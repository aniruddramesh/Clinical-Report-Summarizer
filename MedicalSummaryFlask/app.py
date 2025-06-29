from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel
import torch
from PyPDF2 import PdfReader  # Make sure to install: pip install PyPDF2

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load tokenizer and base model
base_model = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(base_model)
base = T5ForConditionalGeneration.from_pretrained(base_model)

# Load LoRA adapter
model = PeftModel.from_pretrained(base, "./t5_patient_lora_final1")

# Set model to eval mode and move to device
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Summarization function
def summarize(report_text):
    input_text = f"summarize: {report_text}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=128,
            num_beams=6,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Endpoint to accept and summarize PDF file
@app.route("/summarize-pdf", methods=["POST"])
def summarize_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read and extract text from PDF
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            return jsonify({"error": "No text found in the PDF"}), 400

        # Generate summary
        summary = summarize(text)
        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from peft import PeftModel
# from PyPDF2 import PdfReader
# import torch
#
# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)
#
# # Load base T5 model and LoRA adapter
# base_model = "t5-base"
# adapter_path = "./t5_patient_lora_final_v2"
#
# tokenizer = T5Tokenizer.from_pretrained(base_model)
# base = T5ForConditionalGeneration.from_pretrained(base_model)
# model = PeftModel.from_pretrained(base, adapter_path)
#
# # Move model to appropriate device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.eval().to(device)
#
# # Summarization function with prompt engineering
# def summarize(report_text):
#     # Force model to interpret structured clinical data emotionally
#     input_text = (
#         f"summarize: Please explain the following clinical parameters in simple, kind, and reassuring language for the patient:\n{report_text}"
#     )
#
#     inputs = tokenizer(
#         input_text,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=512
#     ).to(device)
#
#     with torch.no_grad():
#         output_ids = model.generate(
#             **inputs,
#             max_length=128,
#             num_beams=6,
#             length_penalty=1.2,
#             no_repeat_ngram_size=3,
#             early_stopping=True
#         )
#
#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)
#
# # PDF upload endpoint
# @app.route("/summarize-pdf", methods=["POST"])
# def summarize_pdf():
#     if "file" not in request.files:
#         return jsonify({"error": "No file part"}), 400
#
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
#
#     try:
#         # Extract text from PDF pages
#         pdf = PdfReader(file)
#         text = ""
#         for page in pdf.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"
#
#         if not text.strip():
#             return jsonify({"error": "No text found in the PDF"}), 400
#
#         # Summarize extracted text
#         summary = summarize(text)
#         return jsonify({"summary": summary})
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# # Run server
# if __name__ == "__main__":
#     app.run(debug=True)
