from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel
import torch
import os
import tempfile
from PyPDF2 import PdfReader

# Initialize Flask app
app = Flask(__name__)
CORS(app)  #  Allow requests from React frontend (e.g., http://localhost:5173)

# Load tokenizer and model
base_model = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(base_model)
base = T5ForConditionalGeneration.from_pretrained(base_model)
model = PeftModel.from_pretrained(base, "./flan_t5_patient_emotional_lora_final")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --------- Summarization Function ---------
def summarize(text):
    input_text = f"summarize: {text}"
    inputs = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True, max_length=512
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

# --------- Route: Upload PDF and Summarize ---------
@app.route('/summarize-pdf', methods=['POST'])
def summarize_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    # Extract text from PDF
    try:
        reader = PdfReader(tmp_path)
        extracted_text = " ".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        return jsonify({'error': f'Failed to read PDF: {str(e)}'}), 500
    finally:
        os.remove(tmp_path)

    if not extracted_text.strip():
        return jsonify({'error': 'No text found in the uploaded PDF.'}), 400

    # Summarize
    summary = summarize(extracted_text)
    return jsonify({'summary': summary})

# --------- Run the Flask app ---------
if __name__ == '__main__':
    app.run(debug=True)
