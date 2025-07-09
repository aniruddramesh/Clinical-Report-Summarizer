from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import google.generativeai as genai
import tempfile
import os


genai.configure(api_key="AIzaSyDIeJeU6E1viTeDoRgVxsxJB2pkiZgd9rg")

# Initialize Gemini model
model = genai.GenerativeModel("models/gemini-2.5-flash")

app = Flask(__name__)
CORS(app)

# ----------- Utility Functions -----------

def extract_text_from_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        reader = PdfReader(tmp_path)
        return " ".join([page.extract_text() or "" for page in reader.pages]).strip()
    finally:
        os.remove(tmp_path)

def create_prompt(text):
    return f"""
You are a medical expert. Read the following research paper and generate a **structured summary**.

The output should contain:

- Title (if available)
- Study Design (sample size, method, duration)
- Key Findings (include percentages, p-values if mentioned)
- Clinical Implications
- Limitations
- Recommendations or Conclusion

Write in a formal and professional tone, useful for doctors and researchers.

### Research Paper:
{text}
"""

# ----------- API Route -----------

@app.route("/summarize-doctor", methods=["POST"])
def summarize_doctor_gemini():
    if "file" not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400

    try:
        text = extract_text_from_pdf(request.files["file"])
        if not text or len(text) < 500:
            return jsonify({"error": "PDF content too short or unreadable."}), 400

        prompt = create_prompt(text)

        #  Send to Gemini
        response = model.generate_content(prompt)
        return jsonify({"summary": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5002, debug=True)
