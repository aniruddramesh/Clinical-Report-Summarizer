Welcome to the Clinical Report Summarizer- a smart, patient-friendly tool that transforms complex physician reports into clear, easy-to-understand summaries. Built using a fine-tuned Flan-T5 transformer model with LoRA (Low-Rank Adaptation), our goal is to make healthcare information more accessible, especially for non-experts or patients in rural areas.

Our system was tested using standard evaluation metrics:
1. ROUGE-1: 0.6916
2. ROUGE-2: 0.5045
3. ROUGE-L: 0.6390
4. BLEU Score: 0.4584
5. Flesch Reading Ease: 47.92
6. Grade Level: 9.74

How to Run This Project

1. Clone the repo ->
git clone https://github.com/aniruddramesh/Clinical-Report-Summarizer.git ->
cd Clinical-Report-Summarizer

3. Set up the Backend ->
cd Medical-Report-Summarizer-Backend ->
pip install -r requirements.txt ->
python app.py ->
python docotr_api.py

5. Run the Frontend ->
cd ../Medical-Report-Summarizer-Frontend ->
npm install ->
npm run dev 
