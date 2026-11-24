# Originality AI starter

A small Streamlit website for a college prototype. It compares a submitted text with a local reference collection using TF-IDF + cosine similarity and separately finds exact shared five-word phrases.

## Run on Windows

1. Install Python 3.10 or newer from https://www.python.org/downloads/
   - During installation, tick **Add Python to PATH**.
2. Extract this ZIP folder.
3. Open the extracted `originality_ai_starter` folder.
4. Click the address bar in File Explorer, type `cmd`, and press Enter.
5. Run:

   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

6. A browser tab should open at http://localhost:8501.

## Test it

- Paste a paragraph copied from one of the files in `reference_docs`.
- Click **Analyze text**.
- You should see a high similarity score and exact shared phrases.
- Try an unrelated paragraph and compare the result.

## Add your own reference documents

Put `.txt` files in `reference_docs` and restart the app, or use the in-app reference upload control for a one-session comparison.

## Important limitations

- The app compares only against its included/local reference documents. It does not search Google, the internet, Turnitin, or academic databases.
- TF-IDF cosine similarity measures textual similarity, not a probability of plagiarism.
- Phrase matching detects exact sequences of five or more consecutive words, not every kind of paraphrase.
- A human should review context, quotations, citations, and common phrases.
- Uploaded submissions are processed locally by this app and are not saved to a submission history.

## Project explanation

TF-IDF (term frequency-inverse document frequency) gives greater weight to words that are informative within a document collection. Cosine similarity compares the angle between the submission vector and each reference vector. A score closer to 1 indicates more similar word patterns in this collection.
