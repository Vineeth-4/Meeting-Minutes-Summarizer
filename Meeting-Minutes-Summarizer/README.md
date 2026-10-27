# Minutes Studio — Meeting Minutes Summarizer

A local Streamlit app that turns a plain-text meeting transcript into a structured brief using extractive NLP. It does not require a paid AI API.

## Features

- Upload a `.txt` transcript or paste text directly
- Extractive summary with configurable length
- Decision-related sentence detection
- Heuristic action-item detection, including possible owners and deadlines
- Keyword and phrase extraction
- Sentence-importance scores for review
- Downloads for summary (`.txt`), full minutes (`.md`), actions (`.csv`), keywords (`.csv`), sentence scores (`.csv`), and the original transcript (`.txt`)
- Responsive Streamlit interface and transcript statistics
- Graceful fallbacks for some missing NLTK resources

## Setup

Use Python 3.10 or newer. From the directory containing these files:

```bash
python -m venv .venv
```

Activate the environment:

- **Windows PowerShell:** ` .venv\\Scripts\\Activate.ps1 `
- **macOS/Linux:** `source .venv/bin/activate`

Install dependencies and launch:

```bash
python -m pip install -r requirements.txt
streamlit run meeting_minutes_summarizer.py
```

Streamlit will print a local address to open in your browser.

## Important limitations

This is a traditional, extractive NLP tool, not a generative AI summarizer. It selects sentences from the source transcript; it does not rewrite them into fluent prose. Action owners, deadlines, decisions, and importance scores are heuristic suggestions and must be checked against the original transcript. For best results, use a transcript with clear punctuation and speaker names.
