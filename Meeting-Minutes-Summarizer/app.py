
import re

import nltk
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


# Download the required NLTK resources once.
@st.cache_resource
def setup_nltk():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)

    return set(stopwords.words("english"))


STOP_WORDS = setup_nltk()


# -----------------------------------------------
# 1. Preprocess text
# -----------------------------------------------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s'-]", "", text)

    tokens = word_tokenize(text)

    tokens = [
        token for token in tokens
        if token not in STOP_WORDS
        and len(token) > 1
    ]

    return " ".join(tokens)


# -----------------------------------------------
# 2. Extract important sentences using TF-IDF
# -----------------------------------------------

def summarize_text(text, num_sentences=5):
    sentences = sent_tokenize(text)

    if not sentences:
        return [], []

    if len(sentences) <= num_sentences:
        return sentences, []

    cleaned_sentences = [
        preprocess_text(sentence)
        for sentence in sentences
    ]

    # Avoid empty documents after preprocessing.
    valid_indices = [
        i for i, sentence in enumerate(cleaned_sentences)
        if sentence.strip()
    ]

    if not valid_indices:
        return sentences[:num_sentences], []

    valid_sentences = [
        cleaned_sentences[i] for i in valid_indices
    ]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(valid_sentences)

    # Score sentences by their average TF-IDF weight.
    sentence_scores = matrix.mean(axis=1).A1

    ranked_indices = sorted(
        range(len(valid_indices)),
        key=lambda i: sentence_scores[i],
        reverse=True,
    )

    selected_indices = sorted(
        valid_indices[i]
        for i in ranked_indices[:num_sentences]
    )

    summary = [sentences[i] for i in selected_indices]

    # Return ranked sentence candidates for inspection.
    ranked_sentences = [
        (sentences[valid_indices[i]], float(sentence_scores[i]))
        for i in ranked_indices
    ]

    return summary, ranked_sentences


# -----------------------------------------------
# 3. Extract frequent keywords
# -----------------------------------------------

def extract_keywords(text, top_n=10):
    cleaned_text = preprocess_text(text)

    if not cleaned_text:
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000,
    )

    try:
        matrix = vectorizer.fit_transform([cleaned_text])
    except ValueError:
        return []

    terms = vectorizer.get_feature_names_out()
    scores = matrix.toarray()[0]

    ranked = sorted(
        zip(terms, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {"Keyword": term, "Score": round(float(score), 4)}
        for term, score in ranked[:top_n]
    ]


# -----------------------------------------------
# 4. Find possible action-item sentences
# -----------------------------------------------

def extract_action_candidates(text):
    sentences = sent_tokenize(text)

    action_patterns = re.compile(
        r"\b("
        r"will|must|should|need to|needs to|"
        r"action item|follow up|follow-up|"
        r"responsible for|assigned to|"
        r"deadline|by tomorrow|by monday|"
        r"by tuesday|by wednesday|by thursday|"
        r"by friday"
        r")\b",
        re.IGNORECASE,
    )

    return [
        sentence
        for sentence in sentences
        if action_patterns.search(sentence)
    ]


# -----------------------------------------------
# 5. Streamlit interface
# -----------------------------------------------

st.set_page_config(
    page_title="NLP Meeting Minutes Summarizer",
    page_icon="📝",
    layout="wide",
)

st.title("📝 Meeting Minutes Summarizer")
st.caption(
    "Traditional NLP using TF-IDF and sentence extraction. "
    "No AI API required."
)

uploaded_file = st.file_uploader(
    "Upload a meeting transcript",
    type=["txt"],
)

transcript = ""

if uploaded_file:
    try:
        transcript = uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        st.error("Please upload a UTF-8 text file.")

transcript = st.text_area(
    "Or paste your meeting transcript",
    value=transcript,
    height=250,
    placeholder=(
        "The team discussed the new project. "
        "Priya will finish testing by Friday. "
        "The team agreed to review the results next Monday."
    ),
)

num_sentences = st.slider(
    "Number of sentences in the summary",
    min_value=2,
    max_value=15,
    value=5,
)

if st.button("Summarize Meeting", type="primary"):
    if not transcript.strip():
        st.warning("Please enter or upload a transcript.")
    else:
        try:
            with st.spinner("Processing transcript with NLP..."):
                summary, ranked_sentences = summarize_text(
                    transcript,
                    num_sentences,
                )

                keywords = extract_keywords(transcript)

                action_candidates = extract_action_candidates(
                    transcript
                )

            st.session_state["summary"] = summary
            st.session_state["keywords"] = keywords
            st.session_state["actions"] = action_candidates
            st.session_state["ranked"] = ranked_sentences

        except Exception as exc:
            st.error(f"Unable to process transcript: {exc}")


# -----------------------------------------------
# 6. Display the results
# -----------------------------------------------

if "summary" in st.session_state:
    summary = st.session_state["summary"]

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric("Transcript Words", len(transcript.split()))
    col2.metric("Summary Sentences", len(summary))
    col3.metric(
        "Action Candidates",
        len(st.session_state["actions"]),
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Summary",
        "Keywords",
        "Action Candidates",
        "Sentence Scores",
    ])

    with tab1:
        st.subheader("Meeting Summary")

        summary_text = "\n\n".join(
            f"- {sentence}" for sentence in summary
        )

        st.markdown(summary_text)

        st.download_button(
            "Download Summary",
            data=summary_text,
            file_name="meeting_summary.txt",
            mime="text/plain",
        )

    with tab2:
        st.subheader("Important Keywords")

        keywords = st.session_state["keywords"]

        if keywords:
            st.dataframe(
                pd.DataFrame(keywords),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No meaningful keywords were found.")

    with tab3:
        st.subheader("Possible Action Items")
        st.caption(
            "These sentences match action-related language. "
            "They have not been verified as assigned tasks."
        )

        actions = st.session_state["actions"]

        if actions:
            for action in actions:
                st.markdown(f"- {action}")
        else:
            st.info("No action-related sentences were detected.")

    with tab4:
        st.subheader("Sentence Importance")

        ranked = st.session_state["ranked"]

        if ranked:
            st.dataframe(
                pd.DataFrame(
                    ranked,
                    columns=["Sentence", "TF-IDF Score"],
                ),
                use_container_width=True,
                hide_index=True,
            )
