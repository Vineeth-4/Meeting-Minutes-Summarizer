"""Meeting Minutes Summarizer — local, extractive NLP with Streamlit."""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from io import BytesIO
from typing import Any

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS


# ----------------------------- NLP utilities -----------------------------
@st.cache_resource(show_spinner=False)
def get_stop_words() -> set[str]:
    """Load English stop words, falling back to scikit-learn's local list."""
    try:
        stopwords.words("english")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
        except Exception:
            pass
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return set(ENGLISH_STOP_WORDS)


STOP_WORDS = get_stop_words()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, with a regex fallback if NLTK data is absent."""
    text = text.strip()
    if not text:
        return []
    try:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except LookupError:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n+", text) if s.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower())


def preprocess_text(text: str) -> str:
    return " ".join(
        token for token in tokenize(text)
        if token not in STOP_WORDS and len(token) > 1
    )


def _sentence_features(sentence: str, index: int, total: int) -> float:
    """Lightweight structural cues that complement TF-IDF relevance."""
    words = tokenize(sentence)
    if not words:
        return 0.0
    length_score = min(len(words) / 20, 1.0)
    position_score = 1.0 - (index / max(total - 1, 1))
    cue_pattern = re.compile(
        r"\b(agreed|decided|decision|approved|deadline|priority|risk|blocker|"
        r"action item|next step|launch|deliver|owner|responsible|must|will)\b",
        re.IGNORECASE,
    )
    cue_score = 1.0 if cue_pattern.search(sentence) else 0.0
    return 0.15 * length_score + 0.10 * position_score + 0.15 * cue_score


def summarize_text(text: str, num_sentences: int = 5) -> tuple[list[str], list[tuple[str, float]]]:
    sentences = split_sentences(text)
    if not sentences:
        return [], []
    if len(sentences) <= num_sentences:
        return sentences, [(s, 1.0) for s in sentences]

    cleaned = [preprocess_text(s) for s in sentences]
    valid = [i for i, value in enumerate(cleaned) if value.strip()]
    if not valid:
        selected = sentences[:num_sentences]
        return selected, [(s, 0.0) for s in sentences]

    docs = [cleaned[i] for i in valid]
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(docs)
        raw_scores = matrix.mean(axis=1).A1
    except ValueError:
        raw_scores = [0.0] * len(valid)

    scored: list[tuple[int, float]] = []
    for local_index, original_index in enumerate(valid):
        score = float(raw_scores[local_index]) + _sentence_features(
            sentences[original_index], original_index, len(sentences)
        )
        scored.append((original_index, score))

    ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
    chosen_indices = sorted(index for index, _ in ranked[:num_sentences])
    summary = [sentences[i] for i in chosen_indices]
    ranked_sentences = [(sentences[i], round(score, 5)) for i, score in ranked]
    return summary, ranked_sentences


def extract_keywords(text: str, top_n: int = 12) -> list[dict[str, Any]]:
    sentences = split_sentences(text)
    cleaned = [preprocess_text(s) for s in sentences]
    cleaned = [s for s in cleaned if s]
    if not cleaned:
        return []
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, sublinear_tf=True)
        matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        return []
    terms = vectorizer.get_feature_names_out()
    scores = matrix.sum(axis=0).A1
    counts = Counter(token for sentence in cleaned for token in sentence.split())
    ranked = sorted(
        zip(terms, scores),
        key=lambda item: (float(item[1]), sum(counts.get(t, 0) for t in item[0].split())),
        reverse=True,
    )
    return [
        {"Keyword / phrase": term, "Relevance score": round(float(score), 4)}
        for term, score in ranked[:max(1, top_n)]
    ]


ACTION_PATTERN = re.compile(
    r"\b(will|must|should|need(?:s)? to|have to|has to|assigned|action item|follow[- ]?up|"
    r"responsible for|owner|deadline|by (?:tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"due (?:on|by)|send|prepare|complete|finish|review|schedule|contact|deliver|create|update|investigate|confirm)\b",
    re.IGNORECASE,
)
DECISION_PATTERN = re.compile(
    r"\b(agreed|decided|decision|approved|rejected|selected|chosen|resolved|consensus|will proceed|moving forward)\b",
    re.IGNORECASE,
)
DEADLINE_PATTERN = re.compile(
    r"\b(by\s+(?:today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+|[a-z]+\s+\d{1,2})|"
    r"due\s+(?:on\s+)?[^,;.]+|before\s+[^,;.]+|within\s+\d+\s+(?:days?|weeks?|hours?))\b",
    re.IGNORECASE,
)


def extract_action_items(text: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for sentence in split_sentences(text):
        if not ACTION_PATTERN.search(sentence):
            continue
        normalized = re.sub(r"\s+", " ", sentence).strip()
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deadline_match = DEADLINE_PATTERN.search(sentence)
        # Heuristic owner extraction: names immediately before common action verbs.
        owner_match = re.search(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:will|must|should|needs? to|has to|is responsible for|to)\b",
            sentence,
        )
        items.append({
            "Possible owner": owner_match.group(1) if owner_match else "Not identified",
            "Possible deadline": deadline_match.group(0).strip() if deadline_match else "Not specified",
            "Action / sentence": normalized,
        })
    return items


def extract_decisions(text: str) -> list[str]:
    return list(dict.fromkeys(
        sentence for sentence in split_sentences(text) if DECISION_PATTERN.search(sentence)
    ))


def make_minutes_markdown(title: str, summary: list[str], decisions: list[str], actions: list[dict[str, str]], keywords: list[dict[str, Any]]) -> str:
    date_text = datetime.now().strftime("%d %B %Y")
    lines = [f"# {title or 'Meeting Minutes'}", f"_Generated on {date_text}_", "", "## Summary"]
    lines += [f"- {sentence}" for sentence in summary] or ["- No summary could be generated."]
    lines += ["", "## Decisions"]
    lines += [f"- {item}" for item in decisions] or ["- No decision-related sentences detected."]
    lines += ["", "## Possible Action Items"]
    if actions:
        lines += [f"- **Owner:** {a['Possible owner']} | **Deadline:** {a['Possible deadline']}  \n  {a['Action / sentence']}" for a in actions]
    else:
        lines.append("- No action-related sentences detected.")
    lines += ["", "## Keywords"]
    lines += [f"- {item['Keyword / phrase']}" for item in keywords] or ["- No keywords detected."]
    lines += ["", "---", "_Generated using extractive NLP. Please verify owners, deadlines, and decisions before sharing._", ""]
    return "\n".join(lines)


# ------------------------------- Streamlit UI -------------------------------
st.set_page_config(page_title="Minutes Studio", page_icon="📝", layout="wide")
st.markdown("""
<style>
    .block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1250px;}
    .hero {padding: 1.35rem 1.5rem; border-radius: 16px; border: 1px solid rgba(128,128,128,.25); margin-bottom: 1.2rem;}
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2.15rem;}
    .hero p {margin: 0; opacity: .78; font-size: 1rem;}
    div[data-testid="stMetric"] {padding: .8rem; border: 1px solid rgba(128,128,128,.22); border-radius: 12px;}
    .small-note {font-size: .85rem; opacity: .75;}
</style>
<div class="hero">
  <h1>📝 Minutes Studio</h1>
  <p>Turn a transcript into a structured meeting brief — summary, decisions, possible action items, and keywords.</p>
</div>
""", unsafe_allow_html=True)
st.caption("Local, extractive NLP • No paid AI API • Review automatically detected details before sharing")

with st.sidebar:
    st.header("⚙️ Summary settings")
    num_sentences = st.slider("Summary length", min_value=2, max_value=15, value=5,
                              help="How many important sentences to include in the summary.")
    keyword_count = st.slider("Keywords to show", min_value=5, max_value=25, value=12)
    st.divider()
    st.markdown("**How it works**")
    st.write("Sentence ranking combines TF-IDF relevance with lightweight position and meeting-cue signals. No generative model is used.")
    st.caption("Action owners and deadlines are heuristic guesses, not verified assignments.")

left, right = st.columns([1.5, 1], gap="large")
with left:
    st.subheader("1. Add your transcript")
    uploaded_file = st.file_uploader("Upload a plain-text transcript", type=["txt"], help="UTF-8 and UTF-8-with-BOM files are supported.")
    uploaded_text = ""
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        try:
            uploaded_text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error("This file could not be decoded as UTF-8. Save it as UTF-8 and upload it again.")
    if "transcript_input" not in st.session_state:
        st.session_state.transcript_input = uploaded_text
    if uploaded_file is not None and st.session_state.get("last_uploaded_name") != uploaded_file.name:
        st.session_state.transcript_input = uploaded_text
        st.session_state.last_uploaded_name = uploaded_file.name
    transcript = st.text_area(
        "Transcript text", key="transcript_input", height=310,
        placeholder="Paste meeting notes or a transcript here…\n\nExample: Priya will finish testing by Friday. The team agreed to review the results next Monday.",
        label_visibility="collapsed",
    )
    st.caption(f"{len(transcript.split()):,} words · {len(split_sentences(transcript)):,} detected sentences")
with right:
    st.subheader("2. Meeting details")
    meeting_title = st.text_input("Meeting title", placeholder="e.g. Weekly Product Review")
    st.info("**Tip:** Speaker names, decisions, and dates in the transcript can help the extractor identify likely owners and deadlines.")
    st.markdown("**What you’ll get**")
    st.markdown("- Executive-style sentence summary\n- Decision-related statements\n- Possible tasks, owners, and deadlines\n- Keywords and sentence relevance scores")
    run_clicked = st.button("✨ Generate meeting minutes", type="primary", use_container_width=True, disabled=not transcript.strip())
    if st.button("Clear transcript and results", use_container_width=True):
        for key in ["transcript_input", "results", "last_uploaded_name"]:
            st.session_state.pop(key, None)
        st.rerun()

if run_clicked:
    try:
        with st.spinner("Analysing transcript…"):
            summary, ranked = summarize_text(transcript, num_sentences)
            keywords = extract_keywords(transcript, keyword_count)
            actions = extract_action_items(transcript)
            decisions = extract_decisions(transcript)
            st.session_state.results = {
                "source_text": transcript,
                "title": meeting_title.strip() or "Meeting Minutes",
                "summary": summary,
                "ranked": ranked,
                "keywords": keywords,
                "actions": actions,
                "decisions": decisions,
                "sentences": split_sentences(transcript),
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
    except Exception as exc:
        st.error(f"The transcript could not be processed: {exc}")

results = st.session_state.get("results")
if results:
    st.divider()
    st.subheader("Meeting overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words", f"{len(results['source_text'].split()):,}")
    m2.metric("Sentences", f"{len(results['sentences']):,}")
    m3.metric("Possible actions", len(results["actions"]))
    m4.metric("Decision signals", len(results["decisions"]))

    tab_summary, tab_actions, tab_decisions, tab_keywords, tab_scores, tab_transcript = st.tabs([
        "📌 Summary", "✅ Action items", "⚖️ Decisions", "🔑 Keywords", "📊 Sentence scores", "📄 Transcript"
    ])
    with tab_summary:
        st.subheader(results["title"])
        if results["summary"]:
            for sentence in results["summary"]:
                st.markdown(f"- {sentence}")
        else:
            st.info("No sentences were detected. Check that the transcript contains readable text.")
        st.download_button("⬇️ Download summary (.txt)", "\n".join(f"- {s}" for s in results["summary"]),
                           file_name="meeting_summary.txt", mime="text/plain", key="download_summary")
        markdown = make_minutes_markdown(results["title"], results["summary"], results["decisions"], results["actions"], results["keywords"])
        st.download_button("⬇️ Download full minutes (.md)", markdown, file_name="meeting_minutes.md", mime="text/markdown", key="download_markdown")

    with tab_actions:
        st.caption("Automatically detected candidates only. Confirm task ownership and deadlines against the original meeting.")
        if results["actions"]:
            actions_df = pd.DataFrame(results["actions"])
            st.dataframe(actions_df, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download action items (.csv)", actions_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="meeting_action_items.csv", mime="text/csv", key="download_actions")
        else:
            st.info("No action-related sentences were detected. Try including explicit task language such as ‘will’, ‘assigned’, or ‘by Friday’.")

    with tab_decisions:
        st.caption("These sentences contain decision-related language; they may need human interpretation.")
        if results["decisions"]:
            for decision in results["decisions"]:
                st.markdown(f"- {decision}")
        else:
            st.info("No clear decision cues were detected.")

    with tab_keywords:
        if results["keywords"]:
            keywords_df = pd.DataFrame(results["keywords"])
            st.dataframe(keywords_df, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download keywords (.csv)", keywords_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="meeting_keywords.csv", mime="text/csv", key="download_keywords")
        else:
            st.info("No meaningful keywords were found in this transcript.")

    with tab_scores:
        st.caption("Higher scores indicate stronger ranking signals, not factual certainty.")
        if results["ranked"]:
            scores_df = pd.DataFrame(results["ranked"], columns=["Sentence", "Importance score"])
            st.dataframe(scores_df, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download sentence scores (.csv)", scores_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="sentence_scores.csv", mime="text/csv", key="download_scores")
        else:
            st.info("No sentence scores are available.")

    with tab_transcript:
        st.text_area("Original transcript", value=results["source_text"], height=350, disabled=True)
        st.download_button("⬇️ Download original transcript (.txt)", results["source_text"],
                           file_name="original_transcript.txt", mime="text/plain", key="download_transcript")

    st.caption(f"Generated {results['generated_at']} · NLP output should be reviewed before distribution.")
else:
    st.divider()
    st.subheader("Your minutes will appear here")
    st.write("Upload a `.txt` file or paste a transcript, adjust the summary length, and select **Generate meeting minutes**.")
