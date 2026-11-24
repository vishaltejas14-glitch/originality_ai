import html
import io
import re
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = Path(__file__).parent
REF_DIR = APP_DIR / "reference_docs"
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(
    page_title="Originality AI | Similarity Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Dark theme --------------------
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #080d18; color: #e7ecf5; }
[data-testid="stHeader"] { background: rgba(8,13,24,.92); }
[data-testid="stSidebar"] { background: #0c1424; border-right: 1px solid #1e2a40; }
[data-testid="stSidebar"] * { color: #dce6f7; }
.block-container { max-width: 1360px; padding-top: 1.7rem; padding-bottom: 3rem; }
h1,h2,h3 { color: #f4f7ff !important; letter-spacing: -.025em; }
p,label,.stCaption { color: #aab7cc; }
.hero { padding: 27px 31px; border: 1px solid #263858; border-radius: 20px;
        background: radial-gradient(circle at 85% 10%, rgba(99,102,241,.28), transparent 32%),
                    linear-gradient(120deg,#111d35,#10172a 65%,#171a36);
        margin-bottom: 23px; box-shadow: 0 18px 50px rgba(0,0,0,.22); }
.eyebrow { color: #91a9ff; font-size: .75rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
.hero h1 { margin: 8px 0 6px; font-size: 2.4rem; }
.hero p { margin: 0; color: #b9c7df; font-size: 1rem; }
.panel { background: #101a2b; border: 1px solid #24334d; border-radius: 16px; padding: 18px 20px; }
.metric-card { background: linear-gradient(145deg,#121f34,#101827); border: 1px solid #263958;
               border-radius: 16px; padding: 17px 19px; min-height: 112px; }
.metric-label { color: #9eacc3; font-size: .78rem; margin-bottom: 9px; }
.metric-value { color: #f5f7ff; font-size: 1.9rem; line-height: 1.1; font-weight: 750; }
.metric-sub { color: #7f91ad; font-size: .74rem; margin-top: 7px; }
.pill { display:inline-block; padding:5px 10px; border-radius:999px; background:#172b48; color:#a9c5ff; border:1px solid #2b4770; font-size:.75rem; font-weight:700; }
.match { background:#111d30; border:1px solid #293b58; border-left:4px solid #8b7cff; padding:11px 14px; border-radius:10px; margin:7px 0; color:#e3eafa; }
.muted { color:#8fa0ba; font-size:.86rem; }
div.stButton > button[kind="primary"] { background: linear-gradient(90deg,#5b6cff,#8b5cf6); border:0; color:white; border-radius:11px; font-weight:750; min-height:2.9rem; }
div.stButton > button[kind="primary"]:hover { border:0; filter:brightness(1.12); }
.stTextArea textarea,.stTextInput input { background:#0d1727 !important; color:#edf3ff !important; border:1px solid #2a3a55 !important; border-radius:12px !important; }
[data-testid="stFileUploader"] section { background:#0d1727; border:1px dashed #405477; border-radius:12px; }
[data-testid="stExpander"] { background:#101a2b; border:1px solid #293b58; border-radius:12px; }
mark { background:#6d5dfc; color:#fff; padding:2px 3px; border-radius:4px; }
hr { border-color:#23324a; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# -------------------- Existing reference and TF-IDF engine --------------------
def read_reference_files():
    docs = {}
    REF_DIR.mkdir(exist_ok=True)
    for path in REF_DIR.glob("*.txt"):
        try:
            content = path.read_text(encoding="utf-8").strip()
            if content:
                docs[path.stem] = content
        except Exception:
            pass
    return docs


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def phrase_matches(submission, reference, min_words=5):
    """Return unique exact sequences of min_words consecutive words."""
    words_a = re.findall(r"\b[\w'-]+\b", submission.lower())
    words_b = re.findall(r"\b[\w'-]+\b", reference.lower())
    if len(words_a) < min_words or len(words_b) < min_words:
        return []
    ref_ngrams = {
        tuple(words_b[i:i + min_words])
        for i in range(len(words_b) - min_words + 1)
    }
    found, seen = [], set()
    for i in range(len(words_a) - min_words + 1):
        gram = tuple(words_a[i:i + min_words])
        if gram in ref_ngrams and gram not in seen:
            found.append(" ".join(gram))
            seen.add(gram)
    return found


def analyze(submission, references):
    names = list(references.keys())
    corpus = [clean_text(submission)] + [clean_text(references[n]) for n in names]
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 2), sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    results = []
    for name, score in zip(names, scores):
        results.append({
            "name": name,
            "score": float(score),
            "matches": phrase_matches(submission, references[name]),
            "text": references[name],
            "semantic_score": None,
            "semantic_pairs": [],
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# -------------------- Optional semantic engine --------------------
def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return [part.strip() for part in parts if len(part.split()) >= 5]


@st.cache_resource(show_spinner="Loading the semantic language model for the first time…")
def load_semantic_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SEMANTIC_MODEL_NAME)


def add_semantic_scores(submission, references, results):
    """Compare sentence embeddings. This is a similarity signal, not a plagiarism verdict."""
    model = load_semantic_model()
    submitted_sentences = split_sentences(submission)
    if not submitted_sentences:
        return results

    for item in results:
        reference_sentences = split_sentences(references[item["name"]])
        if not reference_sentences:
            item["semantic_score"] = 0.0
            item["semantic_pairs"] = []
            continue
        a = model.encode(submitted_sentences, convert_to_tensor=False, normalize_embeddings=True)
        b = model.encode(reference_sentences, convert_to_tensor=False, normalize_embeddings=True)
        similarities = cosine_similarity(a, b)
        best_for_each = similarities.max(axis=1)
        item["semantic_score"] = float(best_for_each.mean())
        pairs = []
        for i, row in enumerate(similarities):
            j = int(row.argmax())
            score = float(row[j])
            if score >= 0.45:
                pairs.append({
                    "submission": submitted_sentences[i],
                    "reference": reference_sentences[j],
                    "score": score,
                })
        pairs.sort(key=lambda x: x["score"], reverse=True)
        item["semantic_pairs"] = pairs[:5]
    results.sort(
        key=lambda x: x["semantic_score"] if x["semantic_score"] is not None else -1,
        reverse=True
    )
    return results


# -------------------- Upload, highlighting and PDF utilities --------------------
def extract_uploaded_text(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".txt":
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            st.error("PDF support needs pypdf. Install it with: pip install pypdf")
            return ""
        except Exception as exc:
            st.error(f"Could not read this PDF: {exc}")
            return ""
    st.error("Please upload a .txt or .pdf file.")
    return ""


def highlighted_html(submission, results):
    """Highlight exact phrase matches in submitted text, safely escaping HTML."""
    phrases = sorted(
        {phrase for item in results for phrase in item["matches"]},
        key=len, reverse=True
    )
    escaped = html.escape(submission)
    if not phrases:
        return f"<div class='panel' style='white-space:pre-wrap;line-height:1.8'>{escaped}</div>"
    # Build a regex from escaped source phrases; matching is case-insensitive.
    pattern = re.compile("|".join(re.escape(p) for p in phrases), re.IGNORECASE)
    pieces, last = [], 0
    for match in pattern.finditer(submission):
        pieces.append(html.escape(submission[last:match.start()]))
        pieces.append("<mark>" + html.escape(match.group(0)) + "</mark>")
        last = match.end()
    pieces.append(html.escape(submission[last:]))
    return "<div class='panel' style='white-space:pre-wrap;line-height:1.8'>" + "".join(pieces) + "</div>"


def make_pdf_report(results, submission, word_count, ref_count):
    """Create a downloadable PDF report with ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
        topMargin=16*mm, bottomMargin=16*mm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BrandTitle", parent=styles["Title"], textColor=colors.HexColor("#27356B"), alignment=TA_CENTER, spaceAfter=5))
    styles.add(ParagraphStyle(name="Subtle", parent=styles["Normal"], textColor=colors.HexColor("#526078"), fontSize=9, leading=13))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#3545A5"), spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontSize=8.5, leading=12, wordWrap="CJK"))
    story = [
        Paragraph("ORIGINALITY AI", styles["BrandTitle"]),
        Paragraph("Text Similarity Analysis Report", styles["Heading2"]),
        Paragraph("Prototype report · Local reference collection only", styles["Subtle"]),
        Spacer(1, 8),
    ]
    best = max((r["score"] for r in results), default=0)
    exact_total = sum(len(r["matches"]) for r in results)
    data = [
        ["Highest TF-IDF similarity", f"{best*100:.1f}%"],
        ["Reference documents checked", str(ref_count)],
        ["Exact phrase overlaps", str(exact_total)],
        ["Submission word count", str(word_count)],
    ]
    table = Table(data, colWidths=[85*mm, 70*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#EEF2FF")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#172554")),
        ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#CBD5E1")),
        ("PADDING", (0,0), (-1,-1), 7),
    ]))
    story.extend([table, Spacer(1, 8)])
    story.append(Paragraph("Matching sources", styles["Section"]))
    for item in sorted(results, key=lambda x: x["score"], reverse=True):
        block = [
            Paragraph(f"<b>{html.escape(item['name'])}</b> · TF-IDF cosine similarity: {item['score']*100:.1f}%", styles["SmallBody"])
        ]
        if item["semantic_score"] is not None:
            block.append(Paragraph(f"Semantic sentence similarity signal: {item['semantic_score']*100:.1f}%", styles["SmallBody"]))
        if item["matches"]:
            phrase_text = "; ".join(item["matches"][:12]).encode("latin-1", "replace").decode("latin-1")
            block.append(Paragraph("<b>Exact shared phrases:</b> " + html.escape(phrase_text), styles["SmallBody"]))
        if item["semantic_pairs"]:
            pair_text = " | ".join(
                f"Submission: {p['submission']} / Reference: {p['reference']} ({p['score']*100:.1f}%)"
                for p in item["semantic_pairs"][:3]
            ).encode("latin-1", "replace").decode("latin-1")
            block.append(Paragraph("<b>Potentially related sentence pairs:</b> " + html.escape(pair_text), styles["SmallBody"]))
        story.extend([KeepTogether(block), Spacer(1, 7)])

    story.append(Paragraph("Important interpretation note", styles["Section"]))
    story.append(Paragraph(
        "Similarity scores are signals for human review, not a probability or verdict of plagiarism. "
        "Semantic similarity can reflect shared subject matter or common ideas. Review the source, "
        "context, quotations and citations before drawing conclusions. This report covers only the "
        "reference documents supplied to this local prototype; it does not search the entire internet.",
        styles["SmallBody"
    ]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Submission text (for review)", styles["Section"]))
    safe_text = html.escape(submission.encode("latin-1", "replace").decode("latin-1")).replace("\n", "<br/>")
    story.append(Paragraph(safe_text[:12000], styles["SmallBody"]))
    doc.build(story)
    return buffer.getvalue()


# -------------------- Sidebar navigation --------------------
with st.sidebar:
    st.markdown("<div style='font-size:1.45rem;font-weight:850;color:#f4f7ff'>◈ ORIGINALITY <span style='color:#8b7cff'>AI</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>SIMILARITY INTELLIGENCE</div>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("WORKSPACE", ["Similarity checker", "Reference library", "Analysis report"], label_visibility="visible")
    st.divider()
    st.markdown("**ENGINE STATUS**")
    st.markdown("🟢 TF-IDF engine")
    st.markdown("🟢 Exact phrase matcher")
    try:
        import sentence_transformers  # noqa: F401
        st.markdown("🟢 Semantic package installed")
    except ImportError:
        st.markdown("🟡 Semantic package optional")
    st.caption("Local prototype. Submitted text is held in the current app session for displaying results; it is not written to a history database.")


# -------------------- Header --------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">ACADEMIC INTEGRITY WORKSPACE</div>
  <h1>Originality <span style="color:#9a8cff">AI</span></h1>
  <p>Understand text similarity. Review matching passages. Strengthen your citations.</p>
  <div style="margin-top:18px"><span class="pill">● SYSTEM ONLINE</span> &nbsp; <span class="muted">TF-IDF · SEMANTIC MATCHING · PASSAGE HIGHLIGHTING · PDF REPORTS</span></div>
</div>
""", unsafe_allow_html=True)

refs = read_reference_files()

if page == "Reference library":
    st.markdown("## Reference library")
    st.caption("These local TXT documents are the comparison collection used by the prototype.")
    if refs:
        for name, text in refs.items():
            with st.expander(f"{name} · {len(text.split())} words"):
                st.write(text)
    else:
        st.warning("No TXT references found. Add .txt files to the reference_docs folder.")
    new_refs = st.file_uploader("Add TXT reference documents for this session", type=["txt"], accept_multiple_files=True, key="library_upload")
    if new_refs:
        st.info("These uploads are available for this session only. To retain them, copy the TXT files into reference_docs.")
        for f in new_refs:
            refs[Path(f.name).stem] = f.getvalue().decode("utf-8", errors="ignore")
        st.success(f"{len(new_refs)} reference file(s) loaded for this session.")

elif page == "Similarity checker":
    col_input, col_info = st.columns([1.65, 1], gap="large")
    with col_input:
        st.markdown("### Submit your document")
        st.caption("Paste writing or upload a TXT/PDF file to compare it with the local reference library.")
        pasted = st.text_area("Document text", height=220, placeholder="Paste your assignment text here…", label_visibility="collapsed")
        uploaded = st.file_uploader("Upload document", type=["txt", "pdf"], help="Supported formats: TXT and text-based PDF.")
        uploaded_text = extract_uploaded_text(uploaded) if uploaded else ""
        submitted_text = uploaded_text if uploaded_text.strip() else pasted
        word_count = len(clean_text(submitted_text).split()) if submitted_text else 0
        st.markdown(f"<div class='muted'>{word_count} words detected</div>", unsafe_allow_html=True)
        run_semantic = st.checkbox("Also run semantic sentence comparison (optional model)", value=False)
    with col_info:
        st.markdown("### Analysis engine")
        st.markdown("""
        <div class="panel">
          <div class="eyebrow">HOW IT WORKS</div>
          <p><b>01 · Vectorize</b><br><span class="muted">TF-IDF represents document word patterns.</span></p>
          <p><b>02 · Compare</b><br><span class="muted">Cosine similarity compares the submission with each reference.</span></p>
          <p><b>03 · Locate</b><br><span class="muted">Exact phrase matching finds shared five-word sequences.</span></p>
          <p><b>04 · Understand (optional)</b><br><span class="muted">Sentence embeddings find potentially related sentences with different wording.</span></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:14px' class='muted'>REFERENCE LIBRARY</div><div style='font-size:1.8rem;font-weight:800;color:#f4f7ff'>{len(refs)} <span style='font-size:.9rem;color:#91a1ba;font-weight:500'>documents indexed</span></div>", unsafe_allow_html=True)

    with st.expander("＋ Add reference documents for this session"):
        st.caption("Upload TXT references for this session. For permanent references, place TXT files in reference_docs.")
        extra_files = st.file_uploader("Reference TXT files", type=["txt"], accept_multiple_files=True, key="checker_refs")
        if extra_files:
            for file in extra_files:
                refs[Path(file.name).stem] = file.getvalue().decode("utf-8", errors="ignore")
            st.success(f"{len(extra_files)} additional reference file(s) loaded for this session.")

    if st.button("✦  ANALYZE DOCUMENT", type="primary", use_container_width=True):
        if len(clean_text(submitted_text).split()) < 10:
            st.warning("Please enter at least 10 words so the comparison is meaningful.")
        elif not refs:
            st.error("No reference documents found. Add TXT reference documents first.")
        else:
            with st.spinner("Comparing your document against the reference library…"):
                results = analyze(submitted_text, refs)
                if run_semantic:
                    try:
                        results = add_semantic_scores(submitted_text, refs, results)
                        st.session_state["semantic_available"] = True
                    except ImportError:
                        st.session_state["semantic_available"] = False
                        st.warning("Semantic model package is not installed yet. The TF-IDF and phrase analysis still completed. Install it using the instructions below, then rerun semantic analysis.")
                    except Exception as exc:
                        st.session_state["semantic_available"] = False
                        st.warning(f"Semantic analysis could not run: {exc}. The TF-IDF and phrase analysis still completed.")
                st.session_state["analysis_results"] = results
                st.session_state["analysis_text"] = submitted_text
                st.session_state["analysis_word_count"] = word_count
                st.session_state["analysis_ref_count"] = len(refs)
                st.session_state["analysis_semantic_requested"] = run_semantic

    if st.session_state.get("analysis_results"):
        results = st.session_state["analysis_results"]
        submission = st.session_state.get("analysis_text", "")
        st.divider()
        st.markdown("## Analysis overview")
        best = max((item["score"] for item in results), default=0)
        total_matches = sum(len(item["matches"]) for item in results)
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='metric-card'><div class='metric-label'>HIGHEST TF-IDF SIMILARITY</div><div class='metric-value'>{best*100:.1f}%</div><div class='metric-sub'>Top match in this reference collection</div></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='metric-card'><div class='metric-label'>DOCUMENTS CHECKED</div><div class='metric-value'>{st.session_state.get('analysis_ref_count', len(results))}</div><div class='metric-sub'>Local reference collection</div></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='metric-card'><div class='metric-label'>EXACT PHRASE OVERLAPS</div><div class='metric-value'>{total_matches}</div><div class='metric-sub'>Unique five-word sequences</div></div>", unsafe_allow_html=True)
        st.info("Similarity scores are review signals, not a probability or verdict of plagiarism. Review quotations, citations, context and common phrases.")
        st.markdown("### Matching sources")
        for index, item in enumerate(sorted(results, key=lambda x: x["score"], reverse=True)):
            with st.expander(f"{index + 1}. {item['name']} · {item['score']*100:.1f}% TF-IDF similarity", expanded=(index == 0)):
                st.progress(min(max(item["score"], 0.0), 1.0), text=f"TF-IDF cosine similarity: {item['score']*100:.1f}%")
                if item["semantic_score"] is not None:
                    st.metric("Semantic sentence similarity signal", f"{item['semantic_score']*100:.1f}%")
                    if item["semantic_pairs"]:
                        st.markdown("**Potentially related sentence pairs**")
                        for pair in item["semantic_pairs"]:
                            st.markdown(f"<div class='match'><b>Submission:</b> {html.escape(pair['submission'])}<br/><b>Reference:</b> {html.escape(pair['reference'])}<br/><span class='muted'>Embedding cosine similarity: {pair['score']*100:.1f}%</span></div>", unsafe_allow_html=True)
                if item["matches"]:
                    st.markdown("**Exact shared phrases**")
                    for phrase in item["matches"][:20]:
                        st.markdown(f"<div class='match'>“{html.escape(phrase)}”</div>", unsafe_allow_html=True)
                    if len(item["matches"]) > 20:
                        st.caption(f"Showing 20 of {len(item['matches'])} matching sequences.")
                else:
                    st.write("No exact five-word phrase overlap found.")
                st.markdown("**Reference text preview**")
                st.code(item["text"][:900] + ("…" if len(item["text"]) > 900 else ""), language="text")

        st.markdown("### Matching passages in your submission")
        st.caption("Purple highlights mark exact phrase overlaps found in the reference collection.")
        st.markdown(highlighted_html(submission, results), unsafe_allow_html=True)

        st.markdown("### Export your report")
        try:
            pdf_bytes = make_pdf_report(
                results, submission,
                st.session_state.get("analysis_word_count", len(submission.split())),
                st.session_state.get("analysis_ref_count", len(results))
            )
            st.download_button(
                "⬇  DOWNLOAD PDF ANALYSIS REPORT",
                data=pdf_bytes,
                file_name="originality_ai_analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except ImportError:
            st.warning("PDF export needs ReportLab. Install it with: pip install reportlab")
        except Exception as exc:
            st.error(f"Could not generate the PDF report: {exc}")

        st.markdown("### Recommended next steps")
        st.markdown("""
        - Review every matching passage in context.
        - Cite borrowed ideas and information.
        - Use quotation marks for exact wording taken from a source.
        - Paraphrase thoughtfully and retain attribution.
        - Do not treat similarity or semantic scores as proof of plagiarism.
        """)

elif page == "Analysis report":
    st.markdown("## Analysis report")
    results = st.session_state.get("analysis_results")
    if not results:
        st.info("Run an analysis in Similarity checker first. Your latest report will appear here during this app session.")
    else:
        submission = st.session_state.get("analysis_text", "")
        best = max((r["score"] for r in results), default=0)
        st.metric("Highest TF-IDF similarity", f"{best*100:.1f}%")
        st.write(f"Reference documents checked: {st.session_state.get('analysis_ref_count', len(results))}")
        st.write(f"Submission word count: {st.session_state.get('analysis_word_count', len(submission.split()))}")
        st.markdown("### Submission with exact matches highlighted")
        st.markdown(highlighted_html(submission, results), unsafe_allow_html=True)
        try:
            pdf_bytes = make_pdf_report(results, submission, len(submission.split()), st.session_state.get("analysis_ref_count", len(results)))
            st.download_button("⬇  DOWNLOAD PDF REPORT", pdf_bytes, "originality_ai_analysis_report.pdf", "application/pdf")
        except ImportError:
            st.warning("Install ReportLab using: pip install reportlab")


# -------------------- Footer --------------------
st.divider()
st.markdown("<div class='muted' style='text-align:center'>ORIGINALITY AI · COLLEGE DESIGN THINKING PROTOTYPE · LOCAL REFERENCE COLLECTION ONLY</div>", unsafe_allow_html=True)
