from __future__ import annotations

from pathlib import Path
from typing import List

import streamlit as st
from PyPDF2 import PdfReader
from docx import Document

from resume_screening import (
    ResumeMatchResult,
    compute_similarity_scores,
    extract_job_keywords,
    score_resume,
)


def _load_text_from_uploaded(file) -> str:
    name = (file.name or "").lower()

    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)

    # assume text
    data = file.read()
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("latin-1", errors="ignore")


def run_app() -> None:
    st.set_page_config(page_title="AI Resume Screening", layout="wide")

    st.title("AI-Powered Resume Screening System")
    st.write(
        "Upload a job description and a set of resumes. "
        "The app will analyze them using NLP and compute a job match percentage for each candidate."
    )

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of top candidates to show", min_value=1, max_value=50, value=10)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job description")
        job_text_input = st.text_area(
            "Paste job description text",
            height=220,
            placeholder="Paste the job title, responsibilities, and required skills here...",
        )
        job_file = st.file_uploader(
            "Or upload job description file (txt / pdf / docx)",
            type=["txt", "pdf", "docx"],
            key="job_file",
        )

    with col2:
        st.subheader("Candidate resumes")
        resume_files = st.file_uploader(
            "Upload one or more resumes (txt / pdf / docx)",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
            key="resumes",
        )

    if st.button("Run screening"):
        if not job_text_input and not job_file:
            st.error("Please provide a job description (paste text or upload a file).")
            return

        if not resume_files:
            st.error("Please upload at least one resume.")
            return

        # Determine job description text
        if job_text_input.strip():
            job_text = job_text_input.strip()
        else:
            job_text = _load_text_from_uploaded(job_file)

        if not job_text.strip():
            st.error("Could not read any text from the job description.")
            return

        # Read resumes
        resume_texts: List[str] = []
        resume_names: List[str] = []
        for f in resume_files:
            text = _load_text_from_uploaded(f)
            if not text.strip():
                continue
            resume_texts.append(text)
            resume_names.append(f.name)

        if not resume_texts:
            st.error("Could not read text from any uploaded resumes.")
            return

        st.info(f"Loaded job description and {len(resume_texts)} resume(s). Running analysis...")

        job_keywords = extract_job_keywords(job_text)
        cosine_scores = compute_similarity_scores(job_text, resume_texts)

        results: List[ResumeMatchResult] = []
        for name, text, cos in zip(resume_names, resume_texts, cosine_scores):
            result = score_resume(
                file_path=Path(name),
                job_text=job_text,
                resume_text=text,
                job_keywords=job_keywords,
                cosine_score=cos,
            )
            results.append(result)

        if not results:
            st.warning("No matches computed.")
            return

        results.sort(key=lambda r: r.match_percentage, reverse=True)
        results_top = results[:top_k]

        # Summary table
        table_rows = [
            {
                "Rank": idx + 1,
                "Resume": r.file_path.name,
                "Match %": round(r.match_percentage, 1),
                "Cosine": round(r.cosine_similarity_score, 3),
                "Overlap": round(r.keyword_overlap_score, 3),
            }
            for idx, r in enumerate(results_top)
        ]

        st.subheader("Ranked candidates")
        st.dataframe(table_rows, use_container_width=True)

        # Detailed explanations
        st.subheader("Explanation for top candidates")
        for idx, r in enumerate(results_top, start=1):
            with st.expander(f"[{idx}] {r.file_path.name} — {r.match_percentage:.1f}% match"):
                if r.matched_keywords:
                    st.markdown(
                        "**Matched keywords/skills:** " + ", ".join(sorted(set(r.matched_keywords)))
                    )
                else:
                    st.markdown("**Matched keywords/skills:** None detected.")

                if r.missing_keywords:
                    st.markdown(
                        "**Missing important keywords/skills:** "
                        + ", ".join(sorted(set(r.missing_keywords)))
                    )
                else:
                    st.markdown("**Missing important keywords/skills:** None (all covered).")


if __name__ == "__main__":
    run_app()

