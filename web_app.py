from __future__ import annotations

from pathlib import Path
from typing import List

from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from docx import Document

from resume_screening import (
    ResumeMatchResult,
    compute_similarity_scores,
    extract_job_keywords,
    score_resume,
)

app = Flask(__name__)


def _load_text_from_uploaded(file) -> str:
    name = (file.filename or "").lower()

    if name.endswith(".pdf"):
        reader = PdfReader(file.stream)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        doc = Document(file.stream)
        return "\n".join(p.text for p in doc.paragraphs)

    data = file.stream.read()
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("latin-1", errors="ignore")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", results=None, error=None, top_k=10)

    job_text_input = (request.form.get("job_text") or "").strip()
    job_file = request.files.get("job_file")
    resume_files = request.files.getlist("resumes")
    top_k_str = request.form.get("top_k") or "10"

    try:
        top_k = max(1, min(50, int(top_k_str)))
    except ValueError:
        top_k = 10

    if not job_text_input and (not job_file or not job_file.filename):
        return render_template(
            "index.html",
            results=None,
            error="Please provide a job description (paste text or upload a file).",
            top_k=top_k,
        )

    valid_resumes = [f for f in resume_files if f and f.filename]
    if not valid_resumes:
        return render_template(
            "index.html",
            results=None,
            error="Please upload at least one resume.",
            top_k=top_k,
        )

    if job_text_input:
        job_text = job_text_input
    else:
        job_text = _load_text_from_uploaded(job_file)

    if not job_text.strip():
        return render_template(
            "index.html",
            results=None,
            error="Could not read any text from the job description.",
            top_k=top_k,
        )

    resume_texts: List[str] = []
    resume_names: List[str] = []
    for f in valid_resumes:
        text = _load_text_from_uploaded(f)
        if not text.strip():
            continue
        resume_texts.append(text)
        resume_names.append(f.filename)

    if not resume_texts:
        return render_template(
            "index.html",
            results=None,
            error="Could not read text from any uploaded resumes.",
            top_k=top_k,
        )

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

    results.sort(key=lambda r: r.match_percentage, reverse=True)
    return render_template(
        "index.html",
        results=results[:top_k],
        error=None,
        top_k=top_k,
    )


if __name__ == "__main__":
    app.run(debug=True)

