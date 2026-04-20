from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set

from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS: Set[str] = {
    "the",
    "and",
    "or",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "this",
    "that",
    "it",
    "i",
    "you",
    "he",
    "she",
    "they",
    "we",
    "have",
    "has",
    "had",
    "but",
    "not",
    "can",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
}


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+\-_/]*")


@dataclass
class ResumeMatchResult:
    file_path: Path
    match_percentage: float
    cosine_similarity_score: float
    keyword_overlap_score: float
    matched_keywords: List[str]
    missing_keywords: List[str]


def load_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    # fallback: plain text
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> List[str]:
    tokens = [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in STOPWORDS]
    return tokens


def extract_job_keywords(job_description: str, min_len: int = 4) -> Set[str]:
    tokens = tokenize(job_description)
    return {t for t in tokens if len(t) >= min_len}


def compute_similarity_scores(job_text: str, resume_texts: List[str]) -> List[float]:
    corpus = [job_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    job_vec = tfidf_matrix[0:1]
    resume_vecs = tfidf_matrix[1:]
    sims = cosine_similarity(resume_vecs, job_vec).flatten()
    return sims.tolist()


def score_resume(
    file_path: Path,
    job_text: str,
    resume_text: str,
    job_keywords: Set[str],
    cosine_score: float,
    overlap_weight: float = 0.3,
    cosine_weight: float = 0.7,
) -> ResumeMatchResult:
    job_norm = normalize_text(job_text)
    resume_norm = normalize_text(resume_text)

    resume_tokens = set(tokenize(resume_norm))
    matched = sorted(job_keywords & resume_tokens)
    missing = sorted(job_keywords - resume_tokens)

    keyword_overlap_score = 0.0
    if job_keywords:
        keyword_overlap_score = len(matched) / len(job_keywords)

    combined_score = cosine_weight * cosine_score + overlap_weight * keyword_overlap_score
    match_percentage = max(0.0, min(1.0, combined_score)) * 100.0

    return ResumeMatchResult(
        file_path=file_path,
        match_percentage=match_percentage,
        cosine_similarity_score=cosine_score,
        keyword_overlap_score=keyword_overlap_score,
        matched_keywords=matched[:30],
        missing_keywords=missing[:30],
    )


def find_resume_files(resumes_dir: Path) -> List[Path]:
    exts = {".pdf", ".docx", ".txt"}
    files: List[Path] = []
    for path in resumes_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            files.append(path)
    return sorted(files)


def screen_resumes(job_path: Path, resumes_dir: Path) -> List[ResumeMatchResult]:
    job_text = load_text_from_file(job_path)
    resume_files = find_resume_files(resumes_dir)

    if not resume_files:
        return []

    resume_texts = [load_text_from_file(p) for p in resume_files]
    job_keywords = extract_job_keywords(job_text)
    cosine_scores = compute_similarity_scores(job_text, resume_texts)

    results: List[ResumeMatchResult] = []
    for file_path, resume_text, cos in zip(resume_files, resume_texts, cosine_scores):
        results.append(
            score_resume(
                file_path=file_path,
                job_text=job_text,
                resume_text=resume_text,
                job_keywords=job_keywords,
                cosine_score=cos,
            )
        )

    # sort best match first
    results.sort(key=lambda r: r.match_percentage, reverse=True)
    return results


def print_results(results: Iterable[ResumeMatchResult], top_k: int | None = None) -> None:
    results_list = list(results)
    if top_k is not None:
        results_list = results_list[:top_k]

    if not results_list:
        print("No resumes found or no matches computed.")
        return

    header = f"{'Rank':<5} {'Match%':>7} {'Cosine':>8} {'Overlap':>8}  Resume"
    print(header)
    print("-" * len(header))
    for idx, r in enumerate(results_list, start=1):
        print(
            f"{idx:<5} {r.match_percentage:6.1f} {r.cosine_similarity_score:8.3f} {r.keyword_overlap_score:8.3f}  {r.file_path.name}"
        )

    print("\nDetails for top results:")
    for idx, r in enumerate(results_list, start=1):
        print(f"\n[{idx}] {r.file_path.name} - match {r.match_percentage:.1f}%")
        if r.matched_keywords:
            print("  Matched keywords:", ", ".join(r.matched_keywords))
        if r.missing_keywords:
            print("  Missing keywords:", ", ".join(r.missing_keywords))


def write_csv(results: Iterable[ResumeMatchResult], out_path: Path) -> None:
    results_list = list(results)
    fieldnames = [
        "rank",
        "file_name",
        "file_path",
        "match_percentage",
        "cosine_similarity_score",
        "keyword_overlap_score",
        "matched_keywords",
        "missing_keywords",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, r in enumerate(results_list, start=1):
            writer.writerow(
                {
                    "rank": idx,
                    "file_name": r.file_path.name,
                    "file_path": str(r.file_path),
                    "match_percentage": f"{r.match_percentage:.2f}",
                    "cosine_similarity_score": f"{r.cosine_similarity_score:.4f}",
                    "keyword_overlap_score": f"{r.keyword_overlap_score:.4f}",
                    "matched_keywords": ", ".join(r.matched_keywords),
                    "missing_keywords": ", ".join(r.missing_keywords),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-powered resume screening using NLP and job match percentage."
    )
    parser.add_argument(
        "--job",
        type=str,
        required=True,
        help="Path to job description file (txt, pdf, docx).",
    )
    parser.add_argument(
        "--resumes-dir",
        type=str,
        required=True,
        help="Directory containing resume files (pdf, docx, txt).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to write detailed results as CSV.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top candidates to display (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_path = Path(args.job).expanduser().resolve()
    resumes_dir = Path(args.resumes_dir).expanduser().resolve()

    if not job_path.exists():
        raise SystemExit(f"Job description file not found: {job_path}")
    if not resumes_dir.exists() or not resumes_dir.is_dir():
        raise SystemExit(f"Resumes directory not found or not a directory: {resumes_dir}")

    print(f"Loading job description from: {job_path}")
    print(f"Scanning resumes under: {resumes_dir}")

    results = screen_resumes(job_path, resumes_dir)
    if not results:
        raise SystemExit("No resume files found. Supported extensions: .pdf, .docx, .txt")

    print_results(results, top_k=args.top_k)

    if args.output_csv:
        out_path = Path(args.output_csv).expanduser().resolve()
        write_csv(results, out_path)
        print(f"\nDetailed results written to: {out_path}")


if __name__ == "__main__":
    main()

