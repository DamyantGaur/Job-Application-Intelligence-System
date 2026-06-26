"""
matcher.py — Semantic Match Scoring

Uses a sentence-transformers model to compute cosine similarity between
a resume and a job description, producing a 0–100 match score.
"""

from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# Load the embedding model once at module level.
# all-MiniLM-L6-v2 is lightweight (~80 MB) and fast on CPU.
# ---------------------------------------------------------------------------
_model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")


def compute_match_score(resume_text: str, jd_text: str) -> float:
    """
    Compute a semantic similarity score between a resume and a job description.

    The score is derived from cosine similarity of sentence-level embeddings,
    scaled to a 0–100 range.

    Args:
        resume_text: Plain text of the candidate's resume.
        jd_text:     Plain text of the target job description.

    Returns:
        A float in [0.0, 100.0] rounded to 1 decimal place representing
        how well the resume matches the job description.
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0

    # Encode both texts into dense vectors
    embeddings = _model.encode(
        [resume_text, jd_text],
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    # Cosine similarity returns a value in [-1, 1]
    cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()

    # Scale to 0–100 and clamp
    score = max(0.0, min(100.0, cosine_sim * 100))

    return round(score, 1)


def get_model() -> SentenceTransformer:
    """
    Return the loaded SentenceTransformer model instance.

    Useful for Streamlit's @st.cache_resource pattern to avoid reloading.

    Returns:
        The pre-loaded SentenceTransformer model.
    """
    return _model
