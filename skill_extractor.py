"""
skill_extractor.py — NLP-Based Skill Extraction

Combines spaCy Named Entity Recognition with a curated keyword list
to identify technical and professional skills from text.
"""

import re
import spacy

# ---------------------------------------------------------------------------
# Curated list of 80+ common tech / professional skills for keyword matching.
# These supplement spaCy NER which may miss domain-specific terms.
# ---------------------------------------------------------------------------
SKILLS_KEYWORDS: list[str] = [
    # Programming Languages
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "Go",
    "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB",
    "Perl", "Lua", "Dart", "Shell", "Bash", "PowerShell",
    # Web / Frontend
    "HTML", "CSS", "React", "Angular", "Vue", "Next.js", "Svelte",
    "Tailwind", "Bootstrap", "jQuery", "Webpack", "Vite",
    # Backend / Frameworks
    "Node.js", "Express", "Django", "Flask", "FastAPI", "Spring Boot",
    "Ruby on Rails", "ASP.NET", "Laravel", "NestJS",
    # Data / ML / AI
    "SQL", "NoSQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras",
    "Spark", "Hadoop", "Kafka", "Airflow", "dbt", "Snowflake", "BigQuery",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "LLM", "Generative AI", "Data Science", "Data Engineering",
    # DevOps / Cloud / Infra
    "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitHub Actions",
    "CI/CD", "AWS", "Azure", "GCP", "Linux", "Nginx", "Prometheus", "Grafana",
    # Tools & Practices
    "Git", "Jira", "Confluence", "Figma", "Postman",
    "REST", "GraphQL", "gRPC", "Microservices", "Agile", "Scrum",
    "Unit Testing", "Integration Testing", "TDD", "BDD",
    # Soft / Business Skills
    "Leadership", "Communication", "Problem Solving", "Project Management",
    "Stakeholder Management", "Cross-functional Collaboration",
    "Technical Writing", "Mentoring",
]

# Pre-compile lowercase lookup set for fast matching
_SKILLS_LOWER: set[str] = {s.lower() for s in SKILLS_KEYWORDS}

# Build regex patterns for multi-word skills (so "Machine Learning" matches)
_MULTI_WORD_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(s) + r"\b", re.IGNORECASE), s)
    for s in SKILLS_KEYWORDS
    if " " in s or "." in s or "/" in s
]

# ---------------------------------------------------------------------------
# spaCy model — loaded once at module level
# ---------------------------------------------------------------------------
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    _nlp = None


def extract_skills(text: str) -> list[str]:
    """
    Extract a deduplicated list of skills from the given text.

    Uses two complementary strategies:
      1. spaCy NER to capture entities tagged as ORG, PRODUCT, or SKILL-like.
      2. Keyword / pattern matching against the curated SKILLS_KEYWORDS list.

    Args:
        text: Plain text (resume or job description).

    Returns:
        Sorted, deduplicated list of skill strings found in the text.
    """
    found: set[str] = set()

    # --- Strategy 1: Keyword / pattern matching ---
    text_lower = text.lower()

    # Single-word skills
    for skill in SKILLS_KEYWORDS:
        if " " not in skill and "." not in skill and "/" not in skill:
            # Simple word boundary check
            if re.search(r"\b" + re.escape(skill.lower()) + r"\b", text_lower):
                found.add(skill)

    # Multi-word / special-char skills
    for pattern, canonical in _MULTI_WORD_PATTERNS:
        if pattern.search(text):
            found.add(canonical)

    # --- Strategy 2: spaCy NER ---
    if _nlp is not None:
        doc = _nlp(text[:100_000])  # Limit to avoid memory issues on huge docs
        for ent in doc.ents:
            ent_lower = ent.text.strip().lower()
            # Check if the entity matches a known skill
            if ent_lower in _SKILLS_LOWER:
                # Use canonical casing from our list
                for skill in SKILLS_KEYWORDS:
                    if skill.lower() == ent_lower:
                        found.add(skill)
                        break

    return sorted(found, key=str.lower)
