"""
gap_analyser.py — Skill Gap Detection

Compares skills extracted from a resume against those required by a job
description to identify matches and gaps.
"""


def analyse_gaps(resume_skills: list[str], jd_skills: list[str]) -> dict:
    """
    Analyse the gap between a candidate's skills and a job description's requirements.

    Performs case-insensitive comparison to avoid false negatives from
    differing capitalisation (e.g., "python" vs "Python").

    Args:
        resume_skills: List of skill strings extracted from the candidate's resume.
        jd_skills:     List of skill strings extracted from the job description.

    Returns:
        A dictionary with the following keys:
            - matched_skills   (list[str]): Skills present in both resume and JD.
            - missing_skills   (list[str]): Skills in the JD but absent from the resume.
            - match_percentage (float):     Percentage of JD skills covered by the resume
                                            (0.0–100.0, rounded to 1 decimal).
    """
    # Normalise to lowercase for comparison, but preserve original casing
    resume_lower_map: dict[str, str] = {s.lower(): s for s in resume_skills}
    jd_lower_map: dict[str, str] = {s.lower(): s for s in jd_skills}

    resume_lower_set = set(resume_lower_map.keys())
    jd_lower_set = set(jd_lower_map.keys())

    matched_lower = resume_lower_set & jd_lower_set
    missing_lower = jd_lower_set - resume_lower_set

    # Map back to canonical (JD) casing
    matched_skills = sorted(
        [jd_lower_map[s] for s in matched_lower], key=str.lower
    )
    missing_skills = sorted(
        [jd_lower_map[s] for s in missing_lower], key=str.lower
    )

    # Calculate match percentage
    if len(jd_lower_set) == 0:
        match_percentage = 100.0  # No requirements → perfect match
    else:
        match_percentage = round(
            (len(matched_lower) / len(jd_lower_set)) * 100, 1
        )

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "match_percentage": match_percentage,
    }
