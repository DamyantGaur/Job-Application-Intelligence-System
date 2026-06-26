"""
rewriter.py — LLM-Powered Resume Rewriting & Cover Letter Generation

Uses the Groq API with the Llama 3 8B model to produce professionally
rewritten resume content and tailored cover letters.
"""

import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL = "llama-3.3-70b-versatile"
_MAX_TOKENS = 4096


def _get_client(api_key: str | None = None) -> Groq:
    """
    Create a Groq client using the provided key or the GROQ_API_KEY env var.

    Args:
        api_key: Optional API key. Falls back to the GROQ_API_KEY environment variable.

    Returns:
        An initialised Groq client.

    Raises:
        ValueError: If no API key is available.
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key or key == "your_key_here":
        raise ValueError(
            "Groq API key is required. Set GROQ_API_KEY in your .env file "
            "or enter it in the sidebar."
        )
    return Groq(api_key=key)


def rewrite_resume(
    resume_text: str,
    jd_text: str,
    gaps: dict,
    api_key: str | None = None,
) -> str:
    """
    Rewrite a resume to better match a target job description.

    The rewritten resume:
      - Preserves truthful content from the original resume
      - Improves bullet points with action verbs and quantified achievements
      - Weaves in relevant keywords from the JD and skill gaps
      - Uses a clean, ATS-friendly format

    Args:
        resume_text: The candidate's original resume as plain text.
        jd_text:     The target job description as plain text.
        gaps:        The gap analysis dict from gap_analyser.analyse_gaps().
        api_key:     Optional Groq API key override.

    Returns:
        The rewritten resume as a formatted string.

    Raises:
        RuntimeError: If the Groq API call fails.
    """
    missing = ", ".join(gaps.get("missing_skills", [])) or "None identified"
    matched = ", ".join(gaps.get("matched_skills", [])) or "None identified"

    prompt = f"""You are an expert resume writer and career coach with 15 years of experience 
helping candidates land roles at top companies. Your task is to rewrite the candidate's 
resume so it is optimally tailored to the target job description.

INSTRUCTIONS:
1. PRESERVE all truthful information — do NOT fabricate experience, companies, dates, or degrees.
2. RESTRUCTURE bullet points using the STAR method (Situation, Task, Action, Result).
3. BEGIN every bullet point with a strong action verb (Led, Architected, Optimised, Delivered, etc.).
4. QUANTIFY achievements wherever possible (percentages, dollar amounts, team sizes, time saved).
5. NATURALLY INTEGRATE these keywords that the job description requires: {missing}
   — Only include them where they can be truthfully connected to the candidate's experience.
   — If a skill cannot be honestly claimed, add a "Key Skills" or "Technical Skills" section 
     and include it there ONLY if the candidate plausibly has exposure to it.
6. EMPHASISE these already-matched skills prominently: {matched}
7. Use a CLEAN, ATS-FRIENDLY format with clear section headers:
   — Contact Information, Professional Summary, Experience, Education, Skills, Certifications
8. Write a compelling 2–3 sentence Professional Summary that directly mirrors the JD's key requirements.
9. Keep the tone professional, confident, and concise. Avoid buzzwords without substance.
10. Output ONLY the rewritten resume text — no commentary, explanations, or meta-text.

=== ORIGINAL RESUME ===
{resume_text[:6000]}

=== TARGET JOB DESCRIPTION ===
{jd_text[:3000]}

=== REWRITTEN RESUME ==="""

    try:
        client = _get_client(api_key)
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a world-class resume writer. You produce polished, "
                        "ATS-optimised resumes that help candidates get interviews. "
                        "You never fabricate information."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=_MAX_TOKENS,
            top_p=0.9,
        )
        return response.choices[0].message.content.strip()

    except ValueError:
        raise  # Re-raise missing API key errors
    except Exception as e:
        raise RuntimeError(
            f"Failed to rewrite resume via Groq API: {e}"
        ) from e


def generate_cover_letter(
    resume_text: str,
    jd_text: str,
    company_name: str,
    role_title: str,
    api_key: str | None = None,
) -> str:
    """
    Generate a personalised cover letter for a specific role and company.

    The cover letter:
      - Opens with a compelling hook referencing the company or role
      - Connects the candidate's experience to the JD's top requirements
      - Demonstrates knowledge of the company (based on available info)
      - Closes with a confident call to action

    Args:
        resume_text:  The candidate's resume as plain text.
        jd_text:      The target job description as plain text.
        company_name: Name of the hiring company.
        role_title:   Title of the role being applied for.
        api_key:      Optional Groq API key override.

    Returns:
        The cover letter as a formatted string.

    Raises:
        RuntimeError: If the Groq API call fails.
    """
    prompt = f"""You are an expert career coach and professional writer. Write a personalised, 
compelling cover letter for the candidate applying to the role described below.

INSTRUCTIONS:
1. ADDRESS the letter to "Dear Hiring Manager" (unless a name is available in the JD).
2. OPENING PARAGRAPH: Start with a strong hook — mention the specific role ({role_title}) 
   and company ({company_name}). Express genuine enthusiasm and briefly state why you're 
   an excellent fit.
3. BODY PARAGRAPH 1: Highlight 2–3 of the candidate's most relevant achievements from their 
   resume that directly align with the job description's top requirements. Use specific 
   numbers and results where possible.
4. BODY PARAGRAPH 2: Demonstrate understanding of the company's mission or challenges 
   (infer from the JD). Explain how your unique skills and experience will contribute 
   to their goals.
5. CLOSING PARAGRAPH: Reiterate your enthusiasm, express eagerness for an interview, 
   and include a confident call to action.
6. SIGN OFF with "Sincerely," followed by the candidate's name (extract from resume).
7. TONE: Professional yet personable. Confident but not arrogant. Specific, not generic.
8. LENGTH: 300–400 words. No more.
9. Output ONLY the cover letter — no commentary, subject lines, or meta-text.

=== CANDIDATE'S RESUME ===
{resume_text[:5000]}

=== JOB DESCRIPTION ===
{jd_text[:3000]}

=== ROLE ===
{role_title} at {company_name}

=== COVER LETTER ==="""

    try:
        client = _get_client(api_key)
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional cover letter writer. You write compelling, "
                        "personalised cover letters that get candidates interviews. "
                        "You tailor every letter to the specific role and company."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=_MAX_TOKENS,
            top_p=0.9,
        )
        return response.choices[0].message.content.strip()

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate cover letter via Groq API: {e}"
        ) from e
