"""
app.py — Job Application Intelligence System (Streamlit Frontend)

A full-featured web application that analyses resumes against job descriptions,
providing match scoring, gap analysis, resume rewriting, and cover letter generation.
"""

import streamlit as st
import tempfile
import os

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Job Application Intelligence System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Module imports (after page config)
# ---------------------------------------------------------------------------
from resume_parser import parse_resume
from skill_extractor import extract_skills
from matcher import compute_match_score
from gap_analyser import analyse_gaps
from rewriter import rewrite_resume, generate_cover_letter


# ---------------------------------------------------------------------------
# Cache the sentence-transformers model to avoid reloading on every rerun
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load and cache the sentence-transformers model."""
    from matcher import get_model
    return get_model()


# ---------------------------------------------------------------------------
# Custom CSS for premium styling
# ---------------------------------------------------------------------------
def inject_custom_css():
    """Inject custom CSS for a polished, modern UI."""
    st.markdown("""
    <style>
        /* ---------- Global ---------- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        .stApp {
            font-family: 'Inter', sans-serif;
        }

        /* ---------- Header ---------- */
        .main-header {
            text-align: center;
            padding: 1.5rem 0 1rem 0;
        }
        .main-header h1 {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.3rem;
        }
        .main-header p {
            color: #94a3b8;
            font-size: 1.05rem;
            font-weight: 400;
        }

        /* ---------- Score card ---------- */
        .score-card {
            text-align: center;
            padding: 2rem;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(30,30,50,0.9), rgba(20,20,40,0.95));
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .score-number {
            font-size: 4.5rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.5rem;
        }
        .score-label {
            font-size: 1rem;
            color: #94a3b8;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        .score-green { color: #4ade80; }
        .score-yellow { color: #facc15; }
        .score-red { color: #f87171; }

        /* ---------- Progress bar ---------- */
        .progress-container {
            width: 100%;
            height: 12px;
            background: rgba(255,255,255,0.08);
            border-radius: 6px;
            margin-top: 1rem;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            border-radius: 6px;
            transition: width 1s ease;
        }
        .progress-green { background: linear-gradient(90deg, #22c55e, #4ade80); }
        .progress-yellow { background: linear-gradient(90deg, #eab308, #facc15); }
        .progress-red { background: linear-gradient(90deg, #dc2626, #f87171); }

        /* ---------- Skill badges ---------- */
        .skill-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            margin: 4px;
            transition: transform 0.2s ease;
        }
        .skill-badge:hover {
            transform: translateY(-2px);
        }
        .badge-green {
            background: rgba(74, 222, 128, 0.15);
            color: #4ade80;
            border: 1px solid rgba(74, 222, 128, 0.3);
        }
        .badge-red {
            background: rgba(248, 113, 113, 0.15);
            color: #f87171;
            border: 1px solid rgba(248, 113, 113, 0.3);
        }

        /* ---------- Section headers ---------- */
        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid rgba(255,255,255,0.08);
        }

        /* ---------- Info cards ---------- */
        .info-card {
            background: linear-gradient(135deg, rgba(30,30,50,0.7), rgba(20,20,40,0.8));
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
        }

        /* ---------- Sidebar styling ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15,15,30,0.98), rgba(10,10,25,0.99));
        }

        /* ---------- Tab styling ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 20px;
            font-weight: 500;
        }

        /* ---------- Button ---------- */
        .stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1rem;
            padding: 0.6rem 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> str | None:
    """
    Render the sidebar with instructions and API key input.

    Returns:
        The Groq API key entered by the user, or None.
    """
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Enter your free Groq API key. Get one at console.groq.com",
        )

        st.markdown("---")
        st.markdown("## 📖 How It Works")
        st.markdown("""
        1. **Upload** your resume (PDF or DOCX)
        2. **Paste** the job description
        3. **Enter** company name & role title
        4. **Click Analyse** to get your results

        The system will:
        - 📊 Score your resume fit (0–100%)
        - 🔍 Identify skill gaps
        - ✍️ Rewrite your resume for the role
        - 💌 Generate a tailored cover letter
        """)

        st.markdown("---")
        st.markdown("## 🛠️ Tech Stack")
        st.markdown("""
        - **NLP**: spaCy + sentence-transformers
        - **LLM**: Groq (Llama 3.3 70B)
        - **Parsing**: pdfminer.six + python-docx
        """)

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center; color:#64748b; font-size:0.8rem;'>"
            "Built with ❤️ using Streamlit</p>",
            unsafe_allow_html=True,
        )

    return api_key if api_key else None


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main():
    """Main application entry point."""
    inject_custom_css()

    # Pre-load embedding model in background
    load_embedding_model()

    # Sidebar
    sidebar_api_key = render_sidebar()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Job Application Intelligence System</h1>
        <p>Upload your resume and paste a job description to get AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Initialize session state ---
    for key in [
        "match_score", "gaps", "rewritten_resume",
        "cover_letter", "resume_text", "analysis_done",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    # --- Input section ---
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📄 Your Resume")
        uploaded_file = st.file_uploader(
            "Upload your resume",
            type=["pdf", "docx", "txt"],
            help="Supports PDF, DOCX, and TXT formats",
            label_visibility="collapsed",
        )

        st.markdown("### 🏢 Company Details")
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            company_name = st.text_input(
                "Company Name",
                placeholder="e.g., Google",
            )
        with detail_col2:
            role_title = st.text_input(
                "Role Title",
                placeholder="e.g., Senior Software Engineer",
            )

    with col_right:
        st.markdown("### 📋 Job Description")
        jd_text = st.text_area(
            "Paste the job description here",
            height=280,
            placeholder="Paste the full job description here...",
            label_visibility="collapsed",
        )

    st.markdown("")  # Spacing

    # --- Analyse button ---
    analyse_clicked = st.button("🚀 Analyse & Generate", use_container_width=True)

    # --- Processing pipeline ---
    if analyse_clicked:
        # Validate inputs
        if not uploaded_file:
            st.error("📁 Please upload your resume (PDF or DOCX).")
            return
        if not jd_text or len(jd_text.strip()) < 50:
            st.error("📋 Please paste a complete job description (at least 50 characters).")
            return
        if not company_name.strip():
            st.error("🏢 Please enter the company name.")
            return
        if not role_title.strip():
            st.error("💼 Please enter the role title.")
            return

        api_key = sidebar_api_key

        # Run the pipeline
        with st.status("🔄 Analysing your application...", expanded=True) as status:
            try:
                # Step 1: Parse resume
                st.write("📄 Parsing resume...")
                resume_text = parse_resume(uploaded_file)
                if not resume_text or len(resume_text.strip()) < 20:
                    st.error("Could not extract meaningful text from your resume. Please try a different file.")
                    return
                st.session_state.resume_text = resume_text

                # Step 2: Extract skills
                st.write("🔍 Extracting skills from resume and JD...")
                resume_skills = extract_skills(resume_text)
                jd_skills = extract_skills(jd_text)

                # Step 3: Compute match score
                st.write("📊 Computing semantic match score...")
                match_score = compute_match_score(resume_text, jd_text)
                st.session_state.match_score = match_score

                # Step 4: Analyse gaps
                st.write("🔎 Analysing skill gaps...")
                gaps = analyse_gaps(resume_skills, jd_skills)
                st.session_state.gaps = gaps

                # Step 5: Rewrite resume
                st.write("✍️ Rewriting resume with AI (this may take a moment)...")
                rewritten = rewrite_resume(resume_text, jd_text, gaps, api_key)
                st.session_state.rewritten_resume = rewritten

                # Step 6: Generate cover letter
                st.write("💌 Generating personalised cover letter...")
                cover_letter = generate_cover_letter(
                    resume_text, jd_text, company_name.strip(), role_title.strip(), api_key
                )
                st.session_state.cover_letter = cover_letter

                st.session_state.analysis_done = True
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)

            except ValueError as e:
                st.error(f"⚠️ {e}")
                return
            except RuntimeError as e:
                st.error(f"🤖 AI Error: {e}")
                return
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                return

    # --- Results section ---
    if st.session_state.analysis_done:
        st.markdown("---")
        st.markdown("## 📊 Results")

        tab_score, tab_gaps, tab_resume, tab_letter = st.tabs([
            "📈 Match Score",
            "🔍 Skill Gaps",
            "✍️ Rewritten Resume",
            "💌 Cover Letter",
        ])

        # --- Tab 1: Match Score ---
        with tab_score:
            score = st.session_state.match_score
            gaps = st.session_state.gaps

            # Determine colour
            if score >= 70:
                color_class = "green"
            elif score >= 40:
                color_class = "yellow"
            else:
                color_class = "red"

            col_score, col_info = st.columns([1, 1], gap="large")

            with col_score:
                st.markdown(f"""
                <div class="score-card">
                    <div class="score-label">Semantic Match Score</div>
                    <div class="score-number score-{color_class}">{score}%</div>
                    <div class="progress-container">
                        <div class="progress-fill progress-{color_class}" style="width: {score}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_info:
                st.markdown(f"""
                <div class="info-card">
                    <div class="section-header">📊 Score Breakdown</div>
                    <p><strong>Semantic Similarity:</strong> {score}%</p>
                    <p><strong>Skill Coverage:</strong> {gaps['match_percentage']}%</p>
                    <p><strong>Matched Skills:</strong> {len(gaps['matched_skills'])}</p>
                    <p><strong>Missing Skills:</strong> {len(gaps['missing_skills'])}</p>
                </div>
                """, unsafe_allow_html=True)

                # Interpretation
                if score >= 70:
                    st.success("🎉 **Excellent match!** Your resume aligns well with this role.")
                elif score >= 40:
                    st.warning("⚡ **Moderate match.** Consider tailoring your resume to highlight relevant experience.")
                else:
                    st.error("📝 **Low match.** Significant tailoring is needed. Review the rewritten version below.")

        # --- Tab 2: Skill Gaps ---
        with tab_gaps:
            gaps = st.session_state.gaps

            col_matched, col_missing = st.columns(2, gap="large")

            with col_matched:
                st.markdown(f"""
                <div class="info-card">
                    <div class="section-header" style="color: #4ade80;">
                        ✅ Matched Skills ({len(gaps['matched_skills'])})
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if gaps["matched_skills"]:
                    badges_html = "".join(
                        f'<span class="skill-badge badge-green">{skill}</span>'
                        for skill in gaps["matched_skills"]
                    )
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.info("No matching skills detected.")

            with col_missing:
                st.markdown(f"""
                <div class="info-card">
                    <div class="section-header" style="color: #f87171;">
                        ❌ Missing Skills ({len(gaps['missing_skills'])})
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if gaps["missing_skills"]:
                    badges_html = "".join(
                        f'<span class="skill-badge badge-red">{skill}</span>'
                        for skill in gaps["missing_skills"]
                    )
                    st.markdown(badges_html, unsafe_allow_html=True)
                else:
                    st.success("🎉 No skill gaps detected! Your resume covers all required skills.")

            # Summary metrics
            st.markdown("---")
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Skill Match Rate", f"{gaps['match_percentage']}%")
            with metric_cols[1]:
                st.metric("Skills Matched", len(gaps['matched_skills']))
            with metric_cols[2]:
                st.metric("Skills to Add", len(gaps['missing_skills']))

        # --- Tab 3: Rewritten Resume ---
        with tab_resume:
            st.markdown("""
            <div class="info-card">
                <div class="section-header">✍️ AI-Rewritten Resume</div>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                    Tailored to the job description with optimised keywords and bullet points.
                    Copy the text below and refine as needed.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.text_area(
                "Rewritten Resume",
                value=st.session_state.rewritten_resume,
                height=500,
                label_visibility="collapsed",
                key="rewritten_resume_display",
            )

            st.download_button(
                label="📥 Download Rewritten Resume",
                data=st.session_state.rewritten_resume,
                file_name="rewritten_resume.txt",
                mime="text/plain",
            )

        # --- Tab 4: Cover Letter ---
        with tab_letter:
            st.markdown("""
            <div class="info-card">
                <div class="section-header">💌 Personalised Cover Letter</div>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                    Generated specifically for this role and company.
                    Review and personalise before submitting.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.text_area(
                "Cover Letter",
                value=st.session_state.cover_letter,
                height=400,
                label_visibility="collapsed",
                key="cover_letter_display",
            )

            st.download_button(
                label="📥 Download Cover Letter",
                data=st.session_state.cover_letter,
                file_name="cover_letter.txt",
                mime="text/plain",
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
