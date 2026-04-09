"""Utility functions for building a resume outreach agent."""

from typing import Any, Dict
import httpx
import json
import csv


# Leaderboard base URL
LEADERBOARD_BASE_URL = "http://ai-leaderboard.site"

# The optimized scoring prompt from Lecture 3.
# This is a template — {job_req} gets filled in at runtime.
SCORING_PROMPT_TEMPLATE = """Score this resume against the job requirements below. Return a JSON object with exactly two fields: "score" (integer 0-100) and "reasoning" (1-3 sentences).

---

JOB REQUIREMENTS:
{job_req}

---

SCORING RULES — follow these exactly:

**STEP 1: Check for CORE DISQUALIFIERS (missing any = heavy penalty)**
- No C#/.NET experience → cap score at 55
- No SQL/database experience → cap score at 60
- No backend development experience at all → cap score at 60
- Less than 3 years total experience → cap score at 75

**STEP 2: Assign a BASE SCORE using these anchors**

0-30 = Completely unrelated field (e.g., nurse, teacher, accountant with no tech skills)
31-49 = Tech background but wrong domain (e.g., embedded systems, data science only, mobile-only with no web)
50-69 = Partial tech match but missing multiple core requirements (e.g., backend dev but wrong stack like Java/Python only, no .NET, no SQL)
70-79 = Significant gaps in core requirements. Use this range for:
  - Junior developer (1-3 years experience) who otherwise matches the stack
  - Frontend-only specialist (5+ years JS/React/Angular) with NO backend or .NET experience
  - Candidate with .NET experience but only 2-3 years and missing SQL or cloud
80-84 = Meets most requirements but has notable gaps (e.g., strong .NET/SQL but no frontend framework, OR strong full-stack but only 4 years experience)
85-92 = Strong match: 5+ years, has C#/.NET, SQL, at least one JS framework, Git/CI-CD. May lack AWS or microservices.
93-100 = Excellent match: 8+ years, C#/.NET, SQL Server, React or Angular, AWS, CI-CD, Agile. Matches nearly all required AND preferred qualifications.

**STEP 3: Apply BONUSES (only if base score is 50+)**
- AWS experience (EC2, S3, Lambda, RDS): +3
- Microservices architecture experience: +2
- Security knowledge (OAuth, JWT, OWASP): +2
- Testing frameworks (xUnit, NUnit, Jest): +1
- Docker/containerization: +1
- Do not exceed 100

**STEP 4: Apply PENALTIES**
- Claims skills but no evidence of use in real projects: -5
- Only 3-4 years experience (below 5-year requirement): -5
- No mention of Agile/Scrum: -2
- Frontend-only with zero backend evidence: -10 (in addition to Step 1 cap)

---

SCORING EXAMPLES to calibrate your output:

- Resume: 9 years C#/.NET, React, SQL Server, AWS, CI-CD, Agile → Score: 93-97
- Resume: 6 years C#/.NET, Angular, MySQL, Git, no AWS → Score: 85-89
- Resume: 10 years React/Angular/Vue frontend only, no .NET, no SQL → Score: 70-74
- Resume: 1.5 years experience, some C# and SQL, learning React → Score: 70-74
- Resume: 8 years Java/Spring backend, no .NET, strong SQL → Score: 50-58
- Resume: 5 years Python/Django, no .NET, no SQL Server → Score: 45-53
- Resume: Registered nurse with no software development experience → Score: 5-15

---

BE STRICT: Do not inflate scores. A frontend-only developer, no matter how experienced, must score 70-79 because they lack backend/.NET skills.
A junior developer under 3 years must score 70-79 even with the right stack.
Only candidates with 5+ years AND C#/.NET AND SQL AND a JS framework should score 85+."""


def load_resumes(csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load all resumes from CSV into a dictionary.

    Args:
        csv_path: Path to the resumes CSV file

    Returns:
        Dict mapping resume ID to resume data (ID, Resume_str, Resume_html)
    """
    resumes = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            resumes[row['ID']] = {
                'ID': row['ID'],
                'Resume_str': row['Resume_str'],
                'Resume_html': row['Resume_html']
            }
    return resumes


def load_job_requirements(file_path: str) -> str:
    """Load job requirements from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def structured_llm_call(
    api_key: str,
    prompt: str,
    context_data: Dict[str, Any],
    output_schema: Dict[str, Any],
    model: str = "anthropic/claude-sonnet-4.6",
    temperature: float = 0.2
) -> Dict[str, Any]:
    """
    Generic function for making structured LLM calls with OpenRouter.

    Args:
        api_key: OpenRouter API key
        prompt: The instruction/task description
        context_data: Dictionary of context (e.g., {'resume': '...', 'job_req': '...'})
        output_schema: Dictionary describing the expected JSON structure
        model: Model to use
        temperature: Sampling temperature

    Returns:
        Dict with 'result', 'error', and 'usage'
    """
    # Build context section
    context_str = ""
    for key, value in context_data.items():
        if isinstance(value, str) and len(value) > 5000:
            value = value[:5000] + "\n... (truncated)"
        context_str += f"\n{key.upper()}:\n{value}\n"

    # Build schema description
    schema_str = json.dumps(output_schema, indent=2)

    # Construct full prompt
    full_prompt = f"""{prompt}

{context_str}

Return a JSON object with this exact structure:
{schema_str}

IMPORTANT: Return ONLY valid JSON, no additional text or markdown formatting."""

    # Make API call
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": temperature,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"}
    }

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)

            return {
                "result": result,
                "error": None,
                "usage": data.get("usage", {})
            }
    except Exception as e:
        return {
            "result": None,
            "error": str(e),
            "usage": {}
        }


def submit_outreach(team_name, resume_id, outcome, email_text, score=None, cost=None, api_key="leaderboard-api-key"):
    """Submit an outreach email to the lecture 4 leaderboard."""
    url = f"{LEADERBOARD_BASE_URL}/lecture4/api/submit"
    payload = {
        "team_name": team_name,
        "resume_id": resume_id,
        "outcome": outcome,
        "email_text": email_text,
        "score": score,
        "cost": cost,
    }
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(url, json=payload, headers={"X-API-Key": api_key})
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


def delete_outreach(team_name, resume_id, api_key="leaderboard-api-key"):
    """Delete a single outreach submission."""
    url = f"{LEADERBOARD_BASE_URL}/lecture4/api/submit"
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.request("DELETE", url, json={"team_name": team_name, "resume_id": resume_id}, headers={"X-API-Key": api_key})
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


def delete_team(team_name, api_key="leaderboard-api-key"):
    """Delete all submissions for a team."""
    url = f"{LEADERBOARD_BASE_URL}/lecture4/api/delete_team"
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(url, json={"team_name": team_name}, headers={"X-API-Key": api_key})
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return {"error": str(e)}
