import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

_client = None


def _get_client():
    """Create the client on first use so a missing key doesn't break app startup."""
    global _client
    if _client is None:
        api_key = st.secrets["GEMINI_API_KEY"]
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        _client = genai.Client(api_key=api_key)
    return _client


def analyze_transcript(transcript):
    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty")

    prompt = f"""
You are an English speaking coach.

Analyze this speech transcript and provide only useful, actionable feedback.

TRANSCRIPT:
{transcript}

Focus ONLY on:

1. Summary
   - Summarize the main idea in 1-3 sentences.
   - Skip this section if there is no clear topic.

2. Grammar
   - Identify only important grammar mistakes.
   - Format:
     Original → Correction
     Brief explanation
   - Ignore minor or acceptable conversational grammar.
   - If there are no significant mistakes, say so.

3. Natural English
   - Identify phrases that are grammatically correct but noticeably unnatural.
   - Provide a more natural alternative.
   - Include only meaningful improvements.
   - If there are none, say so.

4. Suggestions
   - Give 2-3 specific, actionable suggestions based on the transcript.
   - Focus on realistic improvements to the speaker's English.

Do NOT analyze:
- Filler words
- Pauses
- Repeated words
- Speaking speed

These are already handled separately by the application.

Rules:
- Keep the response concise.
- Do not praise unnecessarily.
- Do not invent mistakes.
- Do not force corrections when the speaker's English is already natural.
"""

    response = _get_client().models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text