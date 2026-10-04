import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

LLM_MODEL = "openai/gpt-oss-20b"
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing from the environment or .env file.")
        _client = Groq(api_key=api_key)
    return _client


def _ask_json(system_prompt: str, user_prompt: str) -> dict:
    response = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    return json.loads(response.choices[0].message.content)


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def first_question(topic: str, context: str = "") -> str:
    result = _ask_json(
        "You are a concise interviewer. Return JSON with one string field: question.",
        f"Create the opening interview question for this role or topic: {topic}\n"
        f"Optional candidate/job context: {context}",
    )
    return str(result.get("question", "Tell me about your experience with this topic."))


def next_turn(topic: str, context: str, history: list[dict], is_last: bool) -> dict:
    result = _ask_json(
        "You are an interview coach. Evaluate the candidate's latest answer fairly. "
        "Return JSON fields: acknowledgement (string), score (integer 0-10), "
        "feedback (string), missing_points (array of strings), "
        "next_question (string). If this is the final turn, next_question may be empty.",
        f"Role/topic: {topic}\nContext: {context}\n"
        f"Interview history as JSON: {json.dumps(history, ensure_ascii=True)}\n"
        f"Is this the final turn? {is_last}",
    )
    try:
        score = max(0, min(10, int(result.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    return {
        "acknowledgement": str(result.get("acknowledgement", "Thank you for your answer.")),
        "score": score,
        "feedback": str(result.get("feedback", "")),
        "missing_points": _string_list(result.get("missing_points")),
        "next_question": str(result.get("next_question", "")),
    }


def final_report(topic: str, history: list[dict]) -> dict:
    result = _ask_json(
        "You are an interview coach. Summarize performance constructively. "
        "Return JSON fields: overall_score, content_score, communication_score "
        "(integers 0-100), summary (string), strengths (array of strings), "
        "weaknesses (array of strings), missing_points (array of strings), "
        "fluency_feedback (string).",
        f"Role/topic: {topic}\nInterview history as JSON: "
        f"{json.dumps(history, ensure_ascii=True)}",
    )
    report = {
        "summary": str(result.get("summary", "")),
        "strengths": _string_list(result.get("strengths")),
        "weaknesses": _string_list(result.get("weaknesses")),
        "missing_points": _string_list(result.get("missing_points")),
        "fluency_feedback": str(result.get("fluency_feedback", "")),
    }
    for field in ("overall_score", "content_score", "communication_score"):
        try:
            report[field] = max(0, min(100, int(result.get(field, 0))))
        except (TypeError, ValueError):
            report[field] = 0
    return report
