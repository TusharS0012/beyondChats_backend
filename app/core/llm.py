import os
import requests
import json
from typing import List, Dict

HF_MODEL = os.getenv("HF_LLM_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 256))

HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def query_hf_api(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error {response.status_code}: {response.text}"


def answer_chat_query(query: str, context_snippets: List[dict] = []) -> str:
    prompt = ""
    if context_snippets:
        snippet_texts = " ".join(
            [f"According to p.{s.get('page','?')}: {s['text']}" for s in context_snippets]
        )
        prompt += f"Context: {snippet_texts}\n"
    prompt += f"User: {query}\nAssistant:"

    result = query_hf_api(prompt)
    answer = result[len(prompt):].strip()
    return answer


def generate_quiz_from_text(text: str) -> Dict[str, List[dict]]:
    prompt = (
        f"Generate MCQs, SAQs, LAQs from the following text:\n{text}\n"
        "Format the output as valid JSON with keys 'mcq', 'saq', 'laq'. "
        "Each MCQ should have 'question', 'options', and 'answer'. "
        "Each SAQ/LAQ should have 'question' and 'answer'."
    )

    result_text = query_hf_api(prompt)

    try:
        quiz_json = json.loads(result_text)
        # Ensure all keys exist
        return {
            "mcq": quiz_json.get("mcq", []),
            "saq": quiz_json.get("saq", []),
            "laq": quiz_json.get("laq", [])
        }
    except json.JSONDecodeError:
        # If parsing fails, return placeholders
        return {
            "mcq": [{"question": "Sample MCQ?", "options": ["A","B","C"], "answer": "A"}],
            "saq": [{"question": "Sample SAQ?", "answer": "Sample answer"}],
            "laq": [{"question": "Sample LAQ?", "answer": "Sample long answer"}]
        }
