"""Minimal OpenRouter chat helper (self-contained). Set OPENROUTER_API_KEY in a .env file."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.getenv("MODEL", "meta-llama/llama-3.3-70b-instruct")


def chat(system, user, max_tokens=180, model=None, temperature=0.4, retries=3):
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set (see .env.example)")
    body = {
        "model": model or DEFAULT_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=body, timeout=90)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last = e
    raise RuntimeError(f"OpenRouter call failed after {retries} tries: {last}")
