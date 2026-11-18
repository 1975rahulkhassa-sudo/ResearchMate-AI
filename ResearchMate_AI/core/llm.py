"""Provider-configurable LLM client (OpenAI / Gemini) via REST. Returns None on any failure."""
import os, json, re, logging, requests
from dotenv import load_dotenv
load_dotenv()
log = logging.getLogger("llm")

def available():
    p = os.getenv("LLM_PROVIDER", "openai").lower()
    return bool(os.getenv("GEMINI_API_KEY" if p == "gemini" else "OPENAI_API_KEY"))

def llm(prompt, system="You are a careful research assistant. Never invent sources."):
    p = os.getenv("LLM_PROVIDER", "openai").lower()
    try:
        if not available():
            return None
        if p == "gemini":
            m = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
                json={"contents": [{"parts": [{"text": system + "\n\n" + prompt}]}]}, timeout=60)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        r = requests.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
            json={"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:  # never log keys
        log.warning("LLM call failed: %s", type(e).__name__)
        return None

def llm_json(prompt):
    out = llm(prompt + "\nRespond with JSON only.")
    if not out:
        return None
    try:
        return json.loads(re.sub(r"^```(?:json)?|```$", "", out.strip(), flags=re.M).strip())
    except Exception:
        return None
