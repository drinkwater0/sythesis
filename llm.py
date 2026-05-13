import os

from dotenv import load_dotenv
from openai import OpenAI

from config import LLM_MODEL

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    # Lazy so `import llm` works before .env is populated.
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ["LLM_API_BASE"],
            api_key=os.environ["LLM_API_KEY"],
        )
    return _client


def complete(prompt: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content
