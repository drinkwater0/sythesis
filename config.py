from pathlib import Path

PRIMARY_CORPUS = "laminopathy"
SIDE_CORPORA = ["lnp", "bioinformatics"]
ALL_CORPORA = [PRIMARY_CORPUS, *SIDE_CORPORA]

# Embedding backend: "pubmedbert" (symmetric) or "medcpt" (asymmetric article/query)
EMBEDDING_BACKEND = "pubmedbert"

# Resolved relative to this file so cwd (notebook vs repo root) doesn't matter.
CHROMA_PATH = str(Path(__file__).parent / "data" / "chroma_db")
CHROMA_COLLECTION = "laminorag"

PRIMARY_CHUNKS = 6
SIDE_CHUNKS = 2

# LLM via uni AI gateway (OpenAI-compatible). Value is whatever identifier
# the gateway exposes. LLM_API_BASE and LLM_API_KEY come from .env.
LLM_MODEL = "qwen3.6-35b"
