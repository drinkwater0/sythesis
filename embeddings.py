from config import EMBEDDING_BACKEND

if EMBEDDING_BACKEND == "pubmedbert":
    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer("neuml/pubmedbert-base-embeddings")

    def embed_documents(texts: list[str]) -> list[list[float]]:
        return _model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(text: str) -> list[float]:
        return _model.encode([text], normalize_embeddings=True)[0].tolist()

elif EMBEDDING_BACKEND == "medcpt":
    import torch
    from transformers import AutoModel, AutoTokenizer

    _doc_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    _doc_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").eval()
    _query_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    _query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").eval()

    @torch.no_grad()
    def _encode(model, tok, texts: list[str], max_length: int) -> list[list[float]]:
        enc = tok(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        embs = model(**enc).last_hidden_state[:, 0, :]
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        return embs.tolist()

    # MedCPT is asymmetric: 512 tokens for passages, 64 for queries.
    def embed_documents(texts: list[str]) -> list[list[float]]:
        return _encode(_doc_model, _doc_tok, texts, max_length=512)

    def embed_query(text: str) -> list[float]:
        return _encode(_query_model, _query_tok, [text], max_length=64)[0]

else:
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {EMBEDDING_BACKEND!r}")
