import json
from pathlib import Path
from typing import Any

from src.schema import Entry

Tier = str


def _invoke_titan(client: Any, model_id: str, text: str, dimensions: int = 1024) -> list[float]:
    body = {"inputText": (text or "")[:8000], "dimensions": dimensions}
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    out = json.loads(response["body"].read())
    return out.get("embedding") or []


def get_bedrock_client():
    from src.nova_tier import get_bedrock_client as _get
    return _get()


def embed_text(text: str, client: Any = None, model_id: str | None = None, dimensions: int = 1024) -> list[float]:
    from config import TITAN_EMBED_MODEL_ID
    if client is None:
        client = get_bedrock_client()
    model_id = model_id or TITAN_EMBED_MODEL_ID
    return _invoke_titan(client, model_id, text or "", dimensions=dimensions)


def embed_entries(
    entries: list[Entry],
    client: Any = None,
    model_id: str | None = None,
    dimensions: int = 1024,
) -> list[list[float]]:
    from config import TITAN_EMBED_MODEL_ID
    if client is None:
        client = get_bedrock_client()
    model_id = model_id or TITAN_EMBED_MODEL_ID
    embeddings = []
    for i, e in enumerate(entries):
        try:
            vec = _invoke_titan(client, model_id, e.text or "", dimensions=dimensions)
            embeddings.append(vec)
        except Exception as err:
            embeddings.append([0.0] * dimensions)
            if i < 3:
                print(f"[Titan embed] {err} for req {i}, using zero vector.")
    return embeddings


def embed_requirements(
    requirements_with_tiers: list[tuple[Entry, Tier]],
    client: Any = None,
    model_id: str | None = None,
    dimensions: int = 1024,
) -> list[list[float]]:
    entries = [e for e, _ in requirements_with_tiers]
    return embed_entries(entries, client=client, model_id=model_id, dimensions=dimensions)


def embeddings_cache_path(root: Path, doc_id: str) -> Path:
    root = Path(root)
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    safe = (doc_id or "default").replace(" ", "_")[:64]
    return data_dir / f"embeddings_{safe}.json"


def load_embeddings_from_cache(cache_path: Path, doc_id: str, expected_len: int) -> list[list[float]] | None:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("doc_id") != doc_id or len(data.get("embeddings") or []) != expected_len:
            return None
        return data["embeddings"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def save_embeddings_to_cache(cache_path: Path, doc_id: str, embeddings: list[list[float]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dim = len(embeddings[0]) if embeddings else 0
    cache_path.write_text(
        json.dumps({"doc_id": doc_id, "dim": dim, "embeddings": embeddings}, indent=0),
        encoding="utf-8",
    )


def get_embeddings_for_requirements(
    requirements_with_tiers: list[tuple[Entry, Tier]],
    cache_path: Path,
    doc_id: str,
    client: Any = None,
    dimensions: int = 1024,
) -> list[list[float]]:
    entries = [e for e, _ in requirements_with_tiers]
    return get_embeddings_for_entries(entries, cache_path, doc_id, client=client, dimensions=dimensions)


def get_embeddings_for_entries(
    entries: list[Entry],
    cache_path: Path,
    doc_id: str,
    client: Any = None,
    dimensions: int = 1024,
) -> list[list[float]]:
    cached = load_embeddings_from_cache(cache_path, doc_id, len(entries))
    if cached is not None:
        print(f"Loaded {len(cached)} embeddings from {cache_path}.")
        return cached
    try:
        embeddings = embed_entries(entries, client=client, dimensions=dimensions)
        save_embeddings_to_cache(cache_path, doc_id, embeddings)
        print(f"Computed {len(embeddings)} embeddings (dim={dimensions}), saved to {cache_path}.")
        return embeddings
    except Exception as err:
        print(f"[Titan embeddings failed] {err}. Proceeding without embeddings.")
        return []
