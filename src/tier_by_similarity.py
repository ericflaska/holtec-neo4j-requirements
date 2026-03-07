import random
from typing import Any

from src.schema import Entry

Tier = str


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _centroid(embeddings: list[list[float]], indices: list[int]) -> list[float]:
    if not indices:
        return []
    dim = len(embeddings[indices[0]])
    out = [0.0] * dim
    for i in indices:
        if i < len(embeddings):
            for j, x in enumerate(embeddings[i]):
                out[j] += x
    n = len(indices)
    return [x / n for x in out]


def get_tier_exemplars(
    requirements: list[Entry],
    sample_size: int = 60,
    client: Any = None,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Run Nova on a random sample to get labeled exemplars per tier. Returns tier -> list of indices."""
    from src.nova_tier import assign_tiers_to_requirements_with_fallback, get_bedrock_client
    n = len(requirements)
    if n == 0:
        return {"plant": [], "system": [], "component": []}
    k = min(sample_size, n)
    rng = random.Random(seed)
    indices = rng.sample(range(n), k)
    sample_entries = [requirements[i] for i in indices]
    try:
        labeled = assign_tiers_to_requirements_with_fallback(sample_entries, default_tier="system", client=client)
    except Exception:
        labeled = [(e, "system") for e in sample_entries]
    by_tier: dict[str, list[int]] = {"plant": [], "system": [], "component": []}
    for (_, tier), idx in zip(labeled, indices):
        t = (tier or "system").strip().lower()
        if t not in by_tier:
            by_tier[t] = []
        by_tier.setdefault(t, []).append(idx)
    for t in ("plant", "system", "component"):
        if t not in by_tier:
            by_tier[t] = []
    return by_tier


def assign_tiers_by_similarity(
    requirements: list[Entry],
    embeddings: list[list[float]],
    exemplar_indices_by_tier: dict[str, list[int]],
) -> list[tuple[Entry, Tier]]:
    """Assign each requirement to the tier whose exemplar centroid has highest cosine similarity to its embedding."""
    if len(embeddings) != len(requirements):
        return [(e, "system") for e in requirements]
    centroids = {}
    for tier, indices in exemplar_indices_by_tier.items():
        if indices:
            centroids[tier] = _centroid(embeddings, indices)
        else:
            centroids[tier] = []
    result = []
    for i, e in enumerate(requirements):
        vec = embeddings[i] if i < len(embeddings) else []
        best_tier = "system"
        best_sim = -2.0
        for tier, cent in centroids.items():
            if not cent:
                continue
            sim = _cosine_sim(vec, cent)
            if sim > best_sim:
                best_sim = sim
                best_tier = tier
        result.append((e, best_tier))
    return result


def assign_tiers_by_similarity_with_fallback(
    requirements: list[Entry],
    embeddings: list[list[float]],
    sample_size: int = 60,
    client: Any = None,
    default_tier: Tier = "system",
) -> list[tuple[Entry, Tier]]:
    """Get exemplars via Nova on a sample, then assign all requirements by cosine similarity to tier centroids."""
    if not requirements or not embeddings or len(embeddings) != len(requirements):
        return [(e, default_tier) for e in requirements]
    try:
        exemplars = get_tier_exemplars(requirements, sample_size=sample_size, client=client)
        return assign_tiers_by_similarity(requirements, embeddings, exemplars)
    except Exception as err:
        print(f"[Tier-by-similarity failed] {err}. Using tier '{default_tier}' for all.")
        return [(e, default_tier) for e in requirements]
