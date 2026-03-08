import json
import re
from typing import Any

from src.schema import Entry

Tier = str

ALLOWED_TIER_PAIRS = {("component", "system"), ("system", "component"), ("system", "plant"), ("plant", "system")}


def _cosine_sim(a: list[float], b: list[float]) -> float:
    try:
        import numpy as np
        va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)
        return float(n)
    except ImportError:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na * nb <= 0:
            return 0.0
        return dot / (na * nb)


def _top_k_indices(
    query_emb: list[float],
    candidate_indices: list[int],
    embeddings: list[list[float]],
    k: int,
) -> list[int]:
    scored = [(cand_idx, _cosine_sim(query_emb, embeddings[cand_idx])) for cand_idx in candidate_indices]
    scored.sort(key=lambda x: -x[1])
    return [idx for idx, _ in scored[:k]]


def get_tier_indices(
    requirements_with_tiers: list[tuple[Entry, Tier]],
) -> tuple[list[int], list[int], list[int]]:
    plant_idx = [i for i, (_, t) in enumerate(requirements_with_tiers) if (t or "").lower() == "plant"]
    system_idx = [i for i, (_, t) in enumerate(requirements_with_tiers) if (t or "").lower() == "system"]
    component_idx = [i for i, (_, t) in enumerate(requirements_with_tiers) if (t or "").lower() == "component"]
    return plant_idx, system_idx, component_idx


def get_candidate_trace_pairs(
    requirements_with_tiers: list[tuple[Entry, Tier]],
    embeddings: list[list[float]],
    k: int = 3,
) -> list[tuple[int, int]]:
    if len(embeddings) != len(requirements_with_tiers):
        return []
    plant_idx, system_idx, component_idx = get_tier_indices(requirements_with_tiers)
    pairs: list[tuple[int, int]] = []
    for i in component_idx:
        top = _top_k_indices(embeddings[i], system_idx, embeddings, k)
        for j in top:
            pairs.append((i, j))
    for i in system_idx:
        top = _top_k_indices(embeddings[i], plant_idx, embeddings, k)
        for j in top:
            pairs.append((i, j))
    return pairs


def _parse_nova_pairs(raw: str) -> list[tuple[int, int]]:
    raw = (raw or "").strip()
    obj = None
    if raw.startswith("{"):
        end = raw.rfind("}")
        if end != -1:
            try:
                obj = json.loads(raw[: end + 1])
            except json.JSONDecodeError:
                pass
    if obj is None:
        match = re.search(r"\[\s*\[[\d\s,]+\](?:\s*,\s*\[[\d\s,]+\])*\s*\]", raw)
        if match:
            try:
                arr = json.loads(match.group(0))
                if isinstance(arr, list):
                    obj = {"pairs": arr}
            except json.JSONDecodeError:
                pass
    if obj is None or not isinstance(obj, dict):
        return []
    out = []
    for p in obj.get("pairs") or obj.get("pairs_list") or []:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                i, j = int(p[0]), int(p[1])
                if i != j:
                    out.append((i, j))
            except (ValueError, TypeError):
                pass
    return out


TRACE_VERIFY_PROMPT = """You are a requirements engineer. Below are requirements with their index and tier. We list candidate traceability pairs (from_index, to_index). Only adjacent tiers are valid: component-system or system-plant. For each candidate pair, decide if there is a real traceability link (one refines or implements the other). Return JSON: {"pairs": [[from_index, to_index], ...]} containing ONLY the pairs that should be linked. If none, return {"pairs": []}.

Requirements (index, tier, text snippet):
"""
TRACE_VERIFY_SUFFIX = "\n\nCandidate pairs to evaluate (return only those that are valid traceability links):\n"
TRACE_VERIFY_END = "\n\nReturn only the JSON object, no other text."


def filter_candidates_with_nova(
    requirements_with_tiers: list[tuple[Entry, Tier]],
    candidate_pairs: list[tuple[int, int]],
    client: Any | None = None,
    batch_size: int = 20,
) -> list[tuple[int, int]]:
    from config import NOVA_MODEL_ID
    from src.nova_tier import _nova_invoke, get_bedrock_client

    if client is None:
        client = get_bedrock_client()

    tiers = [t for _, t in requirements_with_tiers]
    n = len(requirements_with_tiers)
    approved: list[tuple[int, int]] = []
    for start in range(0, len(candidate_pairs), batch_size):
        batch_pairs = candidate_pairs[start : start + batch_size]
        indices_needed = sorted(set(i for p in batch_pairs for i in p))
        lines = []
        for idx in indices_needed:
            if idx < 0 or idx >= n:
                continue
            e, t = requirements_with_tiers[idx]
            snippet = (e.text or "")[:400].replace("\n", " ")
            lines.append(f"{idx}: [{t}] {snippet}")
        pairs_str = json.dumps(batch_pairs)
        prompt = TRACE_VERIFY_PROMPT + "\n".join(lines) + TRACE_VERIFY_SUFFIX + pairs_str + TRACE_VERIFY_END
        raw = _nova_invoke(client, NOVA_MODEL_ID, prompt, max_tokens=1024)
        for i, j in _parse_nova_pairs(raw):
            if 0 <= i < n and 0 <= j < n:
                ti, tj = tiers[i].lower(), tiers[j].lower()
                if (ti, tj) in ALLOWED_TIER_PAIRS:
                    approved.append((i, j))
    return approved


def get_trace_pairs_similarity_nova(
    requirements_with_tiers: list[tuple[Entry, Tier]],
    embeddings: list[list[float]],
    k: int = 3,
    client: Any | None = None,
    nova_batch_size: int = 20,
) -> list[tuple[int, int]]:
    candidates = get_candidate_trace_pairs(requirements_with_tiers, embeddings, k=k)
    if not candidates:
        return []
    return filter_candidates_with_nova(
        requirements_with_tiers, candidates, client=client, batch_size=nova_batch_size
    )


def get_trace_pairs_similarity_nova_with_fallback(
    requirements_with_tiers: list[tuple[Entry, Tier]],
    embeddings: list[list[float]],
    k: int = 3,
    client: Any | None = None,
    nova_batch_size: int = 20,
) -> list[tuple[int, int]]:
    try:
        return get_trace_pairs_similarity_nova(
            requirements_with_tiers, embeddings, k=k, client=client, nova_batch_size=nova_batch_size
        )
    except Exception as err:
        if "Credential" in type(err).__name__ or "credentials" in str(err).lower():
            print("[Nova trace verify skipped] No AWS credentials.")
        else:
            print(f"[Nova trace verify failed] {err}")
        return []


def get_candidates_per_requirement(
    requirements_with_tiers: list[tuple[Entry, Tier]],
    embeddings: list[list[float]],
    k: int = 5,
) -> list[tuple[int, list[int]]]:
    if len(embeddings) != len(requirements_with_tiers):
        return []
    plant_idx, system_idx, component_idx = get_tier_indices(requirements_with_tiers)
    out: list[tuple[int, list[int]]] = []
    for i in component_idx:
        top = _top_k_indices(embeddings[i], system_idx, embeddings, k)
        if top:
            out.append((i, top))
    for i in system_idx:
        top = _top_k_indices(embeddings[i], plant_idx, embeddings, k)
        if top:
            out.append((i, top))
    return out


SINGLE_CHOICE_PROMPT = """You are a requirements engineer. For each requirement below, we give exactly 5 candidate requirements from the level above (system if this is component, plant if this is system). Choose the ONE candidate (0, 1, 2, 3, or 4) that this requirement traces to. Reply with one number per requirement, in order, comma-separated. Example: 2,0,4,1

"""
SINGLE_CHOICE_END = "\n\nReply with only the comma-separated numbers (one per requirement), nothing else."


def _parse_single_choices(raw: str, count: int) -> list[int]:
    raw = (raw or "").strip()
    parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
    result = []
    for i in range(count):
        if i < len(parts):
            try:
                v = int(parts[i])
                if 0 <= v <= 4:
                    result.append(v)
                else:
                    result.append(0)
            except ValueError:
                result.append(0)
        else:
            result.append(0)
    return result[:count]


def _nova_pick_single_batch(
    requirements_with_tiers: list[tuple[Entry, Tier]],
    batch: list[tuple[int, list[int]]],
    client: Any,
) -> list[tuple[int, int]]:
    from config import NOVA_MODEL_ID
    from src.nova_tier import _nova_invoke

    n = len(requirements_with_tiers)
    lines = []
    for req_idx, candidate_indices in batch:
        e, tier = requirements_with_tiers[req_idx]
        lines.append(f"Requirement ({tier}): {(e.text or '')[:350].replace(chr(10), ' ')}")
        for opt, cand_idx in enumerate(candidate_indices):
            if 0 <= cand_idx < n:
                ce, ct = requirements_with_tiers[cand_idx]
                lines.append(f"  Option {opt}: {(ce.text or '')[:300].replace(chr(10), ' ')}")
        lines.append("")
    prompt = SINGLE_CHOICE_PROMPT + "\n".join(lines) + SINGLE_CHOICE_END
    raw = _nova_invoke(client, NOVA_MODEL_ID, prompt, max_tokens=256)
    choices = _parse_single_choices(raw, len(batch))
    tiers = [(t or "").strip().lower() for _, t in requirements_with_tiers]
    pairs = []
    for (source_idx, candidate_indices), choice in zip(batch, choices):
        if 0 <= choice < len(candidate_indices):
            target_idx = candidate_indices[choice]
        elif candidate_indices:
            target_idx = candidate_indices[0]
        else:
            continue
        if source_idx >= len(tiers) or target_idx >= len(tiers) or tiers[source_idx] == tiers[target_idx]:
            continue
        pairs.append((source_idx, target_idx))
    return pairs


def get_trace_pairs_single_choice_nova(
    requirements_with_tiers: list[tuple[Entry, Tier]],
    embeddings: list[list[float]],
    k: int = 5,
    client: Any | None = None,
    nova_batch_size: int = 5,
) -> list[tuple[int, int]]:
    from src.nova_tier import get_bedrock_client

    if client is None:
        client = get_bedrock_client()

    per_req = get_candidates_per_requirement(requirements_with_tiers, embeddings, k=k)
    if not per_req:
        return []

    all_pairs: list[tuple[int, int]] = []
    for start in range(0, len(per_req), nova_batch_size):
        batch = per_req[start : start + nova_batch_size]
        pairs = _nova_pick_single_batch(requirements_with_tiers, batch, client)
        all_pairs.extend(pairs)
    return all_pairs


def get_trace_pairs_single_choice_nova_with_fallback(
    requirements_with_tiers: list[tuple[Entry, Tier]],
    embeddings: list[list[float]],
    k: int = 5,
    client: Any | None = None,
    nova_batch_size: int = 5,
) -> list[tuple[int, int]]:
    try:
        return get_trace_pairs_single_choice_nova(
            requirements_with_tiers, embeddings, k=k, client=client, nova_batch_size=nova_batch_size
        )
    except Exception as err:
        if "Credential" in type(err).__name__ or "credentials" in str(err).lower():
            print("[Nova single-choice skipped] No AWS credentials.")
        else:
            print(f"[Nova single-choice failed] {err}")
        return []
