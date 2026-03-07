import json
import re
from typing import Any

from src.schema import Entry

Tier = str

TRACE_PROMPT_PREFIX = """You are a requirements engineer. Given a list of requirements with index and tier (plant, system, or component), identify which pairs should be connected by a traceability link.

Rules: Only output pairs with clear traceability (e.g. one refines or implements the other). Return JSON: {"pairs": [[from_index, to_index], ...]}. Use 0-based indices. If no pairs, return {"pairs": []}.

Requirements (index, tier, text snippet):
"""
TRACE_PROMPT_SUFFIX = "\n\nReturn only the JSON object, no other text."


def _parse_pairs(raw: str) -> list[tuple[int, int]]:
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
    pairs = obj.get("pairs") or obj.get("pairs_list") or []
    out = []
    for p in pairs:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                i, j = int(p[0]), int(p[1])
                if i != j:
                    out.append((i, j))
            except (ValueError, TypeError):
                pass
    return out


def get_trace_pairs_from_nova(
    requirements_with_tiers: list[tuple[Entry, Tier]],
    client: Any | None = None,
    batch_size: int = 12,
) -> list[tuple[int, int]]:
    from config import NOVA_MODEL_ID
    from src.nova_tier import _nova_invoke, get_bedrock_client
    if client is None:
        client = get_bedrock_client()
    all_pairs = []
    n = len(requirements_with_tiers)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = requirements_with_tiers[start:end]
        lines = [f"{i}: [{tier}] {(e.text or '')[:300].replace(chr(10), ' ')}" for i, (e, tier) in enumerate(batch)]
        raw = _nova_invoke(client, NOVA_MODEL_ID, TRACE_PROMPT_PREFIX + "\n".join(lines) + TRACE_PROMPT_SUFFIX, max_tokens=1024)
        if not raw:
            continue
        for i, j in _parse_pairs(raw):
            if 0 <= i < len(batch) and 0 <= j < len(batch):
                all_pairs.append((start + i, start + j))
    return all_pairs


def get_trace_pairs_from_nova_with_fallback(
    requirements_with_tiers: list[tuple[Entry, Tier]],
    client: Any | None = None,
    batch_size: int = 12,
) -> list[tuple[int, int]]:
    try:
        return get_trace_pairs_from_nova(requirements_with_tiers, client=client, batch_size=batch_size)
    except Exception as err:
        if "Credential" in type(err).__name__ or "credentials" in str(err).lower():
            print("[Nova traces skipped] No AWS credentials.")
        else:
            print(f"[Nova traces failed] {err}")
        return []
