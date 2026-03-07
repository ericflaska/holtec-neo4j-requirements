import json
import re
from typing import Literal

from src.schema import Entry

Tier = Literal["plant", "system", "component"]

TIER_PROMPT = """You are classifying engineering requirements by level.

Given the following requirement text, respond with exactly one word: plant, system, or component.

- plant: overall facility/site-level (safety, power, monitoring at the highest level)
- system: major systems (electrical, cooling, control, safety, monitoring systems)
- component: specific equipment or devices (pumps, valves, sensors, controllers, breakers)

Requirement text:
"""


def _normalize_tier(raw: str) -> Tier:
    raw = (raw or "").strip().lower()
    match = re.search(r"\b(plant|system|component)\b", raw)
    if match:
        return match.group(1)
    if "plant" in raw: return "plant"
    if "system" in raw: return "system"
    if "component" in raw: return "component"
    return "component"


def _nova_invoke(client, model_id: str, prompt: str, max_tokens: int = 20) -> str:
    body = {
        "schemaVersion": "messages-v1",
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0},
    }
    if hasattr(client, "converse"):
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        output = response.get("output") or {}
        message = output.get("message") or {}
        content = message.get("content") or []
        if not content:
            return ""
        block = content[0] if isinstance(content[0], dict) else {}
        return (block.get("text") or "").strip() if isinstance(block.get("text"), str) else str(content[0])
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    out = json.loads(response["body"].read())
    output = out.get("output") or {}
    message = output.get("message") or {}
    content = message.get("content") or []
    if not content:
        return ""
    block = content[0] if isinstance(content[0], dict) else {}
    return (block.get("text") or "").strip() if isinstance(block.get("text"), str) else str(content[0])


def get_bedrock_client():
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise ImportError("boto3 required. pip install boto3")
    from config import AWS_REGION, BEDROCK_AWS_PROFILE
    cfg = Config(read_timeout=120)
    if BEDROCK_AWS_PROFILE:
        session = boto3.Session(profile_name=BEDROCK_AWS_PROFILE)
        return session.client("bedrock-runtime", region_name=AWS_REGION, config=cfg)
    return boto3.client("bedrock-runtime", region_name=AWS_REGION, config=cfg)


def classify_requirement_tier(text: str, client=None) -> Tier:
    from config import NOVA_MODEL_ID
    if client is None:
        client = get_bedrock_client()
    raw = _nova_invoke(client, NOVA_MODEL_ID, TIER_PROMPT + text[:4000], max_tokens=20)
    return _normalize_tier(raw) if raw else "component"


def assign_tiers_to_requirements(entries: list[Entry], client=None) -> list[tuple[Entry, Tier]]:
    result = []
    for e in entries:
        result.append((e, classify_requirement_tier(e.text, client=client)))
    return result


def assign_tiers_to_requirements_with_fallback(
    entries: list[Entry], default_tier: Tier = "system", client=None
) -> list[tuple[Entry, Tier]]:
    try:
        return assign_tiers_to_requirements(entries, client=client)
    except Exception as err:
        if "Credential" in type(err).__name__ or "credentials" in str(err).lower():
            print(f"[Nova skipped] No AWS credentials. Using tier '{default_tier}' for all {len(entries)}.")
        else:
            print(f"[Nova failed] {err}. Using tier '{default_tier}' for all.")
        return [(e, default_tier) for e in entries]
