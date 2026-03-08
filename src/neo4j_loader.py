from typing import Any

from neo4j import GraphDatabase

from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from src.schema import Entry, ExtractionResult

Tier = str


def get_driver(uri=None, user=None, password=None):
    return GraphDatabase.driver(uri or NEO4J_URI, auth=(user or NEO4J_USER, password or NEO4J_PASSWORD))


def _entry_id(doc_title: str, page: int, section: str, text_preview: str) -> str:
    import hashlib
    return hashlib.sha256(f"{doc_title}|{page}|{section}|{text_preview[:200]}".encode()).hexdigest()[:16]


def _entry_id_with_index(doc_title: str, page: int, section: str, text_preview: str, index: int) -> str:
    base = _entry_id(doc_title, page, section, text_preview)
    return f"{base}_{index}"


def load_requirements_into_neo4j(result: ExtractionResult, driver=None, document_id=None) -> int:
    requirements = result.requirements_only()
    if not requirements:
        return 0
    doc_id = document_id or result.title.replace(" ", "_")[:64]
    pages = result.meta.get("pages", [])

    def _run(tx):
        tx.run("MERGE (d:Document {id: $id}) SET d.title = $title, d.pages = $pages", id=doc_id, title=result.title, pages=pages)
        count = 0
        for e in requirements:
            req_id = _entry_id(e.doc_title, e.page_number, e.section_title, e.text)
            tx.run(
                """
                MERGE (s:Section {doc_id: $doc_id, title: $section_title}) ON CREATE SET s.doc_title = $doc_title
                WITH s MATCH (d:Document {id: $doc_id}) MERGE (d)-[:HAS_SECTION]->(s)
                MERGE (r:Requirement {id: $req_id}) SET r.page_number = $page_number, r.section_title = $section_title, r.text = $text, r.doc_title = $doc_title
                MERGE (s)-[:CONTAINS]->(r)
                """,
                doc_id=doc_id, doc_title=e.doc_title, section_title=e.section_title, page_number=e.page_number, text=e.text, req_id=req_id,
            )
            count += 1
        return count

    driver = driver or get_driver()
    with driver.session() as session:
        return session.execute_write(_run)


def load_tiered_requirements_into_neo4j(
    requirements_with_tiers: list[tuple[Entry, Tier]],
    driver=None,
    document_id=None,
    embeddings: list[list[float]] | None = None,
) -> int:
    if not requirements_with_tiers:
        return 0
    doc_id = (document_id or "extraction").replace(" ", "_")[:64]
    n = len(requirements_with_tiers)
    use_embeddings = embeddings is not None and len(embeddings) == n

    def _run(tx):
        tx.run("MERGE (p:Plant {id: 'plant', name: 'Plant'})")
        tx.run("MERGE (s:System {id: 'system', name: 'System'})")
        tx.run("MERGE (c:Component {id: 'component', name: 'Component'})")
        count = 0
        for idx, (e, tier) in enumerate(requirements_with_tiers):
            full_id = f"{doc_id}_{_entry_id_with_index(e.doc_title, e.page_number, e.section_title, e.text, idx)}"
            if use_embeddings and idx < len(embeddings):
                tx.run(
                    """
                    MERGE (r:Requirement {id: $req_id})
                    SET r.tier = $tier, r.text = $text, r.doc_title = $doc_title, r.section_title = $section_title, r.page_number = $page_number, r.embedding = $embedding
                    WITH r MATCH (level) WHERE level.id = $tier AND (level:Plant OR level:System OR level:Component) MERGE (level)-[:HAS_REQUIREMENT]->(r)
                    """,
                    req_id=full_id, tier=tier, text=e.text, doc_title=e.doc_title, section_title=e.section_title, page_number=e.page_number,
                    embedding=embeddings[idx],
                )
            else:
                tx.run(
                    """
                    MERGE (r:Requirement {id: $req_id}) SET r.tier = $tier, r.text = $text, r.doc_title = $doc_title, r.section_title = $section_title, r.page_number = $page_number
                    WITH r MATCH (level) WHERE level.id = $tier AND (level:Plant OR level:System OR level:Component) MERGE (level)-[:HAS_REQUIREMENT]->(r)
                    """,
                    req_id=full_id, tier=tier, text=e.text, doc_title=e.doc_title, section_title=e.section_title, page_number=e.page_number,
                )
            count += 1
        return count

    driver = driver or get_driver()
    with driver.session() as session:
        return session.execute_write(_run)


def requirement_full_id(doc_id: str, e: Entry, index: int | None = None) -> str:
    safe_doc = (doc_id or "extraction").replace(" ", "_")[:64]
    if index is not None:
        base = _entry_id_with_index(e.doc_title, e.page_number, e.section_title, e.text, index)
        return f"{safe_doc}_{base}"
    return f"{safe_doc}_{_entry_id(e.doc_title, e.page_number, e.section_title, e.text)}"


BATCH_SIZE_TRACE_EDGES = 2000


def delete_trace_edges(driver=None) -> int:
    driver = driver or get_driver()
    with driver.session() as session:
        result = session.run("MATCH ()-[r:TRACES_TO]->() DELETE r")
        summary = result.consume()
        return getattr(summary.counters, "relationships_deleted", 0) or 0


def create_trace_edges(
    requirement_full_ids: list[str],
    pairs: list[tuple[int, int]],
    driver=None,
    batch_size: int = BATCH_SIZE_TRACE_EDGES,
    progress_callback=None,
) -> int:
    if not pairs or not requirement_full_ids:
        return 0
    valid = []
    for i, j in pairs:
        if 0 <= i < len(requirement_full_ids) and 0 <= j < len(requirement_full_ids):
            valid.append((requirement_full_ids[i], requirement_full_ids[j]))
    if not valid:
        return 0

    driver = driver or get_driver()
    total_batches = (len(valid) + batch_size - 1) // batch_size
    total = 0
    with driver.session() as session:
        for batch_idx, start in enumerate(range(0, len(valid), batch_size)):
            batch = valid[start : start + batch_size]
            batch_param = [{"a_id": a_id, "b_id": b_id} for a_id, b_id in batch]

            def _run(tx, param=batch_param):
                result = tx.run(
                    "UNWIND $pairs AS p MATCH (a:Requirement {id: p.a_id}), (b:Requirement {id: p.b_id}) "
                    "MERGE (a)-[:TRACES_TO]->(b) MERGE (b)-[:TRACES_TO]->(a)",
                    pairs=param,
                )
                result.consume()
                return len(param) * 2

            total += session.execute_write(_run)
            if progress_callback:
                progress_callback(batch_idx + 1, total_batches, len(batch) * 2)
    return total


ADJACENT_TIERS = {("plant", "system"), ("system", "plant"), ("system", "component"), ("component", "system")}


def filter_trace_pairs_to_adjacent_tiers(
    requirements_with_tiers: list[tuple[Entry, Tier]], pairs: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    if not requirements_with_tiers or not pairs:
        return []
    tiers = [(t or "").strip().lower() for _, t in requirements_with_tiers]
    n = len(tiers)
    out = []
    for i, j in pairs:
        if i < 0 or i >= n or j < 0 or j >= n:
            continue
        ti, tj = tiers[i], tiers[j]
        if ti == tj:
            continue
        if (ti, tj) in ADJACENT_TIERS:
            out.append((i, j))
    return out


def tier_based_trace_pairs(requirements_with_tiers: list[tuple[Entry, Tier]]) -> list[tuple[int, int]]:
    plant_idx = [i for i, (_, t) in enumerate(requirements_with_tiers) if t == "plant"]
    system_idx = [i for i, (_, t) in enumerate(requirements_with_tiers) if t == "system"]
    component_idx = [i for i, (_, t) in enumerate(requirements_with_tiers) if t == "component"]
    pairs = []
    for i in plant_idx:
        for j in system_idx:
            pairs.append((i, j))
    for i in system_idx:
        for j in component_idx:
            pairs.append((i, j))
    return pairs
