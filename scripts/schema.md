# 3-tier schema

**Tiers:** Plant, System, Component (one organizer node each).

**Nodes:** Plant, System, Component (entry points); Requirement (tier, text, …).

**Edges:** (Level)-[:HAS_REQUIREMENT]->(Requirement). (Requirement)-[:TRACES_TO]-(Requirement) in both directions.

**Cypher:** `MATCH (p:Plant)-[:HAS_REQUIREMENT]->(r) RETURN r` | `MATCH (a:Requirement)-[:TRACES_TO]-(b) WHERE a.id = $id RETURN b`
