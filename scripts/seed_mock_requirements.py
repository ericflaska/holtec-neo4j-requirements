import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neo4j import GraphDatabase
from config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


MOCK_PLANT = [
    ("P1", "The plant shall maintain overall safety compliance."),
    ("P2", "The plant shall provide power to all systems within capacity limits."),
    ("P3", "The plant shall support remote monitoring and control."),
]
MOCK_SYSTEM = [
    ("S1", "The electrical system shall distribute power to all subsystems."),
    ("S2", "The control system shall implement plant-level safety interlocks."),
    ("S3", "The cooling system shall maintain temperatures within spec."),
    ("S4", "The monitoring system shall collect data from all components."),
    ("S5", "The safety system shall enforce emergency shutdown when required."),
]
MOCK_COMPONENT = [
    ("C1", "Pump P-101 shall maintain flow rate between 10–50 GPM."),
    ("C2", "Sensor T-201 shall report temperature every 5 seconds."),
    ("C3", "Valve V-301 shall open on ESD signal."),
    ("C4", "Controller PLC-1 shall run the main control loop at 100 ms."),
    ("C5", "RTU-1 shall forward alarms to the monitoring system."),
    ("C6", "Breaker B-1 shall trip on overcurrent above 120%."),
]
TRACES = [
    ("P1", "S2"), ("P1", "S5"), ("P2", "S1"), ("P2", "S3"), ("P3", "S4"), ("P3", "S1"),
    ("S1", "C6"), ("S2", "C3"), ("S3", "C1"), ("S4", "C2"), ("S4", "C5"), ("S5", "C3"),
]


def to_full_id(suffix: str) -> str:
    if suffix.startswith("P"): return "plant_" + suffix
    if suffix.startswith("S"): return "system_" + suffix
    if suffix.startswith("C"): return "component_" + suffix
    return suffix


def seed(tx):
    tx.run("MATCH (n) WHERE n:Plant OR n:System OR n:Component OR n:Requirement DETACH DELETE n")
    tx.run("CREATE (p:Plant {id: 'plant', name: 'Plant'})")
    tx.run("CREATE (s:System {id: 'system', name: 'System'})")
    tx.run("CREATE (c:Component {id: 'component', name: 'Component'})")

    def add_reqs(tier: str, label: str, items: list):
        for sid, text in items:
            req_id = f"{tier}_{sid}"
            tx.run(
                f"CREATE (r:Requirement {{id: $id, tier: $tier, text: $text}}) WITH r MATCH (level:{label} {{id: $tier}}) CREATE (level)-[:HAS_REQUIREMENT]->(r)",
                id=req_id, tier=tier, text=text,
            )
    add_reqs("plant", "Plant", MOCK_PLANT)
    add_reqs("system", "System", MOCK_SYSTEM)
    add_reqs("component", "Component", MOCK_COMPONENT)

    for from_suffix, to_suffix in TRACES:
        tx.run(
            "MATCH (a:Requirement {id: $from_id}), (b:Requirement {id: $to_id}) CREATE (a)-[:TRACES_TO]->(b), (b)-[:TRACES_TO]->(a)",
            from_id=to_full_id(from_suffix), to_id=to_full_id(to_suffix),
        )


def main():
    driver = get_driver()
    try:
        with driver.session() as session:
            session.execute_write(seed)
        print("Mock graph seeded.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
