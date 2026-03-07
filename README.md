# PDF → Qwen VL → Neo4j

Stitch two PDF pages → run Qwen VL → parse JSON → load requirements into Neo4j (Plant/System/Component + TRACES_TO).

**Setup:** `pip install -r requirements.txt`. Set `.env` (see env.example). Neo4j running. For GovCloud + SSO: set `BEDROCK_AWS_PROFILE=govcloud` and `AWS_REGION=us-gov-west-1` (or your GovCloud region), then run `aws sso login --profile govcloud` before Nova/Bedrock calls.

**Run:** `python run_pipeline.py path/to/doc.pdf` | `--page 0` | `--no-neo4j` | `--no-nova` | `--out-json out.json`

**Temp notebook:** `temp_train_jsonl_to_neo4j.ipynb` — reads train.jsonl, filters requirements, Nova tier + traces, loads to Neo4j.

**Mock graph:** `python scripts/seed_mock_requirements.py`
