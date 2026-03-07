import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME_OR_PATH = os.getenv("QWEN_MODEL_PATH", "path/to/your/qwen-4b-vl3")
DEVICE = os.getenv("DEVICE", "cuda")
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "bfloat16")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "5000000"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_AWS_PROFILE = os.getenv("BEDROCK_AWS_PROFILE") or os.getenv("AWS_PROFILE") or None
NOVA_MODEL_ID = os.getenv("NOVA_MODEL_ID", "amazon.nova-pro-v1:0")
TITAN_EMBED_MODEL_ID = os.getenv("TITAN_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
