import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEVICE
from src.inference import load_model_and_processor, run_inference
from src.json_utils import parse_extraction
from src.neo4j_loader import load_requirements_into_neo4j, load_tiered_requirements_into_neo4j
from src.nova_tier import assign_tiers_to_requirements
from src.pdf_utils import stitch_pdf_pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--no-neo4j", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--no-nova", action="store_true")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: PDF not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    image, page_numbers = stitch_pdf_pages(args.pdf_path, first_page=args.page)
    model, processor, torch_dtype = load_model_and_processor()
    raw_output = run_inference(image, model, processor, DEVICE, torch_dtype)

    try:
        result = parse_extraction(raw_output)
    except Exception as e:
        print(f"Parse error: {e}", file=sys.stderr)
        if args.out_json:
            args.out_json.write_text(raw_output, encoding="utf-8")
        sys.exit(1)

    if "pages" not in result.meta:
        result.meta["pages"] = page_numbers

    requirements = result.requirements_only()
    requirements_with_tiers = None
    if requirements and not args.no_nova:
        try:
            requirements_with_tiers = assign_tiers_to_requirements(requirements)
        except Exception as e:
            print(f"Nova failed: {e}", file=sys.stderr)

    if args.out_json:
        out = result.model_dump()
        if requirements_with_tiers:
            out["requirements_with_tiers"] = [{"text": e.text[:200], "tier": t} for e, t in requirements_with_tiers]
        args.out_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    if not args.no_neo4j and requirements:
        if requirements_with_tiers:
            n = load_tiered_requirements_into_neo4j(requirements_with_tiers)
        else:
            n = load_requirements_into_neo4j(result)
        print(f"Loaded {n} requirement nodes.")
    elif args.no_neo4j:
        print("Skipping Neo4j.")


if __name__ == "__main__":
    main()
