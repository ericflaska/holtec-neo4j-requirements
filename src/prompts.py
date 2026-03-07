EXTRACTION_INSTRUCTION = """You are analyzing a document image that shows two PDF pages stitched together (top page first, then bottom page).

Your task: extract all document entries (sections, paragraphs, or bullet points) and classify whether each is a requirement or not.

Output a single valid JSON object with this exact structure. No other text before or after the JSON.

{
    "title": "<short title for this document or these two pages>",
    "entries": [
        {
            "doc_title": "<document or section title>",
            "page_number": <1 or 2, the page this entry appears on within this image>,
            "section_title": "<section or heading name>",
            "text": "<the full text of this entry>",
            "is_requirement": "yes"
        },
        {
            "doc_title": "...",
            "page_number": 1,
            "section_title": "...",
            "text": "...",
            "is_requirement": "no"
        }
    ],
    "meta": {
        "pages": [1, 2]
    }
}

Rules:
- page_number must be 1 for content on the first (top) page and 2 for the second (bottom) page.
- For each distinct block of content (paragraph, list item, requirement line), add one entry.
- Set is_requirement to "yes" only for statements that are actual requirements (e.g. "The system shall...", "Must support..."). Use "no" for explanatory text, titles, or non-requirement content.
- meta.pages should be [1, 2] for this two-page image.
- Output only the JSON object, no markdown code fences."""


def build_extraction_prompt(instruction: str = EXTRACTION_INSTRUCTION) -> str:
    return instruction
