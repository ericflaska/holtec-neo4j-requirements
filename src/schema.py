from pydantic import BaseModel, Field


class Entry(BaseModel):
    doc_title: str = Field(...)
    page_number: int = Field(...)
    section_title: str = Field(...)
    text: str = Field(...)
    is_requirement: str = Field(...)


class ExtractionResult(BaseModel):
    title: str = Field(...)
    entries: list[Entry] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)
    model_config = {"extra": "allow"}

    def requirements_only(self) -> list[Entry]:
        yes_values = {"yes", "true", "1", "requirement"}
        return [e for e in self.entries if str(e.is_requirement).strip().lower() in yes_values]
