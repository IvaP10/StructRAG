from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    LIST = "list"
    HEADER = "header"
    FOOTER = "footer"
    CODE = "code"
    EQUATION = "equation"


class FormatType(str, Enum):
    PLAIN = "plain"
    MARKDOWN = "markdown"
    HTML = "html"
    LATEX = "latex"


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

    def dict(self, *args, **kwargs):
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    parent_id: Optional[UUID] = None
    text: str
    chunk_type: ChunkType = ChunkType.TEXT
    format_type: FormatType = FormatType.PLAIN
    page_number: int
    bbox: Optional[BoundingBox] = None
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk text cannot be empty")
        return v

    @property
    def is_parent(self) -> bool:
        return self.parent_id is None

    @property
    def is_child(self) -> bool:
        return self.parent_id is not None


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    filename: str
    file_path: str
    status: ProcessingStatus = ProcessingStatus.UPLOADING
    total_pages: int = 0
    total_chunks: int = 0
    total_parent_chunks: int = 0
    total_child_chunks: int = 0
    upload_time: datetime = Field(default_factory=datetime.now)
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    file_size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
