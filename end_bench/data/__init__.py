"""Text-dataset contract for text-only tracks (no activation arrays)."""

from end_bench.data.text_dataset import (
    TEXT_DATASET_SCHEMA_VERSION,
    TextDatasetError,
    export_text_dataset,
    is_text_dataset,
    load_text_dataset,
    save_text_dataset,
)

__all__ = [
    "TEXT_DATASET_SCHEMA_VERSION",
    "TextDatasetError",
    "export_text_dataset",
    "is_text_dataset",
    "load_text_dataset",
    "save_text_dataset",
]
