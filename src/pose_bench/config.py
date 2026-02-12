"""Configuration management for pose benchmarking."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    images_root: str
    annotations_json: Optional[str] = None


@dataclass
class OutputConfig:
    """Output configuration."""
    dir: str


@dataclass
class BenchmarkConfig:
    """Benchmark run configuration."""
    max_images: Optional[int] = None
    min_conf: float = 0.3
    models: list[str] = field(default_factory=lambda: ["mediapipe", "yolov8-pose", "movenet"])


@dataclass
class Config:
    """Top-level configuration."""
    dataset: DatasetConfig
    output: OutputConfig
    benchmark: BenchmarkConfig
    
    def get_output_dir(self, subdir: str = "") -> Path:
        """Get output directory path, optionally with subdirectory."""
        base = Path(self.output.dir)
        if subdir:
            return base / subdir
        return base
    
    def get_dataset_type(self) -> str:
        """Get dataset type from name."""
        name_lower = self.dataset.name.lower()
        if "coco" in name_lower:
            return "coco"
        elif "mpii" in name_lower:
            return "mpii"
        else:
            return "gym"
