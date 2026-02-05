"""Configuration management for pose benchmarking."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    images_root: str
    annotations_json: str


@dataclass
class OutputConfig:
    """Output configuration."""
    dir: str


@dataclass
class BenchmarkConfig:
    """Benchmark run configuration."""
    max_images: Optional[int] = None
    min_conf: float = 0.2
    models: list[str] = field(default_factory=lambda: ["mediapipe"])


@dataclass
class Config:
    """Top-level configuration."""
    dataset: DatasetConfig
    output: OutputConfig
    benchmark: BenchmarkConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(
            dataset=DatasetConfig(**data["dataset"]),
            output=OutputConfig(**data["output"]),
            benchmark=BenchmarkConfig(**data["benchmark"]),
        )
    
    def get_output_dir(self, subdir: str = "") -> Path:
        """Get output directory path, optionally with subdirectory."""
        base = Path(self.output.dir)
        if subdir:
            return base / subdir
        return base
