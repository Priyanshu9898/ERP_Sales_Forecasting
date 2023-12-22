from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path


@dataclass
class DataProcessingConfig:
    root_dir: Path
    data_file: Path
    preprocessed_file: Path
    isValid: bool


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    data_file: Path
    result_file: Path
