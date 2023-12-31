from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    account_name : str
    account_key : str
    container_name : str
    download_file_path : Path
