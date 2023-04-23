from pathlib import Path
from typing import Union


def verify_dir(dir: Union[str, Path]) -> None:
    """Create dir if it does not exist"""
    Path(dir).mkdir(parents=True, exist_ok=True)
