    
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

import gdown
import torch
import tyro
from typing_extensions import Annotated
    
def download(save_dir: Path):
    """Download the blender dataset."""
    # TODO: give this code the same structure as download_nerfstudio

    # https://drive.google.com/uc?id=1aS-WyrmJbaDyuriPzNcMU870Cc9pdpDz
    
    blender_file_id = "1aS-WyrmJbaDyuriPzNcMU870Cc9pdpDz"

    final_path = save_dir / Path("blender")
    if os.path.exists(final_path):
        shutil.rmtree(str(final_path))
    url = f"https://drive.google.com/uc?id={blender_file_id}"
    download_path = save_dir / "blender_data.zip"
    gdown.download(url, output=str(download_path))
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(str(save_dir))
    unzip_path = save_dir / Path("nerf_synthetic")
    final_path = save_dir / Path("blender")
    unzip_path.rename(final_path)
    if download_path.exists():
        download_path.unlink()
        
if __name__ == "__main__":
    download(Path("."))
    print("Blender dataset downloaded and extracted successfully.")