import csv
import os
from typing import Any, Callable, Dict, List, Optional, TypeVar

from filelock import FileLock
import json
import shlex
import subprocess


def ensure_directory_exists(path: str):
    """Create `path` if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def shell(args: List[str]):
    """Executes the shell command in `args`."""
    cmd = shlex.join(args)
    # hlog(f"Executing: {cmd}")
    exit_code = subprocess.call(args)
    # if exit_code != 0:
    # hlog(f"Failed with exit code {exit_code}: {cmd}")

# @htrack(None) 본격적으로 수정


def ensure_file_downloaded(
    source_url: str,
    target_path: str,
    unpack: bool = False,
    downloader_executable: str = "wget",
    unpack_type: Optional[str] = None,
):
    """Download `source_url` to `target_path` if it doesn't exist."""
    with FileLock(f"{target_path}.lock"):
        if os.path.exists(target_path):
            # Assume it's all good
            # hlog(f"Not downloading {source_url} because {target_path} already exists")
            return

        # Download
        # gdown is used to download large files/zip folders from Google Drive.
        # It bypasses security warnings which wget cannot handle.
        if source_url.startswith("https://drive.google.com"):
            import gdown
            downloader_executable = "gdown"
        tmp_path: str = f"{target_path}.tmp"
        shell([downloader_executable, source_url, "-O", tmp_path])

        # Unpack (if needed) and put it in the right location
        if unpack:
            if unpack_type is None:
                if source_url.endswith(".tar") or source_url.endswith(".tar.gz"):
                    unpack_type = "untar"
                elif source_url.endswith(".zip"):
                    unpack_type = "unzip"
                elif source_url.endswith(".zst"):
                    unpack_type = "unzstd"
                else:
                    raise Exception(
                        "Failed to infer the file format from source_url. Please specify unpack_type.")

            tmp2_path = target_path + ".tmp2"
            ensure_directory_exists(tmp2_path)
            if unpack_type == "untar":
                shell(["tar", "xf", tmp_path, "-C", tmp2_path])
            elif unpack_type == "unzip":
                shell(["unzip", tmp_path, "-d", tmp2_path])
            elif unpack_type == "unzstd":
                dctx = zstandard.ZstdDecompressor()
                with open(tmp_path, "rb") as ifh, open(os.path.join(tmp2_path, "data"), "wb") as ofh:
                    dctx.copy_stream(ifh, ofh)
            else:
                raise Exception("Invalid unpack_type")
            files = os.listdir(tmp2_path)
            if len(files) == 1:
                # If contains one file, just get that one file
                shell(["mv", os.path.join(tmp2_path, files[0]), target_path])
                os.rmdir(tmp2_path)
            else:
                shell(["mv", tmp2_path, target_path])
            os.unlink(tmp_path)
        else:
            # Don't decompress if desired `target_path` ends with `.gz`.
            if source_url.endswith(".gz") and not target_path.endswith(".gz"):
                gzip_path = f"{target_path}.gz"
                shell(["mv", tmp_path, gzip_path])
                # gzip writes its output to a file named the same as the input file, omitting the .gz extension
                shell(["gzip", "-d", gzip_path])
            else:
                shell(["mv", tmp_path, target_path])
        # hlog(f"Finished downloading {source_url} to {target_path}")
