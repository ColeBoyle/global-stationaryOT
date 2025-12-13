import os
import pathlib
import hashlib
import requests
import yaml

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent  # adjust if needed

DEFAULT_DATA_DIR = ROOT / "data"


def _load_config():
    cfg_path = ROOT / "data" / "osf_resources.yml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _download_file(url, dest_path, chunk_size=2**20):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    tmp = dest_path.with_suffix(dest_path.suffix + ".part")

    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

    tmp.rename(dest_path)


def get_dataset(dataset_name: str, data_dir: str | os.PathLike | None = None):
    """
    Ensure the requested dataset is present locally.
    Downloads from OSF if missing.

    Parameters
    ----------
    dataset_name : str
        Name as defined in osf_resources.yml
    data_dir : path-like, optional
        Base directory where data will live.
        Defaults to repo_root/data

    Returns
    -------
    pathlib.Path
        Path to the *dataset directory*.
    """
    cfg = _load_config()
    if dataset_name not in cfg["datasets"]:
        raise KeyError(f"Unknown dataset: {dataset_name}")

    ds_cfg = cfg["datasets"][dataset_name]

    base_dir = pathlib.Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    target_dir = base_dir / ds_cfg["target_dir"]

    for f in ds_cfg["files"]:
        local_path = target_dir / f["name"]
        if not local_path.exists():
            print(f"Downloading {f['name']} from OSF...")
            _download_file(f["osf_url"], local_path)
        else:
            print(f"Found existing file: {local_path}")

    return target_dir