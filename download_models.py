import os
import urllib.request

import config


def _download(url, dest_path, label):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        print(f"[download] {label} already present at {dest_path}, skipping.")
        return
    print(f"[download] Fetching {label} from {url} ...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"[download] Saved {label} ({size_kb:.0f} KB) -> {dest_path}")
    except Exception as e:
        print(f"[download] FAILED to fetch {label}: {e}")
        print(f"[download] You can also download it manually from:\n  {url}\n  and save it as: {dest_path}")


if __name__ == "__main__":
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    _download(config.YUNET_URL, config.YUNET_MODEL_PATH, "YuNet face detector")
    _download(config.SFACE_URL, config.SFACE_MODEL_PATH, "SFace face recognizer")