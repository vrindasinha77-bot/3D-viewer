import sys
from pathlib import Path

import requests

STATIC_DIR = Path(__file__).resolve().parent / "static"

# file name -> (list of download addresses to try, text the real file contains, minimum size in bytes)
FILES = {
    "three.min.js": (
        [
            "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
            "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js",
            "https://unpkg.com/three@0.128.0/build/three.min.js",
        ],
        "REVISION",
        100_000,
    ),
    "OrbitControls.js": (
        [
            "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js",
            "https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js",
        ],
        "OrbitControls",
        10_000,
    ),
}


def download(name: str, urls: list, marker: str, min_size: int) -> bool:
    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as err:
            print(f"  could not use {url} ({type(err).__name__})")
            continue

        text = response.text
        if len(response.content) < min_size or marker not in text:
            print(f"  {url} did not look like the real file, trying the next address")
            continue

        (STATIC_DIR / name).write_text(text, encoding="utf-8")
        print(f"  saved {name} ({len(response.content) // 1024} KB)")
        return True
    return False


def main() -> int:
    STATIC_DIR.mkdir(exist_ok=True)
    failed = []
    for name, (urls, marker, min_size) in FILES.items():
        print(f"Downloading {name} ...")
        if not download(name, urls, marker, min_size):
            failed.append((name, urls[0]))

    if failed:
        print("\nSome files could not be downloaded.")
        print("Open each address below in Chrome, press Ctrl+S, and save it into the static folder:")
        for name, url in failed:
            print(f"  {name}: {url}")
        return 1

    print("\nDone. Now start the server with:  python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
