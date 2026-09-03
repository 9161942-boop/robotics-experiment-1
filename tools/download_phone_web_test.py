"""Download a traceable set of public phone images for an external test."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dataset" / "varify" / "phone_web"
API = "https://commons.wikimedia.org/w/api.php"

# One representative product photo per model; avoid multi-phone collages and screenshots.
TITLES = [
    "File:ACER Liquid Z5 smartphone (15021094340).jpg",
    "File:BlackBerry Curve 9300.webp",
    "File:Blu View 3.jpg",
    "File:Gionee Elife S5.1, world's slimmest smartphone (16948492858).jpg",
    "File:Google Nexus S smartphone.jpg",
    "File:Google Pixel XL smartphone (30155264605).jpg",
    "File:HONOR V40 Lite Luxury Edition Front.jpg",
    "File:Huawei P10 front.jpg",
    "File:Huawei P8 smartphone (16540595414).jpg",
    "File:IPhone 16e black front view.jpg",
    "File:LYF WATER 2 Smartphone.JPG",
    "File:Motorola Edge 50 Neo Front View Nautical Blue.png",
    "File:Nexus 5 Front View.png",
    "File:Nokia 4-2 front.jpg",
    "File:OPPO A57 LineageOS.jpg",
    "File:OPPO Find N5 Foldable Phone Front View.png",
    "File:OnePlus 8T Front View.png",
    "File:Redmi Note8 Pro.png",
    "File:Samsung Galaxy Y (GT-S5360).png",
    "File:Samsung S3 Neo Phone Black (Front).jpg",
]


def api_query() -> dict:
    query = urlencode(
        {
            "action": "query",
            "titles": "|".join(TITLES),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size",
            "iiurlwidth": "1280",
            "format": "json",
        }
    )
    request = Request(f"{API}?{query}", headers={"User-Agent": "robotics-experiment-1/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def text_metadata(info: dict, key: str) -> str:
    value = info.get("extmetadata", {}).get(key, {})
    return str(value.get("value", "")) if isinstance(value, dict) else str(value)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = api_query()
    pages = data.get("query", {}).get("pages", {})
    manifest: list[dict[str, object]] = []
    missing: list[str] = []
    for index, title in enumerate(TITLES, 1):
        page = next((item for item in pages.values() if item.get("title") == title), None)
        if not page or "imageinfo" not in page:
            missing.append(title)
            continue
        info = page["imageinfo"][0]
        url = info.get("thumburl") or info.get("url")
        if not url:
            missing.append(title)
            continue
        suffix = Path(str(info.get("url", url)).split("?", 1)[0]).suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        filename = f"phone_web_{index:02d}{suffix}"
        destination = OUTPUT / filename
        if not destination.exists():
            request = Request(str(url), headers={"User-Agent": "robotics-experiment-1/1.0"})
            with urlopen(request, timeout=90) as response:
                destination.write_bytes(response.read())
        manifest.append(
            {
                "sample_id": f"phone_web_{index:02d}",
                "file": filename,
                "expected_class": "phone",
                "title": title,
                "page_url": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                "source_url": info.get("url"),
                "download_url": url,
                "author": text_metadata(info, "Artist"),
                "license": text_metadata(info, "LicenseShortName"),
                "license_url": text_metadata(info, "LicenseUrl"),
            }
        )
    if missing:
        raise SystemExit(f"Could not resolve {len(missing)} titles: {missing}")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {len(manifest)} phone images to {OUTPUT}")


if __name__ == "__main__":
    main()
