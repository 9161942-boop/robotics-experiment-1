"""Complete the phone screening folder after manual image selection."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dataset" / "varify" / "phone"
API = "https://commons.wikimedia.org/w/api.php"

# These entries reflect the user's retained images plus three clear, recent
# single-phone product photos used to bring the folder back to 20.
FINAL_FILES = [
    ("phone_02.jpg", "File:IPhone 13 Pro.jpg"),
    ("phone_03.jpg", "File:IPhone 14 Pro Deep Purple.jpg"),
    ("phone_04.png", "File:IPhone 15 pro.png"),
    ("phone_05.jpg", "File:IPhone 16 Pro Max White 256g.jpg"),
    ("phone_07.png", "File:GalaxyS21.png"),
    ("phone_09.png", "File:Galaxy S23.png"),
    ("phone_10.jpg", "File:Samsung-Galaxy-S24-Ultra-Front.jpg"),
    ("phone_14.jpg", "File:OnePlus 8T.jpg"),
    ("phone_15.jpg", "File:Nothing Phone.jpg"),
    ("phone_16.jpg", "File:Xiaomi 13.jpg"),
    ("phone_19.jpg", "File:Samsung Galaxy Z Flip 6.jpg"),
    ("phone_20.jpg", "File:Motorola Edge 60 Pro.jpg"),
    ("phone_23.jpg", "File:Xiaomi 14 (July 10, 2026).jpg"),
    ("phone_25.jpg", "File:Vivo X100.jpg"),
    ("phone_26.jpg", "File:Xiaomi Civi 4 Pro (1).jpg"),
    ("phone_27.jpg", "File:Samsung Galaxy Z Flip 7 Unfolded.jpg"),
    ("phone_28.jpg", "File:Xiaomi 17 Alpine Pink.jpg"),
    ("phone_29.jpg", "File:Samsung Galaxy S25+.jpg"),
    ("phone_30.jpg", "File:Oppo Find X9 Titanium Grey.jpg"),
    ("phone_31.jpg", "File:Xiaomi 15 (1).jpg"),
]


def metadata(info: dict, key: str) -> str:
    value = info.get("extmetadata", {}).get(key, {})
    return str(value.get("value", "")) if isinstance(value, dict) else str(value)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    titles = [title for _filename, title in FINAL_FILES]
    query = urlencode(
        {
            "action": "query",
            "titles": "|".join(titles),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size",
            "iiurlwidth": "1280",
            "format": "json",
        }
    )
    request = Request(f"{API}?{query}", headers={"User-Agent": "robotics-experiment-1/1.0"})
    pages = json.loads(urlopen(request, timeout=60).read().decode("utf-8")).get("query", {}).get("pages", {})
    by_title = {page.get("title"): page for page in pages.values()}
    manifest: list[dict[str, object]] = []
    for filename, title in FINAL_FILES:
        page = by_title.get(title)
        if not page or "imageinfo" not in page:
            raise SystemExit(f"Could not resolve {title}")
        info = page["imageinfo"][0]
        download_url = info.get("thumburl") or info.get("url")
        source_url = info.get("url") or download_url
        if not download_url or not source_url:
            raise SystemExit(f"No image URL for {title}")
        destination = OUTPUT / filename
        if not destination.exists():
            image_request = Request(str(download_url), headers={"User-Agent": "robotics-experiment-1/1.0"})
            with urlopen(image_request, timeout=90) as response:
                destination.write_bytes(response.read())
        manifest.append(
            {
                "sample_id": destination.stem,
                "file": filename,
                "expected_class": "phone",
                "title": title,
                "page_url": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                "source_url": source_url,
                "download_url": download_url,
                "author": metadata(info, "Artist"),
                "license": metadata(info, "LicenseShortName"),
                "license_url": metadata(info, "LicenseUrl"),
            }
        )
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Phone screening set ready: {len(manifest)} images")


if __name__ == "__main__":
    main()
