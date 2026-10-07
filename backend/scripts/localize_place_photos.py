"""Download seeded Pexels place photos and rewrite their database URLs locally."""

from __future__ import annotations

import argparse
import json
import sqlite3
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

import httpx

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = BACKEND_ROOT / "travel_agent.db"
DEFAULT_OUTPUT = BACKEND_ROOT / "app" / "static" / "place_photos"
STATIC_PREFIX = "/static/place_photos"
IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MAX_BYTES = 5_000_000


def _download(client: httpx.Client, url: str, output: Path) -> tuple[Path, str]:
    digest = sha256(url.encode("utf-8")).hexdigest()
    for extension in IMAGE_TYPES.values():
        existing = output / f"{digest}{extension}"
        if existing.is_file() and existing.stat().st_size > 0:
            return existing, extension
    with client.stream("GET", url) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        extension = IMAGE_TYPES.get(content_type)
        if extension is None:
            raise ValueError(f"unsupported content type {content_type!r}")
        destination = output / f"{digest}{extension}"
        temporary = output / f".{digest}.tmp"
        size = 0
        try:
            with temporary.open("wb") as target:
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError(f"image exceeds {MAX_BYTES} bytes")
                    target.write(chunk)
            if size == 0:
                raise ValueError("empty image")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination, extension


def localize(db_path: Path, output: Path) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    rows = connection.execute(
        "SELECT id, type, image_url FROM places WHERE image_url IS NOT NULL ORDER BY id"
    ).fetchall()
    updates: list[tuple[str, int]] = []
    manifest: dict[str, dict[str, str | int]] = {}
    downloaded = 0
    reused = 0
    invalid = 0
    with httpx.Client(timeout=httpx.Timeout(20.0), follow_redirects=True) as client:
        for place_id, category, url in rows:
            parsed = urlparse(str(url))
            if parsed.hostname == "images.pexels.com" and parsed.scheme == "https":
                before = {path.name for path in output.glob(f"{sha256(url.encode('utf-8')).hexdigest()}.*")}
                path, _ = _download(client, url, output)
                downloaded += int(path.name not in before)
                reused += int(path.name in before)
                local_url = f"{STATIC_PREFIX}/{path.name}"
                updates.append((local_url, place_id))
                manifest[str(place_id)] = {"source_url": url, "local_url": local_url}
            elif parsed.hostname == "example.com":
                safe_category = category if category in {"attraction", "hotel", "restaurant"} else "generic"
                updates.append((f"/static/place_images/{safe_category}.svg", place_id))
                invalid += 1
    try:
        connection.executemany("UPDATE places SET image_url = ? WHERE id = ?", updates)
        connection.commit()
    finally:
        connection.close()
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"rewritten": len(updates), "downloaded": downloaded, "reused": reused, "invalid_replaced": invalid}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(localize(args.db.resolve(), args.output.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
