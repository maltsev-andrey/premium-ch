#!/usr/bin/env python3
"""
s00_download.py — premium-ch data layer, stage 0.

Downloads the BAG health-insurance premium open data (priminfo) via the
opendata.swiss CKAN API:

    dataset: health-insurance-premiums
    - current premium year:  Prämien_CH.csv, Versichertenbestand_CH.csv,
      Einzugsgebiete.xlsx, Erläuterungen, Tarife
    - archives:              Archiv_Praemien_2011.zip .. Archiv_Praemien_2024.zip

Resource URLs are resolved at runtime from CKAN metadata (package_show),
so nothing breaks when BAG rotates the bagnet.ch links. Archives are
unpacked into data/raw/archive/<year>/. Every downloaded file is recorded
in data/raw/manifest.json with sha256, size and source URL.

Idempotent: existing files with a matching manifest sha256 are skipped.
Run with --force to re-download everything.

Usage (srv-anaconda-01, conda env premium-ch):
    python pipeline/s00_download.py
    python pipeline/s00_download.py --force
"""

import argparse
import base64
import hashlib
import json
import re
import sys
import time
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("/nfs/shared/premium-ch")
RAW_DIR = BASE_DIR / "data" / "raw"
ARCHIVE_DIR = RAW_DIR / "archive"
CURRENT_DIR = RAW_DIR / "current"
MANIFEST_PATH = RAW_DIR / "manifest.json"

CKAN_API = "https://ckan.opendata.swiss/api/3/action/package_show"
DATASET_ID = "health-insurance-premiums"

# PDFs (cover letters) are not data; everything else we take.
SKIP_EXTENSIONS = {".pdf"}

TIMEOUT = 60
RETRIES = 3
RETRY_WAIT = 10  # seconds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url: str, stream: bool = False) -> requests.Response:
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT, stream=stream)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_err = e
            log(f"  attempt {attempt}/{RETRIES} failed: {e}")
            if attempt < RETRIES:
                time.sleep(RETRY_WAIT)
    raise RuntimeError(f"GET failed after {RETRIES} attempts: {url}") from last_err


def sanitize_filename(name: str) -> str:
    """ASCII-fold umlauts and strip characters unsafe for NFS paths."""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^A-Za-z0-9._\- ]+", "_", name).strip()
    return name or "unnamed"


def resolve_filename(resource: dict) -> str:
    """
    Best-effort real filename for a CKAN resource.

    bagnet.ch URLs look like:
        https://opendata.bagnet.ch/?r=/download&path=<base64 of /Praemien/Prämien_CH.csv>
    so the base64 'path' param carries the true filename. Fall back to the
    URL basename, then the resource title.
    """
    url = resource.get("download_url") or resource.get("url") or ""
    qs = parse_qs(urlparse(url).query)
    if "path" in qs:
        try:
            decoded = base64.b64decode(unquote(qs["path"][0])).decode("utf-8")
            candidate = Path(decoded).name
            if candidate:
                return sanitize_filename(candidate)
        except Exception:
            pass
    basename = Path(urlparse(url).path).name
    if basename and "." in basename:
        return sanitize_filename(basename)
    title = resource.get("title") or resource.get("name") or ""
    if isinstance(title, dict):  # opendata.swiss multilingual titles
        title = title.get("de") or title.get("en") or next(iter(title.values()), "")
    fmt = (resource.get("format") or "bin").lower()
    return sanitize_filename(f"{title}.{fmt}")


def fetch_resources() -> list[dict]:
    log(f"querying CKAN metadata for dataset '{DATASET_ID}'")
    r = http_get(f"{CKAN_API}?id={DATASET_ID}")
    payload = r.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN package_show returned success=false: {payload}")
    resources = payload["result"].get("resources", [])
    log(f"CKAN lists {len(resources)} resources")
    return resources


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    r = http_get(url, stream=True)
    size = 0
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            size += len(chunk)
    tmp.rename(dest)
    log(f"  saved {dest.name} ({size / 1e6:.1f} MB)")


def unpack_archive(zip_path: Path, year: str) -> list[str]:
    """Unpack one Archiv_Praemien_<year>.zip into archive/<year>/, flat."""
    target = ARCHIVE_DIR / year
    target.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            # flatten: some years nest files in a folder inside the zip
            out_name = sanitize_filename(Path(member.filename).name)
            out_path = target / out_name
            with zf.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            extracted.append(str(out_path.relative_to(RAW_DIR)))
    log(f"  unpacked {len(extracted)} files -> archive/{year}/")
    return extracted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download all files")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    if MANIFEST_PATH.exists() and not args.force:
        manifest = json.loads(MANIFEST_PATH.read_text())

    resources = fetch_resources()

    n_downloaded = n_skipped = n_ignored = 0
    archive_re = re.compile(r"Archiv_Praemien_(\d{4})\.zip", re.IGNORECASE)

    for res in resources:
        url = res.get("download_url") or res.get("url")
        if not url:
            continue
        fname = resolve_filename(res)
        ext = Path(fname).suffix.lower()
        if ext in SKIP_EXTENSIONS:
            n_ignored += 1
            continue

        m = archive_re.search(fname)
        dest = (RAW_DIR / fname) if m else (CURRENT_DIR / fname)

        entry = manifest.get(fname)
        if entry and dest.exists() and not args.force:
            if sha256_of(dest) == entry.get("sha256"):
                log(f"skip {fname} (manifest sha256 match)")
                n_skipped += 1
                continue
            log(f"re-downloading {fname} (checksum mismatch vs manifest)")

        log(f"downloading {fname}")
        download(url, dest)

        entry = {
            "url": url,
            "path": str(dest.relative_to(RAW_DIR)),
            "sha256": sha256_of(dest),
            "size_bytes": dest.stat().st_size,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if m:
            entry["extracted"] = unpack_archive(dest, m.group(1))
        manifest[fname] = entry
        n_downloaded += 1

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log(
        f"done: {n_downloaded} downloaded, {n_skipped} skipped, "
        f"{n_ignored} ignored (pdf). Manifest: {MANIFEST_PATH}"
    )

    years = sorted(p.name for p in ARCHIVE_DIR.glob("[0-9]" * 4))
    log(f"archive years on disk: {', '.join(years) if years else 'none'}")
    if CURRENT_DIR.exists():
        log(f"current-year files: {len(list(CURRENT_DIR.iterdir()))} in data/raw/current/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
