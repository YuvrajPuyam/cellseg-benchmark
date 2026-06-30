#!/usr/bin/env python3
"""Download + verify TissueNet from users.deepcell.org (token-gated).

Mirrors deepcell.utils._auth.fetch_data, but uses ONLY the Python standard library so it
runs on a bare cluster login node (no tensorflow / deepcell install needed just to download).

Requires DEEPCELL_ACCESS_TOKEN in the environment (see .env). Flow:
  POST https://users.deepcell.org/api/getData/  (header X-Api-Key, body s3_key=<asset>)
    -> {"url": <presigned S3 url>, "size": "4.1 GB"}
  GET <presigned url> -> stream to <out>/raw/, md5-verify, unzip to <out>/tissuenet/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API_ENDPOINT = "https://users.deepcell.org/api/getData/"
VERSIONS = {
    "1.1": {"asset_key": "data/tissuenet/tissuenet_v1-1.zip", "md5": "cab3b8f242aaee02035557b93546d9dc"},
    "1.0": {"asset_key": "data/tissuenet/tissuenet_1-0.zip", "md5": "f080c7732dd6de71e8e72e95a314e904"},
}
# Public, no-token single-image sample — handy for unit-test fixtures + local loader dev.
SAMPLE_URL = "https://deepcell-data.s3.us-west-1.amazonaws.com/multiplex/tissuenet-sample.npz"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_url(asset_key: str, token: str) -> tuple[str, str]:
    """POST to getData; return (presigned_url, human_size). Raises SystemExit on bad token."""
    body = urllib.parse.urlencode({"s3_key": asset_key}).encode()
    req = urllib.request.Request(API_ENDPOINT, data=body, headers={"X-Api-Key": token})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise SystemExit("ERROR: token rejected (403). Regenerate at https://users.deepcell.org")
        if e.code == 404:
            raise SystemExit(f"ERROR: asset not found: {asset_key}")
        raise
    return data["url"], data.get("size", "?")


def _download(url: str, dest: Path, expect_md5: str | None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expect_md5 and dest.exists() and _md5(dest) == expect_md5:
        print(f"[cache] {dest.name} already present + md5 OK", flush=True)
        return dest
    req = urllib.request.Request(url, headers={"user-agent": "Wget/1.20 (linux-gnu)"})
    t0, mb = time.time(), 0
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)  # 1 MB
            if not chunk:
                break
            f.write(chunk)
            mb += 1
            if mb % 100 == 0:
                print(f"  {mb} MB in {time.time() - t0:.0f}s", flush=True)
    print(f"[done] wrote {dest} ({mb} MB in {time.time() - t0:.0f}s)", flush=True)
    if expect_md5:
        got = _md5(dest)
        if got != expect_md5:
            raise SystemExit(f"ERROR: md5 mismatch: got {got}, expected {expect_md5}")
        print(f"[ok] md5 verified: {expect_md5}", flush=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Download/verify TissueNet (token-gated).")
    ap.add_argument("--version", default="1.1", choices=list(VERSIONS))
    ap.add_argument("--out", default="data", help="root: zip -> <out>/raw, extract -> <out>/tissuenet")
    ap.add_argument("--check", action="store_true", help="validate token + report size only (no download)")
    ap.add_argument("--sample", action="store_true", help="fetch the public no-token sample npz too")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("DEEPCELL_ACCESS_TOKEN")
    if not token:
        raise SystemExit("ERROR: DEEPCELL_ACCESS_TOKEN not set (see .env)")

    asset = VERSIONS[args.version]
    url, size = _resolve_url(asset["asset_key"], token)
    print(f"[token OK] TissueNet v{args.version} = {size}", flush=True)
    if args.check:
        print("CHECK_OK")
        return

    out = Path(args.out)
    zpath = _download(url, out / "raw" / Path(asset["asset_key"]).name, asset["md5"])

    if not args.no_extract:
        ex = out / "tissuenet"
        ex.mkdir(parents=True, exist_ok=True)
        print(f"[extract] {zpath} -> {ex}", flush=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(ex)
        print("[extract] top-level contents:", flush=True)
        for p in sorted(ex.iterdir()):
            sz = f"{p.stat().st_size // (1 << 20)} MB" if p.is_file() else "<dir>"
            print(f"    {p.name}  {sz}", flush=True)

    if args.sample:
        _download(SAMPLE_URL, out / "raw" / "tissuenet-sample.npz", None)

    print("DOWNLOAD_COMPLETE")


if __name__ == "__main__":
    main()
