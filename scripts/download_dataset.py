"""Download the fixed lightweight BraTS 2015 FLAIR/OT cohort over HTTPS.

Uses the Archive.org web seed listed by the Academic Torrents .torrent file.
Only Python's standard library is required, so this runs on Windows before
the project virtual environment and ML dependencies are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


INFOHASH = "c4f39a0a8e46e8d2174b8a8a81b9887150f44d50"
TORRENT_URL = f"https://academictorrents.com/download/{INFOHASH}.torrent"
ARCHIVE_WEBSEED = "https://archive.org/download/BRATS2015/"
ARCHIVE_METADATA_URL = "https://archive.org/metadata/BRATS2015"
SELECTION_VERSION = "archive-webseed-v2"
ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "data" / "metadata" / "brats2015.torrent"
RAW_ROOT = ROOT / "data" / "raw" / "BRATS2015"
SOURCE_MANIFEST = ROOT / "data" / "source_manifest.json"
SPLIT_MANIFEST = ROOT / "data" / "splits_v1.csv"
DOWNLOAD_REPORT = ROOT / "data" / "download_report.json"
USER_AGENT = "BraTS2015LightBenchmark/1.0 (Academic Torrents web seed)"


def bdecode(blob: bytes, offset: int = 0):
    token = blob[offset : offset + 1]
    if token == b"i":
        end = blob.index(b"e", offset)
        return int(blob[offset + 1 : end]), end + 1
    if token == b"l":
        items = []
        cursor = offset + 1
        while blob[cursor : cursor + 1] != b"e":
            item, cursor = bdecode(blob, cursor)
            items.append(item)
        return items, cursor + 1
    if token == b"d":
        items = {}
        cursor = offset + 1
        while blob[cursor : cursor + 1] != b"e":
            key, cursor = bdecode(blob, cursor)
            if not isinstance(key, bytes):
                raise ValueError("Invalid torrent dictionary key")
            value, cursor = bdecode(blob, cursor)
            items[key] = value
        return items, cursor + 1
    colon = blob.index(b":", offset)
    size = int(blob[offset:colon])
    start = colon + 1
    return blob[start : start + size], start + size


def bencode(value) -> bytes:
    if isinstance(value, int):
        return b"i" + str(value).encode("ascii") + b"e"
    if isinstance(value, bytes):
        return str(len(value)).encode("ascii") + b":" + value
    if isinstance(value, list):
        return b"l" + b"".join(bencode(item) for item in value) + b"e"
    if isinstance(value, dict):
        return b"d" + b"".join(bencode(key) + bencode(value[key]) for key in sorted(value)) + b"e"
    raise TypeError(f"Unsupported bencode type: {type(value)!r}")


def request_bytes(url: str, timeout: int) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout) as response:
        return response.read()


def load_torrent(timeout: int) -> tuple[dict, str]:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if METADATA_PATH.is_file():
        blob = METADATA_PATH.read_bytes()
    else:
        blob = request_bytes(TORRENT_URL, timeout)
        METADATA_PATH.write_bytes(blob)
    try:
        torrent, end = bdecode(blob)
        if end != len(blob) or not isinstance(torrent, dict):
            raise ValueError("Malformed torrent metadata")
        info = torrent[b"info"]
        actual_infohash = hashlib.sha1(bencode(info)).hexdigest()
        if actual_infohash != INFOHASH:
            raise ValueError(f"Torrent infohash mismatch: {actual_infohash}")
        if info[b"name"].decode("utf-8") != "BRATS2015":
            raise ValueError("Unexpected torrent root name")
        webseeds = torrent.get(b"url-list", [])
        if isinstance(webseeds, bytes):
            webseeds = [webseeds]
        if ARCHIVE_WEBSEED.encode("ascii") not in webseeds:
            raise ValueError("Expected Archive.org web seed is absent from torrent")
    except Exception:
        METADATA_PATH.unlink(missing_ok=True)
        raise
    return info, hashlib.sha256(blob).hexdigest()


def safe_path(parts: list[bytes]) -> str:
    decoded = [part.decode("utf-8") for part in parts]
    if any(part in ("", ".", "..") or any(char in part for char in "/\\:") for part in decoded):
        raise ValueError(f"Unsafe torrent path: {decoded!r}")
    return "/".join(decoded)


def load_archive_inventory(timeout: int) -> tuple[dict[str, dict], str]:
    """Get files actually hosted by the web seed; its copy is incomplete."""
    metadata = json.loads(request_bytes(ARCHIVE_METADATA_URL, timeout))
    inventory = {}
    for item in metadata["files"]:
        name = item["name"]
        if name.startswith("BRATS2015/") and "size" in item and "sha1" in item:
            inventory[name.removeprefix("BRATS2015/")] = {
                "bytes": int(item["size"]), "sha1": item["sha1"].lower()
            }
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return inventory, hashlib.sha256(canonical).hexdigest()


def make_plan(info: dict, torrent_sha256: str, archive_inventory: dict[str, dict], inventory_sha256: str) -> tuple[dict, list[dict]]:
    groups: dict[str, dict[str, dict[str, dict]]] = {"HGG": {}, "LGG": {}}
    license_file = None
    for item in info[b"files"]:
        path = safe_path(item[b"path"])
        size = int(item[b"length"])
        pieces = path.split("/")
        source = archive_inventory.get(path)
        if path == "License_CC_BY_NC_SA_3.0.txt":
            if source is not None and source["bytes"] == size:
                license_file = {"path": path, "bytes": size, "sha1": source["sha1"]}
            continue
        if source is None or source["bytes"] != size:
            continue
        if len(pieces) != 4 or pieces[0] != "training" or pieces[1] not in groups:
            continue
        grade, patient_id, filename = pieces[1], pieces[2], pieces[3]
        modality = "flair" if ".MR_Flair." in filename else "mask" if ".OT." in filename else None
        if modality is None:
            continue
        bucket = groups[grade].setdefault(patient_id, {})
        if modality in bucket:
            raise ValueError(f"More than one {modality} file for {grade}/{patient_id}")
        bucket[modality] = {"path": path, "bytes": size, "sha1": source["sha1"]}

    if license_file is None:
        raise ValueError("Dataset license is absent or differs on the Archive.org web seed")
    available_counts = {
        grade: sum(set(files) == {"flair", "mask"} for files in groups[grade].values())
        for grade in ("HGG", "LGG")
    }
    selected = []
    quotas = {"HGG": 80, "LGG": 20}
    split_quotas = {"HGG": (56, 12, 12), "LGG": (14, 3, 3)}
    for grade in ("HGG", "LGG"):
        valid = {patient: files for patient, files in groups[grade].items() if set(files) == {"flair", "mask"}}
        if len(valid) < quotas[grade]:
            raise ValueError(f"Only {len(valid)} complete {grade} cases; need {quotas[grade]}")
        patients = sorted(
            valid,
            key=lambda patient: (
                hashlib.sha256(f"brats2015-light-v1:{patient}".encode()).hexdigest(), patient
            ),
        )[: quotas[grade]]
        cases = [
            {"case_id": f"{grade}/{patient}", "grade": grade, "patient_id": patient, **valid[patient]}
            for patient in patients
        ]
        cases.sort(
            key=lambda case: (
                hashlib.sha256(f"brats2015-split-v1:{case['case_id']}".encode()).hexdigest(),
                case["case_id"],
            )
        )
        train_count, val_count, test_count = split_quotas[grade]
        for index, case in enumerate(cases):
            case["split"] = "train" if index < train_count else "val" if index < train_count + val_count else "test"
        assert len(cases) == train_count + val_count + test_count
        selected.extend(cases)

    selected.sort(key=lambda case: case["case_id"])
    manifest = {
        "dataset": "BraTS2015",
        "selection_version": SELECTION_VERSION,
        "source_page": f"https://academictorrents.com/details/{INFOHASH}",
        "torrent_url": TORRENT_URL,
        "infohash_sha1": INFOHASH,
        "torrent_sha256": torrent_sha256,
        "webseed": ARCHIVE_WEBSEED,
        "archive_metadata_url": ARCHIVE_METADATA_URL,
        "archive_inventory_sha256": inventory_sha256,
        "available_complete_cases": available_counts,
        "raw_root": "data/raw/BRATS2015",
        "selection": "From Archive.org files matching the verified torrent, with source SHA-1: 80 HGG + 20 LGG; SHA-256 order per grade; FLAIR and OT only",
        "license_file": license_file,
        "cases": selected,
    }
    return manifest, selected


def load_pinned_plan(info: dict, torrent_sha256: str) -> tuple[dict, list[dict]] | None:
    """Reuse the committed case list so a later mirror change cannot change the benchmark."""
    if not SOURCE_MANIFEST.is_file():
        return None
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("selection_version") != SELECTION_VERSION:
        return None
    if manifest.get("infohash_sha1") != INFOHASH or manifest.get("torrent_sha256") != torrent_sha256:
        raise ValueError("Pinned source manifest does not match the verified torrent")
    torrent_files = {safe_path(item[b"path"]): int(item[b"length"]) for item in info[b"files"]}
    cases = manifest["cases"]
    if len(cases) != 100 or len({case["case_id"] for case in cases}) != 100:
        raise ValueError("Pinned manifest must list exactly 100 unique cases")
    if Counter((case["grade"], case["split"]) for case in cases) != Counter({
        ("HGG", "train"): 56, ("HGG", "val"): 12, ("HGG", "test"): 12,
        ("LGG", "train"): 14, ("LGG", "val"): 3, ("LGG", "test"): 3,
    }):
        raise ValueError("Pinned manifest has an unexpected patient split")
    for case in cases:
        grade, patient = case["grade"], case["patient_id"]
        if not patient or any(char in patient for char in "/\\:") or case["case_id"] != f"{grade}/{patient}":
            raise ValueError(f"Unsafe or inconsistent case ID: {case['case_id']}")
        for part in ("flair", "mask"):
            if not case[part]["path"].startswith(f"training/{grade}/{patient}/"):
                raise ValueError(f"Pinned {part} is outside case folder: {case['case_id']}")
    for item in [manifest["license_file"]] + [case[part] for case in cases for part in ("flair", "mask")]:
        if torrent_files.get(item["path"]) != item["bytes"]:
            raise ValueError(f"Pinned file differs from torrent: {item['path']}")
        if len(item.get("sha1", "")) != 40 or any(char not in "0123456789abcdef" for char in item["sha1"]):
            raise ValueError(f"Pinned file lacks a valid source SHA-1: {item['path']}")
    return manifest, cases


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def save_plan(manifest: dict, cases: list[dict]) -> None:
    atomic_write(SOURCE_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    headers = ["case_id", "grade", "patient_id", "split", "flair_relpath", "mask_relpath", "flair_bytes", "mask_bytes"]
    lines = [",".join(headers)]
    for case in cases:
        fields = [
            case["case_id"], case["grade"], case["patient_id"], case["split"],
            "data/raw/BRATS2015/" + case["flair"]["path"],
            "data/raw/BRATS2015/" + case["mask"]["path"],
            str(case["flair"]["bytes"]), str(case["mask"]["bytes"]),
        ]
        # Torrent paths are validated and contain no commas in this dataset.
        lines.append(",".join(fields))
    atomic_write(SPLIT_MANIFEST, "\n".join(lines) + "\n")


def file_url(path: str) -> str:
    return ARCHIVE_WEBSEED + "BRATS2015/" + "/".join(quote(part, safe="") for part in path.split("/"))


def download_one(item: dict, timeout: int, retries: int) -> dict:
    relative_path = item["path"]
    expected_size = item["bytes"]
    target = RAW_ROOT.joinpath(*relative_path.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == expected_size:
        digest = hashlib.sha1()
        with target.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() == item["sha1"]:
            return {"path": relative_path, "bytes": expected_size, "status": "already_present"}
    url = file_url(relative_path)
    temporary = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        temporary.unlink(missing_ok=True)
        try:
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout) as response:
                if response.status != 200:
                    raise OSError(f"HTTP {response.status}")
                digest = hashlib.sha1()
                with temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                        digest.update(chunk)
            actual_size = temporary.stat().st_size
            if actual_size != expected_size:
                raise OSError(f"Size mismatch: expected {expected_size}, got {actual_size}")
            if digest.hexdigest() != item["sha1"]:
                raise OSError(f"SHA-1 mismatch for {relative_path}")
            os.replace(temporary, target)
            return {"path": relative_path, "bytes": expected_size, "status": "downloaded"}
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 8))
    raise RuntimeError("unreachable")


def main() -> int:
    # Windows PowerShell 5.1 may expose a legacy console encoding that cannot
    # print Vietnamese project paths. Escaping preserves a readable path.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
        sys.stderr.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="Generate source/split manifests without downloading images")
    parser.add_argument("--limit-cases", type=int, default=0, help="Download first N selected cases for a smoke test")
    parser.add_argument("--workers", type=int, default=4, help="Parallel HTTPS downloads (default: 4)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for each network read in seconds")
    args = parser.parse_args()
    if args.limit_cases < 0 or args.workers < 1 or args.workers > 8 or args.timeout < 5:
        parser.error("Use --limit-cases >= 0, 1 <= --workers <= 8 and --timeout >= 5")

    info, torrent_sha256 = load_torrent(args.timeout)
    pinned = load_pinned_plan(info, torrent_sha256)
    if pinned is None:
        archive_inventory, inventory_sha256 = load_archive_inventory(args.timeout)
        manifest, cases = make_plan(info, torrent_sha256, archive_inventory, inventory_sha256)
    else:
        manifest, cases = pinned
    save_plan(manifest, cases)
    all_files = [case[part] for case in cases for part in ("flair", "mask")]
    selected_bytes = sum(item["bytes"] for item in all_files)
    print(f"Verified torrent {INFOHASH}; selected {len(cases)} cases / {len(all_files)} MRI files")
    print(f"Expected transfer: {selected_bytes / 2**30:.3f} GiB; destination: {RAW_ROOT}")
    print(f"Manifests: {SOURCE_MANIFEST} and {SPLIT_MANIFEST}")
    if args.plan_only:
        return 0

    to_download = cases[: args.limit_cases] if args.limit_cases else cases
    files = [manifest["license_file"]] + [case[part] for case in to_download for part in ("flair", "mask")]
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, item, args.timeout, 4): item["path"] for item in files}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[{len(results)}/{len(files)}] {result['status']}: {path}")
            except Exception as exc:
                errors.append({"path": path, "error": str(exc)})
                print(f"FAILED: {path}: {exc}")
    report = {
        "selected_cases": len(cases),
        "downloaded_cases_this_run": len(to_download),
        "expected_files_this_run": len(files),
        "completed_files_this_run": len(results),
        "errors": errors,
    }
    atomic_write(DOWNLOAD_REPORT, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if errors:
        print(f"{len(errors)} file(s) failed; rerun the same command to retry incomplete files")
        return 1
    print("Download complete for this run. Rerunning skips files with the expected size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
