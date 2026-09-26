"""Verify the selected BraTS 2015 files and patient-level split manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source_manifest.json"
SPLITS = ROOT / "data" / "splits_v1.csv"
CHECKSUMS = ROOT / "data" / "file_sha256.csv"


def file_hashes(path: Path) -> tuple[str, str]:
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            sha1.update(chunk)
            sha256.update(chunk)
    return sha1.hexdigest(), sha256.hexdigest()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
        sys.stderr.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true", help="Allow missing MRI files during a smoke download")
    parser.add_argument("--write-checksums", action="store_true", help="Write data/file_sha256.csv after a full successful download")
    args = parser.parse_args()
    if not SOURCE.is_file() or not SPLITS.is_file():
        parser.error("Run scripts/download_dataset.py first to create the manifests")

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    cases = source["cases"]
    if len(cases) != 100 or len({case["case_id"] for case in cases}) != 100:
        raise ValueError("Source manifest must contain 100 unique case IDs")
    if Counter(case["grade"] for case in cases) != Counter({"HGG": 80, "LGG": 20}):
        raise ValueError("Incorrect HGG/LGG case counts")

    with SPLITS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100 or {row["case_id"] for row in rows} != {case["case_id"] for case in cases}:
        raise ValueError("Split manifest does not match the selected cases")
    cases_by_id = {case["case_id"]: case for case in cases}
    for row in rows:
        case = cases_by_id[row["case_id"]]
        expected = {
            "grade": case["grade"], "patient_id": case["patient_id"], "split": case["split"],
            "flair_relpath": "data/raw/BRATS2015/" + case["flair"]["path"],
            "mask_relpath": "data/raw/BRATS2015/" + case["mask"]["path"],
            "flair_bytes": str(case["flair"]["bytes"]),
            "mask_bytes": str(case["mask"]["bytes"]),
        }
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError(f"Split row differs from the source manifest: {row['case_id']}")
    if Counter(row["split"] for row in rows) != Counter({"train": 70, "val": 15, "test": 15}):
        raise ValueError("Incorrect train/val/test counts")
    if Counter((row["grade"], row["split"]) for row in rows) != Counter({
        ("HGG", "train"): 56, ("HGG", "val"): 12, ("HGG", "test"): 12,
        ("LGG", "train"): 14, ("LGG", "val"): 3, ("LGG", "test"): 3,
    }):
        raise ValueError("Incorrect split counts by grade")

    raw_root = ROOT / source["raw_root"]
    items = [source["license_file"]] + [case[part] for case in cases for part in ("flair", "mask")]
    expected_hashes = {}
    if CHECKSUMS.is_file():
        with CHECKSUMS.open(encoding="utf-8", newline="") as handle:
            checksum_rows = list(csv.DictReader(handle))
        expected_hashes = {row["path"]: row["sha256"] for row in checksum_rows}
        if len(checksum_rows) != len(items) or set(expected_hashes) != {item["path"] for item in items}:
            raise ValueError("SHA-256 inventory does not cover exactly the selected files")

    missing, invalid, present = [], [], []
    for item in items:
        relative = item["path"]
        path = raw_root.joinpath(*relative.split("/"))
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size != item["bytes"]:
            invalid.append(f"size mismatch: {relative}")
            continue
        if relative.endswith(".mha"):
            with path.open("rb") as handle:
                header = handle.read(1024)
            # BraTS 2015 mixes full MetaImage headers with shorter valid ones
            # that start at NDims and omit ObjectType.
            if b"NDims = 3" not in header or b"ElementDataFile = LOCAL" not in header:
                invalid.append(f"invalid MHA header: {relative}")
                continue
        source_sha1, digest = file_hashes(path)
        if source_sha1 != item.get("sha1"):
            invalid.append(f"source SHA-1 mismatch: {relative}")
            continue
        if relative in expected_hashes and expected_hashes[relative] != digest:
            invalid.append(f"SHA-256 mismatch: {relative}")
            continue
        present.append((relative, item["bytes"], digest))

    print(f"Cases: {len(cases)}; split: 70 train / 15 val / 15 test")
    print(f"Files: {len(present)}/{len(items)} valid; {len(missing)} missing; {len(invalid)} invalid")
    for issue in (invalid + missing)[:12]:
        print(f"  {issue}")
    if args.write_checksums:
        if missing or invalid:
            print("Cannot write checksums before every selected file is valid")
            return 1
        with CHECKSUMS.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "bytes", "sha256"])
            writer.writerows(present)
        print(f"Wrote SHA-256 inventory: {CHECKSUMS}")
    if invalid or (missing and not args.allow_partial):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
