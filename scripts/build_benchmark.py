#!/usr/bin/env python3
"""Curate a full-image reading benchmark manifest from BSTD.

BSTD is a scene-text-in-the-wild dataset: real photographs taken across
India, each annotated with word/phrase-level polygons, transcribed text,
and a script-language label. This script turns that raw annotation file
into a benchmark manifest for a single task: show a model the *whole*
photograph and ask it to transcribe every piece of legible text it can
read. Scoring against the manifest happens in score.py.

Cleanup performed here, all driven by inspecting the raw data first:
  - `script_language` is free text and noisy (typos, stray non-language
    values) -> normalized via LANGUAGE_NORMALIZE, unrecognized values dropped.
  - Annotations where the *text itself* is an illegibility placeholder
    ("UNK", "NA") are dropped -- these are not gold answers.
  - Images with zero legible annotations (58 of BSTD's 1,319 test-split
    images -- mostly annotated but every annotation marked "UNK", i.e.
    illegible even to the human annotator) are KEPT by default, not
    dropped: excluding them from the manifest would mean the model's
    behavior on them -- does it correctly produce nothing, or hallucinate
    text that isn't legibly there? -- never gets exercised at all. There
    is no gold text to compute recall against for these, so score.py
    reports them separately rather than folding them into recall.
  - --min-gold/--max-gold (default 0 / uncapped) can still exclude images
    with very little or very much legible text, if you deliberately want
    a stricter, more uniform-difficulty subset instead of full coverage --
    at the cost of scoring against a different set of images than BSTD's
    own split actually contains.

Usage:
    python3 build_benchmark.py --split test --per-script-cap 999999 --out dataset/manifest_test.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every raw script_language value observed in BSTD_v17.57.json, mapped to a
# canonical label, or to None to drop the annotation (illegible / not a
# language).
LANGUAGE_NORMALIZE = {
    "english": "english", "english\n": "english", "english ": "english",
    "engish": "english", "engilsh": "english",
    "hindi": "hindi", "hini": "hindi",
    "assamese": "assamese", "assamese\n": "assamese",
    "bengali": "bengali", "bengalibengali": "bengali",
    "tamil": "tamil",
    "gujarati": "gujarati", "\t\nધામ\ngujarati\n": "gujarati",
    "urdu": "urdu",
    "punjabi": "punjabi", "punjabi ": "punjabi",
    "telugu": "telugu",
    "odia": "odia",
    "kannada": "kannada",
    "malayalam": "malayalam",
    "marathi": "marathi",
    "meitei": "meitei",
    "vivo": None, "india": None,
    "unk": None, "unk ": None, "na": None,
}

ILLEGIBLE_TEXT_MARKERS = {"unk", "na", "n/a"}


def normalize_language(raw):
    if raw is None:
        return None
    if raw in LANGUAGE_NORMALIZE:
        return LANGUAGE_NORMALIZE[raw]
    return LANGUAGE_NORMALIZE.get(raw.lower())


def clean_text(raw):
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.lower() in ILLEGIBLE_TEXT_MARKERS:
        return None
    return text


def image_relpath(entry, bstd_root_relpath):
    # entry["image_name"] (e.g. "A/image_2287.JPG") preserves each file's
    # actual extension case -- BSTD mixes .jpg and .JPG. Reconstructing the
    # path from the dict key instead (as an earlier version of this script
    # did) silently hardcodes lowercase ".jpg" and drops every .JPG file as
    # "missing" on a case-sensitive filesystem: 160 images dataset-wide.
    return f"{bstd_root_relpath}/{entry['image_name']}"


def build_records(data, bstd_root_relpath, split=None, min_gold=0, max_gold=None):
    # min_gold=0 (the default) keeps every image, including ones with no
    # legible gold text at all -- score.py reports these separately (does
    # the model correctly output nothing, or hallucinate text that isn't
    # there?) rather than folding them into recall, since there's no gold
    # string to check a transcription against either way. Raise min_gold
    # only if you want those images excluded from the manifest entirely.
    records = []
    dropped = Counter()
    for key, entry in data.items():
        if split is not None and entry.get("split") != split:
            dropped["wrong_split"] += 1
            continue

        relpath = image_relpath(entry, bstd_root_relpath)
        if not (REPO_ROOT / relpath).is_file():
            dropped["no_file_on_disk"] += 1
            continue

        gold, lang_counts, seen = [], Counter(), set()
        for ann in entry.get("annotations", {}).values():
            text = clean_text(ann.get("text"))
            lang = normalize_language(ann.get("script_language"))
            if text is None or lang is None or (text, lang) in seen:
                continue
            seen.add((text, lang))
            gold.append({"text": text, "script_language": lang})
            lang_counts[lang] += 1

        if len(gold) < min_gold:
            dropped["zero_legible" if len(gold) == 0 else "too_few_legible"] += 1
            continue
        if max_gold is not None and len(gold) > max_gold:
            dropped["too_dense_to_grade"] += 1
            continue

        records.append({
            "id": key,
            "image_path": relpath,
            "split": entry.get("split"),
            "environment": entry.get("environment"),
            "lighting": entry.get("lighting"),
            "source_url": entry.get("url"),
            "dominant_script": lang_counts.most_common(1)[0][0] if lang_counts else None,
            "script_counts": dict(lang_counts),
            "num_gold": len(gold),
            "gold_texts": gold,
        })

    print(f"kept {len(records)} images; dropped {dict(dropped)}", file=sys.stderr)
    return records


def stratified_sample(records, per_script_cap, sample_size, seed):
    """Cap per dominant script, then (optionally) round-robin across
    scripts down to an exact total so a small pilot still spans languages
    instead of being dominated by whichever script has the most data."""
    rng = random.Random(seed)  # nosec B311 -- reproducible dataset sampling, not a security context
    by_script = defaultdict(list)
    for r in records:
        by_script[r["dominant_script"]].append(r)

    buckets = []
    # dominant_script is None for zero-gold images (no legible text at all
    # -- see build_records) -- sort key keeps that group last instead of
    # comparing None to str, which Python rejects outright.
    for script, items in sorted(by_script.items(), key=lambda kv: (kv[0] is None, kv[0] or "")):
        rng.shuffle(items)
        buckets.append(items[:per_script_cap])
        print(f"  {script}: {len(buckets[-1])}/{len(items)} images", file=sys.stderr)

    if sample_size is None:
        sample = [r for bucket in buckets for r in bucket]
    else:
        sample = []
        i = 0
        while len(sample) < sample_size and any(buckets):
            bucket = buckets[i % len(buckets)]
            if bucket:
                sample.append(bucket.pop())
            i += 1
            if i % len(buckets) == 0 and not any(buckets):
                break

    rng.shuffle(sample)
    return sample


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bstd-json", default=str(REPO_ROOT / "BSTD/detection/BSTD_v17.57.json"))
    ap.add_argument("--bstd-root", default="BSTD/detection",
                     help="path to the BSTD detection dir, relative to the repo root")
    ap.add_argument("--out", required=True, help="output manifest path")
    ap.add_argument("--split", choices=["train", "test"], default=None,
                     help="restrict to BSTD's own train/test split field (default: both)")
    ap.add_argument("--per-script-cap", type=int, default=20,
                     help="max images to keep per dominant script before sampling; "
                          "use a large number (e.g. 999999) for effectively uncapped")
    ap.add_argument("--sample-size", type=int, default=None,
                     help="if set, round-robin across scripts down to this many images total")
    ap.add_argument("--min-gold", type=int, default=0,
                     help="drop images with fewer than this many legible gold texts "
                          "(default 0: keep everything, including images with no legible "
                          "text at all -- score.py reports these separately)")
    ap.add_argument("--max-gold", type=int, default=None,
                     help="drop images with more than this many legible gold texts "
                          "(default: uncapped)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    with open(args.bstd_json) as f:
        data = json.load(f)
    print(f"loaded {len(data)} images from {args.bstd_json}", file=sys.stderr)

    records = build_records(data, args.bstd_root, split=args.split,
                             min_gold=args.min_gold, max_gold=args.max_gold)

    print("sampling by dominant script:", file=sys.stderr)
    sample = stratified_sample(records, args.per_script_cap, args.sample_size, args.seed)
    print(f"final manifest: {len(sample)} images", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
