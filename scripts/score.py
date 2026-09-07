#!/usr/bin/env python3
"""Score a run's predictions against the benchmark manifest's gold text.

Two complementary metrics, both reported per gold string / predicted line
respectively, each in an exact and a fuzzy variant:

  RECALL (of BSTD's gold text): for every gold string BSTD annotated in an
  image, does it appear (exact substring, or a fuzzy-matching line) anywhere
  in the model's output? Answers "of the text actually annotated as ground
  truth, how much did the model correctly read?"

  PRECISION (of the model's output): for every line the model output, is it
  supported by some gold string (exact substring either direction, or a
  fuzzy match)? Answers "of what the model claimed to read, how much is
  actually backed by an annotation?" -- the complement to recall: on images
  with ZERO gold text (BSTD's annotators found nothing legible), no model
  abstains -- every predicted line on those images is automatically
  unsupported, which precision quantifies directly.

  Both aggregate into F1 (harmonic mean) per scope.

  exact : substring match (direction depends on the metric -- see below).
  fuzzy : within FUZZY_THRESHOLD similarity (difflib ratio). Forgives minor
          OCR-style slips (a dropped matra, a swapped character) without
          forgiving wrong answers.

CAVEAT ON PRECISION, read before trusting the number: BSTD's annotation
coverage is not guaranteed exhaustive -- a busy photo may have real,
legible text nobody bothered to annotate. Recall is unaffected by this
(it only checks whether annotated text was found), but precision is
directly vulnerable to it: a model correctly reading real text BSTD's
annotators skipped counts as a "false positive" here even though the
model did nothing wrong. Treat this precision number as an upper bound
on apparent hallucination, not a clean ground truth measurement -- it
is real signal (especially the zero-gold-image case, where there is
nothing to skip), just not as trustworthy as recall.

Usage:
    python3 score.py --manifest dataset/manifest_test.json \
        --predictions results/<run_name>/predictions.json \
        --out results/<run_name>/scores.json
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

FUZZY_THRESHOLD = 0.8


def normalize(text):
    text = unicodedata.normalize("NFC", text or "")
    text = text.strip().casefold()
    return re.sub(r"\s+", " ", text)


def score_recall(gold_texts, prediction_text):
    """Per gold item: does it appear in the model's output?"""
    pred_lines = [normalize(line) for line in (prediction_text or "").splitlines() if line.strip()]
    pred_blob = normalize(prediction_text or "")

    rows = []
    for gold in gold_texts:
        gold_norm = normalize(gold["text"])
        exact_hit = bool(gold_norm) and gold_norm in pred_blob
        best_ratio = max((SequenceMatcher(None, gold_norm, line).ratio() for line in pred_lines), default=0.0)
        fuzzy_hit = exact_hit or best_ratio >= FUZZY_THRESHOLD
        rows.append({
            "text": gold["text"],
            "script_language": gold["script_language"],
            "exact_hit": exact_hit,
            "fuzzy_hit": fuzzy_hit,
            "best_ratio": round(best_ratio, 3),
        })
    return rows


def score_precision(gold_texts, prediction_text):
    """Per predicted line: is it supported by some gold item?"""
    gold_norms = [g for g in (normalize(gold["text"]) for gold in gold_texts) if g]
    pred_lines = [line for line in (prediction_text or "").splitlines() if line.strip()]

    rows = []
    for line in pred_lines:
        line_norm = normalize(line)
        if not line_norm:
            continue
        # substring either direction: covers exact matches, a gold string
        # embedded in a longer output line, and a truncated/partial line
        # that is itself a substring of a longer gold string.
        exact_hit = any(g in line_norm or line_norm in g for g in gold_norms)
        best_ratio = max((SequenceMatcher(None, line_norm, g).ratio() for g in gold_norms), default=0.0)
        fuzzy_hit = exact_hit or best_ratio >= FUZZY_THRESHOLD
        rows.append({
            "line": line,
            "exact_hit": exact_hit,
            "fuzzy_hit": fuzzy_hit,
            "best_ratio": round(best_ratio, 3),
        })
    return rows


def aggregate(rows, prefix, count_key):
    total = len(rows)
    if total == 0:
        return {count_key: 0, f"{prefix}_exact": None, f"{prefix}_fuzzy": None}
    return {
        count_key: total,
        f"{prefix}_exact": round(sum(r["exact_hit"] for r in rows) / total, 4),
        f"{prefix}_fuzzy": round(sum(r["fuzzy_hit"] for r in rows) / total, 4),
    }


def f1(precision, recall):
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return round(2 * precision * recall / (precision + recall), 4)


def merge_with_f1(recall_stats, precision_stats):
    merged = {**recall_stats, **precision_stats}
    merged["f1_exact"] = f1(merged["precision_exact"], merged["recall_exact"])
    merged["f1_fuzzy"] = f1(merged["precision_fuzzy"], merged["recall_fuzzy"])
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = {r["id"]: r for r in json.load(open(args.manifest))}
    run = json.load(open(args.predictions))
    predictions = {p["id"]: p for p in run["predictions"]}

    missing = set(manifest) - set(predictions)
    if missing:
        print(f"WARNING: {len(missing)} manifest images have no prediction: {sorted(missing)[:5]}...",
              file=sys.stderr)

    per_image = []
    all_recall_rows, by_script_recall_rows = [], defaultdict(list)
    all_precision_rows, by_script_precision_rows = [], defaultdict(list)
    n_excluded = 0
    zero_gold_abstained, zero_gold_nonempty = [], []

    for image_id, record in manifest.items():
        pred = predictions.get(image_id, {})
        script = record["dominant_script"]
        error = pred.get("error")

        if error:
            # The model never got a chance to attempt this image (server
            # error, timeout, etc.) -- excluded from both metrics entirely
            # rather than scored as a 0, which would conflate "never
            # attempted" with "attempted and got it wrong."
            n_excluded += 1
            per_image.append({
                "id": image_id, "dominant_script": script, "error": error,
                "num_gold_items": len(record["gold_texts"]),
                "recall_exact": None, "recall_fuzzy": None,
                "num_predicted_lines": 0, "precision_exact": None, "precision_fuzzy": None,
            })
            continue

        prediction_text = pred.get("prediction")
        precision_rows = score_precision(record["gold_texts"], prediction_text)
        all_precision_rows.extend(precision_rows)
        by_script_precision_rows[script].extend(precision_rows)

        if len(record["gold_texts"]) == 0:
            # No recall to compute (nothing to check a transcription
            # against) but precision still applies in full: every line the
            # model produced here is automatically unsupported, since
            # there is no legible gold text at all for it to match.
            if normalize(prediction_text):
                zero_gold_nonempty.append(image_id)
            else:
                zero_gold_abstained.append(image_id)
            per_image.append({
                "id": image_id, "dominant_script": script, "error": None,
                "num_gold_items": 0, "recall_exact": None, "recall_fuzzy": None,
                **aggregate(precision_rows, "precision", "num_predicted_lines"),
            })
            continue

        recall_rows = score_recall(record["gold_texts"], prediction_text)
        all_recall_rows.extend(recall_rows)
        by_script_recall_rows[script].extend(recall_rows)
        per_image.append({
            "id": image_id, "dominant_script": script, "error": None,
            **aggregate(recall_rows, "recall", "num_gold_items"),
            **aggregate(precision_rows, "precision", "num_predicted_lines"),
        })

    if n_excluded:
        print(f"NOTE: {n_excluded} image(s) excluded from scoring (request failed, model never attempted them)",
              file=sys.stderr)

    script_key = lambda kv: (kv[0] is None, kv[0] or "")  # noqa: E731 -- None (zero-gold) sorts last
    all_scripts = set(by_script_recall_rows) | set(by_script_precision_rows)
    by_script = {
        script: merge_with_f1(
            aggregate(by_script_recall_rows.get(script, []), "recall", "num_gold_items"),
            aggregate(by_script_precision_rows.get(script, []), "precision", "num_predicted_lines"),
        )
        for script in all_scripts
    }
    by_script = dict(sorted(by_script.items(), key=script_key))

    report = {
        "run_name": run.get("run_name"),
        "model": run.get("model"),
        "num_images": len(manifest),
        "num_excluded_failed_requests": n_excluded,
        "overall": merge_with_f1(
            aggregate(all_recall_rows, "recall", "num_gold_items"),
            aggregate(all_precision_rows, "precision", "num_predicted_lines"),
        ),
        "by_script": by_script,
        "zero_gold_images": {
            "num_images": len(zero_gold_abstained) + len(zero_gold_nonempty),
            "num_abstained": len(zero_gold_abstained),
            "num_nonempty_output": len(zero_gold_nonempty),
            "nonempty_output_ids": sorted(zero_gold_nonempty),
        },
        "per_image": sorted(per_image, key=lambda r: r["id"]),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    def fmt(x):
        return f"{x:.3f}" if x is not None else "  n/a"

    print(f"\n{report['run_name']}  ({report['model']})  -- {report['num_images']} images\n")
    o = report["overall"]
    print(f"{'overall':<12} recall(n={o['num_gold_items']:<6}) exact={fmt(o['recall_exact'])} fuzzy={fmt(o['recall_fuzzy'])}  |  "
          f"precision(n={o['num_predicted_lines']:<6}) exact={fmt(o['precision_exact'])} fuzzy={fmt(o['precision_fuzzy'])}  |  "
          f"F1 exact={fmt(o['f1_exact'])} fuzzy={fmt(o['f1_fuzzy'])}")
    for script, s in report["by_script"].items():
        if script is None:
            continue  # the zero-gold bucket has no recall; reported separately below
        print(f"{script:<12} recall(n={s['num_gold_items']:<6}) exact={fmt(s['recall_exact'])} fuzzy={fmt(s['recall_fuzzy'])}  |  "
              f"precision(n={s['num_predicted_lines']:<6}) exact={fmt(s['precision_exact'])} fuzzy={fmt(s['precision_fuzzy'])}  |  "
              f"F1 exact={fmt(s['f1_exact'])} fuzzy={fmt(s['f1_fuzzy'])}")

    zg = report["zero_gold_images"]
    if zg["num_images"]:
        none_bucket = report["by_script"].get(None, {})
        print(f"\nzero-gold images (no legible text at all): {zg['num_images']}")
        print(f"  abstained (empty output, arguably correct): {zg['num_abstained']}")
        print(f"  produced non-empty output anyway: {zg['num_nonempty_output']}")
        if none_bucket.get("num_predicted_lines"):
            print(f"  of those output lines, precision exact={fmt(none_bucket['precision_exact'])} "
                  f"fuzzy={fmt(none_bucket['precision_fuzzy'])} (expected near 0 -- nothing there to support them)")

    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
