#!/usr/bin/env python3
"""Run the full-image reading task against a vision model served by vLLM.

This script is only the *client*. Start the server first, with any
vision-capable checkpoint vLLM can serve:

    vllm serve <org>/<model-name> --port 8000

...then point a run config at it (see configs/*.yaml for examples) and run:

    python3 scripts/run_inference.py --config configs/<your-config>.yaml

Each manifest image is sent once, full-frame, with the prompt from
prompts.py, at temperature 0 for reproducibility. Raw model output is saved
verbatim -- scoring is a separate step (score.py) so the model's actual
text is always available for re-scoring if the metric changes.

Per-image latency and token counts are recorded alongside each prediction
(wall-clock time for that request, including retries; not a measure of
reading accuracy -- see score.py for that) and summarized at the end, so a
run's cost/throughput is on record next to its results.

Everything printed during the run is also written to a persistent
run.log next to the output predictions.json (not just this process's
stdout/stderr, which disappears once the terminal/session is gone) --
config used, per-image progress and latency, any failures, and the final
timing summary, each timestamped.
"""
import argparse
import base64
import datetime
import json
import mimetypes
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from openai import OpenAI

import prompts

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS = {"READ_ALL_TEXT": prompts.READ_ALL_TEXT}


class Logger:
    """Writes every message to stderr (as before) and to a persistent
    run.log file, each line timestamped, flushed immediately so the log
    is readable while the run is still in progress."""

    def __init__(self, log_path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(log_path, "w")

    def __call__(self, msg):
        line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, file=sys.stderr)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def format_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s" if hours else f"{minutes}m{seconds:02d}s"


def image_data_url(path):
    # BSTD mixes .jpg/.JPG/.jpeg/.png/.PNG -- declare the real MIME type
    # rather than hardcoding image/jpeg for every file.
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/jpeg"
    data = path.read_bytes()
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def run_one(client, cfg, record, prompt_text):
    image_path = REPO_ROOT / record["image_path"]
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
        ],
    }]

    # No fallback default here, deliberately: a config that doesn't set
    # repetition_penalty means "don't override it" -- vLLM then applies
    # whatever the model's own generation_config.json specifies, or its
    # own neutral default if the model ships none. Hardcoding a borrowed
    # number here for every model (as an earlier version of this script
    # did) is exactly the unfair, unreproducible per-model tuning this
    # benchmark's methodology explicitly rejects -- see README's Scoring
    # section and the repetition_penalty investigation notes.
    extra_body = {}
    if "repetition_penalty" in cfg:
        extra_body["repetition_penalty"] = cfg["repetition_penalty"]

    last_error = None
    started = time.monotonic()
    for attempt in range(cfg.get("max_retries", 3)):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                temperature=cfg.get("temperature", 0.0),
                max_tokens=cfg.get("max_tokens", 1024),
                extra_body=extra_body,
            )
            return {
                "id": record["id"],
                "image_path": record["image_path"],
                "dominant_script": record["dominant_script"],
                "prediction": resp.choices[0].message.content,
                "latency_sec": round(time.monotonic() - started, 3),
                "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
            }
        except Exception as exc:  # noqa: BLE001 -- transient network/server errors, retried
            last_error = exc
            time.sleep(2 ** attempt)
    return {
        "id": record["id"],
        "image_path": record["image_path"],
        "dominant_script": record["dominant_script"],
        "prediction": None,
        "error": str(last_error),
        "latency_sec": round(time.monotonic() - started, 3),
        "completion_tokens": None,
        "prompt_tokens": None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_path = REPO_ROOT / cfg["output"]
    log = Logger(out_path.parent / "run.log")
    log(f"run_name={cfg['run_name']} model={cfg['model']} base_url={cfg['base_url']}")
    log(f"manifest={cfg['manifest']} prompt={cfg.get('prompt', 'READ_ALL_TEXT')} "
        f"temperature={cfg.get('temperature', 0.0)} max_tokens={cfg.get('max_tokens', 1024)} "
        f"repetition_penalty={cfg.get('repetition_penalty', 'model default')} "
        f"concurrency={cfg.get('concurrency', 4)} max_retries={cfg.get('max_retries', 3)}")

    manifest_path = REPO_ROOT / cfg["manifest"]
    with open(manifest_path) as f:
        manifest = json.load(f)
    log(f"loaded {len(manifest)} images from {manifest_path}")

    prompt_text = PROMPTS[cfg.get("prompt", "READ_ALL_TEXT")]
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg.get("api_key", "EMPTY"))

    results = []
    concurrency = cfg.get("concurrency", 4)
    wall_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_one, client, cfg, r, prompt_text) for r in manifest]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            status = f"{result['latency_sec']}s" if result.get("prediction") is not None else f"FAILED: {result.get('error')}"
            elapsed = time.monotonic() - wall_start
            eta_sec = elapsed / i * (len(manifest) - i)
            log(f"{result['id']}  {i}/{len(manifest)} done ({status}) "
                f"[elapsed {format_duration(elapsed)}, ETA {format_duration(eta_sec)}]")
    wall_sec = round(time.monotonic() - wall_start, 1)

    results.sort(key=lambda r: r["id"])
    n_errors = sum(1 for r in results if r.get("error"))
    if n_errors:
        log(f"WARNING: {n_errors} images failed after retries")

    latencies = [r["latency_sec"] for r in results if r.get("prediction") is not None]
    timing = {
        "wall_clock_sec": wall_sec,
        "concurrency": concurrency,
        "num_requests": len(results),
        "num_failed": n_errors,
        "latency_sec": {
            "mean": round(statistics.mean(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
            "p95": round(statistics.quantiles(latencies, n=20)[18], 3) if len(latencies) >= 20 else None,
        } if latencies else None,
    }
    log(f"timing: wall_clock={wall_sec}s over {len(manifest)} images at concurrency={concurrency}; "
        f"per-image latency mean={timing['latency_sec']['mean']}s "
        f"median={timing['latency_sec']['median']}s max={timing['latency_sec']['max']}s"
        if latencies else "timing: no successful requests to summarize")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "run_name": cfg["run_name"],
            "model": cfg["model"],
            "prompt": cfg.get("prompt", "READ_ALL_TEXT"),
            "manifest": cfg["manifest"],
            "timing": timing,
            "predictions": results,
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote {out_path}")
    log.close()


if __name__ == "__main__":
    main()
