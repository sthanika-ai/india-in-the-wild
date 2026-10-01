# India in the Wild

A scene-text reading benchmark that tests whether vision-language models can read what India actually looks like: hand-painted signage, fare boards and multi-script hoardings, photographed in the wild.

[![License: MIT](https://img.shields.io/badge/code-MIT-56BF4F?style=flat-square&labelColor=1E281F)](LICENSE)
[![Data: BSTD](https://img.shields.io/badge/data-BSTD%20(CC%20BY--SA%204.0)-56BF4F?style=flat-square&labelColor=1E281F)](#license)
[![Report](https://img.shields.io/badge/report-sthanika.ai-56BF4F?style=flat-square&labelColor=1E281F&logo=firefox&logoColor=white)](https://sthanika.ai/research/india-in-the-wild-2026)

## What it measures

Whether a model can both find and read text in real street photographs. It is shown one full, unedited photo and asked to transcribe every legible piece of text in its original script, one item per line (exact prompt in `scripts/prompts.py`). There is no cropping and no hint about where the text is. Every item has a verifiable gold answer, so scoring is programmatic and not model-judged.

The source is BSTD (BharatSceneTextDataset), real photographs from across India annotated at the word or phrase level across 14 scripts: English, Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Odia, Assamese, Urdu and Meitei. Eight open-weight VLMs have been run over BSTD's full test split (1,319 images, 24,987 gold text items). It does not cover handwritten forms or printed documents, because BSTD is street-scene photography. Report: [sthanika.ai](https://sthanika.ai/research/india-in-the-wild-2026)

## Quickstart

```bash
git clone https://github.com/sthanika-ai/india-in-the-wild.git
cd india-in-the-wild
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Serve a vision-capable model with vLLM (separate terminal)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8234

# 2. Run the benchmark
python3 scripts/run_inference.py --config configs/qwen7b_test_split.yaml

# 3. Score it
python3 scripts/score.py \
  --manifest dataset/manifest_test.json \
  --predictions results/test-split-qwen2.5-vl-7b/predictions.json \
  --out results/test-split-qwen2.5-vl-7b/scores.json
```

The manifest `dataset/manifest_test.json` is already committed. To rebuild it from BSTD:

```bash
python3 scripts/build_benchmark.py --out dataset/manifest_test.json --split test --per-script-cap 999999
```

Notes:

- **Other models.** Each file in `configs/` is a drop-in swap for step 2, and its header comment carries the exact `vllm serve` command it needs. Configs exist for Qwen2.5-VL-7B and 32B, Gemma-3-27B, Aria, MiniCPM-V-2.6, Pixtral-12B, LLaVA-1.6-34B and Aya-Vision-8B.
- **Hosted APIs.** `run_inference.py` only talks to an OpenAI-compatible endpoint (set `base_url` and model in the config), so any vLLM-served checkpoint or hosted API that accepts chat-completions image input works without changing the script.
- **Outputs.** A run writes `results/<run_name>/` with `predictions.json` (raw output, latency and token counts per image), `run.log`, and once scored `scores.json` (recall, precision and F1, overall and per dominant script). These are generated locally, not committed.
- **Cleaning.** `build_benchmark.py` normalises noisy `script_language` labels, drops 3,143 "UNK"/"NA" placeholder annotations in the test split, and reads each entry's own file extension, which recovers 60 test images an earlier version dropped as missing.

## Results

All eight models saw the identical manifest (1,319 images) and prompt, at temperature 0, `max_tokens` 1024 and concurrency 8, less the one or two images per model that failed every retry. Rows are ordered by fuzzy recall.

| # | model | recall (exact / fuzzy) | precision (exact / fuzzy) | F1 (exact / fuzzy) |
|---|---|---|---|---|
| 1 | Qwen2.5-VL-32B | 0.444 / 0.460 | 0.461 / 0.483 | 0.453 / 0.471 |
| 2 | Qwen2.5-VL-7B | 0.386 / 0.397 | 0.548 / 0.564 | 0.453 / 0.466 |
| 3 | Gemma-3-27B | 0.292 / 0.302 | 0.414 / 0.434 | 0.343 / 0.356 |
| 4 | Aria | 0.176 / 0.179 | 0.242 / 0.245 | 0.204 / 0.207 |
| 5 | MiniCPM-V-2.6 | 0.151 / 0.154 | 0.376 / 0.395 | 0.216 / 0.221 |
| 6 | Pixtral-12B | 0.119 / 0.121 | 0.322 / 0.330 | 0.173 / 0.178 |
| 7 | LLaVA-1.6-34B | 0.115 / 0.116 | 0.439 / 0.442 | 0.182 / 0.184 |
| 8 | Aya-Vision-8B | 0.056 / 0.059 | 0.297 / 0.308 | 0.094 / 0.098 |

How scoring works:

- **Recall:** of BSTD's gold text, how much did the model find? A gold string counts if it appears anywhere in the output, as an exact substring or a fuzzy-matching line.
- **Precision:** of the model's output, how much is supported by some gold string?
- **Exact vs fuzzy:** exact is a normalised substring match. Fuzzy also accepts a match within 0.8 similarity (`difflib.SequenceMatcher`), which forgives a dropped matra or one swapped character but not a wrong reading. Neither checks ordering.

Caveats:

- BSTD's annotations are not guaranteed exhaustive. A model that correctly reads real text the annotators skipped is counted as a false positive, so treat precision as an upper bound on apparent hallucination. Recall is unaffected. See the `scripts/score.py` docstring for the full reasoning.
- Pixtral-12B and LLaVA-1.6-34B ran at `repetition_penalty` 1.30 and the other six at 1.05, so comparisons involving those two rows are not strictly controlled.

Full report: [sthanika.ai](https://sthanika.ai/research/india-in-the-wild-2026)

## Citation

If you use this benchmark, cite this repo and BSTD's underlying paper per the BSTD README.

```bibtex
@software{india_in_the_wild2026,
  title  = {India in the Wild: A Scene-Text Reading Benchmark for Indic Scripts},
  author = {{sthanika-ai}},
  year   = {2026},
  url    = {https://github.com/sthanika-ai/india-in-the-wild}
}
```

## License

Code (everything under `scripts/` and `configs/`) is MIT, see [LICENSE](LICENSE). The data belongs to BSTD and is not covered by that grant: BSTD is Apache-2.0, and its README states every image is CC BY-SA 4.0, sourced from Wikimedia Commons.

## Related

- BSTD (BharatSceneTextDataset): the source of the images and gold annotations
- Companion work from sthanika-ai: [BKP-500 model runs](https://github.com/sthanika-ai/BKP-500-model-runs), [token_fertility](https://github.com/sthanika-ai/token_fertility)
- Site: [sthanika.ai](https://sthanika.ai)
