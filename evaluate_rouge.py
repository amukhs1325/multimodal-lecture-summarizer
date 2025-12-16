import os
import json
from collections import defaultdict

from rouge_score import rouge_scorer

LECTURE_ID = "lecture"
OUTPUT_DIR = "outputs"

GOLD_PATH = os.path.join(OUTPUT_DIR, "gold_summaries.json")
SPEECH_ONLY_PATH = os.path.join(OUTPUT_DIR, f"{LECTURE_ID}_slide_summaries_speech_only.json")
SLIDES_ONLY_PATH = os.path.join(OUTPUT_DIR, f"{LECTURE_ID}_slide_summaries_slides_only.json")
FULL_PATH = os.path.join(OUTPUT_DIR, f"{LECTURE_ID}_slide_summaries_full.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_system(path):
    """Return dict: slide_id -> summary text, and slide_id -> slide_text baseline (if present)."""
    data = load_json(path)
    sys_summaries = {}
    ocr_baseline = {}
    for entry in data:
        sid = entry["slide_id"]
        sys_summaries[sid] = entry.get("summary", "")
        if "slide_text" in entry:
            ocr_baseline[sid] = entry["slide_text"]
    return sys_summaries, ocr_baseline


def main():
    print("=== Starting ROUGE evaluation ===")

    # ---- Check files ----
    for p in [GOLD_PATH, SPEECH_ONLY_PATH, SLIDES_ONLY_PATH, FULL_PATH]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing file: {p}")
    if not os.path.exists(GOLD_PATH):
        print("Gold file missing; nothing to do.")
        return

    # ---- Load gold ----
    gold_list = load_json(GOLD_PATH)
    gold_by_id = {item["slide_id"]: item["gold"] for item in gold_list}
    print(f"Loaded {len(gold_by_id)} gold summaries for slides: {sorted(gold_by_id.keys())}")

    # ---- Load systems ----
    speech_sys, ocr_from_speech = load_system(SPEECH_ONLY_PATH)
    slides_sys, _ = load_system(SLIDES_ONLY_PATH)
    full_sys, _ = load_system(FULL_PATH)

    systems = {
        "raw_slides_ocr": ocr_from_speech,
        "speech_only": speech_sys,
        "slides_only": slides_sys,
        "full_multimodal": full_sys,
    }

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    results = {name: defaultdict(list) for name in systems}

    # ---- Compute scores ----
    for slide_id, gold in gold_by_id.items():
        gold_text = gold.strip()
        print(f"\n--- Slide {slide_id} ---")
        for name, sys_dict in systems.items():
            if slide_id not in sys_dict:
                print(f"[WARN] {name} has no summary for slide {slide_id}")
                continue
            pred = sys_dict[slide_id].strip()
            scores = scorer.score(gold_text, pred)
            r1 = scores["rouge1"].fmeasure
            rL = scores["rougeL"].fmeasure
            results[name]["rouge1"].append(r1)
            results[name]["rougeL"].append(rL)
            print(f"{name:15s} | R1={r1:.3f}  RL={rL:.3f}")

    # ---- Averages ----
    print("\n=== Average ROUGE over gold slides ===")
    for name, metrics in results.items():
        if not metrics["rouge1"]:
            print(f"{name:15s} | (no scores)")
            continue
        avg_r1 = sum(metrics["rouge1"]) / len(metrics["rouge1"])
        avg_rL = sum(metrics["rougeL"]) / len(metrics["rougeL"])
        print(f"{name:15s} | R1={avg_r1:.3f}  RL={avg_rL:.3f}")


if __name__ == "__main__":
    main()
