import os
import json
from collections import defaultdict

from bert_score import score as bertscore

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
    """Return dict: slide_id -> summary text, plus dict slide_id -> slide_text for OCR baseline."""
    data = load_json(path)
    sys_summaries = {}
    ocr_baseline = {}
    for entry in data:
        sid = entry["slide_id"]
        sys_summaries[sid] = entry.get("summary", "") or ""
        if "slide_text" in entry:
            ocr_baseline[sid] = entry.get("slide_text", "") or ""
    return sys_summaries, ocr_baseline


def main():
    print("=== Starting BERTScore evaluation ===")

    # ---- Check files ----
    for p in [GOLD_PATH, SPEECH_ONLY_PATH, SLIDES_ONLY_PATH, FULL_PATH]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing file: {p}")
            return

    # ---- Load gold ----
    gold_list = load_json(GOLD_PATH)
    gold_by_id = {item["slide_id"]: item["gold"].strip() for item in gold_list}
    slide_ids = sorted(gold_by_id.keys())
    print(f"Loaded {len(slide_ids)} gold summaries for slides: {slide_ids}")

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

    # You can change model_type for speed vs accuracy
    # Good defaults:
    # - "roberta-large" (stronger, slower)
    # - "distilroberta-base" (faster)
    model_type = "distilroberta-base"
    lang = "en"

    results = {}

    for name, sys_dict in systems.items():
        cands = []
        refs = []
        used_ids = []

        for sid in slide_ids:
            if sid not in sys_dict:
                print(f"[WARN] {name} missing slide {sid}, skipping that slide for this system.")
                continue
            cand = (sys_dict[sid] or "").strip()
            ref = gold_by_id[sid]

            # avoid empty strings crashing metrics usefulness
            if not cand:
                cand = " "
            cands.append(cand)
            refs.append(ref)
            used_ids.append(sid)

        if not cands:
            print(f"[WARN] No usable slides for system {name}.")
            continue

        print(f"\nScoring system: {name} on slides: {used_ids}")
        P, R, F1 = bertscore(cands, refs, lang=lang, model_type=model_type, verbose=False)

        # Convert tensors to floats
        P = [float(x) for x in P]
        R = [float(x) for x in R]
        F1 = [float(x) for x in F1]

        results[name] = {"P": P, "R": R, "F1": F1, "slide_ids": used_ids}

        # Print per-slide
        for sid, p, r, f1 in zip(used_ids, P, R, F1):
            print(f"Slide {sid:02d} | P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

        # Print averages
        avg_p = sum(P) / len(P)
        avg_r = sum(R) / len(R)
        avg_f1 = sum(F1) / len(F1)
        print(f"AVG {name:15s} | P={avg_p:.3f}  R={avg_r:.3f}  F1={avg_f1:.3f}")

    print("\n=== Summary (Average F1) ===")
    for name in results:
        avg_f1 = sum(results[name]["F1"]) / len(results[name]["F1"])
        print(f"{name:15s} | F1={avg_f1:.3f}")


if __name__ == "__main__":
    main()
