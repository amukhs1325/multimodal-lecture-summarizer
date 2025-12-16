import os
import re
import json
from collections import defaultdict

from rouge_score import rouge_scorer
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
    """Return dict: slide_id -> summary, and dict slide_id -> slide_text baseline."""
    data = load_json(path)
    sys_summaries = {}
    ocr_text = {}
    for entry in data:
        sid = entry["slide_id"]
        sys_summaries[sid] = (entry.get("summary", "") or "").strip()
        if "slide_text" in entry:
            ocr_text[sid] = (entry.get("slide_text", "") or "").strip()
    return sys_summaries, ocr_text


def tokenize_simple(s: str):
    return re.findall(r"[A-Za-z]+", s.lower())


def classify_slide(ocr_text: str):
    """
    Buckets:
      - table-heavy: lots of digits or degree symbols, or many numeric tokens
      - text-heavy: OCR token count high
      - diagram-heavy: OCR token count low
    """
    t = ocr_text or ""
    words = tokenize_simple(t)
    word_count = len(words)

    # table heuristics
    digits = re.findall(r"\d", t)
    deg = "°" in t or "deg" in t.lower() or "c" in t.lower()  # loose, but helps for °C slides
    num_tokens = re.findall(r"\b\d+(\.\d+)?\b", t)

    # "table-heavy" if many digits or many numeric tokens or degrees
    if len(digits) >= 20 or len(num_tokens) >= 8 or "°c" in t.lower() or deg and len(num_tokens) >= 5:
        return "table-heavy", word_count

    # text-heavy vs diagram-heavy based on OCR word count
    if word_count >= 35:
        return "text-heavy", word_count
    else:
        return "diagram-heavy", word_count


def safe_get(d, sid):
    return (d.get(sid, "") or "").strip()


def compute_rouge(golds, preds):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1_list, rl_list = [], []
    for g, p in zip(golds, preds):
        if not p.strip():
            p = " "
        scores = scorer.score(g, p)
        r1_list.append(scores["rouge1"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)
    return sum(r1_list) / len(r1_list), sum(rl_list) / len(rl_list)


def compute_bertscore(golds, preds, model_type="distilroberta-base"):
    # bert-score expects lists of strings
    preds = [p if p.strip() else " " for p in preds]
    P, R, F1 = bertscore(preds, golds, lang="en", model_type=model_type, verbose=False)
    f1 = [float(x) for x in F1]
    return sum(f1) / len(f1)


def main():
    print("=== Stratified Evaluation by Slide Type ===")

    for p in [GOLD_PATH, SPEECH_ONLY_PATH, SLIDES_ONLY_PATH, FULL_PATH]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing file: {p}")
            return

    gold_list = load_json(GOLD_PATH)
    gold_by_id = {x["slide_id"]: x["gold"].strip() for x in gold_list}
    slide_ids = sorted(gold_by_id.keys())
    print(f"Gold slides: {slide_ids}")

    speech_sys, ocr_from_speech = load_system(SPEECH_ONLY_PATH)
    slides_sys, _ = load_system(SLIDES_ONLY_PATH)
    full_sys, _ = load_system(FULL_PATH)

    systems = {
        "raw_slides_ocr": ocr_from_speech,
        "speech_only": speech_sys,
        "slides_only": slides_sys,
        "full_multimodal": full_sys,
    }

    # classify slides using OCR baseline text
    buckets = defaultdict(list)
    meta = {}

    for sid in slide_ids:
        ocr = safe_get(ocr_from_speech, sid)
        bucket, wc = classify_slide(ocr)
        buckets[bucket].append(sid)
        meta[sid] = {"bucket": bucket, "ocr_words": wc}

    print("\n=== Bucket assignments ===")
    for b in ["table-heavy", "text-heavy", "diagram-heavy"]:
        ids = buckets.get(b, [])
        print(f"{b:12s}: {ids}")

    print("\n=== Results: ROUGE + BERTScore by bucket ===")
    # store results: bucket -> system -> metrics
    for bucket_name, ids in buckets.items():
        if not ids:
            continue

        golds = [gold_by_id[sid] for sid in ids]

        print(f"\n--- Bucket: {bucket_name} (n={len(ids)}) ---")

        for sys_name, sys_dict in systems.items():
            preds = []
            used = 0
            for sid in ids:
                if sid not in sys_dict:
                    continue
                preds.append(safe_get(sys_dict, sid))
                used += 1

            if used == 0:
                print(f"{sys_name:15s} | (no outputs)")
                continue

            # if a system is missing some slides in the bucket, align golds to preds
            aligned_golds = []
            aligned_preds = []
            for sid in ids:
                if sid in sys_dict:
                    aligned_golds.append(gold_by_id[sid])
                    aligned_preds.append(safe_get(sys_dict, sid))

            r1, rl = compute_rouge(aligned_golds, aligned_preds)
            bs_f1 = compute_bertscore(aligned_golds, aligned_preds)

            print(f"{sys_name:15s} | R1={r1:.3f}  RL={rl:.3f}  BERT-F1={bs_f1:.3f}")


if __name__ == "__main__":
    main()
