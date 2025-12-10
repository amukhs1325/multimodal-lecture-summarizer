import os
import json
import argparse
from textwrap import shorten

LECTURE_ID = "lecture"
OUTPUT_DIR = "outputs"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_slide_ocr():
    path = os.path.join(OUTPUT_DIR, f"{LECTURE_ID}_slides_ocr.json")
    return load_json(path)


def load_utterances_with_slides():
    path = os.path.join(OUTPUT_DIR, f"{LECTURE_ID}_utterances_with_slides.json")
    return load_json(path)


def load_summaries(tag: str):
    path = os.path.join(OUTPUT_DIR, f"{LECTURE_ID}_slide_summaries_{tag}.json")
    return load_json(path)


def get_summary_for_slide(summaries, slide_id: int) -> str:
    for item in summaries:
        if int(item["slide_id"]) == slide_id:
            return item.get("summary", "").strip()
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Inspect one slide across different summarization modalities."
    )
    parser.add_argument(
        "--slide_id",
        type=int,
        required=True,
        help="Slide index (0-based, e.g. 0, 1, 2, ...)",
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=800,
        help="Max characters to print for long fields.",
    )
    args = parser.parse_args()
    slide_id = args.slide_id
    trunc = args.truncate

    # Load data
    slide_ocr = load_slide_ocr()
    utterances = load_utterances_with_slides()
    speech_only = load_summaries("speech_only")
    slides_only = load_summaries("slides_only")
    full = load_summaries("full")

    # Slide OCR text
    if slide_id < 0 or slide_id >= len(slide_ocr):
        raise ValueError(f"slide_id {slide_id} out of range (0..{len(slide_ocr)-1})")

    slide_info = slide_ocr[slide_id]
    slide_text = slide_info.get("ocr_text", "").strip()

    # Collect all speech for this slide
    speech_list = [u["text"] for u in utterances if int(u["slide_id"]) == slide_id]
    speech_text = " ".join(speech_list).strip()

    # Summaries
    sum_speech_only = get_summary_for_slide(speech_only, slide_id)
    sum_slides_only = get_summary_for_slide(slides_only, slide_id)
    sum_full = get_summary_for_slide(full, slide_id)

    print("=" * 80)
    print(f"SLIDE {slide_id}")
    print("=" * 80)

    print("\n[SLIDE OCR TEXT]")
    print(shorten(slide_text, width=trunc, placeholder=" ..."))

    print("\n[SPEECH TEXT (assigned to this slide)]")
    print(shorten(speech_text, width=trunc, placeholder=" ..."))

    print("\n[SPEECH-ONLY SUMMARY]")
    print(shorten(sum_speech_only, width=trunc, placeholder=" ..."))

    print("\n[SLIDES-ONLY SUMMARY]")
    print(shorten(sum_slides_only, width=trunc, placeholder=" ..."))

    print("\n[FULL MULTIMODAL SUMMARY (SLIDES + SPEECH)]")
    print(shorten(sum_full, width=trunc, placeholder=" ..."))

    print("\nDone. Use different --slide_id values to explore other slides.")


if __name__ == "__main__":
    main()
