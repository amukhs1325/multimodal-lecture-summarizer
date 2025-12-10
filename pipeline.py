import os
import json
from dataclasses import dataclass, asdict
from typing import List

import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

import librosa
from transformers import pipeline as hf_pipeline
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


# ========== CONFIG ==========
LECTURE_ID = "lecture"   # will use your lecture_slides.pdf, lecture_video.mp4, lecture_transcript.pdf

SLIDES_DIR = "slides"
AUDIO_DIR = "audio"
TRANSCRIPTS_DIR = "transcripts"
OUTPUT_DIR = "outputs"

SLIDES_PDF = os.path.join(SLIDES_DIR, f"{LECTURE_ID}_slides.pdf")
AUDIO_FILE = os.path.join(AUDIO_DIR, f"{LECTURE_ID}_video.mp4")
TRANSCRIPT_PDF = os.path.join(TRANSCRIPTS_DIR, f"{LECTURE_ID}_transcript.pdf")


# ========== DATA CLASSES ==========

@dataclass
class SlideInfo:
    slide_id: int
    image_path: str
    ocr_text: str


@dataclass
class Utterance:
    idx: int
    start: float
    end: float
    text: str
    slide_id: int


@dataclass
class SlideSummary:
    slide_id: int
    slide_text: str
    speech_text: str
    summary: str


# ========== UTILS ==========

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ========== STEP 1: SLIDES → IMAGES + OCR ==========

def process_slides(slides_pdf: str) -> List[SlideInfo]:
    """
    Convert each page of the PDF into an image and run OCR over the whole slide.
    For now we treat the whole slide as one region (simple but works).
    """
    print(f"[slides] Converting {slides_pdf} to images and running OCR...")
    out_img_dir = os.path.join(OUTPUT_DIR, "slide_images")
    ensure_dir(out_img_dir)

    pages = convert_from_path(slides_pdf, dpi=200)

    slide_infos: List[SlideInfo] = []

    for i, page in enumerate(tqdm(pages, desc="Slides")):
        img_path = os.path.join(out_img_dir, f"slide_{i:02d}.png")
        page.save(img_path, "PNG")

        # Simple full-slide OCR
        ocr_text = pytesseract.image_to_string(Image.open(img_path))

        slide_infos.append(SlideInfo(
            slide_id=i,
            image_path=img_path,
            ocr_text=ocr_text.strip()
        ))

    # Save for inspection
    slides_json_path = os.path.join(OUTPUT_DIR, f"{LECTURE_ID}_slides_ocr.json")
    with open(slides_json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in slide_infos], f, indent=2, ensure_ascii=False)
    print(f"[slides] Saved slide OCR info to {slides_json_path}")

    return slide_infos


# ========== STEP 2: AUDIO → DURATION (and room for prosody later) ==========

def get_audio_duration(audio_path: str) -> float:
    print(f"[audio] Loading audio from {audio_path} for duration...")
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"[audio] Duration: {duration:.2f} seconds")
    return duration


# ========== STEP 3: TRANSCRIPT PDF → TEXT SENTENCES → APPROX TIMESTAMPS ==========

def extract_text_from_pdf(pdf_path: str) -> str:
    print(f"[transcript] Extracting raw text from {pdf_path} ...")
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    full_text = "\n".join(texts)
    return full_text


def build_utterances_from_transcript(pdf_path: str, audio_duration: float) -> List[Utterance]:
    """
    Heuristic:
    - Read transcript PDF -> plain text
    - Split into sentences
    - Assign start/end times by spreading sentences evenly over the audio duration

    This is a simplification because we don't have precise timestamps, but it
    still lets us demonstrate multimodal alignment and segmentation.
    """
    raw_text = extract_text_from_pdf(pdf_path)

    # Basic cleaning – you can tweak this if MIT transcript has headers/footers you want to cut
    raw_text = raw_text.replace("\r", " ").strip()

    print("[transcript] Splitting into sentences...")
    sentences = sent_tokenize(raw_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    print(f"[transcript] Found {len(sentences)} sentences.")

    if not sentences:
        raise ValueError("No sentences found in transcript. Check PDF format.")

    # Distribute uniformly across duration
    duration_per_sent = audio_duration / len(sentences)

    utterances: List[Utterance] = []
    for i, sent in enumerate(sentences):
        start = i * duration_per_sent
        end = (i + 1) * duration_per_sent
        utterances.append(Utterance(
            idx=i,
            start=start,
            end=end,
            text=sent,
            slide_id=-1,  # fill later
        ))

    # Save for inspection
    utt_json_path = os.path.join(OUTPUT_DIR, f"{LECTURE_ID}_utterances.json")
    with open(utt_json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(u) for u in utterances], f, indent=2, ensure_ascii=False)
    print(f"[transcript] Saved {len(utterances)} utterances with approx timestamps to {utt_json_path}")

    return utterances


# ========== STEP 4: SIMPLE SLIDE-AWARE SEGMENTATION ==========

def assign_utterances_to_slides(utterances: List[Utterance],
                                slide_infos: List[SlideInfo],
                                audio_duration: float) -> List[Utterance]:
    """
    Simple heuristic:
    - Assume slides are used sequentially over the lecture.
    - Map time to slide by splitting the duration evenly across #slides.
    - In later refinements, you can use prosody or actual slide-change metadata.
    """
    num_slides = len(slide_infos)
    seconds_per_slide = audio_duration / num_slides

    print(f"[segmentation] Assigning utterances to {num_slides} slides "
          f"({seconds_per_slide:.2f} seconds per slide).")

    for u in utterances:
        slide_id = int(u.start // seconds_per_slide)
        if slide_id >= num_slides:
            slide_id = num_slides - 1
        u.slide_id = slide_id

    # Save segmented data
    seg_json_path = os.path.join(OUTPUT_DIR, f"{LECTURE_ID}_utterances_with_slides.json")
    with open(seg_json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(u) for u in utterances], f, indent=2, ensure_ascii=False)
    print(f"[segmentation] Saved utterances with slide ids to {seg_json_path}")

    return utterances


# ========== STEP 5: MULTIMODAL SUMMARIZATION PER SLIDE ==========

def build_summarizer():
    """
    Uses a standard abstractive summarization model.
    You can swap the model name for something else if needed.
    """
    print("[summary] Loading summarization model (this may take a bit the first time)...")
    summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer


def summarize_per_slide(
    slide_infos: List[SlideInfo],
    utterances: List[Utterance],
    summarizer,
    use_slide_text: bool = True,
    use_speech_text: bool = True,
    tag: str = "full",
) -> List[SlideSummary]:
    """
    Generate slide-level summaries under a given modality setting.

    use_slide_text:
        If True, include OCR'd slide text.
    use_speech_text:
        If True, include transcript text assigned to that slide.
    tag:
        Suffix used in output filenames (e.g., 'speech_only', 'slides_only', 'full').
    """
    summaries: List[SlideSummary] = []

    num_slides = len(slide_infos)

    for slide_id in range(num_slides):
        slide = slide_infos[slide_id]

        # collect all speech assigned to this slide
        speech_texts = [u.text for u in utterances if u.slide_id == slide_id]
        speech_concat = " ".join(speech_texts)

        # if nothing, skip
        if (not slide.ocr_text.strip() and not speech_concat.strip()):
            continue

        # Build an input that clearly separates slide text and spoken text
        prompt_parts = []
        if use_slide_text and slide.ocr_text.strip():
            prompt_parts.append("SLIDE TEXT:\n" + slide.ocr_text.strip())
        if use_speech_text and speech_concat.strip():
            prompt_parts.append("\nINSTRUCTOR SPEECH:\n" + speech_concat.strip())

        if not prompt_parts:
            # both modalities disabled / empty for this slide
            continue

        prompt = "\n\n".join(prompt_parts)

        # Truncate if too long (BART has a max token limit)
        if len(prompt) > 4000:
            prompt = prompt[:4000]

        print(f"[summary-{tag}] Summarizing slide {slide_id} ...")
        try:
            out = summarizer(
                prompt,
                max_length=220,
                min_length=60,
                do_sample=False,
            )
            summary_text = out[0]["summary_text"].strip()
        except Exception as e:
            print(f"[summary-{tag}] Error summarizing slide {slide_id}: {e}")
            summary_text = ""

        summaries.append(SlideSummary(
            slide_id=slide_id,
            slide_text=slide.ocr_text,
            speech_text=speech_concat,
            summary=summary_text
        ))

    # Save JSON
    sum_json_path = os.path.join(OUTPUT_DIR, f"{LECTURE_ID}_slide_summaries_{tag}.json")
    with open(sum_json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2, ensure_ascii=False)
    print(f"[summary-{tag}] Saved slide-level summaries to {sum_json_path}")

    # Also save a nice text file
    txt_path = os.path.join(OUTPUT_DIR, f"{LECTURE_ID}_slide_summaries_{tag}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for s in summaries:
            f.write(f"===== SLIDE {s.slide_id} =====\n")
            f.write("SUMMARY:\n")
            f.write(s.summary + "\n\n")
    print(f"[summary-{tag}] Saved human-readable summaries to {txt_path}")

    return summaries


# ========== MAIN PIPELINE ==========

def main():
    ensure_dir(OUTPUT_DIR)

    # Step 1: slides → OCR
    slide_infos = process_slides(SLIDES_PDF)

    # Step 2: audio → duration
    audio_duration = get_audio_duration(AUDIO_FILE)

    # Step 3: transcript → utterances with approximate timestamps
    utterances = build_utterances_from_transcript(TRANSCRIPT_PDF, audio_duration)

    # Step 4: assign utterances to slides (simple slide-aware segmentation)
    utterances = assign_utterances_to_slides(utterances, slide_infos, audio_duration)

    # Step 5: multimodal summarization per slide (three modality settings)
    summarizer = build_summarizer()

    # a) Speech-only baseline
    summarize_per_slide(
        slide_infos,
        utterances,
        summarizer,
        use_slide_text=False,
        use_speech_text=True,
        tag="speech_only",
    )

    # b) Slides-only baseline
    summarize_per_slide(
        slide_infos,
        utterances,
        summarizer,
        use_slide_text=True,
        use_speech_text=False,
        tag="slides_only",
    )

    # c) Full multimodal (slides + speech)
    summarize_per_slide(
        slide_infos,
        utterances,
        summarizer,
        use_slide_text=True,
        use_speech_text=True,
        tag="full",
    )

    print("\nDone! Check the 'outputs/' folder for JSON + TXT results.")


if __name__ == "__main__":
    main()
