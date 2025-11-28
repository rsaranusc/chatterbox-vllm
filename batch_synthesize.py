#!/usr/bin/env python3

import os
import sys
import time
import re
import argparse
from pathlib import Path
from typing import List

import torch
import torchaudio as ta

from chatterbox_vllm.tts import ChatterboxTTS

MAX_CHUNK_SIZE = 400  # characters

# Process in batches of X chunks at a time. Tune this based on your GPU memory.
#   15 seems to work for 8GB VRAM
#   40 seems to work for 16GB VRAM
#   80 seems to work for 24GB VRAM
# You may need to adjust the batch size based on your GPU memory.
BATCH_SIZE = 15

AUDIO_PROMPT = "docs/en-Rob_man.mp3"
INPUT_DIR = "input"
OUTPUT_DIR = "output"


def split_text_by_sentence(text: str) -> List[str]:
    """
    Given a line of text, split it into chunks of at most MAX_CHUNK_SIZE characters
    at sentence boundaries, forming roughly equal-sized chunks
    """
    sentences = text.split(". ")
    n_chunks_needed = len(text) // MAX_CHUNK_SIZE + 1
    approx_chunk_size = len(text) // n_chunks_needed

    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        # Remove excess whitespace like consecutive spaces, newlines, etc.
        sentence = " ".join(sentence.split())
        sentence = sentence.strip()

        if current_length + len(sentence) > approx_chunk_size:
            chunks.append(". ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)

    if current_chunk:
        chunks.append(". ".join(current_chunk))

    # If chunks end with a-z0-9, add a period to the end
    chunks = [chunk + "." if re.match(r"[a-zA-Z0-9]", chunk[-1]) else chunk for chunk in chunks if len(chunk) > 0]

    return chunks


def process_text_content(text: str) -> List[str]:
    """
    Process text content: remove comments, split by newlines, and chunk by sentences
    """
    # Remove lines starting with # (comments)
    text = "\n".join([line for line in text.split("\n") if not line.startswith("#")])

    # Chunk text by newlines
    text_lines = [i.strip() for i in text.split("\n") if len(i.strip()) > 0]

    # Split text into chunks
    chunked_text = [split_text_by_sentence(line) for line in text_lines]

    # Flatten list
    return [item for sublist in chunked_text for item in sublist]


def synthesize_file(input_file: Path, output_file: Path, model: ChatterboxTTS, audio_prompt_path: str):
    """
    Synthesize audio for a single text/markdown file
    """
    print(f"[INFO] Processing: {input_file}")

    # Read and process text content
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Process text content
    text_chunks = process_text_content(text)

    if not text_chunks:
        print(f"[WARNING] No valid text content found in {input_file}, skipping...")
        return

    print(f"[INFO] Text chunked into {len(text_chunks)} chunks")

    # Generate audio
    audios = model.generate(
        text_chunks,
        audio_prompt_path=audio_prompt_path,
        exaggeration=0.5,
        min_p=0.1,
        top_p=0.8,
    )

    # Stitch audio chunks together
    full_audio = torch.cat(audios, dim=-1)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save audio file
    ta.save(str(output_file), full_audio, model.sr)
    print(f"[INFO] Audio saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Batch synthesize audio from text and markdown files")
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Input directory containing text/markdown files (default: {INPUT_DIR})")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for audio files (default: {OUTPUT_DIR})")
    parser.add_argument("--audio-prompt", default=AUDIO_PROMPT,
                        help="Path to audio prompt file (default: {AUDIO_PROMPT})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for processing (default: {BATCH_SIZE})")
    parser.add_argument("--max-chunk-size", type=int, default=MAX_CHUNK_SIZE,
                        help=f"Maximum chunk size in characters (default: {MAX_CHUNK_SIZE})")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    # Validate input directory
    if not input_path.exists() or not input_path.is_dir():
        print(f"[ERROR] Input directory '{input_path}' does not exist or is not a directory")
        sys.exit(1)

    # Find all text and markdown files
    text_files = list(input_path.glob("*.txt")) + list(input_path.glob("*.md"))
    if not text_files:
        print(f"[WARNING] No .txt or .md files found in {input_path}")
        sys.exit(0)

    print(f"[INFO] Found {len(text_files)} text/markdown files to process")

    # Initialize TTS model
    print("[INFO] Loading TTS model...")
    start_time = time.time()
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=args.batch_size,
        max_model_len=args.max_chunk_size * 3,  # Rough heuristic
    )
    model_load_time = time.time()
    print(f"[INFO] Model loaded in {model_load_time - start_time:.2f} seconds")

    # Process each file
    total_start_time = time.time()
    processed_count = 0

    for text_file in text_files:
        try:
            # Create output filename with .mp3 extension
            output_file = output_path / f"{text_file.stem}.mp3"
            synthesize_file(text_file, output_file, model, args.audio_prompt)
            processed_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {text_file}: {str(e)}")
            continue

    # Cleanup
    model.shutdown()

    total_time = time.time() - total_start_time
    print(f"\n[SUCCESS] Processed {processed_count} files in {total_time:.2f} seconds")
    print(f"[INFO] Audio files saved to: {output_path}")


if __name__ == "__main__":
    main()