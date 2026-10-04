import json
import re
from pathlib import Path

INPUT_DIR = Path("data/cleaned_transcripts")
OUTPUT_DIR = Path("data/chunks")

TARGET_WORDS = 250
MAX_WORDS = 320
OVERLAP_WORDS = 50


def word_count(text):
    return len(text.split())


def ends_sentence(text):
    return bool(re.search(r'[.!?]["\']?$', text.strip()))


def chunk_transcript(data):
    segments = data["segments"]
    chunks = []

    start_idx = 0
    chunk_id = 0

    while start_idx < len(segments):
        selected = []
        total_words = 0
        end_idx = start_idx

        while end_idx < len(segments):
            segment = segments[end_idx]
            selected.append(segment)
            total_words += word_count(segment["text"])
            end_idx += 1

            if total_words >= TARGET_WORDS:
                if ends_sentence(segment["text"]):
                    break

                if total_words >= MAX_WORDS:
                    break

        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(s["text"] for s in selected),
            "start": selected[0]["start"],
            "end": selected[-1]["start"] + selected[-1]["duration"],
            "word_count": total_words
        })

        chunk_id += 1

        if end_idx >= len(segments):
            break

        # create ~50 word overlap
        overlap_words = 0
        next_start = end_idx

        for i in range(end_idx - 1, start_idx - 1, -1):
            overlap_words += word_count(segments[i]["text"])
            next_start = i

            if overlap_words >= OVERLAP_WORDS:
                break

        start_idx = max(next_start, start_idx + 1)

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in INPUT_DIR.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = chunk_transcript(data)

        output = {
            "video_id": data["video_id"],
            "title": data["title"],
            "url": data["url"],
            "source": data["source"],
            "chunks": chunks
        }

        output_file = OUTPUT_DIR / file_path.name

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"{data['title']}: {len(chunks)} chunks")


if __name__ == "__main__":
    main()