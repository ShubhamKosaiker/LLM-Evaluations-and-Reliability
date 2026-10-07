import json
from pathlib import Path

INPUT_DIR = Path("data/transcripts")
OUTPUT_DIR = Path("data/cleaned_transcripts")
CORRECTIONS_FILE = Path("data/eval/transcript_corrections.json")


def clean_text(text):
    return " ".join(text.split())


def apply_corrections(data, rules):
    entries = rules.get(data["video_id"], {}).get("corrections", [])

    for entry in entries:
        matches = [
            segment for segment in data["segments"]
            if abs(segment["start"] - entry["start"]) < 0.001
        ]

        if len(matches) != 1:
            raise ValueError(f"Correction timestamp mismatch: {entry}")

        segment = matches[0]

        if segment["text"].count(entry["old"]) != 1:
            raise ValueError(f"Correction text mismatch: {entry}")

        segment["text"] = segment["text"].replace(
            entry["old"], entry["new"], 1
        )

    return len(entries)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CORRECTIONS_FILE, encoding="utf-8") as f:
        rules = json.load(f)

    for file_path in INPUT_DIR.glob("*.json"):
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        applied = apply_corrections(data, rules)
        print(f"{data['video_id']}: {applied} corrections applied")

        cleaned_segments = []

        for segment in data["segments"]:
            text = clean_text(segment["text"])

            if not text:
                continue

            cleaned_segments.append({
                "text": text,
                "start": segment["start"],
                "duration": segment["duration"]
            })

        data["segments"] = cleaned_segments
        output_file = OUTPUT_DIR / file_path.name

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Cleaned: {output_file}")


if __name__ == "__main__":
    main()