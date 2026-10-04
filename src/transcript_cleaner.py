import json
from pathlib import Path

INPUT_DIR = Path("data/transcripts")
OUTPUT_DIR = Path("data/cleaned_transcripts")


def clean_text(text):
    return " ".join(text.split())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in INPUT_DIR.glob("*.json"):

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

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