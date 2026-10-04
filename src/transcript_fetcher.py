import csv
import json
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi


VIDEO_LIST = Path("data/videos/videos.csv")
OUTPUT_DIR = Path("data/transcripts")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api = YouTubeTranscriptApi()

    with open(VIDEO_LIST, newline="", encoding="utf-8") as file:
        videos = csv.DictReader(file)

        for video in videos:

            if video["enabled"].lower() != "true":
                continue

            video_id = video["video_id"]
            title = video["title"]

            print(f"Fetching: {title}")

            try:
                transcript = api.fetch(video_id)

                segments = [
                    {
                        "text": item.text,
                        "start": item.start,
                        "duration": item.duration,
                    }
                    for item in transcript
                ]

                data = {
                    "video_id": video_id,
                    "title": title,
                    "url": video["url"],
                    "source": video["source"],
                    "segments": segments,
                }

                output_file = OUTPUT_DIR / f"{video_id}.json"

                with open(output_file, "w", encoding="utf-8") as outfile:
                    json.dump(data, outfile, indent=2, ensure_ascii=False)

                print(f"Saved: {output_file}")

            except Exception as error:
                print(f"FAILED: {title}")
                print(error)


if __name__ == "__main__":
    main()