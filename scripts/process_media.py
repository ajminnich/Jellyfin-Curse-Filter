from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from curse_filter.pipeline import ProcessingOptions, process_media  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a word-timestamped bleep audio track")
    parser.add_argument("media", type=Path)
    parser.add_argument("captions", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path(r"C:\Program Files\Jellyfin\Server\ffmpeg.exe"),
    )
    parser.add_argument("--model", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--transcription-model", default="small.en")
    parser.add_argument("--mode", default="auto", choices=("auto", "captions", "audio"))
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--dry-run", action="store_true", help="Inventory matching caption cues only")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.output or args.media.with_name(f"{args.media.stem}.default.filtered.eng.mka")
    options = ProcessingOptions(
        ffmpeg_path=args.ffmpeg,
        profanity_path=PROJECT_ROOT / "config" / "profanity.en.txt",
        model_dir=PROJECT_ROOT / ".models",
        report_dir=PROJECT_ROOT / "work" / "reports",
        model_name=args.model,
        transcription_model=args.transcription_model,
        analysis_mode=args.mode,
        device=args.device,
    )
    if args.dry_run:
        from curse_filter.pipeline import candidate_cues, load_profanity, parse_srt

        if args.captions is None:
            raise SystemExit("--dry-run requires a caption file")
        profanity = load_profanity(options.profanity_path)
        candidates = candidate_cues(parse_srt(args.captions), profanity)
        print(
            json.dumps(
                {
                    "media": str(args.media),
                    "captions": str(args.captions),
                    "candidate_cue_count": len(candidates),
                    "candidate_cues": [
                        {"index": cue.index, "start": cue.start, "end": cue.end, "text": cue.text}
                        for cue in candidates
                    ],
                },
                indent=2,
            )
        )
        return 0
    report = process_media(args.media, args.captions, output, options)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
