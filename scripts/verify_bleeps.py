from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def decode_mono(ffmpeg: Path, audio: Path, start: float, duration: float) -> np.ndarray:
    result = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(audio),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4")


def dominant_frequency(samples: np.ndarray, sample_rate: int = 16000) -> float:
    if len(samples) < 2:
        return 0.0
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(samples), 1 / sample_rate)
    return float(frequencies[int(np.argmax(spectrum))])


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that every reported censor interval contains a bleep")
    parser.add_argument("report", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path(r"C:\Program Files\Jellyfin\Server\ffmpeg.exe"),
    )
    parser.add_argument("--frequency", type=float, default=1000.0)
    parser.add_argument("--tolerance", type=float, default=35.0)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    failures: list[dict[str, float]] = []
    peaks: list[float] = []
    for interval in report["intervals"]:
        start = float(interval["start"])
        duration = float(interval["end"]) - start
        samples = decode_mono(args.ffmpeg, args.audio, start, duration)
        peak = dominant_frequency(samples)
        peaks.append(peak)
        if abs(peak - args.frequency) > args.tolerance:
            failures.append({"start": start, "end": float(interval["end"]), "peak": peak})

    summary = {
        "intervals": len(peaks),
        "verified": len(peaks) - len(failures),
        "mean_peak_hz": round(float(np.mean(peaks)), 2) if peaks else None,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
