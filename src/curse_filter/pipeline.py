from __future__ import annotations

import html
import json
import os
import re
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence


TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|[A-Za-z](?:\*+)[A-Za-z]*")
TAG_RE = re.compile(r"<[^>]+>|\{\\[^}]+\}")
SRT_BLOCK_RE = re.compile(r"\r?\n\s*\r?\n")
SRT_TIME_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


@dataclass(frozen=True)
class CaptionToken:
    text: str
    normalized: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class Cue:
    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class CensorInterval:
    start: float
    end: float
    word: str
    cue_index: int
    alignment: str
    probability: float | None = None


@dataclass
class ProcessingOptions:
    ffmpeg_path: Path
    profanity_path: Path
    model_dir: Path
    report_dir: Path
    model_name: str = "facebook/wav2vec2-base-960h"
    transcription_model: str = "small.en"
    analysis_mode: str = "auto"
    device: str = "cuda"
    language: str = "en"
    cue_padding: float = 0.75
    censor_padding: float = 0.04
    minimum_censor_duration: float = 0.16
    beep_frequency: int = 1000
    beep_volume: float = 0.20
    audio_bitrate: str = "192k"


def _timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _clean_caption_text(value: str) -> str:
    value = TAG_RE.sub("", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def parse_srt(path: Path) -> list[Cue]:
    content = path.read_text(encoding="utf-8-sig", errors="replace")
    cues: list[Cue] = []
    for fallback_index, block in enumerate(SRT_BLOCK_RE.split(content.strip()), start=1):
        lines = [line.strip("\ufeff") for line in block.splitlines()]
        if not lines:
            continue
        time_line_index = next((i for i, line in enumerate(lines) if "-->" in line), -1)
        if time_line_index < 0:
            continue
        match = SRT_TIME_RE.search(lines[time_line_index])
        if not match:
            continue
        try:
            cue_index = int(lines[0]) if time_line_index > 0 else fallback_index
        except ValueError:
            cue_index = fallback_index
        text = _clean_caption_text(" ".join(lines[time_line_index + 1 :]))
        cues.append(
            Cue(
                index=cue_index,
                start=_timestamp_seconds(match.group("start")),
                end=_timestamp_seconds(match.group("end")),
                text=text,
            )
        )
    return cues


def normalize_word(value: str) -> str:
    value = value.lower().replace("\u2019", "'")
    value = value.replace("0", "o").replace("1", "i").replace("3", "e").replace("$", "s")
    return re.sub(r"[^a-z]", "", value)


def caption_tokens(text: str) -> list[CaptionToken]:
    return [
        CaptionToken(
            text=match.group(0),
            normalized=normalize_word(match.group(0)),
            start_offset=match.start(),
            end_offset=match.end(),
        )
        for match in TOKEN_RE.finditer(text)
    ]


def load_profanity(path: Path) -> set[str]:
    terms: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.partition("#")[0].strip()
        if line:
            terms.add(normalize_word(line))
    return terms


def candidate_cues(cues: Sequence[Cue], profanity: set[str]) -> list[Cue]:
    return [cue for cue in cues if any(t.normalized in profanity for t in caption_tokens(cue.text))]


def _padded_interval(
    start: float,
    end: float,
    word: str,
    cue_index: int,
    alignment: str,
    probability: float | None,
    options: ProcessingOptions,
) -> CensorInterval:
    midpoint = (start + end) / 2
    duration = max(options.minimum_censor_duration, end - start)
    return CensorInterval(
        start=round(max(0.0, midpoint - duration / 2 - options.censor_padding), 3),
        end=round(midpoint + duration / 2 + options.censor_padding, 3),
        word=word,
        cue_index=cue_index,
        alignment=alignment,
        probability=round(probability, 4) if probability is not None else None,
    )


def _run(command: Sequence[str | Path]) -> None:
    result = subprocess.run(
        [str(part) for part in command],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {detail}")


def _extract_cue_audio(
    ffmpeg_path: Path,
    media_path: Path,
    cue: Cue,
    cue_padding: float,
    output_path: Path,
) -> float:
    clip_start = max(0.0, cue.start - cue_padding)
    clip_end = cue.end + cue_padding
    _run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{clip_start:.3f}",
            "-i",
            media_path,
            "-t",
            f"{clip_end - clip_start:.3f}",
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
    )
    return clip_start


def _load_alignment_model(options: ProcessingOptions) -> tuple[Any, Any]:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA analysis was requested, but PyTorch cannot access the NVIDIA GPU")
    options.model_dir.mkdir(parents=True, exist_ok=True)
    shared_arguments = {
        "cache_dir": str(options.model_dir),
        "local_files_only": True,
    }
    try:
        processor = Wav2Vec2Processor.from_pretrained(options.model_name, **shared_arguments)
        model = Wav2Vec2ForCTC.from_pretrained(options.model_name, **shared_arguments)
    except OSError:
        shared_arguments["local_files_only"] = False
        processor = Wav2Vec2Processor.from_pretrained(options.model_name, **shared_arguments)
        model = Wav2Vec2ForCTC.from_pretrained(options.model_name, **shared_arguments)
    model = model.to(options.device)
    model.eval()
    return processor, model


def _read_mono_pcm16(path: Path) -> tuple[Any, int]:
    import numpy as np

    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError(f"Expected mono 16-bit PCM WAV: {path}")
        sample_rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    return samples.astype("float32") / 32768.0, sample_rate


def _ctc_viterbi_path(log_probs: Any, target_ids: Sequence[int], blank_id: int) -> list[int]:
    """Return the best expanded-CTC state for every emission frame."""
    import torch

    frame_count = int(log_probs.shape[0])
    expanded: list[int] = [blank_id]
    for token_id in target_ids:
        expanded.extend((int(token_id), blank_id))
    state_count = len(expanded)
    if state_count > frame_count * 2 + 1:
        raise ValueError("Caption text is too long for the available audio frames")

    negative_infinity = torch.tensor(float("-inf"), dtype=log_probs.dtype)
    previous = torch.full((state_count,), negative_infinity, dtype=log_probs.dtype)
    previous[0] = log_probs[0, blank_id]
    if state_count > 1:
        previous[1] = log_probs[0, expanded[1]]
    backpointers = torch.full((frame_count, state_count), -1, dtype=torch.int16)

    for frame in range(1, frame_count):
        current = torch.full((state_count,), negative_infinity, dtype=log_probs.dtype)
        for state, token_id in enumerate(expanded):
            choices = [(previous[state], state)]
            if state > 0:
                choices.append((previous[state - 1], state - 1))
            if (
                state > 1
                and token_id != blank_id
                and token_id != expanded[state - 2]
            ):
                choices.append((previous[state - 2], state - 2))
            best_score, best_state = max(choices, key=lambda choice: float(choice[0]))
            current[state] = best_score + log_probs[frame, token_id]
            backpointers[frame, state] = best_state
        previous = current

    final_states = [state_count - 1]
    if state_count > 1:
        final_states.append(state_count - 2)
    state = max(final_states, key=lambda candidate: float(previous[candidate]))
    path = [state]
    for frame in range(frame_count - 1, 0, -1):
        state = int(backpointers[frame, state])
        if state < 0:
            raise RuntimeError("Unable to backtrace CTC alignment")
        path.append(state)
    path.reverse()
    return path


def _forced_alignment_words(
    processor: Any,
    model: Any,
    wav_path: Path,
    cue: Cue,
    device: str,
) -> list[dict[str, Any]]:
    import torch

    audio, sample_rate = _read_mono_pcm16(wav_path)
    words = [token for token in caption_tokens(cue.text) if token.normalized]
    if not words:
        return []
    vocabulary = processor.tokenizer.get_vocab()
    delimiter = processor.tokenizer.word_delimiter_token
    target_ids: list[int] = []
    target_word_indexes: list[int] = []
    usable_words: list[CaptionToken] = []
    for source_word in words:
        letters = source_word.normalized.upper()
        if not letters or any(letter not in vocabulary for letter in letters):
            continue
        if usable_words:
            target_ids.append(vocabulary[delimiter])
            target_word_indexes.append(-1)
        word_index = len(usable_words)
        usable_words.append(source_word)
        for letter in letters:
            target_ids.append(vocabulary[letter])
            target_word_indexes.append(word_index)
    if not target_ids:
        return []

    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    with torch.inference_mode():
        logits = model(input_values).logits[0]
    log_probs = logits.log_softmax(dim=-1).float().cpu()
    blank_id = int(processor.tokenizer.pad_token_id)
    state_path = _ctc_viterbi_path(log_probs, target_ids, blank_id)
    seconds_per_frame = (len(audio) / sample_rate) / int(log_probs.shape[0])

    aligned: list[dict[str, Any]] = []
    for word_index, source_word in enumerate(usable_words):
        token_positions = [
            token_position
            for token_position, mapped_word in enumerate(target_word_indexes)
            if mapped_word == word_index
        ]
        states = {2 * token_position + 1 for token_position in token_positions}
        frames = [frame for frame, state in enumerate(state_path) if state in states]
        if not frames:
            continue
        confidences = []
        for frame in frames:
            state = state_path[frame]
            token_position = (state - 1) // 2
            confidences.append(float(log_probs[frame, target_ids[token_position]].exp()))
        aligned.append(
            {
                "word": source_word.text,
                "start": frames[0] * seconds_per_frame,
                "end": (frames[-1] + 1) * seconds_per_frame,
                "probability": sum(confidences) / len(confidences),
            }
        )
    return aligned


def _proportional_interval(cue: Cue, token: CaptionToken) -> tuple[float, float]:
    text_length = max(1, len(cue.text))
    start = cue.start + cue.duration * token.start_offset / text_length
    end = cue.start + cue.duration * token.end_offset / text_length
    return start, end


def _match_expected_word(
    expected: CaptionToken,
    expected_position: float,
    words: Sequence[dict[str, Any]],
    used: set[int],
) -> tuple[int, dict[str, Any]] | None:
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for index, word in enumerate(words):
        if index in used:
            continue
        normalized = normalize_word(str(word.get("word", "")))
        if not normalized:
            continue
        similarity = SequenceMatcher(None, expected.normalized, normalized).ratio()
        if similarity < 0.72:
            continue
        temporal_distance = abs(float(word.get("start", 0.0)) - expected_position)
        score = similarity * 3.0 - min(2.0, temporal_distance)
        candidates.append((score, index, word))
    if not candidates:
        return None
    _, index, word = max(candidates, key=lambda candidate: candidate[0])
    return index, word


def align_cue(
    cue: Cue,
    clip_start: float,
    words: Sequence[dict[str, Any]],
    profanity: set[str],
    options: ProcessingOptions,
) -> list[CensorInterval]:
    tokens = caption_tokens(cue.text)
    profane_tokens = [token for token in tokens if token.normalized in profanity]
    used_words: set[int] = set()
    intervals: list[CensorInterval] = []
    clip_duration = cue.duration + 2 * options.cue_padding

    for token in profane_tokens:
        proportional_start, proportional_end = _proportional_interval(cue, token)
        expected_clip_position = proportional_start - clip_start
        match = _match_expected_word(token, expected_clip_position, words, used_words)
        if match:
            word_index, word = match
            used_words.add(word_index)
            raw_start = clip_start + max(0.0, float(word.get("start", 0.0)))
            raw_end = clip_start + min(clip_duration, float(word.get("end", 0.0)))
            alignment = "wav2vec2-ctc"
            probability = float(word["probability"]) if word.get("probability") is not None else None
        else:
            raw_start, raw_end = proportional_start, proportional_end
            alignment = "caption-proportional-fallback"
            probability = None

        intervals.append(
            _padded_interval(
                raw_start,
                raw_end,
                token.normalized,
                cue.index,
                alignment,
                probability,
                options,
            )
        )
    return intervals


def transcription_intervals(
    transcription: dict[str, Any],
    profanity: set[str],
    options: ProcessingOptions,
) -> list[CensorInterval]:
    """Convert Whisper word timestamps into exact whole-word censor intervals."""
    intervals: list[CensorInterval] = []
    for fallback_index, segment in enumerate(transcription.get("segments", []), start=1):
        cue_index = int(segment.get("id", fallback_index))
        for word in segment.get("words", []):
            normalized = normalize_word(str(word.get("word", "")))
            if normalized not in profanity:
                continue
            start = float(word.get("start", segment.get("start", 0.0)))
            end = float(word.get("end", segment.get("end", start)))
            probability_value = word.get("probability")
            probability = float(probability_value) if probability_value is not None else None
            intervals.append(
                _padded_interval(
                    start,
                    end,
                    normalized,
                    cue_index,
                    "whisper-audio",
                    probability,
                    options,
                )
            )
    return intervals


def _transcribe_audio(
    media_path: Path,
    profanity: set[str],
    options: ProcessingOptions,
) -> tuple[list[CensorInterval], dict[str, Any]]:
    import whisper

    options.model_dir.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(
        options.transcription_model,
        device=options.device,
        download_root=str(options.model_dir),
    )
    result = model.transcribe(
        str(media_path),
        language=options.language,
        task="transcribe",
        word_timestamps=True,
        fp16=options.device == "cuda",
        verbose=False,
    )
    return transcription_intervals(result, profanity, options), result


def merge_intervals(intervals: Iterable[CensorInterval], maximum_gap: float = 0.03) -> list[CensorInterval]:
    ordered = sorted(intervals, key=lambda interval: (interval.start, interval.end))
    if not ordered:
        return []
    merged = [ordered[0]]
    for interval in ordered[1:]:
        previous = merged[-1]
        if interval.start <= previous.end + maximum_gap:
            merged[-1] = CensorInterval(
                start=previous.start,
                end=max(previous.end, interval.end),
                word=f"{previous.word}+{interval.word}",
                cue_index=previous.cue_index,
                alignment=(
                    previous.alignment
                    if previous.alignment == interval.alignment
                    else f"{previous.alignment}+{interval.alignment}"
                ),
                probability=min(
                    value
                    for value in (previous.probability, interval.probability)
                    if value is not None
                )
                if previous.probability is not None or interval.probability is not None
                else None,
            )
        else:
            merged.append(interval)
    return merged


def _duration_seconds(ffprobe_path: Path, media_path: Path) -> float:
    result = subprocess.run(
        [
            str(ffprobe_path),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _timeline_expression(intervals: Sequence[CensorInterval]) -> str:
    return "+".join(f"between(t,{interval.start:.3f},{interval.end:.3f})" for interval in intervals) or "0"


def render_bleep_audio(
    media_path: Path,
    output_path: Path,
    intervals: Sequence[CensorInterval],
    options: ProcessingOptions,
) -> None:
    if not intervals:
        raise ValueError("Refusing to render a filtered track with no censor intervals")
    ffprobe_path = options.ffmpeg_path.with_name("ffprobe.exe")
    duration = _duration_seconds(ffprobe_path, media_path)
    commands: list[str] = []
    for interval in intervals:
        # Runtime audio commands take effect on the next decoded frame. Send
        # the mute/bleep transition slightly early so short words do not leak
        # before that frame boundary; the reported censor interval is unchanged.
        command_start = max(0.0, interval.start - 0.06)
        commands.extend(
            (
                f"{command_start:.3f} volume@clean volume 0",
                f"{command_start:.3f} volume@beep volume {options.beep_volume:.3f}",
                f"{interval.end:.3f} volume@clean volume 1",
                f"{interval.end:.3f} volume@beep volume 0",
            )
        )
    command_expression = ";".join(commands)
    filter_graph = (
        f"[0:a:0]volume@clean=1,asendcmd=c='{command_expression}'[clean];"
        "[1:a]volume@beep=0,"
        "pan=stereo|c0=c0|c1=c0[beep];"
        "[clean][beep]amix=inputs=2:duration=first:normalize=0[out]"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".ffilter",
            prefix="curse-filter-",
            dir=output_path.parent,
            delete=False,
        ) as filter_script:
            filter_script.write(filter_graph)
            filter_script_path = Path(filter_script.name)
        _run(
            [
                options.ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-i",
                media_path,
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={options.beep_frequency}:sample_rate=48000:duration={duration:.3f}",
                "-filter_complex_script",
                filter_script_path,
                "-map",
                "[out]",
                "-c:a",
                "aac",
                "-b:a",
                options.audio_bitrate,
                "-metadata:s:a:0",
                "language=eng",
                "-metadata:s:a:0",
                "title=Filtered English (Bleep)",
                "-disposition:a:0",
                "default",
                output_path,
            ]
        )
    finally:
        if filter_script_path is not None:
            filter_script_path.unlink(missing_ok=True)


def process_media(
    media_path: Path,
    caption_path: Path | None,
    output_path: Path,
    options: ProcessingOptions,
) -> Path:
    media_path = media_path.resolve()
    caption_path = caption_path.resolve() if caption_path is not None else None
    output_path = output_path.resolve()
    if options.analysis_mode not in {"auto", "captions", "audio"}:
        raise ValueError(f"Unsupported analysis mode: {options.analysis_mode}")
    if options.analysis_mode == "captions" and caption_path is None:
        raise ValueError("Caption analysis mode requires a caption file")
    ffmpeg_directory = str(options.ffmpeg_path.resolve().parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if ffmpeg_directory.lower() not in {entry.lower() for entry in path_entries}:
        os.environ["PATH"] = ffmpeg_directory + os.pathsep + os.environ.get("PATH", "")
    profanity = load_profanity(options.profanity_path)
    intervals: list[CensorInterval] = []
    candidate_count = 0
    transcription_word_count = 0
    options.report_dir.mkdir(parents=True, exist_ok=True)
    clip_root = options.report_dir.parent / "clips"
    clip_root.mkdir(parents=True, exist_ok=True)

    use_captions = options.analysis_mode != "audio" and caption_path is not None
    selected_mode = "captions" if use_captions else "audio"
    if use_captions:
        cues = parse_srt(caption_path)
        candidates = candidate_cues(cues, profanity)
        candidate_count = len(candidates)
        if candidates:
            processor, model = _load_alignment_model(options)
            with tempfile.TemporaryDirectory(prefix="alignment-", dir=clip_root) as temp_dir:
                for cue in candidates:
                    wav_path = Path(temp_dir) / f"cue-{cue.index}.wav"
                    clip_start = _extract_cue_audio(
                        options.ffmpeg_path,
                        media_path,
                        cue,
                        options.cue_padding,
                        wav_path,
                    )
                    try:
                        words = _forced_alignment_words(processor, model, wav_path, cue, options.device)
                    except ValueError:
                        # Dense or accessibility-heavy subtitle cues can contain more
                        # characters than the short audio window has CTC frames. Keep
                        # the rest of the item word-aligned and use the bounded
                        # caption-proportional fallback for only this cue.
                        words = []
                    intervals.extend(align_cue(cue, clip_start, words, profanity, options))
    else:
        intervals, transcription = _transcribe_audio(media_path, profanity, options)
        transcription_word_count = sum(
            len(segment.get("words", [])) for segment in transcription.get("segments", [])
        )

    merged = merge_intervals(intervals)
    report_path = options.report_dir / f"{media_path.stem}.censor.json"
    report = {
        "schema_version": 2,
        "media_path": str(media_path),
        "caption_path": str(caption_path) if caption_path is not None else None,
        "output_path": str(output_path),
        "analysis_mode": selected_mode,
        "settings": {
            **asdict(options),
            "ffmpeg_path": str(options.ffmpeg_path),
            "profanity_path": str(options.profanity_path),
            "model_dir": str(options.model_dir),
            "report_dir": str(options.report_dir),
        },
        "candidate_cue_count": candidate_count,
        "transcription_word_count": transcription_word_count,
        "interval_count": len(merged),
        "intervals": [asdict(interval) for interval in merged],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if merged:
        render_bleep_audio(media_path, output_path, merged, options)
    return report_path
