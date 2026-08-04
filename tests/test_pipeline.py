from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from curse_filter.pipeline import (  # noqa: E402
    Cue,
    _ctc_viterbi_path,
    candidate_cues,
    caption_tokens,
    load_profanity,
    merge_intervals,
    ProcessingOptions,
    transcription_intervals,
    CensorInterval,
)


def test_whole_word_matching_does_not_match_innocent_longer_word() -> None:
    terms = {"ass"}
    cues = [
        Cue(1, 0, 1, "That was a classy pass."),
        Cue(2, 1, 2, "He called me an ass."),
    ]
    assert [cue.index for cue in candidate_cues(cues, terms)] == [2]


def test_caption_tokens_normalize_apostrophes() -> None:
    assert [token.normalized for token in caption_tokens("That\u2019s damn bad")] == ["thats", "damn", "bad"]


def test_merge_overlapping_intervals() -> None:
    merged = merge_intervals(
        [
            CensorInterval(1.0, 1.3, "first", 1, "whisper", 0.9),
            CensorInterval(1.31, 1.5, "second", 1, "whisper", 0.8),
        ]
    )
    assert len(merged) == 1
    assert merged[0].start == 1.0
    assert merged[0].end == 1.5


def test_ctc_viterbi_path_visits_each_target_token() -> None:
    import torch

    # Blank=0, A=1, B=2. Each frame strongly favors the intended CTC path.
    probabilities = torch.tensor(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.90, 0.05, 0.05],
            [0.05, 0.05, 0.90],
            [0.90, 0.05, 0.05],
        ]
    )
    path = _ctc_viterbi_path(probabilities.log(), [1, 2], blank_id=0)
    assert 1 in path  # expanded state for A
    assert 3 in path  # expanded state for B


def test_whisper_word_timestamps_create_caption_free_intervals(tmp_path: Path) -> None:
    options = ProcessingOptions(
        ffmpeg_path=tmp_path / "ffmpeg.exe",
        profanity_path=tmp_path / "profanity.txt",
        model_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
    )
    transcription = {
        "segments": [
            {
                "id": 7,
                "words": [
                    {"word": "Classy", "start": 1.0, "end": 1.4, "probability": 0.99},
                    {"word": "shit!", "start": 1.5, "end": 1.8, "probability": 0.91},
                ],
            }
        ]
    }

    intervals = transcription_intervals(transcription, {"ass", "shit"}, options)

    assert len(intervals) == 1
    assert intervals[0].word == "shit"
    assert intervals[0].start == 1.46
    assert intervals[0].end == 1.84
    assert intervals[0].cue_index == 7
    assert intervals[0].alignment == "whisper-audio"
    assert intervals[0].probability == 0.91
