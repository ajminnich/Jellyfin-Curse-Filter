from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from curse_filter.pipeline import (  # noqa: E402
    Cue,
    _canonical_aac_layout_suffix,
    _ctc_viterbi_path,
    _filtered_audio_codec,
    _forced_alignment_words,
    _verify_cue_placement,
    candidate_cues,
    caption_tokens,
    load_profanity,
    merge_intervals,
    ProcessingOptions,
    transcription_intervals,
    CensorInterval,
)


def test_aac_side_surround_layout_is_normalized_without_downmixing() -> None:
    assert _canonical_aac_layout_suffix("5.1(side)") == (
        ",channelmap="
        "map=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:channel_layout=5.1"
    )
    assert _canonical_aac_layout_suffix("5.1") == ""
    assert _canonical_aac_layout_suffix("stereo") == ""


def test_multichannel_dolby_sources_use_eac3_for_hdmi_compatibility() -> None:
    assert _filtered_audio_codec("eac3", 6) == "eac3"
    assert _filtered_audio_codec("ac3", 6) == "eac3"
    assert _filtered_audio_codec("aac", 6) == "aac"
    assert _filtered_audio_codec("eac3", 2) == "aac"


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


def test_placement_check_retimes_a_bleep_with_bounded_audio_evidence() -> None:
    cue = Cue(1, 10.0, 12.0, "Oh, shit.")
    proposed = CensorInterval(9.5, 9.8, "shit", 1, "wav2vec2-ctc", 0.25)
    bounded = CensorInterval(10.3, 10.6, "shit", 1, "wav2vec2-ctc", 0.8)

    selected, checks = _verify_cue_placement(cue, [proposed], [bounded])

    assert selected == [CensorInterval(10.3, 10.6, "shit", 1, "wav2vec2-ctc-cue-bounded", 0.8)]
    assert checks[0]["status"] == "retimed"


def test_placement_check_trims_an_early_start_when_both_paths_agree_on_the_end() -> None:
    cue = Cue(1, 10.0, 12.0, "Damn it.")
    proposed = CensorInterval(9.5, 10.5, "damn", 1, "wav2vec2-ctc", 0.21)
    bounded = CensorInterval(10.26, 10.5, "damn", 1, "wav2vec2-ctc", 0.17)

    selected, checks = _verify_cue_placement(cue, [proposed], [bounded])

    assert selected[0].start == 10.26
    assert checks[0]["status"] == "retimed"


def test_placement_check_skips_a_weak_guess_and_flags_a_strong_unchecked_word() -> None:
    cue = Cue(1, 10.0, 12.0, "Damn, damn, shit.")
    first = CensorInterval(10.2, 10.5, "damn", 1, "wav2vec2-ctc", 0.6)
    repeat = CensorInterval(10.8, 11.1, "damn", 1, "wav2vec2-ctc", 0.02)
    strong = CensorInterval(11.5, 11.8, "shit", 1, "wav2vec2-ctc", 0.85)
    missing = CensorInterval(11.65, 11.89, "shit", 1, "caption-proportional-fallback")

    selected, checks = _verify_cue_placement(cue, [first, repeat, strong], [first, repeat, missing])

    assert selected == [first, strong]
    assert [check["status"] for check in checks] == ["confirmed", "skipped", "unconfirmed"]


def test_both_caption_alignments_use_one_model_call(monkeypatch) -> None:
    from types import SimpleNamespace

    import torch

    from curse_filter import pipeline

    frames = torch.full((20, 5), -10.0)
    frames[:, 0] = 0.0
    for frame, label in [(1, 1), (2, 2), (3, 3), (4, 4), (12, 1), (13, 2), (14, 3), (15, 4)]:
        frames[frame, label] = 10.0 if frame < 10 else 8.0

    class Processor:
        tokenizer = SimpleNamespace(
            get_vocab=lambda: {"D": 1, "A": 2, "M": 3, "N": 4},
            word_delimiter_token="|",
            pad_token_id=0,
        )

        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(input_values=torch.zeros(1, 32000))

    model_calls = []

    def model(input_values):
        model_calls.append(input_values)
        return SimpleNamespace(logits=frames.unsqueeze(0))

    monkeypatch.setattr(pipeline, "_read_mono_pcm16", lambda _path: ([0] * 32000, 16000))
    cue = Cue(1, 1.0, 1.8, "Damn")

    first, bounded = _forced_alignment_words(Processor(), model, Path("cue.wav"), cue, 0.0, "cpu")

    assert len(model_calls) == 1
    assert first[0]["start"] < cue.start
    assert bounded[0]["start"] >= cue.start
