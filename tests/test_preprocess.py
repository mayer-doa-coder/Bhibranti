"""Tests for src/preprocess.py - one test per rule, each with a plain-language name.

Run from the repository root:

    python -m pytest tests/test_preprocess.py -v

Small made-up records, so these run in a second. The checks that need the real corpus
live in `python src/preprocess.py --check`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import (  # noqa: E402  (the import has to come after the path is set)
    FORMATS,
    MAX_TOKENS,
    SEPARATOR,
    ModelInput,
    as_text,
    as_tokens,
    count_tokens,
    cut_to_tokens,
    format_record,
    record_parts,
)

PASSAGE = "বাংলাদেশের রাজধানী ঢাকা। ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত। শহরটি অনেক পুরনো।"
QUESTION = "বাংলাদেশের রাজধানী কোথায়?"
ANSWER = "ঢাকা"


def record(answer: str = ANSWER, passage: str = PASSAGE, question: str = QUESTION) -> dict:
    return {
        "candidate_answer": answer,
        "context": passage,
        "question": question,
        "label": 1,
        "condition": "has_context" if passage else "no_context",
    }


# -------------------------------------------------------------- truncation ---

def test_cutting_keeps_the_start_of_the_text():
    assert cut_to_tokens("এক দুই তিন চার", 2) == "এক দুই"


def test_cutting_to_zero_leaves_nothing():
    assert cut_to_tokens("এক দুই তিন", 0) == ""


def test_a_short_text_is_left_alone():
    assert cut_to_tokens("এক দুই", 50) == "এক দুই"


def test_the_cut_text_is_a_real_piece_of_the_original():
    """Rejoining tokens with spaces would NOT give back the original, so this matters."""
    text = "ডা. ফখরুল আমিন খান ছিলেন চিকিৎসক।"
    assert text.startswith(cut_to_tokens(text, 4))


def test_nothing_is_cut_when_everything_fits():
    parts = record_parts(record())
    assert parts.context == PASSAGE and parts.question == QUESTION and parts.answer == ANSWER


def test_only_the_passage_is_shortened_when_space_runs_out():
    parts = record_parts(record(), budget=12)
    assert parts.question == QUESTION, "the question must never be cut"
    assert parts.answer == ANSWER, "the answer must never be cut"
    assert len(parts.context) < len(PASSAGE), "the passage should have been cut"
    assert PASSAGE.startswith(parts.context)


def test_the_answer_survives_even_an_impossible_budget():
    """With a budget too small for the question and answer alone, the passage goes to nothing
    but the answer still reaches the model - otherwise it has nothing to judge."""
    parts = record_parts(record(), budget=1)
    assert parts.answer == ANSWER
    assert parts.question == QUESTION
    assert parts.context == ""


def test_a_longer_answer_leaves_less_room_for_the_passage():
    tight = record_parts(record(answer="ঢাকা " * 20), budget=40)
    roomy = record_parts(record(), budget=40)
    assert len(tight.context) < len(roomy.context)


# ----------------------------------------------------------------- formats ---

def test_f1_holds_only_the_question_and_answer():
    model_input = format_record(record(), "F1")
    assert model_input.text_b is None
    assert QUESTION in model_input.text_a and ANSWER in model_input.text_a
    assert "বুড়িগঙ্গা" not in model_input.text_a, "F1 must not carry the passage"


def test_f2_holds_everything_in_one_piece():
    model_input = format_record(record(), "F2")
    assert model_input.text_b is None
    for piece in (PASSAGE, QUESTION, ANSWER):
        assert piece in model_input.text_a


def test_f3_keeps_the_passage_and_the_question_apart():
    model_input = format_record(record(), "F3")
    assert model_input.text_a == PASSAGE
    assert model_input.text_b is not None
    assert QUESTION in model_input.text_b and ANSWER in model_input.text_b


def test_a_record_with_no_passage_always_comes_out_as_f1():
    for fmt in FORMATS:
        model_input = format_record(record(passage=""), fmt)
        assert model_input.text_b is None
        assert model_input.text_a == f"{QUESTION} {ANSWER}"


def test_an_unknown_format_is_an_error():
    try:
        format_record(record(), "F9")
    except ValueError:
        return
    raise AssertionError("an unknown format must raise ValueError")


def test_the_two_segments_can_be_joined_into_one_string():
    assert ModelInput("ক", "খ").flat() == f"ক {SEPARATOR} খ"
    assert ModelInput("ক").flat() == "ক"


def test_the_answer_is_in_every_format():
    """The rule the whole file exists for."""
    for fmt in FORMATS:
        assert ANSWER in as_text(record(), fmt)
        assert ANSWER in as_text(record(), fmt, budget=8)


# ------------------------------------------------------------------ tokens ---

def test_the_separator_stays_one_token():
    tokens = as_tokens(record(), "F2")
    assert SEPARATOR in tokens
    assert "<" not in tokens and "SEP" not in tokens


def test_the_token_list_ends_with_the_answer():
    assert as_tokens(record(), "F2")[-1] == ANSWER


def test_the_passage_comes_first_in_f2():
    """Order is passage, then question, then answer - with a separator before each new part.

    Note the answer word "ঢাকা" also appears inside the passage, so the LAST separator is
    what marks where the answer starts.
    """
    tokens = as_tokens(record(), "F2")
    assert tokens[0] == "বাংলাদেশের", "the passage comes first"
    assert tokens.count(SEPARATOR) == 2, "one separator before the question, one before the answer"
    assert tokens[-2] == SEPARATOR and tokens[-1] == ANSWER


def test_f1_tokens_leave_the_passage_out():
    assert "বুড়িগঙ্গা" not in as_tokens(record(), "F1")


def test_the_m10_variants_all_work():
    for variant in ("V0", "V1", "V2", "V3", "V4"):
        tokens = as_tokens(record(), "F2", variant=variant)
        assert tokens, variant


def test_counting_tokens_matches_the_tokenizer():
    assert count_tokens("এক দুই তিন") == 3


# -------------------------------------------------------------- written out ---

def test_writing_a_split_leaves_out_the_label_revealing_fields(tmp_path):
    from preprocess import dump_split

    path = dump_split("dev", "F3", tmp_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows, "the file should not be empty"
    assert set(rows[0]) == {"id", "pair_id", "text_a", "text_b", "label",
                            "condition", "difficulty", "subject"}
    for banned in ("hallucination_type", "type_source", "annotator_1", "adjudicated"):
        assert banned not in rows[0], f"{banned} gives the label away"


def test_the_default_budget_is_the_project_default():
    assert MAX_TOKENS == 256


# --------------------------------- as_text and as_tokens are different readers ---

def test_f2_and_f3_give_identical_tokens():
    """Not a defect - a flat token list has no segments, so F3 cannot mean anything here.

    It is asserted so the fact stays visible: anyone comparing F2 against F3 on a
    bag-of-words model is comparing a thing with itself.
    """
    record = {"question": "প্রশ্ন কী", "candidate_answer": "উত্তর",
              "context": "অনুচ্ছেদ এখানে", "condition": "has_context"}
    assert as_tokens(record, "F2") == as_tokens(record, "F3")


def test_as_text_does_distinguish_f2_from_f3():
    """The encoder-facing view must keep them apart, or the M9 sweep is meaningless."""
    record = {"question": "প্রশ্ন কী", "candidate_answer": "উত্তর",
              "context": "অনুচ্ছেদ এখানে", "condition": "has_context"}
    assert as_text(record, "F2") != as_text(record, "F3")
    assert format_record(record, "F3").text_b is not None, "F3 must keep two segments"
    assert format_record(record, "F2").text_b is None, "F2 is one segment"


def test_the_token_view_marks_every_part_boundary():
    """as_tokens separates all three parts; as_text does not. Deliberate, and documented in
    as_tokens' docstring - a recurrent model has no other way to find the answer."""
    record = {"question": "প্রশ্ন কী", "candidate_answer": "উত্তর",
              "context": "অনুচ্ছেদ এখানে", "condition": "has_context"}
    assert as_tokens(record, "F2").count(SEPARATOR) == 2
    assert as_text(record, "F2").count(SEPARATOR) == 0


def test_train_classical_refuses_the_encoder_only_format():
    """A run logged as F3 that is really F2 would be a false experimental result."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, "-X", "utf8", str(root / "src" / "train_classical.py"),
         "--model", "bow_nb", "--format", "F3"],
        capture_output=True, text=True, cwd=root)
    assert done.returncode != 0, "F3 must be rejected, not silently treated as F2"
    assert "F3" in done.stderr or "invalid choice" in done.stderr
