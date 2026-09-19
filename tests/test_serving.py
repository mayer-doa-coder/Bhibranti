"""Tests for src/serving.py and src/app.py - the demo.

Run from the repository root:

    python -m pytest tests/test_serving.py -v

Most of these run without any trained model, because the things most worth protecting are
not the predictions - they are the rules around them: that a typed-in record never carries a
label, that the scoreboard on the page agrees with the experiment log, and that nothing in
the demo can reach the test split.

The few tests that need real models skip themselves if `python src/serving.py --build` has
not been run, so a fresh clone can still run the suite.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import serving  # noqa: E402

needs_models = pytest.mark.skipif(
    not serving.is_built(),
    reason="models not built; run `python src/serving.py --build`")


# ----------------------------------------------------- the record being judged ---

def test_a_typed_in_record_carries_no_label():
    """The single most important rule in this file.

    A record typed in by a person has no ground truth. If `label` were present - even as a
    placeholder 0 - any feature code that happened to read it would be reading an answer the
    person never gave, and the demo would look far cleverer than it is.
    """
    record = serving.make_record("প্রশ্ন?", "উত্তর", "অনুচ্ছেদ")
    assert "label" not in record
    assert "hallucination_type" not in record


def test_a_record_without_a_passage_is_marked_closed_book():
    assert serving.make_record("q", "a")["condition"] == "no_context"
    assert serving.make_record("q", "a", "p")["condition"] == "has_context"


def test_whitespace_is_trimmed_off_the_input():
    record = serving.make_record("  q  ", "  a  ", "  p  ")
    assert (record["question"], record["candidate_answer"], record["context"]) == ("q", "a", "p")


def test_a_passage_of_only_spaces_counts_as_no_passage():
    assert serving.make_record("q", "a", "    ")["condition"] == "no_context"


# ------------------------------------------------------------------ evidence ---

def test_evidence_reports_an_exact_match():
    record = serving.make_record("রাজধানী কোথায়?", "ঢাকা", "বাংলাদেশের রাজধানী ঢাকা।")
    rows = {r["key"]: r for r in serving.evidence(record)}
    assert rows["exact_in_passage"]["value"] == "yes"
    assert rows["token_overlap"]["value"] == "100%"


def test_evidence_reports_a_missing_answer():
    record = serving.make_record("রাজধানী কোথায়?", "লন্ডন", "বাংলাদেশের রাজধানী ঢাকা।")
    rows = {r["key"]: r for r in serving.evidence(record)}
    assert rows["exact_in_passage"]["value"] == "no"


def test_evidence_says_so_rather_than_printing_a_misleading_zero():
    """With no passage, "0% of the answer must change" would read as a perfect match."""
    rows = {r["key"]: r for r in serving.evidence(serving.make_record("q", "উত্তর"))}
    for key in ("exact_in_passage", "token_overlap", "edit_passage", "lm_passage"):
        assert rows[key]["missing"] is True
        assert rows[key]["value"] == "no passage"


def test_evidence_still_reports_what_it_can_without_a_passage():
    rows = {r["key"]: r for r in serving.evidence(serving.make_record("q", "উত্তর"))}
    assert rows["answer_len"]["missing"] is False


# --------------------------------------------------------------- the rules ---

def test_the_exact_rule_says_correct_when_the_answer_is_in_the_passage():
    record = serving.make_record("রাজধানী?", "ঢাকা", "বাংলাদেশের রাজধানী ঢাকা।")
    rules = {r["rule"]: r for r in serving.rule_verdicts(record)}
    assert rules["Exact string match"]["verdict"] == 1


def test_the_exact_rule_says_hallucinated_when_it_is_not():
    record = serving.make_record("রাজধানী?", "প্যারিস", "বাংলাদেশের রাজধানী ঢাকা।")
    rules = {r["rule"]: r for r in serving.rule_verdicts(record)}
    assert rules["Exact string match"]["verdict"] == 0


def test_the_rules_decline_to_answer_without_a_passage():
    """Both rules work by looking in the passage, so with no passage they have no opinion.
    Reporting 'hallucinated' would be scoring the record on a coin flip."""
    for rule in serving.rule_verdicts(serving.make_record("q", "a")):
        assert rule["verdict"] is None


# ------------------------------------------------------------- the scoreboard ---

def test_the_scoreboard_covers_every_model_and_is_ranked_by_hard():
    rows = serving.scoreboard()
    assert len(rows) == len(serving.SERVING_MODELS)
    hard = [r["hard"] for r in rows if r["hard"] is not None]
    assert hard == sorted(hard, reverse=True), "must be ranked by the hard-subset score"
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))


def test_the_scoreboard_averages_over_seeds_rather_than_taking_the_last_row():
    """Table 5 of the guide reports the MEAN over three seeds. If this page took only the
    most recent row it would print a different number than the write-up, and one of the two
    would look wrong."""
    rows = {r["model"]: r for r in serving.scoreboard()}
    multi = rows["skipgram_mean_xgb"]
    assert multi["seeds"] == 3
    assert abs(multi["overall"] - 0.503) < 0.002, multi["overall"]


def test_a_model_that_did_not_converge_is_flagged_on_the_scoreboard():
    rows = {r["model"]: r for r in serving.scoreboard()}
    assert rows["bow_svm"]["unreliable"] is True
    assert rows["tfidf_svm"]["unreliable"] is False


def test_every_model_has_a_plain_english_name_and_description():
    """The page is read by someone who has not seen the code."""
    for name in serving.SERVING_MODELS:
        assert serving.MODEL_TITLE.get(name), name
        assert serving.MODEL_SEES.get(name), name


# --------------------------------------------- the test split is out of bounds ---

def executable_source(path: Path) -> str:
    """The module's real code, with comments and docstrings removed.

    Parsing rather than reading line by line, because both of these files *discuss* the test
    split at length in their docstrings - saying they do not touch it. A plain text search
    finds those sentences and fails on the very comment that documents the rule.
    """
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body.pop(0)
    return ast.unparse(tree)          # comments are not preserved by the parser at all


def test_nothing_in_the_demo_reads_the_test_split():
    """The test split is opened exactly once, at the end of the project (PRD). A demo that
    quietly scored on it would burn that one use without anyone noticing."""
    for module in ("serving.py", "app.py"):
        code = executable_source(ROOT / "src" / module)
        assert "load_split('test')" not in code, module
        assert 'load_split("test")' not in code, module
        assert "test.jsonl" not in code, module


def test_that_guard_would_actually_catch_a_violation():
    """A guard nobody has seen fail is not a guard. This proves the check has teeth."""
    import ast
    import tempfile
    offending = 'x = load_split("test")\n'
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write('"""A docstring mentioning test.jsonl innocently."""\n' + offending)
        temporary = Path(fh.name)
    try:
        code = executable_source(temporary)
        assert "load_split('test')" in code or 'load_split("test")' in code
        assert "test.jsonl" not in code, "the docstring should have been stripped"
    finally:
        temporary.unlink()
        del ast


def test_the_demo_serves_the_settings_everything_was_measured_under():
    """If the demo served a different format or preprocessing, the scores shown next to each
    prediction would belong to a different model than the one that made it."""
    assert (serving.SERVING_SEED, serving.SERVING_FORMAT, serving.SERVING_VARIANT) == (42, "F2", "V1")


# ------------------------------------------------- with real models on disk ---

@needs_models
def test_every_model_answers_with_a_valid_verdict():
    ensemble = serving.Ensemble.load()
    record = serving.make_record("রাজধানী কোথায়?", "ঢাকা", "বাংলাদেশের রাজধানী ঢাকা।")
    predictions = ensemble.predict_one(record)
    assert len(predictions) == len(serving.SERVING_MODELS)
    for p in predictions:
        assert p.verdict in (0, 1), p.model
        assert p.label in ("correct", "hallucinated")
        assert p.milliseconds >= 0
        assert p.scale in ("probability", "margin", "gap")


@needs_models
def test_the_same_record_gives_the_same_answer_every_time():
    """A saved model is fixed. If the demo wandered between clicks, nothing shown on it
    could be trusted."""
    ensemble = serving.Ensemble.load()
    record = serving.make_record("রাজধানী কোথায়?", "ঢাকা", "বাংলাদেশের রাজধানী ঢাকা।")
    first = {p.model: p.verdict for p in ensemble.predict_one(record)}
    second = {p.model: p.verdict for p in ensemble.predict_one(record)}
    assert first == second


@needs_models
def test_a_probability_is_never_printed_for_a_model_that_has_none():
    """An SVM reports distance from its dividing line, not a probability. Printing '87%'
    for it would invent a certainty it never expressed."""
    ensemble = serving.Ensemble.load(["tfidf_svm", "tfidf_logreg"])
    record = serving.make_record("রাজধানী কোথায়?", "ঢাকা", "বাংলাদেশের রাজধানী ঢাকা।")
    by_model = {p.model: p for p in ensemble.predict_one(record)}
    assert by_model["tfidf_svm"].scale == "margin"
    assert "%" not in by_model["tfidf_svm"].confidence_text
    assert by_model["tfidf_logreg"].scale == "probability"
    assert "%" in by_model["tfidf_logreg"].confidence_text


@needs_models
def test_a_saved_model_predicts_what_the_training_code_predicts():
    """The demo and the results table must be the same model, not two that resemble
    each other. This re-fits one model the way `--all` does and checks the saved copy
    agrees on every one of the first 200 dev records."""
    from splits import load_split
    from train_classical import fit_model

    dev = load_split("dev")[:200]
    train = load_split("train")
    fresh = fit_model("tfidf_logreg", train, serving.SERVING_SEED,
                      serving.SERVING_FORMAT, serving.SERVING_VARIANT, None)
    saved = serving.Ensemble.load(["tfidf_logreg"]).models["tfidf_logreg"]
    assert fresh.predict(dev) == saved.predict(dev)


# ------------------------------------------------------------- the web pages ---

@needs_models
def test_the_pages_all_load():
    import app as webapp
    webapp.boot()
    client = webapp.app.test_client()
    for path in ("/", "/models", "/health"):
        assert client.get(path).status_code == 200, path


@needs_models
def test_an_empty_answer_is_refused():
    import app as webapp
    webapp.boot()
    client = webapp.app.test_client()
    assert client.post("/predict", data={"question": "q", "answer": ""}).status_code == 400


@needs_models
def test_a_prediction_page_shows_every_model_and_the_rules():
    import app as webapp
    webapp.boot()
    client = webapp.app.test_client()
    page = client.post("/predict", data={
        "question": "রাজধানী কোথায়?", "answer": "ঢাকা",
        "passage": "বাংলাদেশের রাজধানী ঢাকা।"}).get_data(as_text=True)
    for name in serving.MODEL_TITLE.values():
        assert name in page, name
    for rule in serving.rule_verdicts(serving.make_record("q", "ঢাকা", "ঢাকা শহর")):
        assert rule["short"] in page, rule["short"]


def test_every_rule_has_both_a_formal_name_and_a_plain_one():
    """The log and the guide call it 'Exact string match'; the page says 'Answer is in the
    passage'. Both belong with the data, so the page invents no wording of its own."""
    for with_passage in (True, False):
        record = serving.make_record("q", "a", "passage" if with_passage else "")
        for rule in serving.rule_verdicts(record):
            assert rule["rule"] and rule["short"] and rule["why"]
            assert rule["short"] != rule["rule"]


@needs_models
def test_every_model_appears_once_on_a_prediction_page():
    """One click must ask every model exactly once - not skip any, not double-count."""
    import app as webapp
    webapp.boot()
    client = webapp.app.test_client()
    page = client.post("/predict",
                       data={"answer": "ঢাকা", "question": "q", "passage": "ঢাকা শহর"}
                       ).get_data(as_text=True)
    for title in serving.MODEL_TITLE.values():
        assert page.count(title) == 1, title


@needs_models
def test_the_example_records_come_in_pairs_and_never_from_test():
    import app as webapp
    from splits import load_split
    examples = webapp.dev_examples()
    assert examples, "the page needs something to click"
    dev_ids = {r["pair_id"] for r in load_split("dev")}
    for e in examples:
        assert e["pair_id"] in dev_ids, "an example came from outside dev"
        assert e["correct_answer"] != e["wrong_answer"]


@needs_models
def test_the_example_records_are_the_same_every_time():
    """A teacher watching the demo twice should see the same page."""
    import app as webapp
    assert [e["pair_id"] for e in webapp.dev_examples()] == \
           [e["pair_id"] for e in webapp.dev_examples()]
