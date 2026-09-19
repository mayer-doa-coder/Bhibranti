"""The demo: type in a question and an answer, see what every model makes of it.

    python src/serving.py --build      # once, about six minutes
    python src/app.py                  # then open http://127.0.0.1:8000

WHAT THE PAGES ARE FOR
  /            Type a question, an answer and (optionally) a passage. Every one of the
               twelve models answers it, and the page shows the evidence behind the
               decision rather than only the decision.
  /models      The scoreboard: what each model scored on dev, and which rules it has to
               beat before it counts as having learned anything.
  /health      Is the server up and are the models loaded.

A NOTE ON WHAT THIS DEMO CAN HONESTLY SHOW
These are the classical models - counting words, averaging word vectors, comparing strings.
On the hard questions they score between 0.49 and 0.61, where 0.50 is a coin flip. The demo
does not hide that. Every prediction is shown next to the two no-learning rules on the same
record, because a model that agrees with a string matcher has not demonstrated anything.

The test split is never touched here. The demo trains on train, quotes dev scores, and the
example records it offers come from dev.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from serving import (
    BASELINES,
    Ensemble,
    SERVING_FORMAT,
    SERVING_SEED,
    SERVING_VARIANT,
    evidence,
    is_built,
    make_record,
    missing,
    rule_verdicts,
    scoreboard,
)
from splits import load_split

app = Flask(__name__,
            template_folder=str(ROOT / "web" / "templates"),
            static_folder=str(ROOT / "web" / "static"))

# Without this, Flask caches the compiled templates whenever debug mode is off, so editing a
# page in web/templates/ appears to do nothing until the server is restarted - which is
# confusing enough to waste a long time on. Re-reading a handful of small files per request
# costs nothing on a local demo.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

# Flask tells browsers to cache static files for 12 hours when debug mode is off. That means
# an edit to web/static/app.css appears to do nothing - the page keeps using the copy the
# browser already has, and no amount of restarting the server changes it. Zero here, because
# this is a local demo where seeing the current file matters more than saving a request.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# =============================================================================
# LOADING, ONCE
# =============================================================================

ENSEMBLE: Ensemble | None = None
EXAMPLES: list[dict] = []


def dev_examples(how_many: int = 8) -> list[dict]:
    """A handful of real records to click, so a demo does not depend on typing Bengali.

    Taken from DEV, never test. Each one is a pair, so the same question can be shown with
    the answer the corpus calls correct and the one it calls hallucinated - which is the
    clearest way to see what the task actually is.
    """
    dev = load_split("dev")
    pairs: dict[str, list[dict]] = {}
    for record in dev:
        pairs.setdefault(record["pair_id"], []).append(record)

    usable = []
    for pair_id, records in pairs.items():
        if len(records) != 2:
            continue
        correct = next((r for r in records if r["label"] == 1), None)
        wrong = next((r for r in records if r["label"] == 0), None)
        if not correct or not wrong:
            continue
        if len(wrong["candidate_answer"]) > 120 or len(correct["question"]) > 150:
            continue
        usable.append({
            "pair_id": pair_id,
            "question": correct["question"],
            "passage": correct["context"],
            "correct_answer": correct["candidate_answer"],
            "wrong_answer": wrong["candidate_answer"],
            "difficulty": wrong["difficulty"],
            "condition": wrong["condition"],
            "error_type": wrong.get("hallucination_type") or "unlabeled",
            "subject": wrong.get("subject", ""),
        })

    # A fixed seed, so the demo shows the same examples every time it is opened. A teacher
    # watching twice should see the same page.
    random.Random(SERVING_SEED).shuffle(usable)
    hard = [e for e in usable if e["difficulty"] == "hard"]
    easy = [e for e in usable if e["difficulty"] != "hard"]
    chosen = hard[: how_many - 2] + easy[:2]
    return chosen[:how_many]


def boot() -> None:
    """Load the trained models and the example records, once, at start-up."""
    global ENSEMBLE, EXAMPLES
    ENSEMBLE = Ensemble.load()
    EXAMPLES = dev_examples()


# =============================================================================
# PAGES
# =============================================================================

@app.route("/")
def index():
    # The example links below the form arrive as query parameters, so clicking one fills the
    # form in without needing any JavaScript - and the resulting URL can be pasted or
    # bookmarked, which is handy when demonstrating the same record twice.
    return render_template("index.html", examples=EXAMPLES, results=None,
                           model_count=len(ENSEMBLE.models) if ENSEMBLE else 0)


@app.route("/predict", methods=["POST"])
def predict():
    question = request.form.get("question", "").strip()
    answer = request.form.get("answer", "").strip()
    passage = request.form.get("passage", "").strip()

    if not answer:
        return render_template("index.html", examples=EXAMPLES, results=None,
                               model_count=len(ENSEMBLE.models),
                               error="An answer is required - that is the thing being judged."), 400

    record = make_record(question, answer, passage)
    predictions = ENSEMBLE.predict_one(record)
    saying_correct = sum(1 for p in predictions if p.verdict == 1)

    ranking = {row["model"]: row for row in scoreboard()}
    predictions.sort(key=lambda p: -(ranking.get(p.model, {}).get("hard") or 0))

    # When the plain string rule and most of the models disagree, the page says so outright.
    # Otherwise a visitor who types an obviously-correct answer, sees most models call it
    # hallucinated, and concludes the demo is broken - when what they are actually looking at
    # is the project's central finding: these models cannot check an answer against a
    # passage, so on the easy cases a two-line rule beats nearly all of them.
    rules = rule_verdicts(record)
    exact = next((r["verdict"] for r in rules if r["rule"] == "Exact string match"), None)
    majority = 1 if saying_correct * 2 > len(predictions) else 0
    divergent = exact is not None and exact != majority

    results = {
        "record": record,
        "predictions": predictions,
        "ranked": ranking,
        "evidence": evidence(record),
        "rules": rules,
        "divergent": divergent,
        "rule_verdict": exact,
        "majority": majority,
        "saying_correct": saying_correct,
        "total": len(predictions),
        "total_ms": sum(p.milliseconds for p in predictions),
        "best_model": next(iter(sorted(
            predictions, key=lambda p: -(ranking.get(p.model, {}).get("hard") or 0))), None),
    }
    return render_template("index.html", examples=EXAMPLES, results=results,
                           model_count=len(ENSEMBLE.models),
                           form={"question": question, "answer": answer, "passage": passage})


@app.route("/models")
def models():
    return render_template("models.html", rows=scoreboard(), baselines=BASELINES,
                           seed=SERVING_SEED, fmt=SERVING_FORMAT, variant=SERVING_VARIANT)


@app.route("/health")
def health():
    ready = ENSEMBLE is not None and len(ENSEMBLE.models) > 0
    return ({"status": "ok" if ready else "no models loaded",
             "models": len(ENSEMBLE.models) if ENSEMBLE else 0}, 200 if ready else 503)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run the BangHallu demo server.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not is_built():
        print("  The models have not been built yet. Missing:")
        for name in missing():
            print(f"    - {name}")
        print("\n  Run this first (about six minutes, CPU only):")
        print("    python src/serving.py --build")
        return 1

    boot()
    print(f"  {len(ENSEMBLE.models)} models loaded, {len(EXAMPLES)} dev examples ready")
    print(f"  open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
