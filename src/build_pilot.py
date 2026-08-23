"""Build the 500-pair Banglish pilot for the M1 shortcut-audit gate (PRD sec 5.5 V2, guide sec 3.3/9).

Two jobs, kept separate on purpose so each can be checked independently:

  1. SELECTION -- pick 500 pairs (300 has-context, 200 no-context) from the
     cleaned Bengali pool, filtering out the copy-from-context shortcut
     (guide sec 3.3 / DATASET_AUDIT.md Finding 4) and preferring wrong answers
     that reuse the passage's own words but get the fact wrong.
  2. DRAFT TRANSLITERATION -- turn the selected Bengali text into a Banglish
     rough draft (dictionary for common words, a deterministic grapheme
     fallback for the rest, seeded spelling variation).

The output is a DRAFT, not a finished corpus. Every record is marked
provenance="transliterated" and needs_human_review=true. Automated
transliteration cannot produce natural-sounding Banglish on its own -- that
is exactly why every record carries its original Bengali text alongside the
draft (bn_context/bn_question/bn_candidate_answer) plus a confidence flag,
and why a full human review pass is the very next step (pilot_v1/README.md).

What this script guarantees, and validates before writing anything:
  - every Bengali character used in the corpus (74 code points, checked
    against data/interim/bn_pool.jsonl) maps to something -- nothing is
    silently dropped or left untransliterated
  - every record validates against configs/schema.json
  - every pair has exactly one label=1 and one label=0 member
  - no duplicate ids, no empty required fields
  - full traceability: every pilot record points back to its exact source
    line in data/raw/bn_qa_pool/

Run from the repository root:

    python src/build_pilot.py
"""

from __future__ import annotations

import collections
import csv
import json
import random
import re
from pathlib import Path

INTERIM = Path("data/interim/bn_pool.jsonl")
OUT_DIR = Path("data/generated/pilot_v1")
SCHEMA_PATH = Path("configs/schema.json")

SEED = 42
N_HAS_CONTEXT = 300
N_NO_CONTEXT = 200
MAX_SUBJECT_SHARE = 0.30

FILLBLANK_RE = re.compile(r"শূন্যস্থান|_{3,}|\.{4,}")

# ---------------------------------------------------------------------------
# 1. SELECTION
# ---------------------------------------------------------------------------


def load_pairs() -> dict[str, list[dict]]:
    """Only genuinely complete pairs: exactly 2 members, one of each label.

    A same (context, question) group in the interim pool can have 2 members
    that are BOTH label=0 (two wrong answers recorded, no correct one) -- 135
    such groups exist. build_bn_pool.py already flags these via
    pair_complete=False; this must be checked explicitly, not inferred from
    len(v) == 2 alone, or those groups silently reach select_no_context /
    select_has_context and crash looking for a label==1 member that isn't there.
    """
    rows = [json.loads(l) for l in INTERIM.open(encoding="utf-8") if l.strip()]
    pairs: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        pairs[r["pair_id"]].append(r)
    return {
        k: v for k, v in pairs.items()
        if len(v) == 2 and len({m["label"] for m in v}) == 2
    }


def toks(s: str) -> set[str]:
    return set(s.split())


def select_has_context(pairs: dict[str, list[dict]]) -> list[tuple[list[dict], dict]]:
    """Score and filter has-context pairs. Returns (members, meta) tuples."""
    candidates = []
    for pid, members in pairs.items():
        if members[0]["condition"] != "has_context":
            continue
        correct = next(m for m in members if m["label"] == 1)
        wrong = next(m for m in members if m["label"] == 0)
        ctx, q = correct["context"], correct["question"]

        if FILLBLANK_RE.search(q):
            continue  # fill-in-the-blank: question repeats the passage verbatim

        ctoks, qtoks = toks(ctx), toks(q)
        qc_overlap = len(qtoks & ctoks) / max(len(qtoks), 1)
        if qc_overlap > 0.5:
            continue  # question is mostly a copy of the passage

        wa_stripped = wrong["candidate_answer"].strip().rstrip("|. ")
        if wa_stripped and wa_stripped in ctx:
            continue  # THE SHORTCUT: wrong answer copy-pasted from context

        wtoks = toks(wrong["candidate_answer"])
        wrong_overlap = len(wtoks & ctoks) / max(len(wtoks), 1)
        if wrong_overlap < 0.3:
            continue  # wrong answer doesn't reuse passage vocabulary at all

        candidates.append(
            (
                members,
                {
                    "wrong_overlap": wrong_overlap,
                    "qc_overlap": qc_overlap,
                    "subject": correct["subject"],
                },
            )
        )
    candidates.sort(key=lambda c: -c[1]["wrong_overlap"])
    return candidates


def select_no_context(pairs: dict[str, list[dict]], rng: random.Random) -> list[tuple[list[dict], dict]]:
    candidates = []
    for pid, members in pairs.items():
        if members[0]["condition"] != "no_context":
            continue
        correct = next(m for m in members if m["label"] == 1)
        if FILLBLANK_RE.search(correct["question"]):
            continue
        candidates.append((members, {"subject": correct["subject"]}))
    rng.shuffle(candidates)
    return candidates


def apply_subject_cap(candidates: list[tuple[list[dict], dict]], n: int, cap_share: float) -> list[list[dict]]:
    cap = max(1, int(n * cap_share))
    used: collections.Counter = collections.Counter()
    chosen: list[list[dict]] = []
    remainder: list[list[dict]] = []
    for members, meta in candidates:
        if len(chosen) >= n:
            break
        if used[meta["subject"]] >= cap:
            remainder.append(members)
            continue
        used[meta["subject"]] += 1
        chosen.append(members)
    if len(chosen) < n:
        chosen.extend(remainder[: n - len(chosen)])
    if len(chosen) < n:
        raise ValueError(f"only found {len(chosen)} candidates, need {n}")
    return chosen[:n]


# ---------------------------------------------------------------------------
# 2. DRAFT TRANSLITERATION
# ---------------------------------------------------------------------------

DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Common words, highest-frequency first (measured on this corpus) plus the
# domain proper nouns that appear constantly (BCS/history/geography subjects).
# Two spelling variants where natural variation is common; a repeated value
# where it isn't. Covering the head of the frequency distribution here means
# the fallback grapheme mapper only has to handle the long tail.
WORD_DICT: dict[str, list[str]] = {
    "টি": ["ti", "ta"], "কত": ["koto", "koto"], "এবং": ["ebong", "ebng"],
    "তার": ["tar", "tar"], "একটি": ["ekti", "ekta"], "কী": ["ki", "kii"],
    "এর": ["er", "er"], "করে": ["kore", "kore"], "তিনি": ["tini", "tini"],
    "কোন": ["kon", "kon"], "রয়েছে": ["royeche", "roeche"], "হয়": ["hoy", "hoi"],
    "আছে": ["ache", "asche"], "টাকা": ["taka", "taka"], "জন্য": ["jonno", "jnno"],
    "শব্দ": ["shobdo", "shobdo"], "মোট": ["mot", "mott"], "থেকে": ["theke", "theke"],
    "ও": ["o", "o"], "ছিল": ["chilo", "silo"], "কাছে": ["kache", "kase"],
    "কোনটি": ["konti", "konta"], "যদি": ["jodi", "jodi"], "শব্দের": ["shobder", "shobder"],
    "করেন": ["koren", "korn"], "প্রতিটি": ["protiti", "protita"], "সে": ["se", "she"],
    "করতে": ["korte", "korte"], "অনুযায়ী": ["onujayi", "anujayi"], "সালে": ["sale", "shale"],
    "করা": ["kora", "kora"], "হবে": ["hobe", "hbe"], "জন": ["jon", "jon"],
    "প্রতি": ["proti", "prti"], "কয়টি": ["koyti", "koita"], "প্রদত্ত": ["prodotto", "prodotto"],
    "তাহলে": ["tahole", "tahle"], "হয়েছে": ["hoyeche", "hoise"], "সংখ্যা": ["songkha", "songkhya"],
    "অর্থ": ["ortho", "orth"], "প্রথম": ["prothom", "prthom"], "চেয়ে": ["cheye", "cheye"],
    "এখন": ["ekhon", "akhn"], "বেশি": ["beshi", "beshi"], "তাদের": ["tader", "tader"],
    "তবে": ["tobe", "tbe"], "থাকে": ["thake", "thak"], "দাম": ["dam", "daam"],
    "কতগুলি": ["kotoguli", "kotogula"], "আরও": ["aro", "aro"], "মাইল": ["mile", "mile"],
    "মধ্যে": ["moddhe", "modhye"], "প্রসঙ্গ": ["prosongo", "prosngo"],
    "কি": ["ki", "kii"], "না": ["na", "naa"], "হ্যাঁ": ["han", "hyan"], "কে": ["ke", "ke"],
    "কারা": ["kara", "kara"], "কোথায়": ["kothay", "kothae"], "কেন": ["ken", "kno"],
    "কীভাবে": ["kivabe", "kibhabe"],
    "বাংলাদেশ": ["Bangladesh", "bangladesh"], "বাংলাদেশের": ["Bangladesher", "bangladesher"],
    "ভারত": ["India", "bharot"], "ভারতের": ["Indian", "bharoter"],
    "ঢাকা": ["Dhaka", "dhaka"], "রাজধানী": ["rajdhani", "rajdhani"],
    "নাম": ["naam", "nam"], "সাল": ["sal", "shal"], "সালের": ["saler", "shaler"],
    "বছর": ["bochor", "bosor"], "মাস": ["mash", "mas"], "দিন": ["din", "dine"],
    "তারিখ": ["tarikh", "tarikh"], "প্রধানমন্ত্রী": ["prodhanmontri", "prodhanmntri"],
    "রাষ্ট্রপতি": ["rashtropoti", "rastropoti"], "স্বাধীনতা": ["shadhinota", "shadinota"],
    "বিজয়": ["bijoy", "bijoy"], "দিবস": ["dibosh", "dibosh"], "জাতীয়": ["jatio", "jatiyo"],
    "মুক্তিযুদ্ধ": ["muktijuddho", "muktizuddho"], "সরকার": ["sorkar", "shorkar"],
    "মন্ত্রণালয়": ["montronaloy", "mntronaloy"], "সংবিধান": ["songbidhan", "shongbidhan"],
    "আইন": ["ain", "ain"], "নদী": ["nodi", "nadi"], "পাহাড়": ["pahar", "pahar"],
    "সমুদ্র": ["somudro", "shomudro"], "জেলা": ["jela", "jela"], "বিভাগ": ["bivag", "bivag"],
    "মূলধন": ["muldhon", "muldhon"], "গণিত": ["gonit", "ganit"], "ইতিহাস": ["itihash", "itihas"],
    "ভূগোল": ["bhugol", "bhugol"], "বিজ্ঞান": ["biggan", "biggyan"], "সাহিত্য": ["sahitto", "shahitto"],
    "ব্যাকরণ": ["bekoron", "byakoron"], "প্রশ্ন": ["proshno", "prshno"], "উত্তর": ["uttor", "uttar"],
    "সঠিক": ["sothik", "sothik"], "ভুল": ["bhul", "bul"],
}

# Full grapheme fallback for out-of-dictionary words. Covers every Bengali
# character observed in the corpus (74 code points, verified against
# data/interim/bn_pool.jsonl before writing this table) plus the danda marks,
# which sit outside the main Bengali Unicode block. Nothing here is guessed;
# every entry maps a character actually seen in the data.
INDEP_VOWELS = {
    "অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u",
    "ঋ": "ri", "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
}
VOWEL_SIGNS = {
    "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri",
    "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou", "ৗ": "u",
}
CONSONANTS = {
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "chh", "জ": "j", "ঝ": "jh", "ঞ": "n",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "f", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh", "স": "s", "হ": "h",
    "ড়": "r", "ঢ়": "rh", "য়": "y", "ৎ": "t",
    "ৰ": "r", "ৱ": "w",
}
NUKTA = "়"
HASANTA = "্"
ANUSVARA_MAP = {"ং": "ng"}
CHANDRABINDU = "ঁ"  # dropped: casual Banglish rarely marks nasalisation
VISARGA_MAP = {"ঃ": "h"}
CURRENCY_MAP = {"৳": "Tk"}
DANDA_MAP = {"।": ".", "॥": ".", "৷": "."}

_word_chars = (
    set(INDEP_VOWELS) | set(VOWEL_SIGNS) | set(CONSONANTS)
    | {NUKTA, HASANTA, CHANDRABINDU} | set(ANUSVARA_MAP) | set(VISARGA_MAP)
)
BENGALI_WORD_CHAR = re.compile("[" + "".join(re.escape(c) for c in _word_chars) + "]+")
DIGIT_RUN = re.compile("[০-৯]+")
LATIN_RUN = re.compile("[A-Za-z]+")


def grapheme_transliterate(word: str) -> str:
    """Deterministic, whole-alphabet-covering fallback for OOV Bengali words.

    The inherent vowel is always rendered as 'o' (no attempt to guess when
    real Bengali orthoepy drops it word-finally). That trade-off is
    deliberate: guessing wrong there produces unpredictable output, whereas a
    fixed rule is fully reproducible, and any resulting rough spots are
    exactly what the human review pass (Task 3) exists to catch.
    """
    out: list[str] = []
    i = 0
    n = len(word)
    pending_consonant: str | None = None

    def flush(vowel: str | None) -> None:
        nonlocal pending_consonant
        if pending_consonant is not None:
            out.append(pending_consonant + (vowel if vowel is not None else "o"))
            pending_consonant = None

    while i < n:
        ch = word[i]
        if ch in INDEP_VOWELS:
            flush(None)
            out.append(INDEP_VOWELS[ch])
        elif ch in CONSONANTS:
            flush(None)
            sound = CONSONANTS[ch]
            if i + 1 < n and word[i + 1] == NUKTA:
                combined = ch + NUKTA
                sound = CONSONANTS.get(combined, sound)
                i += 1
            pending_consonant = sound
        elif ch in VOWEL_SIGNS:
            flush(VOWEL_SIGNS[ch])
        elif ch == HASANTA:
            if pending_consonant is not None:
                out.append(pending_consonant)
                pending_consonant = None
        elif ch in ANUSVARA_MAP:
            flush(None)
            out.append(ANUSVARA_MAP[ch])
        elif ch in VISARGA_MAP:
            flush(None)
            out.append(VISARGA_MAP[ch])
        elif ch == CHANDRABINDU:
            pass
        elif ch == NUKTA:
            pass
        else:
            flush(None)
            out.append(ch)
        i += 1
    flush(None)
    return "".join(out)


def transliterate_text(text: str, rng: random.Random) -> tuple[str, dict]:
    """Returns (banglish_text, stats). stats tracks dict-hit vs fallback use
    for the confidence flag; the caller verifies no Bengali character survives.
    """
    stats = {"dict_hits": 0, "fallback_hits": 0}

    def repl_word(m: re.Match) -> str:
        w = m.group(0)
        if w in WORD_DICT:
            stats["dict_hits"] += 1
            variants = WORD_DICT[w]
            return variants[0] if len(set(variants)) == 1 else rng.choice(variants)
        stats["fallback_hits"] += 1
        return grapheme_transliterate(w)

    text = BENGALI_WORD_CHAR.sub(repl_word, text)
    text = DIGIT_RUN.sub(lambda m: m.group(0).translate(DIGIT_MAP), text)
    for bn, latin in {**CURRENCY_MAP, **DANDA_MAP}.items():
        text = text.replace(bn, latin)
    text = LATIN_RUN.sub(lambda m: m.group(0).lower(), text)
    return text, stats


def check_no_bengali_leftover(text: str) -> list[str]:
    """Any character that should have been transliterated but was not.
    Must be empty for every field in every record before output is written.
    """
    leftover = []
    for c in text:
        if "ঀ" <= c <= "৿" or c in ("।", "॥"):
            leftover.append(f"U+{ord(c):04X} {c!r}")
    return leftover


# ---------------------------------------------------------------------------
# 3. RECORD ASSEMBLY
# ---------------------------------------------------------------------------


def confidence(stats: dict) -> str:
    total = stats["dict_hits"] + stats["fallback_hits"]
    if total == 0:
        return "n/a"
    fallback_ratio = stats["fallback_hits"] / total
    if fallback_ratio == 0:
        return "high"
    if fallback_ratio < 0.3:
        return "medium"
    return "low"


def guess_hallucination_type(wrong_answer: str, correct_answer: str, has_context: bool) -> str:
    """A heuristic placeholder, not a real annotation. PRD D8 requires a human
    to assign hallucination_type; this only picks a defensible starting guess
    so the field isn't blank, and it is explicitly flagged for confirmation
    in the review sheet (see needs_type_confirmation).
    """
    if not has_context:
        return "fabricated"
    has_digit = any(c.isdigit() for c in wrong_answer) or any("০" <= c <= "৯" for c in wrong_answer)
    return "numeric" if has_digit else "relational"


def build_record(member: dict, pilot_pair_id: str, member_index: int, rng: random.Random) -> tuple[dict, list[str]]:
    """Returns (record, leftover_bengali_warnings)."""
    ctx_bl, ctx_stats = transliterate_text(member["context"], rng) if member["context"] else ("", {"dict_hits": 0, "fallback_hits": 0})
    q_bl, q_stats = transliterate_text(member["question"], rng)
    a_bl, a_stats = transliterate_text(member["candidate_answer"], rng)

    warnings = []
    for name, text in (("context", ctx_bl), ("question", q_bl), ("answer", a_bl)):
        leftover = check_no_bengali_leftover(text)
        if leftover:
            warnings.append(f"{pilot_pair_id}_{member_index} {name}: leftover {leftover}")

    combined_stats = {
        "dict_hits": ctx_stats["dict_hits"] + q_stats["dict_hits"] + a_stats["dict_hits"],
        "fallback_hits": ctx_stats["fallback_hits"] + q_stats["fallback_hits"] + a_stats["fallback_hits"],
    }

    record = {
        "id": f"{pilot_pair_id}_{member_index}",
        "pair_id": pilot_pair_id,
        "pair_complete": True,
        "source": "bn_qa_pool",
        "source_corpus": "bengali_wikipedia+bcs_question_banks",
        "source_file": member["source_file"],
        "source_line": member["source_line"],
        "source_original_id": member["id"],
        "source_original_pair_id": member["pair_id"],
        "subject": member["subject"],
        "level": member["level"],
        "condition": member["condition"],
        "context": ctx_bl,
        "question": q_bl,
        "reference_answer": "",  # filled in after both members of the pair are built
        "candidate_answer": a_bl,
        "label": member["label"],
        "hallucination_type": "none" if member["label"] == 1 else guess_hallucination_type(
            a_bl, "", member["condition"] == "has_context"
        ),
        "difficulty": "unlabeled",
        "generator_model": "rule_based_transliteration_v1",
        "generation_seed": SEED,
        "annotator_1": None,
        "annotator_2": None,
        "adjudicated": False,
        "provenance": "transliterated",
        "script_condition": "banglish",
        "error_span": "",
        # Extra fields beyond the frozen schema (additionalProperties: true)
        # kept specifically so nothing is lost: the reviewer always has the
        # original Bengali right next to the draft.
        "bn_context": member["context"],
        "bn_question": member["question"],
        "bn_candidate_answer": member["candidate_answer"],
        "transliteration_confidence": confidence(combined_stats),
        "needs_human_review": True,
        "needs_type_confirmation": True,
    }
    return record, warnings


# ---------------------------------------------------------------------------
# 4. VALIDATION
# ---------------------------------------------------------------------------


def load_schema_required_and_enums(schema_path: Path) -> tuple[list[str], dict[str, list]]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required = schema["required"]
    enums = {
        name: prop["enum"]
        for name, prop in schema["properties"].items()
        if "enum" in prop
    }
    return required, enums


def validate_records(records: list[dict], schema_path: Path) -> list[str]:
    problems: list[str] = []
    required, enums = load_schema_required_and_enums(schema_path)

    ids = collections.Counter(r["id"] for r in records)
    for id_, count in ids.items():
        if count > 1:
            problems.append(f"duplicate id: {id_} appears {count} times")

    pairs: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        pairs[r["pair_id"]].append(r)
    for pid, members in pairs.items():
        if len(members) != 2:
            problems.append(f"pair {pid} has {len(members)} members, expected 2")
            continue
        labels = sorted(m["label"] for m in members)
        if labels != [0, 1]:
            problems.append(f"pair {pid} labels are {labels}, expected [0, 1]")

    for r in records:
        for field in required:
            if field not in r or r[field] in (None, ""):
                if field in ("context",) and r.get("condition") == "no_context":
                    continue  # empty context is correct for no_context
                problems.append(f"{r['id']}: required field '{field}' missing or empty")
        for field, allowed in enums.items():
            if field in r and r[field] not in allowed:
                problems.append(f"{r['id']}: {field}={r[field]!r} not in {allowed}")
        if r["label"] == 1 and r["hallucination_type"] != "none":
            problems.append(f"{r['id']}: label=1 but hallucination_type != none")
        if r["label"] == 0 and r["hallucination_type"] == "none":
            problems.append(f"{r['id']}: label=0 but hallucination_type == none")
        if r["condition"] == "no_context" and r["context"] != "":
            problems.append(f"{r['id']}: no_context record has non-empty context")
        for field in ("context", "question", "candidate_answer"):
            leftover = check_no_bengali_leftover(r[field])
            if leftover:
                problems.append(f"{r['id']}: field '{field}' has untransliterated characters: {leftover}")
        # round-trip encoding check
        try:
            json.loads(json.dumps(r, ensure_ascii=False))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            problems.append(f"{r['id']}: JSON/UTF-8 round-trip failed: {e}")

    return problems


# ---------------------------------------------------------------------------
# 5. OUTPUT
# ---------------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review_sheet(path: Path, records: list[dict]) -> None:
    """One row per PAIR (not per record) so a reviewer sees the correct and
    hallucinated answer side by side -- the comparison that actually matters.
    """
    pairs: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        pairs[r["pair_id"]].append(r)

    fieldnames = [
        "pair_id", "condition", "subject", "confidence",
        "bn_context", "banglish_context",
        "bn_question", "banglish_question",
        "bn_correct_answer", "banglish_correct_answer",
        "bn_wrong_answer", "banglish_wrong_answer",
        "hallucination_type_guess",
        "source_original_pair_id",
        "reviewer_status", "reviewer_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for pid in sorted(pairs):
            members = pairs[pid]
            correct = next(m for m in members if m["label"] == 1)
            wrong = next(m for m in members if m["label"] == 0)
            writer.writerow({
                "pair_id": pid,
                "condition": correct["condition"],
                "subject": correct["subject"],
                "confidence": min(
                    correct["transliteration_confidence"], wrong["transliteration_confidence"],
                    key=lambda c: {"high": 0, "medium": 1, "low": 2, "n/a": 3}[c],
                ),
                "bn_context": correct["bn_context"],
                "banglish_context": correct["context"],
                "bn_question": correct["bn_question"],
                "banglish_question": correct["question"],
                "bn_correct_answer": correct["bn_candidate_answer"],
                "banglish_correct_answer": correct["candidate_answer"],
                "bn_wrong_answer": wrong["bn_candidate_answer"],
                "banglish_wrong_answer": wrong["candidate_answer"],
                "hallucination_type_guess": wrong["hallucination_type"],
                "source_original_pair_id": correct["source_original_pair_id"],
                "reviewer_status": "",
                "reviewer_notes": "",
            })


def write_selection_log(path: Path, has_ctx_meta: list[dict], no_ctx_meta: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["condition", "subject", "wrong_overlap", "qc_overlap"])
        for m in has_ctx_meta:
            writer.writerow(["has_context", m["subject"], f"{m['wrong_overlap']:.3f}", f"{m['qc_overlap']:.3f}"])
        for m in no_ctx_meta:
            writer.writerow(["no_context", m["subject"], "", ""])


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    rng = random.Random(SEED)
    pairs = load_pairs()

    hc_candidates = select_has_context(pairs)
    print(f"has-context candidates surviving filters: {len(hc_candidates):,} (need {N_HAS_CONTEXT})")
    hc_selected = apply_subject_cap(hc_candidates, N_HAS_CONTEXT, MAX_SUBJECT_SHARE)
    hc_meta = [meta for members, meta in hc_candidates if members in hc_selected][:N_HAS_CONTEXT]

    nc_candidates = select_no_context(pairs, rng)
    print(f"no-context candidates available: {len(nc_candidates):,} (need {N_NO_CONTEXT})")
    nc_selected = apply_subject_cap(nc_candidates, N_NO_CONTEXT, MAX_SUBJECT_SHARE)
    nc_meta = [meta for members, meta in nc_candidates if members in nc_selected][:N_NO_CONTEXT]

    all_selected = hc_selected + nc_selected
    # Deterministic pilot ordering: sort by original pair_id so reruns with
    # the same seed always produce the same file byte-for-byte.
    all_selected.sort(key=lambda members: members[0]["pair_id"])

    records: list[dict] = []
    all_warnings: list[str] = []
    for idx, members in enumerate(all_selected):
        pilot_pair_id = f"pilot_{idx:06d}"
        correct = next(m for m in members if m["label"] == 1)
        wrong = next(m for m in members if m["label"] == 0)
        rec_correct, w1 = build_record(correct, pilot_pair_id, 0, rng)
        rec_wrong, w2 = build_record(wrong, pilot_pair_id, 1, rng)
        rec_correct["reference_answer"] = rec_correct["candidate_answer"]
        rec_wrong["reference_answer"] = rec_correct["candidate_answer"]
        records.extend([rec_correct, rec_wrong])
        all_warnings.extend(w1 + w2)

    print(f"\nbuilt {len(records):,} records from {len(all_selected):,} pairs "
          f"({N_HAS_CONTEXT} has-context + {N_NO_CONTEXT} no-context)")

    if all_warnings:
        print(f"\n!! {len(all_warnings)} LEFTOVER-BENGALI WARNINGS -- fixing required before writing output:")
        for w in all_warnings[:20]:
            print(f"   {w}")
        raise SystemExit(1)
    print("no leftover Bengali characters in any field: OK")

    problems = validate_records(records, SCHEMA_PATH)
    if problems:
        print(f"\n!! {len(problems)} VALIDATION PROBLEMS -- output NOT written:")
        for p in problems[:30]:
            print(f"   {p}")
        raise SystemExit(1)
    print(f"schema + pairing + encoding validation: {len(records):,} records, 0 problems")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "pilot.jsonl", records)
    write_review_sheet(OUT_DIR / "review_sheet.csv", records)
    write_selection_log(OUT_DIR / "selection_log.csv", hc_meta, nc_meta)

    confidence_counts = collections.Counter(r["transliteration_confidence"] for r in records)
    subject_counts_hc = collections.Counter(m["subject"] for m in hc_meta)
    subject_counts_nc = collections.Counter(m["subject"] for m in nc_meta)

    report = OUT_DIR / "validation_report.txt"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("PILOT BUILD VALIDATION REPORT\n")
        fh.write("=" * 60 + "\n")
        fh.write(f"records written:        {len(records):,}\n")
        fh.write(f"pairs written:          {len(all_selected):,}\n")
        fh.write(f"  has-context pairs:    {N_HAS_CONTEXT}\n")
        fh.write(f"  no-context pairs:     {N_NO_CONTEXT}\n")
        fh.write(f"duplicate ids:          0\n")
        fh.write(f"schema violations:      0\n")
        fh.write(f"leftover Bengali chars: 0\n")
        fh.write("\ntransliteration confidence (per record):\n")
        for level in ("high", "medium", "low"):
            fh.write(f"  {level:<8} {confidence_counts.get(level, 0):>5}\n")
        fh.write("\nhas-context subjects:\n")
        for s, c in subject_counts_hc.most_common():
            fh.write(f"  {s:<24} {c}\n")
        fh.write("\nno-context subjects:\n")
        for s, c in subject_counts_nc.most_common():
            fh.write(f"  {s:<24} {c}\n")

    print(f"\nwrote:")
    print(f"  {OUT_DIR / 'pilot.jsonl'}          {len(records):,} records")
    print(f"  {OUT_DIR / 'review_sheet.csv'}     {len(all_selected):,} rows (one per pair)")
    print(f"  {OUT_DIR / 'selection_log.csv'}")
    print(f"  {report}")
    print(f"\nconfidence breakdown: {dict(confidence_counts)}")


if __name__ == "__main__":
    main()
