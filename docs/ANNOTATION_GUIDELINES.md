# Annotation Guidelines — Phase 1 (Bengali)

**Read this before you label anything.** Both annotators must read this same file.

Everything you will label is in **Bangla script**. There is no Banglish in Phase 1.

Written before annotation begins (PRD R4). Revise it after the 100-item agreement round —
whatever you and your co-annotator disagreed on becomes a new rule here.

---

## 1. The one question you are answering

Each item shows you:

- A **question** (প্রশ্ন)
- Sometimes a **passage** (অনুচ্ছেদ) — sometimes not
- One **answer** (উত্তর)

You decide: **is this answer correct, or wrong?**

One yes/no decision per item. Everything below is just rules for the confusing cases.

---

## 2. Two kinds of items — ask a different question for each

### A) Items WITH a passage (has-context)

**Ask: "Does the passage support this answer?"**

Not "is this true in real life" — only "does this passage say it." If the passage says it, mark
**correct**, even if you personally think it is historically wrong. If the passage does not say
it, mark **wrong**, even if it happens to be true.

**Real example from your corpus:**

> **অনুচ্ছেদ:** তামিম ইকবালের টেস্ট অভিষেক হয় ২০০৮ সালের ৪ জানুয়ারি নিউজিল্যান্ডের বিপক্ষে ডানেডিনে।
> তাঁর ওয়ানডে অভিষেক হয় ২০০৭ সালে জিম্বাবুয়ের বিপক্ষে।
>
> **প্রশ্ন:** তামিম ইকবালের ওয়ানডে অভিষেক কোন দলের বিপক্ষে?
>
> **উত্তর ১:** জিম্বাবুয়ের বিপক্ষে। → the passage says this → **correct**
> **উত্তর ২:** নিউজিল্যান্ডের বিপক্ষে। → New Zealand is in the passage, but for the **Test**
> debut, not the ODI → **wrong**

Notice how the wrong answer uses a real word from the passage. That is the whole point — you
cannot decide by checking whether the words appear in the text. You have to read what the
passage actually says about *this* question.

### B) Items WITHOUT a passage (no-context)

**Ask: "Is this correct, as far as you know?"**

These cover the whole subject range: Bangla language questions (grammar, বিপরীত শব্দ, সমার্থক
শব্দ, বাগধারা, শব্দার্থ), arithmetic, and also **general-knowledge questions** — law, science,
literature, BCS-style items.

> **প্রশ্ন:** 'অন্ধকার' শব্দের বিপরীতার্থক শব্দ-
> **উত্তর ১:** আলো → **correct**
> **উত্তর ২:** ইহলৗকিক → **wrong**

### You may look things up — but only for no-context items

Some no-context questions ask for a publication year, a constitutional article, or a scientific
name. Nobody knows all of those from memory. So:

| | Looking it up |
|---|---|
| **no-context** (no passage) | **Allowed and encouraged.** Check it, then label. Note in `your_notes` that you looked it up. |
| **has-context** (passage given) | **Not allowed.** Judge only against the passage. Looking it up will make you *wrong* — see Rule 5. |

That difference is the whole point. A has-context item asks "does this passage support this?"
An outside source cannot answer that question and will mislead you.

**If you cannot label an item honestly, write `unsure`.** See §2C — it covers more than "I don't
know the answer".

### 2C. `unsure` means "this item cannot be labelled" — not "I am not confident"

Use `unsure` whenever the **item itself** is broken, not just when you are hesitant. All of these
qualify, and all of them are worth flagging:

| Situation | Example |
|---|---|
| The passage does not answer the question at all | Q asks when §377 was introduced; the passage only discusses the 2018 ruling |
| The answer type does not match the question | Q asks **কবে** (when); the answer describes **what kind of person** he is |
| The two answers differ only by a typo | `শিক্ষা` vs `শিক্সা` (Rule 12) |
| The text is garbled or truncated | `...কিসের শ. ওপর নিভর্র করে? ড়ড় পব` |
| You genuinely cannot verify a no-context fact even after checking | an obscure sales figure |

Write a short reason in `your_notes` — "passage doesn't answer", "typo-pair", "asked when,
answered what" is enough.

**Why this matters.** `unsure` items are excluded from the agreement score rather than counted as
disagreements, and they are reported grouped by subject. A cluster in one subject is how a corpus
defect gets found. Guessing instead hides the defect *and* corrupts the score, so it is strictly
worse than saying `unsure`.

`unreadable` is still available for text you literally cannot read; in practice it and `unsure`
are handled identically.

---

## 3. Hard cases — the actual rules

These cause most disagreements. Read all fifteen before you start. Rules 11-15 were added
after the first agreement round (kappa = 0.717) — they are the cases the two annotators actually
disagreed on, so do not skip them.

### Rule 1 — Different words, same meaning = still correct

Do not require a word-for-word match with the passage.

> **অনুচ্ছেদ:** ১৯৩১ সালে এই গোপন সাক্ষাতের সময় আত্মগোপনে থাকা মাষ্টারদার সাথে ছিলেন **বিপ্লবী** নির্মল
> সেন, তারকেশ্বর দস্তিদার, শৈলেশ্বর চক্রবর্তী **এবং** কালীকিংকর দে।
>
> **প্রশ্ন:** ... তার সঙ্গে কারা ছিলেন?
>
> **উত্তর:** নির্মল সেন, তারকেশ্বর দস্তিদার, শৈলেশ্বর চক্রবর্তী **ও** কালীকিংকর দে
> → the same four people. `এবং` became `ও`, and `বিপ্লবী` was dropped. Neither changes the
> meaning → **correct**

### Rule 2 — Right topic, wrong fact = wrong

The answer talks about the right thing but gets a name, number, or relationship wrong.

> **অনুচ্ছেদ:** হাইড্রোজেন … তাই এটি সাধারণত **বিজারক** পদার্থ হিসেবে কাজ করে।
> **প্রশ্ন:** হাইড্রোজেন সাধারণত কী হিসেবে কাজ করে?
> **উত্তর:** জারক হিসেবে। → the exact opposite of the passage → **wrong**

### Rule 3 — Partly correct = wrong

If part is right and part is wrong, the whole answer is **wrong**. There is no half-correct
option in Phase 1. Note it in your comments — it matters for later.

> Question asks for start **and** end year. Answer gives the right start and the wrong end
> → **wrong**.

### Rule 4 — Hedged answers ("সম্ভবত", "হয়তো")

Judge the fact, ignore the hedge. If the fact underneath is right, it is **correct**. Do not
mark something wrong just because it says "সম্ভবত".

### Rule 5 — True in real life, but not in the passage (has-context only)

If there is a passage and the answer is true in general but the passage never says it, mark it
**wrong**. Has-context items are judged only against the passage.

### Rule 6 — Numbers, dates, and single-letter differences

A near-miss is still **wrong**. These are the hardest and most important items in the corpus.

> **প্রশ্ন:** 'পুনঃ' ও 'মিলন' শব্দ দুটির সন্ধিতে কোন শব্দ গঠিত হয়?
> **correct:** পুনর্মিলন  **wrong:** পুনঃমিলন

One visarga apart. Read carefully — do not skim these.

### Rule 7 — The question has more than one acceptable answer

If the given answer is *one of* the valid answers, mark it correct, even if it is not the one you
would have chosen.

### Rule 8 — Spelling does not matter, meaning does

Do not mark an answer wrong for a spelling slip if the intended word is clear. But be careful:
in সন্ধি, সমাস, and বানান questions the spelling **is** the answer (see Rule 6). Use judgement —
if the question is *about* the form of the word, the form matters.

### Rule 9 — Unreadable or broken text

If an item is garbled and you cannot read it, do not guess. Write `unreadable` in
`your_label` and explain in your notes. These get removed, not labelled.

### Rule 10 — When you simply do not know (no-context only)

Look it up first (see §2B). If you still cannot establish the answer, write `unsure` — never
guess.

This happens most on `law`, `science`, `bcs`, and `literature`, which ask for specific dates,
article numbers, and scientific names. Those subjects are deliberately in the corpus, so this is
expected rather than a problem.

Note this is only *one* of the reasons to write `unsure` — see §2C for the full list. In the
first agreement round, most `unsure` labels were not "I don't know" at all; they were items where
the passage did not answer the question.

### Rule 11 — The passage simply does not answer the question

This caused more disagreement than anything else in the first round. Distinguish two cases:

| What the passage does | Label |
|---|---|
| **Contradicts** the answer | **wrong** |
| **Says nothing** about it | **wrong** — for has-context, "not supported" and "contradicted" are both wrong |

Rule 5 still holds: for a has-context item you judge **only** against the passage, so an answer
the passage never supports is wrong even when it is true in real life.

> **Real case from round 1:** the passage describes Jagadish Chandra Bose's schooling and says he
> became interested in **physics**. The question asked which *literature* interested him, and the
> answer was `বাংলার লোক অভিনয়, যাত্রা-পালাগান, রামায়ণ ও মহাভারত` — true of his life, but absent
> from the passage. **Correct label: wrong.**

**But judge meaning, not words.** "Not supported" does not mean "the words are missing". If the
passage says the same thing in different words, it *is* supported (Rule 1).

If you think the passage is unrelated to the question, say so in your notes — a run of those in
one subject is a corpus problem worth reporting.

### Rule 12 — The two answers differ only by an obvious typo

Sometimes the correct and hallucinated answers are one character apart, with no change in
meaning:

> `…সম্প্রীতির শিক্ষা` vs `…সম্প্রীতির শিক্সা`
> `…তৃতীয় ওডিআইয়ে` vs `…তৃতীয় ওডিআঈয়ে`

These are corrupted duplicates, not hallucinations. Rule 8 says spelling does not make an answer
wrong, so you cannot label them honestly.

**Write `unsure` and note "typo-pair"** (§2C). Do not guess. About 2% of items look like this,
and flagging them is how they get removed.

**Do not confuse this with a changed number or a changed name.** Those *are* hallucinations even
when only one character moves:

| | |
|---|---|
| `১৬` vs `১৩`, `অনুচ্ছেদ ৩২` vs `অনুচ্ছেদ ২৭` | **wrong** — `numeric` |
| `ক্যাথোডে` vs `অ্যানোডে` | **wrong** — `entity` / `contradiction` |
| `হাবিবুল বাশার` vs `হাবিবুল বাসার` | **typo-pair** — same name, spelling variant → `unsure` |

The test: *does the change alter the fact, or only the spelling of the same fact?*

### Rule 13 — শাব্দিক অর্থ means the literal meaning

When a question asks for **শাব্দিক অর্থ**, give the literal, word-by-word sense — not the idiom.

> *"কপালপোড়া" এর শাব্দিক অর্থ কী?* → `পোড়া কপাল (burnt forehead)` → **correct**

Even though the idiom means "unlucky", the literal gloss is what was asked for. If the question
says **ভাবার্থ** instead, the idiomatic meaning is what is wanted.

### Rule 14 — The answer is a whole sentence copied from the passage

Some answers repeat an entire passage sentence instead of answering directly.

> Q: *রেগি টোরিয়ান কবে জন্মগ্রহণ করেন?* ("when was he born?")
> A: `রেগি টোরিয়ান (জন্ম ২২ এপ্রিল ১৯৭৫) একজন মার্কিন হার্ডলার।`

It feels like the wrong answer to the question, but **the asked-for fact is inside it** (the birth
date), and the passage supports it. **Label it correct**, and note "sentence dump".

Only mark it wrong if the required fact is genuinely absent or contradicted.

### Rule 15 — Ignore `[1]`, `[2][3]` in the passage

About one passage in seven still carries Wikipedia footnote markers. They are leftover formatting.
Read straight through them; they never affect the label.

---

## 4. Labelling the type of wrong answer

For every **wrong** answer, also pick one type. For every **correct** answer the type is `none`.

**If there was a passage (has-context):**

| Type | What it means | Example |
|---|---|---|
| `entity` | Wrong person, place, or organisation | Passage says জিম্বাবুয়ে, answer says নিউজিল্যান্ড |
| `numeric` | Wrong number, date, quantity, or unit | Passage says ১৭৭৪, answer says ১৭৭৫ |
| `relational` | Right entities, but who-did-what is scrambled | Which item was seized vs detained gets swapped |
| `contradiction` | Says the opposite of the passage | Passage says বিজারক, answer says জারক |

**If there was no passage (no-context):**

| Type | What it means |
|---|---|
| `fabricated` | Simply not a real or true answer |
| `overclaim` | Confidently answers something that has no fixed answer |

If you genuinely cannot tell `entity` from `relational`, pick your best guess and write a note.
Do not spend more than 10 seconds on the type — the correct/wrong decision matters far more.

---

## 5. Marking difficulty

Mark **easy** or **hard** for every item:

- **Easy** — the wrong answer is obviously unrelated
- **Hard** — the wrong answer is a close, believable near-miss

For **correct** answers, judge how obvious it is that the answer is right: **easy** if plainly
correct, **hard** if you had to read carefully to confirm it.

Do not force a ratio. Label what you actually see.

---

## 6. When you and your co-annotator disagree

**Label alone. Do not compare answers while working.** Disagreements are resolved afterwards,
not during — that is what the whole test measures.

When you compare at the end, count how many items you disagreed on, and for each write down
*why*. Those reasons become new rules in this file.

---

## 7. Checklist before you submit

- [ ] Judged has-context items only against the passage, never outside knowledge
- [ ] Judged no-context items using ordinary Bangla knowledge
- [ ] Marked partly-correct answers as wrong
- [ ] Read the near-miss items carefully (one letter can be the whole difference)
- [ ] Picked a type for every wrong answer
- [ ] Marked easy/hard for every item
- [ ] Looked up no-context items where needed, and noted that you did
- [ ] Never looked anything up for a has-context item
- [ ] Used `unsure` instead of guessing, `unreadable` for broken text, and `unsure` + a
      "typo-pair" note where the two answers differ only by spelling (Rule 12)
- [ ] Did not discuss items with your co-annotator
