# Annotation Guidelines — BanglishHallu Phase 1

**Read this before you label anything.** Both annotators must read this same file. Don't
label from memory or from what "feels right" — when you're unsure, come back here first.

Written before annotation begins, per PRD R4. Based on `data/generated/pilot_v1/` and the
patterns found while building it. Revise this file after the 100-item test round (Step 4 below)
— whatever you and your friend disagreed on becomes a new rule here.

---

## 1. The one question you're answering

For every item, you see:

- A **question**
- Sometimes a **passage** (context) — sometimes not
- One **answer**

You decide: **is this answer correct or wrong?**

That's it. One yes/no decision per item. Everything below is just rules for the confusing cases.

---

## 2. The two kinds of items — ask a different question for each

### A) Items WITH a passage (has-context)

**Ask: "Does the passage support this answer?"**

Not "is this true in real life" — **only** "does the passage say this." Even if you personally
know the answer is historically wrong, if the passage says it, mark it **correct**. Even if the
answer happens to be true in reality, if the passage doesn't say it, mark it **wrong**.

**Real example from your data:**

> Passage: "...১৭৭৪ সালের সনদে কলকাতায় সুপ্রিম কোর্ট অব জুডিকেচার প্রতিষ্ঠিত হয়।"
> ("...by the 1774 charter, the Supreme Court of Judicature was established in Calcutta.")
>
> Answer A: "১৭৭৪ সালের সনদে..." (1774) → passage says this → **correct**
> Answer B: "১৭৭৫ সালের সনদে..." (1775) → passage says 1774, not 1775 → **wrong**

### B) Items WITHOUT a passage (no-context)

**Ask: "Is this true, as far as you know?"**

Here you *do* use your own knowledge, since there's no passage to check against.

---

## 3. Hard cases — the actual rules

These are the cases that cause disagreement. Read all of them before you start.

### Rule 1 — Different words, same meaning = still correct

If the answer says the same thing as the passage but in different words, it's still correct.
Don't require an exact word-for-word match.

> Passage: "গবাদি পশু আটক করা হয়।" (Cattle were seized.)
> Answer: "গবাদি পশু বাজেয়াপ্ত করা হয়েছে।" (Cattle were confiscated.)
> → Same meaning, different words → **correct**

### Rule 2 — Right topic, wrong fact = wrong

If the answer talks about the right thing but gets a name, number, date, or relationship
wrong, it's wrong — even though most of the sentence looks right.

> Passage lists: drugs seized, cattle detained, cash and gold seized.
> Answer: "গবাদি পশু বাজেয়াপ্ত এবং মাদক দ্রব্য আটক" (swaps which item got "seized" vs
> "detained") → **wrong** — this is exactly the kind of wrong answer we want in the dataset,
> because it can't be caught by just checking if the words appear in the passage.

### Rule 3 — Partly correct = wrong

If part of the answer is right and part is wrong, mark the whole answer **wrong**. We don't have
a "half-correct" option in Phase 1 — it's binary. Note it in your comments column though, it
matters for later.

> Question: "মুক্তিযুদ্ধ কোন সালে শুরু ও শেষ হয়?" (What year did the war start and end?)
> Answer: "১৯৭১ সালে শুরু, ১৯৭৩ সালে শেষ" (Started 1971, ended 1973 — end year is wrong)
> → **wrong**, even though the start year is right.

### Rule 4 — Hedged answers ("maybe", "possibly", "I think")

Judge the *fact*, ignore the hedge. If the fact underneath is right, it's correct, whether or
not it's hedged.

> "সম্ভবত ১৯৭১ সালে" (possibly in 1971) — if 1971 is the right answer → **correct**.
> Don't mark it wrong just because it says "possibly."

*(If you notice the hedged answers and the confident answers seem to line up with different
labels overall, tell me — that's a pattern the audit script should check for.)*

### Rule 5 — True in real life, but not in the passage (has-context only)

If there's a passage, and the answer is true in general knowledge but the passage never says
it, mark it **wrong**. Has-context items are only ever checked against the passage, never
against outside knowledge.

> Passage is about Bangladesh's independence day. Answer says something true about a
> different historical fact not mentioned in the passage → **wrong**, because the passage
> doesn't support it, even though it's a true fact somewhere else.

### Rule 6 — Numbers and dates that are close but different

A near-miss (9.8 vs 9.6, 1971 vs 1972) is still **wrong**. Also mark it as **hard** difficulty
(see Section 5) — near-misses are supposed to be the harder half of the dataset.

### Rule 7 — The question has more than one acceptable answer

Some questions genuinely have multiple correct phrasings or multiple correct entities (e.g. "a
prime number less than 50" has many right answers). If the given answer is *one of the valid*
answers, mark it correct, even if it's not the specific one you'd have picked.

### Rule 8 — Spelling doesn't matter

You're reading Banglish, and the same word may be spelled several different ways ("kobe" /
"kobey", "ebong" / "ebng"). **Never mark an answer wrong because of spelling.** Only judge the
meaning.

---

## 4. Labeling the type of wrong answer

For every answer you mark **wrong**, also pick one type. For every answer you mark **correct**,
the type is always `none`.

**If there was a passage (has-context), pick one:**

| Type | What it means | Example from your data |
|---|---|---|
| `entity` | Right kind of thing, wrong specific one — wrong person, place, or organisation | Passage says "Sheikh Mujib," answer says "Tajuddin Ahmad" |
| `numeric` | Wrong number, date, quantity, or unit | Passage says 1774, answer says 1775 |
| `relational` | Right entities, but who-did-what-to-whom is scrambled | Rule 2's cattle/drugs example above |
| `contradiction` | Directly says the opposite of what the passage says | Passage: "the law was passed." Answer: "the law was rejected." |

**If there was no passage (no-context), pick one:**

| Type | What it means |
|---|---|
| `fabricated` | Made up — not true and not real; invented fact |
| `overclaim` | Confidently answers something that isn't really knowable or has no fixed answer |

**When you genuinely can't tell `entity` from `relational`, or `fabricated` from `overclaim`:**
pick your best guess and write a note. Don't spend more than 10 seconds on this — the main
correct/wrong decision is what matters most; the type is secondary.

---

## 5. Marking difficulty

For every item, also mark **easy** or **hard**:

- **Easy** — the wrong answer is obviously unrelated (wrong subject entirely, or way off)
- **Hard** — the wrong answer is a close, believable near-miss (off-by-one date, a very similar
  name, a plausible-sounding swap)

Aim for roughly 6 easy for every 4 hard, across the whole set — but don't force it item by item,
just label what you actually see.

---

## 6. What to do when you and your friend disagree

Don't argue it out and change your answer to match — during Steps 3/4, **label alone**, and
disagreements get resolved *after*, not during. When you compare, count how many items you
disagreed on, and for each one, write down *why* the disagreement happened. That "why" is what
becomes a new rule in this file.

---

## 7. Quick checklist before you submit each batch

- [ ] Judged has-context items only against the passage, never against outside knowledge
- [ ] Judged no-context items using your own knowledge
- [ ] Ignored spelling differences
- [ ] Marked partly-correct answers as wrong
- [ ] Picked a type for every wrong answer
- [ ] Marked easy/hard for every item
- [ ] Didn't discuss items with your co-annotator before submitting
