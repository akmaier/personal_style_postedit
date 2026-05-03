# Paper requirements (RRPR 2026, Track 2 — RR Results)

This document is the **acceptance checklist** for the paper. After we draft
it, we will go through every requirement here and confirm yes/no. The list
covers (a) what the venue formally requires, (b) what good RR-Track-2
papers contain, (c) audience-fit checks, and (d) **negative checks** that
catch the obvious LLM-paper failure modes (hallucinated references, wrong
page counts, missing citations to the source paper, etc.).

The numbers in brackets like `[A.1]` are reference IDs we will use when we
later build a `paper_check.md` review of the drafted paper.

---

## A. Formal venue requirements (CFP and submission page)

- **[A.1] Format:** LNCS Springer layout (`llncs.cls`, `splncs04.bst`)
  — exactly the template we have under `paper/template/`. Use it
  unmodified; do not redefine fonts or geometry.
- **[A.2] Length:** **6–14 pages** including figures and references.
  Extra pages can be requested if needed. Plan target = **12 pages**
  (8 body + 4 references/appendix), which leaves slack on both sides.
- **[A.3] Track:** Track 2 (RR Results). The CFP says authors of papers
  already accepted (or under review) at ICPR may propose a **companion
  paper describing the quality of the reproducible aspects**; that
  framing fits us — Baumler et al. is at ACL 26, and we are submitting a
  Track 2 RR paper companioning it.
- **[A.4] Track-2 content emphasis** (from the CFP, all 6 are encouraged):
  algorithmic implementation details; influence of parameters on result
  quality and how to optimize them; integration of source code into
  another framework; known limitations / difficult cases; future
  improvements; installation procedure. Cover **at least four** of the
  six in the paper.
- **[A.5] Reviewers will check the source code**, not just the text. Our
  paper must therefore include a working DOI/URL to the GitHub repo, the
  exact commit hash used to produce the figures, and a short
  reproducibility recipe (the `make` invocations).
- **[A.6] Blind review.** Anonymize: remove author names, anonymize the
  GitHub URL (e.g., a separate "submission/anonymized" branch on a
  pseudonymous account, or an anonymous archive). The submission PDF must
  not leak our names or affiliations in margins, watermarks, or PDF
  metadata.
- **[A.7] IAPR ethical declaration:** the paper is original, not under
  submission elsewhere, contains no plagiarism, and will be presented in
  person.
- **[A.8] ORCID encouraged** for all authors.
- **[A.9] Submission system:** EasyChair (the submission instructions
  page also references Microsoft CMT — we follow whichever the workshop
  page lists at submission time; do not assume).
- **[A.10] Deadline:** main paper submission **May 15, 2026**. Notification
  June 15. Camera-ready July 15. Track this in `paper_plan.md` §6.
- **[A.11] Proceedings:** Springer LNCS post-proceedings (separate from
  the main ICPR proceedings).

## B. Required content for any RR Track 2 paper

Every one of these must be in the paper, with a section pointer:

- **[B.1] Reference to the original paper.** Cite Baumler et al.
  (arXiv:2604.24444) on page 1, by name, and clearly say "this paper is
  a companion paper following the RRPR 2026 Track 2 framework".
- **[B.2] Brief restatement of the original paper's research question
  and main result**, so the reader does not need to fetch the original.
- **[B.3] What we reproduced** — list the H1–H3 hypotheses, the figures,
  and the rmcorr correlation, and report the numbers we obtained
  side-by-side with the paper's numbers. Include at least one of these
  side-by-side tables verbatim from `REPRODUCTION_REPORT.md`.
- **[B.4] How we reproduced it** — pipeline diagram (data → tidy →
  LUAR → similarity tables → tests → figures → report), the LUAR-MUD
  revision SHA (`9204529...`), the seeded RNG, and the
  permutation-test floor (`p = 1 / (n_perm + 1)` with `n_perm = 10 000`).
- **[B.5] Reproducibility verdict.** A short paragraph rating the
  original paper from a Track-2 perspective: data fully released, code
  fully not released, and what that meant for our reproduction effort.
- **[B.6] Beyond reproduction.** The follow-up experiment: leakage-free
  held-out protocol, n=324 paired tasks, GPT-5.5 and Claude Opus 4.7
  mimics, all-pairs paired permutation tests with BH-FDR. Report
  Hedges' *g*, 95% CIs, p-values, and the per-task win-rate vs human-edit.
- **[B.7] Limitations and known difficulties** (covers CFP item 4):
  self-experiment caveat, n_demos=1, workflow-not-skill comparison,
  style-fidelity-not-quality, the one Opus draft outlier.
- **[B.8] Code & data availability.** A clearly labelled section with
  GitHub URL + commit hash, plus instructions to reproduce from a fresh
  clone in <5 minutes on a CPU-only machine.
- **[B.9] Future improvements** (covers CFP item 5): the symmetric
  two-split LOOCV, full Likert agreement check, Pangram replication
  if API access is granted, and an independent third generator.

## C. Style and audience requirements

- **[C.1] Audience.** ICPR/RRPR is a pattern-recognition and computer-
  vision workshop, not an NLP venue. Readers know LUAR-style author
  embeddings the way a CV reader knows perceptual similarity metrics
  (i.e., conceptually but not always operationally). The paper must
  briefly explain LUAR-MUD in one paragraph at first mention, with a
  citation to Rivera-Soto et al.
- **[C.2] Tone (Maier-group anchor).** Use the structured, didactic,
  hybrid-modelling-aware tone of Maier et al. (Z Med Phys, 2019,
  *Gentle introduction to deep learning in medical image processing*) —
  numbered subsections, motivated definitions, explicit "what we mean by
  this term" paragraphs.
- **[C.3] Theory anchoring (Maier-group anchor).** Where it helps the
  reader, frame the LLM mimic as a **hybrid / known-operator-style
  approach** to text generation: the prompt encodes a known stylistic
  constraint, the LLM provides the unstructured generation. Cite
  Maier et al. ICPR 2018 (precision learning) and Maier et al. PBE 2022
  (known-operator review) when making this framing.
- **[C.4] Agentic-evaluation framing (Zaiss-group anchor).** Cite Zaiss
  et al. (arXiv:2604.13282, *Agentic MR sequence development*) when
  motivating LLM-vs-human comparisons via tightly-defined tasks. Their
  Table 1 (LLMs vs human, with cost and time) is a great model for our
  own LLM-vs-human comparison table.
- **[C.5] Plain-language sub-headlines.** Each section must start with a
  one-sentence "what this section is for". This is exactly how Maier et
  al. structure their reviews; readers love it.
- **[C.6] Numbers, not adjectives.** Effect sizes (Hedges' *g* with
  95 % CI) and p-values must accompany every claim. No bare "the model
  did better".
- **[C.7] No emoji, no first-person plural except in the standard
  scientific sense, no marketing language ("revolutionary", "powerful").**

## D. Negative / failure-mode checks (catch obvious LLM-paper flaws)

These are the items the user explicitly asked us to check. Each one will
become a bullet in `paper_check.md` after the draft exists.

- **[D.1] No hallucinated citations.** Every reference in `refs.bib`
  must have been verified during this planning turn or during a
  later, explicitly-named verification step. Maintain a
  `paper/references_verified.md` table that lists, for each citation,
  the URL or DOI we read and the date.
- **[D.2] No hallucinated numbers.** Every numerical claim in the paper
  must be traceable to a specific row in a CSV in `results/` or to a
  named LUAR/embedding cache. Build a `paper/numbers_traceability.md`
  in the verification step that maps "section X.Y, sentence Z" →
  "results/foo.csv row N column M".
- **[D.3] No missing citation to the source paper.** Baumler et al. must
  be cited (a) in the abstract or first paragraph, (b) at every place
  we report a number from the paper, (c) in the related-work section
  proper. Triple-check.
- **[D.4] No missing citation to LUAR.** Rivera-Soto et al. (LUAR,
  EMNLP 2021) must be cited the first time we say "LUAR" or "LUAR-MUD".
- **[D.5] Page-count check.** After the draft is built, run
  `pdfinfo paper/main.pdf | grep Pages` and confirm 6 ≤ pages ≤ 14.
  If it overflows, tighten methods/results, not the limitations.
- **[D.6] Figures all referenced from text.** No floating Figure that
  isn't mentioned by `\ref{fig:foo}` somewhere in the body.
- **[D.7] Tables all referenced from text.** Same as figures.
- **[D.8] All results in the paper match the committed CSVs.** A small
  shell script (`paper/scripts/check_numbers.sh`) will grep for the
  three headline numbers (g = +1.02, win rate 79 %, r = 0.244) and
  fail if they don't appear in the corresponding CSV files.
- **[D.9] No private/internal URLs.** No `cursor.com` URLs, no
  agent-temp paths, no `/opt/cursor/artifacts/`. Submission build is
  via plain `pdflatex` from a fresh clone of the GitHub repo.
- **[D.10] Anonymization for review.** PDF metadata stripped (`pdftk` or
  `exiftool`). No author affiliation strings in `\author{...}`.
- **[D.11] No prompt-engineering disclosure leaks.** The paper should
  describe the methodology of the mimic experiment without leaking the
  literal prompt of one specific participant; show only the **template**.
- **[D.12] Reproducibility footnote sanity.** Footnote on page 1 should
  give the GitHub repo URL + commit hash + the line:
  "All figures, tables, and statistical tests in this paper can be
  regenerated by running `make all && make mimic-compare && make
  final-assessment` after `pip install -r requirements.txt`."
- **[D.13] References stylistic consistency.** Use `splncs04.bst`
  (provided by LNCS); do not switch bib styles mid-paper.

## E. Hard "would block submission" checks

Listed separately because these are showstoppers, not nice-to-haves.

- **[E.1]** PDF builds cleanly with `pdflatex` + `bibtex` + 2× `pdflatex`
  on a fresh checkout. No errors. Warnings may be tolerated only if they
  do not affect typesetting.
- **[E.2]** Page count is in `[6, 14]`.
- **[E.3]** Source paper (Baumler et al.) is cited.
- **[E.4]** `references_verified.md` exists, every entry has a URL/DOI
  and verification date, and the count there equals the count in
  `refs.bib`.
- **[E.5]** A working GitHub URL appears in the paper.
- **[E.6]** The PDF passes the anonymization check (no author names in
  metadata or text).

---

## How this document gets used

After the paper is drafted, we will create `paper/paper_check.md` with
exactly these item IDs and a `[ok]` / `[issue]` / `[fix]` status next to
each one. No item gets a green check from "the agent thinks it's fine" —
each one is backed by a concrete, scriptable test (page count via
`pdfinfo`, citation presence via `grep`, number traceability via the CSV
mapping, etc.). That's the discipline that keeps this honest.
