# RRPR 2026 paper

This subfolder contains everything needed to write and build a submission to
the [Sixth Workshop on Reproducible Research in Pattern Recognition (RRPR
2026)](https://tc22-team.github.io/rrpr2026/), an ICPR 2026 satellite
workshop. We are targeting **Track 2 — RR Results**, with our paper acting
as a companion to Baumler et al. (ACL 26).

## Layout

```
paper/
\u251c\u2500\u2500 README.md                  # this file
\u251c\u2500\u2500 paper_requirements.md      # what the final paper MUST satisfy (used for checks later)
\u251c\u2500\u2500 paper_plan.md              # outline, figures, citations, build plan
\u251c\u2500\u2500 template/                  # *unmodified* Springer LNCS template (downloaded May 2026)
\u2502   \u251c\u2500\u2500 samplepaper.tex
\u2502   \u251c\u2500\u2500 llncs.cls
\u2502   \u251c\u2500\u2500 splncs04.bst
\u2502   \u251c\u2500\u2500 fig1.eps
\u2502   \u251c\u2500\u2500 llncsdoc.pdf           # official LNCS author guide
\u2502   \u251c\u2500\u2500 history.txt
\u2502   \u2514\u2500\u2500 readme.txt
\u2514\u2500\u2500 (later turns will add main.tex, refs.bib, figures/, etc.)
```

## Status

This first turn delivers the **template, requirements, and plan**. No paper
text has been written yet — we deliberately separate planning from drafting
so that later checks against `paper_requirements.md` are meaningful.

## Building (once main.tex exists)

```
cd paper/
pdflatex -interaction=nonstopmode main.tex
bibtex   main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Sources verified during planning

- **Workshop CFP and submission instructions:** fetched from
  [tc22-team.github.io/rrpr2026/callForPapers.html](https://tc22-team.github.io/rrpr2026/callForPapers.html)
  and `submission.html` on May 2, 2026. Page-length window 6\u201314 pages. LNCS
  format. Track 2 emphasis on companion-paper-to-existing-paper structure.
- **Original arXiv paper to companion:** Baumler et al., *Can You Make It
  Sound Like You? Post-Editing LLM-Generated Text for Personal Style*,
  arXiv:2604.24444, ACL 26.
- **Style-inspiration papers** (verified during planning):
  - Zaiss, Aly, Endres, Dornstetter, Weinm\u00fcller, Maier, *Agentic MR
    sequence development*, arXiv:2604.13282 (2026). Read in full.
  - Maier, Schebesch, Syben, W\u00fcrfl, Steidl, Choi, Fahrig, *Precision
    Learning: Towards Use of Known Operators in Neural Networks*,
    arXiv:1712.00374 / ICPR 2018, pp. 183-188. Read in full.
  - Maier, K\u00f6stler, Heisig, Krau\u00df, Yang, *Known Operator Learning and
    Hybrid Machine Learning in Medical Imaging \u2014 A Review of the Past, the
    Present, and the Future*, arXiv:2108.04543 / Prog. Biomed. Eng. 4
    022002 (2022). Read in full.
  - **One source-link correction:** the user's third style link
    (`sciencedirect.com/.../S093938891830120X`) is **not** the *Pattern
    Recognition Letters* article it might appear to be \u2014 it is in fact
    Maier, Syben, Lasser, Riess, *A gentle introduction to deep learning in
    medical image processing*, **Z. Med. Phys. 29(2):86\u2013101 (2019)**,
    DOI 10.1016/j.zemedi.2018.12.003. The pedagogical, structured,
    "what does this mean for the field" tone of that article is exactly
    what we want, so we cite it accordingly.
