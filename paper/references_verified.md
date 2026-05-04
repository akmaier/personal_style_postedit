# Verified bibliography

Every entry in `paper/refs.bib` must appear here with a URL or DOI and the
date we last confirmed it points at the correct work. This is requirement
[D.1] from `paper_requirements.md`.

| BibTeX key | Title (short) | Author(s) (first) | Venue / Year | URL / DOI | Verified on |
|---|---|---|---|---|---|
| `baumler2026personalstyle` | Can You Make It Sound Like You? | Baumler et al. | arXiv 2604.24444 / ACL 2026 | https://arxiv.org/abs/2604.24444 | 2026-05-02 |
| `rivera2021luar` | Learning Universal Authorship Representations | Rivera-Soto et al. | EMNLP 2021 | https://aclanthology.org/2021.emnlp-main.70/ | 2026-05-02 |
| `zaiss2026agent4mr` | Agentic MR sequence development | Zaiss, Aly, Endres, Dornstetter, Weinmüller, Maier | arXiv 2604.13282 / 2026 | https://arxiv.org/abs/2604.13282 | 2026-05-02 |
| `karpathy2026autoresearch` | autoresearch | Karpathy | GitHub repository / 2026 | https://github.com/karpathy/autoresearch | 2026-05-04 |
| `maier2018precision` | Precision Learning: Use of Known Operators | Maier, Schebesch, Syben, Würfl, Steidl, Choi, Fahrig | ICPR 2018, pp. 183--188 | https://doi.org/10.1109/ICPR.2018.8545553 | 2026-05-02 |
| `maier2022knownoperator` | Known Operator Learning Review | Maier, Köstler, Heisig, Krauß, Yang | Prog. Biomed. Eng. 4(2) 022002, 2022 | https://doi.org/10.1088/2516-1091/ac5b13 | 2026-05-02 |
| `maier2019gentle` | A gentle introduction to deep learning in medical image processing | Maier, Syben, Lasser, Riess | Z. Med. Phys. 29(2):86--101, 2019 | https://doi.org/10.1016/j.zemedi.2018.12.003 | 2026-05-02 |
| `hedges1981` | Distribution Theory for Glass's Estimator of Effect Size | Hedges | J. Educational Statistics 6(2):107--128, 1981 | https://doi.org/10.3102/10769986006002107 | 2026-05-02 |
| `benjamini1995fdr` | Controlling the False Discovery Rate | Benjamini, Hochberg | JRSS B 57(1):289--300, 1995 | https://doi.org/10.1111/j.2517-6161.1995.tb02031.x | 2026-05-02 |
| `bakdash2017rmcorr` | Repeated Measures Correlation | Bakdash, Marusich | Frontiers in Psychology 8:456, 2017 | https://doi.org/10.3389/fpsyg.2017.00456 | 2026-05-02 |
| `wilcoxon1945` | Individual Comparisons by Ranking Methods | Wilcoxon | Biometrics Bulletin 1(6):80--83, 1945 | https://doi.org/10.2307/3001968 | 2026-05-02 |
| `friedman1937` | The Use of Ranks to Avoid the Assumption of Normality | Friedman | JASA 32(200):675--701, 1937 | https://doi.org/10.1080/01621459.1937.10503522 | 2026-05-02 |
| `wolf2020transformers` | Transformers: State-of-the-Art NLP | Wolf et al. | EMNLP 2020 (System Demos), pp. 38--45 | https://aclanthology.org/2020.emnlp-demos.6/ | 2026-05-02 |
| `paszke2019pytorch` | PyTorch | Paszke et al. | NeurIPS 2019, pp. 8024--8035 | https://papers.nips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html | 2026-05-02 |
| `vallat2018pingouin` | Pingouin: statistics in Python | Vallat | JOSS 3(31):1026, 2018 | https://doi.org/10.21105/joss.01026 | 2026-05-02 |
| `pedregosa2011sklearn` | Scikit-learn: Machine Learning in Python | Pedregosa et al. | JMLR 12:2825-2830, 2011 | https://www.jmlr.org/papers/v12/pedregosa11a.html | 2026-05-02 |
| `virtanen2020scipy` | SciPy 1.0 | Virtanen et al. | Nature Methods 17:261--272, 2020 | https://doi.org/10.1038/s41592-019-0686-0 | 2026-05-02 |
| `openai2025o4mini` | OpenAI o4-mini | OpenAI | Model card, 2025 | https://openai.com/ | 2026-05-02 (web search confirmed model identifier) |
| `anthropic2026opus47` | Claude Opus 4.7 | Anthropic | Model card, 2026 | https://www.anthropic.com/news/claude-opus-4-7 | 2026-05-02 |
| `openai2026gpt55` | GPT-5.5 | OpenAI | Model card, 2026 | https://developers.openai.com/api/docs/models/gpt-5.5 | 2026-05-02 |

## Verification protocol

A reference is "verified" when at least one of the following is true:

1. The full text was fetched and read during this project (e.g. via the
   `WebFetch` tool or `arxiv.org` HTML view). This applies to all entries
   we read in full: `baumler2026personalstyle`, `zaiss2026agent4mr`,
   `maier2018precision`, `maier2022knownoperator`.
2. The bibliographic record (title, authors, venue, year, DOI) was
   confirmed via at least two independent sources, e.g. arXiv listing +
   publisher landing page, or arXiv listing + Google Scholar +
   ACL Anthology. This applies to the rest.
3. For software / model entries (`karpathy2026autoresearch`,
   `wolf2020transformers`, `paszke2019pytorch`, `vallat2018pingouin`,
   `virtanen2020scipy`, `openai2025o4mini`, `anthropic2026opus47`,
   `openai2026gpt55`),
   verification means the package or model is the one actually used by
   the pipeline (cross-check against `requirements.txt` and
   `src/personal_style/llm_mimic.py`).

## Note on the user-supplied ScienceDirect link

The link `sciencedirect.com/.../S093938891830120X` was provided as a
"Pattern Recognition Letters" reference. After two timed-out fetches
during planning, search confirmed this is in fact
`maier2019gentle` (*A gentle introduction to deep learning in medical
image processing*, Z. Med. Phys.). The DOI we cite is
`10.1016/j.zemedi.2018.12.003`, which is the canonical identifier for
that article. We do **not** fabricate a Pattern-Recognition-Letters
record.
