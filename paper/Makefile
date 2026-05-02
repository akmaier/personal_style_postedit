PDFLATEX ?= pdflatex
BIBTEX   ?= bibtex
LATEXMK  ?= latexmk

PAPER       = main
PAPER_ARXIV = main_arxiv

.PHONY: all clean tables figures anonymize arxiv check distclean

all: tables figures $(PAPER).pdf

# Build the LaTeX tables from the canonical results CSVs.
tables:
	python3 scripts/build_tables.py

# Build the standalone figure PDFs that main.tex \includegraphics's.
figures: figures/_build/fig1_pipeline.pdf figures/_build/fig2_repro_montage.pdf

figures/_build/%.pdf: figures/%.tex
	@mkdir -p figures/_build
	-cd figures && $(PDFLATEX) -output-directory=_build -interaction=nonstopmode $*.tex >/dev/null
	@test -f figures/_build/$*.pdf || (echo "BUILD FAILED for $<:" && tail -40 figures/_build/$*.log && exit 1)

# Compile the main paper. Uses TEXINPUTS to find llncs.cls / splncs04.bst.
$(PAPER).pdf: $(PAPER).tex refs.bib tables figures
	-TEXINPUTS=.:./template: BSTINPUTS=.:./template: $(PDFLATEX) -interaction=nonstopmode $(PAPER).tex >/dev/null
	-TEXINPUTS=.:./template: BSTINPUTS=.:./template: $(BIBTEX)   $(PAPER) >/dev/null
	-TEXINPUTS=.:./template: BSTINPUTS=.:./template: $(PDFLATEX) -interaction=nonstopmode $(PAPER).tex >/dev/null
	-TEXINPUTS=.:./template: BSTINPUTS=.:./template: $(PDFLATEX) -interaction=nonstopmode $(PAPER).tex >/dev/null
	@test -f $(PAPER).pdf || (echo "BUILD FAILED:" && tail -40 $(PAPER).log && exit 1)
	@echo "==>" $$(pdfinfo $(PAPER).pdf | awk '/^Pages:/ {print $$2 " pages"}')

# Strip metadata for blind review.
anonymize: $(PAPER).pdf
	cp $(PAPER).pdf $(PAPER)_anonymous.pdf
	exiftool -overwrite_original -all= $(PAPER)_anonymous.pdf >/dev/null

# Author-credited build for arXiv: same source, anonymous switch flipped off.
# Uses -jobname so the aux files do not collide with the blind-review build.
arxiv: tables figures
	-TEXINPUTS=.:./template: BSTINPUTS=.:./template: $(PDFLATEX) -interaction=nonstopmode -jobname=$(PAPER_ARXIV) "\\AtBeginDocument{\\anonymousfalse}\\input{$(PAPER).tex}" >/dev/null
	-TEXINPUTS=.:./template: BSTINPUTS=.:./template: $(BIBTEX) $(PAPER_ARXIV) >/dev/null
	-TEXINPUTS=.:./template: BSTINPUTS=.:./template: $(PDFLATEX) -interaction=nonstopmode -jobname=$(PAPER_ARXIV) "\\AtBeginDocument{\\anonymousfalse}\\input{$(PAPER).tex}" >/dev/null
	-TEXINPUTS=.:./template: BSTINPUTS=.:./template: $(PDFLATEX) -interaction=nonstopmode -jobname=$(PAPER_ARXIV) "\\AtBeginDocument{\\anonymousfalse}\\input{$(PAPER).tex}" >/dev/null
	@test -f $(PAPER_ARXIV).pdf || (echo "ARXIV BUILD FAILED" && tail -40 $(PAPER_ARXIV).log && exit 1)
	@echo "==>" $$(pdfinfo $(PAPER_ARXIV).pdf | awk '/^Pages:/ {print $$2 " pages (arXiv build)"}')

# Run the requirement-checker (automated portion of paper_check.md).
check: $(PAPER).pdf
	python3 scripts/check_paper.py

# Remove only build artefacts (kept committed: figures/, tables/).
clean:
	rm -f $(PAPER).pdf $(PAPER)_anonymous.pdf
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).toc
	rm -rf figures/_build
	rm -f tables/*.tex

distclean: clean
