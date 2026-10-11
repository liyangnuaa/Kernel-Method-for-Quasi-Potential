# Complete revised paper and reproducible experiments — 23 September 2026

The clean article is 32 pages and presents verified numerical results without audit warnings, reviewer discussion, or references to earlier manuscript versions. The full blue/red comparison retains removed material. All TeX sources, bibliography inputs, native PDF figures, plotting code, fitted numerical arrays, and experiment code are included.

## Compile all eight documents

Install a full TeX Live or MiKTeX distribution with `latexmk`, `pdflatex`, and `bibtex`, then run:

```sh
python3 build.py --force
```

No scientific Python packages or network access are needed to compile. Precompiled copies of all eight documents are supplied in `pdf/`. New PDFs and logs are written under `build/`. Individual targets are `clean`, `marked`, `rebuttal`, `changes`, `notes`, `guide`, `baseline`, and `reproduction`.

For Overleaf, upload the ZIP and select `main_revised_23092026_norebutal.tex` or `main_revised_23092026_tracked.tex`, using pdfLaTeX. All figures are pre-generated. The unchanged original source is compiled through `original_baseline_compile.tex`, which supplies compilation adapters without changing the original file.

## Rerun the main-text experiments

```sh
python -m pip install -r requirements-reproduction.txt
python code/run_polynomial_experiments.py
python code/refit_selected_kernel.py
python code/plot_verified_experiments.py
```

The first program fits the two-dimensional systems from HJ residuals and local Riccati curvature, selects polynomial degree using independent HJ validation, and saves every candidate and seed. The second refits the anchored degree-two kernel in identifiable feature coordinates for N=50,100,200,400,800. Reference potentials do not enter fitting or selection. The third generates the seven main figures as native vector PDFs and writes the two example tables.

Saved inputs, coefficients, validation scores, solver termination data, prediction checks, and independent test outputs are in `results/accuracy/`. Achieved RMSEs include:

- Maier–Stein, original 10×10 broad evaluation grid, seed 42: **1.1343e-10**.
- Double well, 101×101 wider reference grid, seed 42: **2.8731e-7**; independent right-basin test: **5.2240e-8**.
- Multiplicative example, N=200, original 20³ grid: **1.3547e-9**.

These are the documented direct/stabilized implementations described in the clean paper, not a claim that every historical mixture-selection result has been replicated. The polynomial benchmarks contain the true functions in their candidate spaces, and the three-dimensional Riccati warm start is exact. Detailed provenance and exact-replay results are kept in the separate reproduction report.

## Rerun the appendix comparisons

```sh
python code/run_experiments.py
python code/make_tables.py
python code/generate_figures.py
```

These experiments were also rerun. All 36 method/seed records and fitted models, including the structured eight-dimensional test, are in `results/experiments.json` and `results/fitted_models.npz`. The appendix reports their objectives, domains, timing protocol, and limitations.

## Replot the delivered data without fitting

```sh
python code/generate_figures.py
python code/plot_verified_experiments.py
python code/check_vector_pdfs.py
python build.py --force
```

The checker rejects raster objects and unembedded fonts. No original PNG is used to make the clean figures. The original PNGs remain unchanged in the comparison archive and appear in red in the full marked paper.

## Recovered code and computational history

Eight unchanged scripts and path/hash provenance are in `code/recovered_originals/`. The following commands reproduce diagnostic stages separately:

```sh
python code/replay_maier.py
python code/replay_maier_alternative.py
python code/replay_example3.py
python code/rerun_example3_analytic_gradient.py
python code/make_reproduction_report.py
```

Literal replays retain the recovered settings; the analytic-gradient rerun discloses its larger budgets and output normalization. Some diagnostic stages fail to converge. Their records are preserved in the separate computational report, not inserted as warnings in the clean paper. `code/plot_recovered_runs.py` writes historical diagnostic plots to `figures/audit/`, which cannot overwrite the clean figures.

Refitting may vary slightly by numerical libraries and platform; timings vary by machine. The delivered figures and TeX tables regenerate from the saved arrays. The TeX prose contains the rounded delivered values, so a materially different refit requires reconciling the paper.

## Source inventory

- `main_revised_23092026_norebutal.tex`: complete clean paper.
- `main_revised_23092026_tracked.tex`: full blue-addition/red-removal manuscript.
- `rebuttal_full_revision_2026-09-23.tex`: point-by-point response to the five comments.
- `additions_and_removals_2026-09-23.tex`: exact embedded source diff.
- `changes_from_requested_baseline.diff`: the same diff as plain text.
- `revision_notes_2026-09-23.tex`, `source_bundle_guide_2026-09-23.tex`: revision inventory and build guide.
- `reproduction_report_2026-09-23.tex`: separate numerical reproduction and accuracy report.
- `main_revised_05032026_norebutal.tex`, `original_baseline_compile.tex`: unchanged baseline and build adapter.
- `references_tracked_2026-09-23.tex`: marked bibliography fragment.
- Five `results/*.tex` table fragments: two main-text tables and three appendix tables.
- `verification/`, `preservation_audit.json`, `FIGURE_PROVENANCE.json`, `MANIFEST.sha256`: verification, source accounting, provenance, and file hashes.

The original 28-page source is byte-identical, all three systems remain, and all 55 original citation keys remain cited. The numerical tables and figures in the clean paper were updated after recomputation; the full source record remains available in the baseline and marked comparison.
