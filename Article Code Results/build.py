"""Compile the supplied TeX documents using only package-local inputs.

Requires a TeX distribution with latexmk, pdfLaTeX, and BibTeX on PATH.
No Python packages are needed unless --figures is requested.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DOCUMENTS = {
    "clean": "main_revised_23092026_norebutal.tex",
    "marked": "main_revised_23092026_tracked.tex",
    "rebuttal": "rebuttal_full_revision_2026-09-23.tex",
    "changes": "additions_and_removals_2026-09-23.tex",
    "notes": "revision_notes_2026-09-23.tex",
    "guide": "source_bundle_guide_2026-09-23.tex",
    "baseline": "original_baseline_compile.tex",
    "reproduction": "reproduction_report_2026-09-23.tex",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", choices=["all", *DOCUMENTS], default="all")
    parser.add_argument("--figures", action="store_true",
                        help="Regenerate all nine native PDF figures from saved results first")
    parser.add_argument("--force", action="store_true", help="Force a complete latexmk rebuild")
    args = parser.parse_args()
    for tool in ["latexmk", "pdflatex", "bibtex"]:
        if shutil.which(tool) is None:
            parser.error(f"{tool} is not on PATH. Install a full TeX Live or MiKTeX distribution.")
    if args.figures:
        for script in ["generate_figures.py", "plot_verified_experiments.py"]:
            subprocess.run([sys.executable, str(ROOT / "code" / script)], check=True)
    output = ROOT / "build"
    output.mkdir(exist_ok=True)
    reports = []
    selected = DOCUMENTS if args.document == "all" else {args.document: DOCUMENTS[args.document]}
    for name, filename in selected.items():
        print(f"Building {name}: {filename}", flush=True)
        command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                   "-file-line-error", "-outdir=build"]
        if args.force:
            command.append("-g")
        command.append(filename)
        with (output / f"{name}_build.txt").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"Build failed for {name}; see build/{name}_build.txt")
        stem = Path(filename).stem
        transcript = (output / f"{stem}.log").read_text(errors="replace")
        faults = re.findall(r"^.*(?:undefined|multiply defined|multiply-defined|Overfull|Float too large|^!).*$",
                            transcript, flags=re.M | re.I)
        if faults:
            raise SystemExit(f"Unresolved TeX diagnostics for {name}:\n" + "\n".join(faults))
        pages = re.search(r"Output written on .*?\((\d+) pages?", transcript, flags=re.S)
        pdf = output / f"{stem}.pdf"
        if not pdf.is_file():
            raise SystemExit(f"Expected PDF was not produced: {pdf}")
        reports.append({"document": name, "source": filename, "pdf": str(pdf.relative_to(ROOT)),
                        "pages": int(pages.group(1)) if pages else None,
                        "status": "compiled without errors, missing references, or overfull boxes"})
    (output / "build_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"PASS: {len(reports)} documents. PDFs and logs are in {output}")


if __name__ == "__main__":
    main()
