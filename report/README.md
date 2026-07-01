# Internship report (LaTeX)

`report.tex` + `references.bib` + `figures/` — a technical report on the
quadruped LiDAR navigation stack.

## Compile

**Easiest — Overleaf:** create a new project, upload `report.tex`,
`references.bib`, and the `figures/` folder, then compile (menu set to pdfLaTeX).

**Locally** (needs a TeX distribution: TeX Live / MacTeX / MiKTeX):

```bash
cd report
pdflatex report
bibtex   report
pdflatex report
pdflatex report
```

(Or simply `latexmk -pdf report.tex`, which runs the passes automatically.)

## Before submitting — two manual fixes

1. **Author/advisor:** edit the `\author{...}` block in `report.tex` (currently a
   placeholder) and add your advisor / the master's student you report to.
2. **The reference paper citation** (`@inproceedings{refpaper}` in
   `references.bib`): the PDF's body text is glyph-encoded, so only the *title* and
   *affiliations* could be extracted automatically. Fill in the **authors**,
   **venue**, and **year** from the IEEE Xplore page you downloaded it from. All
   other references are complete and verified.

## Regenerating the figures

The figures come straight from the project scripts (run from the repo root):

```bash
python3 demo.py                                   # -> out.png        (Fig. 2)
python3 examples/test_real_kitti.py --bin test-scans/000001.bin --out out_scan1.png  # Fig. 3
python3 examples/test_real_room.py                # -> out_room.png   (Fig. 4)
python3 examples/paper_navigation_demo.py         # -> out_paper.png  (Fig. 5)
python3 examples/paper_navigation_demo.py --mode follow --out out_paper_follow.png   # Fig. 6
python3 examples/viz3d.py                          # -> out_3d.png    (Fig. 7)
```

Then copy the PNGs into `report/figures/`.
