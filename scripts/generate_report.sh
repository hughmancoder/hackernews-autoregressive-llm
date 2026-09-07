#!/bin/bash
set -e

# Navigate to root directory
cd "$(dirname "$0")/.."

cleanup() {
    rm -f docs/*.aux docs/*.log docs/*.out docs/*.toc docs/*.fls docs/*.synctex* docs/*.bbl docs/*.blg docs/*.latex docs/*.pdf 2> /dev/null || true
}

# Always clean up auxiliary files on exit
trap cleanup EXIT

echo "Compiling LaTeX Report"

# Check if pdflatex or xelatex is available
if command -v pdflatex &> /dev/null; then
    LATEX_CMD="pdflatex"
elif command -v /Library/TeX/texbin/pdflatex &> /dev/null; then
    LATEX_CMD="/Library/TeX/texbin/pdflatex"
elif command -v xelatex &> /dev/null; then
    LATEX_CMD="xelatex"
elif command -v /Library/TeX/texbin/xelatex &> /dev/null; then
    LATEX_CMD="/Library/TeX/texbin/xelatex"
else
    echo "Error: Neither pdflatex nor xelatex found in PATH. Please install a TeX distribution (e.g. MacTeX/TeXLive)."
    exit 1
fi

TEX_SRC="docs/Report.tex"
if [ ! -f "$TEX_SRC" ] && [ -f "docs/Report.latex" ]; then
    TEX_SRC="docs/Report.latex"
fi

if [ ! -f "$TEX_SRC" ]; then
    echo "Error: Source LaTeX file not found ($TEX_SRC)."
    exit 1
fi

echo "Using compiler: $LATEX_CMD on $TEX_SRC"

# Initial cleanup of any previous residual files
cleanup

# First pass
if ! $LATEX_CMD -interaction=nonstopmode -output-directory=docs "$TEX_SRC" > /dev/null; then
    echo "LaTeX compilation failed on first pass. Relevant log snippet:"
    tail -n 25 docs/report.log 2>/dev/null || true
    exit 1
fi

# Second pass for resolving references, citations, and table labels
if ! $LATEX_CMD -interaction=nonstopmode -output-directory=docs "$TEX_SRC" > /dev/null; then
    echo "LaTeX compilation failed on second pass. Relevant log snippet:"
    tail -n 25 docs/report.log 2>/dev/null || true
    exit 1
fi

# Determine the generated PDF name
BASE_NAME=$(basename "${TEX_SRC%.*}")
PDF_OUTPUT="docs/${BASE_NAME}.pdf"

if [ ! -f "$PDF_OUTPUT" ]; then
    echo "Error: Expected PDF output $PDF_OUTPUT was not generated."
    exit 1
fi

# Copy generated PDF to root and mainrun/ folder as required
cp "$PDF_OUTPUT" Report.pdf
mkdir -p mainrun
cp "$PDF_OUTPUT" mainrun/Report.pdf

echo "Done! Report generated successfully at:"
echo " - Report.pdf (root)"
echo " - mainrun/Report.pdf"
