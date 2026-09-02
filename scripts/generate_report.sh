#!/bin/bash

# Navigate to root directory
cd "$(dirname "$0")/.."

echo "Generating Report.pdf from docs/Report.md..."

# Use md-to-pdf via npx to convert Markdown to PDF
npx -y md-to-pdf docs/Report.md

# Move the generated PDF to the root folder
mv docs/Report.pdf Report.pdf

echo "Done! Report.pdf is now in the root folder."
