#!/usr/bin/env bash
#
# Compile paper.md + paper.bib into josspdf/paper.pdf for proofreading.
#
#   ./josspdf/build.sh
#
# This is a local draft build (pandoc + xelatex with a JOSS-like template),
# not the official JOSS layout. The authoritative render is produced by the
# openjournals inara tool at submission time:
#   docker run --rm -v $PWD:/data openjournals/inara -o pdf paper.md

set -euo pipefail
cd "$(dirname "$0")/.."

pandoc paper.md \
    --from markdown+autolink_bare_uris \
    --citeproc \
    --bibliography paper.bib \
    --template josspdf/template.tex \
    --pdf-engine xelatex \
    --metadata link-citations=true \
    -o josspdf/paper.pdf

echo "wrote josspdf/paper.pdf"
