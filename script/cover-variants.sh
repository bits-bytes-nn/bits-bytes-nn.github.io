#!/usr/bin/env bash
# Writes the 960px and 1280px copies of each hero cover that _includes/cover.html
# offers phones and tablets in place of the 1920px original: <name>-960.jpg and
# <name>-1280.jpg beside it. Run after adding or replacing a cover. macOS only
# (sips); jpegtran makes the copies progressive without touching their pixels.
#
# Usage: script/cover-variants.sh <cover.jpg>...
set -euo pipefail

[ "$#" -gt 0 ] || { echo "usage: $0 <cover.jpg>..." >&2; exit 2; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for src in "$@"; do
  for width in 960 1280; do
    out="${src%.*}-${width}.${src##*.}"
    sips -s format jpeg -s formatOptions 85 --resampleWidth "$width" "$src" --out "$tmp/cover.jpg" >/dev/null
    jpegtran -copy all -optimize -progressive -outfile "$out" "$tmp/cover.jpg"
    echo "$out"
  done
done
