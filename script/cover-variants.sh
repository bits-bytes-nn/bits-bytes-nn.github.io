#!/usr/bin/env bash
# Writes the copies of each hero cover that _includes/cover.html offers in place
# of the 1920px original: <name>-960.jpg and <name>-1280.jpg beside it, and a
# WebP of all three sizes (<name>.webp, <name>-960.webp, <name>-1280.webp),
# which browsers that read WebP take instead: about half the bytes of the JPEG
# at the same size. Run after adding or replacing a cover. macOS only (sips);
# jpegtran makes the JPEG copies progressive without touching their pixels, and
# cwebp (brew install webp) encodes each WebP from the original.
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
  for width in 960 1280 1920; do
    out="${src%.*}-${width}.webp"
    [ "$width" = 1920 ] && out="${src%.*}.webp"
    cwebp -quiet -q 82 -metadata icc -resize "$width" 0 "$src" -o "$out"
    echo "$out"
  done
done
