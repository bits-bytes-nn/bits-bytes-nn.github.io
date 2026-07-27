#!/bin/bash
# Render Excalidraw diagrams to PNG with handwriting font.
#
# Why this dance: excalidraw-to-png embeds the font as woff2 @font-face, which
# Resvg (the SVG→PNG backend) can't decode, so it falls back to a plain system
# font. The fix is to reference the font by NAME and have that name installed
# system-wide as a TTF. So we:
#   1. export to SVG,
#   2. strip the woff2 @font-face and rewrite font-family → Excalifont,
#   3. render that SVG to PNG (Resvg picks up the installed Excalifont.ttf).
#
# Use Excalifont, NOT Caveat: the repo's earlier diagrams (agentic-loop.png,
# architecture-overview.png, …) were exported by Excalidraw itself and carry
# real Excalifont — upright and monoline. Caveat is slanted and cursive, so
# mixing the two makes new posts visibly inconsistent with old ones.
#
# One-time setup on a new machine — install the full TTF that ships with
# @excalidraw/utils (the woff2 subsets in @excalidraw/excalidraw won't do):
#   cp "$(find ~/.npm /usr/lib/node_modules -name Excalifont.ttf | head -1)" ~/.fonts/
#   fc-cache -f ~/.fonts
#
# Paths are derived from this script's location so it runs on any machine
# (macOS laptop, Linux dev box) without editing.
set -e
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
# @resvg/resvg-js lives in the excalidraw-to-png install, wherever that is.
e2p="$(readlink -f "$(command -v excalidraw-to-png)")"
resvg=""
for d in "$(dirname "$e2p")" "$(dirname "$e2p")/.."; do
  [[ -d "$d/node_modules/@resvg/resvg-js" ]] && resvg="$(cd "$d" && pwd)/node_modules/@resvg/resvg-js"
done
[[ -n "$resvg" ]] || { echo "resvg-js not found near $e2p" >&2; exit 1; }
for name in "$@"; do
  src="assets/images/$name.excalidraw"
  svg="/tmp/$name.svg"
  out="assets/images/$name.png"
  excalidraw-to-png "$src" "$svg" --svg >/dev/null 2>&1
  python3 - "$svg" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = re.sub(r'@font-face\s*\{[^}]*\}', '', s)              # drop undecodable woff2
s = re.sub(r"font-family:\s*[^;\"}]+", "font-family: 'Excalifont', 'Caveat', cursive", s)
s = re.sub(r'font-family="[^"]*"', "font-family=\"'Excalifont', 'Caveat', cursive\"", s)
open(p, 'w').write(s)
PY
  node -e "
    const {Resvg}=require('$resvg');
    const fs=require('fs');
    const svg=fs.readFileSync('$svg','utf8');
    const r=new Resvg(svg,{fitTo:{mode:'zoom',value:2},font:{loadSystemFonts:true,defaultFontFamily:'Excalifont'}});
    fs.writeFileSync('$out',r.render().asPng());
  "
  echo "rendered $out"
done
