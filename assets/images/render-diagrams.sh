#!/bin/bash
# Render Excalidraw diagrams to PNG with handwriting font.
#
# Why this dance: excalidraw-to-png embeds the font as woff2 @font-face, which
# Resvg (the SVG→PNG backend) can't decode, so it falls back to a plain system
# font. The existing hand-drawn diagrams worked because their SVG referenced
# `'Caveat', cursive` by NAME — and Caveat IS installed system-wide. So we:
#   1. export to SVG,
#   2. strip the woff2 @font-face and rewrite font-family → Caveat/cursive,
#   3. render that SVG to PNG (Resvg picks up the installed Caveat.ttf).
set -e
cd /Users/youngmki/Projects/tech-blog
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
s = re.sub(r"font-family:\s*[^;\"}]+", "font-family: 'Caveat', 'Segoe Print', 'Comic Sans MS', cursive", s)
s = re.sub(r'font-family="[^"]*"', "font-family=\"'Caveat', cursive\"", s)
open(p, 'w').write(s)
PY
  node -e "
    const {Resvg}=require('/Users/youngmki/Projects/tools/excalidraw-to-png/node_modules/@resvg/resvg-js');
    const fs=require('fs');
    const svg=fs.readFileSync('$svg','utf8');
    const r=new Resvg(svg,{fitTo:{mode:'zoom',value:2},font:{loadSystemFonts:true,defaultFontFamily:'Caveat'}});
    fs.writeFileSync('$out',r.render().asPng());
  "
  echo "rendered $out"
done
