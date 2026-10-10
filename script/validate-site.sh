#!/usr/bin/env bash
#
# Post-build checks on _site/ that htmlproofer does not cover.
#
# htmlproofer validates links, images and HTML structure. It says nothing about
# whether the site is *discoverable*: whether the sitemap and feed parse, whether
# every page carries a usable description, or whether those descriptions are
# distinct. Each check below corresponds to a defect this repo actually shipped.
#
# Usage: script/validate-site.sh [site_dir]   (default: _site)

set -uo pipefail

SITE="${1:-_site}"
# Resolved from the script's own location, not the caller's cwd — the timezone
# check below reads _config.yml, and keying it on cwd meant it silently vanished
# whenever the script was run from anywhere but the repo root.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$ROOT/_config.yml"
failures=0

fail() { printf '  FAIL  %s\n' "$1" >&2; failures=$((failures + 1)); }
pass() { printf '  ok    %s\n' "$1"; }

# XML well-formedness through Ruby's bundled REXML rather than xmllint, so this
# script needs no toolchain beyond the Ruby the build already requires — CI used
# to apt-install libxml2-utils purely for this. rexml is in the bundle
# (Gemfile.lock, via html-proofer), so this works under `bundle exec` too.
xml_wellformed() {
  ruby -rrexml/document -e 'REXML::Document.new(File.read(ARGV[0]))' "$1" 2>/dev/null
}

if [ ! -d "$SITE" ]; then
  echo "no such directory: $SITE" >&2
  exit 1
fi

if [ ! -f "$CONFIG" ]; then
  echo "no _config.yml at $CONFIG" >&2
  exit 1
fi

# Read the canonical URL from config rather than repeating it here, so the gate
# cannot end up checking a domain the site no longer uses.
BASE_URL=$(sed -nE 's/^url:[[:space:]]*"?([^"[:space:]]+)"?[[:space:]]*$/\1/p' "$CONFIG" | head -1)
if [ -z "$BASE_URL" ]; then
  echo "could not read 'url:' from $CONFIG" >&2
  exit 1
fi

echo "Validating $SITE"

# --- XML feeds ---------------------------------------------------------------
# A byte before the declaration (BOM, stray newline from front matter) makes
# strict parsers reject the whole document.
for rel in sitemap.xml feed.xml; do
  f="$SITE/$rel"
  if [ ! -f "$f" ]; then
    fail "$rel is missing"
    continue
  fi
  if [ "$(head -c 5 "$f")" != "<?xml" ]; then
    fail "$rel does not start with '<?xml' (BOM or leading whitespace?)"
    continue
  fi
  if ! xml_wellformed "$f"; then
    fail "$rel is not well-formed XML"
    continue
  fi
  pass "$rel parses and starts at byte 0"
done

# --- sitemap contents --------------------------------------------------------
if [ -f "$SITE/sitemap.xml" ]; then
  locs=$(grep -oE '<loc>[^<]+</loc>' "$SITE/sitemap.xml" | sed -E 's|</?loc>||g')
  n=$(printf '%s\n' "$locs" | grep -c . )
  if [ "$n" -lt 1 ]; then
    fail "sitemap.xml lists no URLs"
  else
    pass "sitemap.xml lists $n URLs"
  fi
  bad=$(printf '%s\n' "$locs" | grep -v "^$BASE_URL/" || true)
  if [ -n "$bad" ]; then
    fail "sitemap.xml has URLs outside $BASE_URL/:"
    printf '%s\n' "$bad" | sed 's/^/          /' >&2
  else
    pass "every sitemap URL is absolute and under $BASE_URL/"
  fi
fi

# --- reproducible post URLs --------------------------------------------------
# Post permalinks embed :year/:month/:day and front-matter dates carry no offset,
# so an unset `timezone` resolves them in the build machine's timezone: a build
# from KST instead of CI's UTC moves afternoon-dated posts a day. Nothing in
# _site/ shows this, so the invariant has to be checked at the config.
if grep -qE '^timezone:[[:space:]]*\S' "$CONFIG"; then
  pass "_config.yml pins a timezone (post URLs are build-host independent)"
else
  fail "_config.yml sets no timezone — post URLs depend on the build machine"
fi

# --- robots.txt --------------------------------------------------------------
# jekyll-sitemap writes robots.txt whenever the source has none, so it always
# names the sitemap at the configured url. A hand-written one replaces that.
if [ ! -f "$SITE/robots.txt" ]; then
  fail "robots.txt is missing"
else
  if grep -qE '^[[:space:]]*Disallow:[[:space:]]*/[[:space:]]*$' "$SITE/robots.txt"; then
    fail "robots.txt disallows the whole site"
  else
    pass "robots.txt does not block crawling"
  fi
  if grep -qxF "Sitemap: $BASE_URL/sitemap.xml" "$SITE/robots.txt"; then
    pass "robots.txt points at $BASE_URL/sitemap.xml"
  else
    fail "robots.txt does not name $BASE_URL/sitemap.xml"
  fi
fi

# --- rendered pages ----------------------------------------------------------
# Only Jekyll-rendered documents; the search-engine ownership-verification files
# also end in .html but are plain text by design.
pages=$(grep -rl '<!DOCTYPE html>' "$SITE" --include='*.html' | sort)
page_count=$(printf '%s\n' "$pages" | grep -c .)
if [ "$page_count" -lt 1 ]; then
  fail "no rendered pages found in $SITE — the six per-page checks below would pass vacuously"
else
  pass "$page_count rendered pages found"
fi

extract_meta() { # file, meta-name
  grep -m1 -oE "<meta name=\"$2\" content=\"[^\"]*\"" "$1" 2>/dev/null |
    sed -E "s/.*content=\"//; s/\"$//"
}

missing_desc=""
missing_canonical=""
bad_h1=""
short_desc=""
descs=""
titles=""
while IFS= read -r f; do
  [ -n "$f" ] || continue
  d=$(extract_meta "$f" description)
  [ -n "$d" ] || missing_desc="$missing_desc$f"$'\n'
  grep -q 'rel="canonical"' "$f" || missing_canonical="$missing_canonical$f"$'\n'
  h1=$(grep -oE '<h1[^>]*>' "$f" | grep -c .)
  [ "$h1" = "1" ] || bad_h1="$bad_h1$h1  $f"$'\n'
  t=$(grep -m1 -oE '<title>[^<]*</title>' "$f" | sed -E 's|</?title>||g')
  if [ -n "$d" ] && [ "${#d}" -le "${#t}" ]; then
    short_desc="$short_desc${#d} vs ${#t}  $f"$'\n'"          title: $t"$'\n'"          desc : $d"$'\n'
  fi
  descs="$descs$d"$'\n'
  titles="$titles<title>$t</title>"$'\n'
done <<< "$pages"

# The layout renders the title as the page's h1. A post that also opens with
# "# Title", or uses "#" for its own sections, adds more.
if [ -n "$bad_h1" ]; then
  fail "pages without exactly one <h1>:"
  printf '%s' "$bad_h1" | sed 's/^/          /' >&2
else
  pass "every page has exactly one <h1>"
fi

# A heading outline that jumps h2 -> h4 breaks screen-reader navigation. The usual
# cause is a "### TL;DR" above the "##" sections it precedes.
skips=$(ruby -e '
  files = STDIN.read.split("\n").reject(&:empty?)
  tag = /<[^>]+>/
  files.each do |f|
    prev = nil
    File.read(f, encoding: "UTF-8", invalid: :replace).scan(%r{<h([1-6])[^>]*>(.*?)</h\1>}m) do |lvl, txt|
      lvl = lvl.to_i
      if prev && lvl > prev + 1
        puts "h#{prev} -> h#{lvl}  #{f}  \"#{txt.gsub(tag, "").strip[0, 40]}\""
      end
      prev = lvl
    end
  end
' <<< "$pages")
if [ -n "$skips" ]; then
  fail "heading outlines that skip a level:"
  printf '%s\n' "$skips" | sed 's/^/          /' >&2
else
  pass "no heading outline skips a level"
fi

if [ -n "$missing_desc" ]; then
  fail "pages with an empty meta description:"
  printf '%s' "$missing_desc" | sed 's/^/          /' >&2
else
  pass "every page has a meta description"
fi

# A description no longer than the page's own title cannot be adding information
# — e.g. a subtitle that only translates the title ("Qwen3 Technical Report" ->
# "Qwen3 기술 보고서"). Each such restatement is unique, so the duplicate check
# below cannot see it.
if [ -n "$short_desc" ]; then
  fail "pages whose meta description is no longer than their <title>:"
  printf '%s' "$short_desc" | sed 's/^/          /' >&2
else
  pass "every meta description is longer than its page title"
fi

if [ -n "$missing_canonical" ]; then
  fail "pages with no rel=canonical:"
  printf '%s' "$missing_canonical" | sed 's/^/          /' >&2
else
  pass "every page has rel=canonical"
fi

# Duplicates appear when descriptions come from a block every post shares, such
# as a "TL;DR" heading at the top of each paper post.
dup_desc=$(printf '%s' "$descs" | sort | uniq -d | grep -c .)
if [ "$dup_desc" -gt 0 ]; then
  fail "$dup_desc meta description(s) used on more than one page:"
  printf '%s' "$descs" | sort | uniq -d | cut -c1-90 | sed 's/^/          /' >&2
else
  pass "every meta description is unique"
fi

dup_title=$(printf '%s' "$titles" | sort | uniq -d | grep -c .)
if [ "$dup_title" -gt 0 ]; then
  fail "$dup_title <title> value(s) used on more than one page:"
  printf '%s' "$titles" | sort | uniq -d | sed 's/^/          /' >&2
else
  pass "every <title> is unique"
fi

# --- authoring sources must not ship ----------------------------------------
leaked=$(find "$SITE" -type f \( -name '*.excalidraw' -o -name '*.sh' -o -name '*.rb' \) | sort)
if [ -n "$leaked" ]; then
  fail "authoring sources copied into the site:"
  printf '%s\n' "$leaked" | sed 's/^/          /' >&2
else
  pass "no diagram sources, scripts or tests in the published output"
fi

echo
if [ "$failures" -gt 0 ]; then
  echo "$failures check(s) failed." >&2
  exit 1
fi
echo "All checks passed."
