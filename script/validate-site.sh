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
BASE_URL="https://bits-bytes-nn.github.io"
failures=0

fail() { printf '  FAIL  %s\n' "$1" >&2; failures=$((failures + 1)); }
pass() { printf '  ok    %s\n' "$1"; }

# XML well-formedness through Ruby's bundled REXML rather than xmllint, so this
# script needs no toolchain beyond the Ruby the build already requires — CI used
# to apt-install libxml2-utils purely for this.
#
# RUBYOPT/BUNDLE_GEMFILE are cleared because bundler injects bundler/setup, which
# restricts $LOAD_PATH to the Gemfile's gems; rexml is not one of them.
xml_wellformed() {
  env -u RUBYOPT -u BUNDLE_GEMFILE -u BUNDLE_BIN_PATH \
    ruby -rrexml/document -e 'REXML::Document.new(File.read(ARGV[0]))' "$1" 2>/dev/null
}

if [ ! -d "$SITE" ]; then
  echo "no such directory: $SITE" >&2
  exit 1
fi

echo "Validating $SITE"

# --- XML feeds ---------------------------------------------------------------
# A byte before the declaration (BOM, stray newline from front matter) makes
# strict parsers reject the whole document.
for rel in sitemap.xml sitemap-index.xml feed.xml; do
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
# so an unset `timezone` resolves them in the build machine's timezone. Building
# from KST instead of CI's UTC moved 19 URLs by a day. Nothing in _site/ shows
# this, so the invariant has to be checked at the config.
if [ -f _config.yml ]; then
  if grep -qE '^timezone:[[:space:]]*\S' _config.yml; then
    pass "_config.yml pins a timezone (post URLs are build-host independent)"
  else
    fail "_config.yml sets no timezone — post URLs depend on the build machine"
  fi
fi

# --- robots.txt --------------------------------------------------------------
if [ ! -f "$SITE/robots.txt" ]; then
  fail "robots.txt is missing"
elif grep -qE '^[[:space:]]*Disallow:[[:space:]]*/[[:space:]]*$' "$SITE/robots.txt"; then
  fail "robots.txt disallows the whole site"
else
  pass "robots.txt does not block crawling"
fi

# --- rendered pages ----------------------------------------------------------
# Only Jekyll-rendered documents; the search-engine ownership-verification files
# also end in .html but are plain text by design.
pages=$(grep -rl '<!DOCTYPE html>' "$SITE" --include='*.html' | sort)
page_count=$(printf '%s\n' "$pages" | grep -c .)
pass "$page_count rendered pages found"

extract_meta() { # file, meta-name
  grep -m1 -oE "<meta name=\"$2\" content=\"[^\"]*\"" "$1" 2>/dev/null |
    sed -E "s/.*content=\"//; s/\"$//"
}

missing_desc=""
missing_canonical=""
bad_h1=""
descs=""
titles=""
while IFS= read -r f; do
  [ -n "$f" ] || continue
  d=$(extract_meta "$f" description)
  [ -n "$d" ] || missing_desc="$missing_desc$f"$'\n'
  grep -q 'rel="canonical"' "$f" || missing_canonical="$missing_canonical$f"$'\n'
  h1=$(grep -oE '<h1[^>]*>' "$f" | grep -c .)
  [ "$h1" = "1" ] || bad_h1="$bad_h1$h1  $f"$'\n'
  descs="$descs$d"$'\n'
  titles="$titles$(grep -m1 -oE '<title>[^<]*</title>' "$f")"$'\n'
done <<< "$pages"

# The layout renders the title as the page's h1. A post that also opens with
# "# Title", or uses "#" for its own sections, adds more.
if [ -n "$bad_h1" ]; then
  fail "pages without exactly one <h1>:"
  printf '%s' "$bad_h1" | sed 's/^/          /' >&2
else
  pass "every page has exactly one <h1>"
fi

# A heading outline that jumps h2 -> h4 breaks screen-reader navigation. Every
# paper post used to do worse than that: "### TL;DR" above the "##" sections it
# preceded, and "#" reused for the post's own sections.
skips=$(env -u RUBYOPT -u BUNDLE_GEMFILE ruby -e '
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

if [ -n "$missing_canonical" ]; then
  fail "pages with no rel=canonical:"
  printf '%s' "$missing_canonical" | sed 's/^/          /' >&2
else
  pass "every page has rel=canonical"
fi

# Duplicates are what made 28 paper posts share one description: the template
# derived it from Jekyll's excerpt, which for those posts was a heading.
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
leaked=$(find "$SITE" -type f \( -name '*.excalidraw' -o -name '*.sh' \) | sort)
if [ -n "$leaked" ]; then
  fail "authoring sources copied into the site:"
  printf '%s\n' "$leaked" | sed 's/^/          /' >&2
else
  pass "no diagram sources or scripts in the published output"
fi

echo
if [ "$failures" -gt 0 ]; then
  echo "$failures check(s) failed." >&2
  exit 1
fi
echo "All checks passed."
