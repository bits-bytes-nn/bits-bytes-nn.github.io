# Writing a post

How a post gets here, what its front matter means, and the rules the lint holds
it to. The checks themselves are in [operations.md](operations.md); how the site
turns a post into pages is in [architecture.md](architecture.md).

- [Where posts come from](#where-posts-come-from)
- [Front matter](#front-matter)
- [Dates](#dates)
- [The search snippet: description](#the-search-snippet-description)
- [Categories, URLs and covers](#categories-urls-and-covers)
- [Tags](#tags)
- [Translations](#translations)
- [Math](#math)
- [Images and diagrams](#images-and-diagrams)
- [What the lint catches](#what-the-lint-catches)

## Where posts come from

- **Generated posts**: [scholar-lens](https://github.com/bits-bytes-nn/scholar-lens)
  writes paper reviews, paper summaries, paper surveys and tech guides and opens
  them as pull requests. The PR runs the same gates as any other change; check
  the rendered post in a local build before merging.
- **Hand-written posts** (Insights, and any other post) are written directly or
  with the local `/write-post` skill (under the gitignored `.claude/`), which
  runs research → draft → proofread.

Two rules exist because scholar-lens finds a generated post by its slug to
update it:

- **Never rename a post's file or its images** (`<slug>-<n>.png`). The slug is
  also the URL.
- **A hand fix to a generated post** is overwritten when the paper is
  regenerated. List the post and each fix in the PR description, with the
  source value for factual or math errors ("paper 272/122/199, post 27/21/22"),
  so the generator can be corrected.

After a post merges, record its URL (see
[operations.md](operations.md#published-urls)).

## Front matter

A post is `_posts/YYYY-MM-DD-slug.md`, Korean by default. A **paper post** is a
Paper Reviews or Paper Summaries post (the types with `paper: true` in the
conventions): it reviews one paper, so it has `paper_author` and a required
`subtitle`. Paper Surveys walk through several papers and are not paper posts.
Every post starts with:

```yaml
---
layout: post
title: "<Post Title>"
subtitle: "<one-line pitch>"       # paper posts: required, not the title; else optional
date: YYYY-MM-DD HH:MM:SS          # paper posts: the paper's date; else the posting date
paper_author: "<Org>"              # the paper's org, not people; paper posts only
paper_doi: "10.48550/arXiv.<id>"   # optional, paper posts only: the citation count in the meta panel
seo_title: "<shorter title>"       # optional: for <title> when the title is too long
description: >-                    # optional, plain text; see below
  <search snippet, ~150 chars>
categories: ["<Type>", "<Topic>"]
tags: ["<Tag-1>", "<Tag-2>"]
cover: /assets/images/<topic>.(jpg|png)   # the type's or topic's image; see below
use_math: true                     # only if the post has equations (loads MathJax)
lang: ko                           # required with translation_id (ko | en)
translation_id: <shared-slug>      # on both posts of a ko/en pair
---
```

- **`subtitle`** renders under the title. On a paper post it is the paper's
  title in Korean (`Gemma 3 기술 보고서`), and must differ from `title`.
- **No H1 in the body.** The layout renders the title; a leading `# Title`
  makes two `<h1>`s and leaks into the search snippet.
- **No `author:`.** It would read as this post's author, which is always the
  site's. The paper's organisation goes in `paper_author` and shows as
  "Paper by …".
- **`<title>`.** A paper post's `<title>` is its title (or `seo_title`) plus
  "논문 리뷰" / "논문 요약", the words a Korean reader searches with. English
  twins and social cards keep the plain title.
- **`cover`** is not free: see [covers](#categories-urls-and-covers).

## Dates

The file name's date is the day the post went up. `date:` is that day too,
except on paper posts, where it is the paper's publication date.

`date:` builds the URL, so it never changes once published. Everything else
goes by the posting day: `_plugins/posted_date.rb` sets `posted_at` (the
file-name day when it is later than `date:`, else `date:`), which orders the
home page, section pages, `/tags/`, `/categories/`, search results,
previous/next links and the feed, and is the "Published" date on the page, in
the feed and in the structured data. A paper post also shows its `date:` as
"Paper".

Posts that went up on one day keep `date:` order among themselves. A paper
post's `posted_at` is midnight UTC of its file-name day, so on a shared day it
sorts below a guide or Insights post that has a time.

## The search snippet: description

`description:` is what Google shows under the link, what social cards quote,
and the feed's `<summary>`. Omit it and `_plugins/post_description.rb` derives
one from the first prose paragraph, which is usually fine. Write it by hand
when that paragraph is a pull quote or a disclosure note (most Insights posts).

It must be unique across the site and longer than the page title; the build
fails otherwise. It ships as plain text, so the lint rejects `$$`, `*emphasis*`,
backticks, links and tags in it.

## Categories, URLs and covers

Categories are two levels:

- **`categories[0]`, the type:** `Paper Reviews`, `Paper Summaries`,
  `Paper Surveys`, `Tech Guides` or `Insights`. It picks the nav tab. The Paper
  Reviews page groups its posts by topic; the others list them flat and show an
  empty-state line while they have none.
- **`categories[1]`, the topic:** one of `topics` in
  [`_data/post_conventions.yml`](../_data/post_conventions.yml). The lint
  rejects anything else.

Jekyll slugifies both and adds the date to build the path:

```
categories: ["Paper Reviews", "Language-Models"] + date: 2025-01-23
        ↓
/paper-reviews/language-models/2025/01/23/<slug>.html
```

So **changing a published post's categories or date changes its URL** and
breaks inbound links and search results. If it cannot be avoided, add the old
URL to the post's `redirect_from`; the build fails when a recorded URL is no
longer built.

**Covers.** The cover is fixed by the conventions: the type's image when it has
one (`insights.jpg` for Insights), else the topic's. To add a topic:

1. Add it under `topics` in `_data/post_conventions.yml` with its image path
   (list the change in the PR; scholar-lens keeps a copy of this file).
2. Put the image in `assets/images/`, exactly 1920px wide, and run
   `script/cover-variants.sh assets/images/<name>.jpg` (macOS: it uses `sips`
   and `jpegtran`) for the 960 and 1280px copies smaller screens get. A test
   fails without them or when the original is not 1920px wide.
3. Credit it in `_data/covers.yml` (title, credit, url). The page prints the
   credit under the hero, and the build fails without it.

## Tags

Tags are free-form and hyphenated, and come in two layers:

- **Specific tags** name the post's own contribution
  (`Fine-Grained-Expert-Segmentation`). Precise, but they connect nothing.
- **At least one controlled topic tag** from `tags.controlled` in
  `_data/post_conventions.yml`. "Related reading" needs a shared tag, so a post
  with only specific tags ships without it.

Add the topic tag; don't swap it for the specific ones. Reuse an existing tag
before coining one: three names for one idea leave every post with a tag no
other post shares. When two tags turn out to be one, keep one and move the
other to `tags.aliases`; the lint then rejects the retired name. Renaming a tag
changes its `/tags/#…` anchor.

When a post shows no "Related reading", fix its tags, not the ranking in
`_plugins/related_posts.rb`: loosened to match on categories alone, it would
recommend unrelated posts that merely share a topic.

## Translations

An English twin lives at `<slug>-en.md` with `lang: en`; the Korean post gets
`lang: ko`, and both get the same `translation_id`: the Korean post's slug (its
file name without the date). The lint checks the pair and the name.
Only the Korean post appears in listings; the twin is linked from its language
switcher, the sitemap, the feed and search, and `hreflang` points search engines
at each.

## Math

Write `$$…$$` for inline and display math alike, and set `use_math: true`.

- **Never single `$…$`.** kramdown does not treat it as math, so its Markdown
  pass turns `_` and `*` inside into `<em>`/`<strong>` before MathJax runs:
  `$a*b*c$` becomes `$a<em>b</em>c$`. With `$$`, kramdown passes the contents
  through untouched. Prose dollar signs (`$10M`) are fine; inside math, write a
  literal dollar as `\$`.
- Display `$$` lines need a blank line before and after, in lists too.
- Inside `$$`, write `_` unescaped. In a table, write `\lvert … \rvert` (or
  `\mid`) for a bar and `\lVert … \rVert` for a norm: kramdown splits a cell at
  `|`, even `\|`.
- Only the macros in `_includes/head.html` exist (`math.defined_macros` in the
  conventions lists them).

## Images and diagrams

- Images and diagrams go flat in `assets/images/`, referenced by absolute path
  (`/assets/images/<name>.png`). Some older posts use `assets/<slug>/`.
- Every image needs real alt text; the build rejects an empty one. The alt also
  becomes the lightbox caption.
- A figure from someone else needs a visible credit line under it, with the
  license and a link when the license asks for one (CC BY does). When
  hotlinking an arXiv figure, prefer a pinned version (`arxiv.org/html/<id>v1/…`):
  a revision can renumber the figures.
- Diagrams are drawn in Excalidraw (`.excalidraw` beside the PNG; the sources
  are excluded from the site). `assets/images/render-diagrams.sh` renders them.
  Write ASCII arrows (`->`) in diagram text: Excalifont has no `→`, and one such
  glyph sets the whole label in a fallback font. If labels still come out in a
  plain font, the script's renderer has not found Excalifont (see its header):
  export with `excalidraw-to-png <name>.excalidraw out.svg --svg`, which keeps
  the font embedded, and screenshot the SVG in Chrome at 2x. `docs/diagrams/`
  are rendered that way; the script covers `assets/images/` only.

## What the lint catches

`ruby script/lint-posts.rb` names the line and the fix for each finding. Besides
the front-matter rules above, the ones easiest to trip are each a page that
renders wrong with every other gate green:

- `{{ … }}` or `{% … %}` in the body (a prompt template, GitHub Actions'
  `${{ … }}`): posts are rendered as Liquid first. Wrap it in
  `{% raw %}…{% endraw %}`.
- A model token such as `<think>` or `<s>` outside code, in the body, title or
  subtitle: the browser drops it as an unknown tag. Write it in `code`.
- Math written `\\(…\\)`, TeX beside a single `$`, `\_` inside `$$`, a display
  `$$` without blank lines, or `$$` without `use_math: true`.
- A code fence with more than a language after the marker, or one never closed.
- A file name that does not start with a date (`YYYY-MM-DD-`), or a `date:` after it.
- A hand-written `<a class="glightbox">` whose `href` is not its image's `src`:
  the lightbox opens the page in an iframe the site refuses.
- An ar5iv link: ar5iv answers a lost figure with a placeholder and HTTP 200.

Two prose rules are style, matching the ones scholar-lens holds its posts to: no
translationese without a natural reading ("보여지는", "에 있어서는",
"학습함에 있어"), and no sentence ending in a colon ("다음과 같습니다:").
Headings, quotations, tables, alt text, code and the reference list are exempt.

The full rule list is the script and `test/test_lint_posts.rb`; each rule's
comment names the defect it prevents.
