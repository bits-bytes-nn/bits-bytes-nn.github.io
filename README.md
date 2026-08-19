<div align="center">

# 🧠 Bits, Bytes and Neural Networks

**In-depth AI/ML paper reviews, summaries, and tech guides — published as a blog.**

A Jekyll static site, deployed to GitHub Pages.

[![Deploy](https://github.com/bits-bytes-nn/bits-bytes-nn.github.io/actions/workflows/jekyll.yml/badge.svg)](https://github.com/bits-bytes-nn/bits-bytes-nn.github.io/actions/workflows/jekyll.yml)
![Ruby](https://img.shields.io/badge/ruby-3.3%2B-red)
![Jekyll](https://img.shields.io/badge/SSG-Jekyll-CC0000)
![GitHub Pages](https://img.shields.io/badge/hosting-GitHub%20Pages-222)

🇰🇷 [한국어 README](./README.ko.md)

![Bits, Bytes and Neural Networks](assets/my_header_image_full.jpg)

</div>

---

## Quick start

If you've never run a Jekyll site before, here's the whole loop.

**1. Install Ruby 3.3+ and Bundler.** Check what you have:

```bash
ruby --version     # need 3.3 or newer (see .ruby-version)
bundle --version   # ships with Ruby; if missing: gem install bundler
```

On macOS the system Ruby is old — use [rbenv](https://github.com/rbenv/rbenv) or
asdf to install 3.3+. (The repo pins the version in `.ruby-version`, so a version
manager will pick it up automatically.)

**2. Install the project's gems** (Jekyll, plugins, html-proofer):

```bash
bundle install
```

**3. Run the dev server.** It rebuilds on save and serves at
`http://localhost:4000`:

```bash
bundle exec jekyll serve
```

Edit a file under `_posts/`, `_sass/`, or `_includes/`, save, and refresh — most
changes appear immediately. (Changes to `_config.yml` need a server restart.)

**4. Build for production** (what CI does) when you want the final output in
`_site/`:

```bash
bundle exec jekyll build
```

> **Why plain `jekyll` and not `github-pages`?** This site uses custom Ruby
> plugins in `_plugins/` (read time, lazy images, post descriptions), which the
> sandboxed `github-pages` gem disallows. So both local builds and CI run Jekyll
> directly.

---

## Project structure

```
_posts/            Posts — YYYY-MM-DD-slug.md (Korean; English twin is -en.md)
_layouts/          Page templates: default → post / page / archive
_includes/         Reusable fragments: head, header, footer, nav_links,
                   page_divider, category-posts, language_switcher,
                   related_posts
_sass/             Styles: _layout, _post, _tags, _syntax (Rouge code theme),
                   _dark (dark mode), base/*
                   ⚠ bourbon/ and neat/ are vendored frameworks — don't edit
_plugins/          reading_time.rb      (KO/EN-aware read time)
                   lazy_images.rb       (adds loading="lazy" to <img>)
                   post_description.rb  (fills page.description for posts)
                   related_posts.rb     (fills page.related for posts)
css/               main.scss (Sass entry point) · search.css (search page only)
js/                main.js (theme toggle, code-copy, TOC, menu, image zoom…)
                   search.js (drives the search box)
assets/images/     Shared cover images, reused across posts by topic
assets/<slug>/     Per-post figures, one folder per post
search.json        Full-text search index (consumed by simple-jekyll-search)
test/              minitest unit tests for the _plugins/ logic
script/            validate-site.sh (post-build discoverability checks)
sitemap-index.xml  Sitemap index — the URL to submit to Search Console
.github/workflows/ CI: build → html-proofer → validate-site → deploy on push to main
```

**Top-level pages:** `index.html` (home), plus `paper-reviews.md`,
`paper-summaries.md`, `tech-guides.md`, `insights.md` (the four section tabs),
`categories.html`, `tags.html`, `search.md`, and `about.md`.

---

## Writing a post

The easiest path is the `/write-post` skill, which runs the whole
research → draft → proofread workflow. To add one by hand, create
`_posts/YYYY-MM-DD-slug.md` starting with this front matter:

```yaml
---
layout: post
title: "<Post Title>"
subtitle: "<one-line pitch>"       # optional — shown under the title in the header
date: YYYY-MM-DD HH:MM:SS
author: "<Author>"                 # the paper's org; omit for Insights/opinion posts
description: >-                    # optional — see below
  <search-snippet, ~150 chars>
categories: ["<Type>", "<Topic>"]
tags: ["<Tag-1>", "<Tag-2>"]
cover: /assets/images/<topic>.(jpg|png)
use_math: true                     # ONLY if the post has equations (loads MathJax)
lang: ko                           # optional — with translation_id below…
translation_id: <shared-slug>      # …links a Korean post to its -en twin
---
```

**Don't repeat the title as an H1 in the body.** The layout already renders it,
so a leading `# Title` produces two `<h1>`s and leaks into the search snippet.
Put a tagline in `subtitle:` instead.

### `description:` — the search snippet

`description:` is what Google shows under the link, what social cards quote, and
what the RSS `<summary>` carries. If you omit it, `_plugins/post_description.rb`
derives one from the post's first real prose paragraph, which is usually good
enough. Write it by hand when the first paragraph opens on a pull quote or a
disclosure note — that is, on most Insights posts.

Descriptions must be **unique across the site**; CI fails the build if two pages
share one.

### Categories drive the URL

Categories are **two levels**:

- `categories[0]` — the **type**: `Paper Reviews`, `Paper Summaries`,
  `Tech Guides`, or `Insights`. This decides which nav tab the post appears under.
- `categories[1]` — the **topic**: `Language-Models`, `Multimodal-Learning`,
  `Finetuning`, `Retrieval-Augmented-Generation`, `Agentic-AI`, … (add new ones
  freely).

Jekyll combines them with the date to build the output path:

```
categories: ["Paper Reviews", "Language-Models"] + date: 2025-01-23
        ↓
_site/paper reviews/language-models/2025/01/23/<slug>.html
```

### Math: always use `$$…$$`

Write `$$…$$` for both inline and display math, and set `use_math: true`.

**Never use single `$…$`.** kramdown doesn't treat single `$` as math, so its
Markdown pass turns `_`/`*` inside the span into `<em>`/`<strong>` *before*
MathJax runs — e.g. `$a*b*c$` becomes `$a<em>b</em>c$` and renders broken. With
`$$`, kramdown emits verbatim `\(…\)` and leaves the contents alone. (Prose
dollar signs like `$10M` are fine — they're not math.)

### Validate before pushing

```bash
ruby test/run_all.rb                                  # plugin logic still correct?
bundle exec jekyll build                              # does it build clean?
bundle exec htmlproofer ./_site --disable-external    # any broken links/images?
script/validate-site.sh                               # sitemap, feed, metadata, headings
```

CI runs the same three checks, so catching it locally saves a failed deploy.

`test/` covers `_plugins/` — the description derivation, the read-time estimate,
and the lazy-image rewrite. Plain `ruby`, not `bundle exec`: the plugins guard
their Jekyll/Liquid registration behind `defined?` so their logic loads
standalone, and minitest ships with Ruby. **Anything you change in `_plugins/`
changes every page on the site**, so add a case there before changing behaviour.

> **If the checks report something impossible, look for a running `jekyll serve`
> first.** It watches the tree and rewrites `_site/` behind you, it overrides
> `site.url` with `http://localhost:4000` (so every sitemap URL looks wrong), and
> it keeps the `_config.yml` it started with — so `exclude` entries added since
> then don't apply. Either stop it, or build somewhere else:
>
> ```bash
> ps aux | grep '[j]ekyll serve'
> bundle exec jekyll build --destination /tmp/site-verify
> script/validate-site.sh /tmp/site-verify
> ```

---

## Deployment

Pushing to `main` triggers `.github/workflows/jekyll.yml`, which:

1. runs **`ruby test/run_all.rb`** (the `_plugins/` unit tests),
2. builds the site with `JEKYLL_ENV=production`,
3. runs **html-proofer** over `_site/` (internal links, images, anchors),
4. runs **`script/validate-site.sh`** (sitemap/feed parse, one `h1` per page, no
   heading-level skips, per-page description and canonical, no duplicate
   descriptions or titles), and
5. deploys to GitHub Pages.

If the workflow fails, it's almost always step 3 or 4 — open the Actions log,
which names the exact link, image, or page. No manual deploy step is needed.

> **⚠ Don't add `google*.html` / `naver*.html` to `_config.yml`'s `exclude`.**
> They're Search Console / Naver ownership-verification tokens that must ship to
> the site root. Excluding them silently breaks ownership verification.

### If Search Console says it can't fetch the sitemap

Check the file first — it is usually fine:

```bash
curl -sI  https://bits-bytes-nn.github.io/sitemap.xml   # expect 200, application/xml
curl -sS  https://bits-bytes-nn.github.io/sitemap.xml | xmllint --noout -
curl -sS  https://bits-bytes-nn.github.io/robots.txt
```

If those pass, the failure is a cached Search Console verdict, not the site.
Search Console keys a sitemap by URL and keeps the first result it recorded, so
re-submitting the same path reuses the stale entry. Remove the entry and submit
**`sitemap-index.xml`** instead — a URL it has not seen before — then use URL
Inspection → Request Indexing on a couple of posts to prompt a crawl.

---

## License

MIT — see [LICENSE](LICENSE). Built on the
[Centrarium](http://jekyllthemes.org/themes/centrarium/) Jekyll theme.
