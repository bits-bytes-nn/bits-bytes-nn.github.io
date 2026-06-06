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
> plugins in `_plugins/` (read time, lazy images), which the sandboxed
> `github-pages` gem disallows. So both local builds and CI run Jekyll directly.

---

## Project structure

```
_posts/            Posts — YYYY-MM-DD-slug.md (Korean; English twin is -en.md)
_layouts/          Page templates: default → post / page / archive
_includes/         Reusable fragments: head, header, footer, nav_links,
                   page_divider, category-posts, language_switcher
_sass/             Styles: _layout, _post, _tags, _syntax (Rouge code theme),
                   _dark (dark mode), base/*
                   ⚠ bourbon/ and neat/ are vendored frameworks — don't edit
_plugins/          reading_time.rb (KO/EN-aware read time)
                   lazy_images.rb  (adds loading="lazy" to <img>)
css/               main.scss (Sass entry point) · search.css (search page only)
js/                main.js (theme toggle, code-copy, TOC, menu, image zoom…)
                   search.js (drives the search box)
assets/images/     Shared cover images, reused across posts by topic
assets/<slug>/     Per-post figures, one folder per post
search.json        Full-text search index (consumed by simple-jekyll-search)
.github/workflows/ CI: build → html-proofer link check → deploy on push to main
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
date: YYYY-MM-DD HH:MM:SS
author: "<Author>"                 # the paper's org; omit for Insights/opinion posts
categories: ["<Type>", "<Topic>"]
tags: ["<Tag-1>", "<Tag-2>"]
cover: /assets/images/<topic>.(jpg|png)
use_math: true                     # ONLY if the post has equations (loads MathJax)
lang: ko                           # optional — with translation_id below…
translation_id: <shared-slug>      # …links a Korean post to its -en twin
---
```

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
bundle exec jekyll build                              # does it build clean?
bundle exec htmlproofer ./_site --disable-external    # any broken links/images?
```

CI runs the same html-proofer check, so catching it locally saves a failed
deploy.

---

## Deployment

Pushing to `main` triggers `.github/workflows/jekyll.yml`, which:

1. builds the site with `JEKYLL_ENV=production`,
2. runs **html-proofer** over `_site/` (internal links, images, anchors), and
3. deploys to GitHub Pages.

If the workflow fails, it's almost always step 2 — open the Actions log to see
exactly which link or image is broken. No manual deploy step is needed.

> **⚠ Don't add `google*.html` / `naver*.html` to `_config.yml`'s `exclude`.**
> They're Search Console / Naver ownership-verification tokens that must ship to
> the site root. Excluding them silently breaks search indexing — the symptom is
> "Google can't read the sitemap."

---

## License

MIT — see [LICENSE](LICENSE). Built on the
[Centrarium](http://jekyllthemes.org/themes/centrarium/) Jekyll theme.
