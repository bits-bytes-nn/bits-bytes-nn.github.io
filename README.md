# Bits, Bytes and Neural Networks

A tech blog exploring AI/ML research through in-depth paper reviews and analysis of emerging trends. Built with Jekyll (based on the [Centrarium theme](http://jekyllthemes.org/themes/centrarium/)) and deployed to GitHub Pages.

Live at **[bits-bytes-nn.github.io](https://bits-bytes-nn.github.io/)**.

![Bits, Bytes and Neural Networks](assets/my_header_image_full.jpg)

## Quick start

```bash
bundle install                 # install gems (Jekyll 4.4, plugins, html-proofer)
bundle exec jekyll serve       # dev server with live reload → http://localhost:4000
bundle exec jekyll build       # one-off production build into _site/
```

Requires **Ruby 3.3+** (matches `.ruby-version` and CI) and Bundler. Custom
`_plugins/` are used, so the build runs plain `jekyll` (not the github-pages gem).

## Repository layout

| Path | Purpose |
|------|---------|
| `_posts/YYYY-MM-DD-slug.md` | Blog posts (Markdown + YAML front matter). Korean is primary; English translations live at `<slug>-en.md`. |
| `_layouts/` | Page templates: `default` → `post` / `page` / `archive`. |
| `_includes/` | Reusable fragments: `head`, `header`, `footer`, `nav_links`, `page_divider`, `category-posts`, `language_switcher`. |
| `_sass/` | Styles. Project partials are `_layout`, `_post`, `_tags`, `_syntax` (Rouge theme), `_dark` (dark mode), and `base/*`; `bourbon/` and `neat/` are vendored — don't edit. |
| `_plugins/` | `reading_time.rb` (KO/EN-aware read time), `lazy_images.rb` (lazy-load `<img>`). |
| `css/main.scss` | Sass entry point (imports `_sass` partials). `css/search.css` styles the search page only. |
| `js/main.js` | Site behaviors (theme toggle, code-copy, TOC, menu, image zoom…). `js/search.js` drives search. |
| `assets/images/<topic>.{jpg,png}` | Shared cover images, reused across posts by topic. |
| `assets/<post-slug>/` | Per-post figures, one folder per post. |
| `index.html`, `paper-reviews.md`, `paper-summaries.md`, `tech-guides.md`, `insights.md`, `categories.html`, `tags.html`, `search.md`, `about.md` | Top-level pages. |
| `search.json` + `js/search.js` | Client-side full-text search (simple-jekyll-search). |
| `.github/workflows/jekyll.yml` | CI: build → html-proofer link check → deploy to GitHub Pages on push to `main`. |

For architecture and internals, see **[tech-doc.md](tech-doc.md)** (Korean).

## Writing a post

Use the `/write-post` skill for the full research-to-proofread workflow, or add
a file manually as `_posts/YYYY-MM-DD-slug.md` with this front matter:

```yaml
---
layout: post
title: "<Post Title>"
date: YYYY-MM-DD HH:MM:SS
author: "<Author>"                       # omit for non-paper posts (e.g. Insights)
categories: ["<Category>", "<Subcategory>"]
tags: ["<Tag-1>", "<Tag-2>"]
cover: /assets/images/<topic>.(jpg|png)
use_math: true                           # set ONLY when the post has equations
lang: ko                                 # optional; with translation_id, links ko/en
translation_id: <shared-slug>            # optional; pairs a ko post with its -en twin
---
```

- **`categories[0]`** — one of `Paper Reviews`, `Paper Summaries`, `Tech Guides`,
  `Insights`. Each drives a dedicated nav tab.
- **`categories[1]`** — subcategory, e.g. `Language-Models`, `Multimodal-Learning`,
  `Finetuning`, `Retrieval-Augmented-Generation`, `Agentic-AI`. The two levels
  together form the output path: `_site/<category>/<subcategory>/YYYY/MM/DD/<slug>.html`.
- **Math** — write **`$$...$$`** for both inline and display math, and set
  `use_math: true` (MathJax loads only on those posts). Do **not** use single
  `$...$`: kramdown applies Markdown emphasis (`_`, `*`) inside single-`$` spans
  before MathJax runs, mangling formulas. See [tech-doc.md](tech-doc.md).

Validate locally with `bundle exec jekyll build` (or `bundle exec htmlproofer ./_site
--disable-external` to catch broken links) before pushing.

## Deployment

Pushing to `main` triggers `.github/workflows/jekyll.yml`, which builds with
`JEKYLL_ENV=production`, runs html-proofer on the output, and deploys to GitHub
Pages. No manual step required.

The root-level `google*.html` and `naver*.html` files are Search Console / Naver
ownership-verification tokens — they must ship to the site root, so they are **not**
excluded in `_config.yml`.

## License

MIT — see [LICENSE.md](LICENSE.md).
