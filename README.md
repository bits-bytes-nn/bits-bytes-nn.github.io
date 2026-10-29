<div align="center">

# Bits, Bytes and Neural Networks

**In-depth AI/ML paper reviews, summaries, and tech guides — published as a blog.**

[bits-bytes-nn.github.io](https://bits-bytes-nn.github.io) · a Jekyll static site on GitHub Pages

[![Deploy](https://github.com/bits-bytes-nn/bits-bytes-nn.github.io/actions/workflows/jekyll.yml/badge.svg)](https://github.com/bits-bytes-nn/bits-bytes-nn.github.io/actions/workflows/jekyll.yml)
![Ruby](https://img.shields.io/badge/ruby-3.3%2B-red)
![Jekyll](https://img.shields.io/badge/SSG-Jekyll-CC0000)
![GitHub Pages](https://img.shields.io/badge/hosting-GitHub%20Pages-222)

[한국어 README](./README.ko.md)

![Bits, Bytes and Neural Networks](assets/my_header_image_full.jpg)

</div>

## What's here

Posts are Korean first, some with an English twin, in five sections: Paper
Reviews, Paper Summaries, Paper Surveys, Tech Guides and Insights. Paper posts,
surveys and tech guides arrive as pull requests from
[scholar-lens](https://github.com/bits-bytes-nn/scholar-lens); Insights are
written by hand. Every pull request to `main` runs the full set of checks, and a
merge deploys.

## Quick start

1. **Ruby 3.3 or newer and Bundler.** The repo pins the version in
   `.ruby-version`, so rbenv or asdf picks it up; macOS's system Ruby is too old.
2. **Install the gems:** `bundle install`
3. **Serve with live reload** at `http://localhost:4000`:
   `bundle exec jekyll serve`. Restart it after changing `_config.yml`.
4. **Check a change the way CI does:**

   ```bash
   ruby test/run_all.rb && ruby script/lint-posts.rb
   bundle exec jekyll build --strict-front-matter --destination /tmp/site-verify
   bundle exec htmlproofer /tmp/site-verify --disable-external --allow-hash-href --no-ignore-empty-alt
   script/validate-site.sh /tmp/site-verify
   ```

## Documentation

| Read | For |
|---|---|
| [docs/writing.md](docs/writing.md) | writing a post: front matter, dates, categories and covers, tags, translations, math, images, the lint |
| [docs/operations.md](docs/operations.md) | the checks, CI and deploy, recovering a failed deploy, recording published URLs, scholar-lens, Search Console, troubleshooting |
| [docs/architecture.md](docs/architecture.md) | how the build works: pipeline, plugins, URLs and listings, head and feed, search, styles, math, validation, known limits |

## License

Code is MIT; the writing (posts and their diagrams) is CC BY 4.0. Figures and
images by others keep their own terms, credited where they appear. Details,
including the James Webb cover credits (ESA/Webb, CC BY 4.0), are in
[LICENSE](LICENSE). Built on the
[Centrarium](https://github.com/bencentra/centrarium) Jekyll theme.
