# Operations

Checking a change, how it deploys, recovering when it doesn't, and keeping the
site visible in search. For writing posts see [writing.md](writing.md); for how
the build works see [architecture.md](architecture.md).

- [Checks before pushing](#checks-before-pushing)
- [CI and deploy](#ci-and-deploy)
- [When a deploy fails](#when-a-deploy-fails)
- [Published URLs](#published-urls)
- [scholar-lens](#scholar-lens)
- [Search Console](#search-console)
- [Troubleshooting](#troubleshooting)

## Checks before pushing

The gates CI runs, in its order. Build to a throwaway directory, not `_site/`
(see [Troubleshooting](#troubleshooting)):

```bash
ruby test/run_all.rb                                        # plugin logic and the lint, unit-tested
ruby script/lint-posts.rb                                   # post sources follow the conventions
bundle exec jekyll build --strict-front-matter \
  --destination /tmp/site-verify                            # builds clean
bundle exec htmlproofer /tmp/site-verify --disable-external \
  --allow-hash-href --no-ignore-empty-alt                   # internal links, images, anchors, https, alt text
script/validate-site.sh /tmp/site-verify                    # sitemap, feed, metadata, headings, plugin output
shellcheck script/*.sh assets/images/*.sh && actionlint .github/workflows/*.yml   # workflow-lint.yml
```

- `test/` covers the pure logic in `_plugins/` (one file per plugin) and
  `script/lint-posts.rb`. Plain `ruby`, not `bundle exec`: the plugins guard
  their Jekyll registrations behind `defined?`, and minitest ships with Ruby.
  **A change in `_plugins/` changes every page**, so add a test case first.
- CI's htmlproofer also passes `--swap-urls` for the Pages base path, which a
  local build does not have.
- Each check in `validate-site.sh` and `script/check-plugin-output.rb` (which it
  runs) carries a comment naming the defect it exists for; the scripts are the
  list.

## CI and deploy

| Workflow | Runs on | Does |
|---|---|---|
| `jekyll.yml` | PRs to `main`, pushes to `main`, manual | tests → lint → production build → htmlproofer → validate-site; on `main` only, deploys to GitHub Pages |
| `workflow-lint.yml` | PRs and pushes to `main`, manual | actionlint and shellcheck |
| `external-links.yml` | Mondays, manual | htmlproofer with external links on; fails (and notifies) on a dead link, blocks nothing |
| Dependabot | weekly | PRs for GitHub Actions and gems |

- The production build (`JEKYLL_ENV=production`) is the only one that loads
  Google Analytics.
- Permissions are per job: the workflow default is `contents: read`, and only
  `deploy` holds Pages write and the OIDC token. Actions are pinned to commit
  SHAs.
- `jekyll.yml` runs are grouped per ref: a new commit cancels the PR's previous
  run, and a started `main` run always finishes. `workflow-lint.yml` cancels
  any superseded run.
- `external-links.yml` ignores 403/429/999 (bot walls and rate limits) and two
  hosts that fail any scripted request while working in a browser; its comment
  says which.

## When a deploy fails

GitHub Pages keeps serving the last good deploy, so a failure leaves the live
site unchanged and `main` red.

1. Open the failed run in Actions. It is almost always htmlproofer or
   validate-site, and the log names the link, image or page.
2. Fix it in a PR; merging deploys.
3. For a failure that was not the change's fault (a runner or Pages outage),
   re-run the job, or run the workflow manually on `main` (Actions → "Build
   and deploy Jekyll site to GitHub Pages" → Run workflow).

## Published URLs

Every URL a post has been published under is in `script/published-urls.txt`,
and `validate-site.sh` fails when one is no longer built (a renamed post, a
changed date or category, a lost redirect stub). CI also checks every URL in
the live `sitemap.xml` (`PUBLISHED_SITEMAP` in `jekyll.yml`), so a post is
protected from its first deploy even if nobody records it; when the live
sitemap cannot be fetched, that part is skipped with a warning.

Record new URLs to the list after a post merges, and again after a move (the
new URL alongside the old one in `redirect_from`). `--record` reads the posts
through Jekyll, so no build is needed (the site directory argument is required
but unused), and prints what it added:

```bash
bundle exec ruby script/check-plugin-output.rb _site --record
```

## scholar-lens

scholar-lens generates paper reviews, summaries, surveys and tech guides as PRs
here, and keeps a copy of `_data/post_conventions.yml`; its CI fails when the
parsed data of the two differs (a comment-only change does not trip it).

- Changing `post_conventions.yml`: list the changed keys in the PR description,
  then update scholar-lens's `assets/post_conventions.yml` to match.
- A new `script/lint-posts.rb` rule: put the rule and its pass/fail examples in
  the PR description. A rule that is data (word lists, macros) goes in the spec.
- A MathJax macro added to or removed from `head.html`: update
  `math.defined_macros`.
- A hand fix to a generated post: see [writing.md](writing.md#where-posts-come-from).

## Search Console

The property is the URL prefix `https://bits-bytes-nn.github.io/`, verified by
the `google*.html` file at the site root (`naver*.html` does the same for
Naver). **Don't add either to `exclude` in `_config.yml`**: they must ship at
the root, and excluding them silently drops ownership (crawling and indexing
carry on). `defaults` in `_config.yml` keeps them out of the sitemap. Two sitemaps are submitted:
`sitemap.xml` and `feed.xml` (Google reads an Atom feed as a sitemap; it lists
the newest posts).

**Check a post with URL Inspection**, not with the sitemap rows:

- *URL is unknown to Google* with *No referring sitemaps detected*: Google has
  found the URL neither in a sitemap nor through a link. Use **Request
  indexing**. The quota is about ten a day; start with the posts people search
  for. Requested URLs have been indexed within days.
- *Excluded by 'noindex' tag* on a URL that should rank: a page or redirect stub
  carries `noindex`. Only pages kept out of the sitemap (search, 404) should, and
  `check-plugin-output.rb` fails a post or stub that does.
- Old `/paper%20reviews/…` URLs are redirect stubs (instant refresh plus
  canonical, no `noindex`). Google reads an instant refresh as a permanent
  redirect, so a recrawled stub should move to "Page with redirect".

**If a sitemap row says "Couldn't fetch"**, check the file first:

```bash
curl -sI  https://bits-bytes-nn.github.io/sitemap.xml   # expect 200, application/xml
curl -sS  https://bits-bytes-nn.github.io/sitemap.xml -o /tmp/s.xml && \
  ruby -rrexml/document -e 'REXML::Document.new(File.read("/tmp/s.xml")); puts "well-formed"'
curl -sS  https://bits-bytes-nn.github.io/robots.txt     # names the sitemap
```

If it passes, and URL Inspection's live test on the sitemap says "URL is
available to Google", no change to the file will help. The status means Google
has not fetched it yet: its scheduling, common on `*.github.io` properties.
Resubmit after a large URL change and rely on Request indexing meanwhile. The
only change reported to get it fetched is a custom domain verified as a Domain
property, which moves every URL and is a decision of its own.

## Troubleshooting

- **A check reports something impossible: look for a running `jekyll serve`
  first** (`ps aux | grep '[j]ekyll serve'`). It rewrites `_site/` behind you,
  forces `site.url` to `http://localhost:4000` (every sitemap URL looks wrong),
  and keeps the `_config.yml` it started with, so later `exclude` entries don't
  apply. Building to another destination avoids all three.
- **`_config.yml` changes don't show locally:** restart `jekyll serve`.
- **Sass fails to build after a gem update:** `jekyll-sass-converter` must stay
  on 2.x (libsass). 3.x (dart-sass) rejects the vendored Bourbon/Neat's old `/`
  division.
- **Search Console lost ownership:** the verification files were excluded or
  removed (see [Search Console](#search-console)).
- **Diagram text renders in a plain font:** see
  [writing.md](writing.md#images-and-diagrams).
- More build-level traps (dark-mode text, anchors, Sass in inline `<style>`) are
  in [architecture.md](architecture.md#pitfalls).
