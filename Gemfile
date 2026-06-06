source 'https://rubygems.org'

# Matches CI (.github/workflows/jekyll.yml) and .ruby-version. Lower bound only,
# so newer local Rubies (3.3/3.4) still resolve.
ruby '>= 3.2'

gem 'jekyll', '~> 4.3'
# Pin to the libsass-based converter (2.x). The 3.x line switched to dart-sass,
# which errors on the vendored Bourbon/Neat frameworks' legacy `/` division.
gem 'jekyll-sass-converter', '~> 2.0'
gem 'jekyll-paginate'
gem 'jekyll-sitemap'
gem 'jekyll-feed'
gem 'kramdown-parser-gfm'

gem 'base64'
gem 'bigdecimal'
gem 'logger'
gem 'webrick'

# CI quality gate: validates internal links, images, and HTML structure of _site/
gem 'html-proofer', '~> 5.0'
