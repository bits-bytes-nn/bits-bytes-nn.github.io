source 'https://rubygems.org'

# Matches CI (.github/workflows/jekyll.yml) and .ruby-version. 3.3+ is required
# by html-proofer's async→console dependency. Lower bound only, so newer local
# Rubies (3.4) still resolve.
ruby '>= 3.3'

gem 'jekyll', '~> 4.3'
# Pin to the libsass-based converter (2.x). The 3.x line switched to dart-sass,
# which errors on the vendored Bourbon/Neat frameworks' legacy `/` division.
gem 'jekyll-sass-converter', '~> 2.0'
gem 'jekyll-paginate'
gem 'jekyll-sitemap'
gem 'jekyll-redirect-from'
gem 'kramdown-parser-gfm'

gem 'base64'
gem 'bigdecimal'
gem 'logger'
gem 'webrick'

# _plugins/safe_post_html.rb parses each post body with its HTML5 parser.
gem 'nokogiri', '>= 1.12'

# CI quality gate: validates internal links, images, and HTML structure of _site/
gem 'html-proofer', '~> 5.2'

# test/run_all.rb, run under the bundle like the build.
gem 'minitest', '~> 6.1', group: :test
