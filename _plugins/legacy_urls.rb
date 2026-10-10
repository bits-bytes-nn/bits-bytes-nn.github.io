# frozen_string_literal: true
#
# Keeps each post's pre-slug URL working.
#
# Post URLs were built from plain `:categories`, which keeps spaces, so "Paper
# Reviews" posts were published, indexed and shared under /paper%20reviews/….
# The permalink in _config.yml now slugifies them. For every post whose URL
# changed, this adds the old path to `redirect_from`, and jekyll-redirect-from
# writes a stub there that redirects to the new URL and names it as canonical.
module LegacyUrls
  TEMPLATE = "/:categories/:year/:month/:day/:title:output_ext"

  # `legacy` and `current` are unescaped paths. Returns the post's redirect_from
  # list with `legacy` added when it differs from `current`; entries the post
  # declares itself are kept.
  def self.redirect_from(existing, legacy, current)
    paths = Array(existing)
    return paths if legacy == current

    (paths + [legacy]).uniq
  end
end

# Guarded so test/ can require this file without Jekyll loaded. :post_read runs
# before the generators, so jekyll-redirect-from sees the added paths.
if defined?(Jekyll::Hooks)
  Jekyll::Hooks.register :site, :post_read do |site|
    site.posts.docs.each do |post|
      legacy = Jekyll::URL.new(template: LegacyUrls::TEMPLATE, placeholders: post.url_placeholders).to_s
      post.data["redirect_from"] = LegacyUrls.redirect_from(
        post.data["redirect_from"],
        Jekyll::URL.unescape_path(legacy),
        Jekyll::URL.unescape_path(post.url)
      )
    end
  end
end
