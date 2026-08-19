# frozen_string_literal: true
#
# Adds loading="lazy" and decoding="async" to <img> tags in rendered post and
# page content, so below-the-fold images don't block initial render.
#
# Runs as a post-render hook (after kramdown produces HTML), so it covers both
# Markdown ![](...) images and hand-written <img> tags. Tags that already set a
# loading attribute are left untouched.
#
# This repo builds via GitHub Actions (not the github-pages gem sandbox), so
# custom _plugins are allowed.
module LazyImages
  # Negative lookahead on `loading=` so an explicit loading="eager" (used for an
  # above-the-fold image) survives.
  BARE_IMG = /<img\b(?![^>]*\bloading=)([^>]*)>/i

  def self.apply(html)
    html.to_s.gsub(BARE_IMG) do
      "<img loading=\"lazy\" decoding=\"async\"#{Regexp.last_match(1)}>"
    end
  end
end

# Guarded so test/ can require this file for LazyImages.apply without Jekyll
# loaded, and without registering the hook.
if defined?(Jekyll::Hooks)
  Jekyll::Hooks.register [:posts, :pages], :post_render do |doc|
    next unless doc.output_ext == ".html"

    doc.output = LazyImages.apply(doc.output)
  end
end
