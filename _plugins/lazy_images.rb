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
Jekyll::Hooks.register [:posts, :pages], :post_render do |doc|
  next unless doc.output_ext == ".html"

  doc.output = doc.output.gsub(/<img\b(?![^>]*\bloading=)([^>]*)>/i) do
    attrs = Regexp.last_match(1)
    "<img loading=\"lazy\" decoding=\"async\"#{attrs}>"
  end
end
