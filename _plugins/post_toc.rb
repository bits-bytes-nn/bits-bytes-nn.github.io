# frozen_string_literal: true
#
# Liquid filter: {{ content | post_toc }} → the post's table of contents, one
# { "id", "text", "num" } per <h2 id> in the rendered body.
#
# Built here, not in the browser: js/main.js used to fill the list on
# DOMContentLoaded, which came after the first paint once the fonts stopped
# blocking it, so the TOC opened and pushed the article down. It also left
# readers without JavaScript no TOC at all.
#
# `text` stays HTML (tags stripped, entities kept), so a heading that names a
# tag ("&lt;think&gt; 토큰") renders as written. `num` is the item's number, or
# nil when most headings number themselves ("1. 배경"), which would otherwise
# read "2 1. 배경". A lone "1. …" among plain headings keeps the numbers.
module PostTocFilter
  H2 = %r{<h2\b[^>]*\sid="([^"]+)"[^>]*>(.*?)</h2>}m.freeze
  SELF_NUMBERED = /\A\s*\d+[.)]\s/.freeze

  def post_toc(html)
    heads = html.to_s.scan(H2).map { |id, inner| [id, inner.gsub(/<[^>]+>/, "").strip] }
    numbered = heads.count { |_, text| text.match?(SELF_NUMBERED) } > heads.size / 2.0
    heads.each_with_index.map do |(id, text), i|
      { "id" => id, "text" => text, "num" => (numbered ? nil : i + 1) }
    end
  end
end

# Guarded so test/ can require this file without Liquid loaded.
Liquid::Template.register_filter(PostTocFilter) if defined?(Liquid::Template)
