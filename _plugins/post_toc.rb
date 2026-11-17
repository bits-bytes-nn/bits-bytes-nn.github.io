# frozen_string_literal: true
#
# Liquid filter: {{ content | post_toc }} → the post's table of contents, one
# { "id", "text", "num", "level" } per <h2 id> in the rendered body. A post
# with fewer than MIN h2s but sections inside them gets its h3s too, in order
# (level 3), so a long post split into two parts still has a contents list.
# _layouts/post.html shows the list when it has MIN entries.
#
# Built at build time, so the list is in the first paint (filled by a script it
# would push the article down) and there without JavaScript.
#
# `text` stays HTML (tags stripped, entities kept), so a heading that names a
# tag ("&lt;think&gt; 토큰") renders as written. `num` is the item's number, or
# nil when most headings number themselves ("1. 배경"), which would otherwise
# read "2 1. 배경". A lone "1. …" among plain headings keeps the numbers.
module PostTocFilter
  HEADING = %r{<h([23])\b[^>]*\sid="([^"]+)"[^>]*>(.*?)</h\1>}m.freeze
  SELF_NUMBERED = /\A\s*\d+[.)]\s/.freeze
  MIN = 3

  def post_toc(html)
    heads = html.to_s.scan(HEADING).map { |level, id, inner| [level.to_i, id, inner.gsub(/<[^>]+>/, "").strip] }
    top = heads.select { |level, _, _| level == 2 }
    heads = top if top.size >= MIN
    # An h3 before the first h2 has no section to sit under: list it as one.
    first_h2 = heads.index { |level, _, _| level == 2 } || heads.size
    heads = heads.each_with_index.map { |(level, id, text), i| [i < first_h2 ? 2 : level, id, text] }
    # Numbers count h2s only: with h3s among them, "1 … 14" would number sections
    # and their parts alike, so the list goes unnumbered and the h3s indented.
    unnumbered = heads.count { |_, _, text| text.match?(SELF_NUMBERED) } > heads.size / 2.0 ||
                 heads.any? { |level, _, _| level == 3 }
    heads.each_with_index.map do |(level, id, text), i|
      { "id" => id, "text" => text, "num" => (unnumbered ? nil : i + 1), "level" => level }
    end
  end
end

# Guarded so test/ can require this file without Liquid loaded.
Liquid::Template.register_filter(PostTocFilter) if defined?(Liquid::Template)
