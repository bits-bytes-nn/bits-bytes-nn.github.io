# frozen_string_literal: true
#
# `plain_text` — turns a rendered post into the text search.json carries.
#
# Replaces `strip_html | strip_newlines`, which got two things wrong:
#
#   1. It left HTML entities encoded. kramdown escapes a literal `>` in prose to
#      `&gt;`, so the index held `"음료 &gt; 탄산음료"` and the search page printed
#      that verbatim to the reader.
#   2. It removed tags without leaving anything behind, so `<p>끝</p><p>시작</p>`
#      indexed as `끝시작` — one token that matches neither word.

require "cgi"

module SearchIndex
  # Elements whose *content* is not prose. Removed outright, not unwrapped.
  NON_PROSE = %r{<(script|style)\b[^>]*>.*?</\1>}mi
  COMMENT = /<!--.*?-->/m
  # Tags that end a run of text. Anything else is inline (`<em>`, `<code>`, `<a>`)
  # and is unwrapped with no space, so `<em>강조</em>된` stays one word.
  BOUNDARY = %r{</?(?:p|div|li|ul|ol|dl|dt|dd|h[1-6]|blockquote|pre|table|thead
                  |tbody|tr|td|th|section|article|header|footer|figure|figcaption
                  |br|hr)\b[^>]*>}xi

  def self.plain_text(html)
    text = html.to_s.gsub(NON_PROSE, " ").gsub(COMMENT, " ")
    # Strip before decoding, never after. A post that quotes markup contains
    # `&lt;script&gt;` as text; decoding first would make it a real tag and the
    # strip would then delete the words the author actually wrote.
    text = text.gsub(BOUNDARY, " ").gsub(/<[^>]*>/, "")
    CGI.unescapeHTML(text).gsub(/\s+/, " ").strip
  end
end

module SearchIndexFilter
  def plain_text(input)
    SearchIndex.plain_text(input)
  end
end

# Guarded so test/ can require this file without Liquid loaded.
Liquid::Template.register_filter(SearchIndexFilter) if defined?(Liquid::Template)
