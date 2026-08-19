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

  # kramdown's mathjax engine leaves the delimiters in the HTML as literal text,
  # so a flattened post reads "메커니즘의 \(O(L^2)\) 계산 복잡도" and an excerpt landing
  # there showed the delimiters to the reader. Only the delimiters go; the TeX
  # inside stays, both because it is what the sentence is about and so that a
  # symbol inside a formula is still findable.
  INLINE_MATH = /\\\((.+?)\\\)/m
  DISPLAY_MATH = /\\\[(.+?)\\\]/m

  def self.plain_text(html)
    text = html.to_s.gsub(NON_PROSE, " ").gsub(COMMENT, " ")
    # Strip before decoding, never after. A post that quotes markup contains
    # `&lt;script&gt;` as text; decoding first would make it a real tag and the
    # strip would then delete the words the author actually wrote.
    text = text.gsub(BOUNDARY, " ").gsub(/<[^>]*>/, "")
    text = CGI.unescapeHTML(text)
    # Display math is its own block, so it may take spaces. Inline math must not:
    # Korean attaches particles directly, and `\(\theta\)로` has to stay one word.
    text = text.gsub(DISPLAY_MATH) { " #{Regexp.last_match(1)} " }
    text = text.gsub(INLINE_MATH) { Regexp.last_match(1) }
    text.gsub(/\s+/, " ").strip
  end
end

module SearchIndexFilter
  def plain_text(input)
    SearchIndex.plain_text(input)
  end
end

# Guarded so test/ can require this file without Liquid loaded.
Liquid::Template.register_filter(SearchIndexFilter) if defined?(Liquid::Template)
