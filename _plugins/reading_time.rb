# frozen_string_literal: true
#
# Liquid filter: {{ content | reading_time }} → estimated minutes (integer).
#
# Mixed Korean/English aware. Jekyll's built-in number_of_words counts each CJK
# glyph as a word, which wildly overcounts Korean posts (a 22k-char post came out
# at 116 min). Here CJK characters are read at ~500/min and Latin words at
# ~220/min, then summed — matching realistic reading speeds for this blog.
module ReadingTimeFilter
  CJK = /[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]/.freeze

  def reading_time(input)
    text = input.to_s.gsub(/<[^>]+>/, " ")
    cjk_chars = text.scan(CJK).length
    latin_words = text.gsub(CJK, " ").scan(/[A-Za-z0-9]+/).length
    minutes = (cjk_chars / 500.0) + (latin_words / 220.0)
    [minutes.ceil, 1].max
  end
end

Liquid::Template.register_filter(ReadingTimeFilter)
