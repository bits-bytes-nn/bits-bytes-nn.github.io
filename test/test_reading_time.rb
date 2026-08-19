# frozen_string_literal: true
#
# Unit tests for _plugins/reading_time.rb.
#
# Jekyll's built-in number_of_words counts each CJK glyph as a word, which put a
# 22k-character Korean post at 116 minutes. This filter reads CJK at ~500 chars
# per minute and Latin at ~220 words per minute, so the two rates and the
# boundary between them are what these tests pin down.

require "minitest/autorun"
require_relative "../_plugins/reading_time"

class TestReadingTime < Minitest::Test
  def setup
    @filter = Object.new.extend(ReadingTimeFilter)
  end

  def test_korean_reads_at_500_characters_per_minute
    assert_equal 1, @filter.reading_time("가" * 500)
    assert_equal 2, @filter.reading_time("가" * 1000)
    assert_equal 4, @filter.reading_time("가" * 2000)
  end

  def test_latin_reads_at_220_words_per_minute
    assert_equal 1, @filter.reading_time((["word"] * 220).join(" "))
    assert_equal 2, @filter.reading_time((["word"] * 440).join(" "))
  end

  def test_rounds_up_rather_than_down
    # 501 CJK chars is just over a minute and must not report 1.
    assert_equal 2, @filter.reading_time("가" * 501)
  end

  def test_counts_mixed_korean_and_latin_additively
    # 500 CJK (1 min) + 220 Latin words (1 min) = 2 min.
    mixed = ("가" * 500) + " " + (["word"] * 220).join(" ")
    assert_equal 2, @filter.reading_time(mixed)
  end

  def test_ignores_html_tags
    plain = @filter.reading_time("가" * 500)
    tagged = @filter.reading_time("<p class=\"x\">" + ("가" * 500) + "</p>")
    assert_equal plain, tagged
  end

  # Tag names must not be counted as Latin words.
  def test_tag_names_do_not_inflate_the_estimate
    html = (["<span>a</span>"] * 100).join
    # 100 Latin words at 220/min rounds up to 1, not several minutes of markup.
    assert_equal 1, @filter.reading_time(html)
  end

  # 800 chars, not 400: at 400 the assertion is 1 minute either way, so the test
  # passed even with Han/Hiragana/Katakana deleted from the CJK class — those
  # characters fell through to the Latin branch, counted 0, and `[_, 1].max`
  # returned 1. 800 only reaches 2 if the script is actually recognised.
  def test_covers_the_other_cjk_scripts
    assert_equal 2, @filter.reading_time("漢" * 800)      # Han
    assert_equal 2, @filter.reading_time("ひ" * 800)      # Hiragana
    assert_equal 2, @filter.reading_time("カ" * 800)      # Katakana
  end

  def test_never_reports_less_than_one_minute
    assert_equal 1, @filter.reading_time("")
    assert_equal 1, @filter.reading_time(nil)
    assert_equal 1, @filter.reading_time("한 문장.")
  end
end
