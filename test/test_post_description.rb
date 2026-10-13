# frozen_string_literal: true
#
# Unit tests for _plugins/post_description.rb.
#
# This module decides the meta description, OpenGraph/Twitter description, the
# JSON-LD description and the Atom <summary> for every post on the site, so a
# silent change in what it extracts is a site-wide content change.

require "minitest/autorun"
require_relative "../_plugins/post_description"

class TestDerive < Minitest::Test
  # Setext headings, rules and indented code are block syntax, not prose.
  def test_skips_setext_headings_rules_and_indented_code
    assert_equal "Body.", PostDescription.derive("Title line\n===\n\nBody.")
    assert_equal "Body.", PostDescription.derive("---\n\nBody.")
    assert_equal "Body.", PostDescription.derive("***\n\nBody.")
    assert_equal "Body.", PostDescription.derive("    def code(): pass\n\nBody.")
  end

  # The shape every paper post on this blog opens with. The bug this module
  # exists to fix: Jekyll's excerpt stopped at the two headings, so all 28 paper
  # posts shared one description.
  def test_skips_headings_and_returns_first_prose
    md = <<~MD
      ## TL;DR
      ### 이 연구를 시작하게 된 배경과 동기는 무엇입니까?

      대규모 언어 모델 기반 에이전트는 근본적인 한계를 마주하고 있습니다.

      두 번째 단락입니다.
    MD
    assert_equal "대규모 언어 모델 기반 에이전트는 근본적인 한계를 마주하고 있습니다.",
                 PostDescription.derive(md)
  end

  def test_skips_blockquote_pull_quotes
    md = <<~MD
      > "인용문입니다."
      > — 출처, 2026

      실제 본문은 여기서 시작합니다.
    MD
    assert_equal "실제 본문은 여기서 시작합니다.", PostDescription.derive(md)
  end

  def test_skips_list_items_and_tables
    md = <<~MD
      - 첫째 항목
      * 둘째 항목
      1. 셋째 항목

      | 열 | 값 |
      |---|---|
      | a | b |

      산문 단락입니다.
    MD
    assert_equal "산문 단락입니다.", PostDescription.derive(md)
  end

  # Prose inside a fenced block is code, and a `#` line inside one is a comment,
  # not a heading.
  def test_skips_fenced_code_blocks
    md = <<~MD
      ```bash
      # this is a shell comment, not a heading
      echo "code prose that must not be used"
      ```

      진짜 산문입니다.
    MD
    assert_equal "진짜 산문입니다.", PostDescription.derive(md)
  end

  def test_skips_tilde_fenced_code_blocks
    md = <<~MD
      ~~~python
      print("not a description")
      ~~~

      진짜 산문입니다.
    MD
    assert_equal "진짜 산문입니다.", PostDescription.derive(md)
  end

  # The "*공개(Disclosure): ...*" preamble on Insights posts is a note, not a
  # summary of the post.
  def test_skips_a_wholly_emphasised_paragraph
    md = <<~MD
      *공개(Disclosure): 필자는 AWS에 소속돼 있습니다.*

      본문 첫 단락입니다.
    MD
    assert_equal "본문 첫 단락입니다.", PostDescription.derive(md)
  end

  def test_skips_a_wholly_emphasised_note_that_names_an_identifier
    md = "*설정값은 `max_tokens`와 snake_case 이름을 따릅니다.*\n\n본문 첫 단락입니다.\n"
    assert_equal "본문 첫 단락입니다.", PostDescription.derive(md)
  end

  # Otherwise the derived description is empty and the post falls back to the
  # site description, which every such post would then share.
  def test_skips_a_paragraph_that_is_only_an_image_or_math
    md = "![구조도](/assets/a.png)\n\n$$E = mc^2$$\n\n본문 첫 단락입니다.\n"
    assert_equal "본문 첫 단락입니다.", PostDescription.derive(md)
  end

  def test_skips_indented_list_items_and_fences
    md = <<~MD
      1. 첫째
         ```python
         x = 1
         ```
        - 하위 항목입니다

      본문 첫 단락입니다.
    MD
    assert_equal "본문 첫 단락입니다.", PostDescription.derive(md)
  end

  def test_skips_kramdown_attribute_lines
    md = "{: .notice}\n본문 첫 단락입니다.\n"
    assert_equal "본문 첫 단락입니다.", PostDescription.derive(md)
  end

  # Begins and ends with emphasis, but no one emphasis spans it: this is prose.
  def test_keeps_a_paragraph_that_only_begins_and_ends_with_emphasis
    md = "**핵심**은 데이터입니다. 그래서 **중요합니다**\n"
    assert_equal "핵심은 데이터입니다. 그래서 중요합니다", PostDescription.derive(md)
  end

  def test_keeps_underscores_inside_identifiers
    md = "설정은 `max_new_tokens`와 `top_p`를 씁니다. _강조_ 표현도 있습니다.\n"
    assert_equal "설정은 max_new_tokens와 top_p를 씁니다. 강조 표현도 있습니다.", PostDescription.derive(md)
  end

  def test_keeps_a_paragraph_with_inline_emphasis
    md = "맥락은 **중요합니다**. 그래서 이 글을 씁니다.\n"
    assert_equal "맥락은 중요합니다. 그래서 이 글을 씁니다.", PostDescription.derive(md)
  end

  def test_strips_links_to_their_text
    md = "정확도가 [Sequeda et al., 2023](https://arxiv.org/abs/2311.07509)에서 올랐습니다.\n"
    assert_equal "정확도가 Sequeda et al., 2023에서 올랐습니다.", PostDescription.derive(md)
  end

  def test_strips_images_entirely
    md = "![다이어그램](/assets/images/x.png) 설명이 이어집니다.\n"
    assert_equal "설명이 이어집니다.", PostDescription.derive(md)
  end

  def test_strips_code_spans_footnotes_and_inline_html
    md = "`text-to-SQL` 정확도[^1]는 <em>16.7%</em>였습니다.\n"
    assert_equal "text-to-SQL 정확도는 16.7%였습니다.", PostDescription.derive(md)
  end

  # Math that reads as text stays: dropping it left "비용은 이고". Math with a
  # TeX command, braces, or a character later passes read as markup (* _ < >)
  # still goes; a post whose first paragraph leans on it needs a hand-written
  # description.
  def test_keeps_inline_math_that_reads_as_text
    md = "비용은 $$O(n^2)$$이고 $$k$$에 비례합니다.\n"
    assert_equal "비용은 O(n^2)이고 k에 비례합니다.", PostDescription.derive(md)
  end

  def test_comments_are_not_prose
    md = "<!--\nTODO: 서론을 다시 쓸 것\n-->\n\n{::comment}\n메모\n{:/comment}\n\n본문입니다.\n"
    assert_equal "본문입니다.", PostDescription.derive(md)
    assert_equal "템플릿은 {{ q }} 자리입니다.", PostDescription.derive("템플릿은 {% raw %}{{ q }}{% endraw %} 자리입니다.\n")
  end

  def test_code_spans_keep_their_text
    assert_equal "추론은 <think> 안에, a\\*b는 그대로.", PostDescription.derive("추론은 `<think>` 안에, `a\\*b`는 그대로.\n")
  end

  def test_backslash_escapes_are_text
    assert_equal "추론은 <think> 태그 안, 비용은 $2, 이름은 a_b입니다.", PostDescription.derive("추론은 \\<think\\> 태그 안, 비용은 \\$2, 이름은 a\\_b입니다.\n")
  end

  # `\d+[.)]\s` read a Korean date opening as an ordered list and dropped it.
  def test_a_korean_date_opening_is_prose
    assert_equal "2024. 5. 14. 공개된 모델입니다.", PostDescription.derive("2024. 5. 14. 공개된 모델입니다.\n")
  end

  def test_strips_strikethrough
    assert_equal "취소된 문장입니다.", PostDescription.derive("~~취소~~된 문장입니다.\n")
  end

  def test_a_line_break_tag_is_a_space
    assert_equal "첫 줄 둘째 줄", PostDescription.derive("첫 줄<br>둘째 줄\n")
  end

  def test_a_paragraph_may_open_with_inline_html
    md = "<strong>핵심</strong>은 데이터 품질입니다.\n\n둘째 문단.\n"
    assert_equal "핵심은 데이터 품질입니다.", PostDescription.derive(md)
    assert_equal "본문.", PostDescription.derive("<div class=\"note\">공지</div>\n\n본문.\n")
    assert_equal "본문.", PostDescription.derive("<strong>핵심</strong>\n\n본문.\n")
    assert_equal "핵심은 품질이고 결론도 품질", PostDescription.derive("<strong>핵심</strong>은 품질이고 결론도 <strong>품질</strong>\n")
    assert_equal "본문.", PostDescription.derive("<small>그림 1: 구조</small>\n\n본문.\n")
    assert_equal "본문.", PostDescription.derive("<em>공개: 저자는 <a href=\"https://aws.amazon.com\">AWS</a> 직원입니다.</em>\n\n본문.\n")
  end

  def test_joins_a_multiline_paragraph_with_spaces
    md = "첫 줄이고\n둘째 줄입니다.\n"
    assert_equal "첫 줄이고 둘째 줄입니다.", PostDescription.derive(md)
  end

  # Without this a heading between two prose lines would splice them together.
  def test_does_not_stitch_a_paragraph_across_a_heading
    md = <<~MD
      먼저 이 문장이 옵니다.
      ## 중간 제목
      그리고 이 문장이 옵니다.
    MD
    assert_equal "먼저 이 문장이 옵니다.", PostDescription.derive(md)
  end

  def test_returns_empty_string_when_there_is_no_prose
    md = "# 제목만 있습니다\n\n## 다른 제목\n"
    assert_equal "", PostDescription.derive(md)
  end

  def test_returns_empty_string_for_blank_input
    assert_equal "", PostDescription.derive("")
    assert_equal "", PostDescription.derive(nil)
  end

  def test_truncates_long_prose_to_the_limit
    md = "가" * 500 + "\n"
    result = PostDescription.derive(md)
    assert_operator result.length, :<=, PostDescription::LIMIT + 1 # +1 for the ellipsis
    assert result.end_with?("…"), "expected an ellipsis, got #{result[-10..].inspect}"
  end
end

class TestTruncate < Minitest::Test
  def test_returns_text_unchanged_when_within_the_limit
    assert_equal "short", PostDescription.truncate("short", 160)
  end

  def test_returns_text_unchanged_at_exactly_the_limit
    text = "a" * 20
    assert_equal text, PostDescription.truncate(text, 20)
  end

  def test_cuts_at_a_word_boundary_for_latin_text
    text = "the quick brown fox jumps over the lazy dog again and again"
    result = PostDescription.truncate(text, 30)
    assert result.end_with?("…")
    # No partial word before the ellipsis.
    assert_includes text.split, result.delete_suffix("…").split.last
  end

  # Degenerate input: the only space sits near the start, so backing off to it
  # would return a 3-character stub instead of using the budget.
  def test_hard_cuts_when_the_only_space_is_too_early
    text = "짧게 " + "가" * 200
    result = PostDescription.truncate(text, 100)
    assert_equal 101, result.length # 100 chars + ellipsis
    assert result.end_with?("…")
  end

  def test_strips_trailing_punctuation_before_the_ellipsis
    text = "one two three, four five six seven eight nine ten"
    result = PostDescription.truncate(text, 14)
    refute_match(/[,\s]…\z/, result)
    assert result.end_with?("…")
  end

  def test_falls_back_to_the_default_limit_for_a_nonpositive_limit
    text = "a" * 400
    assert_equal PostDescription::LIMIT + 1, PostDescription.truncate(text, 0).length
    assert_equal PostDescription::LIMIT + 1, PostDescription.truncate(text, -5).length
  end
end

class TestShortenFilters < Minitest::Test
  # The Liquid filters head.html and index.html call. They take no argument, so a
  # character budget never appears in a template.
  def setup
    @filter = Object.new.extend(DescriptionFilter)
  end

  def test_meta_filter_uses_the_meta_limit
    text = "word " * 200
    assert_operator @filter.shorten_meta(text).length, :<=, PostDescription::META_LIMIT + 1
  end

  def test_card_filter_uses_the_full_limit
    text = "word " * 200
    assert_operator @filter.shorten_card(text).length, :<=, PostDescription::LIMIT + 1
    assert_operator @filter.shorten_card(text).length, :>, PostDescription::META_LIMIT
  end

  def test_filters_strip_surrounding_whitespace
    assert_equal "abc", @filter.shorten_meta("  abc  ")
    assert_equal "abc", @filter.shorten_card("  abc  ")
  end

  def test_filters_handle_nil
    assert_equal "", @filter.shorten_meta(nil)
    assert_equal "", @filter.shorten_card(nil)
  end
end
