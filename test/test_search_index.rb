# frozen_string_literal: true
#
# Unit tests for _plugins/search_index.rb.
#
# This text is both what the client-side search matches against and what the
# result card shows, so a mistake here is either a term that cannot be found or
# markup printed at the reader.

require "minitest/autorun"
require_relative "../_plugins/search_index"

class TestPlainText < Minitest::Test
  def test_unwraps_tags
    assert_equal "본문입니다", SearchIndex.plain_text("<p>본문입니다</p>")
  end

  # The reason this filter exists at all: `strip_html` left entities encoded and
  # the search page printed "음료 &gt; 탄산음료" to the reader.
  def test_decodes_html_entities
    assert_equal "음료 > 탄산음료", SearchIndex.plain_text("<p>음료 &gt; 탄산음료</p>")
    assert_equal "a & b", SearchIndex.plain_text("a &amp; b")
    assert_equal %(그는 "말했다"), SearchIndex.plain_text("그는 &quot;말했다&quot;")
    assert_equal "it's", SearchIndex.plain_text("it&#39;s")
  end

  # Order is load-bearing. Decoding before stripping would turn this text into a
  # real element and the strip would delete the words the author wrote.
  def test_entities_that_spell_a_tag_survive_as_text
    assert_equal "<script>alert(1)</script> 는 위험합니다",
                 SearchIndex.plain_text("<p>&lt;script&gt;alert(1)&lt;/script&gt; 는 위험합니다</p>")
  end

  # The second defect: adjacent blocks indexed as one unsearchable token.
  def test_block_boundaries_become_a_space
    assert_equal "끝 시작", SearchIndex.plain_text("<p>끝</p><p>시작</p>")
    assert_equal "하나 둘", SearchIndex.plain_text("<li>하나</li><li>둘</li>")
    assert_equal "위 아래", SearchIndex.plain_text("위<br>아래")
    assert_equal "머리 값", SearchIndex.plain_text("<tr><th>머리</th><td>값</td></tr>")
  end

  # …but an inline tag inside a word must not split it, or the word stops matching.
  def test_inline_tags_do_not_split_a_word
    assert_equal "강조된 문장", SearchIndex.plain_text("<em>강조</em>된 문장")
    assert_equal "attention", SearchIndex.plain_text("<strong>atten</strong><code>tion</code>")
  end

  def test_drops_script_and_style_content_entirely
    assert_equal "본문", SearchIndex.plain_text("<script>var x = 1;</script><p>본문</p>")
    assert_equal "본문", SearchIndex.plain_text("<style>.a{color:red}</style><p>본문</p>")
  end

  def test_drops_comments
    assert_equal "본문", SearchIndex.plain_text("<!-- 숨김 --><p>본문</p>")
  end

  def test_collapses_all_whitespace_including_newlines
    assert_equal "한 줄로", SearchIndex.plain_text("한\n\n  줄로\t")
  end

  def test_handles_nil_and_empty_input
    assert_equal "", SearchIndex.plain_text(nil)
    assert_equal "", SearchIndex.plain_text("")
    assert_equal "", SearchIndex.plain_text("   \n ")
  end

  # Korean prose has no inter-word spaces to fall back on, so nothing may be
  # dropped on the assumption that a space is nearby.
  def test_preserves_korean_text_without_spaces
    long = "맥락을새겨넣는법" * 20
    assert_equal long, SearchIndex.plain_text("<p>#{long}</p>")
  end

  def test_is_idempotent_on_already_plain_text
    plain = "이미 평문입니다 & 그대로"
    assert_equal plain, SearchIndex.plain_text(plain)
  end
end

class TestMathDelimiters < Minitest::Test
  # kramdown leaves these in the HTML as text, so an excerpt landing on a formula
  # used to show the reader `\(O(L^2)\)`.
  def test_strips_inline_delimiters_and_keeps_the_tex
    assert_equal "메커니즘의 O(L^2) 계산 복잡도",
                 SearchIndex.plain_text("<p>메커니즘의 \\(O(L^2)\\) 계산 복잡도</p>")
  end

  def test_strips_display_delimiters
    assert_equal "앞 E = mc^2 뒤", SearchIndex.plain_text("앞 \\[E = mc^2\\] 뒤")
  end

  # Inline math gets no padding: Korean attaches a particle straight onto the
  # symbol, and a space here would break the word in two.
  def test_inline_math_does_not_gain_a_space_before_a_korean_particle
    assert_equal "비율 \\theta로 나눕니다",
                 SearchIndex.plain_text("비율 \\(\\theta\\)로 나눕니다")
  end

  def test_handles_several_formulas_in_one_paragraph
    assert_equal "a x b y c",
                 SearchIndex.plain_text("<p>a \\(x\\) b \\(y\\) c</p>")
  end

  # The symbol stays searchable — that is why the TeX is kept rather than dropped.
  def test_the_contents_remain_findable
    assert_includes SearchIndex.plain_text("<p>복잡도는 \\(O(L^2)\\)입니다</p>"), "O(L^2)"
  end

  # An opening delimiter with no partner is left alone rather than eating the rest
  # of the post.
  def test_an_unclosed_delimiter_is_left_as_is
    assert_equal "열린 \\( 그리고 나머지 본문",
                 SearchIndex.plain_text("<p>열린 \\( 그리고 나머지 본문</p>")
  end

  def test_a_lone_backslash_paren_in_prose_is_untouched
    assert_equal "함수 f(x) 는 그대로", SearchIndex.plain_text("<p>함수 f(x) 는 그대로</p>")
  end
end
