# frozen_string_literal: true
#
# Unit tests for _plugins/post_toc.rb.

require "minitest/autorun"
require_relative "../_plugins/post_toc"

class TestPostToc < Minitest::Test
  def toc(html)
    Object.new.extend(PostTocFilter).post_toc(html)
  end

  def test_one_entry_per_h2_with_an_id
    html = %(<h2 id="a">One</h2><p>x</p><h3 id="b">Sub</h3><h2>No id</h2><h2 data-id="d">Data</h2><h2 id="c" class="k">Two</h2><h2 id="e">Three</h2>)
    assert_equal [{ "id" => "a", "text" => "One", "num" => 1, "level" => 2 },
                  { "id" => "c", "text" => "Two", "num" => 2, "level" => 2 },
                  { "id" => "e", "text" => "Three", "num" => 3, "level" => 2 }], toc(html)
  end

  # RAPTOR, LLaMA and GraphRAG: two h2s, each with many h3s.
  def test_too_few_h2s_bring_in_the_h3s_unnumbered
    html = %(<h2 id="a">Part 1</h2><h3 id="a1">One</h3><h3 id="a2">Two</h3><h2 id="b">Part 2</h2><h3 id="b1">Three</h3><h4 id="x">No</h4>)
    entries = toc(html)
    assert_equal %w[a a1 a2 b b1], entries.map { |e| e["id"] }
    assert_equal [2, 3, 3, 2, 3], entries.map { |e| e["level"] }
    assert(entries.all? { |e| e["num"].nil? })
  end

  def test_two_h2s_and_no_h3s_stay_too_short_to_show
    assert_equal 2, toc(%(<h2 id="a">A</h2><h2 id="b">B</h2>)).size
  end

  def test_tags_are_stripped_and_entities_kept
    assert_equal "The &lt;think&gt; token", toc(%(<h2 id="t">The <code>&lt;think&gt;</code> token</h2>)).first["text"]
  end

  def test_self_numbered_headings_get_no_number
    assert_nil toc(%(<h2 id="a">1. 배경</h2><h2 id="b">2. 방법</h2><h2 id="c">결론</h2>)).first["num"]
    assert_equal 1, toc(%(<h2 id="a">배경</h2><h2 id="b">방법</h2><h2 id="c">결론</h2>)).first["num"]
    assert_equal 1, toc(%(<h2 id="a">1. 배경</h2><h2 id="b">방법</h2><h2 id="c">결론</h2>)).first["num"]
  end

  def test_no_headings
    assert_equal [], toc("<p>Text</p>")
  end
end
