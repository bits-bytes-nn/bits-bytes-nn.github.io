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
    html = %(<h2 id="a">One</h2><p>x</p><h3 id="b">Sub</h3><h2>No id</h2><h2 data-id="d">Data</h2><h2 id="c" class="k">Two</h2>)
    assert_equal [{ "id" => "a", "text" => "One", "num" => 1 }, { "id" => "c", "text" => "Two", "num" => 2 }], toc(html)
  end

  def test_tags_are_stripped_and_entities_kept
    assert_equal "The &lt;think&gt; token", toc(%(<h2 id="t">The <code>&lt;think&gt;</code> token</h2>)).first["text"]
  end

  def test_self_numbered_headings_get_no_number
    assert_nil toc(%(<h2 id="a">1. 배경</h2><h2 id="b">2. 방법</h2><h2 id="c">결론</h2>)).first["num"]
    assert_equal 1, toc(%(<h2 id="a">1. 배경</h2><h2 id="b">방법</h2><h2 id="c">결론</h2>)).first["num"]
  end

  def test_no_headings
    assert_equal [], toc("<p>Text</p>")
  end
end
