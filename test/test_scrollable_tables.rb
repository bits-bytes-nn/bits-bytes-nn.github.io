# frozen_string_literal: true
#
# Unit tests for _plugins/scrollable_tables.rb.
#
# This runs over every rendered post and page, so a regex that is too greedy would
# swallow content between two tables, and one that is too narrow would leave wide
# tables unscrollable.

require "minitest/autorun"
require_relative "../_plugins/scrollable_tables"

class TestScrollableTables < Minitest::Test
  def test_wraps_a_table
    assert_equal '<div class="table-container"><table><tr><td>a</td></tr></table></div>',
                 ScrollableTables.apply("<table><tr><td>a</td></tr></table>")
  end

  def test_preserves_attributes_on_the_table
    html = '<table class="x" id="y"><tr><td>a</td></tr></table>'
    result = ScrollableTables.apply(html)
    assert_includes result, '<table class="x" id="y">'
    assert result.start_with?('<div class="table-container">')
  end

  # The greedy-regex trap: one wrapper must not span from the first <table> to the
  # last </table>, swallowing the prose in between.
  def test_wraps_each_table_separately
    html = "<table><tr><td>1</td></tr></table><p>between</p><table><tr><td>2</td></tr></table>"
    result = ScrollableTables.apply(html)
    assert_equal 2, result.scan('<div class="table-container">').length
    assert_includes result, "</table></div><p>between</p><div class=\"table-container\"><table>"
  end

  def test_handles_a_table_spanning_many_lines
    html = <<~HTML
      <table>
        <thead><tr><th>h</th></tr></thead>
        <tbody><tr><td>d</td></tr></tbody>
      </table>
    HTML
    result = ScrollableTables.apply(html)
    assert_equal 1, result.scan('<div class="table-container">').length
    assert_includes result, "<thead>"
  end

  def test_leaves_content_without_tables_untouched
    html = "<p>표가 없는 문단입니다.</p>"
    assert_equal html, ScrollableTables.apply(html)
  end

  def test_handles_nil_and_empty_input
    assert_equal "", ScrollableTables.apply(nil)
    assert_equal "", ScrollableTables.apply("")
  end

  def test_wrapped_detects_an_already_processed_document
    once = ScrollableTables.apply("<table><tr><td>a</td></tr></table>")
    assert ScrollableTables.wrapped?(once)
    refute ScrollableTables.wrapped?("<table><tr><td>a</td></tr></table>")
  end

  # The hook skips already-wrapped documents; this pins that a second pass would
  # otherwise nest, so the guard is load-bearing rather than decorative.
  def test_a_second_pass_would_nest_which_is_why_the_hook_guards
    once = ScrollableTables.apply("<table><tr><td>a</td></tr></table>")
    twice = ScrollableTables.apply(once)
    assert_equal 2, twice.scan('<div class="table-container">').length
  end
end
