# frozen_string_literal: true
#
# Unit tests for _plugins/posted_date.rb.

require "minitest/autorun"
require "time"
require_relative "../_plugins/posted_date"

class TestPostedDate < Minitest::Test
  PAPER = Time.parse("2022-12-15 00:00:00 UTC")

  def test_a_post_written_after_its_paper_is_modified_on_its_file_date
    assert_equal Time.utc(2026, 1, 16), PostedDate.last_modified("2026-01-16-constitutional-ai.md", PAPER)
  end

  def test_no_value_when_the_file_date_is_not_later
    assert_nil PostedDate.last_modified("2022-12-15-constitutional-ai.md", PAPER + 3600)
    assert_nil PostedDate.last_modified("2022-12-01-earlier.md", PAPER)
  end

  def test_no_value_for_a_file_name_without_a_date
    assert_nil PostedDate.last_modified("about.md", PAPER)
  end
end
