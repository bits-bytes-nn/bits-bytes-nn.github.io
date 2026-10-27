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

  def test_posted_at_is_the_later_file_date_or_the_date_itself
    assert_equal Time.utc(2026, 1, 16), PostedDate.posted_at("2026-01-16-constitutional-ai.md", PAPER)
    guide = Time.parse("2026-10-23 10:00:00 +0900")
    assert_equal guide, PostedDate.posted_at("2026-10-23-strands-agents.md", guide)
  end

  # Stands in for Jekyll::Document: data, and a <=> by date then path.
  Doc = Struct.new(:data, :path) do
    def <=>(other)
      cmp = data["date"] <=> other.data["date"]
      cmp.zero? ? path <=> other.path : cmp
    end
    prepend PostedDate::Order
  end

  def doc(path, date, posted = nil)
    Doc.new({ "date" => Time.parse(date), "posted_at" => posted && Time.parse(posted) }, path)
  end

  def test_posts_order_by_the_day_they_went_up
    review = doc("old-paper.md", "2022-12-15", "2026-10-16")
    guide = doc("guide.md", "2026-07-27", "2026-07-27")
    assert_equal [guide, review], [review, guide].sort
  end

  def test_a_tie_or_a_missing_posted_at_falls_back_to_date_then_path
    a = doc("a.md", "2023-01-01", "2026-01-16")
    b = doc("b.md", "2024-01-01", "2026-01-16")
    assert_equal [a, b], [b, a].sort
    page = doc("page.md", "2020-01-01")
    assert_equal [page, b], [b, page].sort
  end
end
