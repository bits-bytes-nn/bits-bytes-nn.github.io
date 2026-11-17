# frozen_string_literal: true
#
# Unit tests for _plugins/citations.rb.
#
# The id decides which paper's count a post shows, and the batch answer is
# matched to the ids by position, so a slip in either puts one paper's count on
# another's post.

require "minitest/autorun"
require_relative "../_plugins/citations"

class TestCitations < Minitest::Test
  S2 = "5b6fdb2aea424d02c141da81d62b04d739d62b96"

  def test_an_arxiv_doi_is_asked_by_its_arxiv_id
    assert_equal "arXiv:2507.19457", Citations.lookup_id("10.48550/arXiv.2507.19457")
    assert_equal "arXiv:2507.19457", Citations.lookup_id("10.48550/ARXIV.2507.19457")
  end

  def test_any_other_doi_is_asked_as_a_doi
    assert_equal "DOI:10.18653/v1/2023.acl-long.1", Citations.lookup_id("10.18653/v1/2023.acl-long.1")
  end

  def test_an_override_names_the_record
    doi = "10.48550/arXiv.2412.19437"
    assert_equal S2, Citations.lookup_id(doi, { doi => S2 })
  end

  def test_overrides_must_name_a_post_doi_and_a_paper_id
    doi = "10.48550/arXiv.2412.19437"
    assert_equal({ doi => S2 }, Citations.overrides({ doi => S2 }, [doi]))
    assert_equal({}, Citations.overrides(nil, [doi]))
    assert_raises(ArgumentError) { Citations.overrides({ "10.1/gone" => S2 }, [doi]) }
    assert_raises(ArgumentError) { Citations.overrides({ doi => "arXiv:2412.19437" }, [doi]) }
    assert_raises(ArgumentError) { Citations.overrides([doi], [doi]) }
  end

  def test_a_count_is_shown_with_thousands_separators
    assert_equal "7", Citations.delimited(7)
    assert_equal "704", Citations.delimited(704)
    assert_equal "4,481", Citations.delimited(4481)
    assert_equal "1,234,567", Citations.delimited(1_234_567)
  end

  def test_a_record_with_citations_is_kept
    assert_equal({ "count" => 21_110, "paper_id" => S2 }, Citations.entry("paperId" => S2, "citationCount" => 21_110))
  end

  def test_not_found_zero_or_malformed_is_dropped
    assert_nil Citations.entry(nil)
    assert_nil Citations.entry("paperId" => S2, "citationCount" => 0)
    assert_nil Citations.entry("paperId" => S2, "citationCount" => "12")
    assert_nil Citations.entry("paperId" => "x", "citationCount" => 12)
  end

  def test_records_are_matched_to_ids_by_position
    other = "39d9c3f1cd4bd5069713e50dc7301570575fc055"
    records = [{ "paperId" => S2, "citationCount" => 3 }, nil, { "paperId" => other, "citationCount" => 7 }]
    assert_equal({ "a" => { "count" => 3, "paper_id" => S2 }, "c" => { "count" => 7, "paper_id" => other } },
                 Citations.entries(%w[a b c], records))
  end

  def test_an_answer_that_does_not_line_up_is_refused
    assert_raises(ArgumentError) { Citations.entries(%w[a b], [nil]) }
    assert_raises(ArgumentError) { Citations.entries(%w[a], { "error" => "x" }) }
  end

  Post = Struct.new(:data)

  def test_annotate_sets_the_id_and_the_fetched_count
    doi = "10.48550/arXiv.2412.19437"
    paper = Post.new({ "paper_doi" => doi })
    other = Post.new({ "paper_doi" => "10.48550/arXiv.2507.19457" })
    guide = Post.new({})
    data = { "citation_ids" => { doi => S2 },
             "citations" => { "papers" => { S2 => { "count" => 4481, "paper_id" => S2 }, "arXiv:2507.19457" => "junk" } } }
    Citations.annotate([paper, other, guide], data)
    assert_equal S2, paper.data["citation_id"]
    assert_equal({ "count" => 4481, "paper_id" => S2, "shown" => "4,481" }, paper.data["citation"])
    assert_equal "arXiv:2507.19457", other.data["citation_id"]
    assert_nil other.data["citation"] # a malformed entry is skipped, not fatal
    assert_equal({}, guide.data)
  end

  def test_annotate_without_fetched_counts_sets_only_the_id
    post = Post.new({ "paper_doi" => "10.48550/arXiv.2507.19457" })
    [{}, { "citations" => nil }, { "citations" => { "papers" => [] } }].each do |data|
      Citations.annotate([post], data)
      assert_equal "arXiv:2507.19457", post.data["citation_id"]
      assert_nil post.data["citation"]
    end
  end
end
