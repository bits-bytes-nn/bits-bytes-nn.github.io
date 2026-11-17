# frozen_string_literal: true
#
# Unit tests for script/fetch-citations.rb, with the network, the sleep and the
# clock replaced: it runs before every deploy, and must neither fail a build
# for the API's sake nor keep retrying past its step's timeout.

require "minitest/autorun"
require "stringio"
require_relative "../script/fetch-citations"

class TestFetchCitations < Minitest::Test
  def setup
    @now = 0
    @asked = []
    CitationFetch.clock = -> { @now }
    CitationFetch.pause = ->(s) { @now += s }
    CitationFetch.http = ->(*) { raise "no network in tests" }
    @stderr, $stderr = $stderr, StringIO.new # the retry log
  end

  def teardown
    $stderr = @stderr
    CitationFetch.http = CitationFetch.pause = CitationFetch.clock = nil
  end

  def response(code, body = "", headers = {})
    r = Net::HTTPResponse::CODE_TO_OBJ[code.to_s].new("1.1", code.to_s, "")
    r.instance_variable_set(:@read, true)
    r.instance_variable_set(:@body, body)
    headers.each { |k, v| r[k] = v }
    r
  end

  # Answers with `answers` in turn, each taking `took` seconds.
  def answering(*answers, took: 0)
    CitationFetch.http = lambda do |_uri, req|
      @asked << req
      @now += took
      a = answers.shift
      a.is_a?(Exception) ? raise(a) : a
    end
  end

  def test_a_success_returns_the_body_and_sends_no_key_by_default
    answering(response(200, "[]"))
    assert_equal "[]", CitationFetch.fetch(%w[arXiv:1])
    assert_nil @asked.first["x-api-key"]
    assert_equal({ "ids" => %w[arXiv:1] }, JSON.parse(@asked.first.body))
  end

  def test_429s_and_network_errors_are_retried_with_backoff
    answering(response(429), SocketError.new("x"), response(503), response(200, "[]"))
    assert_equal "[]", CitationFetch.fetch(%w[arXiv:1])
    assert_equal 4, @asked.size
    assert_equal 35, @now # 5 + 10 + 20
  end

  def test_a_short_retry_after_does_not_shorten_a_wait
    answering(response(429, "", "Retry-After" => "1"), response(200, "[]"))
    CitationFetch.fetch(%w[arXiv:1])
    assert_equal 5, @now
  end

  def test_retry_after_lengthens_a_wait_up_to_the_longest
    answering(response(429, "", "Retry-After" => "30"), response(429, "", "Retry-After" => "999"), response(200, "[]"))
    CitationFetch.fetch(%w[arXiv:1])
    assert_equal 30 + 40, @now
  end

  def test_it_gives_up_after_the_last_try
    answering(*Array.new(5) { response(429) })
    assert_nil CitationFetch.fetch(%w[arXiv:1])
    assert_equal 5, @asked.size
  end

  def test_it_stops_before_the_deadline
    answering(*Array.new(5) { response(429) }, took: 60)
    assert_nil CitationFetch.fetch(%w[arXiv:1])
    assert_operator @now, :<=, CitationFetch::DEADLINE + 60
    assert_operator @asked.size, :<, 5
  end

  def test_another_4xx_is_not_retried
    answering(response(400))
    assert_nil CitationFetch.fetch(%w[arXiv:1])
    assert_equal 1, @asked.size
  end

  def test_a_refused_key_raises_but_a_403_without_one_does_not
    answering(response(403))
    assert_raises(CitationFetch::KeyRefused) { CitationFetch.fetch(%w[arXiv:1], key: "k") }
    assert_equal "k", @asked.first["x-api-key"]
    answering(response(401))
    assert_raises(CitationFetch::KeyRefused) { CitationFetch.fetch(%w[arXiv:1], key: "k") }
    answering(response(403))
    assert_nil CitationFetch.fetch(%w[arXiv:1])
  end

  def test_published_counts_are_taken_only_in_the_shape_written
    good = { "fetched_at" => "2026-11-17T00:00:00Z", "papers" => {} }
    answering(response(200, JSON.generate(good)))
    assert_equal good, CitationFetch.published("https://example.org/citations.json")
    ["null", "[]", '{"papers":{}}', "not json"].each do |body|
      answering(response(200, body))
      assert_nil CitationFetch.published("https://example.org/citations.json"), body
    end
    answering(response(404))
    assert_nil CitationFetch.published("https://example.org/citations.json")
    answering(SocketError.new("x"))
    assert_nil CitationFetch.published("https://example.org/citations.json")
  end
end
