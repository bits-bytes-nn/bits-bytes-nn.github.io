# frozen_string_literal: true
#
# Unit tests for script/fetch-citations.rb, with the network, the sleep and the
# clock replaced: it runs before every deploy, and must neither fail a build
# for the API's sake nor keep retrying past its step's timeout.

require "minitest/autorun"
require "stringio"
require "tmpdir"
require_relative "../_plugins/citations"
require_relative "../script/fetch-citations"

class TestFetchCitations < Minitest::Test
  def setup
    @now = 0
    @asked = []
    CitationFetch.clock = -> { @now }
    CitationFetch.pause = ->(s) { @now += s }
    CitationFetch.http = ->(*) { raise "no network in tests" }
    @stderr, $stderr = $stderr, StringIO.new # the retry log
    @stdout, $stdout = $stdout, StringIO.new
  end

  def teardown
    $stderr = @stderr
    $stdout = @stdout
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
    assert_operator @now, :<=, CitationFetch::DEADLINE
    assert_operator @asked.size, :<, 5
  end

  def test_a_shared_deadline_cuts_a_second_fetch_short
    answering(*Array.new(10) { response(429) }, took: 60)
    stop = @now + CitationFetch::DEADLINE
    CitationFetch.fetch(%w[arXiv:1], stop: stop)
    first = @asked.size
    CitationFetch.fetch(%w[arXiv:1], stop: stop)
    assert_equal first, @asked.size # no try that could run past the shared deadline
  end

  def test_another_4xx_is_not_retried_and_says_why
    answering(response(400, "{\"error\":\"Unrecognized field\"}\n"))
    assert_nil CitationFetch.fetch(%w[arXiv:1])
    assert_equal 1, @asked.size
    assert_includes $stderr.string, "400 {\"error\":\"Unrecognized field\"}"
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

  S2 = "f1c0aad419cd942d5a135b47fcce58d101dbb3bd"
  NOW = Time.utc(2026, 11, 17, 12)
  LIVE = "https://example.org/citations.json"

  # run() in a scratch dir; returns [what it returned, the file, the notes].
  def run_with(*answers, key: "", existing: nil)
    Dir.mktmpdir do |dir|
      out = File.join(dir, "citations.json")
      File.write(out, existing) if existing
      notes = []
      answering(*answers)
      data = CitationFetch.run(%w[arXiv:1 arXiv:2], out: out, published_url: LIVE, key: key,
                               note: ->(m, l) { notes << [l, m] }, now: NOW)
      [data, File.file?(out) ? JSON.parse(File.read(out)) : nil, notes]
    end
  end

  def published(at)
    response(200, JSON.generate("fetched_at" => at, "papers" => { "arXiv:1" => { "count" => 3, "paper_id" => S2 } }))
  end

  def test_run_writes_the_fetched_counts_and_names_the_missing
    data, file, notes = run_with(response(200, JSON.generate([{ "paperId" => S2, "citationCount" => 9 }, nil])))
    assert_equal({ "fetched_at" => "2026-11-17T12:00:00Z", "papers" => { "arXiv:1" => { "count" => 9, "paper_id" => S2 } } }, file)
    assert_equal file, data
    assert_equal [["warning", "no count for arXiv:2 (not found, or 0: see _data/citation_ids.yml)"]], notes
  end

  def test_a_refused_key_is_an_error_and_is_asked_again_without_it
    _, file, notes = run_with(response(403), response(200, JSON.generate([nil, nil])), key: "k")
    assert_equal %w[k] + [nil], @asked.map { |r| r["x-api-key"] }
    assert_equal "error", notes.first[0]
    assert_equal({}, file["papers"])
  end

  def test_without_an_answer_the_published_counts_are_kept
    _, file, notes = run_with(*Array.new(5) { response(429) }, published("2026-11-16T20:41:00Z"))
    assert_equal "2026-11-16T20:41:00Z", file["fetched_at"]
    assert_equal [["warning", "no usable answer from Semantic Scholar; kept the published counts of 2026-11-16T20:41:00Z"]], notes
  end

  def test_published_counts_a_week_old_are_an_error
    _, _, notes = run_with(*Array.new(5) { response(429) }, published("2026-11-09T00:00:00Z"))
    assert_equal "error", notes.last[0]
  end

  def test_an_unusable_answer_falls_back_too
    _, file, notes = run_with(response(200, "[]"), published("2026-11-16T20:41:00Z"))
    assert_equal 3, file.dig("papers", "arXiv:1", "count")
    assert_match(/unusable answer/, notes.first[1])
  end

  def test_with_nothing_to_write_the_file_is_left_alone
    before = JSON.generate("fetched_at" => "2026-11-01T00:00:00Z", "papers" => {})
    data, file, notes = run_with(*Array.new(5) { response(429) }, response(404), existing: before)
    assert_nil data
    assert_equal JSON.parse(before), file
    assert_match(/not updated/, notes.last[1])
  end
end
