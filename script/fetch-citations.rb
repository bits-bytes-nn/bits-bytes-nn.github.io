# frozen_string_literal: true
#
# Fetches every paper post's citation count from Semantic Scholar in one batch
# request and writes _data/citations.json, which the build prints into each
# post (_plugins/citations.rb) and publishes as /citations.json. Run before
# `jekyll build`; CI does on every run (PRs, pushes and the daily one).
#
# The keyless API shares one rate limit worldwide and answers 429 often, to a
# batch too, so the request is retried with backoff, within DEADLINE. When it
# still fails, the counts the live site published last are kept instead, when
# they can be read. With neither, it warns and the site builds without counts;
# the browser then asks for them (js/citations.js).
# SEMANTIC_SCHOLAR_API_KEY, when set, is sent as the API key; a key the API
# refuses is reported as an error and the request is made again without it,
# so a lapsed secret never holds up a deploy.
#
# Usage: bundle exec ruby script/fetch-citations.rb

require "json"
require "net/http"
require "time"

module CitationFetch
  API = URI("https://api.semanticscholar.org/graph/v1/paper/batch?fields=citationCount")
  WAITS = [5, 10, 20, 40].freeze # seconds before each retry, unless the API asks for longer
  TIMEOUTS = { open_timeout: 10, ssl_timeout: 10, read_timeout: 30, write_timeout: 10, max_retries: 0 }.freeze
  TRY = TIMEOUTS.values_at(:open_timeout, :ssl_timeout, :read_timeout, :write_timeout).sum # one try's longest
  DEADLINE = 240 # the batch ends by then; with the fallback's one try, about 5 minutes at worst
  STALE = 7 * 24 * 60 * 60 # published counts older than this are reported as an error

  class KeyRefused < StandardError; end

  # The seams test/test_fetch_citations.rb replaces.
  class << self
    attr_writer :http, :pause, :clock

    def http = @http ||= ->(uri, req) { Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https", **TIMEOUTS) { |h| h.request(req) } }
    def pause = @pause ||= ->(seconds) { sleep seconds }
    def clock = @clock ||= -> { Process.clock_gettime(Process::CLOCK_MONOTONIC) }
  end

  # The batch answer's body, or nil once every try has failed or `stop` (a
  # clock time; DEADLINE from now by default) is near. Raises KeyRefused when
  # the API refuses the key.
  def self.fetch(ids, key: "", stop: clock.call + DEADLINE)
    req = Net::HTTP::Post.new(API, "Content-Type" => "application/json")
    req["x-api-key"] = key unless key.empty?
    req.body = JSON.generate("ids" => ids)
    (WAITS + [nil]).each do |wait|
      return nil if clock.call + TRY > stop # no try that could run past it
      begin
        response = http.call(API, req)
        return response.body if response.is_a?(Net::HTTPSuccess)
        status = response.code
        raise KeyRefused, status if !key.empty? && %w[401 403].include?(status)
        # Another 4xx is this request's fault; asking again will not help.
        if status.start_with?("4") && status != "429"
          warn("fetch-citations: #{status} #{response.body.to_s.lines.first.to_s.strip[0, 200]}")
          return nil
        end
        retry_after = response["Retry-After"].to_i
      rescue KeyRefused
        raise
      rescue StandardError => e # best effort: whatever went wrong, the build goes on
        status = e.class.name
        retry_after = 0
      end
      return nil unless wait
      wait = retry_after.clamp(wait, WAITS.max) if retry_after.positive?
      return nil if clock.call + wait + TRY > stop
      warn("fetch-citations: #{status}, retrying in #{wait}s")
      pause.call(wait)
    end
  end

  # What the live site published last, or nil: a Hash with `fetched_at` and
  # `papers`, as this script writes it.
  def self.published(url)
    uri = URI(url)
    response = http.call(uri, Net::HTTP::Get.new(uri))
    data = response.is_a?(Net::HTTPSuccess) ? JSON.parse(response.body) : nil
    data if data.is_a?(Hash) && data["papers"].is_a?(Hash) && data["fetched_at"].is_a?(String)
  rescue StandardError
    nil
  end
end

# A warning or error: an annotation in CI, and an error also in the run's
# summary, where a scheduled run's shows without opening the log.
def note(message, level = "warning")
  warn(ENV["GITHUB_ACTIONS"] ? "::#{level} title=Citation counts::#{message}" : "fetch-citations: #{message}")
  summary = ENV["GITHUB_STEP_SUMMARY"]
  File.write(summary, "**Citation counts:** #{message}\n", mode: "a") if level == "error" && summary && !summary.empty?
end

if $PROGRAM_NAME == __FILE__
  require "jekyll"
  $stdout.sync = true
  root = File.expand_path("..", __dir__)
  out = File.join(root, "_data/citations.json")

  # Jekyll reads the posts, so `citation_id` is the plugin's answer, the same
  # one the build prints for the browser.
  site = Jekyll::Site.new(Jekyll.configuration("source" => root, "quiet" => true))
  site.reset
  site.read
  ids = site.posts.docs.filter_map { |post| post.data["citation_id"] }.uniq
  abort "fetch-citations: no post has a citation_id; is _plugins/citations.rb loaded?" if ids.empty?

  papers = nil
  stop = CitationFetch.clock.call + CitationFetch::DEADLINE # one deadline for both tries below
  begin
    answer = CitationFetch.fetch(ids, key: ENV["SEMANTIC_SCHOLAR_API_KEY"].to_s, stop: stop)
  rescue CitationFetch::KeyRefused => e
    note("Semantic Scholar refused SEMANTIC_SCHOLAR_API_KEY (#{e.message}); renew the secret. Asking without it", "error")
    answer = CitationFetch.fetch(ids, stop: stop)
  end
  if answer
    begin
      papers = Citations.entries(ids, JSON.parse(answer))
    rescue JSON::ParserError, ArgumentError => e
      note("unusable answer from Semantic Scholar (#{e.message.lines.first.strip})")
    end
  end

  if papers
    File.write(out, JSON.pretty_generate("fetched_at" => Time.now.utc.iso8601, "papers" => papers.sort.to_h) + "\n")
    puts "fetch-citations: #{papers.size} of #{ids.size} papers counted"
    missing = ids - papers.keys
    note("no count for #{missing.join(", ")} (not found, or 0: see _data/citation_ids.yml)") unless missing.empty?
  elsif (last = CitationFetch.published("#{site.config["url"]}#{ENV.fetch("BASE_PATH", site.baseurl)}/citations.json"))
    File.write(out, JSON.pretty_generate(last) + "\n")
    age = Time.now - (Time.iso8601(last["fetched_at"]) rescue Time.at(0))
    note("no usable answer from Semantic Scholar; kept the published counts of #{last["fetched_at"]}",
         age > CitationFetch::STALE ? "error" : "warning")
  else
    note("no usable answer from Semantic Scholar and no published counts; _data/citations.json is not updated")
  end
end
