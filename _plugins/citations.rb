# frozen_string_literal: true
#
# A paper post's citation count: which Semantic Scholar record it is, and the
# count the last build fetched for it.
#
# Every post with a `paper_doi` gets `citation_id`, the id Semantic Scholar is
# asked by. script/fetch-citations.rb asks for all of them at once before the
# build and writes _data/citations.json (not in Git); a post whose record it
# found also gets `citation` ({count, paper_id, shown}). _layouts/post.html
# prints the count, and js/citations.js refreshes it once it is a day old.
# With no file (a local build) or no entry, the browser asks for it.
#
# Semantic Scholar finds an arXiv paper by its arXiv ID, not by the DOI arXiv
# registers for it. A few papers are not there under that ID, or it leads to a
# duplicate record with no citations: _data/citation_ids.yml names the record
# for those.

module Citations
  ARXIV_DOI = %r{\A10\.48550/arXiv\.(.+)\z}i
  PAPER_ID = /\A\h{40}\z/

  # The id for a DOI: the record named in `overrides` (DOI => paperId), else
  # arXiv:<id> for an arXiv DOI, else DOI:<doi>.
  def self.lookup_id(doi, overrides = {})
    return overrides[doi] if overrides.key?(doi)
    arxiv = doi.match(ARXIV_DOI)
    arxiv ? "arXiv:#{arxiv[1]}" : "DOI:#{doi}"
  end

  # _data/citation_ids.yml, checked: each key a DOI some post gives, each value
  # a paperId. Raises on anything else, so a mistyped entry fails the build
  # rather than silently asking for the wrong paper.
  def self.overrides(data, dois)
    data ||= {}
    raise ArgumentError, "_data/citation_ids.yml must map DOIs to paperIds" unless data.is_a?(Hash)
    data.each do |doi, id|
      raise ArgumentError, "_data/citation_ids.yml: #{doi} is no post's paper_doi" unless dois.include?(doi)
      raise ArgumentError, "_data/citation_ids.yml: #{doi} maps to #{id.inspect}, not a 40-hex paperId" unless id.is_a?(String) && id.match?(PAPER_ID)
    end
    data
  end

  # 21110 as "21,110", as js/citations.js prints it (toLocaleString("en-US")).
  def self.delimited(count)
    count.to_s.reverse.scan(/\d{1,3}/).join(",").reverse
  end

  # One record of the batch answer, as _data/citations.json keeps it. nil for
  # a paper not found, or one with no citations: Semantic Scholar answers 0
  # for a duplicate record, and a zero is not shown.
  def self.entry(record)
    return nil unless record.is_a?(Hash)
    count = record["citationCount"]
    id = record["paperId"].to_s
    return nil unless count.is_a?(Integer) && count.positive? && id.match?(PAPER_ID)
    { "count" => count, "paper_id" => id }
  end

  # Sets `citation_id`, and `citation` when the fetched counts have one, on
  # each of `posts` (anything with a `data` Hash) that has a `paper_doi`.
  # `data` is site.data. Raises ArgumentError on a bad citation_ids.yml.
  def self.annotate(posts, data)
    papers = posts.select { |post| post.data["paper_doi"] }
    overrides = overrides(data["citation_ids"], papers.map { |post| post.data["paper_doi"].to_s })
    fetched = data["citations"].is_a?(Hash) ? data["citations"]["papers"] : nil
    fetched = {} unless fetched.is_a?(Hash)
    papers.each do |post|
      id = lookup_id(post.data["paper_doi"].to_s, overrides)
      post.data["citation_id"] = id
      stored = fetched[id].is_a?(Hash) ? fetched[id] : {}
      e = entry("citationCount" => stored["count"], "paperId" => stored["paper_id"])
      post.data["citation"] = e.merge("shown" => delimited(e["count"])) if e
    end
  end

  # `ids` and the batch answer, which lists one record (or null) per id in
  # order. Raises when the two do not line up: a count must not land on
  # another paper.
  def self.entries(ids, records)
    unless records.is_a?(Array) && records.size == ids.size
      raise ArgumentError, "Semantic Scholar answered #{records.is_a?(Array) ? records.size : records.class} records for #{ids.size} ids"
    end
    ids.zip(records).filter_map { |id, record| (e = entry(record)) && [id, e] }.to_h
  end
end

# Guarded so test/ can require this file without Jekyll loaded.
if defined?(Jekyll::Hooks)
  Jekyll::Hooks.register :site, :post_read do |site|
    Citations.annotate(site.posts.docs, site.data)
  rescue ArgumentError => e
    raise Jekyll::Errors::FatalException, e.message
  end
end
