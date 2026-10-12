# frozen_string_literal: true
#
# Checks the Markdown source of every post against _data/post_conventions.yml
# and the authoring rules a built page cannot show any more: by the time
# validate-site.sh sees the HTML, a misaligned table is just a table and a
# currency sign read as math is just math. Each check below is a defect that
# shipped in a post here.
#
# Usage: ruby script/lint-posts.rb [post.md ...]   (default: every _posts/*.md)
# Exits 1 when any post breaks a rule.

require "yaml"
require "date"

module PostLint
  ROOT = File.expand_path("..", __dir__)

  # One parsed post: front matter, body, and the line the body starts on, so
  # findings point at the source line.
  Post = Struct.new(:path, :data, :body, :body_line)

  def self.parse(path, text = File.read(path))
    parts = text.split(/^---\s*$\n?/, 3)
    raise ArgumentError, "#{path}: no front matter" unless parts.size == 3 && parts[0].strip.empty?

    data = YAML.safe_load(parts[1], permitted_classes: [Date, Time]) || {}
    Post.new(path, data, parts[2], parts[1].count("\n") + 3)
  end

  # Returns findings as "line: message" strings; empty when the post passes.
  def self.check(post, spec)
    front_matter(post, spec) + tags(post, spec) + translation(post) + body(post)
  end

  def self.front_matter(post, spec)
    data = post.data
    found = (spec["forbidden_keys"] & data.keys).map { |k| "1: front matter key `#{k}` is not allowed" }

    type, topic = Array(data["categories"])
    unless Array(data["categories"]).size == 2 && spec["types"].key?(type) && spec["topics"].key?(topic)
      return found << "1: categories must be [<type>, <topic>] from _data/post_conventions.yml, got #{data["categories"].inspect}"
    end

    if spec["types"][type]["paper"]
      author = data["paper_author"]
      unless author.is_a?(String) && !author.strip.empty? && !author.match?(/et al/i)
        found << "1: paper_author must name the paper's organisation, got #{author.inspect}"
      end
    end

    cover = spec["types"][type]["cover"] || spec["topics"][topic]
    if cover && data["cover"] != cover
      found << "1: cover for #{type} / #{topic} is #{cover}, got #{data["cover"].inspect}"
    elsif !cover && !File.file?(File.join(ROOT, data["cover"].to_s))
      found << (data["cover"] ? "1: cover #{data["cover"]} does not exist" : "1: no cover")
    end
    found
  end

  def self.tags(post, spec)
    rules = spec["tags"]
    tags = Array(post.data["tags"]).map(&:to_s)
    found = []
    found << "1: no controlled topic tag (one of #{rules["controlled"].join(", ")})" if (tags & rules["controlled"]).empty?
    tags.each do |tag|
      if (replacement = rules["aliases"][tag])
        found << "1: tag #{tag} is retired; use #{replacement}"
      elsif tag !~ /\A[A-Za-z0-9][A-Za-z0-9.+-]*\z/
        found << "1: tag #{tag.inspect} must be words joined by hyphens"
      # Inner words only: a last "-In" or "-On" is a particle (Opt-In, Add-On).
      elsif (word = tag.split("-")[1...-1].find { |w| rules["lowercase_words"].include?(w.downcase) && w != w.downcase })
        found << "1: tag #{tag} should have #{word.downcase}, lowercase"
      end
    end
    found
  end

  # A translation lives at <original-slug>-en.md and both posts use that slug as
  # their translation_id. The pairing itself runs on the id alone (the language
  # switcher, _plugins/translations.rb); the file names keep a pair findable.
  def self.translation(post)
    id = post.data["translation_id"].to_s
    return [] if id.empty?

    slug = File.basename(post.path, ".md").sub(/\A\d{4}-\d{2}-\d{2}-/, "")
    expected = post.data["lang"] == "en" ? "#{id}-en" : id
    slug == expected ? [] : ["1: a post with translation_id #{id} is named #{expected}.md (after the date), not #{slug}.md"]
  end

  FENCE = /\A\s*(`{3,}|~{3,})/

  def self.body(post)
    found = []
    lines = post.body.split("\n", -1)
    fence = nil # the marker that opened the code fence we are in
    block = nil # the line a still-open multi-line $$ block started on
    span = nil # the line an inline $$ span wrapped onto the next line started on
    lines.each_with_index do |line, i|
      n = post.body_line + i
      if (m = line.match(FENCE))
        if fence.nil?
          fence = m[1]
        elsif line.strip.match?(/\A#{Regexp.escape(fence[0])}{#{fence.size},}\z/)
          fence = nil
        end
        next
      end
      next if fence

      # A display block over several lines opens on a line starting with $$ and
      # closes on the next line with an odd number of them: `$$` alone, or
      # `$$x = \begin{cases}` ... `\end{cases}$$`.
      # Inside a blockquote the same rules apply to the text after the markers.
      bare = unquote(line)
      odd = bare.gsub(/`[^`]*`/, "").scan("$$").size.odd?
      if span && bare.strip.empty?
        found << "#{span}: unbalanced $$ in the paragraph"
        span = nil
      end
      if block || (odd && !span && bare.lstrip.start_with?("$$"))
        found.concat(math_findings([line.delete("$")], n))
        if odd && block
          found << "#{n}: display $$ needs a blank line after" unless after_display?(lines[i + 1])
        elsif odd && i.positive? && !unquote(lines[i - 1]).strip.empty?
          found << "#{n}: display $$ needs a blank line before"
        end
        block = (block ? nil : n) if odd
        next
      end
      # An odd line opens an inline span a later line of the paragraph closes.
      span = (span ? nil : n) if odd

      found << "#{n}: H1 in the body; the layout renders the title" if line.match?(/\A# \S/)
      found.concat(math(line, n, lines, i))
      found.concat(table(lines, i, n)) if table_start?(lines, i)
    end
    found << "#{block}: $$ block never closed" if block
    found << "#{span}: unbalanced $$ in the paragraph" if span
    found
  end

  # Brace groups whose content is text, not math: escapes and Korean belong there.
  TEXT_GROUP = /\\(?:text\w*|mathrm|operatorname)\{/

  def self.without_text_groups(math)
    out = +""
    rest = math
    while (m = rest.match(TEXT_GROUP))
      out << m.pre_match
      depth = 1
      k = m.end(0)
      while k < rest.size && depth.positive?
        depth += 1 if rest[k] == "{"
        depth -= 1 if rest[k] == "}"
        k += 1
      end
      rest = rest[k..] || ""
    end
    out << rest
  end

  def self.math_findings(spans, n)
    spans = spans.map { |m| without_text_groups(m) }
    found = []
    if spans.any? { |m| m.include?("\\_") }
      found << "#{n}: `\\_` inside $$...$$ renders a literal underscore; write `_`"
    end
    # Only where the span holds Korean: a currency sign written `$$` opens math
    # and swallows the prose up to the next one. In English prose it goes unseen.
    if spans.any? { |m| m.match?(/\p{Hangul}/) }
      found << "#{n}: Korean inside $$...$$; a currency sign written `$$` opens math (write `$5`)"
    end
    found
  end

  # A display equation needs a blank line (or a kramdown `{: ...}` attribute
  # line) after it, and a blank line before, or kramdown folds it into the
  # paragraph as inline math. That holds inside list items too.
  def self.after_display?(line)
    rest = unquote(line.to_s).strip
    rest.empty? || rest.start_with?("{:")
  end

  # The text of a line after any blockquote markers.
  def self.unquote(line)
    line.sub(/\A\s*(?:>\s?)+/, "")
  end

  def self.math(line, n, lines, i)
    prose = unquote(line).gsub(/`[^`]*`/, "")
    found = math_findings(prose.scan(/\$\$(.+?)\$\$/).flatten, n)
    outside = prose.gsub(/\$\$.+?\$\$/, "")
    # `\\(` reaches MathJax as `\(` but only after Markdown has read the formula:
    # `*` and `_` inside it turn into <em>, and the formula renders as raw TeX.
    if outside.match?(/\\\\[(\[]/)
      found << "#{n}: math written \\\\( or \\\\[; write $$...$$, which Markdown leaves alone"
    end
    # TeX next to a single `$` is math that never opened; a price is neither
    # preceded by `}` nor followed by a command.
    if outside.match?(/\}\$(?!\$)|(?<!\$)\$\\[a-zA-Z]/)
      found << "#{n}: TeX next to a single `$`; write the formula as $$...$$"
    end
    if prose.strip.match?(/\A\$\$(?:(?!\$\$).)+\$\$\z/) && !prose.lstrip.start_with?("|")
      before_ok = i.zero? || unquote(lines[i - 1]).strip.empty?
      unless before_ok && after_display?(lines[i + 1])
        found << "#{n}: display $$ needs a blank line before and after"
      end
    end
    found
  end

  def self.table_start?(lines, i)
    lines[i].lstrip.start_with?("|") && lines[i + 1].to_s.strip.match?(/\A\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\z/) &&
      (i.zero? || !lines[i - 1].lstrip.start_with?("|"))
  end

  # Cells of one table row. `\|`, and pipes inside code or $$...$$, are content.
  def self.cells(row)
    text = row.strip.gsub(/`[^`]*`|\$\$.*?\$\$/) { |m| m.tr("|", "\u0000") }.gsub("\\|", "\u0000")
    text = text.delete_prefix("|").delete_suffix("|")
    text.split("|", -1).map(&:strip)
  end

  def self.table(lines, i, n)
    header = cells(lines[i])
    rows = lines[(i + 2)..].take_while { |l| l.lstrip.start_with?("|") }
    found = []
    if header.drop(1).any?(&:empty?)
      found << "#{n}: table header has an empty cell; merge a two-row header into one (\"AIME 2024 pass@1\")"
    end
    first = rows.first && cells(rows.first)
    if first && first[0].empty? && first.none? { |c| c.match?(/\A[-+]?[\d.,]+%?\z/) }
      found << "#{n + 2}: table row under the header is a second header row; merge it into the header"
    end
    rows.each_with_index do |row, k|
      count = cells(row).size
      found << "#{n + 2 + k}: table row has #{count} cells, the header #{header.size}" if count > header.size
    end
    found
  end

  def self.spec
    YAML.safe_load(File.read(File.join(ROOT, "_data/post_conventions.yml")))
  end
end

if $PROGRAM_NAME == __FILE__
  spec = PostLint.spec
  paths = ARGV.empty? ? Dir[File.join(PostLint::ROOT, "_posts/*.md")].sort : ARGV
  failed = 0
  paths.each do |path|
    findings =
      begin
        PostLint.check(PostLint.parse(path), spec)
      rescue StandardError, Psych::Exception => e
        ["1: could not be checked: #{e.message}"]
      end
    next if findings.empty?

    failed += 1
    rel = path.delete_prefix("#{PostLint::ROOT}/")
    findings.each { |f| warn "#{rel}:#{f}" }
  end
  if failed.positive?
    warn "\n#{failed} of #{paths.size} post(s) break the conventions."
    exit 1
  end
  puts "#{paths.size} posts follow the conventions."
end
