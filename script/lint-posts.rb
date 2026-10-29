# frozen_string_literal: true
#
# Checks the Markdown source of every post against _data/post_conventions.yml
# and the authoring rules a built page cannot show any more: by the time
# validate-site.sh sees the HTML, a misaligned table is just a table and a
# currency sign read as math is just math. Each check is a defect that shipped
# in a post here, or one that ships silently: every gate stays green while the
# page shows it.
#
# Usage: ruby script/lint-posts.rb [post.md ...]   (default: every file under _posts/)
# Exits 1 when any post breaks a rule.

require "yaml"
require "date"
require "set"

module PostLint
  # `$x_i$`, `$a*b$`, `$T$`: one line, not starting with a digit or space, not
  # ending in a space, holding a TeX operator or a lone letter.
  SINGLE_DOLLAR_MATH = /(?<![$\\\w])\$(?=[^$\s\d])(?:[^$\n]{0,80}?[_^*=\\][^$\n]{0,80}?|[A-Za-z])(?<!\s)\$(?![$\w])/
  ROOT = File.expand_path("..", __dir__)

  # One parsed post: front matter, body, and the line the body starts on, so
  # findings point at the source line.
  Post = Struct.new(:path, :data, :body, :body_line)

  def self.parse(path, text = File.read(path))
    parts = text.delete_prefix("\uFEFF").split(/^---[ \t]*\r?$\n?/, 3) # Jekyll reads bom|utf-8
    # Jekyll reads front matter only when `---` is the first line.
    raise ArgumentError, "#{path}: no front matter" unless parts.size == 3 && parts[0].empty?

    data = YAML.safe_load(parts[1], permitted_classes: [Date, Time]) || {}
    Post.new(path, data, parts[2], parts[1].count("\n") + 3)
  end

  # The file name after its date: what the permalink and translation_id use.
  def self.slug(post)
    File.basename(post.path, ".md").sub(/\A\d{4}-\d{2}-\d{2}-/, "")
  end

  # Returns findings as "line: message" strings; empty when the post passes.
  def self.check(post, spec)
    front_matter(post, spec) + tags(post, spec) + translation(post, spec) + body(post) + liquid(post) + lightbox(post)
  end

  # A hand-written zoomable figure opens its href in the lightbox. A page there
  # (the figure's source article) loads in an iframe the site refuses: an empty
  # or endless box. 16 figures in the AI-ready-data pair shipped that way.
  # Attribute values may hold `>` (a caption "A -> B"), so runs skip quoted text.
  ATTRS = %q{(?:[^>"']|"[^"]*"|'[^']*')*}
  LIGHTBOX = %r{<a\b#{ATTRS}\bclass=["'][^"']*\bglightbox\b[^"']*["']#{ATTRS}>\s*(?<img><img\b#{ATTRS}>)}mi

  def self.lightbox(post)
    # Fenced code keeps its line count, so an example there is neither read nor
    # shifts the line numbers after it.
    lines = post.body.split("\n", -1)
    body = fences(lines).zip(lines).map { |state, line| state ? "" : line }.join("\n")
    body.to_enum(:scan, LIGHTBOX).filter_map do
      m = Regexp.last_match
      href = m[0][/\shref=["']([^"']+)/i, 1]
      src = m[:img][/\ssrc=["']([^"']+)/i, 1]
      next unless href && src && href != src

      "#{post.body_line + body[0, m.begin(0)].count("\n")}: a glightbox link opens #{href}, not its image; set href to #{src}"
    end
  end

  # Jekyll renders a post as Liquid first, so a prompt template's {{ question }}
  # in a code block renders as nothing. The Kimi K2 post shipped that way.
  def self.liquid(post)
    return [] if post.data["render_with_liquid"] == false

    raw = false
    post.body.split("\n", -1).each_with_index.filter_map do |line, i|
      # Token by token: a line may close a raw span and open a bare tag.
      hit = false
      line.split(/(\{%-?\s*(?:end)?raw\s*-?%\})/).each do |part|
        if (tag = part[/\A\{%-?\s*((?:end)?raw)\s*-?%\}\z/, 1])
          raw = tag == "raw"
        elsif !raw && part.match?(/\{\{|\{%/)
          hit = true
        end
      end
      next unless hit
      # Wrapped in raw, a post_url ships as a literal href; the fix is a URL.
      if line.match?(/\{%-?\s*(post_url|link)\b/)
        "#{post.body_line + i}: Liquid #{line[/\{%-?\s*(post_url|link)/, 1]} in the body; link the post by its root-relative URL (/paper-reviews/.../slug.html, from the built site)"
      else
        "#{post.body_line + i}: Liquid tag in the body; posts do not use Liquid, so wrap it in {% raw %} ... {% endraw %}"
      end
    end
  end

  # Across posts: a translation is a pair, one `<slug>.md` and one `<slug>-en.md`
  # with translation_id <slug>. Half a pair lists the English post as its own
  # article and shows no language switcher.
  def self.pairs(posts, spec)
    found = Hash.new { |h, k| h[k] = [] }
    posts.each do |post|
      found[post.path] << "1: an -en post must set translation_id" if slug(post).end_with?("-en") && !post.data["translation_id"]
    end
    posts.select { |p| p.data["translation_id"] }.group_by { |p| p.data["translation_id"].to_s }.each do |id, group|
      next if group.map { |p| p.data["lang"].to_s }.sort == spec["languages"].sort

      # Name what is missing: a post without `lang`, or a second post in one language.
      missing = spec["languages"] - group.map { |p| p.data["lang"].to_s }
      group.each do |post|
        hint = post.data["lang"].to_s.empty? ? "set lang: #{missing.first || spec["languages"].join(" or ")}" : "no #{missing.join(", ")} post shares it; give the twin translation_id: #{id} and lang: #{missing.join(", ")}"
        found[post.path] << "1: translation_id #{id} is not on one post in each of #{spec["languages"].join(", ")}; #{hint}"
      end
    end
    found
  end

  def self.front_matter(post, spec)
    data = post.data
    hints = { "author" => "; the paper's organisation goes in paper_author", "last_modified_at" => "; _plugins/posted_date.rb sets it" }
    found = (spec["forbidden_keys"] & data.keys).map { |k| "1: front matter key `#{k}` is not allowed#{hints[k]}" }
    # It ships as plain text (meta, feed, home card); markup would show literally.
    if data["description"].to_s.match?(/\$\$|\*\S[^*]*\*|`|\]\(|<[a-zA-Z\/]|\{[{%]/)
      found << "1: description is plain text; no $$, *emphasis*, backticks, links or tags"
    end
    # Text the reader sees: a tag in it (<think>, <!-- …) is read as HTML.
    %w[title subtitle paper_author].each do |key|
      found << "1: #{key} has a tag; the page reads it as HTML" if data[key].to_s.match?(/<[a-zA-Z\/!]/)
    end

    # Jekyll drops a file without the date prefix, and builds a `date:` past the
    # file's day into a permanent URL. The file-name date is the posting day.
    if (posted = File.basename(post.path)[/\A(\d{4}-\d{2}-\d{2})-/, 1])
      # Jekyll reads a quoted date too, and builds in UTC (_config.yml); so do
      # DateTime.parse and YAML, for a time without an offset.
      date = data["date"].is_a?(String) ? (DateTime.parse(data["date"]).to_time rescue nil) : data["date"]
      day = date.instance_of?(Date) ? date : date.respond_to?(:to_time) && date.to_time.utc.to_date
      found << "1: date: #{day} (UTC) is after the file name's #{posted}" if day && day > Date.parse(posted)
    else
      found << "1: the file name must start with YYYY-MM-DD- (the posting day), or Jekyll skips the post"
    end

    type, topic = Array(data["categories"])
    unless Array(data["categories"]).size == 2 && spec["types"].key?(type) && spec["topics"].key?(topic)
      return found << "1: categories must be [<type>, <topic>], got #{data["categories"].inspect}; types: #{spec["types"].keys.join(", ")}; topics: #{spec["topics"].keys.join(", ")} (a new topic: docs/writing.md, \"Categories, URLs and covers\")"
    end

    if spec["types"][type]["paper"]
      author = data["paper_author"]
      unless author.is_a?(String) && !author.strip.empty? && !author.match?(/et al/i)
        found << "1: paper_author must name the paper's organisation, got #{author.inspect}"
      end
      # Printed under the title (_layouts/post.html): a paper's title is
      # English, and this is what a Korean reader sees there in Korean.
      subtitle = data["subtitle"].to_s.strip
      if subtitle.empty? || subtitle == data["title"].to_s.strip
        found << "1: #{type} posts need a subtitle that is not the title"
      end
      # js/citations.js puts it in the request path as is, so a URL, a `doi:`
      # prefix, or a `?` or `#` in it finds nothing there.
      # arXiv registers one DOI per paper, without the version (…v2).
      doi = data["paper_doi"].to_s
      if data.key?("paper_doi") && !doi.match?(%r{\A10\.\d{4,9}/[^\s?#]+\z})
        found << "1: paper_doi is a bare DOI (10.<registrant>/<suffix>), got #{data["paper_doi"].inspect}"
      elsif doi.match?(%r{\A10\.48550/.*v\d+\z}i)
        found << "1: paper_doi #{doi.inspect} carries an arXiv version; drop the vN"
      end
    else
      paper_types = spec["types"].select { |_, t| t["paper"] }.keys.join(" and ")
      # The layout prints "Paper by …" for any post that has it.
      %w[paper_author paper_doi].each do |k|
        found << "1: #{k} is for #{paper_types} posts" if data.key?(k)
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
  def self.translation(post, spec)
    id = post.data["translation_id"].to_s
    return [] if id.empty?
    # The switcher and hreflang read `lang` as written, and the switcher knows
    # only these two; without it a Korean post links to itself as "English".
    langs = spec["languages"]
    return ["1: a post with translation_id must set lang (#{langs.join(" or ")})"] unless langs.include?(post.data["lang"])

    expected = post.data["lang"] == "en" ? "#{id}-en" : id
    return [] if slug(post) == expected

    # A published post's file name is its URL, so for one of those the id is
    # what to fix; a new twin can still take the expected name.
    right = slug(post).delete_suffix("-en")
    ["1: translation_id #{id} names this post #{expected}.md (after the date), not #{slug(post)}.md; rename a new post, or set translation_id: #{right} on both posts of a published pair"]
  end

  FENCE = /\A\s*(`{3,}|~{3,})/

  # Per line: [:open, match] on a fence's first line, :code inside, :close on
  # its last, nil outside. A fence inside a blockquote ("> ```") is code too,
  # and only such a fence closes on a quoted marker: a "> ```" line inside a
  # bare ```markdown block is part of its example. A blank line ends the
  # blockquote, and with it a quoted fence left open (:unclosed).
  def self.fences(lines)
    fence = nil
    quoted = false
    lines.map do |line|
      text = unquote(line)
      if fence.nil?
        next unless (m = text.match(FENCE))

        fence = m[1]
        quoted = line.match?(/\A\s*>/)
        [:open, m]
      elsif quoted && line.strip.empty?
        fence = nil
        :unclosed
      elsif (quoted ? text : line).strip.match?(/\A#{Regexp.escape(fence[0])}{#{fence.size},}\z/)
        fence = nil
        :close
      else
        :code
      end
    end
  end
  # A code span, opened and closed by the same run of backticks (``$$`` too).
  CODE_SPAN = /(`+).*?\1/

  def self.strip_code(text)
    text.gsub(CODE_SPAN, "")
  end

  # kramdown passes any <name> through as an element, and the browser drops a
  # tag it does not know: "<think> </think>" vanished from a prompt template in
  # the DeepSeek-R1 post. A prompt token named like a form element (<input>,
  # <output>, <template>) or <s> fares no better, so this is the HTML a post
  # writes, not all of HTML.
  HTML_ELEMENTS = %w[
    a abbr audio b blockquote br caption center cite code col colgroup dd del details
    dfn div dl dt em figcaption figure h1 h2 h3 h4 h5 h6 hr i iframe img ins kbd li
    mark ol p picture pre q samp script section small source span strong sub summary
    sup svg table tbody td tfoot th thead time tr track u ul var video wbr
    circle defs g line path polygon polyline rect text tspan use
  ].to_set # the last row: an inline <svg>'s children
  # Elements that mean nothing bare: <img> or <audio> alone is a token, not media.
  NEEDS_ATTRIBUTES = %w[a audio iframe img script source video].to_set

  # The patterns of scholar-lens's prose rules (its markdown_lint.py), so a
  # hand-written post answers to them too. Its scope is wider for the first.
  # Translationese with no natural reading: a double passive ("보여지는") and
  # the calque "에 있어서는", "학습함에 있어". A bare "에 있어서" stays.
  TRANSLATIONESE = /되어[지진져집]|보여[지진져]|에 있어(?:서[는도의](?! 안)|[는의])|[음함]에 있어(?![야요])/
  # A sentence ending glued to a colon ("다음과 같습니다:"): end the sentence.
  SENTENCE_COLON = /다:/
  QUOTED = /"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|「[^」\n]*」/
  REFERENCES = /\A\#{1,6}\s+(?:references|bibliography|참고 ?자료|참고 ?문헌)\b/i

  # Not in a heading, a quotation, a table row (indented under a list too),
  # image alt text (scholar-lens fills it from the heading) or the reference list.
  def self.prose(line, text, in_references)
    return [] if in_references || line.start_with?("#") || line.lstrip.start_with?(">", "|")

    text = text.gsub(QUOTED, "").gsub(/!\[[^\]\n]*\]\([^)\n]*\)/, "")
    [(%(translationese "#{$&}"; reword it) if text =~ TRANSLATIONESE),
     ("a sentence ends in a colon (#{$&}); end it with a period" if text =~ SENTENCE_COLON)].compact
  end

  def self.body(post)
    found = []
    in_references = false
    lines = post.body.split("\n", -1)
    states = fences(lines)
    fence_line = nil # where a still-open code fence started
    block = nil # the line a still-open multi-line $$ block started on
    para = nil # the line a paragraph with an odd number of $$ started on
    math_seen = false
    lines.each_with_index do |line, i|
      n = post.body_line + i
      state = states[i]
      if state.is_a?(Array)
        fence_line = n
        # kramdown's GFM fence takes one word after the marker; with more it
        # renders the whole block as inline code.
        found << "#{n}: code fence info has a space; write just the language" if unquote(line).strip.match?(/\A(`{3,}|~{3,})\s*\S+\s+\S/)
      elsif state == :close
        fence_line = nil
      elsif state == :unclosed
        found << "#{fence_line}: code fence never closed (a nested fence closes the outer one)"
        fence_line = nil
        next
      end
      next if state

      # A display block over several lines opens on a line starting with $$ and
      # closes on the next line with an odd number of them: `$$` alone, or
      # `$$x = \begin{cases}` ... `\end{cases}$$`.
      # Inside a blockquote the same rules apply to the text after the markers.
      bare = strip_code(unquote(line))
      math_seen ||= bare.include?("$$")
      odd = bare.scan("$$").size.odd?
      # An inline span may wrap onto the next line, but not past the paragraph:
      # a lone `$$5` there shows as text.
      if para && bare.strip.empty?
        found << "#{para}: unbalanced $$ in the paragraph"
        para = nil
      end
      if block || (odd && !para && bare.lstrip.start_with?("$$"))
        found.concat(math_findings([line.delete("$")], n))
        if odd && block
          found << "#{n}: display $$ needs a blank line after" unless after_display?(lines[i + 1])
        elsif odd && i.positive? && !unquote(lines[i - 1]).strip.empty?
          found << "#{n}: display $$ needs a blank line before"
        end
        block = (block ? nil : n) if odd
        next
      end
      para = (para ? nil : n) if odd
      found << "#{n}: H1 in the body; the layout renders the title" if line.match?(/\A# \S/)
      in_references = line.match?(REFERENCES) if line.start_with?("#")
      # ar5iv answers a figure it lost with a "NO IMAGE AVAILABLE" PNG and HTTP
      # 200, so no link check notices: 59 figures went blank that way.
      found << "#{n}: ar5iv link; use arxiv.org/html or a copy in assets/images" if bare.include?("ar5iv.")
      # kramdown keeps a javascript: href, and htmlproofer passes it.
      found << "#{n}: javascript: link; link to a page" if bare.match?(/\]\(\s*javascript:/i)
      # A whole tag, outside code and math: `i<j` and `$$a<b$$` are text.
      text = bare.gsub(/\$\$.*?\$\$/, "")
      tags = text.scan(%r{(?<!\\)<(/?)([a-zA-Z][\w-]*)(\s[^<>]*)?/?>})
      if (tag = tags.find { |close, name, attrs| !HTML_ELEMENTS.include?(name.downcase) || (close.empty? && attrs.to_s.strip.empty? && NEEDS_ATTRIBUTES.include?(name.downcase)) }&.[](1))
        found << "#{n}: <#{tag}> is read as HTML, so it does not show as text; write it in `code` or as \\<#{tag}\\>"
      end
      found.concat(prose(line, text, in_references).map { |f| "#{n}: #{f}" })
      found.concat(math(line, n, lines, i))
      found.concat(table(lines, i, n)) if table_start?(lines, i)
    end
    found << "#{block}: $$ block never closed" if block
    found << "#{fence_line}: code fence never closed (a nested fence closes the outer one)" if fence_line
    found << "#{para}: unbalanced $$ in the paragraph" if para
    # MathJax loads only on posts that ask for it; without it $$...$$ ships as TeX.
    found << "1: the post has $$ math but no use_math: true" if math_seen && post.data["use_math"] != true
    found
  end

  # Brace groups whose content is text, not math: escapes and Korean belong there.
  TEXT_GROUP = /\\(?:text\w*|mathrm|mathtt|operatorname)\{/

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
    prose = strip_code(unquote(line))
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
    # A formula between single `$`s: `$a*b*c$` renders as `$a<em>b</em>c$`. A
    # price starts with a digit, and two prices on a line leave a space before
    # the second `$`, so neither reads as one.
    elsif outside.match?(SINGLE_DOLLAR_MATH)
      found << "#{n}: math between single `$`s; write the formula as $$...$$"
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
    # \u0001, not NUL: strip removes NUL, so a cell holding only `\|` read as empty.
    text = row.strip.gsub(/#{CODE_SPAN}|\$\$.*?\$\$/) { |m| m.tr("|", "\u0001") }.gsub("\\|", "\u0001")
    text = text.delete_prefix("|").delete_suffix("|")
    text.split("|", -1).map(&:strip)
  end

  def self.table(lines, i, n)
    header = cells(lines[i])
    rows = lines[(i + 2)..].take_while { |l| l.lstrip.start_with?("|") }
    found = []
    if header.drop(1).any?(&:empty?)
      found << "#{n}: table header has an empty cell; merge a two-row header into one (\"AIME 2024 pass@1\")"
    elsif header.first&.empty?
      # An empty <th> names nothing to a screen reader; nine shipped that way.
      found << "#{n}: table header's first cell is empty; name the column (\"구분\", \"모델\")"
    end
    first = rows.first && cells(rows.first)
    if first && first[0].empty? && first.none? { |c| c.match?(/\A[-+]?[\d.,]+%?\z/) }
      found << "#{n + 2}: table row under the header is a second header row; merge it into the header"
    end
    ([lines[i]] + rows).each_with_index do |row, k|
      # kramdown splits the cell at |, and turns \| into | (a norm into an absolute value).
      next unless strip_code(row).scan(/\$\$(.*?)\$\$/).flatten.any? { |m| m.include?("|") }

      found << "#{k.zero? ? n : n + 1 + k}: a | inside $$ in a table breaks it; write \\mid, \\lvert...\\rvert or \\lVert...\\rVert"
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
  # Every file Jekyll might read as a post, subfolders included; any but .md is
  # a finding, so a post cannot dodge the lint by its extension.
  # What Jekyll skips (EntryFilter#special? and #backup?): a path or file name
  # starting with . _ # or ~ (a stray _posts/_site/ build), or ending in ~.
  posts_dir = File.join(PostLint::ROOT, "_posts")
  skipped = ->(f) { f.end_with?("~") || [f.delete_prefix("#{posts_dir}/"), File.basename(f)].any? { |p| p.match?(/\A[._#~]/) } }
  paths = ARGV.empty? ? Dir[File.join(posts_dir, "**/*")].select { |f| File.file?(f) && !skipped.call(f) }.sort : ARGV
  abort "no posts found under #{PostLint::ROOT}/_posts" if paths.empty?
  parsed = {}
  findings_by_path = paths.to_h do |path|
    findings =
      begin
        if File.extname(path) == ".md"
          post = parsed[path] = PostLint.parse(path)
          PostLint.check(post, spec)
        else
          ["1: a post is a .md file"]
        end
      rescue StandardError => e
        hint = e.is_a?(Psych::SyntaxError) ? " (quote a front-matter value that contains `: `)" : ""
        ["1: could not be checked: #{e.message}#{hint}"]
      end
    [path, findings]
  end
  # Pairs need every post; a run on named files checks those files alone.
  PostLint.pairs(parsed.values, spec).each { |path, f| findings_by_path[path].concat(f) } if ARGV.empty?
  failed = 0
  findings_by_path.each do |path, findings|
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
