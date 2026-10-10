# frozen_string_literal: true
#
# Derives a one-paragraph description for posts that don't declare one, and
# stores it in `page.description`.
#
# Three separate renderers need a post description: the meta/OG/Twitter/JSON-LD
# block in _includes/head.html, the Atom <summary> that jekyll-feed builds, and
# the excerpt card on the home page. Jekyll's `page.excerpt` is the first block
# up to the excerpt separator — and on this blog that block is almost never prose:
#
#   * paper posts open with "### TL;DR" immediately followed by "#### <first
#     question>", so all of them produced the *same* description; and
#   * essay posts open with an H1 repeating the title, so theirs echoed the title.
#
# Deriving it once here (rather than in a Liquid filter) is what lets jekyll-feed
# see it too: the plugin cannot reach into that gem's template, but it can set the
# front-matter key the template already reads. The hook runs on :site, :pre_render,
# which fires after generators have created the feed page and before anything is
# rendered, so every consumer observes the same value.
module PostDescription
  # The two budgets, declared once. Templates call the filters by name rather than
  # passing a number, so a limit cannot drift between Ruby and Liquid.
  LIMIT = 300       # the stored description, and what the feed and home cards use
  META_LIMIT = 160  # <meta name="description"> — roughly what Google renders

  # How much of the budget a word-boundary cut must preserve before it is worth
  # taking; below this, cut hard instead. See truncate.
  MIN_BOUNDARY_FRACTION = 0.6

  # Block-level markdown that is never prose: headings, blockquotes, list items,
  # tables, fences, raw HTML, and reference definitions.
  #
  # `---`/`===` are deliberately absent. In Markdown a line of them *underlines*
  # the preceding line into a heading, so treating them as a skip would make the
  # heading above them the description. Leaving them out means such a line just
  # joins the paragraph and gets stripped later, and a thematic-break `---` is
  # already preceded by a blank line.
  #
  # The ordered-list alternative is capped at two digits: `\d+[.)]\s` also matched
  # a Korean date opening ("2024. 5. 14. 공개된 …"), silently discarding the first
  # paragraph of any post written that way.
  # `{:` is a kramdown attribute list (`{: .notice}`), which styles the block
  # next to it and is not text.
  SKIP_LINE = /\A(?:\#{1,6}\s|>|[-*+]\s|\d{1,2}[.)]\s|\||```|~~~|<|\[\^?[^\]]+\]:|\{:)/

  # A paragraph that is emphasised *end to end* is a note or caption on this blog
  # (the "*공개(Disclosure): ...*" preamble), not a summary of the post.
  #
  # The inner text may not contain the opening delimiter, so this only fires when
  # one emphasis really spans the paragraph — not on prose that merely begins and
  # ends with emphasis ("**핵심**은 데이터입니다. 그래서 **중요합니다**"). Only the
  # delimiter that opened is excluded, so a note naming `snake_case` still counts.
  WHOLLY_EMPHASISED = /\A(?:(\*{1,2})[^*]+\1|(_{1,2})[^_]+\2)\z/m

  class << self
    def derive(markdown, limit = LIMIT)
      para = first_prose_paragraph(markdown.to_s)
      return "" if para.nil?

      truncate(strip_inline(para), limit)
    end

    # Trims to a word boundary and drops trailing punctuation before the ellipsis.
    # Public so the Liquid filter can shorten an author-written description too.
    def truncate(text, limit)
      # A non-positive limit means a caller passed something Liquid coerced to 0.
      # Falling back is safer than returning the text unbounded — that path would
      # put a whole 20,000-character post body into a meta description.
      limit = LIMIT if limit <= 0
      return text if text.length <= limit

      head = text[0, limit]
      cut = head.rindex(" ")
      # Back off to the last space only when that still keeps most of the budget.
      # Not because Korean lacks spaces — it separates 어절 with them, so real
      # prose never hits this. It is a floor against degenerate input: text whose
      # only space sits near the start would otherwise collapse to a stub
      # ("짧게 " + 200 characters, cut at 100, returns "짧게…"). The test pins it.
      head = head[0, cut] if cut && cut > limit * MIN_BOUNDARY_FRACTION
      "#{head.sub(/[\s,.;:·—–-]+\z/, '')}…"
    end

    private

    def first_prose_paragraph(markdown)
      in_fence = false
      paragraph = []

      markdown.each_line do |raw|
        line = raw.rstrip
        # Block syntax is recognised indented too: a nested list item or a fence
        # under a list item is still not prose.
        lead = line.lstrip

        if lead.start_with?("```", "~~~")
          in_fence = !in_fence
          next
        end
        next if in_fence

        if line.empty?
          candidate = accept(paragraph)
          return candidate if candidate
          paragraph = []
          next
        end

        # A skippable line ends the current paragraph. If prose was already
        # accumulating, that prose is the answer — discarding it here would skip
        # past the first paragraph whenever a heading follows with no blank line
        # between them.
        if lead.match?(SKIP_LINE)
          candidate = accept(paragraph)
          return candidate if candidate
          paragraph = []
          next
        end

        paragraph << line
      end

      accept(paragraph)
    end

    # A paragraph qualifies unless it is empty, is nothing but an image or
    # math once markup is stripped, or is entirely emphasised — on this blog that
    # shape is the "*공개(Disclosure): ...*" preamble or a figure caption, never a
    # summary of the post.
    def accept(lines)
      candidate = lines.join(" ").strip
      return nil if strip_inline(candidate).empty?
      return nil if candidate.match?(WHOLLY_EMPHASISED)

      candidate
    end

    def strip_inline(text)
      out = text.dup
      out.gsub!(/\$\$.*?\$\$/m, " ")             # display math
      out.gsub!(/\\\((.*?)\\\)/m, " ")           # inline math
      out.gsub!(/`([^`]*)`/, '\1')               # code spans
      out.gsub!(/!\[[^\]]*\]\([^)]*\)/, " ")     # images
      out.gsub!(/\[([^\]]*)\]\([^)]*\)/, '\1')   # links -> link text
      out.gsub!(/\[\^[^\]]+\]/, "")              # footnote references
      # <br> is a space; every other tag closes up. Korean attaches particles
      # directly to the emphasised word, so "<em>16.7%</em>였습니다" must not
      # become "16.7% 였습니다".
      out.gsub!(%r{<br\s*/?>}i, " ")
      out.gsub!(/<[^>]+>/, "")                   # inline HTML
      out.gsub!(/(\*\*|__)(.*?)\1/m, '\2')       # bold
      out.gsub!(/(\*|_)(?=\S)(.*?)(?<=\S)\1/m, '\2') # italic
      out.gsub!(/~~(.*?)~~/m, '\1')              # strikethrough
      out.gsub(/\s+/, " ").strip
    end
  end
end

module DescriptionFilter
  # Named rather than numeric so no template carries a character budget:
  #   {{ page.description | shorten_meta }}  -> <meta name="description">
  #   {{ post.description | shorten_card }}  -> home-page excerpt cards
  # A numeric argument would also mean a typo Liquid coerced to 0 silently fell
  # back to the larger limit.
  def shorten_meta(input)
    PostDescription.truncate(input.to_s.strip, PostDescription::META_LIMIT)
  end

  def shorten_card(input)
    PostDescription.truncate(input.to_s.strip, PostDescription::LIMIT)
  end
end

# --- Jekyll wiring -----------------------------------------------------------
# Guarded so test/ can require this file for the logic above without Jekyll or
# Liquid loaded, and without registering anything.

if defined?(Jekyll::Hooks)
  # Fill in `description` before anything renders, so head.html, jekyll-feed and
  # index.html all read the same value.
  # `subtitle` is deliberately NOT consulted here. On paper posts the subtitle is
  # the Korean rendering of the English title ("Qwen3 Technical Report" -> "Qwen3
  # 기술 보고서"), so using it would restate the title — and each restatement is
  # unique, so the duplicate-description gate could not catch it. A post whose
  # subtitle *is* the best description says so with `description:`.
  Jekyll::Hooks.register :site, :pre_render do |site|
    site.posts.docs.each do |post|
      next if post.data["description"].to_s.strip != ""

      derived = PostDescription.derive(post.content)
      post.data["description"] = derived unless derived.empty?
    end
  end
end

Liquid::Template.register_filter(DescriptionFilter) if defined?(Liquid::Template)
