# frozen_string_literal: true
#
# Derives a one-paragraph description for posts that don't declare one, and
# stores it in `page.description`.
#
# Three separate renderers need a post description: the meta/OG/Twitter/JSON-LD
# block in _includes/head.html, the Atom <summary> that jekyll-feed builds, and
# the excerpt card on the home page. Each of them used to fall back to Jekyll's
# `page.excerpt`, which is the first block up to the excerpt separator — and on
# this blog that block is almost never prose:
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
  LIMIT = 300

  # Block-level markdown that is never a description: headings, blockquotes,
  # list items, tables, fences, raw HTML, rules, and reference definitions.
  SKIP_LINE = /\A(?:\#{1,6}\s|>|[-*+]\s|\d+[.)]\s|\||```|~~~|<|---|===|\[\^?[^\]]+\]:)/

  # A paragraph that is entirely emphasised is a note or caption on this blog
  # (the "*공개(Disclosure): ...*" preamble), not a summary of the post.
  WHOLLY_EMPHASISED = /\A(\*|_){1,2}[^*_].*\1{1,2}\z/m

  class << self
    def derive(markdown, limit = LIMIT)
      para = first_prose_paragraph(markdown.to_s)
      return "" if para.nil?

      truncate(strip_inline(para), limit)
    end

    # Trims to a word boundary and drops trailing punctuation before the ellipsis.
    # Public so the Liquid filter can shorten an author-written description too.
    def truncate(text, limit)
      return text if limit <= 0 || text.length <= limit

      head = text[0, limit]
      cut = head.rindex(" ")
      # Honour the space only if it keeps most of the budget — Korean runs can go
      # a long way without one, where a hard cut reads better than a short stub.
      head = head[0, cut] if cut && cut > limit * 0.6
      "#{head.sub(/[\s,.;:·—–-]+\z/, '')}…"
    end

    private

    def first_prose_paragraph(markdown)
      in_fence = false
      paragraph = []

      markdown.each_line do |raw|
        line = raw.rstrip

        if line.start_with?("```", "~~~")
          in_fence = !in_fence
          next
        end
        next if in_fence

        if line.empty?
          candidate = paragraph.join(" ").strip
          paragraph = []
          return candidate unless candidate.empty?
          next
        end

        # A skippable line also terminates whatever was accumulating, so a
        # paragraph is never stitched across a heading or a table.
        if line.match?(SKIP_LINE)
          paragraph = []
          next
        end

        paragraph << line
      end

      candidate = paragraph.join(" ").strip
      candidate.empty? ? nil : candidate
    end

    def strip_inline(text)
      out = text.dup
      out.gsub!(/\$\$.*?\$\$/m, " ")             # display math
      out.gsub!(/\\\((.*?)\\\)/m, " ")           # inline math
      out.gsub!(/`([^`]*)`/, '\1')               # code spans
      out.gsub!(/!\[[^\]]*\]\([^)]*\)/, " ")     # images
      out.gsub!(/\[([^\]]*)\]\([^)]*\)/, '\1')   # links -> link text
      out.gsub!(/\[\^[^\]]+\]/, "")              # footnote references
      out.gsub!(/<[^>]+>/, " ")                  # inline HTML
      out.gsub!(/(\*\*|__)(.*?)\1/m, '\2')       # bold
      out.gsub!(/(\*|_)(?=\S)(.*?)(?<=\S)\1/m, '\2') # italic
      out.gsub!(/~~(.*?)~~/m, '\1')              # strikethrough
      out.gsub(/\s+/, " ").strip
    end
  end
end

# Fill in `description` before anything renders, so head.html, jekyll-feed and
# index.html all read the same value.
Jekyll::Hooks.register :site, :pre_render do |site|
  site.posts.docs.each do |post|
    next if post.data["description"].to_s.strip != ""

    # An author-written subtitle is a better pitch than anything derived.
    derived = post.data["subtitle"].to_s.strip
    derived = PostDescription.derive(post.content) if derived.empty?
    post.data["description"] = derived unless derived.empty?
  end
end

module DescriptionFilter
  # {{ page.description | shorten: 160 }} — meta descriptions want ~160 chars,
  # while the feed and the home page cards can take the full paragraph.
  def shorten(input, limit)
    PostDescription.truncate(input.to_s.strip, limit.to_i)
  end
end

Liquid::Template.register_filter(DescriptionFilter)
