# frozen_string_literal: true
#
# Decides which other posts a post links to, and exposes them to the layout:
#
#   page.related          — topically related, ranked (see `rank`)
#   page.adjacent_older   — the "← Previous" link in _layouts/post.html
#   page.adjacent_newer   — the "Next →" link
#
# Both answers obey the same two eligibility rules (`eligible?`), which is why
# they live together: never point a reader at this post's own translation, and
# never cross languages. Jekyll's built-in `page.previous`/`page.next` know
# neither rule, and a translation pair shares a date — so a Korean post's
# "← Previous" would be its own English version.
#
# A shared topic tag is required. Sharing only a category is not a reading
# recommendation: a broad subcategory like "Paper Reviews / Language-Models"
# would put LLaMA under DeepSeek-V3 with nothing actually in common. Categories
# only break ties between posts that already share a tag, and an empty block is
# better than a wrong suggestion.
#
# That strictness depends on tags that can meet. Per-paper contribution tags
# rarely do, so every post also carries a controlled topic tag (the list is in
# README) — a looser rule here is not the fix for a post with no suggestions.
#
# Jekyll's built-in site.related_posts is either "the 10 most recent posts" or
# LSI, which needs the classifier gem and a slow indexing pass. This is cheaper
# and predictable.
module RelatedPosts
  # The category terms must sum to less than TAG_WEIGHT, or they stop being
  # tie-breakers and start deciding the ranking. At 3/2/1 they summed to exactly
  # TAG_WEIGHT, so a candidate sharing one tag plus both categories (3+3) tied
  # with one sharing two tags (6) — and recency, not topic, broke the tie.
  TAG_WEIGHT = 10         # the only thing that qualifies a candidate
  SUBCATEGORY_WEIGHT = 2  # categories[1] — Agentic-AI, Language-Models, … (tie-break)
  CATEGORY_WEIGHT = 1     # categories[0] — Insights, Paper Reviews, …    (tie-break)
  LIMIT = 3

  # `target` and each `candidate` are hashes:
  #   { id:, tags: [], categories: [], lang:, translation_id:, date: }
  # Returns candidate ids, most related first.
  def self.rank(target, candidates, limit: LIMIT)
    scored = candidates.filter_map do |c|
      next if c[:id] == target[:id]
      next unless eligible?(target, c)

      shared = shared_tags(target, c)
      next if shared.empty?

      [c, score(target, c, shared)]
    end

    scored
      .sort_by { |c, s| [-s, -sort_time(c), c[:id].to_s] }
      .first(limit)
      .map { |c, _| c[:id] }
  end

  # Nearest usable post in each direction, as [older, newer]. `ordered` must be
  # sorted oldest-first. Unlike `rank` this never returns nothing for want of a
  # shared tag — date order always has a neighbour — but it does skip past
  # ineligible ones rather than dropping the link, so a post whose immediate
  # neighbour is its own translation still gets the one beyond it.
  def self.adjacent(target, ordered)
    i = ordered.index { |c| c[:id] == target[:id] }
    return [nil, nil] unless i

    older = ordered[0...i].reverse.find { |c| eligible?(target, c) }
    newer = ordered[(i + 1)..].to_a.find { |c| eligible?(target, c) }
    [older, newer]
  end

  # A translation is the same article, and the language switcher already links
  # it; a post in the other language is unreadable to whoever is here.
  def self.eligible?(target, candidate)
    return false if translation_of?(target, candidate)

    lang(candidate) == lang(target)
  end

  def self.shared_tags(a, b)
    normalise(a[:tags]) & normalise(b[:tags])
  end

  # Only meaningful for candidates that already share a tag; the category terms
  # exist to order those, not to qualify anything.
  def self.score(a, b, shared = shared_tags(a, b))
    s = shared.size * TAG_WEIGHT
    a_cats = Array(a[:categories])
    b_cats = Array(b[:categories])
    s += CATEGORY_WEIGHT if a_cats[0] && a_cats[0] == b_cats[0]
    s += SUBCATEGORY_WEIGHT if a_cats[1] && a_cats[1] == b_cats[1]
    s
  end

  def self.translation_of?(a, b)
    tid = a[:translation_id].to_s
    !tid.empty? && tid == b[:translation_id].to_s
  end

  # Posts without an explicit `lang` are Korean — that is the site default in
  # _layouts/default.html.
  def self.lang(post)
    l = post[:lang].to_s
    l.empty? ? "ko" : l
  end

  def self.normalise(tags)
    Array(tags).map { |t| t.to_s.downcase.tr("_", "-") }.reject(&:empty?).uniq
  end

  def self.sort_time(post)
    post[:date].respond_to?(:to_time) ? post[:date].to_time.to_i : 0
  end
end

# --- Jekyll wiring -----------------------------------------------------------
# Guarded so test/ can require this file for the ranking logic above without
# Jekyll loaded, and without registering the hook.

if defined?(Jekyll::Hooks)
  Jekyll::Hooks.register :site, :pre_render do |site|
    docs = site.posts.docs
    meta = docs.map do |d|
      {
        id: d.url,
        tags: d.data["tags"],
        categories: d.data["categories"],
        lang: d.data["lang"],
        translation_id: d.data["translation_id"],
        date: d.data["date"]
      }
    end
    by_url = docs.to_h { |d| [d.url, d] }
    # Sorted here rather than trusting the collection's order, and by id as well
    # as date because the three translation pairs share a date exactly.
    ordered = meta.sort_by { |m| [RelatedPosts.sort_time(m), m[:id].to_s] }

    meta.each_with_index do |target, i|
      ids = RelatedPosts.rank(target, meta)
      docs[i].data["related"] = ids.map { |id| by_url[id] }

      older, newer = RelatedPosts.adjacent(target, ordered)
      docs[i].data["adjacent_older"] = older && by_url[older[:id]]
      docs[i].data["adjacent_newer"] = newer && by_url[newer[:id]]
    end
  end
end
