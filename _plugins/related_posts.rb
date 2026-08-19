# frozen_string_literal: true
#
# Picks topically related posts for each post and exposes them as `page.related`.
#
# A shared topic tag is required. Sharing only a category is not a reading
# recommendation: "Paper Reviews / Language-Models" holds 17 posts, so category
# overlap alone would put LLaMA and Llama 2 under DeepSeek-V3 with nothing
# actually in common. Categories only break ties between posts that already share
# a tag. An empty block is better than a wrong suggestion, so the 13 posts whose
# tags are unique to them simply don't get one.
#
# Jekyll's built-in site.related_posts is either "the 10 most recent posts" or
# LSI, which needs the classifier gem and a slow indexing pass. This is cheaper
# and predictable.
module RelatedPosts
  TAG_WEIGHT = 3          # the only thing that qualifies a candidate
  SUBCATEGORY_WEIGHT = 2  # categories[1] — Agentic-AI, Language-Models, … (tie-break)
  CATEGORY_WEIGHT = 1     # categories[0] — Insights, Paper Reviews, …    (tie-break)
  LIMIT = 3

  # `target` and each `candidate` are hashes:
  #   { id:, tags: [], categories: [], lang:, translation_id:, date: }
  # Returns candidate ids, most related first.
  def self.rank(target, candidates, limit: LIMIT)
    scored = candidates.filter_map do |c|
      next if c[:id] == target[:id]
      # A translation is the same article; the language switcher already links it.
      next if translation_of?(target, c)
      # Never suggest a Korean post to an English reader or the reverse.
      next unless lang(c) == lang(target)

      shared = shared_tags(target, c)
      next if shared.empty?

      [c, score(target, c, shared)]
    end

    scored
      .sort_by { |c, s| [-s, -sort_time(c), c[:id].to_s] }
      .first(limit)
      .map { |c, _| c[:id] }
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

    meta.each_with_index do |target, i|
      ids = RelatedPosts.rank(target, meta)
      docs[i].data["related"] = ids.map { |id| by_url[id] }
    end
  end
end
