# frozen_string_literal: true
#
# Pairs translated posts (a shared `translation_id`) and keeps one of each pair
# out of the post listings.
#
#   page.translation — the other post of the pair, for the language switcher
#                      and the hreflang links in head.html
#   page.hidden      — on the post not in the site's language, when its pair
#                      is: jekyll-paginate skips it and the `listed` filter
#                      below drops it, so a pair is listed once. It is still
#                      built and in the sitemap, the feed and search.
module Translations
  # A post's translation_id; blank is none.
  def self.id(post)
    post[:translation_id].to_s.strip.then { |s| s.empty? ? nil : s }
  end

  # `posts` are hashes of { translation_id: }. Returns { index => index of the
  # other post with its id }. script/lint-posts.rb holds a pair to one post per
  # language, so there is one other.
  def self.counterparts(posts)
    groups = posts.each_index.group_by { |i| id(posts[i]) }
    groups.delete(nil)
    groups.each_value.with_object({}) do |group, out|
      group.each { |i| (other = group.find { |j| j != i }) && out[i] = other }
    end
  end

  # `posts` are hashes of { lang:, translation_id: }. Returns the indexes of
  # the posts to hide: those not in `site_lang` whose translation_id is shared
  # by a post that is. A post with no counterpart in the site's language stays
  # listed, whatever its language. A blank id is no id.
  def self.hidden(posts, site_lang)
    listed_ids = posts.select { |p| p[:lang] == site_lang }.filter_map { |p| id(p) }
    posts.each_index.select do |i|
      p = posts[i]
      p[:lang] != site_lang && listed_ids.include?(id(p))
    end
  end

  # Liquid: `site.tags[tag] | listed`. Every listing goes through this one filter,
  # so a new listing page cannot forget the rule.
  module Filter
    def listed(posts)
      Array(posts).reject { |post| post["hidden"] }
    end
  end
end

# Guarded so test/ can require this file without Jekyll loaded. :post_read runs
# before the paginator generator, so it sees the flag.
if defined?(Jekyll::Hooks)
  Jekyll::Hooks.register :site, :post_read do |site|
    docs = site.posts.docs
    site_lang = site.config["lang"]
    posts = docs.map { |d| { lang: d.data["lang"] || site_lang, translation_id: d.data["translation_id"] } }
    Translations.hidden(posts, site_lang).each { |i| docs[i].data["hidden"] = true }
    Translations.counterparts(posts).each { |i, j| docs[i].data["translation"] = docs[j] }
  end
end

Liquid::Template.register_filter(Translations::Filter) if defined?(Liquid::Template)
