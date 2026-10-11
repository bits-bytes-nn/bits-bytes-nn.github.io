# frozen_string_literal: true
#
# Keeps a translated post out of the post listings.
#
# A translation pair is one article, but the home page, the category pages and
# /tags/ each listed it twice: the Korean post and its English twin, side by
# side. The post in the site's language stays; its counterpart gets `hidden`,
# which jekyll-paginate already skips and the `listed` filter below drops for
# every other listing. The counterpart is still built and in the sitemap, the
# feed and search, and the language switcher on the listed post links to it.
module Translations
  # `posts` are hashes of { lang:, translation_id: }. Returns the indexes of
  # the posts to hide: those not in `site_lang` whose translation_id is shared
  # by a post that is. A post with no counterpart in the site's language stays
  # listed, whatever its language. A blank id is no id.
  def self.hidden(posts, site_lang)
    id = ->(p) { p[:translation_id].to_s.strip.then { |s| s.empty? ? nil : s } }
    listed_ids = posts.select { |p| p[:lang] == site_lang }.filter_map(&id)
    posts.each_index.select do |i|
      p = posts[i]
      p[:lang] != site_lang && listed_ids.include?(id.call(p))
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
  end
end

Liquid::Template.register_filter(Translations::Filter) if defined?(Liquid::Template)
