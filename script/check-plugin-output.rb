# frozen_string_literal: true
#
# Checks that the _plugins/ hooks that change a post's page left their mark,
# that the search index parses and holds every post, that kramdown parsed the
# post's block HTML, that no listing shows a translation beside its original,
# that every cover is credited on its page, that no page repeats an id, that
# the home pages and the feed are in posted order, that every JSON-LD block
# parses, and that no published URL went missing, says noindex, or redirects
# with noindex.
#
# test/ covers the plugins' functions, not the Jekyll hooks that apply them, and
# with a hook gone the build still passes. Jekyll itself reads the posts here,
# so which posts exist (published, future, folders, slug) is its answer, not a
# second guess from the file names.
#
# Usage: bundle exec ruby script/check-plugin-output.rb <site_dir> [--record]
# Prints one line per problem; exits 1 when there is any. --record adds every
# post's URL and redirect stub to script/published-urls.txt instead, and prints
# what it added. PUBLISHED_SITEMAP=<url> also checks every URL in that sitemap.

require "jekyll"
require "json"

site_dir = File.expand_path(ARGV.fetch(0))
root = File.expand_path("..", __dir__)
# The base path the site was built with (CI passes the Pages one), or its links
# and image paths would match nothing here.
overrides = { "source" => root, "destination" => site_dir, "quiet" => true }
overrides["baseurl"] = ENV["BASE_PATH"] if ENV["BASE_PATH"]
config = Jekyll.configuration(overrides)
site = Jekyll::Site.new(config)
site.reset
site.read
posts = site.posts.docs
abort "no posts read from #{root}/_posts" if posts.empty?

problems = []
lang = ->(post) { post.data["lang"] || site.config["lang"] }
per_lang = posts.group_by(&lang).transform_values(&:size)
metas = posts.map { |p| RelatedPosts.meta(p) }
# Where Jekyll writes a URL: a directory one to index.html, a bare one to .html.
built = lambda do |url|
  file = url.end_with?("/") ? "#{url}index.html" : File.extname(url).empty? ? "#{url}.html" : url
  File.join(site_dir, Jekyll::URL.unescape_path(file))
end


# A post's URL comes from its `date:` and categories, so an edit to either moves
# it, and legacy_urls.rb's stub moves with it: both indexed URLs then 404.
# Every URL listed here must still be built, as a page or a redirect stub.
published = File.join(root, "script/published-urls.txt")
listed = File.file?(published) ? File.readlines(published, chomp: true).grep(%r{\A/}) : []
if ARGV.include?("--record")
  urls = posts.flat_map { |p| [p.url] + Array(p.data["redirect_from"]).map { |u| Jekyll::URL.escape_path(u) } }
  File.write(published, File.read(published).lines.grep_v(%r{\A/}).join + (listed | urls).sort.join("\n") + "\n")
  puts((urls - listed).sort.map { |u| "recorded #{u}" })
  exit
end
# The live sitemap too, when CI names it (PUBLISHED_SITEMAP): a post merged
# without --record is protected from its first deploy on. Unreachable, it is
# skipped with a warning, so a network blip cannot block a deploy; the list
# above still applies.
if (sitemap = ENV["PUBLISHED_SITEMAP"])
  require "net/http"
  begin
    uri = URI(sitemap)
    response = Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https", open_timeout: 15, read_timeout: 15) { |http| http.get(uri.request_uri) }
    raise "HTTP #{response.code}" unless response.is_a?(Net::HTTPSuccess)

    live = response.body.scan(%r{<loc>https?://[^/<]+([^<]*)</loc>}).flatten.map { |path| path.delete_prefix(site.baseurl) }
    listed |= live.map { |path| path.empty? ? "/" : path }
  rescue StandardError => e
    warn "warning: #{sitemap} not checked (#{e.message})"
  end
end
listed.each do |url|
  problems << "#{url}: published but no longer built; add it to the post's redirect_from" unless File.file?(built.call(url))
end

# posted_date.rb's order: the home pages and the feed (feed.xml) list posts by
# the day each went up, not by `date:`, which on a paper post is the paper's.
# Ties fall back to `date:` then path, as Document#<=> does.
newest_first = posts.sort_by { |p| [PostedDate.posted_at(File.basename(p.path), p.date), p.date, p.path] }.reverse
home_order = newest_first.reject { |p| p.data["hidden"] }.map { |p| "#{site.baseurl}#{p.url}" }
home = Dir.glob(File.join(site_dir, "{index.html,page/*/index.html}")).sort_by { |f| f[%r{page/(\d+)/}, 1].to_i }
on_home = home.flat_map { |f| File.read(f).scan(/class="post-link" href="([^"]+)"/).flatten }
problems << "the home pages are not in posted order" unless on_home == home_order
feed_order = newest_first.first(site.config.dig("feed", "posts_limit") || posts.size).map { |p| "#{site.config["url"]}#{site.baseurl}#{p.url}" }
in_feed = File.read(File.join(site_dir, "feed.xml")).scan(%r{<link href="([^"]+)" rel="alternate" type="text/html" title=}).flatten
problems << "feed.xml is not in posted order" unless in_feed == feed_order

# search.json, with search_index.rb's plain_text: an index that does not parse
# leaves js/search.js finding nothing, and saying nothing either.
index = begin
  JSON.parse(File.read(File.join(site_dir, "search.json")))
rescue SystemCallError, JSON::ParserError
  nil
end
unless index.is_a?(Array) && index.size == posts.size && index.all? { |e| e["url"] && !e["content"].to_s.strip.empty? }
  problems << "search.json is not a list of every post with its url and text"
end

posts.each do |post|
  name = File.basename(post.path)
  path = built.call(post.url)
  next problems << "#{name}: not built at #{post.url}" unless File.file?(path)

  html = File.read(path, encoding: "UTF-8")
  # head.html marks a page noindex when it is out of the sitemap; a post never is.
  problems << "#{name}: the page says noindex" if html[%r{<head>.*?</head>}m].to_s.include?('name="robots" content="noindex"')

  # legacy_urls.rb: a stub at the pre-slug URL of a post published before the change.
  legacy = Jekyll::URL.new(template: LegacyUrls::TEMPLATE, placeholders: post.url_placeholders).to_s
  if LegacyUrls.legacy?(name) && legacy != post.url && !File.file?(built.call(legacy))
    problems << "#{name}: no redirect stub at #{Jekyll::URL.unescape_path(legacy)}"
  end
  # _layouts/redirect.html: a stub with noindex is dropped, not followed.
  Array(post.data["redirect_from"]).each do |from|
    stub = built.call(Jekyll::URL.escape_path(from))
    problems << "#{name}: the stub at #{from} says noindex" if File.file?(stub) && File.read(stub).include?("noindex")
  end

  # posted_date.rb: published on the file-name day, and a modified time no
  # earlier than a later file-name date.
  published = html[/article:published_time" content="([^"]+)"/, 1]
  unless published && Date.parse(published) == Date.parse(name[0, 10])
    problems << "#{name}: published time #{published.inspect}, posted #{name[0, 10]}"
  end
  if (posted = PostedDate.last_modified(name, post.date))
    modified = html[/article:modified_time" content="([^"]+)"/, 1]
    unless modified && Date.parse(modified) >= posted.to_date
      problems << "#{name}: modified time #{modified.inspect}, posted #{posted.to_date}"
    end
  end

  # related_posts.rb: previous/next links, given another post in the language,
  # and Related reading, given a post it ranks.
  if per_lang[lang.call(post)] > 1 && !html.include?(%(class="post-nav"))
    problems << "#{name}: no previous/next links"
  end
  if RelatedPosts.rank(RelatedPosts.meta(post), metas).any? && !html.include?(%(class="related-posts"))
    problems << "#{name}: no Related reading"
  end

  imgs = html.scan(/<img\b[^>]*>/)
  # lazy_images.rb
  problems << "#{name}: an image without loading=" if imgs.any? { |t| !t.include?("loading=") }
  # image_dimensions.rb: every local image whose header it can read.
  unsized = imgs.filter_map { |t| t.match(ImageDimensions::SIZELESS_IMG)&.[](2) }.select do |src|
    file = ImageDimensions.source_file(src, root, site.baseurl)
    File.file?(file) && ImageDimensions.read(File.binread(file))
  end
  problems << "#{name}: no width=/height= on #{unsized.first}" if unsized.any?
  # scrollable_tables.rb
  if html.scan("<table").size > html.scan(%(class="table-container")).size
    problems << "#{name}: a table outside a scroll container"
  end

  # Not a hook: the cover's credit, which its CC BY license wants visible, and
  # which the layout leaves out without a word when _data/covers.yml lacks it.
  if (cover = post.data["cover"])
    source = site.data.dig("covers", cover)
    if !source.is_a?(Hash) || %w[title credit url].any? { |k| source[k].to_s.strip.empty? }
      problems << "#{name}: no title, credit and url for #{cover} in _data/covers.yml"
    elsif !html.include?(%(class="cover-credit"))
      problems << "#{name}: the cover credit is not on the page"
    end
  end

  # Not a hook: block HTML kramdown could not parse (a one-line <div>…</div>, a
  # <summary> without markdown="span") ships its closing tag as text and leaves
  # the element open over the rest of the post. A closing tag the author
  # escaped on purpose (\</div\> or &lt;/div&gt;, outside code) is text too.
  prose = html[%r{<article class="post-content">.*</article>}m].to_s.gsub(%r{<(pre|code|script)\b.*?</\1>}m, "")
  source = post.content.gsub(/^\s*(`{3,}|~{3,}).*?^\s*\1/m, "").gsub(/(`+)[^\n]*?\1/, "")
  broken = %w[blockquote center details div figcaption figure section span summary].find do |tag|
    prose.scan("&lt;/#{tag}&gt;").size > source.scan(%r{\\</#{tag}\\?>|&lt;/#{tag}&gt;}).size
  end
  problems << "#{name}: </#{broken}> shows as text; put block HTML tags on their own lines" if broken
end

# Not a hook either: a translation pair is one article, so the listings show
# only the original (_plugins/translations.rb). A listing that skips the
# `listed` filter shows the English twin again. Twins come from front matter,
# so a missing hook cannot hide them from this check.
twins = posts.select { |p| p.data["translation_id"] && lang.call(p) != site.config["lang"] }.map(&:url)
Dir.glob(File.join(site_dir, "**/*.html")).each do |file|
  html = File.read(file, encoding: "UTF-8")
  next if !html.include?("<!DOCTYPE html>") || html.include?(%(http-equiv="refresh"))

  # One id per page: a tag spelled two ways (Grouped-query- / Grouped-Query-)
  # makes two tag sections with one id, and the anchor jumps to the first.
  ids = html.gsub(%r{<(pre|code)\b.*?</\1>}m, "").scan(/\sid="([^"]+)"/).flatten
  if (dup = ids.tally.find { |_, count| count > 1 })
    problems << "#{file.delete_prefix(site_dir)}: id \"#{dup[0]}\" is used #{dup[1]} times"
  end
  # A "</script>" inside a JSON-LD value ends the block and runs what follows.
  html.scan(%r{<script type="application/ld\+json">(.*?)</script>}m).flatten.each do |json|
    JSON.parse(json)
  rescue JSON::ParserError
    problems << "#{file.delete_prefix(site_dir)}: its JSON-LD does not parse"
  end
  next if html.include?(%(property="og:type" content="article")) # a post's prev/next may name a twin

  twins.each do |url|
    problems << "#{file.delete_prefix(site_dir)}: lists the translation #{url}" if html.include?(%(href="#{site.baseurl}#{url}")) || html.include?(%(href="#{site.config["url"]}#{site.baseurl}#{url}"))
  end
end

puts problems
exit(problems.empty? ? 0 : 1)
