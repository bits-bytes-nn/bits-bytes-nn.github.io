# frozen_string_literal: true
#
# Checks that the _plugins/ hooks that change a post's page left their mark,
# and that kramdown parsed the post's block HTML.
#
# test/ covers the plugins' functions, not the Jekyll hooks that apply them, and
# with a hook gone the build still passes. Jekyll itself reads the posts here,
# so which posts exist (published, future, folders, slug) is its answer, not a
# second guess from the file names.
#
# Usage: bundle exec ruby script/check-plugin-output.rb <site_dir>
# Prints one line per problem; exits 1 when there is any.

require "jekyll"

site_dir = File.expand_path(ARGV.fetch(0))
root = File.expand_path("..", __dir__)
config = Jekyll.configuration("source" => root, "destination" => site_dir, "quiet" => true)
site = Jekyll::Site.new(config)
site.reset
site.read
posts = site.posts.docs
abort "no posts read from #{root}/_posts" if posts.empty?

problems = []
lang = ->(post) { post.data["lang"] || site.config["lang"] }
per_lang = posts.group_by(&lang).transform_values(&:size)
# Where Jekyll writes a URL: a directory one to index.html, a bare one to .html.
built = lambda do |url|
  file = url.end_with?("/") ? "#{url}index.html" : File.extname(url).empty? ? "#{url}.html" : url
  File.join(site_dir, Jekyll::URL.unescape_path(file))
end

limit = site.config.dig("feed", "posts_limit") || 10 # jekyll-feed's default
# The feed is sorted by `date:`, the paper's date, so past the cap a new review
# of an old paper never reaches it.
problems << "feed.posts_limit is #{limit} for #{posts.size} posts; raise it in _config.yml" if posts.size > limit

posts.each do |post|
  name = File.basename(post.path)
  path = built.call(post.url)
  next problems << "#{name}: not built at #{post.url}" unless File.file?(path)

  html = File.read(path, encoding: "UTF-8")

  # legacy_urls.rb: a stub at the pre-slug URL of a post published before the change.
  legacy = Jekyll::URL.new(template: LegacyUrls::TEMPLATE, placeholders: post.url_placeholders).to_s
  if LegacyUrls.legacy?(name) && legacy != post.url && !File.file?(built.call(legacy))
    problems << "#{name}: no redirect stub at #{Jekyll::URL.unescape_path(legacy)}"
  end

  # posted_date.rb: a modified time no earlier than a later file-name date.
  if (posted = PostedDate.last_modified(name, post.date))
    modified = html[/article:modified_time" content="([^"]+)"/, 1]
    unless modified && Date.parse(modified) >= posted.to_date
      problems << "#{name}: modified time #{modified.inspect}, posted #{posted.to_date}"
    end
  end

  # related_posts.rb: previous/next links, given another post in the language.
  if per_lang[lang.call(post)] > 1 && !html.include?(%(class="post-nav"))
    problems << "#{name}: no previous/next links"
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

  # Not a hook: block HTML kramdown could not parse (a one-line <div>…</div>, a
  # <summary> without markdown="span") ships its closing tag as text and leaves
  # the element open over the rest of the post. A closing tag the author
  # escaped on purpose (\</div\> or &lt;/div&gt;, outside code) is text too.
  prose = html[%r{<article class="post-content">.*</article>}m].to_s.gsub(%r{<(pre|code|script)\b.*?</\1>}m, "")
  source = post.content.gsub(/^(`{3,}|~{3,}).*?^\1/m, "").gsub(/`[^`\n]*`/, "")
  broken = %w[blockquote center details div figcaption figure section span summary].find do |tag|
    prose.scan("&lt;/#{tag}&gt;").size > source.scan(%r{\\</#{tag}\\?>|&lt;/#{tag}&gt;}).size
  end
  problems << "#{name}: </#{broken}> shows as text; put block HTML tags on their own lines" if broken
end

puts problems
exit(problems.empty? ? 0 : 1)
