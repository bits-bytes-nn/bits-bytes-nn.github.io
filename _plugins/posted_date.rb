# frozen_string_literal: true
#
# Sets `last_modified_at` to the day a post went up, when that is later than
# its `date:`.
#
# A paper post's `date:` is the paper's publication date: it orders the archive
# and builds the URL. The file name carries the day the post was published here.
# Left alone, the sitemap's <lastmod>, the feed's <updated> and the JSON-LD
# dateModified all fell back to `date:`, telling crawlers a post written this
# year had not changed since the paper came out. jekyll-sitemap and jekyll-feed
# already read `last_modified_at`; head.html reads it too. Posts never set it
# themselves (forbidden_keys in _data/post_conventions.yml).
require "date"

module PostedDate
  # `basename` is the post's file name; `date` its front-matter Time. Returns
  # the file-name date as a UTC Time when it is a later day, else nil.
  def self.last_modified(basename, date)
    posted = Date.strptime(basename.to_s[0, 10], "%Y-%m-%d")
    Time.utc(posted.year, posted.month, posted.day) if posted > date.to_date
  rescue ArgumentError
    nil
  end
end

# Guarded so test/ can require this file without Jekyll loaded.
if defined?(Jekyll::Hooks)
  Jekyll::Hooks.register :site, :post_read do |site|
    site.posts.docs.each do |post|
      modified = PostedDate.last_modified(File.basename(post.path), post.date)
      post.data["last_modified_at"] = modified if modified
    end
  end
end
