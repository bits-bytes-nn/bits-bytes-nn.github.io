# frozen_string_literal: true
#
# Gives every post `posted_at`, the day it went up here, and orders the posts by
# it. Sets `last_modified_at` to that day when it is later than `date:`.
#
# A paper post's `date:` is the paper's publication date: it builds the URL, so
# it stays. The file name carries the day the post was published here. Read
# from `date:`, a review written this year was listed below last year's posts,
# shown as "Published" in the paper's year, and dated that way in the JSON-LD,
# and the sitemap's <lastmod> and the feed's <updated> said it had not changed
# since the paper came out.
#
# The layouts show and date a post by `posted_at`; the post page also shows
# `date:` as the paper's date. jekyll-sitemap and feed.xml read
# `last_modified_at`. Posts never set it themselves (forbidden_keys in
# _data/post_conventions.yml).
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

  # The day the post went up: the later file-name date, else `date:` itself.
  def self.posted_at(basename, date)
    last_modified(basename, date) || date
  end

  # Jekyll orders posts with Document#<=>, by `date:`: site.posts, site.tags,
  # site.categories and jekyll-paginate's home pages all come from it. Posts
  # compare by `posted_at` first; anything without one, as before.
  module Order
    def <=>(other)
      mine = data["posted_at"]
      theirs = other.respond_to?(:data) ? other.data["posted_at"] : nil
      cmp = mine && theirs ? mine <=> theirs : nil
      cmp.nil? || cmp.zero? ? super : cmp
    end
  end
end

# Guarded so test/ can require this file without Jekyll loaded.
if defined?(Jekyll::Hooks)
  Jekyll::Document.prepend(PostedDate::Order)

  Jekyll::Hooks.register :site, :post_read do |site|
    site.posts.docs.each do |post|
      basename = File.basename(post.path)
      post.data["posted_at"] = PostedDate.posted_at(basename, post.date)
      modified = PostedDate.last_modified(basename, post.date)
      post.data["last_modified_at"] = modified if modified
    end
    # The reader sorted them by `date:`, before `posted_at` was set.
    site.posts.docs.sort!
  end
end
