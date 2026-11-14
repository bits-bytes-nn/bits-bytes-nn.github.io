# frozen_string_literal: true
#
# Fails the build when a post's rendered body holds markup that would run code
# in a reader's browser or load another document in the site's origin: an
# active element (<script>, <object>, <base>, <meta>, <form>, ...), an event
# handler or srcdoc/formaction attribute, or a URL attribute whose scheme is
# not one a post links to (http(s), mailto, tel, a relative path or anchor, or
# an inline image).
#
# It checks what kramdown produced, after conversion and before the layout, so
# it sees every way Markdown can make an attribute (raw HTML, links, images,
# attribute lists) as the browser will read it: entities decoded, attributes
# split by an HTML5 parser (Nokogiri::HTML5, as browsers parse). Code blocks
# are text by then and are not flagged. Posts are generated and then reviewed
# as pull requests, so a failing build blocks the merge (and the deploy)
# instead of silently rewriting the post.
require "nokogiri"

module SafePostHtml
  ACTIVE_ELEMENTS = %w[
    script style object embed applet base link meta form frame frameset noscript
  ].freeze
  ACTIVE_ATTRIBUTES = %w[srcdoc formaction].freeze
  URL_ATTRIBUTES = %w[
    href src action data poster background cite xlink:href srcset
  ].freeze
  SAFE_SCHEMES = %w[http https mailto tel].freeze

  # Whether +url+ is one a post may carry. Spaces and control characters a
  # browser ignores in a URL are ignored here too.
  def self.safe_url?(url, element)
    bare = url.to_s.gsub(/[\u0000- \u007f]/, "").downcase
    scheme, colon, rest = bare.partition(":")
    return true if colon.empty? || scheme.match?(%r{[/?#]})

    return true if SAFE_SCHEMES.include?(scheme)

    # An inline image: only where an image is loaded.
    scheme == "data" && rest.start_with?("image/") && %w[img source].include?(element)
  end

  # The problems of a rendered post body, as "element attribute=value" lines.
  def self.violations(html)
    found = []
    Nokogiri::HTML5::DocumentFragment.parse(html.to_s).traverse do |node|
      next unless node.element?

      name = node.name.downcase
      found << "<#{name}>" if ACTIVE_ELEMENTS.include?(name)
      node.attribute_nodes.each do |attr|
        key = attr.name.downcase
        if key.start_with?("on") || ACTIVE_ATTRIBUTES.include?(key)
          found << "<#{name} #{key}>"
        elsif URL_ATTRIBUTES.include?(key)
          urls = key == "srcset" ? attr.value.split(",").map { |u| u.strip.split(/\s+/).first.to_s } : [attr.value]
          urls.each do |url|
            found << "<#{name} #{key}=\"#{url[0, 80]}\">" unless safe_url?(url, name)
          end
        end
      end
    end
    found.uniq
  end
end

# Guarded so test/ can require this file for SafePostHtml.violations without
# Jekyll loaded, and without registering the hook.
if defined?(Jekyll::Hooks)
  Jekyll::Hooks.register :posts, :post_convert do |post|
    problems = SafePostHtml.violations(post.content)
    next if problems.empty?

    raise Jekyll::Errors::FatalException,
          "#{post.relative_path}: markup that runs code or loads a document " \
          "(#{problems.first(5).join(', ')}). Remove it or put it in a code block."
  end
end
