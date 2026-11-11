# frozen_string_literal: true
#
# Unit tests for _plugins/safe_post_html.rb.
#
# The check runs over every post's rendered body and fails the build, so it must
# catch each way markup can run code (as the browser reads it) and never flag
# what posts legitimately carry: images, links, video embeds, code blocks.

require "minitest/autorun"
require_relative "../_plugins/safe_post_html"

class TestSafePostHtml < Minitest::Test
  ACTIVE = [
    "<script>alert(1)</script>",
    '<img src="x" onerror="alert(1)">',
    "<svg onload=alert(1)></svg>",
    '<a href="javascript:alert(1)">x</a>',
    '<a href="java&#x73;cript:alert(1)">x</a>',
    "<a href=\"java\tscript:alert(1)\">x</a>",
    '<a href="JavaScript&colon;alert(1)">x</a>',
    '<iframe srcdoc="&lt;script&gt;alert(1)&lt;/script&gt;"></iframe>',
    '<iframe src="data:text/html,x"></iframe>',
    "<object data=\"x.swf\"></object>",
    '<form action="https://evil.example"><button>b</button></form>',
    '<button formaction="javascript:alert(1)">b</button>',
    '<meta http-equiv="refresh" content="0;url=https://evil.example">',
    '<base href="https://evil.example/">',
    "<p onmouseover=\"alert(1)\">p</p>"
  ].freeze

  def test_each_way_markup_can_run_code_is_reported
    ACTIVE.each do |html|
      refute_empty SafePostHtml.violations("<p>x</p>#{html}"), html
    end
  end

  SAFE = [
    '<img src="https://example.org/a.png" alt="그림">',
    '<img src="/assets/img/a.png" srcset="/a.png 1x, /a@2x.png 2x">',
    '<img src="data:image/png;base64,iVBORw0KGgo=" alt="점">',
    '<a href="https://example.org/onboarding">안내</a>',
    '<a href="#tldr">위로</a>',
    '<a href="/posts/x/">글</a>',
    '<a href="mailto:a@example.org">메일</a>',
    '<a href="tel:+82-2-0000-0000">전화</a>',
    '<iframe src="https://www.youtube.com/embed/x" allowfullscreen></iframe>',
    '<pre><code>&lt;script&gt;alert(1)&lt;/script&gt; &lt;img onerror=x&gt;</code></pre>',
    '<span data-on="x" title="based on=y">t</span>'
  ].freeze

  def test_what_posts_carry_is_not_reported
    SAFE.each do |html|
      assert_empty SafePostHtml.violations(html), html
    end
  end
end
