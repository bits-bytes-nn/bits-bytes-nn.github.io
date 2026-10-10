# frozen_string_literal: true
#
# Unit tests for _plugins/legacy_urls.rb.

require "minitest/autorun"
require_relative "../_plugins/legacy_urls"

class TestLegacyUrls < Minitest::Test
  OLD = "/paper reviews/language-models/2023/10/10/mistral-7b.html"
  NEW = "/paper-reviews/language-models/2023/10/10/mistral-7b.html"

  def test_adds_the_old_path_when_the_url_changed
    assert_equal [OLD], LegacyUrls.redirect_from(nil, OLD, NEW)
  end

  # Categories without spaces ("Insights", "Agentic-AI") slugify to themselves;
  # a redirect there would point a URL at itself.
  def test_adds_nothing_when_the_url_did_not_change
    same = "/insights/agentic-ai/2026/04/05/evolution.html"
    assert_equal [], LegacyUrls.redirect_from(nil, same, same)
  end

  def test_keeps_redirects_the_post_declares
    assert_equal ["/old-slug.html", OLD], LegacyUrls.redirect_from("/old-slug.html", OLD, NEW)
  end

  def test_does_not_duplicate_a_declared_legacy_path
    assert_equal [OLD], LegacyUrls.redirect_from([OLD], OLD, NEW)
  end
end
