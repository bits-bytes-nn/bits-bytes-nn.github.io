# frozen_string_literal: true
#
# Unit tests for _plugins/translations.rb.

require "minitest/autorun"
require_relative "../_plugins/translations"

class TestTranslations < Minitest::Test
  def post(lang, id = nil)
    { lang: lang, translation_id: id }
  end

  def test_hides_the_counterpart_of_a_post_in_the_site_language
    posts = [post("ko", "a"), post("en", "a")]
    assert_equal [1], Translations.hidden(posts, "ko")
  end

  def test_keeps_posts_without_a_translation
    posts = [post("ko"), post("en")]
    assert_equal [], Translations.hidden(posts, "ko")
  end

  # An English-only article is the only version there is.
  def test_keeps_a_translation_id_with_no_counterpart_in_the_site_language
    posts = [post("en", "a"), post("ko", "b")]
    assert_equal [], Translations.hidden(posts, "ko")
  end

  def test_pairs_by_id_not_by_position
    posts = [post("en", "b"), post("ko", "a"), post("en", "a"), post("ko", "b")]
    assert_equal [0, 2], Translations.hidden(posts, "ko")
  end

  def test_a_blank_id_pairs_nothing
    posts = [post("ko", ""), post("en", "")]
    assert_equal [], Translations.hidden(posts, "ko")
  end

  # The filter reads Liquid drops, which answer to string keys.
  def test_listed_drops_hidden_posts
    filter = Object.new.extend(Translations::Filter)
    posts = [{ "title" => "ko" }, { "title" => "en", "hidden" => true }]
    assert_equal [{ "title" => "ko" }], filter.listed(posts)
    assert_equal [], filter.listed(nil)
  end
end
