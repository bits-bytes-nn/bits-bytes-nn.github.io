# frozen_string_literal: true
#
# Unit tests for _plugins/lazy_images.rb.
#
# The substitution runs over every rendered post and page, so a regex that is too
# greedy would rewrite unrelated markup, and one that is too narrow would silently
# stop lazy-loading images.

require "minitest/autorun"
require_relative "../_plugins/lazy_images"

class TestLazyImages < Minitest::Test
  def test_adds_both_attributes_to_a_bare_img
    assert_equal '<img loading="lazy" decoding="async" src="/a.png">',
                 LazyImages.apply('<img src="/a.png">')
  end

  def test_preserves_existing_attributes_and_their_order
    html = '<img src="/a.png" alt="설명" class="x" width="800">'
    result = LazyImages.apply(html)
    assert_includes result, 'alt="설명"'
    assert_includes result, 'class="x"'
    assert_includes result, 'width="800"'
    assert_includes result, 'loading="lazy"'
    assert_includes result, 'decoding="async"'
  end

  # An explicit loading="eager" on an above-the-fold image must survive.
  def test_leaves_an_img_that_already_sets_loading_alone
    html = '<img loading="eager" src="/hero.png">'
    assert_equal html, LazyImages.apply(html)
  end

  def test_does_not_double_apply
    once = LazyImages.apply('<img src="/a.png">')
    assert_equal once, LazyImages.apply(once)
  end

  def test_rewrites_every_image_on_the_page
    html = '<img src="/a.png"><p>text</p><img src="/b.png">'
    assert_equal 2, LazyImages.apply(html).scan('loading="lazy"').length
  end

  def test_handles_self_closing_and_uppercase_tags
    assert_includes LazyImages.apply('<img src="/a.png" />'), 'loading="lazy"'
    assert_includes LazyImages.apply('<IMG SRC="/a.png">'), 'loading="lazy"'
  end

  # `<image>` and `<imgx>` are not `<img>`; the \b word boundary must hold.
  def test_does_not_touch_other_tags
    [
      '<image src="/a.png">',
      '<imgx src="/a.png">',
      '<div src="/a.png">'
    ].each do |html|
      assert_equal html, LazyImages.apply(html), "rewrote #{html}"
    end
  end

  def test_leaves_content_without_images_untouched
    html = "<p>이미지가 없는 문단입니다.</p>"
    assert_equal html, LazyImages.apply(html)
  end

  def test_handles_nil_and_empty_input
    assert_equal "", LazyImages.apply(nil)
    assert_equal "", LazyImages.apply("")
  end
end
