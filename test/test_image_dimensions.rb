# frozen_string_literal: true
#
# Unit tests for _plugins/image_dimensions.rb.
#
# The headers below are built by hand, so the parser is pinned to the formats
# themselves rather than to whichever files happen to be in assets/.

require "minitest/autorun"
require_relative "../_plugins/image_dimensions"

class TestImageDimensions < Minitest::Test
  def png(width, height)
    ImageDimensions::PNG_SIGNATURE + [13].pack("N") + "IHDR" + [width, height].pack("NN") + "\x08\x06\x00\x00\x00".b
  end

  # SOI, an APP0 segment to skip, then a baseline SOF0 frame header.
  def jpeg(width, height, sof: 0xC0)
    app0 = "\xFF\xE0".b + [16].pack("n") + "JFIF\x00".b + ("\x00".b * 9)
    sof_segment = [0xFF, sof].pack("CC") + [17, 8, height, width].pack("nCnn") + ("\x00".b * 10)
    "\xFF\xD8".b + app0 + sof_segment
  end

  def test_reads_a_png_header
    assert_equal [1200, 630], ImageDimensions.read(png(1200, 630))
  end

  def test_reads_a_jpeg_frame_after_skipping_other_segments
    assert_equal [1920, 600], ImageDimensions.read(jpeg(1920, 600))
  end

  def test_reads_an_svg_root_size_or_its_view_box
    assert_equal [860, 430], ImageDimensions.read(%(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1" width="860" height="430"><path stroke-width="2"/>))
    assert_equal [840, 440], ImageDimensions.read(%(<?xml version="1.0"?>\n<svg width="100%" viewBox="0 0 840 440">))
    assert_nil ImageDimensions.read(%(<svg width="100%">))
    assert_equal [120, 60], ImageDimensions.read(%(<svg width="120px" height="60px">))
    assert_equal [841, 440], ImageDimensions.read(%(<svg viewBox="0 0 840.6 440.2">))
  end

  def test_reads_a_progressive_jpeg
    assert_equal [800, 450], ImageDimensions.read(jpeg(800, 450, sof: 0xC2))
  end

  # C4 is a Huffman table, not a frame, even though it sits in the SOF range.
  def test_does_not_mistake_a_huffman_table_for_a_frame
    dht = "\xFF\xC4".b + [6].pack("n") + "\x00\x01\x02\x03".b
    bytes = "\xFF\xD8".b + dht + jpeg(640, 480).byteslice(2..)
    assert_equal [640, 480], ImageDimensions.read(bytes)
  end

  def test_returns_nil_for_other_or_truncated_data
    assert_nil ImageDimensions.read("<svg/>")
    assert_nil ImageDimensions.read(png(10, 10).byteslice(0, 20))
    assert_nil ImageDimensions.read("\xFF\xD8\xFF\xE0".b)
    assert_nil ImageDimensions.read(nil)
  end

  def test_adds_the_size_of_a_local_image
    out = ImageDimensions.apply('<img src="/assets/a.png" alt="그림">') { png(300, 200) }
    assert_equal '<img width="300" height="200" src="/assets/a.png" alt="그림">', out
  end

  def test_keeps_a_self_closing_tag_intact
    out = ImageDimensions.apply('<img src="/assets/a.png" />') { png(3, 2) }
    assert_equal '<img width="3" height="2" src="/assets/a.png" />', out
  end

  # Hand-written <img> tags in layouts and raw HTML span lines.
  def test_sizes_a_tag_that_spans_lines
    out = ImageDimensions.apply(%(<img\n  loading="eager"\n  src="/assets/logo.png"\n  alt="logo"\n/>)) { png(192, 192) }
    assert out.start_with?('<img width="192" height="192"')
    assert_includes out, 'loading="eager"'
  end

  def test_leaves_external_and_already_sized_images_alone
    external = '<img src="https://arxiv.org/x.png">'
    sized = '<img src="/assets/a.png" width="10">'
    calls = 0
    out = ImageDimensions.apply(external + sized) { calls += 1; png(1, 1) }
    assert_equal external + sized, out
    assert_equal 0, calls
  end

  def test_leaves_the_tag_alone_when_the_file_is_missing_or_unreadable
    html = '<img src="/assets/missing.png">'
    assert_equal html, ImageDimensions.apply(html) { nil }
    assert_equal html, ImageDimensions.apply(html) { "not an image" }
  end
end
