# frozen_string_literal: true
#
# Adds width/height to <img> tags whose file lives in this repo, read from the
# PNG or JPEG header.
#
# Without them the browser reserves no space for a lazy image, so the page grows
# as each one loads: text jumps under the reader, and a jump or a focus scroll
# lands short of its target. With them, `img { height: auto }` keeps the
# reserved box at the image's aspect ratio at any column width.
#
# Only local files: an external (arXiv) image's size is unknown at build time,
# and fetching every one would make the build depend on the network.
module ImageDimensions
  # An <img> that does not declare its own size, with a root-relative src.
  SIZELESS_IMG = %r{<img\b(?![^>]*\b(?:width|height)=)([^>]*?\ssrc="(/[^"]+)"[^>]*)>}i

  PNG_SIGNATURE = "\x89PNG\r\n\x1A\n".b
  # SOF markers carry the frame size. C4 (DHT), C8 (JPG) and CC (DAC) share the
  # range but are not frames.
  JPEG_SOF = (0xC0..0xCF).to_a - [0xC4, 0xC8, 0xCC]

  # [width, height] from the first bytes of a PNG or JPEG, or nil.
  def self.read(bytes)
    bytes = bytes.to_s.b
    if bytes.start_with?(PNG_SIGNATURE)
      return nil if bytes.bytesize < 24

      bytes[16, 8].unpack("NN")
    elsif bytes.start_with?("\xFF\xD8".b)
      jpeg(bytes)
    end
  end

  # `lookup` maps a src path to its bytes, or nil when there is no such file.
  def self.apply(html, &lookup)
    html.to_s.gsub(SIZELESS_IMG) do |tag|
      attrs, src = Regexp.last_match(1), Regexp.last_match(2)
      size = read(lookup.call(src))
      size ? %(<img width="#{size[0]}" height="#{size[1]}"#{attrs}>) : tag
    end
  end

  def self.jpeg(bytes)
    pos = 2
    while pos + 9 <= bytes.bytesize
      return nil unless bytes.getbyte(pos) == 0xFF

      marker = bytes.getbyte(pos + 1)
      if JPEG_SOF.include?(marker)
        height, width = bytes[pos + 5, 4].unpack("nn")
        return [width, height]
      end
      pos += 2 + bytes[pos + 2, 2].unpack1("n")
    end
    nil
  end
  private_class_method :jpeg
end

# Guarded so test/ can require this file without Jekyll loaded.
if defined?(Jekyll::Hooks)
  Jekyll::Hooks.register [:posts, :pages], :post_render do |doc|
    next unless doc.output_ext == ".html"

    site = doc.site
    doc.output = ImageDimensions.apply(doc.output) do |src|
      path = File.join(site.source, src.delete_prefix(site.baseurl.to_s))
      # A header is all `read` needs; JPEG SOF can sit past APP segments.
      File.file?(path) ? File.binread(path, 64 * 1024) : nil
    end
  end
end
