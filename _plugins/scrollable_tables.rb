# frozen_string_literal: true
#
# Wraps every rendered <table> in <div class="table-container"> so a wide table
# scrolls horizontally instead of squeezing its columns to nothing.
#
# The stylesheet has defined `.table-container { overflow-x: auto }` all along
# (_sass/base/_tables.scss) but nothing ever emitted the wrapper — kramdown
# produces a bare <table>. Meanwhile `table { table-layout: fixed; width: 100% }`
# divides the available width evenly no matter how many columns there are: this
# blog ships 258 tables, and at a 375px viewport the widest (15 columns) gives
# each column ~23px, about 1.6 Hangul characters per line.
#
# Keyboard access is deliberately NOT set here. A scroll container needs
# tabindex="0" to be scrollable by keyboard in Safari and Firefox, but only when
# it actually overflows — and that is a runtime measurement. js/main.js does it
# for both tables and <pre> blocks, so tables that fit add no tab stop.
module ScrollableTables
  WRAPPER_OPEN = '<div class="table-container">'
  WRAPPER_CLOSE = "</div>"

  def self.apply(html)
    out = html.to_s
    return out unless out.include?("<table")

    out.gsub(%r{<table\b.*?</table>}m) do |table|
      "#{WRAPPER_OPEN}#{table}#{WRAPPER_CLOSE}"
    end
  end

  # True when this document has already been wrapped, so a re-run is a no-op.
  def self.wrapped?(html)
    html.to_s.include?(WRAPPER_OPEN)
  end
end

# Guarded so test/ can require this file for ScrollableTables.apply without
# Jekyll loaded, and without registering the hook.
if defined?(Jekyll::Hooks)
  Jekyll::Hooks.register [:posts, :pages], :post_render do |doc|
    next unless doc.output_ext == ".html"
    next if ScrollableTables.wrapped?(doc.output)

    doc.output = ScrollableTables.apply(doc.output)
  end
end
