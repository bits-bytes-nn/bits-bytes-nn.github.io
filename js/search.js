/* Client-side search over /search.json (simple-jekyll-search).
 *
 * The index is every post's full body — that is what makes Korean recall work,
 * since simple-jekyll-search does plain substring matching and cannot find a
 * character it never indexed. A search term is rarely in a post's opening lines,
 * so a fixed excerpt from the top would show no visible match and make correct
 * results look wrong.
 *
 * So the excerpt is cut around the match instead. That happens in
 * `templateMiddleware`, a simple-jekyll-search hook called once per {placeholder}
 * per result: it receives the full field value and returns what to render, so the
 * whole body reaches this file and only the window reaches the DOM.
 *
 * Everything returned from the middleware is inserted as HTML by the library, so
 * text is escaped here and the only tags introduced are the <mark>s.
 */
(function () {
  'use strict';

  var searchInput = document.getElementById('search-input');
  var resultsContainer = document.getElementById('results-container');
  var statusEl = document.getElementById('search-status');
  if (!searchInput || !resultsContainer) return;

  var LIMIT = 10;         // must match the `limit` passed to SimpleJekyllSearch
  var WINDOW_CHARS = 220; // length of the excerpt shown
  var LEAD_CHARS = 60;    // run-up kept before the first match, for context
  var SNAP_CHARS = 12;    // how far to reach for a space rather than cut a word

  function queryWords() {
    // The library lowercases and splits on spaces, and requires every word to
    // appear; matching that here keeps the highlight honest.
    return searchInput.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  }

  var ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };
  function escapeHtml(text) {
    return text.replace(/[&<>"]/g, function (c) { return ESCAPES[c]; });
  }

  // Every occurrence of every query word, as non-overlapping [start, end) ranges.
  // Overlaps are merged so a query like "gpt gpt-4" cannot nest one <mark> in
  // another.
  function matchRanges(text, words) {
    var lower = text.toLowerCase();
    var found = [];
    words.forEach(function (word) {
      var from = 0;
      var at = lower.indexOf(word, from);
      while (at !== -1) {
        found.push([at, at + word.length]);
        from = at + word.length;
        at = lower.indexOf(word, from);
      }
    });
    found.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });

    var merged = [];
    found.forEach(function (range) {
      var last = merged[merged.length - 1];
      if (last && range[0] <= last[1]) {
        last[1] = Math.max(last[1], range[1]);
      } else {
        merged.push([range[0], range[1]]);
      }
    });
    return merged;
  }

  // Escape `text`, wrapping each query-word hit in <mark>.
  function mark(text, words) {
    var out = '';
    var cursor = 0;
    matchRanges(text, words).forEach(function (range) {
      out += escapeHtml(text.slice(cursor, range[0])) +
             '<mark>' + escapeHtml(text.slice(range[0], range[1])) + '</mark>';
      cursor = range[1];
    });
    return out + escapeHtml(text.slice(cursor));
  }

  // Where to cut the body so the first match is visible, with context in front.
  function windowAround(text, firstMatch) {
    if (!firstMatch) {
      return { start: 0, end: Math.min(text.length, WINDOW_CHARS) };
    }
    var start = Math.max(0, firstMatch[0] - LEAD_CHARS);
    var end = Math.min(text.length, start + WINDOW_CHARS);

    // Prefer a word boundary, but only if one is within SNAP_CHARS. Korean prose
    // has no inter-word spaces, so an unbounded search for one would run past the
    // match and swallow the window; the cap makes it degrade to a clean character
    // cut. The extra guards stop a snap from hiding the match it exists to show.
    if (start > 0) {
      var next = text.indexOf(' ', start);
      if (next !== -1 && next - start <= SNAP_CHARS && next < firstMatch[0]) {
        start = next + 1;
      }
    }
    if (end < text.length) {
      var prev = text.lastIndexOf(' ', end);
      if (prev !== -1 && end - prev <= SNAP_CHARS && prev > firstMatch[1]) {
        end = prev;
      }
    }
    return { start: start, end: end };
  }

  var MAX_CANDIDATES = 40; // enough for a common word; bounds the work per result

  // With a multi-word query the earliest match is often the wrong place to look:
  // for "vibe coding" the first "coding" can be chapters away from any "vibe",
  // and the excerpt then shows one word of a two-word query. So try the window
  // around each match and keep whichever covers the most distinct words.
  function bestWindow(text, ranges, words) {
    if (!ranges.length) {
      return { start: 0, end: Math.min(text.length, WINDOW_CHARS) };
    }
    if (words.length < 2) return windowAround(text, ranges[0]);

    var best = null;
    var limit = Math.min(ranges.length, MAX_CANDIDATES);
    for (var i = 0; i < limit; i++) {
      var bounds = windowAround(text, ranges[i]);
      var slice = text.slice(bounds.start, bounds.end).toLowerCase();
      var covered = 0;
      for (var w = 0; w < words.length; w++) {
        if (slice.indexOf(words[w]) !== -1) covered++;
      }
      // Strictly greater, so ties keep the earliest window — the opening of a
      // post is likelier to be orienting prose than a passage deep inside it.
      if (!best || covered > best.covered) best = { bounds: bounds, covered: covered };
      if (best.covered === words.length) break;
    }
    return best.bounds;
  }

  function excerpt(text, words) {
    var ranges = matchRanges(text, words);
    // A post can match on its title or tags alone, with the term absent from the
    // body. Then there is no match to centre on and the opening lines are the
    // most useful thing to show.
    var bounds = bestWindow(text, ranges, words);
    var slice = text.slice(bounds.start, bounds.end);
    return (bounds.start > 0 ? '…' : '') +
           mark(slice, words) +
           (bounds.end < text.length ? '…' : '');
  }

  function updateStatus() {
    var query = searchInput.value.trim();
    if (!query) {
      statusEl.textContent = '';
      return;
    }
    var n = resultsContainer.querySelectorAll('.search-result').length;
    if (n === 0) {
      statusEl.textContent = 'No posts match "' + query + '".';
    } else if (n >= LIMIT) {
      // The library stops scanning at `limit`, so the real total is unknown —
      // saying "10 posts found" would be a guess dressed as a count.
      statusEl.textContent = 'Showing the first ' + LIMIT + ' matches.';
    } else {
      statusEl.textContent = n + (n === 1 ? ' post' : ' posts') + ' found.';
    }
  }

  SimpleJekyllSearch({
    searchInput: searchInput,
    resultsContainer: resultsContainer,
    json: searchInput.getAttribute('data-index'),
    searchResultTemplate:
      '<div class="search-result">' +
        '<h3><a href="{url}">{title}</a></h3>' +
        '<p class="result-meta"><span class="post-date">{date}</span>' +
        '<span class="post-category">{category}</span></p>' +
        '<p class="result-snippet">{content}</p>' +
        '<p class="result-tags">{tags}</p>' +
      '</div>',
    // Returning undefined leaves a field to the library's default, which is what
    // {url} and {date} want — they are attributes and plain text, not prose.
    templateMiddleware: function (prop, value) {
      var words = queryWords();
      if (prop === 'content') return excerpt(String(value == null ? '' : value), words);
      if (prop === 'title' || prop === 'tags' || prop === 'category') {
        return mark(String(value == null ? '' : value), words);
      }
      return undefined;
    },
    noResultsText: '',
    limit: LIMIT,
    fuzzy: false,
    success: function () {
      // This callback runs before the library fetches the JSON and registers its
      // own input handler, so this listener is always the earlier of the two.
      // Deferring by a turn lets the results render first, so the count is read
      // from the DOM that the visitor is actually looking at.
      searchInput.addEventListener('input', function () {
        setTimeout(updateStatus, 0);
      });
    }
  });
})();
