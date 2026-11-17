/* Client-side search over /search.json.
 *
 * The index is every post's full body: matching is plain substring matching,
 * without stemming, so a word never indexed can never be found, and indexing
 * whole posts is what makes Korean recall work. A search term is rarely in a
 * post's opening lines, so a fixed excerpt from the top would show no visible
 * match and make correct results look wrong; the excerpt is cut around the
 * match instead.
 *
 * A post matches when every query word is in it (title, tags, category or
 * body). Matches are ranked by where the words are (WEIGHTS: a title hit beats
 * a passing mention), then newest first, and a translation pair counts once.
 *
 * Results are built as HTML here, so all text is escaped and the only tags
 * introduced are the <mark>s.
 */
(function () {
  'use strict';

  var searchInput = document.getElementById('search-input');
  var resultsContainer = document.getElementById('results-container');
  var statusEl = document.getElementById('search-status');
  if (!searchInput || !resultsContainer || !statusEl) return;

  var LIMIT = 10;         // results shown at first, and added by "Show more"
  var WINDOW_CHARS = 220; // length of the excerpt shown
  var LEAD_CHARS = 60;    // run-up kept before the first match, for context
  var SNAP_CHARS = 12;    // how far to reach for a space rather than cut a word
  // Score per query word, by where it is found. A body hit adds up to one more
  // for frequency (BODY_CAP occurrences), so a post about the word outranks one
  // that names it once, but never a title hit.
  var WEIGHTS = { title: 8, tags: 4, category: 2, body: 1 };
  var BODY_CAP = 20;

  function queryWords() {
    // Every word must appear; the highlight marks the same words.
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

    // Prefer a word boundary, but only if one is within SNAP_CHARS. A long run
    // without spaces (a URL, a formula, a CJK passage) would otherwise send the
    // search past the match and swallow the window; the cap makes it degrade to
    // a clean character cut. The extra guards stop a snap from hiding the match it exists to show.
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

  // The query in the address bar (?q=), so a search can be shared and survives
  // going back to it.
  // Debounced: Safari throws after 100 replaceState calls in 30 seconds.
  var syncTimer = null;
  function writeUrl() {
    clearTimeout(syncTimer);
    syncTimer = null;
    var url = new URL(window.location.href);
    var query = searchInput.value.trim();
    if (query) url.searchParams.set('q', query); else url.searchParams.delete('q');
    try { history.replaceState(null, '', url); } catch (e) {}
  }
  function syncUrl() {
    clearTimeout(syncTimer);
    syncTimer = setTimeout(writeUrl, 400);
  }
  // Leaving before the wait is over (a result clicked straight after typing)
  // still records the query, so Back returns to it.
  window.addEventListener('pagehide', function () { if (syncTimer !== null) writeUrl(); });

  // Up to BODY_CAP occurrences of `word` in `text` (both lowercase).
  function occurrences(text, word) {
    var n = 0;
    var at = text.indexOf(word);
    while (at !== -1 && n < BODY_CAP) {
      n++;
      at = text.indexOf(word, at + word.length);
    }
    return n;
  }

  // 0 when a word is missing from the post, else the sum of WEIGHTS for where
  // each word was found.
  function score(doc, words) {
    var total = 0;
    for (var i = 0; i < words.length; i++) {
      var w = words[i];
      var body = occurrences(doc.content, w);
      var s = (doc.title.indexOf(w) !== -1 ? WEIGHTS.title : 0) +
              (doc.tags.indexOf(w) !== -1 ? WEIGHTS.tags : 0) +
              (doc.category.indexOf(w) !== -1 ? WEIGHTS.category : 0) +
              (body ? WEIGHTS.body * (1 + body / BODY_CAP) : 0);
      if (!s) return 0;
      total += s;
    }
    return total;
  }

  var docs = [];
  // The index lowercased once, in its order (newest first), for matching.
  function prepare(index) {
    docs = index.map(function (post, order) {
      var lower = function (v) { return String(v == null ? '' : v).toLowerCase(); };
      return { post: post, order: order, title: lower(post.title), tags: lower(post.tags),
               category: lower(post.category), content: lower(post.content) };
    });
  }

  // The site's language (site.lang): of a translation pair that both match, the
  // post in it is shown, at the better score of the two; the twin only when it
  // alone matches (an English query the Korean original lacks).
  var SITE_LANG = searchInput.getAttribute('data-site-lang');

  // Matching posts, best first, newest first on a tie; a pair counts once.
  function rank(words) {
    var matched = docs
      .map(function (doc) { return { doc: doc, score: score(doc, words) }; })
      .filter(function (r) { return r.score > 0; });
    var pairs = {};
    matched.forEach(function (r) {
      var pair = r.doc.post.translation;
      if (!pair) return;
      var kept = pairs[pair];
      if (!kept) { pairs[pair] = r; return; }
      var mine = r.doc.post.lang === SITE_LANG;
      var best = Math.max(kept.score, r.score);
      if (mine) { r.score = best; pairs[pair] = r; } else { kept.score = best; }
    });
    return matched
      .filter(function (r) { var pair = r.doc.post.translation; return !pair || pairs[pair] === r; })
      .sort(function (a, b) { return b.score - a.score || a.doc.order - b.doc.order; });
  }

  function text(v) { return String(v == null ? '' : v); }

  function resultHtml(post, words) {
    var lang = escapeHtml(text(post.lang));
    return '<div class="search-result">' +
        '<h2 lang="' + lang + '"><a href="' + escapeHtml(text(post.url)) + '">' + mark(text(post.title), words) + '</a></h2>' +
        '<p class="result-meta"><span class="post-date">' + escapeHtml(text(post.date)) + '</span>' +
        '<span class="post-category">' + mark(text(post.category), words) + '</span></p>' +
        '<p class="result-snippet" lang="' + lang + '">' + excerpt(text(post.content), words) + '</p>' +
        '<p class="result-tags">' + mark(text(post.tags), words) + '</p>' +
      '</div>';
  }

  var results = [];
  var shown = LIMIT;

  function draw(words) {
    var html = results.slice(0, shown).map(function (r) { return resultHtml(r.doc.post, words); }).join('');
    var rest = results.length - shown;
    if (rest > 0) {
      html += '<button type="button" class="search-more">Show ' + Math.min(LIMIT, rest) + ' more</button>';
    }
    resultsContainer.innerHTML = html;
    announce(words);
  }

  function announce(words) {
    var query = searchInput.value.trim();
    var n = results.length;
    if (!words.length) {
      statusEl.textContent = '';
      return;
    }
    if (n === 0) {
      statusEl.textContent = words.length > 1
        ? 'No post contains every word of "' + query + '". Try fewer words.'
        : 'No post contains "' + query + '".';
    } else {
      statusEl.textContent = n + (n === 1 ? ' post' : ' posts') + ' found' +
        (n > shown ? ', showing the best ' + shown + '.' : '.');
    }
  }

  function run() {
    var words = queryWords();
    shown = LIMIT;
    if (!words.length) {
      results = [];
      resultsContainer.innerHTML = '';
      statusEl.textContent = '';
      return;
    }
    results = rank(words);
    draw(words);
  }

  resultsContainer.addEventListener('click', function (e) {
    if (!e.target.closest || !e.target.closest('.search-more')) return;
    var first = shown;
    shown += LIMIT;
    draw(queryWords());
    // Keyboard and screen-reader users continue at the first new result, not
    // back at the top.
    var next = resultsContainer.querySelectorAll('.search-result a')[first];
    if (next) next.focus();
  });

  // The index is every post's full text, about 1 MB over the wire, so it is
  // fetched when the visitor first reaches for the box, not with the page, and
  // a query typed while it downloads runs once it lands.
  var ready = false;
  var started = false;
  function start() {
    if (started) return;
    started = true;
    statusEl.textContent = 'Loading the search index…';
    fetch(searchInput.getAttribute('data-index'))
      .then(function (response) {
        if (!response.ok) throw new Error(response.status);
        return response.json();
      })
      .then(function (index) {
        prepare(index);
        ready = true;
        statusEl.textContent = '';
        if (searchInput.value.trim()) run();
      })
      .catch(function () {
        statusEl.textContent = 'Search could not load. Reload the page to try again.';
      });
  }
  ['focus', 'pointerdown', 'input'].forEach(function (type) {
    searchInput.addEventListener(type, start, { once: true });
  });

  // The results follow every keystroke, a syllable still being composed
  // included: a Korean IME (or a phone keyboard composing a whole word) keeps
  // the last one open until Space, Enter or blur.
  var runTimer;
  function scheduleRun() {
    if (!ready) return;
    clearTimeout(runTimer);
    runTimer = setTimeout(run, 80);
  }
  // The status is shown throughout but spoken (aria-live) only once the
  // syllable is done: mid-composition it would read out a half-typed one.
  searchInput.addEventListener('compositionstart', function () {
    statusEl.setAttribute('aria-live', 'off');
  });
  searchInput.addEventListener('compositionend', function () {
    statusEl.setAttribute('aria-live', 'polite');
    statusEl.textContent = ''; // so the same words are announced again
    scheduleRun();
  });
  searchInput.addEventListener('input', function () {
    scheduleRun();
    syncUrl();
  });

  var linked = new URLSearchParams(window.location.search).get('q');
  if (linked && !searchInput.value) searchInput.value = linked;
  // Typed into, or linked with a query, before this script ran: start now.
  if (document.activeElement === searchInput || searchInput.value) start();
})();
