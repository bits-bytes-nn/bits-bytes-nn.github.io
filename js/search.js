/* Client-side search over /search.json (simple-jekyll-search).
 *
 * Refactor notes:
 *  - search.json carries a 40-word `snippet` for display *and* each post's full
 *    body in `content`, plus `category` and `tags`. `content` is what makes this
 *    full-text rather than title-only: no `searchFields` is configured, so
 *    simple-jekyll-search matches every field. That is deliberate — see
 *    "fix(search): index full body for Korean recall".
 *    The cost is real and should not be discovered by surprise: search.json is
 *    3.2 MB raw / ~970 KB gzip, fetched only by this page. An earlier version of
 *    this comment claimed the body had been dropped for a ~70 KB index; it never
 *    was, and the claim was wrong for months.
 *  - Keyword highlighting runs once per render via a debounced handler — the old
 *    code stacked a fresh setTimeout on every keystroke, racing itself.
 *  - A live status line reports match counts for screen readers and sighted users.
 */
(function () {
  'use strict';

  var searchInput = document.getElementById('search-input');
  var resultsContainer = document.getElementById('results-container');
  var statusEl = document.getElementById('search-status');
  if (!searchInput || !resultsContainer) return;

  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function highlight(keyword) {
    if (!keyword) return;
    var regex = new RegExp('(' + escapeRegExp(keyword) + ')', 'gi');
    var targets = resultsContainer.querySelectorAll('.search-result h3 a, .search-result .result-snippet, .search-result .result-tags');
    targets.forEach(function (el) {
      // Re-derive from the original text each time so highlights don't compound.
      var original = el.getAttribute('data-original') || el.textContent;
      el.setAttribute('data-original', original);
      el.innerHTML = original.replace(regex, '<mark>$1</mark>');
    });
  }

  function updateStatus() {
    var q = searchInput.value.trim();
    if (!q) { statusEl.textContent = ''; return; }
    var n = resultsContainer.querySelectorAll('.search-result').length;
    statusEl.textContent = n === 0
      ? 'No posts match "' + q + '".'
      : n + (n === 1 ? ' post' : ' posts') + ' found.';
  }

  // Debounce: coalesce bursts of keystrokes into one post-render pass.
  var timer = null;
  function onRendered() {
    if (timer) clearTimeout(timer);
    timer = setTimeout(function () {
      highlight(searchInput.value.trim());
      updateStatus();
    }, 120);
  }

  SimpleJekyllSearch({
    searchInput: searchInput,
    resultsContainer: resultsContainer,
    json: '/search.json',
    searchResultTemplate:
      '<div class="search-result">' +
        '<h3><a href="{url}">{title}</a></h3>' +
        '<p class="result-meta"><span class="post-date">{date}</span>' +
        '<span class="post-category">{category}</span></p>' +
        '<p class="result-snippet">{snippet}</p>' +
        '<p class="result-tags">{tags}</p>' +
      '</div>',
    noResultsText: '',
    limit: 10,
    fuzzy: false,
    success: function () {
      searchInput.addEventListener('input', onRendered);
    }
  });
})();
