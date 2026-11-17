// A paper post's citation count, in the meta panel field _layouts/post.html
// prints for a post with `paper_doi`.
// The build fetches every count in one request (script/fetch-citations.rb) and
// prints it, so a reader sees it without asking anyone. This asks Semantic
// Scholar only when neither the build nor this browser has asked for a day,
// and shows the newer of the two counts. The keyless API shares one rate limit
// worldwide and often answers 429, so a failure keeps the count shown, or none.
(function () {
  var el = document.querySelector('[data-citation-id]');
  if (!el) return;

  // arXiv:<id>, DOI:<doi> or a paperId, chosen at build (_plugins/citations.rb).
  var id = el.getAttribute('data-citation-id');
  var DAY = 24 * 60 * 60 * 1000;
  // The daily build can start late, and deploys some minutes after it asks.
  var BUILD_DAY = DAY + 6 * 60 * 60 * 1000;
  var key = 'citations:' + id;
  var now = Date.now();

  // The build's link, when it printed one; same text and label as here.
  var link = el.querySelector('a.meta-value');
  function show(entry) {
    var count = entry.count.toLocaleString('en-US');
    if (!link) {
      link = document.createElement('a');
      link.className = 'meta-value';
      link.title = 'Semantic Scholar';
      el.appendChild(link);
    }
    // `title` reaches neither touch screen readers nor the keyboard.
    link.setAttribute('aria-label', count + (entry.count === 1 ? ' citation' : ' citations') + ' on Semantic Scholar');
    link.href = 'https://www.semanticscholar.org/paper/' + encodeURIComponent(entry.paperId);
    link.textContent = count;
    el.hidden = false;
  }

  // An entry: the count, its record, when it was fetched (`at`), and, in this
  // browser's, when it last asked (`checked`), which a failure moves on too.
  var built = el.hasAttribute('data-count') ? {
    count: Number(el.getAttribute('data-count')),
    paperId: el.getAttribute('data-paper-id'),
    at: Date.parse(el.getAttribute('data-fetched-at'))
  } : null;
  var cached = null;
  try { cached = JSON.parse(localStorage.getItem(key)); } catch (e) {}
  if (!cached || typeof cached.checked !== 'number') cached = null;

  // Zero is never shown: Semantic Scholar answers it for a paper's duplicate
  // records, and it stands for "not found" here. So a zero never hides a count.
  function counted(e) { return e && e.count > 0 && e.paperId && !isNaN(e.at); }
  var newest = [built, cached].filter(counted).sort(function (a, b) { return b.at - a.at; })[0];
  if (newest && newest !== built) show(newest);

  // The build's time is the CI clock's, so a reader's clock behind it is not
  // a reason to ask; a cached time ahead of the clock (one set back) is.
  var builtAge = built ? now - built.at : NaN;
  var cachedAge = cached ? now - cached.checked : NaN;
  if (builtAge < BUILD_DAY || (cachedAge >= 0 && cachedAge < DAY)) return;

  function remember(entry) {
    try { localStorage.setItem(key, JSON.stringify(entry)); } catch (e) {}
  }
  var controller = new AbortController();
  var timer = setTimeout(function () { controller.abort(); }, 5000);
  fetch('https://api.semanticscholar.org/graph/v1/paper/' + encodeURI(id) + '?fields=citationCount',
        { signal: controller.signal })
    .then(function (res) {
      // Not found is remembered as a zero, so it is asked again tomorrow
      // rather than on every view.
      if (res.status === 404) return { citationCount: 0 };
      if (!res.ok) throw new Error('Semantic Scholar answered ' + res.status);
      return res.json();
    })
    .then(function (data) {
      if (typeof data.citationCount !== 'number') throw new Error('no citationCount for ' + id);
      var entry = { count: data.citationCount, paperId: data.paperId, at: Date.now(), checked: Date.now() };
      if (counted(entry)) {
        remember(entry);
        show(entry);
      } else {
        // Keep the count this browser had; only the time asked moves on.
        remember(counted(cached) ? { count: cached.count, paperId: cached.paperId, at: cached.at, checked: entry.checked } : entry);
      }
    })
    .catch(function (err) {
      console.warn('Citation count:', err); // expected: the keyless API often answers 429
      // Asked again tomorrow, not on every view of the paper.
      remember(cached ? { count: cached.count, paperId: cached.paperId, at: cached.at, checked: Date.now() }
                      : { count: 0, at: 0, checked: Date.now() });
    })
    .finally(function () { clearTimeout(timer); });
})();
