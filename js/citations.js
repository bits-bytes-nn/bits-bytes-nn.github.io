// A paper post's citation count, from Semantic Scholar, filled into the meta
// panel field _layouts/post.html prints for a post with `paper_doi`. Fetched in the
// browser rather than at build time, so a static page never shows a stale count.
// The keyless API shares one rate limit worldwide and often answers 429, so a
// result is kept for a day, and a failure shows the last one or nothing.
(function () {
  var el = document.querySelector('[data-paper-doi]');
  if (!el) return;

  var doi = el.getAttribute('data-paper-doi');
  var DAY = 24 * 60 * 60 * 1000;
  var key = 'citations:' + doi;
  // Semantic Scholar finds an arXiv paper by its arXiv ID, not by the DOI
  // arXiv registers for it.
  var arxiv = doi.match(/^10\.48550\/arXiv\.(.+)$/i);
  var id = arxiv ? 'arXiv:' + arxiv[1] : 'DOI:' + doi;

  // Zero is not a count worth showing: Semantic Scholar answers it for a
  // duplicate record of a paper, and it stands in for "not found" below.
  function show(entry) {
    if (!(entry.count > 0) || !entry.paperId) return;
    var count = entry.count.toLocaleString('en-US');
    var link = document.createElement('a');
    link.className = 'meta-value';
    link.title = 'Semantic Scholar';
    // `title` reaches neither touch screen readers nor the keyboard.
    link.setAttribute('aria-label', count + ' citations on Semantic Scholar');
    link.href = 'https://www.semanticscholar.org/paper/' + encodeURIComponent(entry.paperId);
    link.textContent = count;
    el.appendChild(link);
    el.hidden = false;
  }

  var cached = null;
  try { cached = JSON.parse(localStorage.getItem(key)); } catch (e) {}
  if (cached && typeof cached.count !== 'number') cached = null;
  var age = cached && Date.now() - cached.at;
  if (cached && age >= 0 && age < DAY) return show(cached); // a clock set back is not fresh

  var controller = new AbortController();
  var timer = setTimeout(function () { controller.abort(); }, 5000);
  fetch('https://api.semanticscholar.org/graph/v1/paper/' + encodeURI(id) + '?fields=citationCount',
        { signal: controller.signal })
    .then(function (res) {
      // Not found stays not found: cached as a zero, it is asked again tomorrow
      // rather than on every view. A count already known is kept instead.
      if (res.status === 404 && !cached) return { citationCount: 0 };
      if (!res.ok) throw new Error('Semantic Scholar answered ' + res.status);
      return res.json();
    })
    .then(function (data) {
      if (typeof data.citationCount !== 'number') throw new Error('no citationCount for ' + id);
      var entry = { count: data.citationCount, paperId: data.paperId, at: Date.now() };
      try { localStorage.setItem(key, JSON.stringify(entry)); } catch (e) {}
      show(entry);
    })
    .catch(function (err) {
      console.warn('Citation count:', err); // expected: the keyless API often answers 429
      if (!cached) return;
      // Kept for another day, or every view of a paper gone missing asks again.
      cached.at = Date.now();
      try { localStorage.setItem(key, JSON.stringify(cached)); } catch (e) {}
      show(cached);
    })
    .finally(function () { clearTimeout(timer); });
})();
