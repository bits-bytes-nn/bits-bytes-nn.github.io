---
layout: page
title: Search
permalink: /search/
description: >-
  Full-text search across every post on this blog — by title, body, category,
  or tag.
main_nav: true
nav_order: 7
lang: en
stylesheet: /css/search.css
# An empty shell until JavaScript runs, so there is nothing here for a search
# engine to index: out of the sitemap, and head.html marks it noindex.
sitemap: false
---

<div id="search-container">
  <input type="search" id="search-input" data-index="{{ '/search.json' | prepend: site.baseurl }}" placeholder="Search posts by title, content, category, or tag…"
         autocomplete="off" autocapitalize="off" spellcheck="false" aria-label="Search posts" lang="en">
  <p id="search-status" class="search-status" role="status" aria-live="polite" lang="en"></p>
  <noscript><p class="search-status">Search runs in the browser and needs JavaScript.</p></noscript>
  <div id="results-container"></div>
</div>

<!-- Pinned version + SRI for reproducible, tamper-evident loading (was @latest).
     async, and search.js waits for it: a CDN that hangs or fails holds nothing
     else on the page. data-failed records an error that came before search.js
     was listening. -->
<script src="https://unpkg.com/simple-jekyll-search@1.10.0/dest/simple-jekyll-search.min.js"
	integrity="sha384-UN+lyciv8Ta643YxZ9sY2tdTSmk3KE61Qq84ZIXG9NRTbD9+NFXy38m9h6Exxx3n" crossorigin="anonymous" id="search-lib" async onerror="this.dataset.failed = ''"></script>
<script src="{{ '/js/search.js' | prepend: site.baseurl }}" defer></script>
