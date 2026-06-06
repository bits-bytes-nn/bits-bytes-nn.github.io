---
layout: page
title: Search
permalink: /search/
main_nav: true
nav_order: 6
head_extra: <link rel="stylesheet" href="/css/search.css">
---

<div id="search-container">
  <input type="text" id="search-input" placeholder="Search posts by title, content, category, or tag…"
         autocomplete="off" autocapitalize="off" spellcheck="false" aria-label="Search posts">
  <p id="search-status" class="search-status" role="status" aria-live="polite"></p>
  <div id="results-container"></div>
</div>

<!-- Pinned version + SRI for reproducible, tamper-evident loading (was @latest). -->
<script src="https://unpkg.com/simple-jekyll-search@1.10.0/dest/simple-jekyll-search.min.js"
	integrity="sha384-UN+lyciv8Ta643YxZ9sY2tdTSmk3KE61Qq84ZIXG9NRTbD9+NFXy38m9h6Exxx3n" crossorigin="anonymous"></script>
<script src="/js/search.js"></script>
