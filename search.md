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
# An empty shell until JavaScript runs, so there is nothing here for a search
# engine to index: out of the sitemap, and head.html marks it noindex.
sitemap: false
---

<div id="search-container">
  <input type="search" id="search-input" data-index="{{ '/search.json' | prepend: site.baseurl }}" placeholder="Search titles, text and tags…"
         autocomplete="off" autocapitalize="off" spellcheck="false" aria-label="Search posts" lang="en">
  <p id="search-status" class="search-status" role="status" aria-live="polite" lang="en"></p>
  <noscript><p class="search-status">Search runs in the browser and needs JavaScript.</p></noscript>
  <div id="results-container"></div>
</div>

<script src="{{ '/js/search.js' | prepend: site.baseurl }}" defer></script>
