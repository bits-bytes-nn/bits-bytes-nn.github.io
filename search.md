---
layout: page
title: Search
permalink: /search/
main_nav: true
nav_order: 5
---

<div id="search-container">
  <input type="text" id="search-input" placeholder="Search posts...">
  <div id="results-container"></div>
</div>

<script src="https://unpkg.com/simple-jekyll-search@latest/dest/simple-jekyll-search.min.js"></script>
<script>
  SimpleJekyllSearch({
    searchInput: document.getElementById('search-input'),
    resultsContainer: document.getElementById('results-container'),
    json: '/search.json',
    searchResultTemplate: '<div class="search-result"><h3><a href="{url}">{title}</a></h3><p><span class="post-date">{date}</span> <span class="post-category">in {category}</span></p><p>{content}</p></div>',
    noResultsText: '<p>No results found</p>',
    limit: 10,
    fuzzy: false
  });
</script>

<style>
#search-container {
  margin-bottom: 2em;
}

#search-input {
  width: 100%;
  padding: 0.5em;
  font-size: 1em;
  border: 1px solid #ddd;
  border-radius: 3px;
  margin-bottom: 1em;
}

#search-input:focus {
  outline: none;
  border-color: #2980b9;
}

.search-result {
  margin-bottom: 2em;
  padding-bottom: 1em;
  border-bottom: 1px solid #ddd;
}

.search-result:last-child {
  border-bottom: none;
}

.search-result h3 {
  margin: 0 0 0.5em;
}

.search-result h3 a {
  color: #333;
  text-decoration: none;
}

.search-result h3 a:hover {
  color: #2980b9;
}

.post-date {
  color: #999;
  font-size: 0.9em;
}

.post-category {
  color: #999;
  font-size: 0.9em;
}

.search-result p {
  margin: 0.5em 0;
  color: #333;
}
</style>
