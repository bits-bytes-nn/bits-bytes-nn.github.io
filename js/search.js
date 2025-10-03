SimpleJekyllSearch({
  searchInput: document.getElementById('search-input'),
  resultsContainer: document.getElementById('results-container'),
  json: '/search.json',
  searchResultTemplate: '<div class="search-result"><h3><a href="{url}">{title}</a></h3><p><span class="post-date">{date}</span> <span class="post-category">in {category}</span></p><p>{content}</p></div>',
  noResultsText: '<p>No results found</p>',
  limit: 10,
  fuzzy: false
});
