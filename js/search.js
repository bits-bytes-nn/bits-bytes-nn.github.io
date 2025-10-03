function highlightKeyword(text, keyword) {
  if (!keyword || !text) return text;
  const regex = new RegExp(`(${keyword})`, 'gi');
  return text.replace(regex, '<mark>$1</mark>');
}

const searchInput = document.getElementById('search-input');
const resultsContainer = document.getElementById('results-container');

SimpleJekyllSearch({
  searchInput: searchInput,
  resultsContainer: resultsContainer,
  json: '/search.json',
  searchResultTemplate: '<div class="search-result"><h3><a href="{url}">{title}</a></h3><p><span class="post-date">{date}</span> <span class="post-category">in {category}</span></p><p>{content}</p></div>',
  noResultsText: '<p>No results found</p>',
  limit: 10,
  fuzzy: false,
  success: function() {
    searchInput.addEventListener('input', function() {
      setTimeout(function() {
        const keyword = searchInput.value.trim();
        if (keyword) {
          const results = resultsContainer.querySelectorAll('.search-result');
          results.forEach(function(result) {
            const title = result.querySelector('h3 a');
            const content = result.querySelector('p:last-child');

            if (title && title.textContent) {
              const originalTitle = title.getAttribute('data-original') || title.textContent;
              title.setAttribute('data-original', originalTitle);
              title.innerHTML = highlightKeyword(originalTitle, keyword);
            }

            if (content && content.textContent) {
              const originalContent = content.getAttribute('data-original') || content.textContent;
              content.setAttribute('data-original', originalContent);
              content.innerHTML = highlightKeyword(originalContent, keyword);
            }
          });
        }
      }, 100);
    });
  }
});
