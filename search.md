---
layout: page
title: Search
permalink: /search/
main_nav: true
---

<div class="search-container" style="min-height: 60vh">
    <input type="text" id="search-input" placeholder="Enter search term..." aria-label="Search blog posts">
    <div id="results-container"></div>
</div>

<script src="https://unpkg.com/simple-jekyll-search@latest/dest/simple-jekyll-search.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('search-input');
    const resultsContainer = document.getElementById('results-container');

    function escapeRegExp(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function highlightText(element, searchTerm) {
        if (!element || !searchTerm) return;
        element.innerHTML = element.innerHTML.replace(/<\/?mark>/g, '');
        const regex = new RegExp(`(${escapeRegExp(searchTerm)})`, 'gi');
        element.innerHTML = element.innerHTML.replace(regex, '<mark>$1</mark>');
    }

    function getSnippet(text, searchTerm, snippetLength) {
        snippetLength = snippetLength || 100;
        const lowerText = text.toLowerCase();
        const lowerSearchTerm = searchTerm.toLowerCase();
        const index = lowerText.indexOf(lowerSearchTerm);
        if (index === -1) {
            return text.substring(0, snippetLength) + (text.length > snippetLength ? '...' : '');
        }
        let start = Math.max(0, index - Math.floor(snippetLength / 2));
        let end = start + snippetLength;
        if (end > text.length) {
            end = text.length;
            start = Math.max(0, end - snippetLength);
        }
        let snippet = text.substring(start, end);
        if (start > 0) snippet = '...' + snippet;
        if (end < text.length) snippet = snippet + '...';
        return snippet;
    }

    SimpleJekyllSearch({
        searchInput: searchInput,
        resultsContainer: resultsContainer,
        json: '/search.json',
        searchResultTemplate: `
            <div class="search-result">
                <h3><a href="{url}">{title}</a></h3>
                <div class="post-meta">
                    <span class="date">{date}</span>
                    <span class="category">{category}</span>
                </div>
                <div class="search-snippet" data-fullcontent="{content}">{content}</div>
            </div>
        `,
        noResultsText: '<p class="no-results">No results found</p>',
        limit: 10,
        fuzzy: false,
        success: () => {
            let debounceTimeout;
            searchInput.addEventListener('input', () => {
                clearTimeout(debounceTimeout);
                debounceTimeout = setTimeout(() => {
                    const searchTerm = searchInput.value.trim();
                    if (!searchTerm) return;

                    const results = resultsContainer.querySelectorAll('.search-result');
                    results.forEach((result) => {
                        const titleElement = result.querySelector('h3 a');
                        if (titleElement) {
                            highlightText(titleElement, searchTerm);
                        }
                        const snippetElement = result.querySelector('.search-snippet');
                        if (snippetElement) {
                            const fullText = snippetElement.getAttribute('data-fullcontent') || snippetElement.textContent;
                            let snippet = getSnippet(fullText, searchTerm, 100);
                            snippetElement.innerHTML = snippet;
                            highlightText(snippetElement, searchTerm);
                        }
                    });
                }, 300);
            });
        }
    });
});
</script>

<style>
:root {
    --primary-color: #1976D2;
    --primary-hover: #1565C0;
    --background-color: #fff;
    --shadow-color: rgba(0, 0, 0, 0.05);
    --mark-bg: #FFF9C4;
    --mark-hover-bg: #FFF176;
    --no-results-bg: #f5f5f5;
    --border-color: #e0e0e0;
    --focus-border-color: #2196F3;
}

.search-container {
    margin: 2rem auto;
    max-width: 800px;
    padding: 0 1rem;
}

#search-input {
    width: 100%;
    padding: 1rem;
    font-size: 1.1rem;
    border: 2px solid var(--border-color);
    border-radius: 8px;
    margin-bottom: 2rem;
    transition: all 0.3s ease;
}

#search-input:focus {
    outline: none;
    border-color: var(--focus-border-color);
    box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.1);
}

.search-result {
    margin-bottom: 2rem;
    padding: 1.5rem;
    border-radius: 8px;
    background-color: var(--background-color);
    box-shadow: 0 2px 8px var(--shadow-color);
    transition: transform 0.2s ease;
}

.search-result:hover {
    transform: translateY(-2px);
}

.search-result h3 {
    margin: 0 0 0.8rem;
    line-height: 1.4;
}

.search-result h3 a {
    color: var(--primary-color);
    text-decoration: none;
    transition: color 0.2s ease;
}

.search-result h3 a:hover {
    color: var(--primary-hover);
}

.post-meta {
    display: flex;
    gap: 1rem;
    font-size: 0.9rem;
    color: #666;
    margin-bottom: 1rem;
}

.post-meta span {
    display: inline-flex;
    align-items: center;
}

.post-meta .date::before {
    content: "Date:";
    margin-right: 0.4rem;
}

.post-meta .category::before {
    content: "Category:";
    margin-right: 0.4rem;
}

.search-snippet {
    font-size: 0.95rem;
    line-height: 1.6;
    color: #444;
}

mark {
    background-color: var(--mark-bg);
    color: #000;
    padding: 0.1em 0.3em;
    border-radius: 3px;
    transition: background-color 0.2s ease;
}

mark:hover {
    background-color: var(--mark-hover-bg);
}

.no-results {
    color: #666;
    font-style: italic;
    text-align: center;
    padding: 2rem;
    background-color: var(--no-results-bg);
    border-radius: 8px;
}

@media (max-width: 600px) {
    .search-container {
        padding: 0 0.5rem;
    }

    .search-result {
        padding: 1rem;
    }
}
</style>
