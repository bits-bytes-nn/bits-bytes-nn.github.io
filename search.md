---
layout: page
title: Search
permalink: /search/
main_nav: true
---

<div class="search-page">
    <div class="search-container" style="min-height: 62.5vh">
        <div class="search-box">
            <input type="text" id="search-input" placeholder="Enter search term..." aria-label="Search blog posts">
        </div>
        <div id="results-container"></div>
    </div>
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

    function getSnippet(text, searchTerm, snippetLength = 200) {
        const lowerText = text.toLowerCase();
        const lowerSearchTerm = searchTerm.toLowerCase();
        const index = lowerText.indexOf(lowerSearchTerm);

        if (index === -1) {
            return text.substring(0, snippetLength) + (text.length > snippetLength ? '...' : '');
        }

        const start = Math.max(0, index - Math.floor(snippetLength / 2));
        const end = Math.min(text.length, start + snippetLength);
        let snippet = text.substring(start, end);

        if (start > 0) snippet = '...' + snippet;
        if (end < text.length) snippet += '...';

        return snippet;
    }

    function formatDate(dateString) {
        return new Date(dateString).toLocaleDateString('en-US', {
            month: 'long',
            day: 'numeric',
            year: 'numeric'
        });
    }

    SimpleJekyllSearch({
        searchInput: searchInput,
        resultsContainer: resultsContainer,
        json: '/search.json',
        searchResultTemplate: `
            <div class="search-result">
                <h3><a href="{url}">{title}</a></h3>
                <div class="post-meta">
                    <div class="meta-line">
                        <span class="date">{date}</span>
                        <span class="category">in {category}</span>
                    </div>
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

                    resultsContainer.querySelectorAll('.search-result').forEach(result => {
                        const titleElement = result.querySelector('h3 a');
                        const snippetElement = result.querySelector('.search-snippet');
                        const dateElement = result.querySelector('.date');
                        const categoryElement = result.querySelector('.category');

                        if (titleElement) highlightText(titleElement, searchTerm);

                        if (snippetElement) {
                            const fullText = snippetElement.getAttribute('data-fullcontent') || snippetElement.textContent;
                            snippetElement.innerHTML = getSnippet(fullText, searchTerm, 200);
                            highlightText(snippetElement, searchTerm);
                        }

                        if (dateElement) {
                            dateElement.textContent = formatDate(dateElement.textContent);
                        }

                        if (categoryElement) {
                            categoryElement.textContent = categoryElement.textContent.replace(/-/g, ' ');
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
    --primary-color: #{$blue};
    --primary-hover: #{$deep-blue};
    --background-color: #{$base-background-color};
    --shadow-color: rgba(#{$dark-gray}, 0.05);
    --mark-bg: rgba(255, 235, 59, 0.4);
    --mark-hover-bg: rgba(255, 235, 59, 0.6);
    --no-results-bg: #{$light-gray};
    --border-color: #{mix($accent-purple, $light-gray, 15%)};
    --focus-border-color: #{$accent-purple};
}

.search-page {
    background: linear-gradient(135deg, #{mix($light-gray, $white, 40%)}, #{$white} 60%);
    margin: -#{$base-spacing * 2} -#{$base-spacing} -#{$base-spacing};
    padding: #{$base-spacing * 2} #{$base-spacing} #{$base-spacing};
}

.search-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 0 #{$base-spacing};
}

.search-box {
    background: #{$white};
    padding: #{$base-spacing};
    border-radius: #{$base-border-radius * 2};
    box-shadow: 0 8px 30px rgba(#{$dark-gray}, 0.08);
    margin-bottom: #{$base-spacing * 2};
}

#search-input {
    width: 100%;
    padding: #{$base-spacing * 0.8} #{$base-spacing * 1.2};
    font-size: #{$base-font-size * 1.1};
    font-family: #{$base-font-family};
    border: 2px solid #{mix($light-gray, $white, 60%)};
    border-radius: #{$base-border-radius * 1.5};
    background: #{mix($light-gray, $white, 5%)};
    transition: all 0.3s ease;
}

#search-input:focus {
    outline: none;
    border-color: var(--focus-border-color);
    box-shadow: 0 4px 12px rgba(#{$accent-purple}, 0.1);
    background: #{$white};
}

#search-input::placeholder {
    color: #{$medium-gray};
}

.search-result {
    margin-bottom: 2.5rem;
    padding: 2rem;
    border-radius: #{$base-border-radius * 3};
    background-color: var(--background-color);
    box-shadow: 0 4px 16px var(--shadow-color);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(#{$light-gray}, 0.1);
}

.search-result:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(#{$dark-gray}, 0.1);
}

.search-result h3 {
    margin: 0 0 #{$small-spacing};
    font-family: #{$heading-font-family};
    font-weight: 600;
}

.search-result h3 a {
    color: #{$dark-gray};
    text-decoration: none;
    transition: color 0.2s ease;
}

.search-result h3 a:hover {
    color: var(--primary-color);
}

.meta-line {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: #{$base-spacing};
    font-size: #{$base-font-size * 0.9};
}

.post-meta span {
    display: inline-flex;
    align-items: center;
    padding: #{$small-spacing * 0.3} #{$small-spacing * 0.8};
    background: #{mix($light-gray, $white, 30%)};
    border-radius: #{$base-border-radius};
    color: #{$medium-gray};
}

.search-snippet {
    color: #{mix($dark-gray, $medium-gray, 70%)};
    line-height: #{$base-line-height};
}

mark {
    background-color: var(--mark-bg);
    color: #{$dark-gray};
    padding: 0.1em 0.4em;
    border-radius: #{$base-border-radius};
    transition: all 0.2s ease;
    font-weight: 600;
    text-shadow: 0 1px 2px rgba(#{$dark-gray}, 0.1);
    box-shadow: 0 1px 3px rgba(#{$dark-gray}, 0.05);
}

mark:hover {
    background-color: var(--mark-hover-bg);
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(#{$dark-gray}, 0.1);
}

.no-results {
    text-align: center;
    padding: #{$base-spacing * 2};
    background: #{mix($light-gray, $white, 10%)};
    border-radius: #{$base-border-radius};
    color: #{$medium-gray};
    font-style: italic;
}

@media (max-width: 768px) {
    .search-page {
        margin: -#{$base-spacing} -#{$base-spacing} -#{$base-spacing};
        padding: #{$base-spacing} #{$base-spacing * 0.5} #{$base-spacing};
    }

    .search-container {
        padding: 0 #{$base-spacing * 0.5};
    }

    .search-box {
        padding: #{$base-spacing * 0.75};
    }

    .search-result {
        padding: #{$base-spacing};
    }

    .meta-line {
        flex-direction: column;
        gap: #{$small-spacing * 0.5};
    }
}
</style>
