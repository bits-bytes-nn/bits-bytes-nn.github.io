---
layout: page
title: "Posts"
permalink: /posts/
main_nav: true
---

{% assign sorted_cats = site.categories | sort %}
{% for category in sorted_cats %}
{% capture cat %}{{ category | first }}{% endcapture %}
<div class="category-section" style="min-height: 65vh">
  <h2 id="{{cat}}" class="category-title">{{ cat | replace: '-', ' ' }}</h2>

  {% for desc in site.descriptions %}
  {% if desc.cat == cat %}
  <p class="category-description"><em>{{ desc.desc }}</em></p>
  {% endif %}
  {% endfor %}

  <ul class="posts-list">
    {% for post in site.categories[cat] %}
    <li class="post-item">
      <strong>
        <a href="{{ post.url | prepend: site.baseurl }}" class="post-link">{{ post.title }}</a>
      </strong>
      <span class="post-date">{{ post.date | date: "%B %-d, %Y"  }}</span>
    </li>
    {% endfor %}
  </ul>
</div>

{% if forloop.last == false %}
<hr class="category-divider" />
{% endif %}
{% endfor %}

<style>
  .category-section {
    margin: #{$base-spacing} 0;
    padding: #{$base-spacing};
    background: linear-gradient(
      135deg,
      rgba($light-gray, 0.85),
      rgba($white, 0.95)
    );
    border-radius: #{$base-border-radius * 2};
    box-shadow: 0 6px 12px rgba($dark-gray, 0.06);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
  }

  .category-section:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 20px rgba($dark-gray, 0.1);
  }

  .category-title {
    color: $dark-gray;
    margin-bottom: #{$base-spacing};
    padding-bottom: #{$small-spacing};
    border-bottom: $base-border;
    font-family: $heading-font-family;
    line-height: $heading-line-height;
  }

  .category-description {
    color: $medium-gray;
    font-size: 0.95rem;
    margin-bottom: #{$base-spacing};
    font-family: $base-font-family;
    line-height: $base-line-height;
  }

  .posts-list {
    list-style: none;
    padding: 0;
  }

  .post-item {
    margin: #{$small-spacing} 0;
    padding: #{$small-spacing};
    border-radius: $base-border-radius;
    transition: all 0.3s ease;
    background: linear-gradient(
      135deg,
      rgba($white, 0.95),
      rgba($light-gray, 0.85)
    );
  }

  .post-item:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, rgba($blue, 0.1), rgba($white, 0.95));
    box-shadow: 0 4px 8px rgba($blue, 0.1);
  }

  .post-link {
    color: $blue;
    text-decoration: none;
    transition: all 0.3s ease;
    font-family: $heading-font-family;
    font-weight: 500;
  }

  .post-link:hover {
    color: $deep-blue;
    text-shadow: 0 0 20px rgba($blue, 0.15);
  }

  .post-date {
    color: $medium-gray;
    font-size: 0.9rem;
    margin-left: #{$small-spacing};
    font-family: $base-font-family;
  }

  .category-divider {
    margin: #{$base-spacing} 0;
    border: 0;
    height: 1px;
    background: linear-gradient(
      to right,
      rgba($light-gray, 0),
      rgba($base-border-color, 1),
      rgba($light-gray, 0)
    );
  }
</style>
