---
layout: page
title: Summaries
permalink: /summaries/
main_nav: true
nav_order: 2
---

{% assign summary_posts = site.posts | where: "categories", "Summaries" %}
{% if site.categories['Summaries'].size > 0 %}
  <h2 id="summaries">Summaries</h2>
  <p class="desc"><em>Concise summaries of AI/ML research papers, blog posts, videos, and other resources, highlighting main contributions and insights.</em></p>
  <ul class="posts-list">
  {% for post in site.categories['Summaries'] %}
    <li>
      <strong>
        <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      </strong>
      <span class="post-date">- {{ post.date | date_to_long_string }}</span>
    </li>
  {% endfor %}
  </ul>
{% else %}
  <h2 id="summaries">Summaries</h2>
  <p class="desc"><em>Concise summaries of AI/ML research papers, blog posts, videos, and other resources, highlighting main contributions and insights.</em></p>
  <p><em>Coming Soon! Summaries are being prepared. Check back soon for concise insights from various AI/ML resources.</em></p>
{% endif %}
<br>
