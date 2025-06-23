---
layout: page
title: "Quick Reads"
permalink: /quick-reads/
main_nav: true
nav_order: 2
---

{% assign quick_posts = site.posts | where: "category", "quick-reads" %}
{% if quick_posts.size > 0 %}
  <h2>Quick Reads</h2>
  <p class="desc"><em>Brief insights, quick tips, and bite-sized thoughts on AI/ML trends and technologies.</em></p>
  <ul class="posts-list">
  {% for post in quick_posts limit: 20 %}
    <li>
      <strong>
        <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      </strong>
      <span class="post-date">- {{ post.date | date_to_long_string }}</span>
    </li>
  {% endfor %}
  </ul>
{% else %}
  <h2>Quick Reads</h2>
  <p class="desc"><em>Brief insights, quick tips, and bite-sized thoughts on AI/ML trends and technologies.</em></p>
  <p><em>Coming Soon! Quick reads are being prepared. Check back soon for bite-sized AI/ML insights.</em></p>
{% endif %}
<br>

