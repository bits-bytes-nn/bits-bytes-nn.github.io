---
layout: page
title: Paper Summaries
permalink: /paper-summaries/
main_nav: true
nav_order: 2
---

<h2 id="paper-summaries">Paper Summaries</h2>
<p class="desc"><em>Concise summaries of AI/ML research papers, blog posts, videos, and other resources, highlighting main contributions and insights.</em></p>

{% if site.categories['Summaries'].size > 0 %}
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
  <p><em>Coming Soon! Paper summaries are being prepared. Check back soon for concise insights from various AI/ML resources.</em></p>
{% endif %}
<br>
