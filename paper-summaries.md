---
layout: page
title: Paper Summaries
permalink: /paper-summaries/
main_nav: true
nav_order: 2
---

{% assign summary_posts = site.posts | where: "categories", "Paper Summaries" %}
{% if site.categories['Paper Summaries'].size > 0 %}
  <h2 id="paper-summaries">Paper Summaries</h2>
  <p class="desc"><em>Concise summaries of key AI/ML research papers, highlighting main contributions and insights.</em></p>
  <ul class="posts-list">
  {% for post in site.categories['Paper Summaries'] %}
    <li>
      <strong>
        <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      </strong>
      <span class="post-date">- {{ post.date | date_to_long_string }}</span>
    </li>
  {% endfor %}
  </ul>
{% else %}
  <h2 id="paper-summaries">Paper Summaries</h2>
  <p class="desc"><em>Concise summaries of key AI/ML research papers, highlighting main contributions and insights.</em></p>
  <p><em>Coming Soon! Paper summaries are being prepared. Check back soon for concise research insights.</em></p>
{% endif %}
<br>
