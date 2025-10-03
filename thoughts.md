---
layout: page
title: Thoughts
permalink: /thoughts/
main_nav: true
nav_order: 4
---

{% assign thoughts_posts = site.posts | where: "categories", "thoughts" %}
{% if site.categories['Thoughts'].size > 0 %}
  <h2 id="thoughts">Thoughts</h2>
  <p class="desc"><em>Personal reflections, thoughts on the future of AI, industry observations, and philosophical perspectives on technology.</em></p>
  <ul class="posts-list">
  {% for post in site.categories['Thoughts'] %}
    <li>
      <strong>
        <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      </strong>
      <span class="post-date">- {{ post.date | date_to_long_string }}</span>
    </li>
  {% endfor %}
  </ul>
{% else %}
  <h2 id="thoughts">Thoughts</h2>
  <p class="desc"><em>Personal reflections, thoughts on the future of AI, industry observations, and philosophical perspectives on technology.</em></p>
  <p><em>Thoughts Brewing... Personal musings and reflections on AI/ML are coming soon. Stay tuned for philosophical insights and industry observations.</em></p>
{% endif %}
<br>
