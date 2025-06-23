---
layout: page
title: "Musings"
permalink: /musings/
main_nav: true
nav_order: 4
---

{% assign musing_posts = site.posts | where: "category", "musings" %}
{% if musing_posts.size > 0 %}
  <h2>Musings</h2>
  <p class="desc"><em>Personal reflections, thoughts on the future of AI, industry observations, and philosophical perspectives on technology.</em></p>
  <ul class="posts-list">
  {% for post in musing_posts limit: 15 %}
    <li>
      <strong>
        <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      </strong>
      <span class="post-date">- {{ post.date | date_to_long_string }}</span>
    </li>
  {% endfor %}
  </ul>
{% else %}
  <h2>Musings</h2>
  <p class="desc"><em>Personal reflections, thoughts on the future of AI, industry observations, and philosophical perspectives on technology.</em></p>
  <p><em>Thoughts Brewing... Personal musings and reflections on AI/ML are coming soon. Stay tuned for philosophical insights and industry observations.</em></p>
{% endif %}
<br>

