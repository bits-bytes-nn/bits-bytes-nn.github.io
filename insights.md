---
layout: page
title: Insights
permalink: /insights/
main_nav: true
nav_order: 4
---

<h2 id="insights">Insights</h2>

{% if site.categories['Insights'].size > 0 %}
  <ul class="posts-list">
  {% for post in site.categories['Insights'] %}
    <li>
      <strong>
        <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      </strong>
      <span class="post-date">- {{ post.date | date_to_long_string }}</span>
    </li>
  {% endfor %}
  </ul>
{% else %}
  <p><em>Insights Brewing... Personal musings and reflections on AI/ML are coming soon. Stay tuned for philosophical insights and industry observations.</em></p>
{% endif %}
<br>
