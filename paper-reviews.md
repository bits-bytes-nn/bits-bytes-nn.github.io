---
layout: page
title: Paper Reviews
permalink: /paper-reviews/
main_nav: true
nav_order: 3
---

{% for category in site.categories %}
  {% capture cat %}{{ category | first }}{% endcapture %}
  {% assign cat_temp = cat | replace: "-", " " | replace: "_", " " %}
  {% assign words = cat_temp | split: " " %}
  {% assign cat_display = "" %}
  {% for word in words %}
    {% capture capitalized_word %}{{ word | capitalize }}{% endcapture %}
    {% if forloop.first %}
      {% assign cat_display = capitalized_word %}
    {% else %}
      {% assign cat_display = cat_display | append: " " | append: capitalized_word %}
    {% endif %}
  {% endfor %}
  <h2 id="{{cat}}">{{ cat_display }}</h2>
  {% for desc in site.descriptions %}
    {% if desc.cat == cat %}
      <p class="desc"><em>{{ desc.desc }}</em></p>
    {% endif %}
  {% endfor %}
  <ul class="posts-list">
  {% for post in site.categories[cat] %}
    <li>
      <strong>
        <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      </strong>
      <span class="post-date">- {{ post.date | date_to_long_string }}</span>
    </li>
  {% endfor %}
  </ul>
  {% if forloop.last == false %}<hr>{% endif %}
{% endfor %}
<br>

