---
layout: page
title: Paper Reviews
permalink: /paper-reviews/
main_nav: true
nav_order: 3
---

<p class="desc"><em>Deep teardowns of key AI/ML papers — LLMs, multimodal, fine-tuning, RAG — down to their design decisions and trade-offs, grouped by topic.</em></p>

{%- comment -%} Only "Paper Reviews" posts, grouped by their secondary topic — prevents other top-level categories (e.g. Insights) from leaking in via shared subcategories {%- endcomment -%}
{% assign review_posts = site.categories['Paper Reviews'] %}
{% assign topics = "" | split: "" %}
{% for post in review_posts %}
  {% for c in post.categories %}
    {% unless c == 'Paper Reviews' %}{% assign topics = topics | push: c %}{% endunless %}
  {% endfor %}
{% endfor %}
{% assign topics = topics | uniq | sort %}

{% for cat in topics %}
  {% assign cat_display = cat | replace: "-", " " | replace: "_", " " %}
  <h2 id="{{cat}}">{{ cat_display }}</h2>
  <ul class="posts-list">
  {% for post in review_posts %}
    {% if post.categories contains cat %}
    <li>
      <strong>
        <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
      </strong>
      <span class="post-date">- {{ post.date | date_to_long_string }}</span>
    </li>
    {% endif %}
  {% endfor %}
  </ul>
  {% unless forloop.last %}<hr>{% endunless %}
{% endfor %}
<br>
