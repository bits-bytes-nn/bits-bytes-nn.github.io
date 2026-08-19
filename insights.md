---
layout: page
title: Insights
permalink: /insights/
description: >-
  The threads a single paper can't hold — agentic-AI architecture, industry
  shifts, and the design decisions underneath them.
main_nav: true
nav_order: 5
---

{%- comment -%}
  No heading here. _layouts/page.html already renders the front-matter `title` as
  this page's top-level heading, so a level-2 heading repeating it printed
  "Insights" twice in a row — which is what /paper-reviews/ never did.

  A Liquid comment, not an HTML one: an HTML comment ships to the reader, and
  spelling a heading tag inside it also trips the "exactly one h1 per page" check
  in script/validate-site.sh, which counts occurrences in the served markup.
{%- endcomment -%}
<p class="desc"><em>The threads a single paper can't hold — agentic-AI architecture, industry shifts, and the design decisions underneath them.</em></p>

{% include category-posts.html category="Insights" empty="The first piece is on its way — check back soon." %}
<br>
