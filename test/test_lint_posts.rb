# frozen_string_literal: true
#
# Unit tests for script/lint-posts.rb. Each case is a defect that shipped in a
# post here, cut down to the lines that show it.

require "minitest/autorun"
require_relative "../script/lint-posts"

class TestLintPosts < Minitest::Test
  SPEC = PostLint.spec

  FRONT = <<~YAML
    layout: post
    title: "Mistral 7B"
    paper_author: "Mistral AI"
    categories: ["Paper Reviews", "Language-Models"]
    tags: ["Sliding-Window-Attention", "Efficient-Attention"]
    cover: /assets/images/language-models.jpg
    use_math: true
  YAML

  def lint(body = "Text.\n", front: FRONT, path: "_posts/2025-02-27-mistral-7b.md")
    PostLint.check(PostLint.parse(path, "---\n#{front}---\n#{body}"), SPEC)
  end

  def assert_finding(pattern, findings)
    assert findings.any? { |f| f.match?(pattern) }, "expected #{pattern.inspect} in #{findings.inspect}"
  end

  def test_a_conforming_post_passes
    assert_equal [], lint("Intro.\n\n$$W_0 x$$\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
  end

  def test_rejects_author_and_a_person_list_as_paper_author
    found = lint(front: FRONT.sub('paper_author: "Mistral AI"', 'author: "Preston Rasmussen et al."'))
    assert_finding(/key `author`/, found)
    assert_finding(/key `last_modified_at`/, lint(front: "#{FRONT}last_modified_at: 2026-01-01\n"))
    assert_finding(/paper_author must name/, found)
    assert_finding(/paper_author must name/, lint(front: FRONT.sub('"Mistral AI"', '"Rasmussen et al."')))
  end

  def test_an_insights_post_takes_the_type_cover
    # Language-Models has its own image; the type's still wins.
    front = FRONT.sub('"Paper Reviews", "Language-Models"', '"Insights", "Language-Models"').sub("paper_author: \"Mistral AI\"\n", "")
    assert_equal [], lint(front: front.sub("language-models.jpg", "insights.jpg"))
    assert_finding(/cover for Insights/, lint(front: front))
  end

  def test_a_topic_without_an_image_takes_any_existing_one
    front = FRONT.sub('"Paper Reviews", "Language-Models"', '"Tech Guides", "Agentic-AI"').sub("paper_author: \"Mistral AI\"\n", "")
    assert_equal [], lint(front: front.sub("language-models.jpg", "insights.jpg"))
    assert_finding(/does not exist/, lint(front: front.sub("language-models.jpg", "nope.jpg")))
  end

  def test_rejects_an_unknown_topic_and_a_wrong_cover
    assert_finding(/categories must be/, lint(front: FRONT.sub("Language-Models", "Training-&-Inference-Optimization")))
    assert_finding(/cover for Paper Reviews/, lint(front: FRONT.sub("language-models.jpg", "default.jpg")))
  end

  def test_tags_need_a_controlled_tag_and_current_well_formed_names
    found = lint(front: FRONT.sub('"Efficient-Attention"', '"Sparse-Mixture-of-Experts", "Public-Dataset-Only Training", "Community-Detection-For-RAG"'))
    assert_finding(/no controlled topic tag/, found)
    assert_finding(/retired; use Mixture-of-Experts/, found)
    assert_finding(/joined by hyphens/, found)
    assert_finding(/should have for/, found)
  end

  def test_a_translation_is_named_after_its_original
    front = "#{FRONT}lang: en\ntranslation_id: claude-code-source-map-leak-analysis\n"
    assert_finding(/is named claude-code-source-map-leak-analysis-en/, lint(front: front, path: "_posts/2026-03-31-claude-code-architecture-analysis.md"))
    assert_equal [], lint(front: front, path: "_posts/2026-03-31-claude-code-source-map-leak-analysis-en.md")
    assert_finding(/must set lang/, lint(front: "#{FRONT}translation_id: mistral-7b\n"))
    assert_finding(/must set lang/, lint(front: "#{FRONT}translation_id: mistral-7b\nlang: kr\n"))
  end

  def test_math_escapes_currency_and_display_spacing
    assert_finding(/literal underscore/, lint("$$ e\\_i \\in \\mathcal{E}\\_e $$ text\n"))
    assert_equal [], lint("$$\\text{patch\\_embedding} = W x$$ text\n")
    assert_equal [], lint("$$\\mathrm{top\\_k} = 40$$ text\n")
    assert_finding(/Korean inside/, lint("가격을 $$2로 가정할 때 총 비용이 $$5.576M입니다.\n"))
    assert_finding(/blank line before and after/, lint("Intro\n$$y = mx$$\n"))
  end

  def test_tables_with_a_flattened_two_row_header
    sub_header = "| 모델 | AIME 2024 | |\n|---|---|---|\n| | pass@1 | cons@64 |\n| R1 | 79.8 | 86.7 |\n"
    found = lint(sub_header)
    assert_finding(/empty cell/, found)
    assert_finding(/second header row/, found)
    assert_finding(/row has 5 cells, the header 3/, lint("| 모델 | A | B |\n|---|---|---|\n| V3 | 1 | 2 | 3 | 4 |\n"))
    assert_finding(/row has 4 cells, the header 3/, lint("| 모델 | A | B |\n|---|---|---|\n| V3 | 1 | 2 | 3 |\n"))
  end

  # A blank first cell over numbers is a data row (a sub-row of the one above).
  def test_a_numeric_row_with_a_blank_first_cell_is_data
    assert_equal [], lint("| 모델 | A |\n|---|---|\n| | 85.2 |\n")
  end

  def test_pipes_inside_code_and_math_are_not_cell_breaks
    assert_equal [], lint("| Flag | Tool |\n|---|---|\n| `A` \\|\\| `B` | `$$\\lvert x \\rvert$$` |\n")
  end

  def test_no_ar5iv_links
    assert_finding(/ar5iv link/, lint("![그림](https://ar5iv.labs.arxiv.org//html/2310.06825/assets/x1.png)\n"))
    assert_equal [], lint("![그림](https://arxiv.org/html/2401.04088v1/x1.png)\n")
  end

  def test_a_body_h1_outside_code
    assert_finding(/H1 in the body/, lint("# Title again\n"))
    # A blank line after the front matter still counts toward the line number.
    assert_equal ["11: H1 in the body; the layout renders the title"], lint("\n# Title again\n")
    assert_equal [], lint("```bash\n# a comment\n```\n")
  end

  # kramdown folds a tight display equation into the paragraph, list item or not.
  def test_display_math_needs_blank_lines_even_when_indented
    assert_finding(/blank line before and after/, lint("• 기존 멀티헤드 어텐션:\n   $$y = mx$$\n\nNext.\n"))
    assert_equal [], lint("Intro.\n\n$$y = mx$$\n{: .eq}\n")
  end

  def test_a_multi_line_display_block_is_checked_inside_and_around
    assert_finding(/literal underscore/, lint("Intro.\n\n$$\ne\\_i = 1\n$$\n\nTail.\n"))
    assert_finding(/blank line before/, lint("Intro\n$$\nx\n$$\n\nTail.\n"))
    assert_finding(/blank line after/, lint("Intro.\n\n$$\nx\n$$\nTail.\n"))
    assert_equal [], lint("Intro.\n\n$$\nx_i\n$$\n\nTail.\n")
    # Content on the opening and closing lines, as cases environments are written.
    assert_equal [], lint("Intro.\n\n$$y = \\begin{cases}\n1 & x > 0 \\\\\n0\n\\end{cases}$$\n\nTail.\n")
    assert_finding(/literal underscore/, lint("Intro.\n\n$$y = \\begin{cases}\ne\\_i\n\\end{cases}$$\n\nTail.\n"))
    assert_finding(/unbalanced/, lint("가격은 $$5입니다.\n\n다음 문단.\n"))
    # Two stray $$ in two paragraphs do not pair up.
    assert_equal 2, lint("가격은 $$5입니다.\n\n비용은 $$3입니다.\n").grep(/unbalanced/).size
    assert_equal [], lint("식 $$a +\n\\frac{1}{b}$$ 입니다.\n")
    assert_equal [], lint("> 인용\n>\n> $$\n> a = b\n> $$\n>\n> 끝\n")
    assert_finding(/blank line before and after/, lint("> 문장\n> $$x=1$$\n> 문장\n"))
    assert_finding(/never closed/, lint("Intro.\n\n$$a$$ and $$b\n# Title again\n"))
    # Two spans on a line are inline math, whatever the line starts with.
    assert_equal [], lint("Intro\n$$x$$는 $$y$$보다 큽니다.\n")
  end

  def test_a_non_html_tag_renders_as_nothing
    assert_finding(/<think> is read as HTML/, lint("답변은 <think> 태그로 감쌉니다.\n"))
    assert_equal [], lint("Write ``$$`` to open math, and `` `<think>` `` in prose.\n")
    assert_finding(/<s> is read as HTML/, lint("문장 앞에 <s>를 붙입니다.\n"))
    assert_finding(/<input> is read as HTML/, lint("질문은 <input>에 넣습니다.\n"))
    assert_equal [], lint("모델은 \\<think> 로 시작합니다.\n")
    assert_finding(/<img> is read as HTML/, lint("이미지 자리에 <img> 토큰이 들어갑니다.\n"))
    assert_equal [], lint("<svg width=\"10\"><rect width=\"1\"/><path d=\"M0\"/></svg>\n")
    assert_equal [], lint("<center><svg width=\"1\"></svg></center> <script src=\"https://gist.github.com/x.js\"></script>\n")
    # Math is stripped first: <y> inside $$...$$ is TeX, not a tag.
    assert_equal [], lint("For $$x<y>z$$ and x<y z we sum.\n")
    assert_equal [], lint("답변은 `<think>` 또는 \\<think\\> 태그, <br> 줄바꿈, <https://a.b> 링크.\n")
  end

  def test_fences_kramdown_does_not_read_as_fences
    assert_finding(/info has a space/, lint("```python title=\"a.py\"\nx\n```\n"))
    assert_finding(/never closed/, lint("````md\nx\n```\n"))
  end

  def test_fences_close_only_on_their_own_marker
    assert_equal [], lint("````md\n```\n# not a heading\n```\n````\n")
    assert_equal [], lint("~~~\n```\n# not a heading\n~~~\n")
    assert_finding(/H1 in the body/, lint("```\ncode\n```\n# Title again\n"))
  end

  def test_math_outside_double_dollars
    assert_finding(/write \$\$/, lint("별표(\\\\(*\\\\))는 \\\\(\\phi^{*}\\\\)입니다.\n"))
    assert_finding(/write \$\$/, lint("Intro.\n\n\\\\[ y = x \\\\]\n\nTail.\n"))
    assert_finding(/single `\$`/, lint("/\\pi_{\\text{ref}}$를 초기화합니다.\n"))
    assert_finding(/single `\$`/, lint("정책 $\\pi$를 초기화합니다.\n"))
    # A line break inside display math, a price, and code are not the defect.
    assert_equal [], lint("Intro.\n\n$$a \\\\[4pt] b$$\n\nTail.\n")
    assert_equal [], lint("It cost $10.2B and `\\\\(x\\\\)` is how not to write it.\n")
  end

  def test_text_inside_math_and_code_spans_is_left_alone
    assert_equal [], lint("`$$x\\_y$$` is how not to write it.\n")
    assert_equal [], lint("Inline $$\\text{a \\textbf{가} b}$$ here.\n")
  end

  def test_tag_edge_cases
    # YAML reads a bare 2024 as an Integer; it is checked as text, not a crash.
    assert_equal [], lint(front: FRONT.sub('"Sliding-Window-Attention"', '2024'))
    assert_equal [], lint(front: FRONT.sub('"Sliding-Window-Attention"', '"Opt-In"'))
    assert_equal [], lint(front: FRONT.sub('"Sliding-Window-Attention"', '"Leiden-Algorithm-for-Text"'))
  end

  # YAML 1.1 reads a bare `on` as true, which would drop it from the list.
  def test_spec_word_lists_are_strings
    assert SPEC["tags"]["lowercase_words"].all?(String)
    assert_finding(/should have on/, lint(front: FRONT.sub('"Sliding-Window-Attention"', '"Attention-On-Demand"')))
  end

  # scholar-lens warns on any other macro, so the list must name what head.html defines.
  def test_spec_macros_are_the_ones_mathjax_defines
    head = File.read(File.join(PostLint::ROOT, "_includes/head.html"))
    defined = head[/macros: \{(.*?)\n\s*\}/m, 1].scan(/^\s*(\w+):/).flatten
    assert_equal SPEC["math"]["defined_macros"].sort, defined.sort
  end

  def test_math_needs_use_math
    without = FRONT.sub("use_math: true\n", "")
    assert_finding(/no use_math/, lint("식 $$x$$입니다.\n", front: without))
    assert_finding(/no use_math/, lint("백틱(`` ` ``)입니다.\n\n식 $$x$$입니다.\n\n`a.yml`\n", front: without))
    assert_equal [], lint("```bash\necho $$\n```\n", front: without)
  end

  # kramdown splits a table cell at | even inside $$, and reads \| as |.
  def test_a_bar_inside_math_in_a_table
    assert_finding(/inside \$\$ in a table/, lint("| a | b |\n|---|---|\n| $$p(y|x)$$ | 2 |\n"))
    assert_finding(/inside \$\$ in a table/, lint("| $$p(y|x)$$ | b |\n|---|---|\n| 1 | 2 |\n"))
    assert_finding(/inside \$\$ in a table/, lint("| a | b |\n|---|---|\n| $$\\|\\theta\\|$$ | 2 |\n"))
    assert_equal [], lint("| Syntax | Example |\n|---|---|\n| `$$` | $$\\lvert x \\rvert$$ |\n")
  end

  def test_a_translation_is_a_pair
    ko = PostLint.parse("_posts/2026-01-01-x.md", "---\n#{FRONT}translation_id: x\nlang: ko\n---\nText.\n")
    en = PostLint.parse("_posts/2026-01-01-x-en.md", "---\n#{FRONT}translation_id: x\nlang: en\n---\nText.\n")
    bare_en = PostLint.parse("_posts/2026-01-01-x-en.md", "---\n#{FRONT}---\nText.\n")
    assert_empty PostLint.pairs([ko, en], SPEC)
    assert_match(/one post in each of ko, en/, PostLint.pairs([ko, bare_en], SPEC)[ko.path].first)
    other_ko = PostLint.parse("_posts/2026-02-01-y.md", "---\n#{FRONT}translation_id: x\nlang: ko\n---\nText.\n")
    assert_match(/one post in each of ko, en/, PostLint.pairs([ko, other_ko], SPEC)[ko.path].first)
    assert_match(/must set translation_id/, PostLint.pairs([ko, bare_en], SPEC)[bare_en.path].first)
  end

  def test_a_lightbox_link_opens_its_image
    assert_finding(/not its image/, lint(%(<a href="https://arxiv.org/abs/1" class="glightbox">\n<img src="/assets/images/a.png" alt="a" />\n</a>\n)))
    assert_equal [], lint(%(<a href="/assets/images/a.png" class="glightbox">\n<img src="/assets/images/a.png" alt="a" />\n</a>\n))
    assert_finding(/not its image/, lint(%(<a class='glightbox wide' href='https://a.b/page'><img src='/assets/images/a.png' alt='a'></a>\n)))
    assert_finding(/not its image/, lint(%(<a href="https://a.b" class="glightbox" data-glightbox="title: A -> B"><img src="/x.png" alt="a > b"></a>\n)))
    assert_finding(/not its image/, lint(%(<a href="https://a.b" class="glightbox"><img alt="A -> B" src="/x.png"></a>\n)))
    assert_finding(/not its image/, lint(%(<a href="https://a.b" class="glightbox"><IMG SRC="/x.png" alt="x"></a>\n)))
    # After a fence the finding still names its own line.
    found = lint(%(```\nx\n```\n<a href="https://a.b" class="glightbox"><img src="/x.png" alt="x"></a>\n))
    assert_match(/\A13: /, found.first)
    assert_equal [], lint(%(```html\n<a href="https://a.b" class="glightbox"><img src="/x.png" alt="x"></a>\n```\n))
  end

  def test_liquid_in_the_body_needs_raw
    assert_finding(/Liquid tag/, lint("```text\nQuestion: {{ question }}\n```\n"))
    assert_equal [], lint("{% raw %}\n```text\nQuestion: {{ question }}\n```\n{% endraw %}\n")
    assert_finding(/Liquid tag/, lint("`{% raw %}{{ q }}{% endraw %}`와 `{{ context }}`\n"))
    assert_equal [], lint("`{% raw %}{{ q }}{% endraw %}`\n")
    assert_equal [], lint("`{%- raw -%}{{ q }}{%endraw%}`\n")
    assert_equal [], lint("`{{ q }}`\n", front: "#{FRONT}render_with_liquid: false\n")
  end

  def test_dates_and_file_names
    assert_finding(/after the file name/, lint(front: "#{FRONT}date: 2025-03-01 12:00:00\n"))
    assert_finding(/after the file name/, lint(front: "#{FRONT}date: 2025-02-28\n"))
    assert_finding(/after the file name/, lint(front: "#{FRONT}date: \"2025-03-01 12:00:00\"\n"))
    assert_finding(/after the file name/, lint(front: "#{FRONT}date: 2025-02-27 20:00:00 -0800\n"))
    assert_equal [], lint(front: "#{FRONT}date: 2025-02-28 01:00:00 +0900\n")
    assert_equal [], lint(front: "#{FRONT}date: 2023-10-10 17:54:58\n")
    assert_finding(/YYYY-MM-DD-/, lint(path: "_posts/mistral-7b.md"))
  end

  def test_description_is_plain_text
    assert_finding(/plain text/, lint(front: "#{FRONT}description: \"비용은 $$O(n)$$입니다\"\n"))
    assert_finding(/plain text/, lint(front: "#{FRONT}description: \"R1은 <think> 블록을 씁니다\"\n"))
    assert_finding(/plain text/, lint(front: "#{FRONT}description: \"*강조*된 `코드`와 [링크](https://a.b)\"\n"))
    assert_finding(/plain text/, lint(front: "#{FRONT}description: \"{{ question }} 자리\"\n"))
    assert_equal [], lint(front: "#{FRONT}description: \"Mixtral 8*7B와 C++ [1]\"\n")
  end

  def test_front_matter_starts_on_line_one
    assert_raises(ArgumentError) { PostLint.parse("_posts/2025-02-27-x.md", "\n---\n#{FRONT}---\nText.\n") }
    assert_equal "Mistral 7B", PostLint.parse("_posts/2025-02-27-x.md", "\uFEFF---\n#{FRONT}---\nText.\n").data["title"]
  end

  def test_paper_author_is_one_string_without_et_al
    assert_equal [], lint(front: FRONT.sub('"Mistral AI"', '"Google DeepMind, Google Research, Stanford"'))
    assert_finding(/paper_author/, lint(front: FRONT.sub('"Mistral AI"', '["A", "B"]')))
  end

  # The Mistral, Llama 3 and Gemini posts shipped these.
  def test_translationese_and_a_colon_ending_in_prose
    assert_finding(/translationese "보여지"/, lint("그림에서 보여지는 곡선입니다.\n"))
    assert_finding(/translationese "에 있어서는"/, lint("추론에 있어서는 그렇습니다.\n"))
    assert_finding(/translationese "함에 있어"/, lint("학습함에 있어 중요합니다.\n"))
    assert_finding(/ends in a colon/, lint("다음과 같이 작동합니다:\n\n1. 하나\n"))
    # A bare "에 있어서", and a colon in a heading, a quote, a table, code or the references.
    assert_equal [], lint("이 점에 있어서 중요합니다.\n")
    assert_equal [], lint("## 정리합니다: 요약\n\n> 그는 말했다: 아니다\n\n\"이렇다: 저렇다\"라고 합니다.\n\n`a다:b`\n")
    assert_equal [], lint("| 바다: 깊이 | 1 |\n|---|---|\n| 2 | 3 |\n")
    assert_equal [], lint("### References\n\n* 보여지는 것들: 논문\n")
  end

  def test_paper_doi_is_a_bare_doi_on_paper_posts_only
    with_doi = ->(doi) { FRONT.sub("paper_author: \"Mistral AI\"\n", "\\0paper_doi: #{doi}\n") }
    assert_equal [], lint(front: with_doi.call('"10.48550/arXiv.2310.06825"'))
    assert_equal [], lint(front: with_doi.call('"10.18653/v1/2024.acl-long.1"'))
    ['"https://doi.org/10.48550/arXiv.2310.06825"', '"arXiv:2310.06825"', '""', "10.48550", '"10.1000/a?b"', '"10.1000/a#b"', '"10.48550/arXiv.2310.06825v2"'].each do |doi|
      assert_finding(/paper_doi is a bare DOI/, lint(front: with_doi.call(doi)))
    end
    guide = with_doi.call('"10.48550/arXiv.2310.06825"').sub('"Paper Reviews", "Language-Models"', '"Tech Guides", "Agentic-AI"')
    assert_finding(/paper_doi is for Paper Reviews and Paper Summaries posts/, lint(front: guide))
    assert_finding(/paper_author is for Paper Reviews and Paper Summaries posts/, lint(front: guide))
    survey = FRONT.sub('"Paper Reviews"', '"Paper Surveys"').sub("paper_author: \"Mistral AI\"\n", "")
    assert_equal [], lint(front: survey)
  end
end

