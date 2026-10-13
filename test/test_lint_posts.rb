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
    assert_finding(/Korean inside/, lint("가격을 $$2로 가정할 때 총 비용이 $$5.576M입니다.\n"))
    assert_finding(/blank line before and after/, lint("Intro\n$$y = mx$$\n"))
  end

  def test_tables_with_a_flattened_two_row_header
    sub_header = "| 모델 | AIME 2024 | |\n|---|---|---|\n| | pass@1 | cons@64 |\n| R1 | 79.8 | 86.7 |\n"
    found = lint(sub_header)
    assert_finding(/empty cell/, found)
    assert_finding(/second header row/, found)
    assert_finding(/row has 5 cells, the header 3/, lint("| 모델 | A | B |\n|---|---|---|\n| V3 | 1 | 2 | 3 | 4 |\n"))
  end

  def test_pipes_inside_code_and_math_are_not_cell_breaks
    assert_equal [], lint("| Flag | Tool |\n|---|---|\n| `A` \\|\\| `B` | `$$\\lvert x \\rvert$$` |\n")
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
    assert_equal [], lint("식 $$a +\n\\frac{1}{b}$$ 입니다.\n")
    assert_equal [], lint("> 인용\n>\n> $$\n> a = b\n> $$\n>\n> 끝\n")
    assert_finding(/blank line before and after/, lint("> 문장\n> $$x=1$$\n> 문장\n"))
    assert_finding(/never closed/, lint("Intro.\n\n$$a$$ and $$b\n# Title again\n"))
    # Two spans on a line are inline math, whatever the line starts with.
    assert_equal [], lint("Intro\n$$x$$는 $$y$$보다 큽니다.\n")
  end

  def test_a_non_html_tag_renders_as_nothing
    assert_finding(/<think> is read as HTML/, lint("답변은 <think> 태그로 감쌉니다.\n"))
    assert_finding(/<s> is read as HTML/, lint("문장 앞에 <s>를 붙입니다.\n"))
    assert_finding(/<input> is read as HTML/, lint("질문은 <input>에 넣습니다.\n"))
    assert_equal [], lint("모델은 \\<think> 로 시작합니다.\n")
    assert_finding(/<img> is read as HTML/, lint("이미지 자리에 <img> 토큰이 들어갑니다.\n"))
    assert_equal [], lint("<svg width=\"10\"><rect width=\"1\"/><path d=\"M0\"/></svg>\n")
    assert_equal [], lint("<center><svg width=\"1\"></svg></center> <script src=\"https://gist.github.com/x.js\"></script>\n")
    assert_equal [], lint("For $$i<j \\le n$$ and x<y z we sum.\n")
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

  def test_a_tag_is_spelled_one_way
    a = PostLint.parse("_posts/2026-01-01-a.md", "---\n#{FRONT}---\nText.\n")
    b = PostLint.parse("_posts/2026-01-02-b.md", "---\n#{FRONT.sub('"Sliding-Window-Attention"', '"Sliding-window-Attention"')}---\nText.\n")
    assert_match(/also written Sliding-Window-Attention/, PostLint.tag_cases([a, b], SPEC)[b.path].first)
    assert_empty PostLint.tag_cases([a], SPEC)
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
    assert_finding(/after the file name/, lint(front: "#{FRONT}date: \"2025-03-01 12:00:00\"\n"))
    assert_finding(/after the file name/, lint(front: "#{FRONT}date: 2025-02-27 20:00:00 -0800\n"))
    assert_equal [], lint(front: "#{FRONT}date: 2025-02-28 01:00:00 +0900\n")
    assert_equal [], lint(front: "#{FRONT}date: 2023-10-10 17:54:58\n")
    assert_finding(/YYYY-MM-DD-/, lint(path: "_posts/mistral-7b.md"))
  end

  def test_description_is_plain_text
    assert_finding(/plain text/, lint(front: "#{FRONT}description: \"비용은 $$O(n)$$입니다\"\n"))
    assert_finding(/plain text/, lint(front: "#{FRONT}description: \"R1은 <think> 블록을 씁니다\"\n"))
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
end

