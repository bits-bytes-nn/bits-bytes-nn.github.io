# frozen_string_literal: true
#
# Unit tests for _plugins/related_posts.rb.
#
# The ranking decides which internal links every post carries, so it shapes both
# the reader's path through the blog and how relevance flows between pages.

require "minitest/autorun"
require "time"
require_relative "../_plugins/related_posts"

def post(id, tags: [], categories: [], lang: nil, translation_id: nil, date: "2026-01-01")
  { id: id, tags: tags, categories: categories,
    lang: lang, translation_id: translation_id, date: Time.parse(date) }
end

class TestScore < Minitest::Test
  def test_a_shared_tag_outweighs_a_shared_subcategory
    a = post("/a", tags: %w[Harness-Engineering], categories: ["Insights", "Agentic-AI"])
    tag_match = post("/b", tags: %w[Harness-Engineering], categories: ["Insights", "Other"])
    cat_match = post("/c", tags: %w[Unrelated], categories: ["Insights", "Agentic-AI"])
    assert_operator RelatedPosts.score(a, tag_match), :>, RelatedPosts.score(a, cat_match)
  end

  # Among candidates that already share a tag, the closer category wins.
  def test_subcategory_outweighs_top_category_as_a_tie_break
    a = post("/a", tags: %w[X], categories: ["Insights", "Agentic-AI"])
    sub = post("/b", tags: %w[X], categories: ["Paper Reviews", "Agentic-AI"])
    top = post("/c", tags: %w[X], categories: ["Insights", "Language-Models"])
    assert_operator RelatedPosts.score(a, sub), :>, RelatedPosts.score(a, top)
  end

  def test_scores_add_up
    a = post("/a", tags: %w[X Y], categories: ["Insights", "Agentic-AI"])
    b = post("/b", tags: %w[X Y], categories: ["Insights", "Agentic-AI"])
    expected = (2 * RelatedPosts::TAG_WEIGHT) +
               RelatedPosts::CATEGORY_WEIGHT + RelatedPosts::SUBCATEGORY_WEIGHT
    assert_equal expected, RelatedPosts.score(a, b)
  end

  def test_unrelated_posts_score_zero
    a = post("/a", tags: %w[X], categories: ["Insights", "Agentic-AI"])
    b = post("/b", tags: %w[Y], categories: ["Paper Reviews", "Language-Models"])
    assert_equal 0, RelatedPosts.score(a, b)
  end

  def test_tag_matching_ignores_case_and_underscores
    a = post("/a", tags: ["Harness_Engineering"])
    b = post("/b", tags: ["harness-engineering"])
    assert_equal RelatedPosts::TAG_WEIGHT, RelatedPosts.score(a, b)
  end

  def test_duplicate_tags_are_not_counted_twice
    a = post("/a", tags: %w[X X])
    b = post("/b", tags: %w[X X])
    assert_equal RelatedPosts::TAG_WEIGHT, RelatedPosts.score(a, b)
  end
end

class TestRank < Minitest::Test
  def test_excludes_the_post_itself
    a = post("/a", categories: ["Insights", "Agentic-AI"])
    assert_equal [], RelatedPosts.rank(a, [a])
  end

  # The language switcher already links the twin; suggesting it as "related" is
  # just the same article again.
  def test_excludes_the_translation_twin
    ko = post("/ko", tags: %w[X], categories: ["Insights", "Agentic-AI"], lang: "ko", translation_id: "t1")
    en = post("/en", tags: %w[X], categories: ["Insights", "Agentic-AI"], lang: "en", translation_id: "t1")
    other = post("/other", tags: %w[X], categories: ["Insights", "Agentic-AI"], lang: "ko")
    assert_equal ["/other"], RelatedPosts.rank(ko, [ko, en, other])
  end

  def test_never_crosses_languages
    ko = post("/ko", tags: %w[X], categories: ["Insights", "Agentic-AI"], lang: "ko")
    en = post("/en", tags: %w[X], categories: ["Insights", "Agentic-AI"], lang: "en")
    assert_equal [], RelatedPosts.rank(ko, [en])
  end

  # The strictness rule: "both are Paper Reviews / Language-Models" describes 17
  # posts and is not a reason to read one after the other.
  def test_requires_a_shared_tag_and_never_relates_on_category_alone
    a = post("/a", tags: %w[Multi-Head-Latent-Attention], categories: ["Paper Reviews", "Language-Models"])
    same_cats_no_tags = post("/b", tags: %w[Something-Else], categories: ["Paper Reviews", "Language-Models"])
    assert_equal [], RelatedPosts.rank(a, [same_cats_no_tags])
  end

  def test_an_untagged_post_relates_to_nothing
    a = post("/a", categories: ["Insights", "Agentic-AI"])
    b = post("/b", categories: ["Insights", "Agentic-AI"])
    assert_equal [], RelatedPosts.rank(a, [b])
  end

  # Most paper posts declare no `lang`; the site default is Korean.
  def test_treats_a_missing_lang_as_korean
    bare = post("/bare", tags: %w[X], categories: ["Paper Reviews", "Language-Models"])
    explicit_ko = post("/ko", tags: %w[X], categories: ["Paper Reviews", "Language-Models"], lang: "ko")
    assert_equal ["/ko"], RelatedPosts.rank(bare, [explicit_ko])
  end

  def test_drops_candidates_with_no_overlap_at_all
    a = post("/a", tags: %w[X], categories: ["Insights", "Agentic-AI"])
    unrelated = post("/u", tags: %w[Y], categories: ["Paper Reviews", "Language-Models"])
    assert_equal [], RelatedPosts.rank(a, [unrelated])
  end

  def test_orders_by_score_then_recency
    a = post("/a", tags: %w[X Y], categories: ["Insights", "Agentic-AI"])
    strong = post("/strong", tags: %w[X Y], categories: ["Insights", "Agentic-AI"], date: "2020-01-01")
    weak_new = post("/weak-new", tags: %w[X], categories: ["Insights", "Agentic-AI"], date: "2026-06-01")
    weak_old = post("/weak-old", tags: %w[X], categories: ["Insights", "Agentic-AI"], date: "2024-06-01")
    assert_equal ["/strong", "/weak-new", "/weak-old"],
                 RelatedPosts.rank(a, [weak_old, weak_new, strong])
  end

  def test_honours_the_limit
    a = post("/a", tags: %w[X], categories: ["Insights", "Agentic-AI"])
    many = (1..10).map { |i| post("/p#{i}", tags: %w[X], categories: ["Insights", "Agentic-AI"]) }
    assert_equal 3, RelatedPosts.rank(a, many).length
    assert_equal 2, RelatedPosts.rank(a, many, limit: 2).length
  end

  def test_is_deterministic_for_equal_scores_and_dates
    a = post("/a", tags: %w[X], categories: ["Insights", "Agentic-AI"])
    b = post("/b", tags: %w[X], categories: ["Insights", "Agentic-AI"])
    c = post("/c", tags: %w[X], categories: ["Insights", "Agentic-AI"])
    assert_equal RelatedPosts.rank(a, [b, c]), RelatedPosts.rank(a, [c, b])
  end
end

class TestTheActualHarnessCluster < Minitest::Test
  # The real front matter of the five posts this feature was built for. The
  # Claude Code posts share no tags with the harness chronicle — they are related
  # only by category, which is deliberately not enough.
  def setup
    @evolution_ko = post("/evolution-ko", lang: "ko", translation_id: "evo", date: "2026-04-05",
                         tags: %w[Prompt-Engineering Context-Engineering Harness-Engineering
                                  Agentic-Patterns LLM-Architecture Vibe-Coding],
                         categories: ["Insights", "Agentic-AI"])
    @evolution_en = post("/evolution-en", lang: "en", translation_id: "evo", date: "2026-04-05",
                         tags: @evolution_ko[:tags], categories: ["Insights", "Agentic-AI"])
    @agentcore = post("/agentcore", date: "2026-04-12",
                      tags: %w[AgentCore AWS-Bedrock Harness-Engineering Agentic-Infrastructure
                               MCP Cedar-Policy Managed-RAG Agent-Registry],
                      categories: ["Insights", "Agentic-AI"])
    @claude_ko = post("/claude-ko", lang: "ko", translation_id: "cc", date: "2026-03-31",
                      tags: %w[Claude-Code Agentic-Architecture Context-Compaction
                               Multi-Agent-Orchestration Security-Architecture],
                      categories: ["Insights", "Agentic-AI"])
    @claude_en = post("/claude-en", lang: "en", translation_id: "cc", date: "2026-03-31",
                      tags: @claude_ko[:tags], categories: ["Insights", "Agentic-AI"])
    @all = [@evolution_ko, @evolution_en, @agentcore, @claude_ko, @claude_en]
  end

  # Harness-Engineering is the one tag they share, and it is the query cluster
  # this feature exists to consolidate.
  def test_agentcore_and_the_harness_chronicle_relate_to_each_other
    assert_equal ["/evolution-ko"], RelatedPosts.rank(@agentcore, @all)
    assert_equal ["/agentcore"], RelatedPosts.rank(@evolution_ko, @all)
  end

  # Both are Insights/Agentic-AI, but they share no topic tag. "Also about agents"
  # is not a reason to send a reader from one to the other.
  def test_the_claude_code_teardown_is_not_linked_on_category_alone
    assert_equal [], RelatedPosts.rank(@claude_ko, @all)
  end

  def test_an_english_post_only_sees_english_relatives
    # /claude-en shares no tag with /evolution-en, and /agentcore is Korean.
    assert_equal [], RelatedPosts.rank(@evolution_en, @all)
  end

  def test_no_post_is_offered_its_own_translation
    @all.each do |p|
      related = RelatedPosts.rank(p, @all)
      twin = @all.find { |o| o[:id] != p[:id] && o[:translation_id] == p[:translation_id] && p[:translation_id] }
      refute_includes related, twin[:id] if twin
    end
  end
end
