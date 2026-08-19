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

  # The invariant the weights exist to satisfy. Asserted on the constants
  # themselves, because the previous values (3/2/1) satisfied every behavioural
  # test while quietly letting categories outrank a whole extra shared tag.
  def test_category_bonuses_cannot_outweigh_one_shared_tag
    assert_operator RelatedPosts::SUBCATEGORY_WEIGHT + RelatedPosts::CATEGORY_WEIGHT,
                    :<, RelatedPosts::TAG_WEIGHT
  end

  # The ordering that the old weights got wrong.
  def test_two_shared_tags_beat_one_tag_plus_both_categories
    a = post("/a", tags: %w[X Y], categories: ["Insights", "Agentic-AI"])
    two_tags = post("/two", tags: %w[X Y], categories: ["Paper Reviews", "Other"], date: "2020-01-01")
    one_tag  = post("/one", tags: %w[X],   categories: ["Insights", "Agentic-AI"], date: "2026-01-01")
    assert_equal ["/two", "/one"], RelatedPosts.rank(a, [one_tag, two_tags])
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

class TestAdjacent < Minitest::Test
  def ordered(*posts)
    posts.sort_by { |p| [p[:date].to_time.to_i, p[:id]] }
  end

  def test_returns_the_neighbours_in_date_order
    old = post("/old", date: "2024-01-01")
    mid = post("/mid", date: "2025-01-01")
    new = post("/new", date: "2026-01-01")
    older, newer = RelatedPosts.adjacent(mid, ordered(old, mid, new))
    assert_equal "/old", older[:id]
    assert_equal "/new", newer[:id]
  end

  def test_nil_at_each_end_of_the_archive
    old = post("/old", date: "2024-01-01")
    new = post("/new", date: "2026-01-01")
    list = ordered(old, new)
    assert_nil RelatedPosts.adjacent(old, list)[0]
    assert_nil RelatedPosts.adjacent(new, list)[1]
  end

  # The live defect: all three translation pairs share a date exactly, so
  # Jekyll's page.previous pointed the newest Korean post at its own English
  # version.
  def test_skips_the_translation_twin_even_on_an_identical_date
    ko = post("/ko", lang: "ko", translation_id: "t1", date: "2026-07-27")
    en = post("/en", lang: "en", translation_id: "t1", date: "2026-07-27")
    before = post("/before", lang: "ko", date: "2026-04-12")
    older, newer = RelatedPosts.adjacent(ko, ordered(before, en, ko))
    assert_equal "/before", older[:id]
    assert_nil newer
  end

  # Skipping past an ineligible neighbour, rather than dropping the link: the
  # reader still gets a previous post, just not the untranslated one.
  def test_steps_over_an_other_language_neighbour
    target = post("/target", lang: "ko", date: "2026-03-01")
    english = post("/english", lang: "en", date: "2026-02-01")
    korean = post("/korean", lang: "ko", date: "2026-01-01")
    older, = RelatedPosts.adjacent(target, ordered(korean, english, target))
    assert_equal "/korean", older[:id]
  end

  def test_an_english_post_navigates_english_posts
    en_a = post("/en-a", lang: "en", date: "2026-01-01")
    ko = post("/ko", lang: "ko", date: "2026-02-01")
    en_b = post("/en-b", lang: "en", date: "2026-03-01")
    older, newer = RelatedPosts.adjacent(en_b, ordered(en_a, ko, en_b))
    assert_equal "/en-a", older[:id]
    assert_nil newer
  end

  # Most paper posts declare no lang; they must not be treated as a third
  # language and cut off from each other.
  def test_posts_without_an_explicit_lang_are_neighbours
    a = post("/a", date: "2024-01-01")
    b = post("/b", date: "2025-01-01", lang: "ko")
    older, = RelatedPosts.adjacent(b, ordered(a, b))
    assert_equal "/a", older[:id]
  end

  def test_returns_nothing_when_the_target_is_not_in_the_list
    assert_equal [nil, nil], RelatedPosts.adjacent(post("/x"), [post("/y")])
  end

  def test_a_lone_post_has_no_neighbours
    only = post("/only")
    assert_equal [nil, nil], RelatedPosts.adjacent(only, [only])
  end
end

class TestTheActualHarnessCluster < Minitest::Test
  # The real front matter of the five Insights/Agentic-AI posts.
  #
  # These five used to share nothing at all: the Claude Code teardown said
  # "Agentic-Architecture", the harness chronicle said "Agentic-Patterns", and
  # AgentCore said "Agentic-Infrastructure" — three names for one topic, so the
  # shared-tag requirement found no pair. The explicit "Agentic-AI" topic tag is
  # what connects them, and it is a tag, not a category, that does it.
  def setup
    @evolution_ko = post("/evolution-ko", lang: "ko", translation_id: "evo", date: "2026-04-05",
                         tags: %w[Prompt-Engineering Context-Engineering Harness-Engineering
                                  Agentic-Patterns LLM-Architecture Vibe-Coding Agentic-AI],
                         categories: ["Insights", "Agentic-AI"])
    @evolution_en = post("/evolution-en", lang: "en", translation_id: "evo", date: "2026-04-05",
                         tags: @evolution_ko[:tags], categories: ["Insights", "Agentic-AI"])
    @agentcore = post("/agentcore", date: "2026-04-12",
                      tags: %w[AgentCore AWS-Bedrock Harness-Engineering Agentic-Infrastructure
                               Model-Context-Protocol Cedar-Policy Managed-RAG Agent-Registry
                               Agentic-AI],
                      categories: ["Insights", "Agentic-AI"])
    @claude_ko = post("/claude-ko", lang: "ko", translation_id: "cc", date: "2026-03-31",
                      tags: %w[Claude-Code Agentic-Architecture Context-Compaction
                               Multi-Agent-Orchestration Security-Architecture Agentic-AI],
                      categories: ["Insights", "Agentic-AI"])
    @claude_en = post("/claude-en", lang: "en", translation_id: "cc", date: "2026-03-31",
                      tags: @claude_ko[:tags], categories: ["Insights", "Agentic-AI"])
    @all = [@evolution_ko, @evolution_en, @agentcore, @claude_ko, @claude_en]
  end

  # Two shared tags (Harness-Engineering and Agentic-AI) put these two ahead of
  # the teardown, which shares only the topic tag. Depth of overlap still orders
  # the list.
  def test_agentcore_and_the_harness_chronicle_rank_first_for_each_other
    assert_equal ["/evolution-ko", "/claude-ko"], RelatedPosts.rank(@agentcore, @all)
    assert_equal ["/agentcore", "/claude-ko"], RelatedPosts.rank(@evolution_ko, @all)
  end

  # The taxonomy fix, pinned: this post had no related block on the live site.
  def test_the_claude_code_teardown_now_connects_through_the_topic_tag
    assert_equal ["/agentcore", "/evolution-ko"], RelatedPosts.rank(@claude_ko, @all)
  end

  # Still true, and still the rule: strip the shared topic tag and shared
  # categories alone must not resurrect the link.
  def test_the_shared_category_alone_would_not_have_connected_them
    bare_claude = @claude_ko.merge(tags: @claude_ko[:tags] - ["Agentic-AI"])
    bare_all = @all.map { |p| p[:id] == bare_claude[:id] ? bare_claude : p }
    assert_equal [], RelatedPosts.rank(bare_claude, bare_all)
  end

  # Only three English posts exist and each is a translation, so an English
  # reader's candidate pool is tiny — the topic tag is what keeps it non-empty.
  def test_an_english_post_only_sees_english_relatives
    assert_equal ["/claude-en"], RelatedPosts.rank(@evolution_en, @all)
  end

  def test_no_post_is_offered_its_own_translation
    @all.each do |p|
      related = RelatedPosts.rank(p, @all)
      twin = @all.find { |o| o[:id] != p[:id] && o[:translation_id] == p[:translation_id] && p[:translation_id] }
      refute_includes related, twin[:id] if twin
    end
  end
end
