---
layout: post
title: "Claude Code Architecture Analysis"
date: 2026-03-31 12:00:01
categories: ["Insights", "Agentic-AI"]
tags: ["Claude-Code", "Agentic-Architecture", "Context-Compaction", "Multi-Agent-Orchestration", "Security-Architecture"]
cover: /assets/images/insights.png
use_math: false
---

# Claude Code Exposed: Anatomy of an Agentic AI Through an npm Source Map Leak

> Between what Anthropic calls "open source" and what is actually open, there was a gap of 4,600 files.

### TL;DR
- An npm source map mishap exposed Claude Code's proprietary core engine — all 4,600+ files of it
- The official "open source" release was just a plugin shell (279 files) — the core engine was commercially closed
- Inside lies a sophisticated production architecture: 8-layer security, 4-tier message compaction, cost-aware error recovery, and more
- Unreleased feature flags reveal the roadmap: voice mode, multi-agent Coordinator, proactive Kairos mode
- This is a rare opportunity to examine the real-world architecture of an agentic AI system

---

## 1. What Happened

On March 31, 2026, security researcher Chaofan Shou posted a few screenshots on X. The entire internal source code of Anthropic's Claude Code was sitting right there inside the npm package. The source map files — `.map` files — that should have been stripped from the production build were left intact.

[Reddit r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1s8ijfb/claude_code_source_code_has_been_leaked_via_a_map/) exploded, and the community began dissecting the entire codebase within hours.

This was not an intentional hack. Nor was it a sophisticated social engineering attack. Someone simply forgot to exclude `.map` files from the production package in the build pipeline. Internal prompt structures, agent orchestration logic, and tool invocation patterns were all reportedly included. It was a moment that simultaneously revealed how meticulously Anthropic had *designed* Claude Code — and how carelessly they had *shipped* it. An unintended gift to both competitors and security researchers alike.

One thing worth clarifying: Anthropic has marketed Claude Code as "open source." There is an [official GitHub repository](https://github.com/anthropics/claude-code) that anyone can browse. But what is actually there?

The plugin system. Hook examples. Configuration file templates. Eleven example plugins.

If this were a restaurant, it would be like publishing the menu and table settings and declaring "our kitchen is open." The actual recipes, cooking techniques, and secret sauces were all hidden behind closed doors. What the npm source map revealed was the entire kitchen — over 4,600 source files across 55+ directories, written in TypeScript/React. The license is not Apache 2.0 but Anthropic Commercial Terms of Service. A far cry from *true* open source.

This post sets aside the legal and ethical debates. Instead, it digs deep into the technical architecture revealed by the leaked code. Frankly, it is quite impressive — except for the part where they accidentally leaked it.

---

## 2. Overall Architecture: Much Bigger Than You Think

<a href="/assets/images/claude-code-analysis/architecture-overview.png" data-lightbox="claude-code" data-title="Overall System Architecture">
  <img src="/assets/images/claude-code-analysis/architecture-overview.png" alt="Overall System Architecture" />
</a>

When you first encounter Claude Code, it is easy to think: "Isn't this just an API wrapper?" You talk to Claude in the terminal, Claude fixes your code, done. Looks simple.

Open the leaked code and that notion shatters.

### Tech Stack

| Layer | Technology | Why This Choice |
|-------|-----------|-----------------|
| Runtime | **Bun** | Faster startup than Node.js with native TypeScript support. The `feature()` bundling API enables build-time dead code elimination |
| Language | **TypeScript** (strict mode) | Type safety is not optional in a 4,600+ file codebase |
| UI | **React 18 + [Ink](https://github.com/vadimdemedes/ink)** | Renders React components in the terminal. Complex UI elements like permission dialogs, progress bars, and multi-panel layouts justify this choice |
| API Client | `@anthropic-ai/sdk` | Official Anthropic SDK |
| MCP Client | `@modelcontextprotocol/sdk` | Standard protocol for external tool server integration |
| Feature Flags | **GrowthBook** | Server-side feature control and A/B testing |
| Bundler | **Bun bundler** | `feature()`-based dead code elimination separates internal and external builds |

Using React for a terminal app might seem like overkill. But when you look at the code — permission dialogs (`PermissionDialog.tsx`), Worker badges (`WorkerBadge`), multi-panel layouts, real-time streaming UI — these complex interactions make the choice understandable. It is a terminal, but far from simple text output. The UI is remarkably rich.

### Official Open Source vs. Leaked Code: The Gap in Numbers

| | Official Repo (`anthropics/claude-code`) | Leaked Code |
|---|---|---|
| File Count | ~279 (scripts/configs) | **4,600+** (full-stack engine) |
| Core Engine | **Not included** | Included |
| Tool Implementations | **Not included** | Fully included (Bash, Read, Write, Edit...) |
| Agentic Loop | **Not included** | Included (`query.ts`, 1,729 lines) |
| Permission System | **Not included** | Included (`permissions.ts`, 52K) |
| API Communication | **Not included** | Included (streaming, caching, fallback) |
| Bridge/Remote | **Not included** | Included (33+ files) |
| MCP Client | **Not included** | Included (`client.ts`, 119K) |
| License | Anthropic Commercial ToS | Proprietary commercial code |

279 vs 4,600. Time to rethink the definition of "open source."

---

## 3. The Agentic Loop: Opening the Heart

<a href="/assets/images/claude-code-analysis/agentic-loop.png" data-lightbox="claude-code" data-title="Agentic Loop State Machine">
  <img src="/assets/images/claude-code-analysis/agentic-loop.png" alt="Agentic Loop State Machine" />
</a>

The heart of Claude Code is `query.ts` — a 1,729-line `while(true)` loop. This "agentic loop" governs the entire cycle: receiving user input, executing tools, and feeding results back to Claude.

### 3.1 Async Generator: An Elegant Design Choice

The first thing that catches the eye is the function signature.

```typescript
export async function* query(
  params: QueryParams,
): AsyncGenerator<
  | StreamEvent
  | RequestStartEvent
  | Message
  | TombstoneMessage
  | ToolUseSummaryMessage,
  Terminal  // Return value: termination reason
>
```

`async function*` — an async generator. It streams events via `yield` while returning a `Terminal` type upon final termination via `return`. Here is why this is clever.

A conventional approach would have used an EventEmitter or callback-based implementation. But with an async generator, you can **handle both the event stream and termination semantics in a single function**. Consumers receive events via `for await...of`, and when the loop ends, they retrieve the termination reason (`Terminal`) from the `return` value. Error propagation is also natural — `throw` inside the generator propagates to the consumer's `try-catch`.

It is a remarkably clean pattern for expressing complex state machines, and it fits the agentic loop domain particularly well.

### 3.2 Immutable Parameters + Mutable State: The Continue Site Pattern

The loop separates two kinds of data.

**Immutable parameters** — things that do not change throughout the loop:
```typescript
type QueryParams = {
  messages: Message[]
  systemPrompt: SystemPrompt
  canUseTool: CanUseToolFn       // Permission check callback
  toolUseContext: ToolUseContext   // Tool execution context
  taskBudget?: { total: number }  // API task_budget (beta)
  maxTurns?: number               // Maximum turn limit
  fallbackModel?: string          // Fallback model
  querySource: QuerySource        // Query source (REPL, agent, etc.)
  // ...
}
```

**Mutable state** — things updated on every iteration:
```typescript
type State = {
  messages: Message[]
  toolUseContext: ToolUseContext
  autoCompactTracking: AutoCompactTrackingState | undefined
  maxOutputTokensRecoveryCount: number
  hasAttemptedReactiveCompact: boolean
  maxOutputTokensOverride: number | undefined
  pendingToolUseSummary: Promise<ToolUseSummaryMessage | null> | undefined
  stopHookActive: boolean | undefined
  turnCount: number
  transition: Continue | undefined  // Continuation reason from previous iteration
}
```

The pattern to note here is the **"Continue Site."** The source code comments explain it directly:

```typescript
// Continue sites write `state = { ... }` instead of 9 separate assignments.
```

Instead of modifying 9 individual fields one by one when changing state, the entire state object is reassigned.

```typescript
state = {
  ...state,
  messages: newMessages,
  turnCount: nextTurnCount,
  transition: { reason: 'next_turn' }
}
```

This pattern has two advantages. First, state transitions are **atomic** — there is no intermediate state where only 5 of 9 fields have been updated before an error occurs. Second, the `transition` field tracks **why** execution continued, enabling tests to assert that recovery paths worked correctly without inspecting message contents.

React's `setState` philosophy has permeated all the way into the backend loop — a glimpse into the Anthropic engineers' love for React.

This pattern matters because, in an agentic loop, state management bugs translate directly into user costs. Corrupted state leads to unnecessary API calls, and those calls cost tokens. Competitors like Cursor or Windsurf presumably implement similar agentic loops, but whether they achieve this level of state management rigor is unknown — their code is not public.

### 3.3 The 6-Stage Per-Turn Pipeline: A Detailed Walkthrough

Each turn goes through the following 6 stages. We will examine them alongside the actual line numbers in the source code.

#### Stage 1: Pre-Request Compaction (lines 365-548)

Conversation history is cleaned up *before* calling the API. Five compaction mechanisms are applied **in sequence**.

```typescript
// 1. Apply Tool Result Budget (lines 369-394)
messagesForQuery = await applyToolResultBudget(
  messagesForQuery,
  toolUseContext.contentReplacementState,
  // Only save replacement records for agent/REPL sources (for resume)
  persistReplacements ? records => void recordContentReplacement(...) : undefined,
)

// 2. Snip Compact — the cheapest option (lines 401-410)
if (feature('HISTORY_SNIP')) {
  const snipResult = snipModule!.snipCompactIfNeeded(messagesForQuery)
  messagesForQuery = snipResult.messages
  snipTokensFreed = snipResult.tokensFreed
}

// 3. Microcompact — cache-aware tool result clearing (lines 413-426)
const microcompactResult = await deps.microcompact(messagesForQuery, toolUseContext, querySource)
messagesForQuery = microcompactResult.messages

// 4. Context Collapse — staged reduction (lines 440-447)
if (feature('CONTEXT_COLLAPSE') && contextCollapse) {
  const collapseResult = await contextCollapse.applyCollapsesIfNeeded(...)
  messagesForQuery = collapseResult.messages
}

// 5. Auto-Compact — full summarization when threshold exceeded (lines 453-543)
const { compactionResult, consecutiveFailures } = await deps.autocompact(...)
```

The ordering matters. The reason Context Collapse comes **before** Auto-Compact is explained directly in the source code comments:

```typescript
// Runs BEFORE autocompact so that if collapse gets us under the
// autocompact threshold, autocompact is a no-op and we keep granular
// context instead of a single summary.
```

If Collapse reduces enough, Auto-Compact (an expensive API call) does not fire. The strategy is to preserve fine-grained context as much as possible while saving costs.

After compaction, the server can no longer see the full history, so the client must track the remaining task budget and report it to the server. Simple in theory, but synchronizing server-client state at compaction boundaries is a subtle problem commonly encountered in agentic systems.

#### Stage 2: API Call & Streaming (lines 659-863)

```typescript
for await (const message of deps.callModel(
  fullSystemPrompt,
  prependUserContext(messagesForQuery, userContext),
  toolUseContext,
  { taskBudget, taskBudgetRemaining, maxOutputTokensOverride, skipCacheWrite },
)) {
  // Process streaming events
}
```

The key here is the **StreamingToolExecutor**. Tools execute in parallel *while* Claude is generating its response. This means files are already being read while Claude is typing "Let me read that file." The reduction in perceived latency comes from tricks like this.

The implementation of this parallel execution is quite sophisticated:

```typescript
// StreamingToolExecutor.ts
private canExecuteTool(isConcurrencySafe: boolean): boolean {
  const executingTools = this.tools.filter(t => t.status === 'executing')
  return (
    executingTools.length === 0 ||
    (isConcurrencySafe && executingTools.every(t => t.isConcurrencySafe))
  )
}
```

Every tool has an `isConcurrencySafe` flag. Read-only tools like `FileReadTool`, `GlobTool`, and `GrepTool` are safe for parallel execution. State-mutating tools like `FileWriteTool` or `BashTool` must execute serially. And when one tool errors, sibling tools are cancelled with `sibling_error`.

```typescript
type AbortReason =
  | 'sibling_error'       // Sibling tool error → cancel me too
  | 'user_interrupted'     // User pressed Ctrl+C / ESC
  | 'streaming_fallback'   // Discarded due to model fallback
```

Model fallback is also handled here. When the primary model fails:

```typescript
if (innerError instanceof FallbackTriggeredError && fallbackModel) {
  currentModel = fallbackModel
  attemptWithFallback = true
  // Create tombstones for orphaned messages
  yield* yieldMissingToolResultBlocks(assistantMessages, 'Model fallback triggered')
  // Reset StreamingToolExecutor and retry
}
```

Tombstone messages leave a record saying "this tool call was discarded due to model fallback." It is a mechanism for maintaining conversation history consistency.

#### Stage 3: Error Recovery Cascade (lines 1062-1256)

This is the most architecturally impressive section. When errors occur, the system does not give up immediately. It attempts recovery **in order from lowest to highest cost**.

**Prompt-too-long (413 error) recovery — 3-stage cascade:**

```
Stage 1: Context Collapse drain (cost: 0)
  └ Flushes already-prepared reductions. Applied instantly with no additional API calls.

Stage 2: Reactive Compact (cost: 1 API call)
  └ Summarizes the entire conversation. Strips images, summarizes, then retries.
  └ "strip retry" — if the summary itself is too large, removes media and tries once more.

Stage 3: Surface the error
  └ If all attempts fail, shows the error to the user.
```

**Max-output-tokens recovery — also 3 stages:**

```
Stage 1: Token cap escalation (cost: 0)
  └ Transparently increases from 8K → 64K (ESCALATED_MAX_TOKENS).
  └ Handled automatically with no meta-message. The user does not notice.

Stage 2: Resume message injection (cost: API re-call, up to 3 times)
  └ Injects a message: "Your previous response was truncated. Please continue from where you left off."
  └ Then calls the model again.
  └ Tracks maxOutputTokensRecoveryCount, attempting up to 3 times.

Stage 3: Recovery exhaustion
  └ If still unresolved after 3 attempts, completes with whatever results are available.
```

The design principle behind this cascade is clear: **the first attempt is always free.** The expensive operation of summarizing the entire conversation is a last resort. This is not simple "retry" logic — it is a recovery strategy with precise cost-effectiveness considerations.

#### Stage 4: Stop Hooks & Token Budget (lines 1267-1355)

Stop hooks execute user-defined validation logic. If you set up a hook saying "don't stop unless tests pass," the hook runs when Claude tries to finish. If it fails, the hook's error message is injected into the conversation and Claude tries again.

The token budget is even more interesting:

```typescript
function checkTokenBudget(tracker, budget, globalTurnTokens) {
  const pct = (turnTokens / budget) * 100
  const isDiminishing = (
    continuationCount >= 3 &&
    deltaSinceLastCheck < 500 &&  // DIMINISHING_THRESHOLD
    lastDeltaTokens < 500
  )

  if (!isDiminishing && turnTokens < budget * 0.9) {
    return { action: 'continue', nudgeMessage: ... }
  }
  return { action: 'stop', completionEvent: { diminishingReturns, ... } }
}
```

**Diminishing returns detection** is built in. If Claude continues 3 consecutive times but produces fewer than 500 tokens each time, the system decides "continuing further is pointless" and stops. It keeps going until 90% of the budget is consumed, but it will not spin its wheels.

Why this logic matters: the most dangerous thing in an agentic loop is an **infinite loop**. Claude repeatedly saying "let me try one more fix" while burning tokens with no progress. This diminishing returns detection automatically blocks those situations.

#### Stage 5: Tool Execution (lines 1363-1520)

Streaming mode (already started in Stage 2) and batch mode coexist.

```typescript
// Streaming: concurrency-safe tools already executing during API response generation
if (streamingToolExecutor) {
  toolResults = await streamingToolExecutor.getRemainingResults()
}

// Batch: remaining tools execute sequentially here
toolResults = await runTools(toolUseBlocks, toolUseContext, canUseTool)
```

Progress signaling is also present. A Promise resolver called `progressAvailableResolve` notifies consumers that "new progress is available." This drives the real-time spinner updates in the terminal UI.

#### Stage 6: Post-Tool & Next Turn Transition (lines 1547-1727)

After tool execution completes, several "side tasks" are processed:

```typescript
// 1. Consume skill discovery — harvest what was prefetched in Stage 1
if (pendingSkillPrefetch?.settledAt !== null) {
  const skillAttachments = await pendingSkillPrefetch.promise
}

// 2. Consume memory attachments — also harvest prefetched results
if (pendingMemoryPrefetch?.settledAt !== null) {
  const memoryAttachments = await pendingMemoryPrefetch.promise
}

// 3. Drain queued commands — slash commands, task notifications, etc.
const queuedCommands = getCommandsByMaxPriority(...)

// 4. Refresh MCP server tools
if (toolUseContext.options.refreshTools) {
  toolUseContext.options.tools = toolUseContext.options.refreshTools()
}
```

Then the state transitions:

```typescript
// Continue Site: proceed to next turn
state = {
  messages: [...messagesForQuery, ...assistantMessages, ...toolResults],
  toolUseContext: toolUseContextWithQueryTracking,
  autoCompactTracking: tracking,
  turnCount: nextTurnCount,
  transition: { reason: 'next_turn' },
}
// Returns to top of while(true) loop
```

### 3.4 Termination Reasons: 9 Ways the Loop Ends

| Exit Reason | Meaning | Trigger Point |
|-------------|---------|---------------|
| `completed` | Normal completion (response ended without tool calls) | line 1264, 1357 |
| `blocking_limit` | Hard token limit reached | line 646 |
| `aborted_streaming` | User interrupted during streaming (Ctrl+C) | line 1051 |
| `aborted_tools` | User interrupted during tool execution | line 1515 |
| `prompt_too_long` | Prompt exceeded limit even after recovery attempts | line 1175, 1182 |
| `image_error` | Image validation failure (size exceeded, etc.) | line 977, 1175 |
| `model_error` | Unexpected model error | line 996 |
| `hook_stopped` | Stop hook blocked continuation | line 1520 |
| `max_turns` | Maximum turn count exceeded (`maxTurns` parameter) | line 1711 |

### 3.5 QueryEngine.ts: The Session-Level Supervisor

If `query.ts` is the loop for a single turn, `QueryEngine.ts` (1,295 lines) manages the **entire session**.

```typescript
class QueryEngine {
  mutableMessages: Message[]       // Full conversation history
  permissionDenials: PermissionDenial[]  // Tool permission denial records
  totalUsage: Usage                // Cumulative token usage
  readFileState: FileStateCache    // File state cache (prevents duplicate reads)
  discoveredSkillNames: Set<string> // Discovered skills (reset per turn)
  loadedNestedMemoryPaths: Set<string> // Loaded memory paths (prevents duplicates)
}
```

The `submitMessage()` method receives user input, calls `query()`, and accumulates results into the session.

```typescript
// Accumulate usage
this.totalUsage = accumulateUsage(this.totalUsage, currentMessageUsage)
```

An asymmetric strategy is applied to transcript recording:

```typescript
// User messages: blocking save (essential for --resume restoration)
await recordTranscript(userMessage)

// Assistant messages: fire-and-forget (async, non-blocking)
recordTranscript(assistantMessage)  // No await
```

Not all messages are saved with equal priority. User messages are saved with blocking I/O because they are essential for session restoration. Assistant responses follow a "save it but don't wait" strategy. A design that balances performance and reliability.

---

## 4. Message Compaction: A Sophisticated War Against the Context Window

<a href="/assets/images/claude-code-analysis/compaction-layers.png" data-lightbox="claude-code" data-title="Message Compaction Layers">
  <img src="/assets/images/claude-code-analysis/compaction-layers.png" alt="Message Compaction Layers" />
</a>

The greatest enemy of an agentic AI tool is the context window limit. In long conversations, earlier content gets truncated or the API returns a 413 error. Claude Code's compaction system addresses this with 4 layers, each with different costs and levels of information loss. The core principle: **"always start with the cheapest option."**

### 4.1 Snip Compact — Cheapest and Most Aggressive

Cost: **Free** (no API calls)
Information loss: **High**

Removes entire blocks of older internal messages and keeps only recent context. Gated behind the `HISTORY_SNIP` feature flag, primarily used in headless sessions.

```typescript
const snipResult = snipModule!.snipCompactIfNeeded(messagesForQuery)
messagesForQuery = snipResult.messages
snipTokensFreed = snipResult.tokensFreed
```

Importantly, `snipTokensFreed` is passed to the auto-compact stage. As the source comments explain:

```typescript
// snipTokensFreed is plumbed to autocompact so its threshold check reflects
// what snip removed; tokenCountWithEstimation alone can't see it
```

If Snip has already freed a significant amount, this prevents Auto-Compact from firing unnecessarily.

### 4.2 Microcompact (530 lines) — Selective Clearing That Respects the Cache

Cost: **Free** (no API calls)
Information loss: **Medium**

Selectively clears tool results — not all tool results, only specific ones.

```typescript
// Clearing targets: file_read, shell, grep, glob, web_search, web_fetch, file_edit, file_write
// Replaces results with "[Old tool result content cleared]"
// Images/documents: estimated at 2,000 tokens
// Text: rough token count estimation
```

The real core of this layer is **cache edit block pinning**. Behind the `CACHED_MICROCOMPACT` feature flag, this feature is designed to work in harmony with the API's prompt caching.

```typescript
const pendingCacheEdits = feature('CACHED_MICROCOMPACT')
  ? microcompactResult.compactionInfo?.pendingCacheEdits
  : undefined
```

It tracks the IDs of already-cached tool results and manages them to maintain cache hits. In prompt caching, a single cache miss means tens of thousands of tokens must be recomputed. Reducing context while maintaining cache hit rates are conflicting goals, and this design attempts to balance them.

### 4.3 Context Collapse — Staged Reduction

Cost: **Low**
Information loss: **Medium**

This is a unique concept called "staged collapse." Instead of compressing the entire conversation at once, the system decides which message blocks to collapse during a preview stage, then actually collapses them during a commit stage.

The source comments explain the core of this design:

```typescript
// Nothing is yielded — the collapsed view is a read-time projection
// over the REPL's full history. Summary messages live in the collapse
// store, not the REPL array. This is what makes collapses persist
// across turns: projectView() replays the commit log on every entry.
```

The collapsed view is a **read-time projection** over the REPL's full history. Original messages remain intact; only the collapse results are stored separately. Conceptually similar to Git's snapshot storage approach — providing a different "view" without touching the originals.

The advantage of this approach is maximizing prompt caching cache hits. Since original messages do not change, cached prefixes are not invalidated.

### 4.4 Auto-Compact (351 lines) — The Last Resort

Cost: **High** (additional Claude API call)
Information loss: **Low** (AI summarizes, so key information is preserved)

Sends the entire conversation history to Claude and requests a summary. The threshold logic is clear:

```typescript
function getAutoCompactThreshold(model: string): number {
  const effectiveContextWindow = getEffectiveContextWindowSize(model)
  return effectiveContextWindow - 13_000  // 13K token buffer
}
```

Reserves 13K tokens from the context window, and triggers auto-compaction when the rest fills up.

There is a **circuit breaker**:

```typescript
MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3
```

Auto-compaction itself can fail — when the conversation is so long that even the summarization request exceeds the context limit. This can lead to an infinite loop of "compact fails → retry → fails again." After 3 consecutive failures, it gives up cleanly. A textbook example of defensive programming.

### 4.5 Full Compaction (1,705 lines) — The Final Card

This is the full compaction logic called internally when Auto-Compact executes.

1. **Image stripping**: Replaces images and documents with `[image]` / `[document]` placeholders (major token savings)
2. **API round grouping**: Groups `tool_use` → `tool_result` pairs together (preserves semantic units)
3. **Thinking block removal**: (Anthropic internal builds only) Removes reasoning process blocks before compaction
4. **PTL (Prompt-Too-Long) retry**: If the compaction request itself exceeds the limit, drops the oldest API round groups at 20% increments

```typescript
truncateHeadForPTLRetry(messages, ptlResponse) {
  // Drop oldest API-round groups until prompt-too-long gap is covered
  // Falls back to dropping 20% of groups
}
```

### 4.6 Token Warning State Machine

A 4-stage warning system provides visual feedback to the user:

```
[Normal]
  ↓ Context window - 20K tokens
[Warning] ← Yellow warning
  ↓ Context window - 20K tokens
[Error] ← Orange warning
  ↓ Context window - 13K tokens
[AutoCompact] ← Auto-compaction triggers
  ↓ Context window - 3K tokens
[BlockingLimit] ← Red, manual compaction only
```

### Insight: Why 4 Tiers?

The 4-tier compaction is not simply "compress progressively harder." Each tier has **different trade-offs**:

- **Snip**: High information loss but no impact on cache hits
- **Microcompact**: Selective loss that respects the cache
- **Context Collapse**: Reduces the view while preserving originals
- **Auto-Compact**: Minimal information loss but incurs API costs

This is a problem of finding the Pareto optimum along two axes: "minimize cost" and "maximize information preservation." The 4 layers represent different points on that Pareto frontier.

---

## 5. Tool System: A Systematically Extensible Swiss Army Knife

What Claude Code can actually *do* is determined by the tool system. Looking at `tools.ts` reveals the full picture of how tools are registered.

### 5.1 Tool Interface

Every tool follows the same interface (from `Tool.ts`):

```typescript
// ToolUseContext — everything needed for tool execution is bundled here
type ToolUseContext = {
  options: { tools: Tools; mainLoopModel: string; mcpClients: MCPServerConnection[]; maxBudgetUsd?: number; ... }
  abortController: AbortController
  readFileState: FileStateCache    // Prevents duplicate file reads
  getAppState(): AppState          // Access to application state
  setAppState(f: (prev: AppState) => AppState): void
  // ... 40+ fields (agent ID, permission tracking, content replacement state, etc.)
}
```

Worth noting is the `ToolPermissionContext`:

```typescript
type ToolPermissionContext = DeepImmutable<{
  mode: PermissionMode
  additionalWorkingDirectories: Map<string, AdditionalWorkingDirectory>
  alwaysAllowRules: ToolPermissionRulesBySource
  alwaysDenyRules: ToolPermissionRulesBySource
  alwaysAskRules: ToolPermissionRulesBySource
  isBypassPermissionsModeAvailable: boolean
  isAutoModeAvailable?: boolean
  strippedDangerousRules?: ToolPermissionRulesBySource
  shouldAvoidPermissionPrompts?: boolean  // For background agents
  awaitAutomatedChecksBeforeDialog?: boolean  // For coordinator workers
  prePlanMode?: PermissionMode  // Saves mode before entering plan mode
}>
```

Wrapped in `DeepImmutable`. The permission context is read-only and cannot be accidentally modified.

### 5.2 Three-Tier Tool Registration Structure

Looking at the `getAllBaseTools()` function, Claude Code's tools fall into three tiers.

**Always active**: Around 20 base tools including `BashTool`, `FileReadTool`, `FileEditTool`, `WebSearchTool`, and `AgentTool`. These constitute Claude Code's core capabilities.

**Conditionally active**: Tools activated based on environment or configuration. For example, `GlobTool` and `GrepTool` are disabled in Anthropic's internal build because `bfs`/`ugrep` are embedded in the Bun binary, making separate tools unnecessary. `PowerShellTool` only activates on Windows, and `LSPTool` requires explicit activation via environment variable.

**Feature flag-gated (unreleased)**: Tools gated by the `feature()` function — `WebBrowserTool`, `WorkflowTool`, `SleepTool`, `PushNotificationTool`, and 15+ others fall into this category. These are covered in detail in Section 7.

Interesting are the Anthropic-internal-only tools, gated by `USER_TYPE === 'ant'`:

```typescript
const REPLTool = process.env.USER_TYPE === 'ant'
  ? require('./tools/REPLTool/REPLTool.js').REPLTool
  : null
```

`REPLTool`, `ConfigTool`, `TungstenTool`, `SuggestBackgroundPRTool` — these tools do not exist for external users. This means Anthropic engineers have a separate set of tools for internal use. `TungstenTool` in particular has a mysterious purpose based on name alone — tungsten (a high-density metal) suggests a tool for handling "heavy" operations.

Why does this 3-tier structure matter? In agentic AI tools, "which capabilities to grant" is a core design decision. Too many tools and the system prompt bloats (increasing token costs); too few and the agent's capabilities are limited. Claude Code solves this with a 3-tier approach of build-time elimination + conditional activation + feature flags. Cursor and Devin likely face similar tool scaling challenges, and this layered approach is worth studying.

### 5.3 Tool Pool Assembly: Designed with Cache Stability in Mind

The `assembleToolPool()` function merges built-in and MCP tools, and here we see interesting processing for prompt cache stability:

```typescript
export function assembleToolPool(permissionContext, mcpTools): Tools {
  const builtInTools = getTools(permissionContext)
  const allowedMcpTools = filterToolsByDenyRules(mcpTools, permissionContext)

  // Sort each partition for prompt-cache stability, keeping built-ins as a
  // contiguous prefix. The server's claude_code_system_cache_policy places a
  // global cache breakpoint after the last prefix-matched built-in tool; a flat
  // sort would interleave MCP tools into built-ins and invalidate all downstream
  // cache keys whenever an MCP tool sorts between existing built-ins.
  const byName = (a: Tool, b: Tool) => a.name.localeCompare(b.name)
  return uniqBy(
    [...builtInTools].sort(byName).concat(allowedMcpTools.sort(byName)),
    'name',
  )
}
```

Built-in tools and MCP tools are **sorted separately then concatenated.** Why not sort everything at once? Because the server's `claude_code_system_cache_policy` places a cache breakpoint after the last built-in tool. A flat sort would interleave MCP tools between built-in tools, invalidating all downstream cache keys whenever an MCP tool is added or removed.

This level of cache optimization clearly comes from real-world production cost experience.

### 5.4 Dynamic Tool Search

This is the `tool-search-2025-10-16` beta feature. When tools number in the dozens, the system prompt alone consumes significant tokens. Instead of showing Claude all tools upfront, this feature **searches and loads them on demand**.

```typescript
// tools.ts:247-249
// Include ToolSearchTool when tool search might be enabled (optimistic check)
// The actual decision to defer tools happens at request time in claude.ts
...(isToolSearchEnabledOptimistic() ? [ToolSearchTool] : []),
```

The phrase "optimistic check" stands out. At registration time, it is optimistically included, while the actual defer decision happens at API request time. This is the LLM version of lazy loading.

---

## 6. Multi-Layer Security: Peeling the Onion

<a href="/assets/images/claude-code-analysis/security-layers.png" data-lightbox="claude-code" data-title="Security Layers">
  <img src="/assets/images/claude-code-analysis/security-layers.png" alt="Security Layers" />
</a>

"An AI agent accessing my filesystem" is a sentence sufficient to make any security researcher shudder. Looking at how Claude Code handles this problem reveals an onion-like multi-layered defense. Eight layers deep.

### 6.1 Layer 1: Build-Time Gate — Security Through Nonexistence

```typescript
// tools.ts:117-119
const WebBrowserTool = feature('WEB_BROWSER_TOOL')
  ? require('./tools/WebBrowserTool/WebBrowserTool.js').WebBrowserTool
  : null
```

The `feature()` function is **evaluated at build time**. The Bun bundler completely eliminates `require()` calls in `false` branches. Anthropic's internal-only tool code *physically does not exist* in external builds. Activating them at runtime by changing environment variables is impossible — the code simply is not in the binary.

A clever approach to splitting "internal" and "external" builds from the same codebase while avoiding the security risks of runtime branching. Of course, the irony is that what leaked this time was precisely the code that should have been eliminated at build time, left behind in source maps.

The `USER_TYPE` environment variable follows the same pattern:

```typescript
// tools.ts:16-19
const REPLTool =
  process.env.USER_TYPE === 'ant'
    ? require('./tools/REPLTool/REPLTool.js').REPLTool
    : null
```

### 6.2 Layer 2: Feature Flags — Server-Side Kill Switches

Even after the build, features can be controlled server-side. GrowthBook-based flags with the `tengu_` prefix serve this role.

| Flag | Role |
|------|------|
| `tengu_amber_quartz_disabled` | Voice mode kill switch |
| `tengu_bypass_permissions_disabled` | Bypass permissions mode kill switch |
| `tengu_auto_mode_config.enabled` | Auto mode circuit breaker |
| `tengu_ccr_bridge` | Remote control eligibility check |
| `tengu_sessions_elevated_auth_enforcement` | Requires trusted device token |

If a security incident occurs, the kill switch can be pulled from the server instantly. Being able to disable features without a client update is critically important from an incident response perspective.

Internally at Anthropic, these can be overridden via environment variable:

```typescript
// CLAUDE_INTERNAL_FC_OVERRIDES (Ant-only)
// '{"my_feature": true, "my_config": {"key": "val"}}'
```

### 6.3 Layer 3: Configuration-Based Rules — Priority Across 8 Sources

```typescript
type PermissionRule = {
  source: PermissionRuleSource  // One of 8 sources
  ruleBehavior: 'allow' | 'deny' | 'ask'
  ruleValue: {
    toolName: string     // "Bash", "FileEdit", etc.
    ruleContent?: string // Pattern, e.g. "python:*"
  }
}
```

Rule source priority:
1. `userSettings` — `~/.claude/settings.json`
2. `projectSettings` — `.claude/settings.json`
3. `localSettings` — `.claude/local.json`
4. `flagSettings` — GrowthBook feature flags
5. `policySettings` — Organization-level policies (enterprise)
6. `cliArg` — Command-line arguments
7. `command` — Runtime rule updates
8. `session` — In-memory session-only rules

### 6.4 Layer 4: Transcript Classifier — AI Watching AI

`yoloClassifier.ts` — a 52K file. (The file is named "YOLO" yet weighs in at 52K — Anthropic engineers clearly have a sense of humor.)

In auto mode (`auto`), instead of asking the user "is this okay?" every time, **an additional Claude API call** judges the safety of tool usage.

```
1. Tool usage request occurs
2. Whitelist check
   └ FileRead, Grep, Glob, Tasks, etc. (read-only) → skip classifier, auto-allow
3. Classifier API call → "allow" or "deny"
4. Denial tracking:
   └ 3 consecutive denials → fall back to prompting mode
   └ 20 total denials → fall back to prompting mode
```

The fallback mechanism in denial tracking is important. If the classifier acts too conservatively, user experience degrades, so after a certain number of denials it switches to "just ask the user."

For debugging, setting the `CLAUDE_CODE_DUMP_AUTO_MODE=1` environment variable dumps the classifier's requests/responses as JSON to `/tmp/claude-code/auto-mode/`. A useful tip when you want to understand why Claude Code's auto mode is rejecting certain operations.

What if the classifier returns an API error? **It falls back to prompting** — not automatic denial. A fail-open strategy of "when in doubt, ask the user." Fail-open is typically dangerous in security, but here the fallback is "let the user decide," which is reasonable.

### 6.5 Layer 5: Dangerous Pattern Detection

```typescript
DANGEROUS_BASH_PATTERNS = [
  'python', 'node', 'ruby', 'perl', 'bash', 'sh', 'zsh', 'ksh',
  'exec', 'eval', 'source', 'curl', 'wget', 'nc', 'ncat',
  'socat', 'dd', 'xxd', 'openssl', 'ssh', 'scp', 'sftp',
  'sudo', 'su', 'chroot', 'unshare', 'docker', 'podman',
  'chmod', 'chown', 'chgrp', 'umask', 'mount', 'umount'
]
```

In auto mode, attempts to set `python:*` or interpreter wildcards as allow rules are blocked.

```typescript
isDangerousBashPermission(rule) {
  // Tool-level allow without content (allow all bash) → dangerous
  // "python:*", "python*", "python -*" → dangerous
  // Interpreter prefix + wildcard → dangerous
}
```

This is because arbitrary code execution through interpreters can bypass Bash sandboxing. It blocks attacks like `python -c "import os; os.system('rm -rf /')"` at the source.

### 6.6 Layer 6: Filesystem Permission Validation (62K)

The largest permission file (62K) is dedicated solely to file path validation.

- **Absolute path normalization**: Handling `.`, `..` in relative paths
- **Symlink escape prevention**: Blocking attacks where symlinks inside allowed directories point outside
- **Safe glob pattern expansion**: Ensuring patterns like `/**/*` do not include unexpected paths
- **CWD-only mode vs. full access mode**: In `acceptEdits` mode, only the current directory
- **Scratchpad support**: Allowing Coordinator worker access to shared directories (`tengu_scratch`)
- **Windows/POSIX path handling**: Cross-platform support

### 6.7 Layer 7: Trust Dialog

The security dialog that appears on first run. It reviews and obtains user consent for:

- Project-scoped MCP server configurations
- Custom Hook configurations
- Bash permission settings
- API key helpers
- AWS/GCP command access
- OTEL headers

File/filesystem operations are blocked until the Trust Dialog is passed.

### 6.8 Layer 8: Bypass Permissions Kill Switch

The measure of last resort. When `tengu_bypass_permissions_disabled` is activated on the GrowthBook server:

```typescript
// bypassPermissionsKillswitch.ts
// Blocks users from entering bypass mode entirely
// Forces revert to previous mode
// Displays diagnostic message
```

### Insight: Security Model Design Philosophy

A consistent set of principles runs through all 8 layers:

1. **Deny by default** — Explicit allow rules are required
2. **Fail to prompting, not to deny** — When judgment is uncertain, ask the user (not deny)
3. **Defense in depth** — If one layer is breached, the next catches it
4. **Server-side kill switch** — Instant deactivation without client updates
5. **Build-time elimination** — If the code does not exist, neither does the vulnerability

---

## 7. Unreleased Features: The Roadmap Got Leaked

Unreleased features hiding behind feature flags were exposed. This is the area that became the most "unintended gift" to competitors.

### 7.1 Voice Mode — Coding by Voice

```typescript
// voice/voiceModeEnabled.ts
export function isVoiceModeEnabled(): boolean {
  return hasVoiceAuth() && isVoiceGrowthBookEnabled()
}

export function hasVoiceAuth(): boolean {
  // OAuth only (requires Claude.ai account)
  // Not available via API key/Bedrock/Vertex
  // Uses voice_stream endpoint
}

export function isVoiceGrowthBookEnabled(): boolean {
  return feature('VOICE_MODE')
    ? !getFeatureValue_CACHED_MAY_BE_STALE('tengu_amber_quartz_disabled', false)
    : false
}
```

Several things can be gleaned:

- A **dedicated `voice_stream` API endpoint** exists, separate from the standard Messages API.
- It is **OAuth-only**. Users on API keys or third-party clouds (Bedrock/Vertex) cannot access it.
- A **GrowthBook kill switch** (`tengu_amber_quartz_disabled`) allows instant server-side deactivation.
- The caching strategy uses `_CACHED_MAY_BE_STALE`, meaning it uses a cached flag value rather than real-time checks. Voice mode activation status is not queried from the server on every call.

### 7.2 Web Browser Tool — Real Browser Automation

```typescript
const WebBrowserTool = feature('WEB_BROWSER_TOOL')
  ? require('./tools/WebBrowserTool/WebBrowserTool.js').WebBrowserTool
  : null
```

The current `WebFetchTool` only fetches static HTML. `WebBrowserTool` is presumably real browser automation leveraging Bun's `WebView` API. This would mean the ability to interact with SPA pages that require JavaScript rendering. Think of it as the CLI version of Computer Use.

### 7.3 Coordinator Mode — Multi-Agent Orchestration (19K)

The most ambitious unreleased feature.

```typescript
// coordinator/coordinatorMode.ts
export function isCoordinatorMode(): boolean {
  if (feature('COORDINATOR_MODE')) {
    return isEnvTruthy(process.env.CLAUDE_CODE_COORDINATOR_MODE)
  }
  return false
}
```

The **Coordinator** does not write code directly. It is a meta-orchestrator that spawns multiple worker agents and distributes tasks.

- Workers are spawned via `AgentTool`
- Results arrive as `<task-notification>` XML blocks
- `SendMessage` sends follow-up instructions to workers
- `TaskStop` terminates workers

The allowed tools are strictly limited:

```typescript
COORDINATOR_MODE_ALLOWED_TOOLS = new Set([
  'AgentTool',           // Spawn worker
  'TaskStop',            // Terminate worker
  'SendMessage',         // Communicate with worker
  'SyntheticOutput',     // Generate output
])
```

The Coordinator **cannot execute Bash**. It cannot read files. It can only manage workers. This is an extreme application of the principle of least privilege — if the orchestrator could execute tools directly, it would become a single point of failure.

**Shared scratchpad** (`tengu_scratch` GrowthBook gate):

A directory for information sharing between workers. It bypasses normal filesystem permission checks — workers need to access the scratchpad outside their CWD.

This is effectively a **microservices architecture for AI agents**. One Claude orchestrating multiple Claudes, with each worker operating independently in isolated contexts.

### 7.4 Kairos — The Proactive Assistant

Kairos, Greek for "the opportune moment," is a mode where **Claude acts first**.

| Feature Flag | Tool | Capability |
|-------------|------|------------|
| `KAIROS` | `SendUserFileTool` | Proactively sends files to the user |
| `KAIROS` \|\| `KAIROS_PUSH_NOTIFICATION` | `PushNotificationTool` | Mobile/desktop push notifications |
| `KAIROS_GITHUB_WEBHOOKS` | `SubscribePRTool` | GitHub PR webhook subscriptions |
| `PROACTIVE` \|\| `KAIROS` | `SleepTool` | Background waiting (timer) |
| `KAIROS_CHANNELS` | (unknown) | Multi-channel integration |
| `KAIROS_BRIEF` | (unknown) | Checkpoints/status updates |

Picture this scenario: A new review comment is posted on a GitHub PR. `SubscribePRTool` detects it, checks the CI results, and sends a notification via `PushNotificationTool`: "PR #123 got a review, and CI failed. Here's what you might want to fix." `SleepTool` periodically wakes to check status.

Claude works even when the user has not opened a session. This represents a transition beyond coding assistant to **autonomous software engineering agent**.

### 7.5 Agent Triggers & Monitoring

```typescript
const cronTools = feature('AGENT_TRIGGERS')
  ? [CronCreateTool, CronDeleteTool, CronListTool]
  : []

const RemoteTriggerTool = feature('AGENT_TRIGGERS_REMOTE')
  ? require('./tools/RemoteTriggerTool/RemoteTriggerTool.js').RemoteTriggerTool
  : null

const MonitorTool = feature('MONITOR_TOOL')
  ? require('./tools/MonitorTool/MonitorTool.js').MonitorTool
  : null
```

Crons created by a teammate are tagged with that agent's `agentId` and routed to that agent's message queue. This means long-running agents can **manage their own schedules**.

### 7.6 UDS Inbox — Multi-Device Messaging

```typescript
const ListPeersTool = feature('UDS_INBOX')
  ? require('./tools/ListPeersTool/ListPeersTool.js').ListPeersTool
  : null
```

"UDS" presumably stands for Unified Device Stack. Agents can query "peers" (other connected devices/instances) and route messages via `bridge://` or `other://` schemas.

A world where the Claude on your laptop and the Claude on your desktop communicate with each other. It may still be distant, but the pipes are already laid.

### 7.7 Workflow Scripts

```typescript
const WorkflowTool = feature('WORKFLOW_SCRIPTS')
  ? (() => {
      require('./tools/WorkflowTool/bundled/index.js').initBundledWorkflows()
      return require('./tools/WorkflowTool/WorkflowTool.js').WorkflowTool
    })()
  : null
```

There are bundled workflows (pre-built automation scripts) with an initialization system. Recursive execution is blocked inside sub-agents (included in `ALL_AGENT_DISALLOWED_TOOLS`).

### Insight: What Unreleased Features Reveal About Direction

Synthesizing these features reveals Anthropic's strategy:

1. **CLI → Platform**: From a single terminal tool to a multi-device, multi-agent platform
2. **Reactive → Proactive**: From awaiting user commands to autonomous monitoring and notifications
3. **Text → Multimodal**: From typing to voice and browser automation
4. **Single → Orchestration**: From one agent to an agent swarm managed by a Coordinator

### 7.8 Bridge — Remote Control System (33+ Files)

The Bridge system is a large subsystem comprising 33+ files. It is the backend for the "Remote Control" feature that controls the local machine's Claude Code from the Claude.ai web interface.

#### Connection Flow

```
1. User → OAuth login to claude.ai
2. CCR (Claude Cloud Runtime) API → Subscription + GrowthBook gate check (tengu_ccr_bridge)
3. Local Claude Code → Obtains environment_id + environment_secret
4. Authenticated tunnel setup (WebSocket)
5. Browser → Bridge API → Local tool execution → Result return
```

#### Security Tiers

| Tier | Authentication Requirements | Use Case |
|------|-----------------------------|----------|
| Standard | OAuth token | Regular remote sessions |
| Elevated | OAuth + **Trusted Device Token** (JWT) | Sessions involving sensitive operations |

In the Elevated tier, device trustworthiness is additionally verified via the `X-Trusted-Device-Token` header.

```typescript
// bridge/trustedDevice.ts
// JWT bound to a physical device
// Guarantees "this request actually came from a registered device"
```

#### Session Isolation

Each remote session is isolated via **Git worktree**. Even if you connect to the same project from two browser tabs, each runs in an independent working directory. An approach that eliminates merge conflicts between sessions at the source.

Session lifecycle is managed by `sessionRunner.ts`, maintaining session handoff security with JWT-signed `WorkSecret`.

### Insight: Design Lessons from Remote Control

The noteworthy aspect of the Bridge system is its **session isolation strategy**. Using Git worktree as the session isolation unit is a pragmatic choice that leverages existing infrastructure (Git) to achieve filesystem-level isolation. It has lower overhead than container or VM-level isolation while still giving each session its own independent working directory.

The 2-tier authentication structure (Standard/Elevated) is also instructive. Requiring maximum authentication for all remote operations degrades user experience; requiring only the minimum weakens security. Varying the authentication level based on operation sensitivity is a realistic approach to balancing security and convenience.

---

## 8. What This Architecture Means for the Industry

Seeing inside Claude Code through leaked code is a rare opportunity to examine the real-world architecture of an agentic AI tool. Here are several key lessons.

### Agentic Systems Are More Complex Than You Think

"Call the LLM API and execute tools — done" is a demo-level understanding. Production agentic systems require:

- **Context management**: 4-tier compaction layers, tool sorting for cache stability, server-client budget synchronization
- **Error recovery**: Cost-aware cascades, diminishing returns detection, circuit breakers
- **Security**: 8 layers of defense in depth, ML-based auto-classification, build-time code elimination
- **Performance**: Parallel tool execution during streaming, deferred concurrent prefetching, asymmetric transcript saving

Claude Code's 4,600 files are evidence of this complexity. Competing products like OpenAI's Codex, Cursor, and Devin likely harbor similar levels of complexity internally. Their code simply has never been made public.

### Cost Drives Architecture

If there is one principle running through Claude Code's architecture, it is **cost-awareness**.

- Error recovery starts with free options first
- Message compaction starts with methods that require no API calls
- Token waste stops when diminishing returns are detected
- Tool sorting optimizes prompt cache hit rates
- Tools are loaded only when needed

In an LLM system where every API call is a cost, "cheapest method first" is not technical elegance — it is an economic survival strategy. Every team building agentic systems will eventually face this problem.

### We Need a New Definition of "Open Source"

The Claude Code case raises questions about what "open source" means in the AI era. The official repository contains only the plugin interface (279 files), while the core engine (4,600+ files) sits behind a commercial license. Can we call this "open source"?

This is not solely Anthropic's issue. Many AI products publish only extension interfaces under the "open" label while keeping the core proprietary. It is important for users and the developer community to recognize this distinction.

---

## 9. Conclusion: What the Leak Tells Us

Claude Code's internals, revealed through the absurd channel of an npm source map, turned out to be not a simple API wrapper but a full-stack agent platform spanning 4,600 files.

Eight layers of defense-in-depth security, 4-tier message compaction, diminishing returns detection, cost-aware error recovery, tool sorting for cache stability — these details are production-level engineering that simply cannot emerge from an "API wrapper."

The unreleased features reveal Anthropic's direction: voice mode, web browser automation, multi-agent Coordinator, proactive Kairos mode. The transition from "coding assistant" to "autonomous software engineering platform" is already underway at the code level.

There is irony here. They implemented a sophisticated security mechanism for build-time dead code elimination, then forgot to strip source maps from that same build pipeline. They built 8 layers of onion-like protection around users' filesystems, while uploading their own source code to npm.

The patterns discussed in this analysis — cost-aware error recovery, multi-tier context compaction, build-time feature gates, diminishing returns detection — are not exclusive to Claude Code. They serve as reference for anyone building agentic AI systems. The principle of "try the cheapest recovery first" in particular applies to any LLM application where API cost is a key variable.

Perhaps this is the essence of software engineering. Even the most sophisticated systems fall to the simplest mistakes. What Anthropic's engineers built is undeniably an impressive agentic architecture. Just next time, hopefully they will remember to add `*.map` to their `.npmignore`.

---

*This post is a technical analysis of leaked source code and does not encourage redistribution or use of the code. All code copyrights belong to Anthropic PBC.*
