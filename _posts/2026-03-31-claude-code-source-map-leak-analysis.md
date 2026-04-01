---
layout: post
title: "Claude Code 내부 아키텍처 분석"
date: 2026-03-31 12:00:00
categories: ["Insights", "Agentic-AI"]
tags: ["Claude-Code", "Agentic-Architecture", "Context-Compaction", "Multi-Agent-Orchestration", "Security-Architecture"]
cover: /assets/images/insights.png
use_math: false
---

# Claude Code의 속살이 드러났습니다. npm Source Map 유출로 본 에이전틱 AI의 해부학

> Anthropic이 "오픈소스"라고 부르는 것과 실제로 오픈된 것 사이에는 4,600개 파일만큼의 간극이 있었습니다.

### TL;DR
- npm source map 실수로 Claude Code의 비공개 코어 엔진(4,600+ 파일) 전체가 노출되었습니다
- 공식 "오픈소스"는 플러그인 껍데기(279개)뿐 — 핵심 엔진은 상용 비공개였습니다
- 내부에는 8겹 보안 레이어, 4단계 메시지 압축, 비용 인식 에러 복구 등 정교한 프로덕션 아키텍처가 있습니다
- 미출시 피처 플래그에서 음성 모드, 멀티 에이전트 Coordinator, 선제적 Kairos 모드 등 로드맵이 드러났습니다
- 에이전틱 AI 시스템의 실전 아키텍처를 들여다볼 수 있는 드문 기회입니다

---

## 1. 무슨 일이 있었나

2026년 3월 31일, 보안 연구자 Chaofan Shou가 X에 스크린샷 몇 장을 올렸습니다. Anthropic의 Claude Code 내부 소스코드 전체가 npm 패키지 안에 고스란히 들어 있었다는 내용이었습니다. Source map(`.map`) 파일은 빌드 과정에서 압축·난독화된 코드를 원본 소스코드로 역추적할 수 있게 해주는 디버깅용 매핑 파일입니다. 프로덕션 배포에서는 제거하는 게 상식인데, 그것이 그대로 남아 있었던 겁니다.

[Reddit r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1s8ijfb/claude_code_source_code_has_been_leaked_via_a_map/)가 들끓었고, 커뮤니티는 몇 시간 만에 전체 코드베이스를 해부하기 시작했습니다.

의도적인 해킹이 아닙니다. 정교한 사회공학 공격도 아닙니다. 빌드 파이프라인에서 `.map` 파일을 프로덕션 패키지에서 제외하는 걸 누락한 것입니다. 내부 프롬프트 구조, 에이전트 오케스트레이션 로직, 툴 호출 패턴이 포함된 것으로 알려졌습니다. Anthropic이 Claude Code를 얼마나 정교하게 *설계*했는지 — 혹은 얼마나 허술하게 *배포*했는지 — 가 동시에 드러나는 순간이었습니다. 경쟁사와 보안 연구자 모두에게 의도치 않은 선물이 된 셈입니다.

한 가지 짚고 넘어가야 할 게 있습니다. Anthropic은 Claude Code를 "오픈소스"라고 마케팅해왔습니다. [공식 GitHub 저장소](https://github.com/anthropics/claude-code)가 있고, 누구나 볼 수 있습니다. 하지만 거기에 있는 건 뭘까요?

플러그인 시스템. Hook 예제. 설정 파일 템플릿. 11개의 예제 플러그인.

요리로 치면, 레스토랑 메뉴판과 테이블 세팅을 공개하고 "우리 주방은 오픈입니다"라고 말한 격입니다. 진짜 레시피, 조리법, 비밀 소스는 전부 뒤에 숨겨져 있었습니다. 이번에 npm source map을 통해 드러난 건 바로 그 주방 전체입니다 — TypeScript/React로 작성된 4,600개 이상의 소스 파일, 55개 이상의 디렉토리. 라이선스는 Apache 2.0이 아니라 Anthropic Commercial Terms of Service입니다. *진정한* 오픈소스와는 거리가 있습니다.

이 글에서는 법적, 윤리적 논의는 제쳐두겠습니다. 대신 유출된 코드가 드러내는 기술적 아키텍처를 깊이 파고들어 봅니다. 솔직히 말씀드리면, 꽤 인상적입니다. 실수로 유출한 부분만 빼면요.

---

## 2. 전체 아키텍처: 생각보다 훨씬 큽니다

<a href="/assets/images/claude-code-analysis/architecture-overview.png" data-lightbox="claude-code" data-title="전체 시스템 아키텍처">
  <img src="/assets/images/claude-code-analysis/architecture-overview.png" alt="전체 시스템 아키텍처" />
</a>

Claude Code를 처음 접하면 "API 래퍼 아닌가?"라고 생각하기 쉽습니다. 터미널에서 Claude에게 말 걸고, Claude가 코드를 고쳐주고, 끝. 심플해 보입니다.

유출된 코드를 열어보면 그 생각이 산산조각 납니다.

### 기술 스택

| 레이어 | 기술 | 왜 이 선택인가 |
|--------|------|----------------|
| 런타임 | **Bun** | Node.js 대비 빠른 시작 시간과 네이티브 TypeScript 지원. `feature()` 번들링 API로 빌드 타임 데드코드 제거가 가능합니다 |
| 언어 | **TypeScript** (strict mode) | 4,600+ 파일 규모의 코드베이스에서 타입 안전성은 선택이 아닌 필수입니다 |
| UI | **React 18 + [Ink](https://github.com/vadimdemedes/ink)** | 터미널에서 React 컴포넌트를 렌더링합니다. 권한 다이얼로그, 진행 표시줄, 멀티패널 레이아웃 등 복잡한 UI가 이 선택을 정당화합니다 |
| API 클라이언트 | `@anthropic-ai/sdk` | Anthropic 공식 SDK |
| MCP 클라이언트 | `@modelcontextprotocol/sdk` | LLM이 외부 도구·데이터 소스와 표준화된 방식으로 통신하기 위한 개방형 프로토콜([Model Context Protocol](https://modelcontextprotocol.io/)) |
| 피처 플래그 | **GrowthBook** | 서버 측 기능 제어 및 A/B 테스트 |
| 번들러 | **Bun bundler** | `feature()` 기반 데드코드 제거로 내부/외부 빌드 분리 |

터미널 앱에 React를 쓴다는 게 과하게 느껴질 수 있습니다. 하지만 코드를 보면 권한 다이얼로그(`PermissionDialog.tsx`), Worker 배지(`WorkerBadge`), 멀티패널 레이아웃, 실시간 스트리밍 UI 같은 복잡한 인터랙션이 이 선택을 이해하게 만듭니다. 터미널이지만 단순한 텍스트 출력이 아니라, 상당히 풍부한 UI를 제공하고 있습니다.

### 공식 오픈소스 vs 유출 코드: 숫자로 보는 괴리

| | 공식 저장소 (`anthropics/claude-code`) | 유출 코드 |
|---|---|---|
| 파일 수 | ~279개 (스크립트/설정 위주) | **4,600개+** (풀스택 엔진) |
| 핵심 엔진 | **미포함** | 포함 |
| 도구 구현체 | **미포함** | 전체 포함 (Bash, Read, Write, Edit...) |
| 에이전틱 루프 | **미포함** | 포함 (`query.ts` 1,729줄) |
| 권한 시스템 | **미포함** | 포함 (`permissions.ts` 52K) |
| API 통신 | **미포함** | 포함 (스트리밍, 캐싱, 폴백) |
| Bridge/원격 | **미포함** | 포함 (33+ 파일) |
| MCP 클라이언트 | **미포함** | 포함 (`client.ts` 119K) |
| 라이선스 | Anthropic Commercial ToS | 비공개 상용 코드 |

279개 vs 4,600개. "오픈소스"라는 단어의 정의에 대해 다시 한번 생각해 볼 필요가 있겠습니다.

---

## 3. 에이전틱 루프: 심장을 열어봅니다

<a href="/assets/images/claude-code-analysis/agentic-loop.png" data-lightbox="claude-code" data-title="에이전틱 루프 상태 머신">
  <img src="/assets/images/claude-code-analysis/agentic-loop.png" alt="에이전틱 루프 상태 머신" />
</a>

Claude Code의 심장은 `query.ts`입니다 — 1,729줄의 `while(true)` 루프입니다. "에이전틱 루프"라고 부르는 이것이 사용자 입력을 받아 도구를 실행하고, 결과를 다시 Claude에게 던지는 전체 사이클을 관장합니다.

### 3.1 Async Generator: 우아한 설계 선택

가장 먼저 눈에 띄는 건 함수 시그니처입니다.

```typescript
export async function* query(
  params: QueryParams,
): AsyncGenerator<
  | StreamEvent
  | RequestStartEvent
  | Message
  | TombstoneMessage
  | ToolUseSummaryMessage,
  Terminal  // 반환값: 종료 이유
>
```

`async function*` — async generator입니다. 이벤트를 `yield`로 스트리밍하면서, 최종 종료 시에는 `Terminal` 타입을 `return`합니다. 이게 왜 영리한 선택인지 설명하겠습니다.

일반적인 접근이라면 이벤트 이미터(EventEmitter)나 콜백 기반으로 구현했을 겁니다. 하지만 async generator를 쓰면 **이벤트 스트림과 종료 시맨틱을 하나의 함수에서 처리**할 수 있습니다. 소비자는 `for await...of`로 이벤트를 받다가, 루프가 끝나면 `return` 값에서 종료 이유(`Terminal`)를 가져옵니다. 에러 전파도 자연스럽습니다 — generator 안에서 `throw`하면 소비자의 `try-catch`로 전파됩니다.

복잡한 상태 머신을 표현하기에 놀라울 정도로 깔끔한 패턴이며, 에이전틱 루프라는 도메인에 특히 잘 맞습니다.

### 3.2 불변 파라미터 + 가변 상태: Continue Site 패턴

루프는 두 종류의 데이터를 분리합니다.

**불변 파라미터** — 루프 전체에서 변하지 않는 것들:
```typescript
type QueryParams = {
  messages: Message[]
  systemPrompt: SystemPrompt
  canUseTool: CanUseToolFn       // 권한 검사 콜백
  toolUseContext: ToolUseContext   // 도구 실행 컨텍스트
  taskBudget?: { total: number }  // API task_budget (beta)
  maxTurns?: number               // 최대 턴 제한
  fallbackModel?: string          // 폴백 모델
  querySource: QuerySource        // 쿼리 출처 (REPL, agent 등)
  // ...
}
```

**가변 상태** — 매 이터레이션마다 갱신되는 것들:
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
  transition: Continue | undefined  // 이전 이터레이션의 계속 사유
}
```

여기서 주목할 패턴은 **"Continue Site"**입니다. 루프 안에서 상태를 갱신하고 다음 이터레이션으로 넘어가는 코드 지점(site)을 가리키는 용어인데, 소스 코드 주석에서 직접 이렇게 설명하고 있습니다.

```typescript
// Continue sites write `state = { ... }` instead of 9 separate assignments.
```

9개의 개별 필드를 하나씩 수정하는 대신, 전체 상태 객체를 한 번에 재할당합니다.

```typescript
state = {
  ...state,
  messages: newMessages,
  turnCount: nextTurnCount,
  transition: { reason: 'next_turn' }
}
```

이 패턴의 장점은 두 가지입니다. 첫째, 상태 전이가 **원자적(atomic)**입니다 — 9개 필드 중 5개만 업데이트되고 에러가 나는 중간 상태가 없습니다. 둘째, `transition` 필드를 통해 **왜 계속했는지**를 추적할 수 있어서, 테스트에서 복구 경로가 제대로 작동했는지 메시지 내용을 들여다보지 않고도 단언(assert)할 수 있습니다.

React의 `setState` 철학이 백엔드 루프에까지 스며든 것으로, Anthropic 엔지니어들의 React 사랑이 엿보이는 부분입니다.

이 패턴이 중요한 이유는, 에이전틱 루프에서 상태 관리 버그가 곧 사용자 비용으로 이어지기 때문입니다. 상태가 꼬이면 불필요한 API 호출이 발생하고, 그건 곧 토큰 비용입니다. Cursor나 Windsurf 같은 경쟁 제품들도 비슷한 에이전틱 루프를 구현하고 있을 텐데, 이 수준의 상태 관리 엄밀성을 갖추고 있는지는 알 수 없습니다. 코드가 공개되지 않았으니까요.

### 3.3 턴당 6단계 파이프라인: 상세 워크스루

각 턴은 다음 6단계를 거칩니다. 소스 코드의 실제 라인 번호와 함께 살펴보겠습니다.

#### 1단계: Pre-Request Compaction (lines 365-548)

API를 호출하기 *전에* 대화 히스토리를 정리합니다. 5개의 압축 메커니즘이 **순서대로** 적용되는데, 각 메커니즘의 상세한 작동 원리는 섹션 4에서 다룹니다. 여기서는 파이프라인 안에서의 흐름만 보겠습니다.

```typescript
// 1. Tool Result Budget 적용 (lines 369-394)
// 도구 실행 결과가 너무 클 때 크기를 제한해서 컨텍스트 윈도우를 효율적으로 쓰기 위한 예산 시스템
messagesForQuery = await applyToolResultBudget(
  messagesForQuery,
  toolUseContext.contentReplacementState,
  // 에이전트/REPL 소스만 교체 기록을 저장 (resume을 위해)
  persistReplacements ? records => void recordContentReplacement(...) : undefined,
)

// 2. Snip Compact — 가장 저렴 (lines 401-410)
if (feature('HISTORY_SNIP')) {
  const snipResult = snipModule!.snipCompactIfNeeded(messagesForQuery)
  messagesForQuery = snipResult.messages
  snipTokensFreed = snipResult.tokensFreed
}

// 3. Microcompact — 캐시 인식 도구 결과 클리어 (lines 413-426)
const microcompactResult = await deps.microcompact(messagesForQuery, toolUseContext, querySource)
messagesForQuery = microcompactResult.messages

// 4. Context Collapse — 단계적 축소 (lines 440-447)
if (feature('CONTEXT_COLLAPSE') && contextCollapse) {
  const collapseResult = await contextCollapse.applyCollapsesIfNeeded(...)
  messagesForQuery = collapseResult.messages
}

// 5. Auto-Compact — 임계값 초과 시 전체 요약 (lines 453-543)
const { compactionResult, consecutiveFailures } = await deps.autocompact(...)
```

이 순서가 중요합니다. Context Collapse가 Auto-Compact **앞에** 오는 이유는 소스 코드 주석에서 직접 설명합니다.

```typescript
// Runs BEFORE autocompact so that if collapse gets us under the
// autocompact threshold, autocompact is a no-op and we keep granular
// context instead of a single summary.
```

Collapse로 충분히 줄어들면 Auto-Compact(비싼 API 호출)가 발동하지 않습니다. 세밀한 컨텍스트를 최대한 보존하면서 비용을 아끼는 전략입니다.

압축 후에는 서버가 전체 히스토리를 볼 수 없으므로, 클라이언트가 task budget의 잔여량을 직접 추적해서 서버에 알려줘야 합니다. 단순해 보이지만, 압축 경계에서 서버-클라이언트 간 상태를 동기화하는 건 에이전틱 시스템에서 흔히 겪는 미묘한 문제입니다.

#### 2단계: API Call & Streaming (lines 659-863)

```typescript
for await (const message of deps.callModel(
  fullSystemPrompt,
  prependUserContext(messagesForQuery, userContext),
  toolUseContext,
  { taskBudget, taskBudgetRemaining, maxOutputTokensOverride, skipCacheWrite },
)) {
  // 스트리밍 이벤트 처리
}
```

여기서의 핵심은 **StreamingToolExecutor**입니다. Claude가 응답을 생성하는 *동안* 도구가 병렬로 실행됩니다. Claude가 "파일을 읽어볼게요"라고 타이핑하는 중에 이미 파일이 읽히고 있다는 뜻입니다. 사용자가 체감하는 대기 시간이 줄어드는 건 이런 트릭 덕분입니다.

소스를 보면 이 병렬 실행의 구현이 상당히 정교합니다.

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

모든 도구에는 `isConcurrencySafe` 플래그가 있습니다. `FileReadTool`, `GlobTool`, `GrepTool` 같은 읽기 전용 도구는 병렬 실행이 안전합니다. `FileWriteTool`이나 `BashTool`처럼 상태를 변경하는 도구는 직렬로 실행해야 합니다. 그리고 한 도구가 에러를 내면 형제 도구들도 `sibling_error`로 취소됩니다.

```typescript
type AbortReason =
  | 'sibling_error'       // 형제 도구 에러 → 나도 취소
  | 'user_interrupted'     // 사용자가 Ctrl+C / ESC
  | 'streaming_fallback'   // 모델 폴백으로 폐기
```

모델 폴백도 여기서 처리됩니다. 주 모델이 실패하면:

```typescript
if (innerError instanceof FallbackTriggeredError && fallbackModel) {
  currentModel = fallbackModel
  attemptWithFallback = true
  // 고아(orphan) 메시지에 tombstone 생성
  yield* yieldMissingToolResultBlocks(assistantMessages, 'Model fallback triggered')
  // StreamingToolExecutor 초기화 후 재시도
}
```

Tombstone(묘비) 메시지는 데이터베이스에서 '이 항목은 삭제됨'을 표시하는 마커에서 빌려온 용어입니다. 여기서는 "이 도구 호출은 모델 폴백으로 인해 폐기되었습니다"라는 기록을 남겨서, 대화 히스토리의 일관성을 유지합니다.

#### 3단계: 에러 복구 캐스케이드 (lines 1062-1256)

이 부분이 아키텍처적으로 가장 인상적인 곳입니다. 에러가 발생하면 즉시 포기하지 않습니다. **저비용에서 고비용 순서로** 복구를 시도합니다.

**Prompt-too-long (413 에러) 복구 — 3단계 캐스케이드:**

```
1단계: Context Collapse 드레인 (비용: 0)
  └ 이미 준비된 축소를 플러시합니다. 추가 API 호출 없이 즉시 적용됩니다.

2단계: Reactive Compact (비용: API 1회 호출)
  └ 전체 대화를 요약합니다. 이미지를 스트리핑하고, 요약 후 재시도합니다.
  └ "strip retry" — 요약 자체도 너무 크면 미디어를 제거한 뒤 한 번 더 시도합니다.

3단계: 에러 표출
  └ 모든 시도가 실패하면 사용자에게 에러를 보여줍니다.
```

**Max-output-tokens 복구 — 역시 3단계:**

```
1단계: 토큰 캡 에스컬레이션 (비용: 0)
  └ 8K → 64K (ESCALATED_MAX_TOKENS)로 투명하게 올립니다.
  └ 메타 메시지 없이 자동으로 처리됩니다. 사용자는 모릅니다.

2단계: Resume 메시지 주입 (비용: API 재호출, 최대 3회)
  └ "이전 응답이 잘렸습니다. 잘린 부분부터 이어서 작성해주세요"
  └ 라는 메시지를 주입하고 모델을 다시 호출합니다.
  └ maxOutputTokensRecoveryCount를 추적해서 최대 3회까지만 시도합니다.

3단계: 복구 소진
  └ 3회 시도 후에도 해결 안 되면 현재까지의 결과로 완료 처리합니다.
```

이 캐스케이드의 설계 원칙은 명확합니다. **첫 번째 시도는 항상 무료(free)입니다.** 전체 대화를 요약하는 비싼 작업은 최후의 수단입니다. 이건 단순한 "재시도"가 아니라, 비용-효과를 정밀하게 고려한 복구 전략입니다.

#### 4단계: Stop Hooks & Token Budget (lines 1267-1355)

Stop hook은 사용자 정의 검증 로직을 실행합니다. "테스트를 통과하지 않으면 멈추지 마"라는 hook을 걸면, Claude가 작업을 끝내려 할 때 hook이 실행되고, 실패하면 hook의 에러 메시지가 대화에 주입되어 Claude가 다시 시도합니다.

토큰 버짓은 더 흥미롭습니다.

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

**감소 수익(diminishing returns) 감지**가 내장되어 있습니다. 3회 연속으로 계속했는데 매번 500 토큰도 생성하지 못하면, "이건 더 해봤자 의미 없다"고 판단하고 멈춥니다. 90% 버짓 소진 전까지는 계속하되, 삽질은 하지 않습니다.

이 로직이 왜 중요한지 생각해보면: 에이전틱 루프에서 가장 위험한 건 **무한 루프**입니다. Claude가 "한 번만 더 고쳐볼게요"를 반복하면서 토큰만 소비하고 진전이 없는 상황. 이 감소 수익 감지는 그런 상황을 자동으로 차단합니다.

#### 5단계: Tool 실행 (lines 1363-1520)

스트리밍 모드(2단계에서 이미 시작)와 배치 모드가 공존합니다.

```typescript
// 스트리밍: 병렬 안전 도구는 API 응답 생성 중에 이미 실행
if (streamingToolExecutor) {
  toolResults = await streamingToolExecutor.getRemainingResults()
}

// 배치: 나머지 도구는 여기서 순차 실행
toolResults = await runTools(toolUseBlocks, toolUseContext, canUseTool)
```

프로그레스 시그널링도 있습니다. `progressAvailableResolve`라는 Promise resolver를 통해 "새 진행 상황이 있다"를 소비자에게 알립니다. 이게 터미널 UI의 실시간 스피너 업데이트를 구동합니다.

#### 6단계: Post-Tool & 다음 턴 전이 (lines 1547-1727)

도구 실행이 끝나면 여러 "부가 작업"을 처리합니다.

```typescript
// 1. 스킬 디스커버리 소비 — 1단계에서 prefetch 시작한 것을 여기서 수확
if (pendingSkillPrefetch?.settledAt !== null) {
  const skillAttachments = await pendingSkillPrefetch.promise
}

// 2. 메모리 어태치먼트 소비 — 역시 prefetch한 것을 여기서 수확
if (pendingMemoryPrefetch?.settledAt !== null) {
  const memoryAttachments = await pendingMemoryPrefetch.promise
}

// 3. 큐된 명령어 드레인 — 슬래시 명령어, 태스크 알림 등
const queuedCommands = getCommandsByMaxPriority(...)

// 4. MCP 서버 도구 새로고침
if (toolUseContext.options.refreshTools) {
  toolUseContext.options.tools = toolUseContext.options.refreshTools()
}
```

그리고 상태를 전이합니다.

```typescript
// Continue Site: 다음 턴으로
state = {
  messages: [...messagesForQuery, ...assistantMessages, ...toolResults],
  toolUseContext: toolUseContextWithQueryTracking,
  autoCompactTracking: tracking,
  turnCount: nextTurnCount,
  transition: { reason: 'next_turn' },
}
// while(true) 루프 상단으로 돌아감
```

### 3.4 종료 이유: 루프가 끝나는 9가지 방법

| Exit Reason | 의미 | 발생 지점 |
|-------------|------|-----------|
| `completed` | 정상 완료 (도구 호출 없이 응답 종료) | line 1264, 1357 |
| `blocking_limit` | 하드 토큰 한도 도달 | line 646 |
| `aborted_streaming` | 스트리밍 중 사용자 중단 (Ctrl+C) | line 1051 |
| `aborted_tools` | 도구 실행 중 사용자 중단 | line 1515 |
| `prompt_too_long` | 복구 시도 후에도 프롬프트 초과 | line 1175, 1182 |
| `image_error` | 이미지 검증 실패 (크기 초과 등) | line 977, 1175 |
| `model_error` | 예상치 못한 모델 에러 | line 996 |
| `hook_stopped` | Stop hook이 진행을 차단 | line 1520 |
| `max_turns` | 최대 턴 수 초과 (`maxTurns` 파라미터) | line 1711 |

### 3.5 QueryEngine.ts: 세션과 턴의 상위 관리자

`query.ts`가 단일 턴의 루프라면, `QueryEngine.ts`(1,295줄)는 **세션 전체**를 관리합니다.

```typescript
class QueryEngine {
  mutableMessages: Message[]       // 전체 대화 히스토리
  permissionDenials: PermissionDenial[]  // 도구 권한 거부 기록
  totalUsage: Usage                // 누적 토큰 사용량
  readFileState: FileStateCache    // 파일 상태 캐시 (중복 읽기 방지)
  discoveredSkillNames: Set<string> // 발견된 스킬 (턴마다 초기화)
  loadedNestedMemoryPaths: Set<string> // 로드된 메모리 경로 (중복 방지)
}
```

`submitMessage()` 메서드가 사용자 입력을 받아서 `query()`를 호출하고, 결과를 세션에 축적합니다.

```typescript
// 사용량 누적
this.totalUsage = accumulateUsage(this.totalUsage, currentMessageUsage)
```

트랜스크립트 기록에도 비대칭적 전략이 적용됩니다.

```typescript
// 사용자 메시지: 블로킹 저장 (--resume 복원에 필수)
await recordTranscript(userMessage)

// 어시스턴트 메시지: fire-and-forget (실행만 하고 완료를 기다리지 않는 비동기 패턴)
recordTranscript(assistantMessage)  // await 없음
```

모든 메시지를 같은 우선순위로 저장하지 않는 것입니다. 사용자 메시지는 세션 복원에 필수적이니 블로킹으로 저장하고, 어시스턴트 응답은 "저장은 하되 기다리지는 않는다"는 전략입니다. 성능과 안정성의 균형점을 찾은 설계입니다.

---

## 4. 메시지 압축: 컨텍스트 윈도우와의 정교한 전쟁

<a href="/assets/images/claude-code-analysis/compaction-layers.png" data-lightbox="claude-code" data-title="메시지 압축 계층">
  <img src="/assets/images/claude-code-analysis/compaction-layers.png" alt="메시지 압축 계층" />
</a>

에이전틱 AI 도구의 최대 적은 컨텍스트 윈도우 한계입니다. 긴 대화에서 이전 내용이 잘려나가거나 API가 413을 반환합니다. Claude Code의 압축 시스템은 이 문제를 4개 계층으로 해결하며, 각 계층은 비용과 정보 손실 정도가 다릅니다. 핵심 원칙은 **"항상 가장 저렴한 것부터"**입니다.

### 4.1 Snip Compact — 가장 저렴하고 가장 과격합니다

비용: **무료** (API 호출 없음)
정보 손실: **높음**

"Snip(잘라내기)"이라는 이름 그대로, 오래된 메시지를 통째로 잘라내고 최근 컨텍스트만 남깁니다. `HISTORY_SNIP` 피처 플래그 뒤에 있으며, 주로 헤드리스(headless, UI 없이 백그라운드에서 실행되는) 세션에서 사용됩니다.

```typescript
const snipResult = snipModule!.snipCompactIfNeeded(messagesForQuery)
messagesForQuery = snipResult.messages
snipTokensFreed = snipResult.tokensFreed
```

`snipTokensFreed`가 자동 압축 단계로 전달되는 게 중요합니다. 소스 주석에 따르면:

```typescript
// snipTokensFreed is plumbed to autocompact so its threshold check reflects
// what snip removed; tokenCountWithEstimation alone can't see it
```

Snip이 이미 상당량을 줄였다면, Auto-Compact가 불필요하게 발동하는 걸 방지합니다.

### 4.2 Microcompact (530줄) — 캐시를 존중하는 선택적 클리어

비용: **무료** (API 호출 없음)
정보 손실: **중간**

"Micro"라는 이름은 전체 대화를 한꺼번에 압축하는 게 아니라, 특정 도구의 결과만 골라서 개별적으로 클리어한다는 뜻입니다.

```typescript
// 클리어 대상: file_read, shell, grep, glob, web_search, web_fetch, file_edit, file_write
// 결과를 "[Old tool result content cleared]"로 교체
// 이미지/문서: 2,000 토큰으로 추정
// 텍스트: 대략적 토큰 카운트로 추정
```

이 계층의 진짜 핵심은 **캐시 편집 블록 피닝**입니다. Anthropic API에는 프롬프트의 앞부분(프리픽스)이 이전 요청과 동일하면 서버가 재계산을 건너뛰는 **프롬프트 캐싱** 기능이 있는데, `CACHED_MICROCOMPACT` 피처 플래그 뒤에 있는 이 기능은 그 캐싱과 조화를 이루도록 설계되어 있습니다.

```typescript
const pendingCacheEdits = feature('CACHED_MICROCOMPACT')
  ? microcompactResult.compactionInfo?.pendingCacheEdits
  : undefined
```

이미 캐시된 도구 결과의 ID를 추적하고, 캐시 히트가 유지되도록 관리합니다. 프롬프트 캐싱에서 캐시 미스 한 번은 수만 토큰의 재계산을 의미합니다. 캐시 히트율을 유지하면서 컨텍스트를 줄이는 건 서로 상충하는 목표인데, 이 설계가 그 균형을 잡으려 합니다.

### 4.3 Context Collapse — 단계적 축소

비용: **낮음**
정보 손실: **중간**

"단계적 축소(staged collapse)"라는 독특한 개념입니다. 전체 대화를 한 번에 압축하는 대신, 프리뷰 단계에서 어떤 메시지 블록을 축소할지 결정하고, 커밋 단계에서 실제로 축소합니다.

소스 주석이 이 설계의 핵심을 설명합니다.

```typescript
// Nothing is yielded — the collapsed view is a read-time projection
// over the REPL's full history. Summary messages live in the collapse
// store, not the REPL array. This is what makes collapses persist
// across turns: projectView() replays the commit log on every entry.
```

축소된 뷰는 REPL의 전체 히스토리에 대한 **읽기 시점 프로젝션(read-time projection)**입니다 — 원본 메시지를 수정하지 않고, 읽을 때마다 축소된 뷰를 그때그때 계산해서 보여주는 방식입니다. 축소 결과는 별도 저장소에 있고, 원본은 그대로 남아있습니다. Git이 원본 파일을 건드리지 않으면서 다른 브랜치의 스냅샷을 보여주는 것과 개념적으로 유사합니다.

이 접근의 장점은 프롬프트 캐싱의 캐시 히트를 최대화할 수 있다는 것입니다. 원본 메시지가 변하지 않으므로, 캐시된 프리픽스가 무효화되지 않습니다.

### 4.4 Auto-Compact (351줄) — 최후의 수단

비용: **높음** (Claude API 추가 호출)
정보 손실: **낮음** (AI가 요약하므로 핵심은 보존)

전체 대화 히스토리를 Claude에게 보내서 요약을 요청합니다. 임계값 로직은 명확합니다.

```typescript
function getAutoCompactThreshold(model: string): number {
  const effectiveContextWindow = getEffectiveContextWindowSize(model)
  return effectiveContextWindow - 13_000  // 13K 토큰 버퍼
}
```

컨텍스트 윈도우에서 13K 토큰을 남겨두고, 나머지가 차면 자동 압축이 발동합니다.

**서킷 브레이커(circuit breaker)** — 전기의 차단기처럼 연속 실패를 감지하면 더 이상 시도하지 않는 패턴 — 가 있습니다.

```typescript
MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3
```

대화가 너무 길어서 요약 요청 자체도 컨텍스트 한도를 넘으면, 자동 압축이 실패합니다. "압축 실패 → 재시도 → 또 실패"의 무한 루프에 빠질 수 있는데, 3회 연속 실패 시 서킷 브레이커가 작동해서 깔끔하게 포기합니다. 방어적 프로그래밍의 교과서적 예시입니다.

### 4.5 Full Compaction (1,705줄) — 최후의 카드

Auto-Compact가 실행할 때 내부적으로 호출하는 전체 압축 로직입니다.

1. **이미지 스트리핑**: 이미지와 문서를 `[image]` / `[document]` 플레이스홀더로 교체 (토큰 대폭 절감)
2. **API 라운드 그룹핑**: `tool_use` → `tool_result` 쌍을 그룹으로 묶어 처리 (의미적 단위 보존)
3. **Thinking 블록 제거**: (Anthropic 내부 빌드 전용) 추론 과정 블록을 압축 전에 제거
4. **PTL(Prompt-Too-Long) 재시도**: 압축 요청 자체가 한도를 넘으면, 가장 오래된 API 라운드 그룹부터 20%씩 드롭

```typescript
truncateHeadForPTLRetry(messages, ptlResponse) {
  // Drop oldest API-round groups until prompt-too-long gap is covered
  // Falls back to dropping 20% of groups
}
```

### 4.6 토큰 경고 상태 시스템

4단계 경고 시스템이 사용자에게 시각적 피드백을 제공합니다.

```
[정상]
  ↓ 컨텍스트 윈도우 - 20K 토큰
[Warning] ← 노란색 경고
  ↓ 컨텍스트 윈도우 - 20K 토큰
[Error] ← 주황색 경고
  ↓ 컨텍스트 윈도우 - 13K 토큰
[AutoCompact] ← 자동 압축 발동
  ↓ 컨텍스트 윈도우 - 3K 토큰
[BlockingLimit] ← 빨간색, 수동 압축만 가능
```

### 인사이트: 왜 4단계인가

4단계 압축이 단순히 "점점 더 세게 압축"하는 게 아닙니다. 각 단계는 **서로 다른 트레이드오프**를 가집니다.

- **Snip**: 정보 손실 크지만 캐시 히트에 영향 없음
- **Microcompact**: 선택적 손실이지만 캐시를 존중함
- **Context Collapse**: 원본 보존하면서 뷰만 축소
- **Auto-Compact**: 정보 손실 최소지만 API 비용 발생

이건 "비용 최소화"와 "정보 보존 최대화"라는 두 축 위에서 파레토 최적점을 찾는 문제입니다. 파레토 프론티어란 하나를 개선하면 반드시 다른 하나가 나빠지는 트레이드오프의 경계선을 말합니다. 4개 계층은 그 경계선 위의 서로 다른 균형점을 나타냅니다.

---

## 5. 도구 시스템: 체계적으로 확장 가능한 스위스 아미 나이프

Claude Code가 실제로 *할 수 있는 것*은 도구 시스템이 결정합니다. `tools.ts`를 보면 도구 등록 방식의 전체 그림이 보입니다.

### 5.1 도구 인터페이스

모든 도구는 같은 인터페이스를 따릅니다 (`Tool.ts`에서):

```typescript
// ToolUseContext — 도구 실행에 필요한 모든 것이 여기 담깁니다
type ToolUseContext = {
  options: { tools: Tools; mainLoopModel: string; mcpClients: MCPServerConnection[]; maxBudgetUsd?: number; ... }
  abortController: AbortController
  readFileState: FileStateCache    // 중복 파일 읽기 방지
  getAppState(): AppState          // 애플리케이션 상태 접근
  setAppState(f: (prev: AppState) => AppState): void
  // ... 40+ 필드 (에이전트 ID, 권한 추적, 콘텐츠 교체 상태 등)
}
```

주목할 점은 `ToolPermissionContext`입니다.

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
  shouldAvoidPermissionPrompts?: boolean  // 백그라운드 에이전트용
  awaitAutomatedChecksBeforeDialog?: boolean  // 코디네이터 워커용
  prePlanMode?: PermissionMode  // 플랜 모드 진입 전 모드 저장
}>
```

`DeepImmutable`로 감싸져 있습니다 — 객체의 모든 중첩 속성까지 재귀적으로 읽기 전용으로 잠그는 TypeScript 유틸리티 타입입니다. 권한 컨텍스트는 실수로라도 수정할 수 없습니다.

### 5.2 도구 등록의 3계층 구조

`getAllBaseTools()` 함수를 보면, Claude Code의 도구는 세 계층으로 나뉩니다.

**항상 활성화**: `BashTool`, `FileReadTool`, `FileEditTool`, `WebSearchTool`, `AgentTool` 등 20여 개의 기본 도구. 이것들이 Claude Code의 핵심 능력을 구성합니다.

**조건부 활성화**: 환경이나 설정에 따라 켜지는 도구들입니다. 예를 들어 `GlobTool`과 `GrepTool`은 Anthropic 내부 빌드에서는 비활성화됩니다 — Bun 바이너리에 `bfs`/`ugrep`가 임베딩되어 있어서 별도 도구가 불필요하기 때문입니다. `PowerShellTool`은 Windows에서만, `LSPTool`은 환경변수로 명시적으로 켜야 합니다.

**피처 플래그 기반 (미출시)**: `feature()` 함수로 게이트된 도구들입니다. `WebBrowserTool`, `WorkflowTool`, `SleepTool`, `PushNotificationTool` 등 15개 이상이 여기에 속합니다. 이것들은 섹션 7에서 자세히 다룹니다.

흥미로운 건 `USER_TYPE === 'ant'`로 분기되는 Anthropic 내부 전용 도구입니다.

```typescript
const REPLTool = process.env.USER_TYPE === 'ant'
  ? require('./tools/REPLTool/REPLTool.js').REPLTool
  : null
```

`REPLTool`, `ConfigTool`, `TungstenTool`, `SuggestBackgroundPRTool` — 이 도구들은 외부 사용자에게는 존재하지 않습니다. Anthropic 엔지니어들이 내부적으로 사용하는 도구 세트가 따로 있다는 뜻입니다. 특히 `TungstenTool`은 이름만으로는 용도를 알 수 없는데, 텅스텐(고밀도 금속)이라는 이름에서 "무거운 작업"을 처리하는 도구로 추정됩니다.

이 3계층 구조가 왜 중요할까요? 에이전틱 AI 도구에서 "어떤 능력을 줄 것인가"는 핵심 설계 결정입니다. 도구가 너무 많으면 시스템 프롬프트가 비대해지고 (토큰 비용 증가), 너무 적으면 에이전트의 능력이 제한됩니다. Claude Code는 이 문제를 빌드 타임 제거 + 조건부 활성화 + 피처 플래그의 3계층으로 해결합니다. Cursor나 Devin도 비슷한 도구 확장 문제를 겪을 텐데, 이런 계층적 접근은 참고할 만합니다.

### 5.3 도구 풀 조립: 캐시 안정성까지 고려한 설계

`assembleToolPool()` 함수에서 빌트인 도구와 MCP 도구를 합치는 로직이 있는데, 여기서 프롬프트 캐시 안정성을 위한 흥미로운 처리가 보입니다.

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

빌트인 도구와 MCP 도구를 **각각 따로 정렬한 뒤 이어 붙입니다.** 왜 전체를 한 번에 정렬하지 않을까요? 서버의 `claude_code_system_cache_policy`가 빌트인 도구의 마지막 항목 뒤에 캐시 브레이크포인트를 놓기 때문입니다. 평면 정렬을 하면 MCP 도구가 빌트인 도구 사이에 끼어들어서, MCP 도구가 추가/제거될 때마다 모든 다운스트림 캐시 키가 무효화됩니다.

이 수준의 캐시 최적화는 실제 프로덕션에서의 비용 경험에서 나온 것으로 보입니다.

### 5.4 동적 도구 검색 (Tool Search)

`tool-search-2025-10-16` 베타 기능입니다. 도구가 수십 개가 되면 시스템 프롬프트만으로도 상당한 토큰을 차지합니다. 이 기능은 모든 도구를 처음부터 Claude에게 보여주는 대신, **필요할 때 검색해서 로드**합니다.

```typescript
// tools.ts:247-249
// Include ToolSearchTool when tool search might be enabled (optimistic check)
// The actual decision to defer tools happens at request time in claude.ts
...(isToolSearchEnabledOptimistic() ? [ToolSearchTool] : []),
```

"optimistic check"라는 표현이 눈에 띕니다. 등록 시점에서는 낙관적으로 포함하고, 실제 지연(defer) 결정은 API 요청 시점에 합니다. 이건 게으른(lazy) 로딩의 LLM 버전입니다.

---

## 6. 다층 보안: 양파를 까보면

<a href="/assets/images/claude-code-analysis/security-layers.png" data-lightbox="claude-code" data-title="보안 레이어">
  <img src="/assets/images/claude-code-analysis/security-layers.png" alt="보안 레이어" />
</a>

"AI 에이전트가 내 파일 시스템에 접근한다"는 문장은 보안 연구자를 떨리게 하기에 충분합니다. Claude Code가 이 문제를 어떻게 다루는지 보면, 양파 같은 다층 방어가 드러납니다. 8개 레이어입니다.

### 6.1 레이어 1: 빌드 타임 게이트 — 코드가 존재하지 않는 보안

```typescript
// tools.ts:117-119
const WebBrowserTool = feature('WEB_BROWSER_TOOL')
  ? require('./tools/WebBrowserTool/WebBrowserTool.js').WebBrowserTool
  : null
```

`feature()` 함수는 **빌드 타임에 평가**됩니다. Bun 번들러가 `false`인 분기의 `require()`를 완전히 제거합니다. 외부 빌드에는 Anthropic 내부 전용 도구의 코드가 *물리적으로 존재하지 않습니다*. 런타임에 환경변수를 바꿔서 활성화하는 건 불가능합니다 — 바이너리에 코드 자체가 없으니까요.

같은 코드베이스에서 "내부용"과 "외부용" 빌드를 분기하면서도, 런타임 분기문의 보안 위험을 피하는 기발한 접근입니다. 물론, 이번에 유출된 건 바로 그 빌드 타임에 제거되었어야 할 코드가 source map에 남아있었기 때문입니다. 아이러니한 상황입니다.

`USER_TYPE` 환경변수도 같은 패턴으로 사용됩니다.

```typescript
// tools.ts:16-19
const REPLTool =
  process.env.USER_TYPE === 'ant'
    ? require('./tools/REPLTool/REPLTool.js').REPLTool
    : null
```

### 6.2 레이어 2: 피처 플래그 — 서버 측 킬 스위치

빌드 후에도 서버 측에서 기능을 제어할 수 있습니다. GrowthBook 기반의 `tengu_` 접두사 플래그들이 그 역할을 합니다. (`tengu`는 Anthropic 내부의 Claude Code 프로젝트 코드네임으로 추정됩니다.)

| 플래그 | 역할 |
|--------|------|
| `tengu_amber_quartz_disabled` | 음성 모드 킬 스위치 |
| `tengu_bypass_permissions_disabled` | 권한 우회 모드 킬 스위치 |
| `tengu_auto_mode_config.enabled` | 자동 모드 서킷 브레이커 |
| `tengu_ccr_bridge` | 원격 제어 자격 확인 |
| `tengu_sessions_elevated_auth_enforcement` | 신뢰 디바이스 토큰 요구 |

보안 사고가 발생하면 서버에서 즉시 킬 스위치를 당길 수 있습니다. 클라이언트 업데이트 없이 기능을 비활성화할 수 있다는 건 인시던트 대응 관점에서 매우 중요합니다.

Anthropic 내부에서는 환경변수로 이를 오버라이드할 수도 있습니다.

```typescript
// CLAUDE_INTERNAL_FC_OVERRIDES (Ant-only)
// '{"my_feature": true, "my_config": {"key": "val"}}'
```

### 6.3 레이어 3: 설정 기반 규칙 — 8개 소스의 우선순위

```typescript
type PermissionRule = {
  source: PermissionRuleSource  // 8개 소스 중 하나
  ruleBehavior: 'allow' | 'deny' | 'ask'
  ruleValue: {
    toolName: string     // "Bash", "FileEdit" 등
    ruleContent?: string // 패턴, 예: "python:*"
  }
}
```

규칙 소스의 우선순위:
1. `userSettings` — `~/.claude/settings.json`
2. `projectSettings` — `.claude/settings.json`
3. `localSettings` — `.claude/local.json`
4. `flagSettings` — GrowthBook 피처 플래그
5. `policySettings` — 조직 수준 정책 (엔터프라이즈)
6. `cliArg` — 명령줄 인수
7. `command` — 런타임 규칙 업데이트
8. `session` — 인메모리 세션 전용 규칙

### 6.4 레이어 4: Transcript Classifier — AI가 AI를 감시합니다

`yoloClassifier.ts` — 52K짜리 파일입니다. (파일 이름이 "YOLO"인데 52K라니, Anthropic 엔지니어의 유머 감각이 엿보입니다.)

자동 모드(`auto`)에서는 사용자에게 매번 "이거 해도 될까요?"라고 묻는 대신, **Claude API를 한 번 더 호출해서** 도구 사용의 안전성을 판정합니다.

```
1. 도구 사용 요청 발생
2. 화이트리스트 체크
   └ FileRead, Grep, Glob, Tasks 등 읽기 전용 → 분류기 건너뛰고 자동 허용
3. Classifier API 호출 → "허용" 또는 "거부"
4. 거부 추적:
   └ 연속 3회 거부 → 프롬프팅 모드로 폴백
   └ 총 20회 거부 → 프롬프팅 모드로 폴백
```

거부 추적의 폴백 메커니즘이 중요합니다. 분류기가 지나치게 보수적으로 작동하면 사용자 경험이 나빠지니까, 일정 횟수 이상 거부하면 "차라리 사용자에게 물어보자"로 전환합니다.

디버깅용으로 `CLAUDE_CODE_DUMP_AUTO_MODE=1` 환경변수를 설정하면 분류기의 요청/응답을 `/tmp/claude-code/auto-mode/`에 JSON으로 덤프합니다. Claude Code의 자동 모드가 왜 특정 작업을 거부하는지 궁금할 때 유용한 팁입니다.

분류기가 API 에러를 반환하면? **프롬프팅으로 폴백**합니다 — 자동 거부가 아닙니다. "판단할 수 없으면 사용자에게 묻는다"는 fail-open — 장애 시 차단(fail-closed)이 아닌 통과 — 전략입니다. 보안에서 fail-open은 보통 위험하지만, 여기서는 "사용자가 직접 판단"이라는 폴백이므로 합리적입니다.

### 6.5 레이어 5: 위험 패턴 감지

```typescript
DANGEROUS_BASH_PATTERNS = [
  'python', 'node', 'ruby', 'perl', 'bash', 'sh', 'zsh', 'ksh',
  'exec', 'eval', 'source', 'curl', 'wget', 'nc', 'ncat',
  'socat', 'dd', 'xxd', 'openssl', 'ssh', 'scp', 'sftp',
  'sudo', 'su', 'chroot', 'unshare', 'docker', 'podman',
  'chmod', 'chown', 'chgrp', 'umask', 'mount', 'umount'
]
```

자동 모드에서 `python:*`이나 인터프리터 와일드카드를 허용 규칙으로 설정하려 하면 차단됩니다.

```typescript
isDangerousBashPermission(rule) {
  // 도구 수준 allow without content (모든 bash 허용) → 위험
  // "python:*", "python*", "python -*" → 위험
  // 인터프리터 프리픽스 + 와일드카드 → 위험
}
```

인터프리터를 통한 임의 코드 실행은 Bash 샌드박싱을 우회할 수 있기 때문입니다. `python -c "import os; os.system('rm -rf /')"` 같은 공격을 원천 봉쇄합니다.

### 6.6 레이어 6: 파일시스템 권한 검증 (62K)

가장 큰 권한 파일(62K)은 파일 경로 검증만 담당합니다.

- **절대 경로 정규화**: 상대 경로의 `.`, `..` 처리
- **심볼릭 링크 탈출 방지**: 허용 디렉토리 안의 symlink가 바깥을 가리키는 공격 차단
- **Glob 패턴 안전 확장**: `/**/*` 같은 패턴이 예상치 못한 경로를 포함하지 않도록
- **CWD 전용 모드 vs 전체 접근 모드**: `acceptEdits` 모드에서는 현재 디렉토리만
- **스크래치패드 지원**: Coordinator 워커의 공유 디렉토리 접근 허용 (`tengu_scratch`)
- **Windows/POSIX 경로 처리**: 크로스 플랫폼 지원

### 6.7 레이어 7: Trust Dialog

첫 실행 시 나타나는 보안 대화상자입니다. 다음을 검토하고 사용자 동의를 받습니다.

- 프로젝트 스코프의 MCP 서버 설정
- 커스텀 Hook 설정
- Bash 권한 설정
- API 키 헬퍼
- AWS/GCP 명령 접근
- OTEL 헤더

Trust Dialog를 통과하지 않으면 파일/파일시스템 작업이 차단됩니다.

### 6.8 레이어 8: Bypass Permissions Kill Switch

최후의 수단입니다. GrowthBook 서버에서 `tengu_bypass_permissions_disabled`를 활성화하면:

```typescript
// bypassPermissionsKillswitch.ts
// 사용자가 bypass 모드에 진입하는 것 자체를 차단
// 이전 모드로 강제 복귀
// 진단 메시지 표시
```

### 인사이트: 보안 모델의 설계 철학

8개 레이어를 관통하는 일관된 원칙이 있습니다.

1. **Deny by default** — 명시적 allow 규칙이 필요합니다
2. **Fail to prompting, not to deny** — 판단 불가 시 사용자에게 묻습니다 (거부가 아님)
3. **Defense in depth** — 한 레이어가 뚫려도 다음 레이어가 잡습니다
4. **Server-side kill switch** — 클라이언트 업데이트 없이 즉시 비활성화 가능
5. **Build-time elimination** — 코드가 존재하지 않으면 취약점도 존재하지 않습니다

---

## 7. 미출시 기능: 로드맵이 유출됐습니다

피처 플래그 뒤에 숨어 있던 미출시 기능들이 드러났습니다. 경쟁사에게 가장 "의도치 않은 선물"이 된 영역입니다.

### 7.1 Voice Mode — 음성으로 코딩하기

```typescript
// voice/voiceModeEnabled.ts
export function isVoiceModeEnabled(): boolean {
  return hasVoiceAuth() && isVoiceGrowthBookEnabled()
}

export function hasVoiceAuth(): boolean {
  // OAuth 전용 (Claude.ai 계정 필요)
  // API 키/Bedrock/Vertex로는 사용 불가
  // voice_stream 엔드포인트 사용
}

export function isVoiceGrowthBookEnabled(): boolean {
  return feature('VOICE_MODE')
    ? !getFeatureValue_CACHED_MAY_BE_STALE('tengu_amber_quartz_disabled', false)
    : false
}
```

몇 가지를 알 수 있습니다.

- **전용 `voice_stream` API 엔드포인트**가 존재합니다. 일반 Messages API와 별도입니다.
- **OAuth 인증 전용**입니다. API 키나 서드파티 클라우드(Bedrock/Vertex) 사용자는 접근할 수 없습니다.
- **GrowthBook 킬 스위치**(`tengu_amber_quartz_disabled`)가 걸려 있어서 서버에서 즉시 끌 수 있습니다.
- 캐싱 전략이 `_CACHED_MAY_BE_STALE`로, 실시간이 아닌 캐시된 플래그 값을 사용합니다. 음성 모드 활성화 여부를 매번 서버에 물어보지 않는다는 뜻입니다.

### 7.2 Web Browser Tool — 진짜 브라우저 자동화

```typescript
const WebBrowserTool = feature('WEB_BROWSER_TOOL')
  ? require('./tools/WebBrowserTool/WebBrowserTool.js').WebBrowserTool
  : null
```

현재 `WebFetchTool`은 정적 HTML만 가져옵니다. `WebBrowserTool`은 Bun의 `WebView` API를 활용한 실제 브라우저 자동화로 추정됩니다. JavaScript 렌더링이 필요한 SPA 페이지와 상호작용할 수 있다는 뜻입니다. Computer Use의 CLI 버전이라고 볼 수 있습니다.

### 7.3 Coordinator Mode — 멀티 에이전트 오케스트레이션 (19K)

가장 야심찬 미출시 기능입니다.

```typescript
// coordinator/coordinatorMode.ts
export function isCoordinatorMode(): boolean {
  if (feature('COORDINATOR_MODE')) {
    return isEnvTruthy(process.env.CLAUDE_CODE_COORDINATOR_MODE)
  }
  return false
}
```

**Coordinator**는 직접 코드를 쓰지 않습니다. 여러 워커 에이전트를 생성하고 작업을 분배하는 메타 오케스트레이터입니다.

- 워커는 `AgentTool`로 스폰됩니다
- 결과는 `<task-notification>` XML 블록으로 도착합니다
- `SendMessage`로 워커에게 후속 지시를 보냅니다
- `TaskStop`으로 워커를 종료합니다

허용되는 도구가 엄격히 제한됩니다.

```typescript
COORDINATOR_MODE_ALLOWED_TOOLS = new Set([
  'AgentTool',           // 워커 생성
  'TaskStop',            // 워커 종료
  'SendMessage',         // 워커와 통신
  'SyntheticOutput',     // 출력 생성
])
```

Coordinator는 **Bash를 실행할 수 없습니다**. 파일도 읽을 수 없습니다. 오직 워커를 관리하는 것만 합니다. 이건 권한 분리(principle of least privilege)의 극단적 적용입니다 — 오케스트레이터가 직접 도구를 실행할 수 있으면, 단일 장애점이 되기 때문입니다.

**공유 스크래치패드** (`tengu_scratch` GrowthBook 게이트):

워커 간 정보 공유를 위한 디렉토리입니다. 일반적인 파일시스템 권한 검사를 우회합니다 — 워커가 CWD 밖의 스크래치패드에 접근할 수 있어야 하니까요.

이건 사실상 **AI 에이전트의 마이크로서비스 아키텍처**입니다. 하나의 Claude가 여러 Claude를 오케스트레이션하는 구조이며, 각 워커는 격리된 컨텍스트에서 독립적으로 작업합니다.

### 7.4 Kairos — 선제적 어시스턴트

그리스어로 "적절한 때"를 의미하는 Kairos는 **Claude가 먼저 행동하는 모드**입니다.

| 피처 플래그 | 도구 | 기능 |
|-------------|------|------|
| `KAIROS` | `SendUserFileTool` | 사용자에게 파일을 선제적으로 전송 |
| `KAIROS` \|\| `KAIROS_PUSH_NOTIFICATION` | `PushNotificationTool` | 모바일/데스크톱 푸시 알림 |
| `KAIROS_GITHUB_WEBHOOKS` | `SubscribePRTool` | GitHub PR 웹훅 구독 |
| `PROACTIVE` \|\| `KAIROS` | `SleepTool` | 백그라운드 대기 (타이머) |
| `KAIROS_CHANNELS` | (미상) | 멀티 채널 통합 |
| `KAIROS_BRIEF` | (미상) | 체크포인트/상태 업데이트 |

시나리오를 그려보면 이렇습니다. GitHub PR에 새 리뷰 코멘트가 달리면 `SubscribePRTool`이 감지하고, CI 결과를 확인한 뒤, `PushNotificationTool`로 "PR #123에 리뷰가 달렸고, CI가 실패했습니다. 이 부분을 수정하면 될 것 같습니다"라는 알림을 보냅니다. `SleepTool`로 주기적으로 깨어나서 상태를 점검합니다.

사용자가 세션을 열지 않아도 Claude가 일하는 구조입니다. 이건 코딩 어시스턴트를 넘어 **자율 소프트웨어 엔지니어링 에이전트**로의 전환을 의미합니다.

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

팀메이트(teammate)가 생성한 크론은 해당 에이전트의 `agentId`로 태그되고, 그 에이전트의 메시지 큐로 라우팅됩니다. 장기 실행 에이전트가 **자기만의 스케줄**을 관리할 수 있다는 뜻입니다.

### 7.6 UDS Inbox — 멀티 디바이스 메시징

```typescript
const ListPeersTool = feature('UDS_INBOX')
  ? require('./tools/ListPeersTool/ListPeersTool.js').ListPeersTool
  : null
```

"UDS"는 Unified Device Stack으로 추정됩니다. 에이전트가 "피어"(다른 연결된 디바이스/인스턴스)를 조회하고, `bridge://`나 `other://` 스키마로 메시지를 라우팅할 수 있습니다.

노트북의 Claude와 데스크톱의 Claude가 서로 통신하는 세계. 아직 먼 이야기일 수 있지만, 파이프는 이미 깔려 있습니다.

### 7.7 Workflow Scripts

```typescript
const WorkflowTool = feature('WORKFLOW_SCRIPTS')
  ? (() => {
      require('./tools/WorkflowTool/bundled/index.js').initBundledWorkflows()
      return require('./tools/WorkflowTool/WorkflowTool.js').WorkflowTool
    })()
  : null
```

번들된 워크플로우(pre-built 자동화 스크립트)가 있고, 초기화 시스템이 있습니다. 서브 에이전트 내부에서는 재귀 실행이 차단됩니다 (`ALL_AGENT_DISALLOWED_TOOLS`에 포함).

### 인사이트: 미출시 기능이 말해주는 방향

이 기능들을 종합하면 Anthropic의 전략이 보입니다.

1. **CLI → 플랫폼**: 단일 터미널 도구에서 멀티 디바이스, 멀티 에이전트 플랫폼으로
2. **반응적 → 선제적**: 사용자 명령 대기에서 자율적 모니터링/알림으로
3. **텍스트 → 멀티모달**: 타이핑에서 음성, 브라우저 자동화로
4. **단일 → 오케스트레이션**: 하나의 에이전트에서 Coordinator가 관리하는 에이전트 스웜으로

### 7.8 Bridge — 원격 제어 시스템 (33+ 파일)

Bridge 시스템은 33개 이상의 파일로 구성된 대형 서브시스템입니다. Claude.ai 웹에서 로컬 머신의 Claude Code를 제어하는 "Remote Control" 기능의 백엔드입니다.

#### 연결 흐름

```
1. 사용자 → OAuth로 claude.ai 로그인
2. CCR(Claude Cloud Runtime) API → 구독 + GrowthBook 게이트 확인 (tengu_ccr_bridge)
3. 로컬 Claude Code → environment_id + environment_secret 획득
4. 인증된 터널 설정 (WebSocket)
5. 브라우저 → Bridge API → 로컬 도구 실행 → 결과 반환
```

#### 보안 티어

| 티어 | 인증 요구사항 | 용도 |
|------|-------------|------|
| Standard | OAuth 토큰 | 일반 원격 세션 |
| Elevated | OAuth + **Trusted Device Token** (JWT) | 민감한 작업이 포함된 세션 |

Elevated 티어에서는 `X-Trusted-Device-Token` 헤더로 디바이스 신뢰성을 추가 검증합니다.

```typescript
// bridge/trustedDevice.ts
// 물리적 디바이스에 바인딩된 JWT
// "이 요청이 실제로 등록된 장치에서 온 것"을 보장
```

#### 세션 격리

각 원격 세션은 **Git worktree**로 격리됩니다. 브라우저에서 두 개의 탭으로 같은 프로젝트에 접속해도, 각각 독립된 작업 디렉토리에서 돌아갑니다. 세션 간 merge conflict를 원천 봉쇄하는 접근입니다.

세션 라이프사이클은 `sessionRunner.ts`가 관리하며, JWT 서명된 `WorkSecret`으로 세션 핸드오프 보안을 유지합니다.

### 인사이트: 원격 제어의 설계 교훈

Bridge 시스템에서 주목할 점은 **세션 격리 전략**입니다. Git worktree를 세션 격리 단위로 사용한 것은 기존 인프라(Git)를 활용해 파일시스템 수준의 격리를 달성하는 실용적 선택입니다. 컨테이너나 VM 수준의 격리보다 오버헤드가 낮으면서도, 각 세션이 독립된 작업 디렉토리를 갖습니다.

2티어 인증 구조(Standard/Elevated)도 배울 점이 있습니다. 모든 원격 작업에 최고 수준의 인증을 요구하면 사용자 경험이 나빠지고, 최저 수준만 요구하면 보안이 약해집니다. 작업의 민감도에 따라 인증 수준을 달리하는 이 접근은, 보안과 편의성 사이의 균형을 잡는 현실적 방법입니다.

---

## 8. 이 아키텍처가 업계에 의미하는 것

유출된 코드를 통해 Claude Code의 내부를 본 것은, 에이전틱 AI 도구의 실전 아키텍처를 들여다볼 수 있는 드문 기회입니다. 몇 가지 핵심 교훈을 정리합니다.

### 에이전틱 시스템은 생각보다 복잡합니다

"LLM API를 호출하고 도구를 실행하면 끝"이라는 건 데모 수준의 이해입니다. 프로덕션 에이전틱 시스템에는 다음이 필요합니다.

- **컨텍스트 관리**: 4단계 압축 계층, 캐시 안정성을 위한 도구 정렬, 서버-클라이언트 간 예산 동기화
- **에러 복구**: 비용 인식 캐스케이드, 감소 수익 감지, 서킷 브레이커
- **보안**: 8개 레이어의 다층 방어, ML 기반 자동 판정, 빌드 타임 코드 제거
- **성능**: 스트리밍 중 병렬 도구 실행, 지연된 동시 프리페치, 비대칭 트랜스크립트 저장

Claude Code의 4,600개 파일은 이 복잡성의 증거입니다. OpenAI의 Codex, Cursor, Devin 같은 경쟁 제품들도 내부적으로는 비슷한 수준의 복잡성을 가지고 있을 겁니다. 다만 그 코드가 공개된 적이 없을 뿐입니다.

### 비용이 아키텍처를 결정합니다

Claude Code 아키텍처를 관통하는 하나의 원칙이 있다면, 그건 **비용 인식(cost-awareness)**입니다.

- 에러 복구는 무료 옵션부터 시도합니다
- 메시지 압축은 API 호출 없는 방법부터 적용합니다
- 감소 수익이 감지되면 토큰 낭비를 멈춥니다
- 도구 정렬은 프롬프트 캐시 히트율을 최적화합니다
- 도구 검색은 필요할 때만 로드합니다

API 호출 한 번이 곧 비용인 LLM 시스템에서, "가장 저렴한 방법부터"라는 원칙은 기술적 우아함이 아니라 경제적 생존 전략입니다. 에이전틱 시스템을 만드는 모든 팀이 결국 이 문제와 마주하게 됩니다.

### "오픈소스"의 새로운 정의가 필요합니다

Claude Code 사례는 AI 시대 "오픈소스"의 의미에 대해 질문을 던집니다. 공식 저장소에는 플러그인 인터페이스(279개 파일)만 있고, 핵심 엔진(4,600+ 파일)은 상용 라이선스 뒤에 있습니다. 이걸 "오픈소스"라고 부를 수 있을까요?

이건 Anthropic만의 문제가 아닙니다. 많은 AI 제품이 "오픈"이라는 라벨 아래 확장 인터페이스만 공개하고 핵심은 비공개로 유지합니다. 사용자와 개발자 커뮤니티가 이 구분을 인식하는 것이 중요합니다.

---

## 9. 마무리: 유출이 말해주는 것

npm source map이라는 어처구니없는 경로로 세상에 나온 Claude Code의 내부는, 단순한 API 래퍼가 아니라 4,600개 파일에 걸친 풀스택 에이전트 플랫폼이었습니다.

8개 레이어의 다층 보안, 4단계 메시지 압축 계층, 감소 수익 감지, 비용 인식 에러 복구, 캐시 안정성을 위한 도구 정렬 — 이런 세부 사항들은 "API 래퍼"에서는 나올 수 없는 프로덕션 레벨의 엔지니어링입니다.

미출시 기능들은 Anthropic의 방향을 보여줍니다. 음성 모드, 웹 브라우저 자동화, 멀티 에이전트 Coordinator, 선제적 Kairos 모드. "코딩 어시스턴트"에서 "자율 소프트웨어 엔지니어링 플랫폼"으로의 전환이 이미 코드 레벨에서 진행 중입니다.

아이러니가 있습니다. 빌드 타임 데드코드 제거라는 세련된 보안 메커니즘을 구현해놓고, 같은 빌드 파이프라인에서 source map을 지우는 걸 누락했습니다. 8겹의 양파 껍질로 사용자의 파일 시스템을 보호하면서, 자기 소스코드는 npm에 올려버린 것입니다.

이 분석에서 소개한 패턴들 — 비용 인식 에러 복구, 다단계 컨텍스트 압축, 빌드 타임 피처 게이트, 감소 수익 감지 — 은 Claude Code에만 유효한 것이 아닙니다. 에이전틱 AI 시스템을 구축하는 누구에게나 참고가 될 수 있습니다. 특히 "가장 저렴한 복구부터 시도한다"는 원칙은, API 비용이 핵심 변수인 모든 LLM 애플리케이션에 적용됩니다.

어쩌면 이게 소프트웨어 엔지니어링의 본질인지도 모릅니다. 가장 정교한 시스템도 가장 단순한 실수에 무너집니다. Anthropic의 엔지니어들이 구축한 것은 분명 인상적인 에이전틱 아키텍처입니다. 다만 다음번에는 `.npmignore`에 `*.map`을 추가하는 걸 잊지 않기를 바랍니다.

---

*이 글은 유출된 소스코드의 기술적 분석이며, 소스코드의 재배포나 사용을 권장하지 않습니다. 모든 코드의 저작권은 Anthropic PBC에 있습니다.*
