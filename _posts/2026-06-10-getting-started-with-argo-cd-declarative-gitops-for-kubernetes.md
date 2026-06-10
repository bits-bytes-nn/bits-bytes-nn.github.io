---
layout: post
title: "Getting Started with Argo CD: Declarative GitOps for Kubernetes"
date: 2026-06-10 19:53:31
author: "bits-bytes-nn"
categories: ["Tech Guides"]
tags: []
cover: /assets/images/tech-guides.jpg
use_math: true
---

## GitOps란 무엇이며 왜 Argo CD인가

Kubernetes 위에서 애플리케이션을 운영하다 보면 곧 한 가지 근본적인 질문에 부딪히게 됩니다. "지금 클러스터에 실제로 떠 있는 상태가, 우리가 의도했던 상태와 정말 일치하는가?" 수십 개의 `Deployment`, `Service`, `ConfigMap`이 여러 환경에 흩어져 있고, 여러 사람이 `kubectl apply`로 직접 변경을 가하는 상황이라면 이 질문에 자신 있게 답하기는 어렵습니다. GitOps는 바로 이 문제를 정면으로 다루는 운영 방식이며, Argo CD는 그 GitOps 패턴을 Kubernetes에서 구현한 도구입니다. 공식 정의에 따르면 **Argo CD는 Kubernetes를 위한 선언적(declarative) GitOps 지속적 배포(continuous delivery) 도구**입니다.

### 선언적이라는 것, 그리고 Git을 진실의 원천으로 삼는다는 것

GitOps의 핵심 철학은 Argo CD 공식 문서가 "왜 Argo CD인가"를 설명하면서 제시하는 두 문장에 압축되어 있습니다.

- 애플리케이션의 정의, 구성, 환경은 **선언적이어야 하고 버전 관리되어야 한다.**
- 애플리케이션의 배포와 생명주기 관리는 **자동화되고, 감사 가능하며(auditable), 이해하기 쉬워야 한다.**

이 두 원칙을 풀어보면 GitOps의 작동 방식이 자연스럽게 드러납니다. "선언적"이라는 말은 *어떻게* 배포할지를 명령형 스크립트로 나열하는 대신, 시스템이 도달해야 할 *최종 상태*를 기술한다는 뜻입니다. Kubernetes 자체가 이미 선언적 API를 갖고 있기 때문에 — 매니페스트에 원하는 상태를 적으면 컨트롤러가 그 상태로 수렴시킵니다 — GitOps는 이 철학을 클러스터 경계 바깥의 Git 저장소까지 확장합니다.

"버전 관리되어야 한다"는 부분이 GitOps의 두 번째 축입니다. Argo CD는 **GitOps 패턴을 따라 Git 저장소를 원하는 애플리케이션 상태를 정의하는 진실의 원천(source of truth)으로 사용**합니다. 즉, 클러스터에 무엇이 떠 있어야 하는지에 대한 단 하나의 권위 있는 기록이 Git 안에 있습니다. 누가, 언제, 무엇을 바꿨는지가 커밋 히스토리로 남기 때문에 변경은 자연스럽게 감사 가능해지고, 문제가 생기면 Git 저장소에 커밋된 과거의 어떤 구성으로든 롤백할 수 있습니다.

이 매니페스트는 한 가지 형식에만 묶이지 않습니다. Argo CD는 Kubernetes 매니페스트를 여러 방식으로 지정할 수 있습니다.

- Kustomize 애플리케이션
- Helm 차트
- Jsonnet 파일
- 평범한 YAML/JSON 매니페스트 디렉터리
- 설정 관리 플러그인(config management plugin)으로 구성한 임의의 커스텀 도구

또한 배포가 추적할 Git 상의 지점도 유연하게 고를 수 있습니다. 특정 브랜치나 태그의 변경을 따라가도록 하거나, 특정 Git 커밋의 매니페스트에 고정(pin)할 수도 있습니다. 이러한 추적 전략의 세부 내용은 별도의 추적 전략 문서에서 다룹니다.

### 조정 루프: OutOfSync를 감지하고 수렴시키는 방식

GitOps를 단순한 "Git에 YAML을 두는 관행"과 구별 짓는 결정적 요소는 **지속적인 조정(reconciliation)**입니다. Argo CD는 하나의 **Kubernetes 컨트롤러로 구현**되어, 실행 중인 애플리케이션을 끊임없이 모니터링하면서 현재의 **라이브 상태(live state)**를 Git 저장소에 기술된 **목표 상태(target state)**와 비교합니다. 라이브 상태가 목표 상태에서 벗어난 배포는 `OutOfSync` 상태로 간주됩니다. Argo CD는 이 차이를 보고하고 시각화하며, 라이브 상태를 다시 목표 상태로 되돌리는 동기화(sync)를 자동 또는 수동으로 수행할 수 있는 수단을 제공합니다.

직관적으로 이 과정을 다음과 같은 끝나지 않는 루프로 생각할 수 있습니다.

```text
반복:
  목표 상태  ← Git 저장소의 매니페스트(특정 revision)
  라이브 상태 ← 클러스터에 실제로 떠 있는 리소스

  if 라이브 상태 == 목표 상태:
      Synced      # 클러스터가 Git이 말하는 그대로다
  else:
      OutOfSync   # 차이를 보고/시각화
      # 정책에 따라 자동 또는 수동으로 sync 수행
```

여기서 두 가지 동작이 구분된다는 점이 중요합니다. Git에서 최신 코드를 가져와 라이브 상태와 비교하며 *무엇이 다른지* 알아내는 과정(refresh)과, 실제로 클러스터에 변경을 적용해 애플리케이션을 목표 상태로 *이동시키는* 과정(sync)은 서로 다릅니다. Git의 목표 상태에 가한 어떤 수정이든 지정된 대상 환경에 자동으로 적용되고 반영되도록 만들 수 있습니다. 이 덕분에 배포는 "누군가 터미널에서 명령을 입력하는 사건"이 아니라 "Git에 머지하면 시스템이 알아서 수렴시키는 흐름"이 됩니다.

![Argo CD 아키텍처 개요](https://argo-cd.readthedocs.io/en/stable/assets/argocd_architecture.png)

위 다이어그램은 Argo CD가 Git 저장소를 입력으로 받아 클러스터의 실제 상태와 비교·조정하는 전체 구조를 나타냅니다. 그림에 등장하는 API Server, Repository Server, Application Controller 같은 개별 구성 요소의 역할과 상호작용은 아키텍처 개요 문서에서 자세히 분석합니다. 지금 단계에서 기억할 점은, Argo CD의 중심에 "Git의 목표 상태와 클러스터의 라이브 상태를 끊임없이 맞춰가는 컨트롤러"가 있다는 사실 하나입니다.

### 명령형 배포 방식과 GitOps 방식의 대비

GitOps가 가져오는 변화를 분명히 하기 위해, 전통적인 명령형 배포 흐름과 Argo CD 기반 GitOps 흐름을 비교하면 다음과 같습니다. (아래 표의 좌측 열은 GitOps가 해결하려는 일반적인 운영상의 불편을, 우측 열은 Argo CD 문서가 명시하는 기능과 원칙을 정리한 것입니다.)

| 관점 | 명령형/수동 배포 | Argo CD GitOps |
| --- | --- | --- |
| 진실의 원천 | 클러스터의 현재 상태(사람마다 기억이 다름) | Git 저장소가 원하는 상태를 정의하는 단일 원천 |
| 변경 방식 | 사람이 직접 `kubectl apply` 등 명령 실행 | Git 커밋을 목표 상태로 삼아 자동/수동 sync |
| 드리프트(drift) 감지 | 별도 도구 없이는 알기 어려움 | 구성 드리프트를 자동 감지하고 `OutOfSync`로 시각화 |
| 감사·추적성 | 누가 언제 무엇을 했는지 추적이 어려움 | Git 히스토리 + 애플리케이션 이벤트·API 호출 감사 추적 |
| 롤백 | 수작업 복구, 재현이 어려움 | Git에 커밋된 임의의 구성으로 롤백/롤-애니웨어 |
| 다중 환경/클러스터 | 환경마다 절차가 갈라지기 쉬움 | 여러 클러스터로의 배포를 관리·배포 가능 |

### Argo CD가 GitOps 위에 더해 주는 것들

순수한 GitOps 원칙만으로도 운영 모델은 크게 개선되지만, Argo CD는 그 위에 실무에서 필요한 기능들을 얹어 줍니다. 공식 문서가 열거하는 주요 기능은 다음과 같습니다.

- 지정한 대상 환경으로의 **애플리케이션 자동 배포**
- Kustomize, Helm, Jsonnet, 평문 YAML 등 **여러 설정 관리/템플릿 도구 지원**
- **여러 클러스터**로의 관리 및 배포 능력
- OIDC, OAuth2, LDAP, SAML 2.0, GitHub, GitLab, Microsoft, LinkedIn 등과의 **SSO 통합**
- 인가를 위한 **멀티 테넌시 및 RBAC 정책**
- Git 저장소에 커밋된 임의의 애플리케이션 구성으로의 **롤백/롤-애니웨어**
- 애플리케이션 리소스의 **헬스 상태 분석**
- **구성 드리프트의 자동 감지 및 시각화**
- 애플리케이션의 목표 상태로의 **자동 또는 수동 동기화**
- 애플리케이션 활동을 실시간으로 보여 주는 **웹 UI**, 자동화와 CI 통합을 위한 **CLI**
- GitHub, BitBucket, GitLab **웹훅 통합**과 자동화를 위한 **액세스 토큰**
- blue/green, canary 같은 복잡한 롤아웃을 지원하기 위한 **PreSync, Sync, PostSync 훅**
- 애플리케이션 이벤트 및 API 호출에 대한 **감사 추적**과 **Prometheus 메트릭**
- Git 안에서 Helm 파라미터를 덮어쓰는 **파라미터 오버라이드**

이 목록을 관통하는 일관된 의도가 있습니다. "원하는 상태는 선언적으로 Git에 두고, 그 상태로의 수렴은 자동화·감사 가능·이해 가능하게 만든다"는 앞의 두 원칙을, 실제 운영 환경에서 쓸 수 있도록 가시성(웹 UI, 메트릭, 헬스 분석)과 통제(RBAC, 멀티 테넌시, 동기화 훅)로 보강한 것입니다.

정리하자면, GitOps는 "원하는 상태를 버전 관리되는 선언으로 기술하고, 시스템이 그것을 자동으로 수렴시킨다"는 운영 모델이며, Argo CD는 그 모델을 Kubernetes 컨트롤러로 구현해 라이브 상태와 목표 상태의 차이를 끊임없이 감지·시각화·조정해 주는 도구입니다. 이 정신적 모형 — Git이라는 목표 상태와 클러스터라는 라이브 상태, 그리고 둘을 잇는 조정 루프 — 을 머릿속에 두고 나면, 이어지는 아키텍처 분석과 핵심 개념, 설치 방식 선택이 모두 하나의 일관된 그림 안에서 자연스럽게 연결됩니다.

## Argo CD 아키텍처 심층 분석

앞에서 Argo CD를 "Git의 목표 상태와 클러스터의 라이브 상태를 끊임없이 맞춰가는 컨트롤러"라는 하나의 정신적 모형으로 정리했습니다. 그러나 실제 Argo CD는 단일 프로세스가 아니라 **역할이 분리된 여러 구성 요소의 협업**으로 동작합니다. 이 컴포넌트 기반 설계를 이해하는 것은 단순히 호기심을 채우는 일이 아닙니다. 어떤 컴포넌트가 Git을 읽고, 어떤 컴포넌트가 클러스터에 쓰며, 어떤 컴포넌트가 사용자 요청을 받는지를 분명히 알아야 — 설치 방식(Multi-Tenant vs. Core)의 차이가 왜 생기는지, 장애가 났을 때 어디를 봐야 하는지, 자격 증명이 어디에 저장되는지를 정확히 추론할 수 있기 때문입니다.

공식 아키텍처 문서는 Argo CD의 핵심을 세 개의 컴포넌트로 분해합니다. **API Server**, **Repository Server**, 그리고 **Application Controller**입니다. 각각이 무엇을 입력으로 받고 무엇을 책임지는지를 먼저 살펴본 뒤, 이들이 하나의 sync 동작에서 어떻게 맞물려 돌아가는지 흐름으로 엮어 보겠습니다.

### 세 핵심 컴포넌트의 역할 분담

#### API Server — 외부 세계와 만나는 관문

API Server는 **gRPC/REST 서버**로, Web UI, CLI, 그리고 CI/CD 시스템이 소비하는 API를 노출합니다. 즉 사람이나 자동화 도구가 Argo CD와 상호작용하는 모든 경로는 이 컴포넌트를 통과합니다. 공식 문서가 명시하는 API Server의 책임은 다음과 같습니다.

- 애플리케이션 관리 및 상태 보고(application management and status reporting)
- 애플리케이션 작업(sync, rollback, 사용자 정의 action 등)의 호출(invoking)
- 저장소 및 클러스터 자격 증명 관리 — 이 자격 증명은 **Kubernetes 시크릿으로 저장**됩니다
- 인증 및 외부 자격 증명 공급자(identity provider)로의 인증 위임
- RBAC 정책 시행(enforcement)
- Git 웹훅 이벤트를 위한 리스너/포워더(listener/forwarder)

여기서 중요한 통찰이 하나 드러납니다. SSO 통합, RBAC, 멀티 테넌시 같은 "사람 중심"의 기능들은 본질적으로 API Server에 묶여 있습니다. 따라서 이 컴포넌트를 포함하지 않는 설치 형태에서는 그러한 기능들이 함께 빠지게 되는데, 이 점이 뒤에서 다루는 Core 모드의 성격을 이해하는 열쇠가 됩니다.

#### Repository Server — Git에서 매니페스트를 만들어 내는 공장

Repository Server는 외부에 노출되지 않는 **내부 서비스(internal service)**로, 애플리케이션 매니페스트를 담고 있는 Git 저장소의 **로컬 캐시**를 유지합니다. 이 컴포넌트의 단 하나의 본질적 책임은 "주어진 입력으로부터 최종 Kubernetes 매니페스트를 생성해 반환하는 것"입니다. 그 입력은 다음과 같습니다.

- 저장소 URL(repository URL)
- revision (commit, tag, branch)
- 애플리케이션 경로(application path)
- 템플릿 도구별 설정: 파라미터, Helm의 `values.yaml` 등

이 정의를 곱씹어 보면 첫 번째 섹션에서 언급한 "Kustomize, Helm, Jsonnet, 평문 YAML 같은 여러 도구 지원"이 구체적으로 어디에서 실현되는지 알 수 있습니다. 바로 Repository Server입니다. 사용자가 어떤 템플릿 도구를 쓰든, 그 도구를 실행해 원시 매니페스트를 산출하는 단계가 이 컴포넌트로 집중되어 있습니다. 즉 "목표 상태(target state)"라는 추상적 개념은, Repository Server가 `(저장소 URL, revision, 경로, 파라미터)`라는 입력을 받아 구체적인 Kubernetes 매니페스트 집합으로 **렌더링**해 낸 결과물로 물질화됩니다.

#### Application Controller — 비교하고 수렴시키는 심장

Application Controller는 **Kubernetes 컨트롤러**로, 실행 중인 애플리케이션을 지속적으로 모니터링하며 현재의 라이브 상태를 저장소에 명시된 목표 상태와 비교합니다(이 조정 루프 자체의 의미는 앞 섹션에서 다뤘습니다). Application Controller가 추가로 담당하는 구체적 책임은 두 가지입니다.

- `OutOfSync` 상태를 **감지**하고, 선택적으로 교정 조치(corrective action)를 취합니다.
- 생명주기 이벤트에 대한 사용자 정의 훅 — **PreSync, Sync, PostSync** — 을 호출(invoke)합니다.

첫 번째 섹션에서 blue/green·canary 같은 복잡한 롤아웃을 위한 PreSync/Sync/PostSync 훅을 기능 목록으로 언급했는데, 그 훅을 실제로 실행시키는 주체가 바로 이 Application Controller입니다.

### 컴포넌트 책임 한눈에 비교하기

세 컴포넌트의 성격 차이를 한 표로 정리하면 다음과 같습니다. 핵심은 "누가 외부 트래픽을 받는가", "누가 Git을 읽는가", "누가 클러스터에 쓰는가"라는 세 축이 깔끔하게 분리되어 있다는 점입니다.

| 컴포넌트 | 유형 | 주된 입력 | 주된 책임 | 외부 노출 |
| --- | --- | --- | --- | --- |
| API Server | gRPC/REST 서버 | Web UI·CLI·CI/CD의 API 호출, Git 웹훅 이벤트 | 애플리케이션 관리·상태 보고, sync/rollback/action 호출, 저장소·클러스터 자격 증명 관리, 인증·인증 위임, RBAC 시행 | 노출됨(UI·CLI 진입점) |
| Repository Server | 내부 서비스 | 저장소 URL, revision, 경로, 템플릿 파라미터/values | Git 저장소의 로컬 캐시 유지, Kubernetes 매니페스트 생성·반환 | 내부 전용 |
| Application Controller | Kubernetes 컨트롤러 | 목표 상태 매니페스트, 라이브 상태 | 라이브 vs 목표 상태 비교, `OutOfSync` 감지·교정, PreSync/Sync/PostSync 훅 호출 | 내부 전용 |

이 분리는 그 자체로 설계 의도를 말해 줍니다. **사용자 인터페이스·인증(API Server)**, **매니페스트 생성(Repository Server)**, **조정·적용(Application Controller)**이 서로 독립된 책임을 가지므로, 필요에 따라 일부만 떼어 내거나(예: UI·API가 필요 없는 경량 설치) 지원되는 컴포넌트를 다중 복제(replica)로 키울 수 있습니다.

### 하나의 Sync가 컴포넌트를 가로지르는 흐름

각 컴포넌트의 역할을 따로 외우는 것보다, 실제 동작 하나를 따라가며 협업을 보는 편이 이해가 깊어집니다. 첫 섹션의 Getting Started 맥락에서 등장하는 명령을 예로 들어 보겠습니다.

```bash
# 사용자가 CLI로 sync(배포)를 요청
argocd app sync guestbook
```

이 한 줄이 실행될 때 내부에서 일어나는 협업을, 위에서 정리한 책임 분담에 근거해 단계로 풀면 다음과 같습니다.

```text
1) CLI 요청 → API Server
   - argocd CLI가 gRPC/REST로 sync 작업을 호출
   - API Server가 인증·RBAC를 시행한 뒤 sync 작업을 트리거

2) 목표 상태 생성: Application Controller ↔ Repository Server
   - Application Controller가 Repository Server에게 매니페스트를 요청
   - 입력: 저장소 URL, revision(commit/tag/branch), 경로, 템플릿 파라미터/values
   - Repository Server가 로컬 캐시된 Git에서 Kubernetes 매니페스트를 생성·반환  → 이것이 "목표 상태"

3) 비교: Application Controller
   - 클러스터의 라이브 상태와 방금 생성된 목표 상태를 비교
   - 차이가 있으면 OutOfSync로 판단

4) 적용·수렴: Application Controller
   - 목표 상태로 수렴시키기 위해 클러스터에 변경을 적용 (kubectl apply)
   - 필요 시 PreSync / Sync / PostSync 훅을 호출
```

이 흐름을 보면, 첫 섹션에서 개념적으로만 구분했던 두 동작 — Git의 최신 코드를 가져와 무엇이 다른지 알아내는 비교 과정과, 실제 클러스터에 변경을 적용하는 과정 — 이 각각 어떤 컴포넌트의 손을 거치는지가 분명해집니다. "무엇이 다른가"를 알기 위한 목표 상태는 **Repository Server**가 만들어 내고, 그 비교와 실제 적용은 **Application Controller**가 수행하며, 사용자의 요청 접수·인증·권한 검사는 **API Server**가 담당합니다.

Git 웹훅의 경로도 같은 그림 안에서 해석됩니다. API Server가 "Git 웹훅 이벤트를 위한 리스너/포워더" 역할을 한다는 것은, 저장소에 푸시가 일어났을 때 그 이벤트를 받아 내부로 전달하는 진입점이 API Server라는 의미입니다.

### Redis 캐시와 자격 증명 저장에 대한 보충

세 핵심 컴포넌트 외에 실무에서 반드시 알아야 할 두 가지 사실이 있습니다.

첫째, **Redis**입니다. Argo CD Core 문서에서 밝히듯, Application Controller는 Redis 없이도 동작할 수 있지만 권장되지는 않습니다. Application Controller는 Redis를 중요한 **캐싱 메커니즘**으로 사용하여 Kube API와 Git에 가해지는 부하를 줄이기 때문입니다. 이런 이유로 Core 설치 방식에서도 Redis는 함께 포함됩니다.

둘째, **자격 증명의 저장 위치**입니다. 앞의 책임 표에서 보았듯 저장소와 클러스터의 자격 증명은 API Server가 관리하며, 이는 **Kubernetes 시크릿으로 저장**됩니다. 외부 클러스터를 등록하는 동작이 결국 클러스터 자격 증명을 등록하는 일이라는 점을, 이 아키텍처적 사실에서 미리 짐작할 수 있습니다.

### 이 분해가 이후 내용과 연결되는 지점

이렇게 컴포넌트 단위로 책임을 나눠 두면, 뒤에서 다룰 여러 주제가 같은 틀 위에서 자연스럽게 설명됩니다. 설치 옵션을 비교하는 부분과 Core 모드를 다루는 부분은 본질적으로 "이 세 컴포넌트 중 무엇을 포함하고 무엇을 빼는가"의 문제이며, 멀티 테넌시·RBAC·SSO가 어느 설치에 따라오고 빠지는지는 그것들이 API Server에 묶여 있다는 사실에서 곧장 따라 나옵니다. 또한 여러 클러스터로의 배포나 외부 클러스터 등록은 API Server가 관리하는 클러스터 자격 증명(Kubernetes 시크릿)이라는 토대 위에서 이루어집니다. 이어지는 부분에서는 이 아키텍처를 떠받치는 Application, Target/Live state, Sync, Refresh 같은 핵심 개념들을 하나씩 정밀하게 다집니다.

## 핵심 개념 완전 정복

앞에서 GitOps의 운영 모델과 Argo CD의 컴포넌트 구조를 살펴보았다면, 이제 그 두 그림을 잇는 **어휘(vocabulary)**를 정밀하게 다질 차례입니다. Argo CD 문서가 별도로 "Core Concepts" 페이지를 두고 있는 데는 이유가 있습니다. `argocd app get`이 출력하는 `Sync Status`, `Health Status` 같은 필드, UI에 표시되는 색상과 라벨, 그리고 뒤에서 다룰 Sync 정책 설정은 모두 이 핵심 개념들을 정확히 가리키는 이름으로 구성되어 있기 때문입니다. 용어를 어림짐작으로 쓰면 "Synced인데 왜 Healthy가 아니지?" 같은 흔한 혼란에 빠지기 쉽습니다. 이 절의 목표는 각 개념을 정의하고, 서로 어떻게 구별되며, 실제 출력에서 어디에 나타나는지를 한 번에 정리하는 것입니다.

문서가 전제하듯, Git·Docker·Kubernetes·지속적 배포·GitOps의 일반 개념에는 어느 정도 익숙하다고 가정하고, 여기서는 **Argo CD에 고유한(specific to Argo CD)** 개념에 집중합니다.

### 개념 사전: Argo CD 고유 용어 한눈에 정리

먼저 전체 지형도를 표로 깔아 두고, 이후에 까다로운 부분을 하나씩 깊게 파고들겠습니다. 아래 표의 정의는 공식 Core Concepts 문서를 그대로 따른 것입니다.

| 개념 | 정의 | 핵심 질문 |
| --- | --- | --- |
| **Application** | 매니페스트로 정의되는 Kubernetes 리소스들의 그룹. 이것은 하나의 Custom Resource Definition(CRD)입니다. | "무엇을 하나의 배포 단위로 묶는가?" |
| **Application source type** | 애플리케이션을 빌드하는 데 사용되는 도구(Tool)가 무엇인가. | "이 매니페스트는 어떤 도구로 만들어지는가?" |
| **Target state** | 애플리케이션이 도달해야 할 원하는 상태. Git 저장소의 파일들로 표현됩니다. | "Git은 무엇이 떠 있어야 한다고 말하는가?" |
| **Live state** | 그 애플리케이션의 실제 상태. 어떤 Pod 등이 배포되어 있는가. | "지금 클러스터에 실제로 무엇이 떠 있는가?" |
| **Sync status** | Live state가 Target state와 일치하는지 여부. 배포된 애플리케이션이 Git이 말하는 것과 같은가? | "현재 상태가 Git과 일치하는가?" |
| **Sync** | 애플리케이션을 Target state로 이동시키는 과정. 예컨대 변경을 Kubernetes 클러스터에 적용하는 것. | "어떻게 일치시키는가?" |
| **Sync operation status** | 하나의 sync가 성공했는지 여부. | "방금의 동기화 시도가 성공했는가?" |
| **Refresh** | Git의 최신 코드를 Live state와 비교해 무엇이 다른지 알아내는 것. | "지금 차이가 있는가? (적용은 하지 않음)" |
| **Health** | 애플리케이션의 건강 상태. 올바르게 실행되고 있는가? 요청을 처리할 수 있는가? | "떠 있는 것이 제대로 동작하는가?" |
| **Tool / Configuration management tool** | 파일 디렉터리로부터 매니페스트를 만들어 내는 도구. 예: Kustomize. (둘은 같은 의미입니다.) | "매니페스트 생성기는 무엇인가?" |
| **Configuration management plugin** | 커스텀 도구. | "표준 도구가 아닌 임의의 도구를 쓰려면?" |

이 표는 외워야 할 항목 모음이 아니라, 이후 모든 설명이 매달리는 골격입니다. 특히 가운데 다섯 줄 — Target state, Live state, Sync status, Sync, Refresh — 이 서로 어떻게 맞물리는지가 Argo CD를 이해하는 핵심이므로, 이제 그 부분을 깊게 들여다봅니다.

### Application: 배포의 기본 단위이자 하나의 CRD

가장 먼저 분명히 해 둘 것은, **Application은 단순한 추상적 개념이 아니라 실재하는 Kubernetes 객체**라는 사실입니다. 정의에 따르면 Application은 "매니페스트로 정의되는 Kubernetes 리소스들의 그룹"이며 동시에 "하나의 CRD"입니다. 이 두 문장이 합쳐지면 중요한 결론이 나옵니다. Argo CD에게 "이 Git 경로의 매니페스트들을 저 클러스터의 저 네임스페이스에 배포하라"고 지시하는 일 자체가, 클러스터에 `Application`이라는 커스텀 리소스 하나를 만드는 일과 같다는 것입니다.

앞에서 다룬 Getting Started 예시에서 `guestbook` 애플리케이션을 생성하는 명령은 사실 이 `Application` 객체를 만드는 것이었습니다.

```bash
argocd app create guestbook \
  --repo https://github.com/argoproj/argocd-example-apps.git \
  --path guestbook \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace default
```

이 명령에 들어간 인자들이 곧 표의 다른 개념들과 정확히 대응됩니다. `--repo`와 `--path`는 **Target state**가 어디에 정의되어 있는지를 가리키고(아키텍처 분석에서 본 Repository Server의 입력인 "저장소 URL, 경로"가 바로 이것입니다), `--dest-server`와 `--dest-namespace`는 그 상태가 실현될 목적지, 즉 **Live state**가 존재하게 될 곳을 지정합니다. Application이 CRD라는 점이 중요한 이유는, GitOps의 "선언적"이라는 원칙이 배포 대상뿐 아니라 배포 정의 자체에도 적용될 수 있기 때문입니다. Application을 명령형 CLI 대신 YAML로 선언해 Git에 둘 수 있다는 가능성이 여기서 열리는데, 그 선언적 설정 방식은 이 가이드의 뒷부분에서 다룹니다.

### Target state와 Live state: 비교의 두 항

조정 루프의 직관 자체는 앞에서 다루었으므로, 여기서는 두 "상태"의 출처와 성질 차이에 집중합니다.

- **Target state(목표 상태)**는 "Git 저장소의 파일들로 표현되는 원하는 상태"입니다. 즉 그 출처가 전적으로 Git이며, 사람이 의도적으로 커밋한 결과입니다. 정적이고 버전 관리되는 선언입니다.
- **Live state(라이브 상태)**는 "실제로 배포되어 있는 것 — 어떤 Pod 등이 떠 있는가"입니다. 그 출처는 클러스터의 현재 모습이며, 스케일링·장애·수동 변경 등으로 끊임없이 변할 수 있는 동적인 사실입니다.

이 둘은 비교의 대상이 되는 좌변과 우변입니다. 한쪽은 "있어야 할 것"(Git), 다른 한쪽은 "있는 것"(클러스터)입니다. 그리고 이 두 항을 비교해 나온 결과가 다음 개념인 Sync status입니다.

### Sync status, Sync, 그리고 Sync operation status — 가장 헷갈리는 세 가지

초보자가 가장 자주 혼동하는 지점이 바로 "sync"라는 단어가 세 가지 다른 것을 가리킨다는 사실입니다. 문서는 이를 세 개의 별도 개념으로 분리해 정의하고 있으며, 이 구분을 또렷이 잡아 두면 UI와 CLI 출력을 정확히 읽을 수 있습니다.

**Sync status**는 *상태(state)*입니다. "Live state가 Target state와 일치하는가?"라는 질문에 대한 답으로, 앞 절들에서 본 `Synced`/`OutOfSync`라는 값을 가집니다. 이것은 어떤 행동의 결과가 아니라, 어느 시점의 일치 여부를 나타내는 명사적 상태입니다.

**Sync**는 *과정(process)*입니다. "애플리케이션을 Target state로 이동시키는 과정, 예컨대 Kubernetes 클러스터에 변경을 적용하는 것"입니다. 즉 `OutOfSync`였던 것을 `Synced`로 만들기 위해 실제로 손을 대는 동작입니다. Getting Started에서 본 `argocd app sync guestbook`이 바로 이 과정을 트리거하며, 문서가 명시하듯 이 명령은 "저장소에서 매니페스트를 가져와 그 매니페스트에 대해 `kubectl apply`를 수행"합니다.

**Sync operation status**는 *그 과정의 성패(success/failure)*입니다. "하나의 sync가 성공했는가?"에 대한 답입니다. 여기서 미묘하지만 결정적인 통찰이 나옵니다. Sync status와 sync operation status는 같은 것이 아닙니다. 하나의 sync 동작이 **성공적으로 완료(sync operation status = succeeded)**되었더라도, 그 직후 누군가가 클러스터를 수동으로 바꾸거나 Git에 새 커밋이 들어오면 sync status는 다시 `OutOfSync`가 될 수 있습니다. 한쪽은 "지금 Git과 같은가?"라는 *지속적 비교 결과*이고, 다른 쪽은 "방금 시도한 적용 작업이 끝까지 잘 됐는가?"라는 *일회성 작업의 결말*입니다.

이 세 가지의 관계를 의사 흐름으로 정리하면 다음과 같습니다.

```text
# Sync status: 비교가 낳는 지속적 상태
Sync status = (Live state == Target state) ? Synced : OutOfSync

# Sync: OutOfSync를 해소하기 위한 일회성 행동
sync() {
    Git에서 매니페스트 가져오기
    kubectl apply 수행          # 클러스터를 Target state로 이동
    return 작업 결과            # → Sync operation status (succeeded / failed)
}

# 주의: sync가 succeeded여도, 이후 드리프트가 생기면
#       Sync status는 다시 OutOfSync가 될 수 있다.
```

### Refresh와 Sync는 다르다 — 읽기 대 쓰기

앞선 절에서 두 동작의 구분을 개념적으로 짚었으니, 여기서는 정의에 근거해 못을 박습니다. **Refresh**는 "Git의 최신 코드를 Live state와 비교해 무엇이 다른지 알아내는 것"입니다. 핵심은 refresh가 **아무것도 적용하지 않는다**는 점입니다. 그저 최신 Target state를 다시 계산해 Live state와 견주어 sync status를 갱신할 뿐입니다. 비유하자면 refresh는 "장부와 창고 재고를 대조해 보는 것"이고, sync는 "대조 결과에 따라 실제로 물건을 채워 넣거나 빼는 것"입니다.

| | Refresh | Sync |
| --- | --- | --- |
| 본질 | 비교(읽기) | 적용(쓰기) |
| 클러스터 변경 | 없음 | 있음 (`kubectl apply`) |
| 결과로 갱신되는 것 | Sync status (`Synced`/`OutOfSync`) | Live state, 그리고 Sync operation status |
| 비유 | 장부와 재고 대조 | 재고를 장부에 맞춰 채우거나 빼기 |

이 구분이 실무에서 왜 중요한가 하면, "지금 Git과 다른지 확인만 하고 싶을 때"와 "실제로 클러스터를 바꾸고 싶을 때"가 명확히 다른 작업이기 때문입니다. 드리프트를 감지·시각화하는 단계까지는 refresh의 영역이고, 그 차이를 실제로 해소하는 단계가 sync의 영역입니다.

### Sync status와 Health는 직교한다

마지막으로, 가장 실수가 잦은 개념 쌍입니다. **Health**는 "애플리케이션의 건강 상태 — 올바르게 실행되고 있는가? 요청을 처리할 수 있는가?"를 뜻합니다. 이것은 sync status와 묻는 질문 자체가 다릅니다.

- **Sync status**: 떠 있는 것이 *Git이 말하는 것과 같은가?* (일치 여부)
- **Health**: 떠 있는 것이 *제대로 동작하는가?* (작동 여부)

이 둘은 서로 독립적인 축, 즉 직교(orthogonal)합니다. 네 가지 조합이 모두 의미를 가집니다.

- `Synced` + `Healthy`: Git대로 배포되었고 정상 동작 중 — 이상적 상태.
- `Synced` + 비정상(예: Pod가 크래시): Git이 말하는 그대로 배포는 되었지만, 그 매니페스트 자체에 문제가 있어 애플리케이션이 제대로 못 돈다. **Git을 고쳐야** 한다.
- `OutOfSync` + `Healthy`: 지금 떠 있는 것은 잘 돌지만 Git과는 다르다(예: 새 버전이 Git에 머지되었으나 아직 sync 전).
- `OutOfSync` + 비정상: 차이도 있고 동작도 문제다.

Getting Started에서 보았던 `argocd app get guestbook`의 출력이 바로 이 두 축을 나란히 보여 줍니다.

```text
$ argocd app get guestbook
Name:               guestbook
Server:             https://kubernetes.default.svc
Namespace:          default
URL:                https://10.97.164.88/applications/guestbook
Repo:               https://github.com/argoproj/argocd-example-apps.git
Target:
Path:               guestbook
SyncPolicy:         <none>
SyncStatus:         OutOfSync from (1ff8a67)
HealthStatus:       Missing

GROUP  KIND        NAMESPACE  NAME          STATUS     HEALTH
apps   Deployment  default    guestbook-ui  OutOfSync  Missing
       Service     default    guestbook-ui  OutOfSync  Missing
```

여기서 `SyncStatus: OutOfSync`는 "아직 배포되지 않아 Live state가 Target state와 다르다"는 비교 결과이고, `HealthStatus: Missing`은 건강 축의 평가입니다. 즉 한 줄의 출력 안에 sync status와 health라는 두 개의 독립된 진단이 동시에 담겨 있는 것입니다. 문서는 이 초기 상태를 "애플리케이션이 아직 배포되지 않았고 어떤 Kubernetes 리소스도 생성되지 않았기 때문에 `OutOfSync`"라고 설명합니다. `1ff8a67`이라는 짧은 해시는 Argo CD가 비교에 사용한 Git revision을 가리키며, 이는 Target state가 항상 특정 커밋에 근거해 계산된다는 사실을 보여 줍니다.

### Tool, Configuration management tool, Configuration management plugin

표에서 정리했듯, **Tool**과 **Configuration management tool**은 같은 의미로, "파일 디렉터리로부터 매니페스트를 만들어 내는 도구"를 가리킵니다. Kustomize가 대표적인 예입니다. 그리고 **Application source type**은 특정 Application이 그중 어떤 도구로 빌드되는지를 나타내는 분류입니다. 첫 절에서 열거한 Kustomize, Helm, Jsonnet, 평문 YAML 같은 선택지가 모두 source type에 해당하며, 이들이 실제로 매니페스트를 산출하는 단계가 아키텍처 분석에서 본 Repository Server에 집중된다는 점을 이제 더 분명히 연결할 수 있습니다.

**Configuration management plugin**은 이 표준 도구들로 충분하지 않을 때를 위한 탈출구로, 한마디로 "커스텀 도구(a custom tool)"입니다. 표준 도구로 표현되지 않는 임의의 방식으로 매니페스트를 생성해야 할 때 사용하는 확장 지점이며, 지원되는 설정 관리 도구들과 함께 이 가이드의 뒷부분에서 더 구체적으로 다룹니다.

이렇게 핵심 어휘를 정렬해 두면, 이어지는 설치 옵션 비교, 첫 Application 생성과 Sync, 그리고 Sync 정책과 추적 전략이 모두 같은 용어 체계 위에서 일관되게 읽힙니다. "Application(CRD)이 Target state와 Live state를 잇고, refresh로 차이를 알아내며, sync로 그 차이를 해소하고, sync status와 health라는 두 축으로 결과를 진단한다" — 이 한 문장이 이후 모든 실습의 해석 틀이 됩니다.
</none>

## 설치 옵션 비교: Multi-Tenant vs. Core vs. HA

지금까지 다룬 아키텍처 — API Server, Repository Server, Application Controller, 그리고 캐싱용 Redis — 를 떠올리면, 설치 방식을 고른다는 것이 결국 **"이 컴포넌트들 중 무엇을 포함하고, 그것들을 어떤 복제 수준으로 배치할 것인가"**의 문제임을 알 수 있습니다. 공식 설치 문서는 이 선택지를 크게 두 부류로 정리합니다. **Argo CD에는 multi-tenant와 core, 두 종류의 설치가 있습니다.** 그리고 multi-tenant 안에서 다시 고가용성(HA) 여부에 따라 매니페스트가 갈립니다. 정리하면 실무에서 마주치는 갈래는 multi-tenant(non-HA), multi-tenant(HA), 그리고 core 세 가지입니다.

이 절의 목표는 세 갈래의 성격 차이를 분명히 하고, 어떤 매니페스트가 어떤 상황을 겨냥하는지를 한눈에 비교하는 것입니다. Core 모드의 사용법 자체와 Kustomize를 활용한 커스텀 설치는 뒤의 해당 절에서 더 깊게 다루므로, 여기서는 "무엇이 어떻게 다른가"라는 선택의 틀에 집중합니다.

### Multi-Tenant 설치: 가장 일반적인 형태

Multi-tenant 설치는 **Argo CD를 설치하는 가장 일반적인 방식**입니다. 문서가 묘사하는 전형적인 사용 맥락은 조직 안의 여러 애플리케이션 개발 팀에게 서비스를 제공하고, 그것을 플랫폼 팀이 유지·관리하는 그림입니다. 최종 사용자는 **API Server를 통해 Web UI나 `argocd` CLI로 Argo CD에 접근**하며, CLI는 `argocd login <server-host>` 명령으로 설정해야 합니다.

여기서 앞의 아키텍처 분석에서 짚어 둔 사실이 그대로 이어집니다. SSO·RBAC·멀티 테넌시 같은 "사람 중심" 기능은 API Server에 묶여 있으므로, API Server를 포함하는 multi-tenant 설치에서만 이 기능들이 온전히 활성화됩니다.

Multi-tenant에는 권한 범위가 다른 두 종류의 매니페스트가 제공됩니다. 이 둘의 차이는 **클러스터 수준 권한이 필요한가, 네임스페이스 수준 권한으로 충분한가**에 있습니다.

| 매니페스트 | 권한 범위 | 의도된 용도 | 비고 |
| --- | --- | --- | --- |
| `install.yaml` | cluster-admin 접근 권한 | Argo CD가 떠 있는 **같은 클러스터**에 애플리케이션을 배포하려는 경우의 표준 설치 | 외부 클러스터에도 자격 증명을 입력해 배포 가능 |
| `namespace-install.yaml` | 네임스페이스 수준 권한만 필요(클러스터 롤 불필요) | Argo CD가 떠 있는 클러스터에는 배포할 필요가 없고, 입력된 클러스터 자격 증명에만 의존하는 경우. 팀별로 여러 Argo CD 인스턴스를 운영하는 시나리오 | **CRD가 포함되지 않아** 별도로 설치해야 함 |

`namespace-install.yaml`에 관해 두 가지 주의점이 있습니다. 첫째, 이 매니페스트에는 Argo CD CRD가 포함되어 있지 않으므로 CRD를 따로 설치해야 합니다. 문서가 제시하는 방법은 다음과 같습니다.

```bash
kubectl apply --server-side --force-conflicts \
  -k https://github.com/argoproj/argo-cd/manifests/crds\?ref\=stable
```

둘째, 기본 제공되는 롤만으로는 같은 클러스터에 Argo CD 리소스(Application, ApplicationSet, AppProject)만 생성할 수 있으며, 실제 워크로드 배포는 외부 클러스터로 이뤄지는 GitOps 모드를 지원합니다. 이 동작을 바꾸려면 새 롤을 정의해 `argocd-application-controller` 서비스 어카운트에 바인딩하면 됩니다. 같은 클러스터에 배포해야 한다면 `argocd cluster add <context> --in-cluster --namespace <your namespace="">` 형태로 자격 증명을 제공할 수 있습니다.

또한 두 매니페스트 모두 `ClusterRoleBinding`이 `argocd` 네임스페이스의 ServiceAccount에 묶여 있다는 점을 기억해야 합니다. 다른 네임스페이스에 설치한다면 이 바인딩을 새 네임스페이스에 맞게 조정하지 않으면 권한 오류가 발생할 수 있습니다(커스텀 네임스페이스 설치는 Kustomize를 다루는 절에서 구체적으로 살펴봅니다).

### High Availability(HA): 같은 컴포넌트, 다중 복제

HA 설치는 **프로덕션 사용에 권장되는** 형태입니다. 핵심은 새로운 컴포넌트를 추가하는 것이 아니라, **multi-tenant와 동일한 컴포넌트를 고가용성과 복원력을 위해 튜닝**한다는 점에 있습니다. 구체적으로는 지원되는 컴포넌트를 다중 복제(replica)로 띄웁니다.

HA 역시 권한 범위에 따라 두 매니페스트로 나뉘며, 각각 non-HA 버전과 대응됩니다.

| 매니페스트 | 대응되는 non-HA 매니페스트 | 차이 |
| --- | --- | --- |
| `ha/install.yaml` | `install.yaml` | 지원되는 컴포넌트를 다중 복제로 배치 |
| `ha/namespace-install.yaml` | `namespace-install.yaml` | 지원되는 컴포넌트를 다중 복제로 배치 |

반대로 non-HA 설치는 **프로덕션 용도로 권장되지 않으며**, 평가 기간 중 시연이나 테스트에 쓰이는 형태입니다. 첫 절에서 본 Quick Start의 `install.yaml`이 바로 이 non-HA 표준 설치에 해당합니다.

### Core: 경량 headless GitOps 엔진

Core 설치는 **Argo CD를 headless 모드로 구동**하는, 성격이 뚜렷이 다른 설치입니다. 이는 멀티 테넌시 기능이 필요 없고 Argo CD를 독립적으로 사용하는 클러스터 관리자에게 가장 적합합니다. **더 적은 컴포넌트를 포함하며 설정이 더 간단하다**는 것이 특징인데, 결정적으로 이 번들에는 **API Server나 UI가 포함되지 않고, 각 컴포넌트의 경량(non-HA) 버전이 설치**됩니다.

API Server가 빠진다는 사실에서 어떤 기능들이 함께 빠지는지는 아키텍처 분석에서 이미 예고된 바와 같습니다. Core 설치에서 사용할 수 없게 되는 기능군은 다음과 같습니다.

- Argo CD RBAC 모델
- Argo CD API
- Argo CD Notification Controller
- OIDC 기반 인증

반면 다음 기능들은 부분적으로 사용할 수 있습니다.

- Argo CD Web UI
- Argo CD CLI
- 멀티 테넌시(엄격히 git push 권한에 기반한 GitOps 방식)

아래 다이어그램은 Core 옵션을 선택했을 때 설치되는 컴포넌트 집합을 보여 줍니다.

![Argo CD Core 컴포넌트 구성](https://argo-cd.readthedocs.io/en/stable/assets/argocd-core-components.png)

그림의 Core 상자 안에 담긴 것이 실제로 설치되는 컴포넌트들입니다. 주목할 점은, 컴포넌트 수가 줄어들어도 핵심 GitOps 기능 — Git에서 목표 상태를 가져와 Kubernetes에 적용하는 일 — 은 그대로 동작한다는 사실입니다. 이것이 가능한 이유는 Argo CD가 처음부터 컴포넌트 기반 아키텍처로 설계되었기 때문이며, 그래서 더 미니멀한 설치가 성립합니다. 다만 이 경우에도 Redis는 함께 포함됩니다. 앞서 다뤘듯 Application Controller는 Redis 없이도 동작할 수 있지만, Kube API와 Git에 가해지는 부하를 줄이는 중요한 캐싱 메커니즘으로서 권장되기 때문입니다.

Core 모드의 구체적 사용법(예: `argocd login --core`로 로컬 API Server를 띄워 CLI를 쓰는 방식)은 Core 모드를 전담하는 절에서 자세히 다룹니다.

### 세 갈래를 한눈에 비교하기

세 설치 갈래의 성격을 핵심 축으로 정리하면 다음과 같습니다.

| 관점 | Multi-Tenant (non-HA) | Multi-Tenant (HA) | Core |
| --- | --- | --- | --- |
| 주 사용자 | 여러 개발 팀 / 플랫폼 팀 | 여러 개발 팀 / 플랫폼 팀 | 클러스터 관리자 단독 |
| API Server·Web UI | 포함 | 포함 | 미포함 (headless) |
| RBAC·OIDC 인증·Notification Controller | 사용 가능 | 사용 가능 | 사용 불가 |
| 멀티 테넌시 | 지원 | 지원 | git push 권한 기반으로만 제한적 |
| 컴포넌트 복제 | 단일(경량) | 다중 복제 | 단일(경량) |
| 대표 매니페스트 | `install.yaml`, `namespace-install.yaml` | `ha/install.yaml`, `ha/namespace-install.yaml` | `core-install.yaml` |
| 권장 용도 | 평가·시연·테스트(프로덕션 비권장) | 프로덕션 권장 | UI/CLI/멀티 테넌시가 불필요한 GitOps 엔진 |

이 표가 던지는 선택의 기준은 두 개의 질문으로 요약됩니다. 첫째, **사람 중심의 기능(UI, SSO, RBAC, 멀티 테넌시)이 필요한가?** 필요하다면 API Server를 포함하는 multi-tenant 계열을, 그렇지 않고 순수 GitOps 엔진만 원한다면 core를 고릅니다. 둘째(multi-tenant를 골랐다면), **가용성을 위해 컴포넌트를 다중 복제할 것인가?** 프로덕션이라면 HA 매니페스트가 권장됩니다. 여기에 권한 범위(같은 클러스터 배포 여부)가 `install.yaml`과 `namespace-install.yaml`의 선택을 가릅니다.
</your></context></server-host>

## Argo CD 설치하기

설치 방식의 갈래는 이미 정리했으니, 이제 실제로 클러스터에 Argo CD를 올리는 작업으로 넘어갑니다. 이 절의 초점은 "어떤 매니페스트를 고를까"가 아니라 "고른 매니페스트를 어떤 명령으로, 어떤 주의사항과 함께 적용하는가"입니다. CLI 설치와 서버 접근(포트 포워딩, 로그인 등)은 이어지는 부분에서 별도로 다루므로, 여기서는 클러스터에 컴포넌트를 띄우는 단계까지만 책임지고 진행합니다.

### 시작 전 충족해야 할 요건

설치 명령을 실행하기 전에 환경이 다음 조건을 만족하는지 확인해야 합니다. 공식 Getting Started 문서가 명시하는 요건은 다음과 같습니다.

- `kubectl` 명령줄 도구가 설치되어 있어야 합니다.
- kubeconfig 파일이 있어야 합니다(기본 위치는 `~/.kube/config`).
- CoreDNS가 필요합니다. microk8s에서는 `microk8s enable dns && microk8s stop && microk8s start`로 활성화할 수 있습니다.

이 요건들은 어렵지 않게 충족되지만, CoreDNS는 종종 간과되는 항목이므로 설치 전에 클러스터 DNS가 동작하는지 확인해 두는 편이 좋습니다.

### 표준 설치 명령

가장 일반적인 multi-tenant(non-HA) 설치는 단 두 줄로 끝납니다.

```bash
kubectl create namespace argocd
kubectl apply -n argocd --server-side --force-conflicts \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

이 명령이 하는 일은 명확합니다. 첫 줄은 모든 Argo CD 서비스와 애플리케이션 리소스가 들어갈 새로운 `argocd` 네임스페이스를 만듭니다. 둘째 줄은 `stable` 브랜치의 공식 매니페스트를 적용해 Argo CD를 설치합니다.

다른 설치 변형을 원한다면 마지막의 매니페스트 URL만 교체하면 됩니다. 예를 들어 프로덕션용 HA 설치는 `manifests/ha/install.yaml`을, headless GitOps 엔진인 Core 설치는 `manifests/core-install.yaml`을 같은 방식으로 적용합니다. 각 매니페스트의 성격 차이는 설치 옵션 비교에서 다룬 그대로입니다.

### 프로덕션에서는 버전을 고정하라

위 명령은 `stable` 브랜치를 가리킵니다. 문서는 **`v3.2.0`과 같이 고정된 버전(pinned version)을 사용할 것을 프로덕션에 권장**합니다. 이때는 URL의 `stable` 부분을 원하는 릴리스 태그로 바꾸면 됩니다.

### `--server-side --force-conflicts`가 필수인 이유

이 두 플래그는 단순한 관용구가 아니라 Argo CD 설치에서 반드시 필요한 장치이므로 그 이유를 정확히 이해해 둘 가치가 있습니다.

`--server-side` 플래그가 필요한 이유는 **CRD 크기 제한** 때문입니다. ApplicationSet 같은 일부 Argo CD CRD는 클라이언트 측 `kubectl apply`가 사용하는 262KB 어노테이션 크기 한계를 초과합니다. 서버 측 apply(server-side apply)는 `last-applied-configuration` 어노테이션을 저장하지 않으므로 해당 제한을 우회합니다.

`--force-conflicts` 플래그는 apply 작업이 **이전에 다른 도구가 관리하던 필드의 소유권을 가져올 수 있게** 합니다. 여기서 "다른 도구"란 Helm이나 과거의 `kubectl apply` 같은 것을 말합니다. 이 동작은 새로 설치하는 경우에는 안전하며, 업그레이드 시에는 필요합니다.

다만 한 가지 부작용을 명확히 알아 두어야 합니다. Argo CD 매니페스트에 정의된 필드(예: `affinity`, `env`, `probes`)에 직접 커스텀 수정을 가했다면, 그 수정은 이 apply로 **덮어쓰여집니다**. 반대로 매니페스트에 정의되지 않은 필드(예: `resources`의 limits/requests, `tolerations`)는 보존됩니다. 이 차이를 정리하면 다음과 같습니다.

| 필드 종류 | 예시 | server-side apply 시 동작 |
| --- | --- | --- |
| Argo CD 매니페스트에 **정의된** 필드 | `affinity`, `env`, `probes` | 덮어쓰여짐(custom 수정 손실) |
| 매니페스트에 **정의되지 않은** 필드 | `resources` limits/requests, `tolerations` | 보존됨 |

따라서 컴포넌트의 리소스 요청량이나 톨러레이션처럼 매니페스트가 건드리지 않는 항목은 직접 패치해 두어도 안전하지만, 매니페스트가 다루는 필드를 커스터마이징하려면 Kustomize 같은 선언적 방식으로 관리하는 편이 안전합니다.

### Redis는 비밀번호 인증을 사용한다

기본 설치에서 Redis는 **비밀번호 인증을 사용**합니다. 이 Redis 비밀번호는 Argo CD가 설치된 네임스페이스의 `argocd-redis`라는 Kubernetes 시크릿에 `auth` 키로 저장됩니다.

### 다른 네임스페이스에 설치할 때의 주의

`argocd`가 아닌 다른 네임스페이스에 설치하려는 경우 반드시 짚어야 할 함정이 있습니다. 설치 매니페스트에 포함된 `ClusterRoleBinding` 리소스가 `argocd` 네임스페이스의 ServiceAccount를 참조하고 있기 때문입니다. 다른 네임스페이스에 설치한다면 이 네임스페이스 참조를 그에 맞게 갱신해야 하며, 그러지 않으면 권한 관련 오류가 발생할 수 있습니다. 커스텀 네임스페이스 설치는 Kustomize 패치로 `ClusterRoleBinding`의 네임스페이스를 교체하는 방식으로 깔끔하게 처리할 수 있습니다.

또한 `namespace-install.yaml`을 선택했다면, 이 매니페스트에는 Argo CD CRD가 포함되어 있지 않으므로 CRD를 별도로 설치해야 한다는 점을 기억해야 합니다. CRD 매니페스트는 `manifests/crds` 디렉터리에 있으며 다음 명령으로 설치할 수 있습니다.

```bash
kubectl apply --server-side --force-conflicts -k https://github.com/argoproj/argo-cd/manifests/crds?ref=stable
```

### 설치 직후의 상태와 다음 단계

설치 명령을 적용하면 클러스터 내부에 컴포넌트가 기동되지만, 기본적으로 Argo CD는 클러스터 외부로 노출되지 않습니다. 따라서 이 시점에서는 아직 브라우저나 CLI로 접근할 수 없습니다. 또한 기본 설치는 자체 서명(self-signed) 인증서를 사용하므로, 약간의 추가 작업 없이는 접근할 수 없습니다.

여기까지가 "설치"의 영역입니다. 이렇게 컴포넌트를 띄운 뒤에 필요한 작업 — `argocd` CLI를 내려받고, 포트 포워딩이나 LoadBalancer로 API Server에 접근하며, 자동 생성된 초기 `admin` 비밀번호로 로그인하는 과정 — 은 이어지는 부분에서 단계별로 다룹니다. UI·SSO·멀티 클러스터 기능이 필요 없다면 Core 컴포넌트만 설치하는 선택지도 있으며, 이는 Core 모드를 전담하는 절에서 사용법까지 함께 살펴봅니다.

## Argo CD CLI 설치 및 서버 접근 설정

앞 절에서 클러스터에 Argo CD 컴포넌트를 띄우는 데까지 진행했습니다. 그러나 그 마지막에서 확인했듯, 기본 설치 상태의 Argo CD는 클러스터 외부로 노출되지 않으며 자체 서명 인증서를 사용하므로, 브라우저나 CLI로 곧장 접근할 수는 없습니다. 이 절에서는 그 마지막 간극을 메웁니다. `argocd` CLI를 내려받고, API Server에 도달할 경로를 확보하며, 자동 생성된 초기 비밀번호로 로그인하는 세 단계를 다룹니다.

이 모든 작업이 **API Server**를 향한다는 점을 먼저 짚어 둘 필요가 있습니다. 아키텍처 분석에서 정리했듯 Web UI·CLI·CI/CD가 소비하는 gRPC/REST API를 노출하는 컴포넌트가 API Server이며, 인증과 RBAC 시행도 여기서 일어납니다. 따라서 "CLI를 설정하고 로그인한다"는 것은 결국 "외부에서 API Server에 닿는 길을 만들고, 그 관문에서 인증을 통과한다"는 의미입니다.

### Argo CD CLI 내려받기

`argocd` CLI는 최신 릴리스 페이지(`https://github.com/argoproj/argo-cd/releases/latest`)에서 직접 내려받을 수 있습니다. 더 자세한 설치 방법은 공식 CLI 설치 문서에 정리되어 있습니다.

macOS, Linux, 그리고 WSL 환경에서는 Homebrew로 간단히 설치할 수 있습니다.

```bash
brew install argocd
```

CLI는 자동화와 CI 통합을 위한 진입점이기도 하므로(첫 절의 기능 목록에서 언급한 "CLI for automation and CI integration"), 이후의 거의 모든 실습은 이 도구를 통해 진행됩니다.

### API Server에 접근하는 세 가지 방법

기본적으로 Argo CD는 클러스터 외부로 노출되지 않으므로, 브라우저나 CLI에서 접근하려면 다음 세 가지 방법 중 하나를 선택해야 합니다.

| 방법 | 핵심 동작 | 적합한 상황 |
| --- | --- | --- |
| Service Type LoadBalancer | `argocd-server` 서비스 타입을 `LoadBalancer`로 변경 | 클라우드 환경에서 외부 IP로 노출하려는 경우 |
| Ingress | Ingress를 통해 Argo CD 노출 | Ingress 컨트롤러를 이미 운영 중인 경우 |
| Port Forwarding | `kubectl port-forward`로 로컬 포트와 연결 | 서비스를 노출하지 않고 빠르게 접근하려는 경우 |

#### Service Type Load Balancer

`argocd-server` 서비스의 타입을 `LoadBalancer`로 바꾸면 클라우드 제공자가 외부 IP를 할당합니다.

```bash
kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "LoadBalancer"}}'
```

잠시 기다리면 클라우드 제공자가 서비스에 외부 IP를 부여합니다. 할당된 IP는 다음 명령으로 확인할 수 있습니다.

```bash
kubectl get svc argocd-server -n argocd \
  -o=jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

#### Ingress

Ingress를 통한 노출은 별도의 Ingress 구성 문서에서 다루는 방식대로 설정합니다. 이미 Ingress 컨트롤러를 운영하는 환경이라면 자연스러운 선택지입니다.

#### Port Forwarding

서비스를 외부로 노출하지 않고도 API Server에 닿고 싶다면 `kubectl port-forward`가 가장 간편합니다.

```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

이렇게 하면 `https://localhost:8080`으로 API Server에 접근할 수 있습니다.

#### 자체 서명 인증서에 대한 대응

세 방법 중 무엇을 택하든, 기본 설치는 자체 서명 인증서를 사용한다는 점을 다시 떠올려야 합니다. 이를 다루는 방법은 다음 세 가지 중 하나입니다.

- 인증서를 정식으로 구성하고, 클라이언트 OS가 그 인증서를 신뢰하도록 설정합니다(TLS 구성 문서 참조).
- 클라이언트 OS가 자체 서명 인증서를 신뢰하도록 설정합니다.
- 이 가이드의 모든 Argo CD CLI 작업에 `--insecure` 플래그를 사용합니다.

### CLI로 로그인하기

API Server에 닿을 경로가 마련되었다면, 이제 인증을 통과할 차례입니다. `admin` 계정의 초기 비밀번호는 자동 생성되어, Argo CD가 설치된 네임스페이스의 `argocd-initial-admin-secret`이라는 시크릿의 `password` 필드에 평문으로 저장됩니다. 이 비밀번호는 다음 명령으로 조회할 수 있습니다.

```bash
argocd admin initial-password -n argocd
```

조회한 비밀번호와 `admin` 사용자명으로, Argo CD의 IP 또는 호스트명을 향해 로그인합니다.

```bash
argocd login <argocd_server>
```

여기서 `<argocd_server>`에는 위에서 확보한 접근 경로(예: LoadBalancer가 할당한 IP, 또는 포트 포워딩 시 `localhost:8080`)를 넣습니다.

로그인이 끝나면 초기 비밀번호를 변경합니다.

```bash
argocd account update-password
```

비밀번호를 바꾼 뒤에는 `argocd-initial-admin-secret`을 네임스페이스에서 삭제하는 것이 권장됩니다. 이 시크릿은 초기 생성된 비밀번호를 평문으로 보관하는 것 외에 다른 용도가 없으며, 언제든 안전하게 삭제할 수 있습니다. 새 admin 비밀번호를 다시 생성해야 할 경우 Argo CD가 필요에 따라 이 시크릿을 재생성합니다.

### CLI가 API Server에 직접 닿지 못할 때

한 가지 흔히 놓치는 함정이 있습니다. CLI를 실행하는 환경은 Argo CD API Server와 통신할 수 있어야 합니다. 만약 위에서 설명한 방식으로 API Server가 직접 접근 가능하지 않다면, CLI가 포트 포워딩을 거쳐 접근하도록 지시할 수 있습니다. 방법은 두 가지입니다.

- 모든 CLI 명령에 `--port-forward-namespace argocd` 플래그를 추가합니다.
- 또는 `ARGOCD_OPTS` 환경 변수를 설정합니다.

```bash
export ARGOCD_OPTS='--port-forward-namespace argocd'
```

### Core 모드라면 이 절을 건너뛸 수 있다

마지막으로, 설치 방식에 따라 위 과정이 통째로 생략될 수 있다는 점을 짚어 둡니다. UI·SSO·멀티 클러스터 기능이 필요 없는 경우에는 `argocd login --core`로 CLI 접근을 구성할 수 있으며, 이때는 접근 경로 확보와 로그인 단계(즉 LoadBalancer/Ingress/포트 포워딩 설정과 `argocd login <argocd_server>`, 클러스터 등록까지)를 건너뜁니다. `--core` 플래그는 CLI 및 Web UI 요청을 처리하기 위해 로컬 Argo CD API Server 프로세스를 띄우는 역할을 합니다. 이 동작의 구체적인 사용법은 Core 모드를 전담하는 절에서 자세히 다룹니다.

이 절을 통해 CLI가 설치되고 API Server에 인증된 채로 연결되었으니, 이제 실제로 Git 저장소로부터 Application을 생성하고 동기화하는 본격적인 GitOps 흐름으로 넘어갈 준비가 끝났습니다.
</argocd_server></argocd_server></argocd_server>

## 첫 번째 Application 생성 및 Sync

이제 앞서 갖춰 둔 모든 토대 — 설치된 컴포넌트, 인증된 CLI, 정렬된 핵심 어휘 — 를 실제 배포 한 번으로 꿰어 볼 차례입니다. 공식 예제는 `guestbook`이라는 애플리케이션을 사용하며, 그 매니페스트는 `https://github.com/argoproj/argocd-example-apps.git` 저장소의 `guestbook` 경로에 들어 있습니다. 이 절의 목표는 명령 자체의 의미를 다시 설명하는 것이 아니라, **"Application을 만든다"와 "실제로 배포한다"가 분리된 두 단계라는 사실을 손으로 직접 확인**하는 데 있습니다.

> 참고: 예제 애플리케이션은 AMD64 아키텍처에서만 호환될 수 있습니다. ARM64·ARMv7 같은 다른 아키텍처에서는 의존성이나 컨테이너 이미지 호환성 문제가 생길 수 있으므로, 필요하다면 아키텍처에 맞는 이미지를 확인하거나 별도로 빌드해야 합니다.

### CLI로 진행하기 전: 네임스페이스 컨텍스트 설정

CLI로 애플리케이션을 다루기 전에 짚어야 할 실무적 함정이 하나 있습니다. 설치 단계의 `kubectl apply` 명령들은 `-n argocd`를 직접 포함하고 있었지만, 이제부터 사용할 일부 명령은 그렇지 않습니다. 그래서 `kubectl`의 현재 컨텍스트 기본 네임스페이스를 `argocd`로 맞춰 두어야 합니다.

```bash
kubectl config set-context --current --namespace=argocd
```

이 한 줄을 빼먹으면 이후 명령이 엉뚱한 네임스페이스를 바라보게 됩니다. 설치 직후에 한 번 설정해 두는 것이 안전합니다.

### Application 생성은 "선언"일 뿐, 배포가 아니다

`argocd app create guestbook`로 애플리케이션을 만드는 명령과 그 인자(`--repo`, `--path`, `--dest-server`, `--dest-namespace`)의 의미는 앞서 핵심 개념을 정리하면서 다룬 그대로입니다. 여기서 주목할 점은 이 명령의 *결과*입니다. 외부 클러스터가 아닌, Argo CD가 떠 있는 같은 클러스터에 배포하는 경우 목적지 K8s API 서버 주소로는 `https://kubernetes.default.svc`를 사용합니다(외부 클러스터로 배포하려면 먼저 자격 증명을 등록해야 하며, 이는 외부 클러스터 등록을 다루는 부분에서 별도로 살펴봅니다).

생성 직후 상태를 조회해 보면, 앞서 본 것처럼 `SyncStatus`가 `OutOfSync`로, `HealthStatus`가 `Missing`으로 나옵니다. 문서가 명확히 설명하듯, 이는 "애플리케이션이 아직 배포되지 않았고 어떤 Kubernetes 리소스도 생성되지 않았기 때문"입니다. 다시 말해 `app create`는 **무엇을 어디에 배포할지를 선언하는 Application 객체를 클러스터에 등록**할 뿐, 매니페스트를 클러스터에 적용하지는 않습니다. Git의 목표 상태는 정의되었지만, 라이브 상태는 여전히 비어 있는 셈입니다. 이 간극을 메우는 것이 바로 다음 단계인 sync입니다.

### UI로 Application 생성하기

CLI 대신 Web UI에서도 동일한 애플리케이션을 만들 수 있으며, 폼을 채우는 과정에서 어떤 정보가 Application에 들어가는지가 시각적으로 드러납니다. 외부 UI에 접속해 로그인한 뒤, `+ New App` 버튼을 누르면 생성 폼이 열립니다.

![+ New App 버튼](https://argo-cd.readthedocs.io/en/stable/assets/new-app.png)

폼은 크게 세 묶음의 정보를 요구하며, 공식 가이드가 지정하는 값은 다음과 같습니다.

| 폼 구획 | 입력 항목 | 예제에서 넣는 값 |
| --- | --- | --- |
| 애플리케이션 기본 정보 | Application Name | `guestbook` |
| | Project | `default` |
| | Sync Policy | `Manual` (그대로 둠) |
| Source(소스) | Repository URL | `https://github.com/argoproj/argocd-example-apps.git` |
| | Revision | `HEAD` (그대로 둠) |
| | Path | `guestbook` |
| Destination(목적지) | Cluster URL | `https://kubernetes.default.svc` (클러스터 이름으로는 `in-cluster`) |
| | Namespace | `default` |

![애플리케이션 기본 정보 입력](https://argo-cd.readthedocs.io/en/stable/assets/app-ui-information.png)

여기서 폼이 CLI 인자와 정확히 대응한다는 점이 한눈에 보입니다. Source 구획의 Repository URL·Revision·Path는 CLI의 `--repo`·revision·`--path`와, Destination 구획의 Cluster URL·Namespace는 `--dest-server`·`--dest-namespace`와 같은 정보입니다. Revision을 `HEAD`로 둔다는 것은 브랜치의 최신 커밋을 따라간다는 의미이며, 특정 태그나 커밋으로 고정하는 추적 전략은 별도의 부분에서 다룹니다.

한 가지 처음 폼을 채울 때 망설이게 되는 항목이 **Sync Policy**입니다. 예제에서는 이를 `Manual`로 둡니다. 즉 Argo CD가 드리프트를 감지하더라도 사람이 직접 sync를 수행하기 전까지는 클러스터를 건드리지 않습니다. 이 선택이 바로 위에서 강조한 "생성 ≠ 배포"의 분리를 가능하게 합니다. 반대로 자동 동기화를 켜는 정책(그리고 그와 관련된 동작)은 Sync 정책과 추적 전략을 다루는 부분에서 본격적으로 살펴봅니다.

![저장소 연결](https://argo-cd.readthedocs.io/en/stable/assets/connect-repo.png)

![목적지 설정](https://argo-cd.readthedocs.io/en/stable/assets/destination.png)

위 정보를 모두 채운 뒤 UI 상단의 `Create` 버튼을 누르면 `guestbook` 애플리케이션이 생성됩니다.

![애플리케이션 생성](https://argo-cd.readthedocs.io/en/stable/assets/create-app.png)

CLI로 만들든 UI로 만들든 결과는 동일합니다. 클러스터에 `Application` 객체가 하나 등록되고, 아직 아무것도 배포되지 않았으므로 `OutOfSync` 상태로 표시됩니다.

### Sync로 라이브 상태를 목표 상태에 맞추기

이제 선언만 되어 있던 목표 상태를 실제 클러스터에 반영합니다. CLI에서는 다음 명령 한 줄입니다.

```bash
argocd app sync guestbook
```

이 명령이 내부적으로 하는 일을 문서는 정확히 한 문장으로 규정합니다. **저장소에서 매니페스트를 가져와, 그 매니페스트에 대해 `kubectl apply`를 수행**합니다. 앞선 아키텍처 분석에 비춰 보면, Repository Server는 `(저장소 URL, revision, 경로)`로부터 목표 상태 매니페스트를 생성해 반환하고, Application Controller는 라이브 상태와 목표 상태를 비교하며 동작합니다. sync가 완료되면 가이드의 표현대로 "guestbook 앱이 이제 실행 중이며, 그 리소스 구성 요소·로그·이벤트, 그리고 평가된 헬스 상태를 확인할 수 있는" 상태가 됩니다. 즉 비어 있던 라이브 상태에 `guestbook-ui` Deployment와 Service가 실제로 생성됩니다.

UI에서 sync하려면, Applications 페이지에서 guestbook 애플리케이션의 `Sync` 버튼을 누릅니다. 그러면 패널이 열리고, 그 안의 `Synchronize` 버튼을 누르면 동기화가 시작됩니다.

![guestbook 애플리케이션](https://argo-cd.readthedocs.io/en/stable/assets/guestbook-app.png)

sync 이후 애플리케이션을 클릭하면 리소스 트리가 펼쳐지며, 어떤 리소스가 어떤 관계로 배포되었는지를 시각적으로 확인할 수 있습니다.

![guestbook 리소스 트리](https://argo-cd.readthedocs.io/en/stable/assets/guestbook-tree.png)

### 무엇이 달라졌는가: create와 sync의 역할 정리

이 절에서 직접 손으로 확인한 두 단계의 성격 차이를 정리하면 다음과 같습니다. 이 구분이 흐릿하면, "분명히 애플리케이션을 만들었는데 왜 클러스터에는 아무것도 없지?" 같은 혼란에 빠지기 쉽습니다.

| 단계 | 명령 / 동작 | 클러스터에 미치는 영향 | 직후의 Sync 상태 |
| --- | --- | --- | --- |
| 생성 | `argocd app create …` 또는 UI `Create` | `Application` 객체만 등록(목표 상태 선언) | `OutOfSync` (리소스 미생성) |
| 동기화 | `argocd app sync …` 또는 UI `Synchronize` | 매니페스트를 `kubectl apply`로 적용(라이브 상태 생성) | 일치하면 `Synced` |

여기서 Sync Policy를 `Manual`로 두었기 때문에 두 단계가 명확히 나뉘었다는 점을 기억할 필요가 있습니다. 만약 자동 동기화를 활성화했다면, Argo CD가 라이브 상태를 목표 상태로 자동 동기화하므로 두 단계가 사용자 눈에는 한 번에 일어난 것처럼 보였을 것입니다. 이처럼 "선언과 적용을 어떻게 연결할 것인가"는 정책의 문제이며, 그 정책의 종류와 트레이드오프는 이어지는 Sync 정책과 추적 전략 부분에서 깊이 다룹니다. 지금 단계에서 확실히 체득해야 할 한 가지는, Argo CD에서 배포란 "터미널에서 명령을 입력하는 사건"이 아니라 "Git에 선언된 목표 상태를, 적용이라는 별도의 동작을 통해 라이브 상태로 수렴시키는 흐름"이라는 사실입니다.

## 외부 클러스터 등록 및 관리

지금까지의 모든 실습은 Argo CD가 떠 있는 바로 그 클러스터에 배포하는 경우를 전제했습니다. 그래서 목적지 K8s API 서버 주소로 `https://kubernetes.default.svc`를 사용했습니다. 그러나 실무에서는 하나의 Argo CD 인스턴스가 여러 클러스터로 배포하는 그림이 흔합니다(첫 절의 기능 목록에 있던 "여러 클러스터로의 관리·배포"가 바로 이것입니다). 이 절은 그 외부 클러스터를 등록하는 단계가 정확히 무엇을 하는 동작인지를 다룹니다.

핵심을 먼저 못 박아 두면, **클러스터 등록은 선택적(optional) 단계이며, 외부 클러스터에 배포할 때에만 필요합니다.** 같은 클러스터 내부로 배포한다면 이 단계 자체가 불필요합니다. 그리고 이 동작의 본질은 아키텍처 분석에서 이미 예고한 그대로입니다. 저장소·클러스터 자격 증명은 API Server가 관리하며 Kubernetes 시크릿으로 저장된다고 했는데, "클러스터를 등록한다"는 것은 곧 **대상 클러스터의 자격 증명을 Argo CD에 등록하는 일**입니다.

### 등록 절차: 컨텍스트 확인 후 추가

등록은 두 단계로 끝납니다. 먼저 현재 kubeconfig에 들어 있는 클러스터 컨텍스트의 이름을 모두 나열합니다.

```bash
kubectl config get-contexts -o name
```

이 목록에서 등록하려는 컨텍스트 이름을 골라 `argocd cluster add CONTEXTNAME`에 넘깁니다. 예를 들어 `docker-desktop` 컨텍스트라면 다음과 같습니다.

```bash
argocd cluster add docker-desktop
```

여기서 컨텍스트 이름은 단순한 라벨이 아니라, Argo CD가 그 클러스터에 접속하기 위한 접속 정보(엔드포인트와 인증 수단)를 kubeconfig에서 읽어 오는 키 역할을 합니다.

### `cluster add`가 대상 클러스터 안에서 실제로 하는 일

이 명령이 단순히 "주소를 적어 두는" 것이 아니라는 점을 이해하는 것이 이 절의 핵심입니다. `argocd cluster add`는 대상이 되는 kubectl 컨텍스트의 클러스터에 다음 두 가지를 설치·구성합니다.

- 그 클러스터의 `kube-system` 네임스페이스에 **`argocd-manager`라는 ServiceAccount**를 설치합니다.
- 그 ServiceAccount를 **admin 수준의 ClusterRole에 바인딩**합니다.

이렇게 만들어진 ServiceAccount의 토큰을 Argo CD가 사용해, 그 외부 클러스터에 대한 관리 작업 — 즉 배포(deploy)와 모니터링(monitoring) — 을 수행합니다. 이 구조 덕분에 Argo CD는 자신이 떠 있는 클러스터의 경계를 넘어, 등록된 대상 클러스터로 배포·관리 작업을 수행할 수 있습니다.

등록이 끝난 뒤에는, Application을 만들 때 목적지(`--dest-server`)로 `https://kubernetes.default.svc` 대신 그 외부 클러스터의 API 서버 주소를 지정하면 해당 클러스터로 배포가 이루어집니다.

### 권한을 좁히기: 최소 권한으로 운영하기

기본값인 admin 수준 ClusterRole이 부담스러운 환경이라면 권한을 좁힐 수 있습니다. 문서가 명시하는 조정의 여지는 다음과 같습니다.

- `argocd-manager-role`의 규칙(rules)을 수정해, **제한된 네임스페이스·그룹·종류(kind)에 대해서만** `create`, `update`, `patch`, `delete` 권한을 갖도록 만들 수 있습니다.
- 다만 `get`, `list`, `watch` 권한은 **클러스터 범위(cluster-scope)에서 반드시 필요**합니다. 이 권한이 없으면 Argo CD가 동작하지 못합니다.

이 비대칭을 표로 정리하면 다음과 같습니다.

| 권한 종류 | 동작 | 범위를 좁힐 수 있는가 |
| --- | --- | --- |
| `get`, `list`, `watch` (읽기·관찰) | 라이브 상태 파악, 드리프트 감지 | 불가 — 클러스터 범위에서 필수 |
| `create`, `update`, `patch`, `delete` (쓰기) | 매니페스트 적용·수렴 | 가능 — 특정 네임스페이스/그룹/종류로 한정 가능 |

### 같은 클러스터를 "등록"해야 하는 특수한 경우

대개 같은 클러스터 내부 배포에는 등록이 필요 없지만, 한 가지 예외가 있습니다. 설치 옵션 비교에서 다룬 `namespace-install.yaml`처럼 네임스페이스 수준 권한만으로 운영하는 경우, 같은 클러스터에 배포하려 해도 입력된 자격 증명에 의존해야 합니다. 이때는 `--in-cluster` 플래그를 사용해 자기 자신을 대상 클러스터로 등록합니다.

```bash
argocd cluster add <context> --in-cluster --namespace <your namespace="">
```

`namespace-install.yaml`은 기본 롤만으로는 같은 클러스터에 Argo CD 리소스(Applications, ApplicationSets, AppProjects)만 배포할 수 있고 실제 워크로드 배포는 외부 클러스터로 이뤄지는 GitOps 모드를 지원합니다. 위 형태는 그 제약을 우회해, 같은 클러스터(`kubernetes.default.svc`)로도 배포할 수 있게 자격 증명을 명시적으로 제공하는 방법입니다.
</your></context>

## 지원되는 Config Management 도구

Argo CD가 매니페스트를 여러 방식으로 지정할 수 있다는 점, 그리고 그 다섯 가지 선택지가 모두 source type에 해당하며 실제 렌더링은 Repository Server로 집중된다는 점은 앞서 정리했습니다. 이 절이 더하는 가치는 정의의 반복이 아니라, **"이미 알려진 다섯 가지가 실제로 어떻게 구성되며, 어느 입력으로 매니페스트를 산출하는가"**, 그리고 **"표준 도구 대신 플러그인을 언제 골라야 하는가"**입니다.

### 다섯 가지 source type이 실제로 받는 입력

아키텍처 분석에서 보았듯, Repository Server는 `(저장소 URL, revision, 경로)`에 더해 **도구별 설정(template specific settings) — 파라미터(parameters)와 Helm의 `values.yaml`** 을 입력으로 받아 최종 Kubernetes 매니페스트를 산출합니다. 즉 source type을 고른다는 것은 "어떤 렌더링 엔진을, 어떤 입력으로 돌릴 것인가"를 정하는 일입니다. 각 도구가 무엇을 입력으로 받고 어떻게 동작하는지, 그리고 더 깊은 구성은 어느 전용 문서에서 다뤄지는지를 정리하면 다음과 같습니다.

| Source type | 매니페스트 생성 방식 | 렌더링 입력(근거가 있는 범위) | 전용 문서 |
| --- | --- | --- | --- |
| Kustomize 애플리케이션 | `kustomize`로 빌드 | 경로 + 파라미터 | [user-guide/kustomize](https://argo-cd.readthedocs.io/en/stable/user-guide/kustomize/) |
| Helm 차트 | `helm`으로 차트를 템플릿 렌더링 | `values.yaml` + Git 안에서의 파라미터 오버라이드 | [user-guide/helm](https://argo-cd.readthedocs.io/en/stable/user-guide/helm/) |
| Jsonnet 파일 | `jsonnet`을 평가해 매니페스트 산출 | 파라미터 | [user-guide/jsonnet](https://argo-cd.readthedocs.io/en/stable/user-guide/jsonnet/) |
| 평문 YAML/JSON 디렉터리 | 디렉터리의 매니페스트를 그대로 사용 | (별도 렌더링 없음) | [user-guide/directory](https://argo-cd.readthedocs.io/en/stable/user-guide/directory/) |
| Config management plugin | 등록된 **커스텀 도구**가 매니페스트 생성 | 플러그인 구성에 따름 | [user-guide/plugins](https://argo-cd.readthedocs.io/en/stable/user-guide/plugins/) |

표의 마지막에서 평문 디렉터리만이 별도의 렌더링 단계 없이 매니페스트를 "있는 그대로" 사용한다는 점이 눈에 띕니다. 나머지는 모두 입력을 받아 매니페스트를 **만들어 내는** 도구이며, 이것이 핵심 개념에서 정의한 "Tool / Configuration management tool"의 본질입니다.

> **Helm의 두 레이어 구성에 주목할 것.** Helm을 source type으로 쓸 때 값은 한 곳이 아니라 두 층위에서 정해집니다. 첫째 층은 차트와 함께 Git에 커밋된 `values.yaml`이고, 둘째 층은 그 위에 Argo CD가 적용하는 **파라미터 오버라이드(parameter overrides for overriding helm parameters in Git)**입니다. 즉 공통 기본값은 `values.yaml`에 두고, 환경별로 달라지는 일부 값만 Application 정의에서 덮어쓰는 식으로 분리할 수 있습니다. 이 분리 덕분에 "기본 차트는 그대로 두되, 이 환경에서는 레플리카 수만 바꾼다"는 변경이 차트 수정 없이 선언적으로 표현됩니다. 오버라이드 자체도 Git에 기록되므로, GitOps의 "선언적·버전 관리" 원칙이 값 수준까지 유지됩니다.

### Config Management Plugin: 등록된 도구로서 동작한다

Configuration management plugin은 핵심 개념에서 이미 "커스텀 도구"로 정의했으므로 그 의미는 반복하지 않습니다. 여기서 강조할 점은 **동작 방식**입니다. 플러그인은 별도의 외부 시스템이 아니라, 위 다섯 source type 중 하나로서 Repository Server의 매니페스트 생성 파이프라인에 끼어드는 **등록된 도구**입니다. 표준 도구(Kustomize·Helm·Jsonnet)가 그러하듯, 플러그인 역시 "디렉터리(와 입력)로부터 최종 매니페스트를 만들어 반환"하는 동일한 계약을 따릅니다. 차이는 그 생성 로직이 Argo CD에 내장된 표준 엔진이 아니라 운영자가 구성한 임의의 도구라는 점뿐입니다.

선택의 기준은 단순합니다.

- **표준 도구로 표현되면 표준 도구를 쓴다.** Kustomize·Helm·Jsonnet·평문 디렉터리 중 하나로 매니페스트를 만들 수 있다면, 굳이 플러그인을 도입할 이유가 없습니다. 유지보수 부담이 적고 동작이 예측 가능합니다.
- **표준 도구로 표현되지 않을 때만 플러그인을 고른다.** 사내 전용 템플릿 시스템, 여러 도구를 엮은 전처리 파이프라인 등 표준 엔진의 입력 모델로는 담기 어려운 생성 방식이 필요할 때가 플러그인의 자리입니다.

플러그인을 실제로 설정하는 구체적 방법은 운영자 매뉴얼의 [Config Management Plugins](https://argo-cd.readthedocs.io/en/stable/operator-manual/config-management-plugins/) 문서에서, 사용 관점의 안내는 [user-guide/plugins](https://argo-cd.readthedocs.io/en/stable/user-guide/plugins/)에서 다룹니다.

정리하면, source type을 고른다는 것은 곧 Repository Server가 어떤 입력으로 어떤 렌더링 엔진을 돌릴지를 정하는 일이며, 표준 네 가지로 충분하지 않을 때의 확장 지점이 플러그인입니다. 어떤 도구를 쓰든 그 산출물은 동일하게 "목표 상태"가 되어, 이후 동기화·추적 메커니즘의 입력으로 그대로 이어집니다.

## Sync 정책과 추적 전략

앞선 실습에서 `guestbook` 애플리케이션을 만들 때 Sync Policy를 `Manual`로 두었고, 그 덕분에 "생성(선언)"과 "동기화(적용)"가 깔끔하게 두 단계로 나뉘는 것을 직접 확인했습니다. 또한 UI 폼에서 Revision을 `HEAD`로 남겨 두었던 것도 기억할 것입니다. 이 두 선택 — **언제 적용할 것인가(sync 정책)**와 **Git의 어느 지점을 따라갈 것인가(추적 전략)** — 은 서로 독립적인 축이며, Argo CD가 목표 상태를 라이브 상태로 수렴시키는 흐름을 실제로 어떻게 운영할지를 결정하는 두 다이얼입니다. 이 절에서는 그 두 축이 각각 무엇을 정하는지를 정리합니다.

### 두 가지 sync 모드: 수동과 자동

첫 절의 기능 목록에서 Argo CD는 "애플리케이션을 목표 상태로 **자동 또는 수동**으로 동기화"한다고 명시했습니다. 이 둘은 같은 조정 루프 위에서 동작하되, `OutOfSync`가 감지된 뒤 *누가 적용을 트리거하는가*가 다릅니다.

**수동(Manual) sync**는 예제에서 쓴 방식입니다. Argo CD가 드리프트를 끊임없이 감지·시각화하더라도, 실제 `kubectl apply`는 사람이 `argocd app sync`(또는 UI의 `Synchronize`)를 누르기 전까지 일어나지 않습니다. 즉 "차이를 보여 주는 일"과 "차이를 해소하는 일" 사이에 사람의 결정이 끼어듭니다. 변경을 적용하기 전에 검토 단계를 두고 싶을 때 적합합니다.

**자동(Automated) sync**는 그 결정을 시스템에 위임합니다. 첫 절에서 인용했듯, "Git 저장소의 목표 상태에 가해진 어떤 수정이든 지정된 대상 환경에 자동으로 적용·반영되도록 만들 수 있습니다." 아키텍처 분석에서 짚은 대로 Application Controller는 `OutOfSync` 상태를 감지하면 **선택적으로 교정 조치(corrective action)를 취하는데**, 자동 sync란 바로 이 교정 조치를 활성화해 두는 것에 해당합니다. 결과적으로 Git에 머지하는 것만으로 배포가 이루어지는 흐름이 됩니다.

| 관점 | 수동(Manual) sync | 자동(Automated) sync |
| --- | --- | --- |
| `OutOfSync` 감지 | 동일하게 지속적으로 감지·시각화 | 동일하게 지속적으로 감지·시각화 |
| 적용 트리거 | 사람이 `argocd app sync` / UI `Synchronize` 실행 | Application Controller가 자동으로 교정 조치 수행 |
| 사람의 검토 지점 | 적용 직전에 개입 가능 | 개입 없이 Git 머지가 곧 배포 |
| 예제에서의 설정 | UI Sync Policy `Manual`, CLI `SyncPolicy: <none>` | 별도로 자동 동기화를 켜야 함 |

자동 sync의 세부 설정은 공식 문서의 [Automated Sync Policy](https://argo-cd.readthedocs.io/en/stable/user-guide/auto_sync/) 가이드에서 구체적으로 다룹니다. 핵심만 기억하면, sync 정책은 "감지된 차이를 *언제, 누구의 손으로* 적용할 것인가"를 결정하는 다이얼입니다.

### 추적 전략: Git의 어느 지점을 목표로 삼는가

sync 정책이 "언제 적용하는가"의 문제라면, 추적 전략은 그와 직교하는 "**무엇을 목표 상태로 삼는가**"의 문제입니다. 핵심 개념에서 본 것처럼 목표 상태는 항상 특정 revision에 근거해 계산되며(`argocd app get` 출력의 `OutOfSync from (1ff8a67)`이 그 증거였습니다), 그 revision을 어떻게 지정하느냐가 추적 전략을 가릅니다.

공식 문서는 애플리케이션 배포가 따라갈 수 있는 Git 상의 지점을 세 가지로 제시합니다. **브랜치(branch)나 태그(tag)의 업데이트를 추적하거나, 특정 Git 커밋의 매니페스트 버전에 고정(pin)**할 수 있습니다. UI 폼이나 CLI의 revision 인자에 무엇을 넣느냐가 곧 이 선택입니다.

| 추적 대상 | revision 지정 예 | 동작 |
| --- | --- | --- |
| 브랜치 | `HEAD`, `main` 등 브랜치 이름 | 그 브랜치에 새 커밋이 들어오면 목표 상태가 따라 이동(움직이는 목표) |
| 태그 | 태그 이름 | 태그가 가리키는 지점을 추적 |
| 특정 커밋에 고정 | 커밋 해시(예: `1ff8a67`) | 그 커밋의 매니페스트에 고정되어 변하지 않음 |

예제에서 Revision을 `HEAD`로 둔 것은 브랜치의 최신 커밋을 따라간다는 의미였습니다. 이 경우 브랜치에 커밋이 추가될 때마다 목표 상태가 갱신되므로, 새 커밋이 머지되면(아직 sync하지 않았다면) sync status가 `OutOfSync`로 바뀝니다. 반대로 특정 커밋 해시에 고정하면 목표 상태가 그 시점에 박제되어, 의도적으로 revision을 바꾸기 전까지는 Git 쪽 변화로 인한 드리프트가 생기지 않습니다.

여기서 두 축이 어떻게 맞물리는지가 분명해집니다. 예컨대 "브랜치 추적 + 자동 sync"는 머지 즉시 배포되는 가장 자동화된 운영 모델이고, "커밋 고정 + 수동 sync"는 변화를 최대한 통제하는 보수적 모델입니다. 어느 조합이 적절한지는 환경의 위험 감수 수준에 달려 있습니다.

추적 전략의 더 자세한 내용과 배포 전략과의 결합은 공식 문서의 [Tracking and Deployment Strategies](https://argo-cd.readthedocs.io/en/stable/user-guide/tracking_strategies/) 문서에서 다룹니다.

### 두 다이얼을 함께 잡기

정리하면, Argo CD의 동기화 운영은 서로 독립적인 두 다이얼로 표현됩니다. 하나는 **sync 정책**으로, 감지된 차이를 수동으로 적용할지 자동으로 적용할지를 정합니다. 다른 하나는 **추적 전략**으로, 목표 상태를 브랜치·태그·고정 커밋 중 무엇으로부터 계산할지를 정합니다. 앞에서 손으로 확인한 "선언과 적용의 분리"는 결국 이 두 다이얼을 각각 `Manual`과 `HEAD`로 둔 특정한 조합이었을 뿐이며, 이 다이얼들을 어떻게 맞추느냐에 따라 같은 조정 루프가 전혀 다른 운영 성격을 띠게 됩니다.
</none>

## Kustomize를 활용한 Argo CD 커스텀 설치

앞선 설치 절에서 표준 `kubectl apply` 방식의 한 가지 특성이 드러났습니다. server-side apply는 매니페스트가 **정의하는** 필드를 다음 apply에서 덮어쓰므로, 그런 필드(예: `affinity`, `env`, `probes`)에 가한 직접 수정은 업그레이드 때 사라집니다. 반면 매니페스트에 명시되지 않은 필드(예: `resources` limits/requests, `tolerations`)는 보존됩니다. 따라서 이런 필드의 커스터마이징은 선언적으로 관리해야 지속됩니다. Argo CD 매니페스트를 **Kustomize로 설치**하는 방식이 바로 그 선언적 관리의 자리입니다. 공식 설치 문서는 **매니페스트를 원격 리소스(remote resource)로 포함하고 그 위에 Kustomize 패치로 추가 커스터마이징을 적용**할 것을 권장합니다.

### 베이스: 원격 매니페스트를 리소스로 참조하기

Kustomize 설치의 출발점은 공식 매니페스트를 원격 리소스로 끌어오는 최소한의 `kustomization.yaml`입니다.

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: argocd
resources:
  - https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

여기서 핵심은 `install.yaml`을 **복사해 오는 것이 아니라 참조한다**는 점입니다. 이 베이스 위에 조직 고유의 변경 사항을 패치로 겹겹이 쌓는 구조이므로, 업스트림 매니페스트와 로컬 커스터마이징이 깨끗하게 분리됩니다.

#### 버전 고정과 업그레이드의 의미

베이스 URL의 `stable`은 움직이는 참조이므로, 프로덕션에서는 고정된 릴리스 태그로 바꾸는 것이 권장됩니다. 예를 들어 다음과 같이 한 줄을 교체하면 됩니다.

```yaml
resources:
  # stable → 고정 태그로 교체 (예시)
  - https://raw.githubusercontent.com/argoproj/argo-cd/v3.2.0/manifests/install.yaml
```

이 구조에서 업그레이드는 별도의 절차가 아니라 **이 한 줄의 태그를 바꾸는 커밋 하나**가 됩니다. 즉 `v3.2.0`을 다음 릴리스 태그로 바꾸어 커밋하는 행위 자체가 곧 업그레이드 의도의 선언이며, 그 변경 이력이 Git에 남습니다. HA 설치나 Core 설치로 베이스를 바꾸고 싶다면 같은 자리에서 `manifests/ha/install.yaml`이나 `manifests/core-install.yaml`을 가리키도록 경로만 교체하면 됩니다.

### Kustomize 패치: 공식 예시의 JSON Patch

베이스가 마련되면, 그 위에 변경을 얹는 수단이 패치입니다. 공식 설치 예시는 `op`/`path`/`value`로 이루어진 연산 목록 형태의 **JSON Patch**를 사용합니다. 이는 "어느 경로에 어떤 연산(replace 등)을 가할지"를 명시적으로 지정하는 방식으로, 배열의 특정 원소를 정확히 짚어 바꿔야 할 때 명료합니다. 커스텀 네임스페이스 설치가 바로 그런 경우입니다.

### 커스텀 네임스페이스 설치: JSON Patch로 ClusterRoleBinding 교정하기

`argocd`가 아닌 다른 네임스페이스에 설치할 때는 ClusterRoleBinding이 가리키는 네임스페이스 참조를 갱신해야 합니다. 여기서는 그 교정을 Kustomize로 **선언적으로** 처리하는 방법과, 패치의 경로가 정확히 무엇을 가리키는지를 들여다봅니다. 공식 문서가 제시하는 `kustomization.yaml`은 다음과 같습니다.

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: <your-custom-namespace>   # 실제 네임스페이스명으로 교체
resources:
  - https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
patches:
  - patch: |-
      - op: replace
        path: /subjects/0/namespace
        value: <your-custom-namespace>   # 실제 네임스페이스명으로 교체
    target:
      kind: ClusterRoleBinding
```

이 구성은 두 부분이 맞물려 동작합니다.

먼저 최상단의 `namespace: <your-custom-namespace>`는 Kustomize가 리소스들을 그 네임스페이스에 배치하도록 합니다. 그러나 설치 매니페스트의 `ClusterRoleBinding`은 `argocd` 네임스페이스의 ServiceAccount에 바인딩되어 있어, 네임스페이스를 바꿀 때 ClusterRoleBinding을 새 네임스페이스에 맞게 조정하지 않으면 권한 관련 오류가 발생할 수 있습니다. 그래서 JSON Patch로 그 참조를 명시적으로 고쳐 주는 것입니다.

`patches` 블록을 한 줄씩 해석하면 다음과 같습니다.

- `op: replace`는 "기존 값을 새 값으로 바꾼다"는 연산입니다.
- `path: /subjects/0/namespace`는 대상 리소스 안에서 바꿀 위치를 가리킵니다. `subjects`는 바인딩이 권한을 부여하는 주체들의 **배열**이고, `0`은 그 배열의 **첫 번째 원소**를 의미하며, `namespace`는 그 원소의 네임스페이스 필드입니다. 즉 "첫 번째 subject의 네임스페이스"를 콕 집어 교체합니다.
- `value`에는 새 네임스페이스명을 넣습니다.
- `target.kind: ClusterRoleBinding`은 이 패치를 **`ClusterRoleBinding` 리소스에 적용**하라는 선택자입니다.

이 패치는 각 `ClusterRoleBinding`의 **첫 번째 subject만** 교정합니다. 설치 매니페스트의 `ClusterRoleBinding`은 ServiceAccount에 바인딩된 구조이므로 첫 번째 원소를 바꾸는 것으로 충분합니다. 만약 어떤 바인딩이 여러 subject를 갖고 그들이 모두 같은 네임스페이스에 있어야 한다면, `/subjects/1/namespace`처럼 인덱스를 달리한 연산을 추가로 나열해 각각을 교정해야 합니다. `target`이 `kind`만 지정하므로 패치 자체는 모든 `ClusterRoleBinding`에 걸리지만, 그 안에서 **어느 subject를 바꿀지는 경로의 인덱스가 결정**한다는 점이 핵심입니다.

이렇게 `kustomization.yaml`을 구성한 뒤에는 디렉터리를 대상으로 적용합니다.

```bash
kubectl apply --server-side --force-conflicts -k .
```

`-k` 플래그는 디렉터리의 `kustomization.yaml`을 평가해 최종 매니페스트를 렌더링한 뒤 적용하라는 의미이며, server-side apply 관련 플래그가 필요한 이유는 앞 절에서 설명한 그대로입니다.

이로써 매니페스트가 정의하는 필드의 커스터마이징, 네임스페이스 교정, 버전 관리가 모두 Git 안의 선언으로 모입니다.
</your-custom-namespace></your-custom-namespace></your-custom-namespace>

## Argo CD Core 모드: 경량 GitOps 엔진

설치 옵션을 비교하면서 Core가 어떤 컴포넌트를 빼고 무엇을 포함하는지는 이미 정리했습니다. API Server와 UI가 빠지고 각 컴포넌트의 경량(non-HA) 버전만 설치되며, RBAC·API·Notification Controller·OIDC 인증이 사라지는 대신 Redis는 캐싱을 위해 그대로 남는다는 그림이었습니다. 이 절이 더하는 것은 그 "성격"이 아니라 **"왜 이런 설치를 고르며, 컴포넌트가 줄어든 상태에서 실제로 어떻게 사용하는가"**입니다. Core 설치는 한마디로 **헤드리스(headless) 모드로 동작하는 완전한 기능의 GitOps 엔진**으로, Git 저장소로부터 목표 상태를 가져와 Kubernetes에 적용하는 일을 그대로 수행합니다.

### Core 모드가 정당화되는 사용 사례

Core를 고를지 말지는 결국 "사람 중심 기능(UI·SSO·RBAC·멀티 테넌시)이 필요한가"라는 앞선 질문으로 귀결되지만, 공식 문서는 Core 설치를 정당화하는 구체적인 사용 사례를 세 가지로 제시합니다. 이를 살펴보면 Core가 무엇을 단순화하려는 설계인지가 분명해집니다.

- **클러스터 관리자로서 Kubernetes RBAC에만 의존하고 싶다.** Argo CD 고유의 RBAC 모델을 따로 두지 않고, 이미 익숙한 Kubernetes의 권한 체계 하나로 접근 통제를 일원화하려는 경우입니다.
- **DevOps 엔지니어로서 새로운 API나 별도의 CLI에 의존해 배포를 자동화하고 싶지 않다. Kubernetes API에만 의존하고 싶다.** 배포 파이프라인이 `Application` 같은 Kubernetes 리소스를 다루는 것만으로 완결되기를 원하는 경우입니다.
- **클러스터 관리자로서 개발자에게 Argo CD UI나 CLI를 제공하고 싶지 않다.** 운영을 클러스터 관리자가 독립적으로 수행하며 사용자 인터페이스 노출 자체를 원치 않는 경우입니다.

세 사례를 관통하는 공통점은 "Argo CD를 사람이 드나드는 플랫폼이 아니라, Kubernetes 위에서 조용히 도는 조정 엔진으로 쓰겠다"는 의도입니다. 멀티 테넌시 기능이 필요 없고 Argo CD를 독립적으로 사용하는 클러스터 관리자에게 가장 적합하다는 문서의 설명이 바로 여기에서 나옵니다.

### Core 설치하기

Core 설치는 필요한 모든 리소스를 담은 단일 매니페스트(`core-install.yaml`) 하나를 적용하는 것으로 끝납니다. 공식 문서가 제시하는 예시는 버전을 환경 변수로 고정해 두고 적용하는 형태입니다.

```bash
export ARGOCD_VERSION=<원하는 Argo CD 릴리스 버전 (예: v2.7.0)>
kubectl create namespace argocd
kubectl apply -n argocd --server-side --force-conflicts \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/$ARGOCD_VERSION/manifests/core-install.yaml
```

명령의 구조는 표준 설치와 동일합니다. `argocd` 네임스페이스를 만들고, server-side apply 관련 플래그와 함께 매니페스트를 적용합니다. `--server-side --force-conflicts`가 필요한 이유는 앞서 설명한 그대로(CRD 크기 제한과 필드 소유권 이전)이며, 다른 점은 적용하는 매니페스트가 `install.yaml`이 아니라 `core-install.yaml`이라는 것뿐입니다. 즉 Core는 "다른 설치 절차"가 아니라 "다른 매니페스트 한 장"의 문제입니다.

### Core 모드에서 상호작용하는 법: GitOps와 Kubernetes 리소스

Core 설치가 끝나면 사용자는 **GitOps에 의존해** Argo CD와 상호작용하게 됩니다. 이때 다룰 수 있는 Kubernetes 리소스는 `Application`과 `ApplicationSet` CRD입니다. 핵심 개념에서 Application이 단순한 추상 개념이 아니라 실재하는 CRD라는 점을 강조했는데, Core 모드에서는 바로 그 사실이 운영의 전부가 됩니다. API Server라는 관문이 없으므로, 배포와 관리는 이 CRD들을 클러스터에 직접 생성·수정하는 일로 환원됩니다.

권한 체계도 이 점에서 단순해집니다. Core는 오직 Kubernetes RBAC에만 의존하므로, CLI를 호출하는 사용자(또는 프로세스)는 **`argocd` 네임스페이스에 접근할 수 있어야 하고, 주어진 명령을 실행하기 위해 `Application`·`ApplicationSet` 리소스에 대한 적절한 권한을 가지고 있어야** 합니다. Argo CD 고유의 RBAC 정책을 따로 설정하는 대신, 누가 어떤 명령을 실행할 수 있는지가 전적으로 Kubernetes의 권한 부여로 결정되는 것입니다.

### CLI를 Core 모드로 쓰기: 로컬 API Server를 띄우는 트릭

흥미로운 점은, API Server 컴포넌트를 설치하지 않았는데도 `argocd` CLI를 여전히 사용할 수 있다는 사실입니다. 이는 `--core` 플래그의 동작 덕분입니다. Core 모드에서 CLI를 쓰려면 `login` 서브커맨드에 `--core` 플래그를 전달합니다.

```bash
kubectl config set-context --current --namespace=argocd  # 현재 kube 컨텍스트를 argocd 네임스페이스로 변경
argocd login --core
```

여기서 일어나는 일이 Core 모드 이해의 열쇠입니다. CLI는 명령을 처리하기 위해 **로컬에 Argo CD API Server 프로세스를 띄우고(spawn)**, 그 프로세스가 명령을 처리합니다. 명령이 끝나면 이 로컬 API Server 프로세스도 함께 종료됩니다. 이 과정은 사용자에게 투명하게(transparently) 일어나며, 별도의 추가 명령이 필요하지 않습니다. 다시 말해 클러스터에 상주하는 API Server를 두는 대신, CLI가 명령을 실행하는 그 순간에만 일회성으로 API Server를 로컬에 세웠다 거두는 것입니다. 멀티 테넌트 설치에서 필요했던 LoadBalancer/Ingress/포트 포워딩 설정과 `argocd login <서버>` 단계가 Core에서 생략될 수 있는 이유가 바로 이 동작에 있습니다.

이 동작의 의미를 멀티 테넌트 모드와 나란히 정리하면 다음과 같습니다.

| 관점 | 멀티 테넌트 모드 | Core 모드 |
| --- | --- | --- |
| API Server | 클러스터에 상주하는 컴포넌트 | 컴포넌트로 설치되지 않음 |
| CLI가 닿는 대상 | 상주 중인 API Server (로그인·인증 필요) | `--core`가 띄우는 로컬 API Server 프로세스 (명령 종료 시 함께 종료) |
| 인증·권한 | API Server의 인증 + Argo CD RBAC | Kubernetes RBAC (`argocd` 네임스페이스의 `Application`·`ApplicationSet` 권한) |
| 배포 정의 | API를 통한 작업 또는 CRD | 주로 `Application`·`ApplicationSet` CRD를 직접 다룸 |

### Web UI도 로컬에서 띄울 수 있다

UI 역시 같은 방식으로 부분적으로 사용할 수 있습니다. 클러스터에 UI를 노출하는 대신, 필요할 때 로컬에서 대시보드를 실행하는 형태입니다.

```bash
argocd admin dashboard -n argocd
```

이 명령을 실행하면 Argo CD Web UI가 `http://localhost:8080`에서 제공됩니다. `--core` 로그인이 그러했듯, 이 또한 상주하는 서버 없이 로컬에서 UI 요청을 처리하는 방식이라는 점에서 Core 모드의 "필요할 때만 로컬에 세운다"는 일관된 철학을 따릅니다.

정리하면, Core 모드는 컴포넌트를 덜어 내 설치와 운영을 단순화하면서도 핵심 GitOps 기능을 온전히 유지하는 설치 형태입니다. 사람이 드나드는 관문(API Server·상주 UI)과 Argo CD 고유의 권한 모델을 걷어 내고, 그 자리를 Kubernetes 리소스(`Application`·`ApplicationSet`)와 Kubernetes RBAC로 대체합니다. CLI나 UI가 필요한 순간에는 로컬 프로세스를 일시적으로 띄워 처리하므로, "독립적으로 Argo CD를 운영하며 Kubernetes API 하나로 모든 것을 다루고 싶은" 클러스터 관리자에게 가장 잘 맞는 선택지가 됩니다.

## Feature Maturity 이해하기

지금까지 다룬 설치 방식과 핵심 기능들은 대체로 안정적인(stable) 기능을 전제로 했습니다. 그러나 Argo CD는 활발히 개발되는 프로젝트이므로, 모든 기능이 동일한 성숙도(maturity)에 도달해 있는 것은 아닙니다. 어떤 기능은 정식으로 굳어진 반면, 어떤 기능은 아직 실험적이어서 향후 릴리스에서 동작이 바뀔 수 있습니다. 이 차이를 무시한 채 최신 기능을 프로덕션에 끌어다 쓰면, Argo CD를 업그레이드하는 순간 예기치 못한 문제에 부딪힐 수 있습니다. 이 절은 Argo CD가 기능에 부여하는 **성숙도 상태(status)**가 무엇을 의미하는지, 그리고 그것을 운영 판단에 어떻게 반영해야 하는지를 정리합니다.

### 성숙도 상태가 말하는 것

Argo CD의 기능에는 안정성과 성숙도를 나타내는 상태가 표시될 수 있습니다. 공식 문서가 명시적으로 열거하는 비안정(non-stable) 상태는 **Alpha**와 **Beta** 두 가지이며, 이 둘에 속하지 않는 기능은 사실상 안정 단계로 취급됩니다. 핵심은 이 상태가 단순한 라벨이 아니라 **하위 호환성(backward compatibility) 보장 수준에 대한 약속**이라는 점입니다.

문서가 분명히 경고하는 내용은 다음과 같습니다. Alpha와 Beta 기능은 **하위 호환성을 보장하지 않으며, 향후 릴리스에서 깨지는 변경(breaking change)이 일어날 수 있습니다.** 그래서 Argo 사용자에게는 — 특히 Argo CD 업그레이드를 직접 통제하지 못하는 환경이라면 — 이러한 기능에 프로덕션에서 의존하지 말 것이 강력히 권고됩니다. 더 나아가, **Alpha 기능의 제거는 업그레이드 이후 리소스를 예측 불가능한 상태로 바꿔 놓을 수 있습니다.** 따라서 어떤 기능이 사용 중인지를 문서로 남기고, 업그레이드 전에 릴리스 노트를 검토하는 습관이 중요합니다.

이 경고가 GitOps의 운영 철학과 어떻게 맞물리는지 짚어 둘 필요가 있습니다. Argo CD는 "선언적이고 버전 관리되는 목표 상태"를 전제로 동작하는데, Alpha 기능이 제거되면서 리소스가 예측 불가능한 상태로 변한다는 것은 곧 그 목표 상태의 해석이 흔들릴 수 있다는 의미입니다. 그렇기에 어떤 기능이 어느 성숙도에 있는지를 아는 것은 단순한 참고 정보가 아니라 업그레이드 안전성을 좌우하는 실질적인 운영 정보입니다.

### 비안정 기능 목록

공식 문서는 비안정 상태에 있는 기능들을, 도입된 버전과 함께 정리해 둡니다. 아래 표는 그 내용을 그대로 옮긴 것으로, "언제 도입되었고 현재 어떤 성숙도에 있는가"를 한눈에 보여 줍니다.

| 기능 | 도입 버전 | 상태 |
| --- | --- | --- |
| AppSet Progressive Syncs | v2.6.0 | Beta |
| Proxy Extensions | v2.7.0 | Beta |
| Skip Application Reconcile | v2.7.0 | Alpha |
| AppSets in any Namespace | v2.8.0 | Beta |
| Cluster Sharding: round-robin | v2.8.0 | Alpha |
| Dynamic Cluster Distribution | v2.9.0 | Alpha |
| Cluster Sharding: consistent-hashing | v2.12.0 | Alpha |
| Service Account Impersonation | v2.13.0 | Alpha |
| Source Hydrator | v2.14.0 | Alpha |

여기서 한 가지 통찰을 얻을 수 있습니다. 도입된 지 오래된 기능이라고 해서 자동으로 안정 단계에 오르는 것은 아니라는 점입니다. 예컨대 `Skip Application Reconcile`은 v2.7.0에 도입되었지만 여전히 Alpha이며, 반대로 `AppSet Progressive Syncs`는 v2.6.0 도입 이후 Beta까지 성숙했습니다. 즉 성숙도는 시간이 아니라 안정화의 진척에 따라 부여되는 것입니다.

### 어디에서 상태가 갈리는가: CRD 속성과 설정의 단위

성숙도 표시는 기능 전체에만 붙는 것이 아니라, 그 기능을 켜는 **구체적인 설정 지점**에까지 내려가 붙습니다. 문서는 비안정 설정을 CRD별로 분류하는데, 이는 "내가 지금 만지려는 이 필드가 안정적인가"를 정확히 확인할 수 있게 해 줍니다.

핵심 개념에서 Application·AppProject·ApplicationSet 같은 CRD가 Argo CD의 선언 단위라는 점을 다뤘는데, 비안정 기능 중 일부는 바로 이 CRD들의 특정 속성을 통해 활성화됩니다.

| CRD | 기능 | 속성 | 상태 |
| --- | --- | --- | --- |
| Application | Skip Application Reconcile | `metadata.annotations[argocd.argoproj.io/skip-reconcile]` | Alpha |
| AppProject | Service Account Impersonation | `spec.destinationServiceAccounts.*` | Alpha |
| ApplicationSet | AppSet Progressive Syncs | `spec.strategy.*` | Beta |
| ApplicationSet | AppSet Progressive Syncs | `status.applicationStatus.*` | Beta |

또한 일부 기능은 CRD가 아니라 **설정용 리소스**(예: `ConfigMap/argocd-cmd-params-cm`)나 **컨트롤러 Deployment의 환경 변수**를 통해 켜집니다. 예를 들어 Proxy Extensions는 `argocd-server`의 `ARGOCD_SERVER_ENABLE_PROXY_EXTENSION` 환경 변수나 `argocd-cmd-params-cm`의 `server.enable.proxy.extension` 값으로 활성화되며 그 상태는 Beta입니다. 마찬가지로 앞서 설치 옵션을 비교하며 언급된 클러스터 샤딩 알고리즘(`controller.sharding.algorithm`의 round-robin·consistent-hashing)도 모두 Alpha 상태의 설정 값으로 분류됩니다. 즉 같은 기능이라도 그것을 켜는 환경 변수와 ConfigMap 키가 짝을 이뤄 동일한 성숙도로 표시되므로, 어느 경로로 설정하든 동일한 안정성 판단을 적용하면 됩니다.

### 실무에서의 활용

정리하면, Feature Maturity 정보는 새 기능을 도입할지 결정할 때 반드시 거쳐야 할 점검 단계입니다. 어떤 기능을 켜기 전에 그 기능 — 그리고 그것을 활성화하는 구체적 속성·환경 변수·ConfigMap 키 — 이 Alpha인지 Beta인지 확인하고, Alpha나 Beta라면 프로덕션 의존을 피하거나 최소한 사용 사실을 문서화한 뒤 업그레이드마다 릴리스 노트를 확인하는 절차를 두는 것이 안전합니다. 이렇게 성숙도를 의식한 기능 선택은, 이 가이드의 뒷부분에서 다룰 프로덕션 고려사항 및 버전 호환성 판단과도 곧장 이어집니다.

## 시작 전 사전 준비: 필수 배경 지식과 도구

지금까지 이 가이드는 설치·접근·동기화·커스텀 설치·Core 모드·기능 성숙도까지, Argo CD를 실제로 다루는 흐름을 따라왔습니다. 그런데 공식 문서는 이 모든 실습에 앞서 한 가지 전제를 깔아 둡니다. Getting Started 안내가 "이 가이드는 Argo CD가 기반으로 삼는 도구들에 대한 기초 지식을 가정한다"고 못 박고, Core Concepts 페이지 역시 "Git, Docker, Kubernetes, 지속적 배포(Continuous Delivery), GitOps의 핵심 개념에 익숙하다고 가정한다"고 명시하기 때문입니다. 이 절은 그 전제를 정면으로 다룹니다. 즉 **Argo CD를 효과적으로 쓰기 위해 미리 갖추어야 할 배경 지식과 환경**을 한자리에 정리하는 것이 목표입니다.

이 절이 다른 절들보다 뒤에 놓였다고 해서 중요도가 낮은 것은 아닙니다. 오히려 앞선 모든 설명이 암묵적으로 의존해 온 토대를 명시적으로 확인하는 점검표에 가깝습니다. 설치 환경 요건(`kubectl`, kubeconfig, CoreDNS)이나 `argocd` CLI 설치처럼 이미 앞에서 절차로 다룬 항목은 여기서 다시 설명하지 않고, 그 항목들이 *왜* 필요한 전제였는지의 맥락에서 짧게만 위치를 짚습니다. 새로 더하는 것은 **개념적 배경 지식**과 그것을 학습할 수 있는 출처입니다.

### 왜 배경 지식을 먼저 확인하는가

Argo CD 문서는 "플랫폼을 효과적으로 사용하기 전에, 그것이 기반으로 삼는 기술을 이해하는 것이 필요하다"고 말합니다. 이 권고는 단순한 의례적 문구가 아닙니다. 앞선 절들을 떠올려 보면, 핵심 개념을 다룬 부분에서 Target state·Live state·Sync 같은 Argo CD 고유 용어를 정의할 때 문서가 "Git, Docker, Kubernetes, CD, GitOps 개념에는 익숙하다고 가정한다"고 선을 그었습니다. 즉 Argo CD가 새로 가르치는 것은 *Git 위에 선언된 목표 상태를 Kubernetes 라이브 상태로 수렴시키는 메커니즘*이지, Git이나 Kubernetes 자체가 아닙니다. 그 토대가 비어 있으면, `kubectl apply`가 무엇을 하는지·매니페스트가 어떻게 생겼는지·브랜치와 커밋이 무엇을 의미하는지 같은 1차 지식이 없는 상태에서 2차 개념을 쌓는 셈이 되어, 설명이 자꾸 미끄러지게 됩니다.

### 갖추어야 할 핵심 배경 지식

공식 "Understand The Basics" 문서가 안내하는 학습 갈래를 정리하면 다음과 같습니다. 각 항목은 Argo CD의 어느 부분과 직접 맞닿는지를 함께 적어 두었습니다.

| 배경 지식 영역 | 왜 필요한가 (Argo CD와의 접점) | 공식 문서가 권하는 출처 |
| --- | --- | --- |
| 컨테이너와 Docker | 배포 대상 워크로드가 컨테이너 이미지로 패키징됨 | A Beginner-Friendly Introduction to Containers, VMs and Docker |
| Kubernetes | 라이브 상태가 존재하는 곳이자 매니페스트가 적용되는 대상 | Introduction to Kubernetes(edX), kubernetes.io의 Tutorials |
| 지속적 배포(CD)·GitOps | Argo CD의 운영 모델 그 자체 | Core Concepts가 전제하는 일반 개념 |
| Git | 목표 상태가 정의되고 버전 관리되는 진실의 원천 | Core Concepts가 전제하는 일반 개념 |

표의 위 두 행은 문서가 구체적인 온라인 강의·튜토리얼 링크까지 제시하는 항목이고, 아래 두 행은 별도 강의를 지정하기보다 "이미 익숙하다고 가정한다"는 형태로 전제되는 개념입니다. Git·CD·GitOps의 일반론은 이 가이드의 첫 부분에서 이미 운영 모델로 풀어 다루었으므로, 그 그림을 머릿속에 두고 있다면 여기서 추가로 학습할 것은 주로 컨테이너와 Kubernetes의 실무 감각입니다.

### 사용 계획에 따라 달라지는 템플릿 도구 지식

배경 지식 중 한 가지는 *모두에게 똑같이 필요한 것이 아니라 사용 계획에 따라 달라집니다*. 문서는 "애플리케이션을 어떻게 템플릿화할 계획인가에 따라" 다음 도구를 익히라고 안내합니다.

- **Kustomize** (kustomize.io)
- **Helm** (helm.sh)

이는 앞서 지원되는 Config Management 도구를 다룬 부분과 직접 이어집니다. Argo CD의 source type 중 어느 것을 고르느냐에 따라, 그 도구 자체에 대한 사전 지식이 전제됩니다. 예컨대 Helm 차트를 source type으로 쓸 계획이라면 `values.yaml`과 파라미터 구조를 이해하고 있어야 하고, Kustomize로 매니페스트를 관리할 계획이라면 그 모델을 알고 있어야 합니다. 흥미롭게도 Kustomize 지식은 단지 *배포 대상* 애플리케이션을 위해서만이 아니라, 앞서 다룬 것처럼 **Argo CD 자체를 Kustomize로 커스텀 설치**할 때에도 그대로 쓰입니다. 따라서 어떤 템플릿 도구를 사전에 익혀 둘지는 "내가 무엇을, 어떻게 배포할 것인가"라는 계획에서 역으로 결정됩니다.

### CI 도구와의 통합을 계획한다면

Argo CD의 CLI가 자동화와 CI 통합을 위한 진입점이라는 점은 앞에서 짚었습니다. 그 통합을 실제로 구성할 계획이라면, 연동할 CI 도구에 대한 사전 지식이 필요합니다. 문서는 대표적인 두 가지를 듭니다.

- **GitHub Actions** (GitHub Actions Documentation)
- **Jenkins** (Jenkins User Guide)

이 지식은 "Git에 머지하면 시스템이 수렴시킨다"는 GitOps 흐름의 *앞단*, 즉 변경을 Git에 만들어 넣고 검증하는 파이프라인을 어떻게 짤지와 관련됩니다. CI 도구 지식은 Argo CD의 동기화 메커니즘과 별개의 축으로 미리 준비해 두는 편이 좋습니다.

### 환경·도구 점검표

마지막으로, 개념 지식과 별개로 *기계 위에 실제로 갖추어져 있어야 하는 것들*을 한눈에 모읍니다. 아래 항목들의 설치·설정 절차는 이 가이드의 앞부분에서 이미 다루었으므로, 여기서는 "무엇이 왜 필요한가"의 체크리스트로만 제시합니다.

| 준비물 | 역할 | 이 가이드에서 다룬 위치 |
| --- | --- | --- |
| `kubectl` | 클러스터에 매니페스트를 적용하고 리소스를 조회하는 기본 도구 | 설치 절의 시작 전 요건 |
| kubeconfig 파일 | 어느 클러스터에 어떤 자격으로 접속할지를 담은 설정(기본 위치 `~/.kube/config`) | 설치 절의 시작 전 요건, 외부 클러스터 등록 |
| CoreDNS | 클러스터 내 DNS — Argo CD 동작의 전제(microk8s에서는 `microk8s enable dns && microk8s stop && microk8s start`로 활성화) | 설치 절의 시작 전 요건 |
| `argocd` CLI | UI 없이 Argo CD와 상호작용하고 자동화하는 진입점 | CLI 설치 및 서버 접근 설정 절 |

이 점검표에서 한 가지만 다시 강조하면, kubeconfig의 의미입니다. 앞서 외부 클러스터 등록을 다룰 때 보았듯 `argocd cluster add`는 kubeconfig의 컨텍스트로부터 접속 정보를 읽어 오고, `kubectl config get-contexts -o name`으로 그 목록을 확인했습니다. 즉 kubeconfig는 단순한 환경 파일이 아니라, "어떤 클러스터들을 다룰 수 있는가"의 출발점이므로 시작 전에 정확히 구성되어 있어야 합니다.

정리하면, Argo CD를 시작하기 전에 갖출 것은 두 갈래입니다. 하나는 **개념적 토대** — 컨테이너·Kubernetes의 실무 감각, 그리고 (계획에 따라) Kustomize·Helm·CI 도구에 대한 지식 — 이고, 다른 하나는 **환경적 토대** — `kubectl`·kubeconfig·CoreDNS·`argocd` CLI가 실제로 준비된 기계 — 입니다. 이 두 토대가 마련되어 있으면, 앞서 따라온 설치와 동기화 흐름이 비로소 미끄러짐 없이 맞물립니다. 이제 남은 것은 그 토대 위에서 *어떤 설치 방식을 고르고 프로덕션에서 무엇을 더 고려할 것인가*, 그리고 *어느 Kubernetes 버전과 호환되며 다음으로 무엇을 학습할 것인가*이며, 이어지는 부분에서 다룹니다.

## 설치 방식 선택 가이드 및 프로덕션 고려사항

지금까지 이 가이드는 세 갈래의 설치(multi-tenant non-HA, multi-tenant HA, core)와 권한 범위에 따른 매니페스트 분기, 그리고 server-side apply의 동작·기능 성숙도까지 개별 주제로 다뤄 왔습니다. 이 절의 목표는 그 흩어진 판단 기준을 **하나의 결정 절차**로 엮고, "이 환경에는 무엇을 설치해야 하는가"라는 질문에 답한 뒤, 그 선택을 프로덕션에서 안전하게 운영하기 위한 점검 항목을 **운영 사고 패턴**과 함께 정리하는 것입니다. 개별 사실의 재설명이 아니라, 이미 학습한 사실들을 "선택 → 사고 예방"이라는 흐름으로 재배치하는 것이 핵심입니다.

### 세 개의 질문으로 매니페스트 고르기

설치 매니페스트의 선택은 세 개의 순차적 질문으로 환원됩니다. 각 질문은 앞서 다룬 아키텍처적 사실(API Server에 사람 중심 기능이 묶여 있다, HA는 같은 컴포넌트를 다중 복제한다, 권한 범위가 매니페스트를 가른다) 위에 그대로 서 있습니다.

```text
Q1. 사람 중심 기능(Web UI · SSO · RBAC · 멀티 테넌시)이 필요한가?
    ├─ 아니오 → core-install.yaml
    │            ※ Core는 경량(non-HA) 단일 버전만 제공된다.
    │              HA/non-HA 선택지가 없으므로 Q2·Q3는 건너뛴다.
    └─ 예 ↓

Q2. 프로덕션이며 가용성·복원력이 필요한가?
    ├─ 예  → ha/ 접두사 매니페스트 (다중 복제)
    └─ 아니오(평가·시연·테스트) → 비-ha 매니페스트 ↓

Q3. Argo CD가 떠 있는 "같은 클러스터"에 워크로드를 배포하는가?
    ├─ 예  → install.yaml          (cluster-admin 범위)
    └─ 아니오(외부 클러스터 전용, 팀별 다중 인스턴스)
           → namespace-install.yaml (네임스페이스 범위)
             ※ 이 매니페스트에는 Argo CD CRD가 포함되지 않는다.
               manifests/crds 를 별도로 설치해야 동작한다.
```

Q1에서 "아니오"로 빠지는 경로에는 한 가지 자주 놓치는 사실이 있습니다. Core 설치는 각 컴포넌트의 **경량(non-HA) 버전**만 담은 단일 매니페스트(`core-install.yaml`)로 제공되며, API Server나 UI를 포함하지 않습니다. multi-tenant처럼 `ha/` 변형이 따로 존재하지 않으므로, Core를 고른 순간 가용성 튜닝의 선택지(Q2)도, 권한 범위의 분기(Q3)도 사라집니다. 즉 Q1의 "아니오"는 나머지 모든 질문을 닫아 버리는 결정입니다.

Q3에서 `namespace-install.yaml`로 빠지는 경로 역시 부수 조건을 동반합니다. 이 매니페스트는 클러스터 롤 없이 네임스페이스 권한만으로 동작하는 대신 Argo CD CRD를 포함하지 않으므로, `manifests/crds`를 먼저 설치하지 않으면 `Application` 같은 리소스를 인식조차 하지 못합니다. 결정 트리에서 이 가지를 택했다면 CRD 별도 설치가 함께 따라온다는 점을 짝지어 기억해야 합니다.

세 질문의 결과를 한 표로 응축하면 다음과 같습니다.

| 조건 | 선택 매니페스트 | 권한 범위 | 동반 제약 |
| --- | --- | --- | --- |
| 사람 중심 기능 불필요 | `core-install.yaml` | (경량 단일 버전) | HA 변형 없음, API Server·UI 부재 |
| 사람 중심 기능 필요 + 프로덕션 + 같은 클러스터 배포 | `ha/install.yaml` | cluster-admin | 다중 복제 리소스 운영 |
| 사람 중심 기능 필요 + 프로덕션 + 외부 클러스터 전용 | `ha/namespace-install.yaml` | 네임스페이스 | CRD 별도 설치 |
| 사람 중심 기능 필요 + 평가·테스트 + 같은 클러스터 | `install.yaml` | cluster-admin | 프로덕션 비권장 |
| 사람 중심 기능 필요 + 평가·테스트 + 외부 클러스터 전용 | `namespace-install.yaml` | 네임스페이스 | CRD 별도 설치 |

### 프로덕션 점검 항목 — 각 항목이 어떤 사고로 이어지는가

매니페스트를 골랐다면, 그 다음은 "선택을 프로덕션에서 어떻게 유지하는가"입니다. 아래 표는 이미 앞 절들에서 *사실*로 다룬 항목들을, 이 절에서는 **누락 시 어떤 운영 사고 패턴으로 번지는가**라는 각도에서 다시 배치한 것입니다. 점검의 목적은 정의를 외우는 데 있지 않고, 각 항목이 무너졌을 때 벌어지는 구체적 연쇄를 미리 그려 두는 데 있습니다.

| 점검 항목 | 권장 조치 | 누락 시 사고 패턴 |
| --- | --- | --- |
| 버전 고정 | `stable` 대신 `v3.2.0` 같은 고정 태그 사용 | `stable`을 추적하면 매니페스트 재적용 시 의도치 않은 버전으로 점프 → 검증되지 않은 릴리스가 무계획적으로 들어옴 |
| 고가용성 | 프로덕션은 `ha/` 매니페스트로 컴포넌트 다중 복제 | 단일 복제 상태에서 컴포넌트가 죽으면 조정 루프 자체가 멈춤 → 드리프트 감지·자동 sync가 중단되어도 알아채기 어려움 |
| 커스터마이징 보존 | 매니페스트가 정의하는 필드는 Kustomize 패치로 선언 관리 | server-side apply가 정의된 필드를 덮어씀 → 업그레이드 후 `affinity` 설정 유실 → 워크로드가 의도치 않은 노드에 스케줄링 |
| 초기 admin 비밀번호 | 비밀번호 변경 후 `argocd-initial-admin-secret` 삭제 | 평문 초기 비밀번호가 네임스페이스에 잔존 → 시크릿 열람 권한자에게 admin 자격이 노출 |
| 커스텀 네임스페이스 | `ClusterRoleBinding`의 네임스페이스 참조를 새 네임스페이스로 교정 | 바인딩이 `argocd` 네임스페이스 ServiceAccount를 가리킨 채 방치 → 권한 관련 오류 발생 |
| 외부 클러스터 쓰기 권한 | `argocd-manager-role`의 `create/update/patch/delete`를 한정된 네임스페이스·그룹·종류로 한정 | admin 수준 ClusterRole을 그대로 두면 폭발 반경이 대상 클러스터 전체로 확대 (아래에서 상술) |
| 비안정 기능 의존 | Alpha/Beta 사용 시 문서화 + 업그레이드마다 릴리스 노트 검토 | Alpha 기능 제거가 리소스를 예측 불가능한 상태로 변형 → 업그레이드 직후 정의의 해석이 흔들림 |

### 권한 최소화가 프로덕션 전환에서 갖는 의미

외부 클러스터 등록을 다루면서, `get/list/watch`는 클러스터 범위에서 필수이고 `create/update/patch/delete`는 한정된 네임스페이스·그룹·종류로 좁힐 수 있다는 비대칭은 이미 정리했습니다. 여기서 굳이 다시 짚는 이유는, 그 비대칭이 **프로덕션 전환 맥락에서 폭발 반경(blast radius)의 문제로 성격이 바뀌기** 때문입니다.

`argocd cluster add`는 대상 클러스터의 kube-system 네임스페이스에 `argocd-manager` ServiceAccount를 설치하고, 이를 admin 수준 ClusterRole에 바인딩합니다. 평가용 단일 클러스터에서는 이 권한이 큰 위험이 아닙니다. 그러나 하나의 multi-tenant Argo CD 인스턴스가 **여러 팀의 프로덕션 클러스터**를 등록해 관리하는 순간, 그 인스턴스가 탈취되거나 잘못된 매니페스트를 sync하면 영향 범위가 단일 애플리케이션이 아니라 **조직 전체의 프로덕션**으로 확장됩니다. 즉 admin ClusterRole을 방치한 외부 클러스터의 수만큼 폭발 반경이 곱해집니다. 그래서 프로덕션에서는 쓰기 권한을 실제 배포가 일어나는 네임스페이스로 좁히는 작업이 "선택적 강화"가 아니라 "사고 시 피해를 국소화하는 필수 조치"가 됩니다. 읽기·관찰 권한(`get/list/watch`)은 클러스터 범위에서 반드시 필요하므로 좁힐 수 없다는 제약은 그대로이며, 좁힐 수 있는 것은 어디까지나 쓰기 권한이라는 점을 다시 명확히 해 둘 필요가 있습니다.

### 업그레이드를 Git 워크플로 안의 검토 가능한 사건으로 만들기

Kustomize 설치에서 보았듯, 베이스 URL의 태그 한 줄을 바꾸는 커밋이 곧 업그레이드 선언이 됩니다. 이 절이 더하는 것은 그 커밋을 **실패 유형별 대응이 가능한 검토 사건**으로 다루는 운영 절차입니다. 단순히 태그를 바꿔 머지하는 것만으로는 앞 표의 두 가지 사고 패턴 — 커스터마이징 유실과 Alpha 기능 제거로 인한 리소스 변형 — 을 막을 수 없기 때문입니다.

프로덕션 업그레이드에서 마주치는 대표적 실패 유형과 그 대응을 정리하면 다음과 같습니다.

| 실패 유형 | 어디에서 비롯되는가 | Git 워크플로에 끼워 넣을 대응 |
| --- | --- | --- |
| 커스텀 필드 유실 | 새 매니페스트가 `affinity`·`env`·`probes` 등을 다시 정의 | 그 필드를 Kustomize 패치로 선언해, 태그 변경 커밋과 함께 패치가 항상 재적용되도록 구성 |
| Alpha 기능 제거 | 비안정 기능이 다음 릴리스에서 깨지거나 사라짐 | 사용 중인 Alpha/Beta 기능 목록을 저장소에 문서화하고, 업그레이드 PR에서 해당 항목의 릴리스 노트를 필수 확인 |
| 미지원 Kubernetes 버전 | 대상 Argo CD 버전이 현재 클러스터 버전을 테스트 범위에 두지 않음 | 태그 변경 PR 검토 시 버전 호환성 표를 대조(다음 부분에서 다룸) |

핵심은 업그레이드를 "운영자가 어느 날 실행하는 명령"이 아니라 "리뷰 가능한 Git 변경"으로 환원하되, 그 변경에 **릴리스 노트 검토를 게이트로 결합**하는 데 있습니다. 비안정 기능에 의존하고 있다면, 어떤 기능을 어느 속성·환경 변수·ConfigMap 키로 켰는지를 저장소의 문서에 남겨 두는 것이 특히 중요합니다. 그래야 태그를 올리는 커밋을 리뷰하는 사람이 "이번 릴리스에서 우리가 쓰는 기능 중 깨지는 것이 있는가"를 릴리스 노트와 대조해 판단할 수 있습니다. 이렇게 하면 업그레이드 실패가 "사후에 발견하는 장애"가 아니라 "머지 전에 걸러지는 리뷰 코멘트"로 위치를 옮기게 됩니다.

정리하면, 설치 방식의 선택은 세 개의 질문으로 결정론적으로 좁혀지고, 그 선택을 프로덕션에서 유지하는 일은 각 점검 항목을 *사고 패턴*과 짝지어 예방하는 작업으로 환원됩니다. 남은 한 가지 축 — 선택한 Argo CD 버전이 어느 Kubernetes 버전과 호환되며, 그 위에서 다음으로 무엇을 학습할 것인가 — 은 이어지는 마지막 부분에서 다룹니다.

## Kubernetes 버전 호환성 및 다음 단계

앞선 프로덕션 고려사항에서 업그레이드를 검토 가능한 Git 변경으로 다루며, 그 PR을 검토할 때 "대상 Argo CD 버전이 현재 클러스터 버전을 테스트 범위에 두는가"를 대조하라는 항목을 남겨 두었습니다. 이 마지막 절은 그 대조표의 실체를 정확히 제시하고, 이 가이드를 마친 뒤 어디로 학습을 이어 갈지를 정리합니다. 즉 "선택한 Argo CD 버전이 어느 Kubernetes 버전 위에서 검증되었는가"라는 호환성 축을 닫고, 그다음 깊이 있는 주제들로 가는 길을 안내하는 것이 목표입니다.

### 테스트된 Kubernetes 버전: 두 가지 다른 약속

먼저 용어를 또렷이 구분해 둘 필요가 있습니다. Argo CD 공식 설치 문서는 버전과 관련해 **지원되는 버전(supported versions)**과 **테스트된 버전(tested versions)**이라는 서로 다른 두 개념을 제시합니다. 이 둘은 같지 않습니다.

- **지원되는 버전**은 Argo CD의 버전 지원 정책에 관한 것이며, 그 세부 내용은 공식 [Release Process and Cadence](https://argo-cd.readthedocs.io/en/stable/developer-guide/release-process-and-cadence/) 문서가 규정합니다.
- **테스트된 버전**은 각 Argo CD 버전이 *어떤 Kubernetes 버전들과 함께 테스트되었는가*를 보여 주는 표입니다. 이것이 업그레이드 PR을 검토할 때 직접 대조해야 하는 호환성 축입니다.

문서가 명시하는 테스트된 버전 조합은 다음과 같습니다.

| Argo CD 버전 | 테스트된 Kubernetes 버전 |
| --- | --- |
| 3.4 | v1.35, v1.34, v1.33, v1.32 |
| 3.3 | v1.35, v1.34, v1.33, v1.32 |
| 3.2 | v1.34, v1.33, v1.32, v1.31 |

이 표를 읽는 방법이 곧 운영 판단의 핵심입니다. 각 Argo CD 마이너 버전은 가장 최근의 Kubernetes 버전 몇 개를 묶어 테스트하며, Argo CD가 새 버전으로 올라갈수록 그 묶음 전체가 더 최신 쪽으로 이동합니다. 예컨대 클러스터가 v1.31에 머물러 있다면, 표에서 v1.31을 테스트 범위에 포함하는 것은 Argo CD 3.2뿐입니다. 3.3 이상으로 올리는 커밋은 "테스트된 조합"을 벗어나게 되므로, 그런 PR은 클러스터를 먼저 올리거나 업그레이드 계획을 재검토해야 한다는 신호입니다. 반대로 클러스터를 v1.35로 올린 환경이라면 3.2는 더 이상 테스트된 조합이 아니며, 3.3 또는 3.4가 검증된 짝입니다.

여기서 "버전 고정" 원칙과 이 표가 어떻게 맞물리는지가 분명해집니다. 공식 문서는 프로덕션에서 `v3.2.0` 같은 고정 버전을 사용할 것을 권장하는데, 이렇게 고정 태그로 박아 두면 *그 고정된 버전이 현재 클러스터 Kubernetes 버전과 검증된 조합 안에 있는지*를 명시적으로 통제할 수 있습니다. 태그를 올리는 업그레이드 커밋과 클러스터의 Kubernetes 버전은 항상 이 표 위에서 함께 고려되어야 하는 한 쌍입니다.

> Argo CD를 Helm으로 설치하는 선택지도 있으나, 그 Helm 차트는 현재 커뮤니티가 유지보수하는(community maintained) 것으로, [argo-helm/charts/argo-cd](https://github.com/argoproj/argo-helm/tree/main/charts/argo-cd)에서 제공됩니다. 이 가이드가 다룬 `install.yaml`·`core-install.yaml` 등의 공식 매니페스트와는 출처가 다르다는 점을 염두에 두는 것이 좋습니다.

### 다음 단계: 어디로 학습을 이어 갈 것인가

이 가이드는 GitOps의 운영 모델에서 출발해 아키텍처, 핵심 개념, 설치·접근·동기화, 외부 클러스터 등록, 설정 관리 도구, 커스텀 설치, Core 모드, 기능 성숙도, 그리고 사전 준비와 프로덕션 판단까지를 하나의 일관된 그림으로 엮었습니다. 그 토대 위에서 더 깊이 들어갈 만한 주제들을 정리하면 다음과 같습니다.

- **자동 동기화와 추적 전략의 심화** — sync 정책과 추적 전략은 각각 [Automated Sync Policy](https://argo-cd.readthedocs.io/en/stable/user-guide/auto_sync/)와 [Tracking and Deployment Strategies](https://argo-cd.readthedocs.io/en/stable/user-guide/tracking_strategies/)에서 구체적인 설정 수준으로 다뤄집니다.
- **선언적 설정(Declarative Setup)** — Application이 하나의 CRD라는 사실에서 열린 가능성, 즉 Application 자체를 YAML로 선언해 Git에 두는 방식은 [Declarative Setup](https://argo-cd.readthedocs.io/en/stable/operator-manual/declarative-setup/) 문서가 본격적으로 안내합니다.
- **여러 클러스터·여러 네임스페이스로의 확장** — 다수의 Application을 생성기로 펼치는 [ApplicationSet](https://argo-cd.readthedocs.io/en/stable/operator-manual/applicationset/)과 [Applications in any namespace](https://argo-cd.readthedocs.io/en/stable/operator-manual/app-any-namespace/)는, 단일 Application 수준을 넘어선 운영으로 가는 자연스러운 다음 단계입니다.
- **사람 중심 기능의 구성** — multi-tenant 설치에서 활성화되는 기능들, 즉 [User Management](https://argo-cd.readthedocs.io/en/stable/operator-manual/user-management/)와 [RBAC Configuration](https://argo-cd.readthedocs.io/en/stable/operator-manual/rbac/), 그리고 외부 노출을 위한 [Ingress Configuration](https://argo-cd.readthedocs.io/en/stable/operator-manual/ingress/)과 [TLS configuration](https://argo-cd.readthedocs.io/en/stable/operator-manual/tls/)이 여기에 속합니다.
- **고가용성과 업그레이드 운영** — 프로덕션 권장 형태인 HA의 세부 튜닝은 [High Availability Overview](https://argo-cd.readthedocs.io/en/stable/operator-manual/high_availability/)에서, 버전 간 변경점은 [Upgrading Overview](https://argo-cd.readthedocs.io/en/stable/operator-manual/upgrading/overview/)와 각 버전별 업그레이드 문서에서 확인할 수 있습니다.
- **운영 가시성과 복구** — [Metrics](https://argo-cd.readthedocs.io/en/stable/operator-manual/metrics/), [Notifications](https://argo-cd.readthedocs.io/en/stable/operator-manual/notifications/), [Disaster Recovery](https://argo-cd.readthedocs.io/en/stable/operator-manual/disaster_recovery/)는 배포 이후의 관찰·알림·복구를 다룹니다.

이 가이드를 관통한 정신적 모형 — Git의 목표 상태와 클러스터의 라이브 상태, 그리고 둘을 잇는 조정 루프 — 은 위의 모든 심화 주제에서도 그대로 유효합니다. 어떤 기능을 더 배우든, 결국 "선언적으로 기술된 목표 상태를, 감사 가능하고 자동화된 방식으로 라이브 상태에 수렴시킨다"는 한 문장으로 되돌아온다는 점을 기억한다면, 이어지는 학습은 새로운 개념의 나열이 아니라 같은 그림을 더 정밀하게 그려 넣는 작업이 될 것입니다.
- - -

### Sources
* <https://argo-cd.readthedocs.io/en/stable/>
* <https://argo-cd.readthedocs.io/en/stable/understand_the_basics/>
* <https://argo-cd.readthedocs.io/en/stable/core_concepts/>
* <https://argo-cd.readthedocs.io/en/stable/getting_started/>
* <https://argo-cd.readthedocs.io/en/stable/operator-manual/>
* <https://argo-cd.readthedocs.io/en/stable/operator-manual/architecture/>
* <https://argo-cd.readthedocs.io/en/stable/operator-manual/installation/>
* <https://argo-cd.readthedocs.io/en/stable/operator-manual/feature-maturity/>
* <https://argo-cd.readthedocs.io/en/stable/operator-manual/core/>
