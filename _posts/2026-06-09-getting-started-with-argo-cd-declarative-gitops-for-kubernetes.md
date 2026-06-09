---
layout: post
title: "Getting Started with Argo CD: Declarative GitOps for Kubernetes"
date: 2026-06-09 21:46:44
author: "bits-bytes-nn"
categories: ["Tech Guides"]
tags: []
cover: /assets/images/tech-guides.jpg
use_math: true
---

## 1. Argo CD란 무엇인가?

**Argo CD는 Kubernetes를 위한 선언적(declarative) GitOps 지속적 배포(Continuous Delivery, CD) 도구입니다.** 이 한 문장에 Argo CD의 정체성이 압축되어 있습니다. Argo CD는 Git 저장소를 애플리케이션의 "원하는 상태(desired state)"를 정의하는 **단일 진실 공급원(source of truth)** 으로 사용하는 GitOps 패턴을 따릅니다.

![Argo CD UI](https://argo-cd.readthedocs.io/en/stable/assets/argocd-ui.gif)

### 어떻게 동작하는가

Argo CD는 본질적으로 **Kubernetes 컨트롤러**로 구현되어 있습니다. 이 컨트롤러는 실행 중인 애플리케이션을 지속적으로 모니터링하면서, 클러스터의 현재 라이브 상태(current, live state)를 Git 저장소에 명시된 원하는 타깃 상태(desired target state)와 비교합니다.

- 배포된 애플리케이션의 라이브 상태가 타깃 상태에서 벗어나면, 해당 애플리케이션은 `OutOfSync` 상태로 간주됩니다.
- Argo CD는 이 차이를 **리포트하고 시각화**하며, 라이브 상태를 다시 원하는 상태로 되돌리는 **자동 또는 수동 동기화(sync)** 수단을 제공합니다.
- Git 저장소의 타깃 상태에 가해진 모든 수정 사항은 지정된 대상 환경에 자동으로 적용되고 반영될 수 있습니다.

Git 저장소에 담기는 Kubernetes 매니페스트는 여러 방식으로 지정할 수 있습니다.

- [Kustomize](https://kustomize.io) 애플리케이션
- [Helm](https://helm.sh) 차트
- [Jsonnet](https://jsonnet.org) 파일
- 플레인(plain) YAML/JSON 매니페스트 디렉터리
- Config Management Plugin으로 구성된 임의의 커스텀 도구

이러한 매니페스트 소스 유형은 「지원되는 Config Management 도구」 섹션에서 더 자세히 다룹니다.

### 핵심 가치: 선언적 · 자동화 · 감사 가능

Argo CD의 설계 철학은 두 가지 원칙으로 요약됩니다.

> - 애플리케이션 정의, 구성, 환경은 **선언적(declarative)** 이고 **버전 관리(version controlled)** 되어야 한다.
> - 애플리케이션 배포와 라이프사이클 관리는 **자동화(automated)** 되고, **감사 가능(auditable)** 하며, 이해하기 쉬워야 한다.

즉, "무엇을 어떤 상태로 배포할지"를 Git에 선언으로 기록하고, Argo CD가 그 선언을 클러스터에 자동으로 적용하며, 모든 변경 이력과 API 호출은 추적 가능한 기록으로 남습니다. 이 가치가 왜 중요한지는 「Argo CD를 사용해야 하는 이유」 섹션에서 설명합니다.

### 주요 기능

Argo CD가 제공하는 핵심 기능들을 범주별로 정리하면 다음과 같습니다.

| 범주 | 기능 |
|------|------|
| 배포 자동화 | 지정된 대상 환경으로의 애플리케이션 자동 배포 / 애플리케이션을 원하는 상태로 자동 또는 수동 동기화 |
| 다중 도구 지원 | Kustomize, Helm, Jsonnet, 플레인 YAML 등 여러 config management/템플릿 도구 지원 |
| 다중 클러스터 | 여러 클러스터를 관리하고 배포 |
| 드리프트 감지 | 애플리케이션 리소스의 헬스 상태 분석 / 구성 드리프트(configuration drift)의 자동 감지 및 시각화 |
| 롤백 | Git에 커밋된 임의의 애플리케이션 구성으로 롤백/롤-애니웨어(roll-anywhere) |
| 인증·인가 | SSO 통합(OIDC, OAuth2, LDAP, SAML 2.0, GitHub, GitLab, Microsoft, LinkedIn) / 멀티테넌시와 인가용 RBAC 정책 / 자동화를 위한 액세스 토큰 |
| 인터페이스 | 애플리케이션 활동을 실시간으로 보여주는 Web UI / 자동화 및 CI 통합용 CLI |
| 통합 | Webhook 통합(GitHub, BitBucket, GitLab) / Prometheus 메트릭 |
| 고급 배포 | PreSync, Sync, PostSync 훅으로 blue/green·canary 같은 복잡한 롤아웃 지원 / Git 내 Helm 파라미터 오버라이드 |
| 거버넌스 | 애플리케이션 이벤트 및 API 호출에 대한 감사 추적(audit trails) |

이러한 기능들이 어떤 컴포넌트에 의해 구현되는지는 「핵심 아키텍처 이해」 섹션에서, `Application`·`Sync Status`·`Health` 등 고유 용어의 정확한 정의는 「Argo CD 핵심 개념 사전」 섹션에서 다룹니다.

> 참고: Argo CD는 커뮤니티에 의해 활발히 개발되고 있으며, 일부 기능은 Alpha/Beta 단계로 표시되어 있습니다. 안정성 단계 구분과 프로덕션 사용 시 주의점은 「Feature Maturity 및 안정성 주의사항」 섹션에서 정리합니다.

## 2. Argo CD를 사용해야 하는 이유

「Argo CD란 무엇인가?」에서 인용한 두 가지 설계 원칙을 다시 떠올려 봅시다.

> - 애플리케이션 정의, 구성, 환경은 **선언적(declarative)** 이고 **버전 관리(version controlled)** 되어야 한다.
> - 애플리케이션 배포와 라이프사이클 관리는 **자동화(automated)** 되고, **감사 가능(auditable)** 하며, 이해하기 쉬워야 한다.

이 두 문장은 단순한 슬로건이 아니라, 기존 배포 방식이 겪던 구체적인 문제들에 대한 답입니다. 이 섹션에서는 "그래서 실제로 무엇이 좋아지는가?"를 하나씩 풀어 봅니다.

### Git을 단일 진실 공급원으로 삼을 때 얻는 것

Argo CD는 GitOps 패턴을 따라 **Git 저장소를 원하는 애플리케이션 상태를 정의하는 source of truth로 사용**합니다. 클러스터에 무엇이 배포되어 있어야 하는지를 Git에 선언으로 기록한다는 사실에서 다음 이점들이 자연스럽게 따라옵니다.

- **버전 관리와 이력 추적**: 애플리케이션 정의·구성·환경이 선언적이고 버전 관리되므로, 변경 이력이 Git에 보존됩니다.
- **재현 가능한 환경**: 배포 상태가 사람의 기억이나 일회성 명령에 의존하지 않고 저장소 내용으로 정의되므로, 동일한 매니페스트로 동일한 상태를 다시 만들 수 있습니다.
- **유연한 추적 전략**: 애플리케이션 배포는 브랜치나 태그의 업데이트를 추적하거나, 특정 Git 커밋의 매니페스트에 **고정(pinned)** 할 수 있습니다. (추적 전략의 세부 사항은 Argo CD 공식 문서의 tracking strategies에서 다룹니다.)

매니페스트는 Kustomize, Helm, Jsonnet, 플레인 YAML/JSON 디렉터리, 또는 config management plugin으로 구성한 커스텀 도구 등 여러 방식으로 지정할 수 있어, 기존에 사용하던 템플릿 도구를 그대로 Git에 담아 활용할 수 있습니다.

### 자동화된 배포와 드리프트(drift) 처리

Argo CD는 Kubernetes 컨트롤러로서 실행 중인 애플리케이션을 **지속적으로 모니터링**하며, 클러스터의 라이브 상태를 Git의 타깃 상태와 비교합니다. 이 비교 루프에서 두 가지 핵심 동작이 나옵니다.

1. **자동 배포**: Git의 타깃 상태에 가해진 수정 사항이 지정된 대상 환경에 자동으로 적용·반영될 수 있습니다.
2. **구성 드리프트의 자동 감지 및 시각화**: 라이브 상태가 타깃 상태에서 벗어나면 해당 애플리케이션은 `OutOfSync`로 표시됩니다. Argo CD는 이 차이를 리포트·시각화하고, 라이브 상태를 다시 원하는 상태로 되돌리는 **자동 또는 수동 동기화** 수단을 제공합니다.

즉, 누군가 클러스터를 직접 손대 Git과 달라진 상황(드리프트)을 사람이 일일이 찾아다닐 필요 없이, 컨트롤러가 그 차이를 드러내 줍니다. `OutOfSync` 상태와 sync 동작의 실제 사용법은 「Application 동기화(Sync) 및 상태 확인」에서 실습합니다.

### 안전한 롤백과 감사 추적

전통적인 스크립트·수동 `kubectl apply` 기반 배포에서 가장 까다로운 부분 중 하나가 "문제가 생겼을 때 이전 상태로 정확히 되돌리기"와 "무슨 일이 있었는지 추적하기"입니다. Argo CD는 두 가지 모두를 구조적으로 해결합니다.

- **Rollback / Roll-anywhere**: Git 저장소에 커밋된 **임의의 애플리케이션 구성**으로 되돌릴 수 있습니다. 롤백 대상이 곧 커밋이므로 "어떤 상태로 돌아가는지"가 명확합니다.
- **감사 추적(audit trails)**: 애플리케이션 이벤트와 API 호출에 대한 감사 추적이 제공되어, 변경의 출처와 시점을 사후에 확인할 수 있습니다.

### 기존 CD 방식 대비 정리

아래 표는 Argo CD가 제공하는 기능을, 그것이 해결하는 운영상의 문제와 연결해 정리한 것입니다. 모든 항목은 앞서 인용한 GitOps 원칙과 기능 목록에서 직접 도출됩니다.

| 운영상의 고민 | Argo CD의 접근 |
|--------------|----------------|
| "지금 클러스터에 정확히 무엇이 배포돼 있지?" | Git의 타깃 상태가 곧 선언이며, 라이브 상태와의 차이를 `OutOfSync`로 시각화 |
| "이 변경은 누가, 언제, 왜 했지?" | 변경이 Git 커밋으로 남고, 이벤트·API 호출에 대한 감사 추적 제공 |
| "배포를 사람이 수동으로 돌려야 한다" | 지정된 대상 환경으로의 자동 배포 및 자동/수동 동기화 |
| "롤백할 정확한 이전 상태를 모르겠다" | Git에 커밋된 임의의 구성으로 롤백/roll-anywhere |
| "클러스터를 누가 직접 손대 설정이 틀어졌다" | 구성 드리프트 자동 감지 및 시각화 |
| "여러 클러스터를 일관되게 배포하기 어렵다" | 여러 클러스터를 관리하고 배포 |
| "기존 템플릿 도구를 버리고 싶지 않다" | Kustomize·Helm·Jsonnet·플레인 YAML 등 다중 도구 지원 |

요약하면, Argo CD를 사용하는 이유는 배포를 **선언으로 만들고(Git), 그 선언을 자동으로 클러스터에 반영하며(컨트롤러), 차이와 이력을 항상 드러내는(시각화·감사)** 일관된 모델을 제공하기 때문입니다. 이 모델이 어떤 컴포넌트로 구현되는지는 「핵심 아키텍처 이해」에서, 위에서 등장한 `Application`·`Target State`·`Live State`·`Sync Status` 등 용어의 정확한 정의는 「Argo CD 핵심 개념 사전」에서 이어 다룹니다.

## 3. 사전 지식 및 준비 사항

Argo CD는 기존 기술 위에 세워진 도구입니다. 공식 문서도 본격적인 사용에 앞서 **"플랫폼이 기반으로 삼는 기술들을 이해하는 것이 필요하다"** 고 분명히 밝히고 있습니다. 이 섹션에서는 Argo CD 학습을 시작하기 전에 갖추어야 할 개념적 배경지식과, 실습 환경에 미리 준비해 두어야 할 CLI 도구를 정리합니다.

### 미리 알고 있어야 할 개념

Argo CD의 「Core Concepts」 문서는 독자가 다음 개념에 이미 익숙하다고 가정합니다.

- **Git** — Argo CD는 Git 저장소를 source of truth로 사용합니다.
- **Docker (컨테이너)** — 컨테이너 기술에 대한 이해입니다.
- **Kubernetes** — Argo CD는 Kubernetes 컨트롤러로 동작하며 Kubernetes 리소스(매니페스트)를 배포합니다.
- **Continuous Delivery (CD)** 와 **GitOps** — Argo CD가 구현하는 배포 패턴 자체에 대한 이해입니다.

이러한 기반 기술이 처음이라면, Argo CD 공식 문서의 "Understand The Basics" 페이지가 Docker·Kubernetes 입문 튜토리얼 링크를 제공하므로 먼저 학습하는 것을 권장합니다.

### 매니페스트 템플릿 도구 (선택)

Git에 담기는 매니페스트는 여러 도구로 표현할 수 있습니다. 공식 문서는 **"애플리케이션을 어떻게 템플릿화할 계획인지에 따라"** 다음 도구를 익혀 둘 것을 안내합니다.

| 도구 | 비고 |
|------|------|
| [Kustomize](https://kustomize.io) | 사용 예정인 경우 학습 권장 |
| [Helm](https://helm.sh) | 사용 예정인 경우 학습 권장 |

이 둘은 "사용할 계획이 있다면" 익히면 되는 **선택 사항**입니다.

또한 Argo CD를 CI 도구와 연동할 계획이라면, GitHub Actions나 Jenkins 같은 CI 도구의 사용법도 미리 알아 두면 도움이 됩니다.

### 실습 환경 요구 사항

「Getting Started」 가이드는 다음 환경을 전제로 합니다.

- **`kubectl` CLI 설치** — Kubernetes 명령줄 도구가 설치되어 있어야 합니다.
- **kubeconfig 파일** — 클러스터 접근 정보가 담긴 kubeconfig가 있어야 하며, 기본 위치는 `~/.kube/config` 입니다.
- **CoreDNS** — 클러스터 내 DNS가 활성화되어 있어야 합니다. 예를 들어 microk8s에서는 다음 명령으로 활성화할 수 있습니다.

```bash
microk8s enable dns && microk8s stop && microk8s start
```

현재 kubeconfig에 등록된 클러스터 컨텍스트 목록은 다음과 같이 확인할 수 있습니다.

```bash
# 현재 kubeconfig에 등록된 클러스터 컨텍스트 목록 확인
kubectl config get-contexts -o name
```

이 명령으로 보이는 컨텍스트 이름은 나중에 외부 클러스터를 등록할 때 `argocd cluster add <context>` 에 사용되므로, 본인의 컨텍스트 이름을 미리 확인해 두면 좋습니다.

> **참고: 아키텍처(CPU) 호환성.** 이후 실습에서 사용하는 공식 guestbook 예제 애플리케이션은 AMD64 아키텍처에서만 호환될 수 있습니다. ARM64나 ARMv7 같은 다른 아키텍처에서 실행 중이라면 의존성이나 컨테이너 이미지 문제로 어려움을 겪을 수 있으므로, 필요하다면 호환성을 미리 확인하거나 해당 아키텍처용 이미지를 빌드하는 것을 고려하세요.

> **로컬 Kubernetes 환경.** Docker Desktop이나 그 밖의 로컬 Kubernetes 환경에서 Argo CD를 실행할 계획이라면, 로컬 클러스터에 맞춘 전체 설정 절차가 별도로 정리되어 있으니 Argo CD 공식 문서의 "Running Argo CD Locally" 가이드를 참고하세요.

이제 기반 지식과 도구가 준비되었다면, 다음으로 Argo CD를 구성하는 컴포넌트들이 어떻게 맞물려 동작하는지 살펴봅니다.
</context>

## 4. 핵심 아키텍처 이해

앞선 섹션들에서 Argo CD가 "Kubernetes 컨트롤러로서 Git의 타깃 상태와 클러스터의 라이브 상태를 비교한다"고 반복해서 설명했습니다. 그렇다면 이 동작은 **구체적으로 어떤 컴포넌트들이 맡아서 수행**할까요? Argo CD는 컴포넌트 기반 아키텍처(component based architecture)로 설계되어 있으며, 핵심은 세 가지 컴포넌트 — **API Server**, **Repository Server**, **Application Controller** — 입니다.

아래는 Argo CD 공식 아키텍처 다이어그램입니다.

![Argo CD Architecture](https://argo-cd.readthedocs.io/en/stable/assets/argocd_architecture.png)

### 세 가지 핵심 컴포넌트

#### API Server

API Server는 Web UI, CLI, 그리고 CI/CD 시스템이 소비하는 API를 노출하는 **gRPC/REST 서버**입니다. 다음과 같은 책임을 가집니다.

- 애플리케이션 관리 및 상태 리포팅(application management and status reporting)
- 애플리케이션 작업(sync, rollback, 사용자 정의 액션 등) 호출(invoking)
- 레포지토리 및 클러스터 자격증명 관리 (Kubernetes Secret으로 저장)
- 인증(authentication) 및 외부 ID 공급자로의 인증 위임(auth delegation)
- RBAC 적용(enforcement)
- Git webhook 이벤트를 위한 리스너/포워더(listener/forwarder)

즉, UI와 CLI, 그리고 등록하는 클러스터 자격증명은 모두 이 API Server를 통해 다루어집니다.

#### Repository Server

Repository Server는 애플리케이션 매니페스트를 담고 있는 Git 레포지토리의 **로컬 캐시(local cache)** 를 유지하는 내부 서비스(internal service)입니다. 핵심 역할은 다음 입력이 주어졌을 때 **Kubernetes 매니페스트를 생성해 반환**하는 것입니다.

- repository URL
- revision (commit, tag, branch)
- application path
- template specific settings: parameters, Helm `values.yaml`

다시 말해, Kustomize·Helm 같은 도구가 실제로 "렌더링된 매니페스트"로 변환되는 작업이 이 컴포넌트에서 일어납니다. 위 입력 항목들은 Application을 생성할 때 사용하는 `--repo`, `--path`, revision 같은 파라미터에 그대로 대응됩니다.

#### Application Controller

Application Controller는 실행 중인 애플리케이션을 **지속적으로 모니터링하는 Kubernetes 컨트롤러**입니다. 클러스터의 현재 라이브 상태(current, live state)를 레포지토리에 명시된 원하는 타깃 상태(desired target state)와 비교합니다.

- `OutOfSync` 애플리케이션 상태를 감지(detect)합니다.
- 선택적으로 교정 조치(corrective action)를 취합니다.
- 라이프사이클 이벤트에 대한 사용자 정의 훅(PreSync, Sync, PostSync)을 호출하는 역할을 담당합니다.

이 컴포넌트가 바로 앞선 섹션에서 설명한 "비교 루프"의 실체이며, `OutOfSync` 상태와 sync 동작은 Application 동기화(Sync) 단계에서 실습합니다.

### 컴포넌트 역할 요약

| 컴포넌트 | 유형 | 핵심 역할 |
|----------|------|-----------|
| **API Server** | gRPC/REST 서버 | UI·CLI·CI/CD가 소비하는 API 노출, 애플리케이션 관리·작업 호출, 자격증명 관리, 인증·인증 위임, RBAC 적용, Git webhook 포워딩 |
| **Repository Server** | 내부 서비스 | Git 레포지토리 로컬 캐시 유지, 입력(repo URL·revision·path·템플릿 설정)으로부터 Kubernetes 매니페스트 생성·반환 |
| **Application Controller** | Kubernetes 컨트롤러 | 라이브 상태와 타깃 상태 비교, `OutOfSync` 감지 및 교정, PreSync/Sync/PostSync 훅 호출 |

### 컴포넌트가 함께 동작하는 흐름

세 컴포넌트의 책임을 이어 붙이면, GitOps 루프가 어떻게 완성되는지 보입니다.

1. 사용자(또는 CI 시스템)가 **API Server** 를 통해 UI·CLI로 sync 같은 작업을 호출합니다.
2. **Repository Server** 가 Git 레포지토리의 캐시에서 지정된 revision·path의 매니페스트를 렌더링해 반환합니다.
3. **Application Controller** 가 이 타깃 상태를 클러스터의 라이브 상태와 비교하고, 차이가 있으면 `OutOfSync`로 표시하며 필요한 교정 조치와 훅을 수행합니다.

이러한 컴포넌트 기반 설계 덕분에, 일부 컴포넌트만 설치하는 **더 미니멀한 설치**도 가능합니다. 예를 들어 UI·SSO·멀티테넌시가 필요 없을 때는 API Server와 UI를 제외하고 핵심 GitOps 기능만 갖춘 형태로 설치할 수 있는데, 이것이 헤드리스 모드로 동작하는 **Core 설치**입니다. 반대로 프로덕션에서는 동일한 컴포넌트를 다중 복제본(replica)으로 튜닝한 **High Availability(HA)** 구성을 사용할 수 있습니다.

> 참고: 위 세 컴포넌트는 각각 `argocd-server`(API Server), `argocd-repo-server`(Repository Server), `argocd-application-controller`(Application Controller)라는 이름의 워크로드로 클러스터에 배포됩니다. 또한 Controller는 Kube API와 Git에 대한 부하를 줄이는 캐싱 메커니즘으로 Redis를 사용하므로, 표준 설치에는 Redis도 함께 포함됩니다.

이제 각 컴포넌트가 다루는 `Application`, `Target State`, `Live State`, `Sync Status` 같은 용어의 정확한 정의가 필요합니다. 이는 Argo CD 핵심 개념을 다루는 다음 부분에서 이어 다룹니다.

## 5. Argo CD 핵심 개념 사전

지금까지 「Argo CD란 무엇인가?」와 「핵심 아키텍처 이해」에서 `Application`, `Target State`, `Live State`, `Sync Status`, `OutOfSync` 같은 단어를 여러 번 사용해 왔습니다. 이제 이 용어들의 정확한 정의를 한곳에 모아 정리합니다. 이 개념들은 Argo CD 공식 문서의 「Core Concepts」에 정의된 것으로, **Git·Docker·Kubernetes·Continuous Delivery·GitOps의 일반 개념에는 이미 익숙하다는 전제** 아래 Argo CD에만 고유한 용어를 다룹니다. (일반 기반 개념은 「사전 지식 및 준비 사항」을 참고하세요.)

### 용어 정의표

| 용어 | 정의 |
|------|------|
| **Application** | 매니페스트로 정의된 Kubernetes 리소스의 그룹. 이는 Custom Resource Definition(CRD)으로 구현됩니다. |
| **Application source type** | 애플리케이션을 빌드하는 데 사용되는 **Tool**(도구)이 무엇인지를 가리킵니다. |
| **Target state** | 애플리케이션의 원하는 상태(desired state). Git 저장소의 파일들로 표현됩니다. |
| **Live state** | 해당 애플리케이션의 라이브 상태(live state). 실제로 어떤 Pod 등이 배포되어 있는지를 말합니다. |
| **Sync status** | 라이브 상태가 타깃 상태와 일치하는지 여부. 즉 "배포된 애플리케이션이 Git이 말하는 그대로인가?"입니다. |
| **Sync** | 애플리케이션을 타깃 상태로 이동시키는 과정. 예를 들어 Kubernetes 클러스터에 변경을 적용(apply)하는 것입니다. |
| **Sync operation status** | sync가 성공했는지 여부. |
| **Refresh** | Git의 최신 코드를 라이브 상태와 비교하여, 무엇이 다른지 파악하는 동작. |
| **Health** | 애플리케이션의 헬스(health). 정상적으로 실행되고 있는가? 요청을 처리할 수 있는가? |
| **Tool** | 파일 디렉터리로부터 매니페스트를 생성하는 도구. 예: Kustomize. (Application Source Type 참조) |
| **Configuration management tool** | **Tool**과 동일한 의미. |
| **Configuration management plugin** | 커스텀 도구(custom tool). |

### 개념들이 서로 어떻게 맞물리는가

위 정의들은 따로 떨어진 단어가 아니라 하나의 흐름으로 연결됩니다. 핵심은 **두 가지 상태의 비교**입니다.

- **Target state(타깃 상태)** 는 Git에 선언된 "있어야 할 모습"입니다.
- **Live state(라이브 상태)** 는 클러스터에 "실제로 있는 모습"입니다.

이 둘을 비교한 결과가 **Sync status**입니다. 두 상태가 일치하지 않으면 「핵심 아키텍처 이해」에서 본 것처럼 `OutOfSync`로 표시됩니다. 이때 라이브 상태를 타깃 상태로 옮기는 행위가 **Sync**이며, 그 결과(성공/실패)는 **Sync operation status**로 따로 추적됩니다.

```text
Git 저장소 (Target state)
        │
        │  ← Refresh: 최신 Git 코드와 Live state를 비교해 차이를 파악
        ▼
   비교 결과 = Sync status  ──►  일치하면 Synced / 다르면 OutOfSync
        │
        │  ← Sync: Live state를 Target state로 이동 (예: kubectl apply)
        ▼
클러스터의 실제 리소스 (Live state) ──► 정상 동작 여부 = Health
```

여기서 **Sync status와 Health는 서로 다른 축**이라는 점에 유의해야 합니다. Sync status는 "Git과 클러스터가 같은가?"라는 *일치 여부*를 묻고, Health는 "그 애플리케이션이 제대로 동작하며 요청을 처리할 수 있는가?"라는 *동작 상태*를 묻습니다. 두 질문은 독립적이므로, 예를 들어 한 애플리케이션이 `Synced`이면서 동시에 헬스 문제를 가질 수도, 그 반대일 수도 있습니다. 이 두 상태를 실제 CLI 출력과 UI에서 어떻게 읽는지는 「Application 동기화(Sync) 및 상태 확인」에서 직접 다룹니다.

### Tool / Configuration management tool / plugin

세 용어는 모두 **"매니페스트를 만들어 내는 도구"** 라는 같은 축에 있습니다.

- **Tool**(= **Configuration management tool**)은 파일 디렉터리로부터 매니페스트를 생성하는 도구를 가리키며, Kustomize가 대표적인 예입니다. 이는 **Application source type** 과 직접 연결됩니다 — 어떤 Tool로 애플리케이션을 빌드하는지가 곧 source type이기 때문입니다.
- **Configuration management plugin**은 이 중에서도 **커스텀 도구**를 의미합니다.

이 도구들이 「핵심 아키텍처 이해」에서 설명한 Repository Server에서 매니페스트로 변환되는 주체이며, Argo CD가 실제로 어떤 도구들을 지원하고 어떻게 자동 감지하는지는 「지원되는 Config Management 도구」에서 자세히 다룹니다.

### Application은 CRD다

마지막으로 가장 중요한 한 가지를 강조해 둡니다. **Application은 단순한 추상 개념이 아니라 Kubernetes의 Custom Resource Definition(CRD)** 입니다. 즉 Application 자체가 클러스터에 저장되는 Kubernetes 리소스이며, 위에서 설명한 Target state·Live state·Sync status·Health 같은 정보가 이 리소스 위에서 관리됩니다. 이 사실은 헤드리스 모드인 **Core 설치**에서 특히 의미가 큰데, API 서버나 UI 없이도 `Application`(과 `ApplicationSet`) CRD를 통해 GitOps를 수행할 수 있기 때문입니다. 이에 대해서는 「Argo CD Core(헤드리스 모드) 사용하기」에서 이어 다룹니다.

이제 용어가 정리되었으니, 다음으로 이 개념들을 구현한 Argo CD를 실제로 **어떤 설치 유형으로 배포할지** 선택하는 단계로 넘어갑니다.

## 6. Argo CD 설치 방법 선택

「핵심 아키텍처 이해」에서 설명한 것처럼 Argo CD는 컴포넌트 기반 아키텍처로 설계되어 있어, **더 미니멀한 설치가 가능합니다.** 공식 문서는 설치 유형을 크게 두 가지 — **Multi-Tenant** 와 **Core** — 로 구분합니다. 본격적으로 명령어를 실행하기 전에, 자신의 사용 시나리오에 맞는 유형을 먼저 결정하는 것이 중요합니다.

### Multi-Tenant vs. Core: 큰 갈래

| 구분 | Multi-Tenant | Core (헤드리스) |
|------|--------------|-----------------|
| 주 사용 대상 | 조직 내 여러 애플리케이션 개발 팀을 지원하며 플랫폼 팀이 운영 | Argo CD를 단독으로 사용하고 멀티테넌시가 필요 없는 클러스터 관리자 |
| 접근 방식 | end-user가 API Server를 통해 Web UI 또는 `argocd` CLI로 접근 | API Server·UI 미포함, GitOps에 의존 |
| 포함 컴포넌트 | 전체 컴포넌트 | 더 적은 컴포넌트 |
| 설치 난이도 | 표준 | 더 쉬움 |
| 비고 | 가장 일반적인 설치 방법 | 각 컴포넌트의 경량(non-HA) 버전을 설치 |

Core 설치는 **API Server나 UI를 포함하지 않고**, 각 컴포넌트의 경량(non-HA) 버전을 설치합니다. UI·SSO·멀티테넌시 기능이 필요 없을 때 적합하며, 매니페스트는 `core-install.yaml` 하나로 제공됩니다. Core 모드의 실제 설치와 사용법(`argocd login --core`, 로컬 대시보드 등)은 「Argo CD Core(헤드리스 모드) 사용하기」에서 별도로 다룹니다.

### Multi-Tenant 설치 매니페스트 세분화

Multi-Tenant 설치에는 **Non-HA** 와 **HA** 두 종류의 매니페스트가 제공되며, 각각 다시 클러스터 권한 수준에 따라 `install.yaml` 과 `namespace-install.yaml` 로 나뉩니다.

| 매니페스트 | HA 여부 | 권한 범위 | 권장 용도 |
|------------|---------|-----------|-----------|
| `install.yaml` | Non-HA | cluster-admin 권한 | Argo CD가 실행되는 동일 클러스터(`kubernetes.svc.default`)에 배포할 때. 입력된 자격증명으로 외부 클러스터 배포도 가능 |
| `namespace-install.yaml` | Non-HA | 네임스페이스 수준 권한만 필요 (ClusterRole 불필요) | 동일 클러스터 배포가 필요 없고 입력된 외부 클러스터 자격증명에만 의존할 때. 팀별로 여러 Argo CD 인스턴스를 운영하는 경우 등 |
| `ha/install.yaml` | HA | cluster-admin 권한 | `install.yaml` 과 동일하되 지원 컴포넌트가 다중 복제본(replica)으로 구성 |
| `ha/namespace-install.yaml` | HA | 네임스페이스 수준 권한 | `namespace-install.yaml` 과 동일하되 다중 복제본 구성 |

핵심 차이를 정리하면 다음과 같습니다.

- **Non-HA vs. HA**: HA 번들은 *동일한 컴포넌트* 를 포함하되 고가용성과 복원력(resiliency)을 위해 튜닝되어, 지원되는 컴포넌트에 대해 다중 복제본을 사용합니다. **Non-HA 설치는 프로덕션 용도로 권장되지 않으며**, 일반적으로 평가·데모·테스트 기간에 사용됩니다. **HA 설치가 프로덕션 용도로 권장됩니다.**
- **`install.yaml` vs. `namespace-install.yaml`**: 전자는 cluster-admin 권한(ClusterRole 포함)이며, 후자는 클러스터 롤 없이 네임스페이스 수준 권한만 요구합니다.

> **주의: `namespace-install.yaml` 의 CRD.** `namespace-install.yaml` 에는 Argo CD CRD가 **포함되어 있지 않으므로** 별도로 설치해야 합니다. CRD 매니페스트는 `manifests/crds` 디렉터리에 있으며 다음 명령으로 설치합니다.

```bash
kubectl apply --server-side --force-conflicts -k https://github.com/argoproj/argo-cd/manifests/crds?ref=stable
```

> **주의: ClusterRoleBinding과 네임스페이스.** 설치 매니페스트의 `ClusterRoleBinding` 은 `argocd` 네임스페이스의 ServiceAccount에 바인딩되어 있습니다. 네임스페이스를 변경할 경우 ClusterRoleBinding을 새 네임스페이스에 맞게 올바르게 조정하지 않으면 권한 관련 오류가 발생할 수 있습니다. 커스텀 네임스페이스 설치 방법은 아래 Kustomize 절을 참고하세요.

`namespace-install.yaml` 의 기본 역할(role)로는 동일 클러스터 안에서 Argo CD 리소스(Applications, ApplicationSets, AppProjects)만 배포할 수 있습니다. 즉 실제 배포는 외부 클러스터로 수행되는 GitOps 모드만 지원합니다. 동일 클러스터에 배포하려면 제공된 자격증명을 사용하면 됩니다.

```bash
# 동일 클러스터를 in-cluster로 등록하는 예 (namespace-install 사용 시)
argocd cluster add <context> --in-cluster --namespace <your namespace="">
```

이 동작을 변경하려면 새 역할을 정의하고 `argocd-application-controller` ServiceAccount에 바인딩하면 됩니다.

### Kustomize로 설치하기

Argo CD 매니페스트는 Kustomize로도 설치할 수 있습니다. 공식 문서는 매니페스트를 **원격 리소스(remote resource)로 포함하고 Kustomize 패치로 추가 커스터마이징을 적용**하는 방식을 권장합니다.

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: argocd
resources:
  - https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

기본값인 `argocd` 가 아닌 **커스텀 네임스페이스**에 설치하려면, ClusterRoleBinding이 해당 네임스페이스의 ServiceAccount를 올바르게 가리키도록 패치를 적용합니다.

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: <your-custom-namespace>
resources:
  - https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
patches:
  - patch: |-
      - op: replace
        path: /subjects/0/namespace
        value: <your-custom-namespace>
    target:
      kind: ClusterRoleBinding
```

이 패치는 ClusterRoleBinding이 커스텀 네임스페이스의 ServiceAccount로 올바르게 매핑되도록 하여 배포 시 권한 관련 문제를 방지합니다.

### Helm으로 설치하기

Argo CD는 Helm으로도 설치할 수 있습니다. 단, 이 **Helm 차트는 현재 커뮤니티에서 유지보수**되며 `argo-helm/charts/argo-cd` 저장소에서 제공됩니다.

### 설치 방법 한눈에 비교

| 설치 방법 | 입력 | 커스터마이징 | 비고 |
|-----------|------|--------------|------|
| `kubectl apply` (원격 매니페스트) | `install.yaml` / `ha/install.yaml` / `core-install.yaml` 등 | 제한적 | 「Argo CD 설치 실습」에서 표준 절차로 다룸 |
| Kustomize | 원격 리소스 + 패치 | 패치로 유연하게 가능 | 커스텀 네임스페이스 설치에 유용 |
| Helm | 커뮤니티 차트 | values로 가능 | 차트는 커뮤니티 유지보수 |

### 지원되는 Kubernetes 버전 호환 표

Argo CD의 버전 지원 정책 자체는 공식 문서의 "Release Process and Cadence"에 정의되어 있습니다. 아래 표는 각 Argo CD 버전이 **테스트된(tested)** Kubernetes 버전을 보여 줍니다.

| Argo CD 버전 | 테스트된 Kubernetes 버전 |
|--------------|--------------------------|
| 3.4 | v1.35, v1.34, v1.33, v1.32 |
| 3.3 | v1.35, v1.34, v1.33, v1.32 |
| 3.2 | v1.34, v1.33, v1.32, v1.31 |

설치 유형을 결정했다면, 이제 실제로 `argocd` 네임스페이스를 만들고 공식 매니페스트를 적용하는 표준 설치 절차로 넘어갑니다. 특히 Quick Start에서 등장한 `--server-side --force-conflicts` 플래그가 왜 필요한지는 다음 「Argo CD 설치 실습」에서 자세히 설명합니다.
</your-custom-namespace></your-custom-namespace></your></context>

## 7. Argo CD 설치 실습

앞선 「Argo CD 설치 방법 선택」에서 자신의 시나리오에 맞는 설치 유형(Multi-Tenant Non-HA / HA, Core)을 결정했다면, 이제 실제로 클러스터에 Argo CD를 배포할 차례입니다. 이 섹션에서는 「Getting Started」 가이드가 제시하는 **표준 설치 절차** — `argocd` 네임스페이스를 만들고 공식 매니페스트를 적용하는 두 줄의 명령어 — 를 단계별로 따라가 보고, 그 안에 등장하는 `--server-side --force-conflicts` 플래그가 왜 필요한지 정확히 짚어 봅니다.

> **사전 확인.** 이 절차는 「사전 지식 및 준비 사항」에서 정리한 요구 사항(`kubectl` 설치, `~/.kube/config` 위치의 kubeconfig, CoreDNS 활성화)이 갖춰져 있다는 전제 위에서 진행됩니다.

### 1단계 — `argocd` 네임스페이스 생성

먼저 Argo CD의 모든 서비스와 애플리케이션 리소스가 위치할 전용 네임스페이스를 만듭니다.

```bash
kubectl create namespace argocd
```

이 명령은 새 `argocd` 네임스페이스를 생성합니다. 모든 Argo CD 서비스(앞서 본 `argocd-server`, `argocd-repo-server`, `argocd-application-controller` 등)와 애플리케이션 리소스가 이 네임스페이스 안에 자리 잡게 됩니다.

### 2단계 — 공식 매니페스트 적용

다음으로 stable 브랜치의 공식 설치 매니페스트를 적용합니다.

```bash
kubectl apply -n argocd --server-side --force-conflicts \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

이 명령은 stable 브랜치의 공식 매니페스트를 적용하여 Argo CD를 설치합니다. 위 예시는 「Argo CD 설치 방법 선택」에서 소개한 Multi-Tenant Non-HA용 `install.yaml`을 사용합니다. HA 설치를 원한다면 같은 자리에 `ha/install.yaml`을, Core 설치를 원한다면 `core-install.yaml`을 지정하면 됩니다(Core의 전체 설치·사용법은 「Argo CD Core(헤드리스 모드) 사용하기」에서 별도로 다룹니다).

> **프로덕션 팁: 버전 고정(pinned version).** 위 명령은 `stable` 브랜치를 가리키므로 적용 시점에 따라 내용이 달라질 수 있습니다. 공식 문서는 프로덕션에서는 `v3.2.0`처럼 **고정된 버전을 사용할 것을 권장**합니다.

### `--server-side --force-conflicts`는 왜 필요한가

이 두 플래그는 단순한 관례가 아니라 **CRD 크기 제한** 때문에 요구되는 것입니다. 공식 가이드의 설명을 정리하면 다음과 같습니다.

| 플래그 | 이유 | 비고 |
|--------|------|------|
| `--server-side` | 일부 Argo CD CRD(예: ApplicationSet)가 client-side `kubectl apply`가 부과하는 **262KB annotation 크기 제한**을 초과하기 때문. server-side apply는 `last-applied-configuration` annotation을 저장하지 않으므로 이 제한을 회피합니다. | — |
| `--force-conflicts` | 이전에 다른 도구(Helm이나 이전 `kubectl apply` 등)가 소유하던 필드의 **소유권을 가져오기(take ownership)** 위해. | 신규 설치에서는 안전하며, 업그레이드 시 필요 |

여기서 client-side apply가 부딪히는 한계는 262KB annotation 크기 제한입니다. server-side apply는 `last-applied-configuration` annotation을 저장하지 않기 때문에 큰 CRD도 문제없이 적용됩니다.

> **주의: 필드 덮어쓰기(overwrite) 동작.** `--force-conflicts`를 사용하면, Argo CD 매니페스트에 **정의된 필드**(예: `affinity`, `env`, `probes`)에 직접 가한 커스텀 수정 사항은 덮어쓰여집니다. 반면 매니페스트에 **명시되지 않은 필드**(예: `resources` limits/requests, `tolerations`)는 보존됩니다. 따라서 기존 설치에 손수 패치한 부분이 있다면 이 차이를 미리 인지해 두어야 합니다.

### 설치 시 주의할 점

- **ClusterRoleBinding과 네임스페이스 일치.** 설치 매니페스트에 포함된 `ClusterRoleBinding`은 `argocd` 네임스페이스를 참조합니다. 만약 다른 네임스페이스에 설치한다면 그 참조를 새 네임스페이스에 맞게 갱신해야 하며, 그렇지 않으면 권한 관련 오류가 발생할 수 있습니다. 커스텀 네임스페이스에 설치하는 Kustomize 패치 방법은 「Argo CD 설치 방법 선택」에서 다루었습니다.
- **기본 인증서는 self-signed.** 이 기본 설치는 self-signed 인증서를 사용하며, 추가 작업 없이는 외부에서 접근할 수 없습니다. 접근 방법(인증서 구성, OS 신뢰 설정, 또는 CLI의 `--insecure` 플래그 사용)은 다음 「Argo CD CLI 설치 및 UI 접근」에서 이어집니다.
- **Redis 비밀번호.** 이 기본 설치의 Redis는 비밀번호 인증을 사용합니다. Redis 비밀번호는 Argo CD가 설치된 네임스페이스의 `argocd-redis` Secret에 `auth` 키로 저장됩니다.

### 설치 확인

매니페스트 적용 후, `argocd` 네임스페이스에 워크로드(Pod)가 생성되었는지 확인할 수 있습니다.

```bash
kubectl get pods -n argocd
```

이후 이어지는 섹션에서 사용할 CLI 명령들을 위해, 현재 `kubectl` 컨텍스트의 기본 네임스페이스를 `argocd`로 설정해 두면 편리합니다.

```bash
kubectl config set-context --current --namespace=argocd
```

> 공식 가이드에 따르면 이 `set-context` 설정은 **뒤이어 나오는 명령들에만 필요**합니다. 위 1~2단계 명령은 이미 `-n argocd`를 포함하고 있기 때문입니다.

이제 Argo CD가 클러스터에서 동작하기 시작했습니다. 다음 단계는 이 설치된 인스턴스와 상호작용하기 위해 `argocd` CLI를 내려받고, 초기 admin 비밀번호를 조회하며, UI에 접근하는 방법을 살펴보는 것입니다.

## Argo CD CLI 설치 및 UI 접근

「Argo CD 설치 실습」에서 클러스터에 Argo CD를 배포했지만, 한 가지 중요한 제약이 남아 있습니다. 공식 가이드가 명시하듯 **기본적으로 Argo CD는 클러스터 외부로 노출되지 않습니다(By default, Argo CD isn't exposed outside the cluster).** 이 섹션에서는 ① `argocd` CLI를 내려받고, ② 초기 admin 비밀번호를 조회하며, ③ 브라우저나 CLI가 API 서버에 도달할 수 있도록 **LoadBalancer · Ingress · Port Forwarding** 세 가지 접근 방식을 다룹니다. 실제 로그인(`argocd login`)과 클러스터 등록은 다음 「CLI로 로그인하고 클러스터 등록하기」에서 이어집니다.

### Argo CD CLI 내려받기

CLI는 GitHub 릴리스 페이지에서 최신 버전을 직접 받을 수 있습니다.

- 최신 릴리스: <https: argo-cd="" argoproj="" github.com="" latest="" releases="">

Mac, Linux, WSL 환경이라면 Homebrew로 더 간단히 설치할 수 있습니다.

```bash
brew install argocd
```

> 더 상세한 설치 방법은 Argo CD 공식 문서의 CLI 설치 문서에서 확인할 수 있습니다. CLI는 「Argo CD란 무엇인가?」에서 소개한 "자동화 및 CI 통합용 CLI"이자, 이후 모든 실습 명령(`argocd app create`, `argocd app sync` 등)을 실행하는 도구입니다.

### 초기 admin 비밀번호 조회

`admin` 계정의 초기 비밀번호는 **자동 생성**되어, Argo CD가 설치된 네임스페이스의 `argocd-initial-admin-secret` Secret에 `password` 필드의 **평문(clear text)** 으로 저장됩니다. CLI로 다음과 같이 조회합니다.

```bash
argocd admin initial-password -n argocd
```

> **보안 경고.** 비밀번호를 변경한 후에는 `argocd-initial-admin-secret`을 Argo CD 네임스페이스에서 **삭제해야 합니다.** 이 Secret은 초기 생성 비밀번호를 평문으로 보관하는 것 외에 다른 용도가 없으며 언제든 안전하게 삭제할 수 있습니다. 새 admin 비밀번호를 다시 생성해야 할 때 Argo CD가 필요에 따라 재생성합니다. 이 권장 사항은 「모범 사례 및 다음 단계」에서 다시 정리합니다.

조회한 비밀번호는 사용자명 `admin`과 함께 다음 섹션의 로그인 단계에서 사용합니다. 로그인 후에는 다음 명령으로 비밀번호를 변경할 수 있습니다.

```bash
argocd account update-password
```

### Argo CD 접근하기: 세 가지 방법

브라우저나 CLI에서 API 서버에 접근하려면 다음 세 가지 방법 중 하나를 사용합니다.

| 방법 | 적합한 상황 | 핵심 동작 |
|------|------------|-----------|
| Service Type LoadBalancer | 클라우드 환경에서 외부 IP로 상시 노출하고 싶을 때 | `argocd-server` 서비스 타입을 `LoadBalancer`로 변경 |
| Ingress | Ingress 컨트롤러로 호스트네임 기반 라우팅을 구성할 때 | 공식 Ingress 문서의 구성 절차를 따름 |
| Port Forwarding | 서비스를 노출하지 않고 로컬에서 빠르게 접근할 때 | `kubectl port-forward`로 로컬 포트와 연결 |

#### 1) Service Type Load Balancer

`argocd-server` 서비스의 타입을 `LoadBalancer`로 변경합니다.

```bash
kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "LoadBalancer"}}'
```

잠시 후 클라우드 제공자가 서비스에 외부 IP 주소를 할당합니다. 다음 명령으로 그 IP를 조회할 수 있습니다.

```bash
kubectl get svc argocd-server -n argocd -o=jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

#### 2) Ingress

Ingress를 통한 노출은 별도의 컨트롤러 구성과 라우팅 설정이 필요합니다. 구성 방법은 Argo CD 공식 문서의 Ingress Configuration 문서를 참고하세요.

#### 3) Port Forwarding

서비스를 외부로 노출하지 않고도 API 서버에 연결할 수 있는 가장 간단한 방법입니다.

```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

이제 API 서버에는 다음 주소로 접근할 수 있습니다.

```text
https://localhost:8080
```

### 기억해 둘 점: self-signed 인증서

「Argo CD 설치 실습」에서 짚었듯, 이 기본 설치는 **self-signed 인증서**를 사용하므로 약간의 추가 작업 없이는 접근할 수 없습니다. 공식 가이드는 다음 중 하나를 수행하라고 안내합니다.

- 인증서를 구성하고(공식 TLS 문서 참고) 클라이언트 OS가 이를 신뢰하도록 설정한다.
- 클라이언트 OS가 self-signed 인증서를 신뢰하도록 설정한다.
- 이 가이드의 모든 Argo CD CLI 작업에 `--insecure` 플래그를 사용한다.

### CLI가 API 서버에 도달하지 못할 때

CLI 환경은 Argo CD API 서버와 통신할 수 있어야 합니다. 위 방법으로 직접 접근이 불가능하다면, **port forwarding을 통해 CLI가 접근하도록** 다음 두 가지 중 하나를 사용할 수 있습니다.

1. 모든 CLI 명령에 `--port-forward-namespace argocd` 플래그를 추가한다.
2. `ARGOCD_OPTS` 환경 변수를 설정한다.

```bash
export ARGOCD_OPTS='--port-forward-namespace argocd'
```

이렇게 접근 경로가 마련되면, 다음 섹션에서 `argocd login`으로 실제 인증을 수행하고 외부 클러스터를 등록하게 됩니다.

> **팁: 접근 단계를 건너뛰는 경우.** UI·SSO·멀티클러스터 기능이 필요 없다면, `argocd login --core`를 사용해 CLI 접근을 구성하고 이 섹션과 다음 섹션(접근·로그인·클러스터 등록)을 건너뛸 수 있습니다. Core 모드의 전체 사용법은 「Argo CD Core(헤드리스 모드) 사용하기」에서 다룹니다.
</https:>

## 9. CLI로 로그인하고 클러스터 등록하기

앞선 「Argo CD CLI 설치 및 UI 접근」에서 ① `argocd` CLI를 설치하고, ② 초기 admin 비밀번호를 조회했으며, ③ LoadBalancer·Ingress·Port Forwarding 중 하나로 API 서버에 도달할 수 있는 경로를 마련했습니다. 이제 그 경로를 통해 **실제로 인증(login)** 하고, 필요하다면 애플리케이션을 배포할 **외부 클러스터를 등록**하는 단계입니다.

> **참고: 이 단계를 건너뛸 수 있는 경우.** UI·SSO·멀티클러스터 기능이 필요 없다면 `argocd login --core`로 CLI 접근을 구성하고 접근·로그인·클러스터 등록 단계를 건너뛸 수 있습니다. Core 모드의 전체 사용법은 「Argo CD Core(헤드리스 모드) 사용하기」에서 다룹니다.

### CLI로 로그인하기

이전 섹션에서 조회한 비밀번호와 사용자명 `admin`을 사용해, Argo CD의 IP 또는 호스트네임으로 로그인합니다.

```bash
argocd login <argocd_server>
```

여기서 `<argocd_server>`에는 「Argo CD CLI 설치 및 UI 접근」에서 마련한 접근 경로의 주소가 들어갑니다. 예를 들어 LoadBalancer로 외부 IP를 할당받았다면 그 IP/호스트네임을, Port Forwarding을 사용했다면 `localhost:8080`을 지정합니다.

다음 사항을 기억해 둡니다.

- **self-signed 인증서.** 기본 설치는 self-signed 인증서를 사용하므로, 인증서를 신뢰하도록 구성하지 않았다면 이 가이드의 모든 Argo CD CLI 작업에 `--insecure` 플래그를 사용할 수 있습니다.
- **CLI가 API 서버에 직접 도달하지 못할 때.** CLI 환경은 Argo CD API 서버와 통신할 수 있어야 합니다. 위 방식대로 직접 접근이 불가능하다면, port forwarding을 통해 접근하도록 다음 중 하나를 사용합니다.
  1. 모든 CLI 명령에 `--port-forward-namespace argocd` 플래그를 추가한다.
  2. `ARGOCD_OPTS` 환경 변수를 설정한다.

```bash
export ARGOCD_OPTS='--port-forward-namespace argocd'
```

로그인이 완료되면, 이후 `argocd app create`, `argocd app sync` 같은 명령을 인증된 상태로 실행할 수 있습니다.

### 어떤 클러스터에 배포할 것인가: 내부 vs. 외부

클러스터 등록을 이해하려면 먼저 **배포 대상이 어디인가**를 구분해야 합니다.

| 배포 대상 | 클러스터 등록 필요 여부 | 사용할 API 서버 주소 |
|-----------|-------------------------|----------------------|
| **내부(internal)** — Argo CD가 실행 중인 바로 그 클러스터 | 불필요 | `https://kubernetes.default.svc` |
| **외부(external)** — 다른 클러스터 | `argocd cluster add`로 등록 필요 | 등록된 클러스터의 K8s API 서버 주소 |

즉, **클러스터 등록은 외부 클러스터에 배포할 때만 필요한 선택적(optional) 단계**입니다. Argo CD가 설치된 동일 클러스터에 배포한다면 별도 등록 없이 애플리케이션의 대상 서버로 `https://kubernetes.default.svc`를 사용하면 됩니다.

### 외부 클러스터 등록하기

외부 클러스터를 등록하는 절차는 두 단계입니다. 먼저 현재 kubeconfig에 등록된 클러스터 컨텍스트 목록을 확인합니다.

```bash
kubectl config get-contexts -o name
```

목록에서 컨텍스트 이름을 골라 `argocd cluster add CONTEXTNAME`에 전달합니다. 예를 들어 `docker-desktop` 컨텍스트라면 다음과 같습니다.

```bash
argocd cluster add docker-desktop
```

#### 이 명령이 실제로 하는 일

위 명령은 대상 클러스터에 다음을 설정합니다.

- 해당 kubectl 컨텍스트의 **`kube-system` 네임스페이스에 `argocd-manager`라는 ServiceAccount**를 설치합니다.
- 이 ServiceAccount를 **admin 수준의 ClusterRole**에 바인딩합니다.

Argo CD는 이 ServiceAccount의 토큰을 사용해 배포·모니터링 등 관리 작업을 수행합니다.

> **권한 범위 조정.** `argocd-manager-role` 역할의 규칙은, 제한된 네임스페이스·그룹·종류(kind)에 대해서만 `create`, `update`, `patch`, `delete` 권한을 갖도록 수정할 수 있습니다. 다만 Argo CD가 정상 동작하려면 **cluster-scope에서 `get`, `list`, `watch` 권한은 반드시 필요**합니다.

### namespace-install 사용 시: 동일 클러스터를 in-cluster로 등록

「Argo CD 설치 방법 선택」에서 다룬 `namespace-install.yaml`을 사용한 경우, 기본 역할로는 GitOps 모드만 지원되며 실제 배포는 외부 클러스터로 수행됩니다. 이때 **동일 클러스터에 배포하려면** 제공된 자격증명으로 in-cluster 등록을 명시할 수 있습니다.

```bash
argocd cluster add <context> --in-cluster --namespace <your namespace="">
```

이렇게 로그인과(필요 시) 클러스터 등록이 끝나면, 이제 실제로 배포할 첫 번째 Application을 만들 차례입니다. 다음 섹션에서는 공식 guestbook 예제 레포지토리를 사용해 `argocd app create`로 Application을 생성합니다.
</your></context></argocd_server></argocd_server>

## 10. 첫 번째 Application 생성 — CLI 방식

앞선 「CLI로 로그인하고 클러스터 등록하기」에서 인증을 마쳤다면, 이제 실제로 배포할 **첫 번째 Application**을 만들 차례입니다. Argo CD는 동작 방식을 시연하기 위한 공식 예제 레포지토리 — guestbook 애플리케이션 — 를 제공합니다.

- 예제 레포지토리: <https: argocd-example-apps.git="" argoproj="" github.com="">

이 섹션에서는 `argocd app create` 명령으로 이 guestbook 애플리케이션을 CLI에서 생성하는 전 과정을 따라갑니다.

> **참고: 아키텍처 호환성.** 이 예제 애플리케이션은 **AMD64 아키텍처에서만 호환될 수 있습니다.** ARM64나 ARMv7 등 다른 아키텍처에서 실행 중이라면 의존성이나 컨테이너 이미지 문제를 만날 수 있습니다.

### 1단계 — 현재 네임스페이스를 `argocd`로 설정

먼저 현재 `kubectl` 컨텍스트의 네임스페이스를 `argocd`로 설정합니다.

```bash
kubectl config set-context --current --namespace=argocd
```

이미 이 명령을 실행했다면 다시 할 필요는 없지만, 안전하게 한 번 더 확인해 두면 좋습니다.

### 2단계 — `argocd app create`로 guestbook 생성

다음 명령으로 guestbook 예제 Application을 생성합니다.

```bash
argocd app create guestbook \
  --repo https://github.com/argoproj/argocd-example-apps.git \
  --path guestbook \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace default
```

### 각 플래그가 의미하는 것

이 명령에 사용된 플래그들은 Repository Server의 입력(repository URL, revision, application path)과 배포 대상 정보에 직접 대응됩니다.

| 플래그 / 인자 | 값 | 의미 |
|----------------|------|------|
| `guestbook`(첫 인자) | `guestbook` | 생성할 Application의 이름 |
| `--repo` | `https://github.com/argoproj/argocd-example-apps.git` | 매니페스트가 담긴 Git 레포지토리 URL (Target state의 출처) |
| `--path` | `guestbook` | 레포지토리 내에서 매니페스트가 위치한 디렉터리 경로 |
| `--dest-server` | `https://kubernetes.default.svc` | 배포 대상 클러스터의 K8s API 서버 주소 |
| `--dest-namespace` | `default` | 리소스가 배포될 대상 네임스페이스 |

여기서 `--dest-server`에 `https://kubernetes.default.svc`를 사용한 점에 주목하세요. 이는 Argo CD가 실행 중인 바로 그 클러스터에 배포하는 **내부(internal) 배포**의 주소 규칙입니다. 외부 클러스터에 배포하려면 먼저 `argocd cluster add`로 등록한 뒤, 그 클러스터의 API 서버 주소를 `--dest-server`에 지정하면 됩니다.

### 생성된 Application의 의미

이 명령이 실행되면 Argo CD에 `guestbook`이라는 Application이 등록됩니다. **Application은 Kubernetes의 CRD**이므로, 이 명령은 사실상 클러스터에 하나의 Application 리소스를 만드는 것입니다.

다만 이 시점에서 한 가지 중요한 점이 있습니다. Application을 **생성(create)** 하는 것과 실제로 클러스터에 리소스를 **배포(sync)** 하는 것은 별개의 단계입니다. Application은 생성되었지만 아직 배포되지 않아 Kubernetes 리소스가 만들어지지 않은 상태 — 즉 `OutOfSync` 상태 — 로 남아 있습니다.

이 `OutOfSync` 상태의 정확한 의미를 읽고, `argocd app sync`로 실제 배포를 수행해 Health/Sync 상태를 확인하는 방법은 「Application 동기화(Sync) 및 상태 확인」에서 이어 다룹니다. 또한 동일한 guestbook Application을 웹 UI의 "New App" 버튼으로 만드는 방법은 바로 다음 「첫 번째 Application 생성 — UI 방식」에서 단계별로 살펴봅니다.
</https:>

## 11. 첫 번째 Application 생성 — UI 방식

「첫 번째 Application 생성 — CLI 방식」에서 `argocd app create`로 만든 것과 **완전히 동일한 guestbook Application**을, 이번에는 웹 UI의 "New App" 버튼을 통해 만들어 봅니다. CLI에 익숙하지 않거나, 어떤 필드가 어디에 매핑되는지 시각적으로 확인하고 싶을 때 UI 방식이 유용합니다.

> **사전 준비.** 이 절차는 「Argo CD CLI 설치 및 UI 접근」에서 마련한 접근 경로(LoadBalancer·Ingress·Port Forwarding 중 하나)를 통해 UI에 접속할 수 있고, 「CLI로 로그인하고 클러스터 등록하기」 단계에서 사용한 `admin` 자격증명을 알고 있다는 전제 위에서 진행됩니다.

### 1단계 — UI에 로그인하고 "New App" 열기

브라우저로 Argo CD 외부 UI에 접속합니다. IP/호스트네임으로 방문해, 「Argo CD CLI 설치 및 UI 접근」에서 조회한 자격증명(사용자명 `admin`)으로 로그인합니다. 로컬 환경에서 실행 중이라면 Argo CD 공식 문서의 "Try Argo CD Locally" 안내를 참고할 수 있습니다.

로그인 후, 아래와 같이 **`+ New App`** 버튼을 클릭합니다.

![+ new app button](https://argo-cd.readthedocs.io/en/stable/assets/new-app.png)

### 2단계 — 애플리케이션 일반 정보 입력

새 애플리케이션 생성 패널에서 다음 값을 채웁니다.

- 앱 이름(name): `guestbook`
- 프로젝트(project): `default`
- 동기화 정책(sync policy): `Manual` 그대로 둡니다.

![app information](https://argo-cd.readthedocs.io/en/stable/assets/app-ui-information.png)

> 여기서 동기화 정책을 `Manual`로 두는 이유는, CLI 방식에서와 마찬가지로 **Application 생성과 실제 배포(sync)를 분리**하기 위해서입니다. 즉 이 단계에서는 Application 리소스만 정의하고, 실제 클러스터 배포는 다음 「Application 동기화(Sync) 및 상태 확인」에서 수행합니다.

### 3단계 — 소스(Source) 레포지토리 연결

guestbook 예제 레포지토리를 Argo CD에 연결합니다. 다음 값을 입력합니다.

- repository URL: `https://github.com/argoproj/argocd-example-apps.git`
- revision: `HEAD` 그대로 둡니다.
- path: `guestbook`

![connect repo](https://argo-cd.readthedocs.io/en/stable/assets/connect-repo.png)

이 세 값은 「핵심 아키텍처 이해」에서 설명한 Repository Server의 입력(repository URL, revision, application path)에 그대로 대응하며, CLI 방식의 `--repo`, `--path` 와 동일한 역할을 합니다.

### 4단계 — 대상(Destination) 지정

배포 대상 클러스터와 네임스페이스를 지정합니다.

- cluster URL: `https://kubernetes.default.svc` (또는 cluster name으로 `in-cluster`)
- namespace: `default`

![destination](https://argo-cd.readthedocs.io/en/stable/assets/destination.png)

여기서 `https://kubernetes.default.svc`는 「CLI로 로그인하고 클러스터 등록하기」에서 다룬 **내부(internal) 배포** 주소 규칙입니다. 즉 Argo CD가 실행 중인 바로 그 클러스터로 배포합니다.

### 5단계 — Create 클릭

위 정보를 모두 채운 뒤, UI 상단의 **`Create`** 버튼을 클릭하면 `guestbook` 애플리케이션이 생성됩니다.

![create app](https://argo-cd.readthedocs.io/en/stable/assets/create-app.png)

### UI 입력값과 CLI 플래그 대응표

UI에서 채운 각 값이 「첫 번째 Application 생성 — CLI 방식」의 어떤 플래그에 해당하는지 정리하면 다음과 같습니다.

| UI 입력 항목 | 입력 값 | 대응되는 CLI 플래그 |
|--------------|---------|---------------------|
| Application Name | `guestbook` | 첫 번째 인자(`guestbook`) |
| Project | `default` | — |
| Sync Policy | `Manual` | — |
| Repository URL | `https://github.com/argoproj/argocd-example-apps.git` | `--repo` |
| Revision | `HEAD` | — |
| Path | `guestbook` | `--path` |
| Cluster URL | `https://kubernetes.default.svc` | `--dest-server` |
| Namespace | `default` | `--dest-namespace` |

### 생성 후 상태

CLI 방식과 마찬가지로, **이 단계까지는 Application이 "생성"되었을 뿐 아직 클러스터에 리소스가 배포된 것은 아닙니다.** 동기화 정책을 `Manual`로 두었기 때문에, Application은 아직 동기화되지 않은 `OutOfSync` 상태로 남아 있습니다.

이 `OutOfSync` 상태의 정확한 의미를 읽고, UI의 **Sync** 버튼(또는 CLI의 `argocd app sync`)으로 실제 배포를 수행하여 Health/Sync 상태를 확인하는 방법은 바로 다음 「Application 동기화(Sync) 및 상태 확인」에서 이어 다룹니다.

## 12. Application 동기화(Sync) 및 상태 확인

「첫 번째 Application 생성 — CLI 방식」과 「첫 번째 Application 생성 — UI 방식」에서 guestbook Application을 만들었지만, 두 경우 모두 **생성만 했을 뿐 아직 클러스터에 리소스가 배포되지 않았습니다.** 이 상태가 바로 `OutOfSync`입니다. 이 섹션에서는 `OutOfSync`의 의미를 실제 출력으로 확인하고, CLI(`argocd app sync`)와 UI의 Sync 버튼으로 애플리케이션을 배포한 뒤 Health/Sync 상태를 읽는 방법을 다룹니다.

### `OutOfSync`란 무엇인가

「Argo CD 핵심 개념 사전」에서 정리했듯, **Sync status**는 "라이브 상태가 타깃 상태와 일치하는가?"라는 질문에 대한 답입니다. guestbook Application은 막 생성되었을 뿐 아직 배포되지 않았고 어떤 Kubernetes 리소스도 생성되지 않았으므로, Git이 말하는 타깃 상태(Deployment, Service 등)가 클러스터에 존재하지 않습니다. 따라서 두 상태가 일치하지 않아 `OutOfSync`로 표시됩니다.

### 1단계 — CLI로 현재 상태 조회

먼저 `argocd app get`으로 생성된 애플리케이션의 상태를 확인합니다.

```bash
argocd app get guestbook
```

출력은 다음과 같습니다.

```text
Name:               guestbook
Server:             https://kubernetes.default.svc
Namespace:          default
URL:                https://10.97.164.88/applications/guestbook
Repo:               https://github.com/argoproj/argocd-example-apps.git
Target:
Path:               guestbook
SyncPolicy:         <none>
SyncStatus:         OutOfSync from  (1ff8a67)
HealthStatus:       Missing

GROUP  KIND        NAMESPACE  NAME           STATUS     HEALTH
apps   Deployment  default    guestbook-ui   OutOfSync  Missing
       Service     default    guestbook-ui   OutOfSync  Missing
```

여기서 두 가지를 함께 읽어야 합니다.

- **SyncStatus: `OutOfSync`** — 애플리케이션이 아직 배포되지 않았고, 어떤 Kubernetes 리소스도 생성되지 않았기 때문에 초기 상태가 `OutOfSync`입니다.
- **HealthStatus: `Missing`** — 리소스(Deployment, Service)가 아직 클러스터에 존재하지 않아 헬스가 `Missing`으로 표시됩니다.

「Argo CD 핵심 개념 사전」에서 강조했듯 **Sync status와 Health는 서로 다른 축**입니다. 위 출력은 "Git과 클러스터가 다르다(OutOfSync)"는 것과 "리소스가 아직 없다(Missing)"는 것을 각각 별개로 보여 줍니다.

### 2단계 — CLI로 Sync(배포)하기

`OutOfSync` 상태의 애플리케이션을 타깃 상태로 옮기려면, 즉 실제로 배포하려면 다음 명령을 실행합니다.

```bash
argocd app sync guestbook
```

이 명령은 **레포지토리에서 매니페스트를 가져와 `kubectl apply`를 수행**합니다. 「핵심 아키텍처 이해」의 흐름과 연결하면, Repository Server가 지정된 revision·path의 매니페스트를 렌더링해 반환하고, Application Controller가 이를 클러스터에 적용하는 과정입니다.

Sync가 완료되면 guestbook 앱이 실행되며, 이제 그 리소스 컴포넌트·로그·이벤트와 평가된 헬스 상태를 확인할 수 있습니다. 배포 후 다시 `argocd app get guestbook`을 실행하면 SyncStatus와 HealthStatus가 변경된 것을 볼 수 있습니다.

### UI로 Sync하기

UI 방식으로 만든 경우(또는 UI에서 배포하고 싶은 경우), 동일한 작업을 버튼으로 수행할 수 있습니다.

1. **Applications** 페이지에서 guestbook 애플리케이션의 **Sync** 버튼을 클릭합니다.

![guestbook app](https://argo-cd.readthedocs.io/en/stable/assets/guestbook-app.png)

2. 패널이 열리면 **Synchronize** 버튼을 클릭합니다.

3. guestbook 애플리케이션을 클릭하면 더 자세한 내용을 확인할 수 있습니다.

![view app](https://argo-cd.readthedocs.io/en/stable/assets/guestbook-tree.png)

### CLI vs. UI 동기화 정리

| 항목 | CLI | UI |
|------|-----|-----|
| 상태 조회 | `argocd app get guestbook` | Applications 페이지에서 앱 상세 보기 |
| 동기화 실행 | `argocd app sync guestbook` | **Sync** 버튼 → **Synchronize** 버튼 |
| 동작 내용 | 레포지토리에서 매니페스트를 가져와 `kubectl apply` 수행 | 동일 (API Server를 통해 sync 작업 호출) |
| 결과 확인 | 출력의 `SyncStatus` / `HealthStatus` | 애플리케이션 상세 화면 |

### 상태를 읽을 때 기억할 점

- **Sync는 생성과 별개의 단계입니다.** 앞선 두 섹션에서 동기화 정책을 `Manual`(CLI에서는 `SyncPolicy: <none>`)로 두었기 때문에, 명시적으로 sync를 실행하기 전까지는 `OutOfSync` 상태로 남아 있습니다. 이 수동 단계를 자동화하는 자동 동기화 정책은 별도의 자동 동기화 정책 문서에서 다룹니다.
- **두 상태축을 분리해서 판단하세요.** `SyncStatus`는 Git과 클러스터의 일치 여부를, `HealthStatus`는 애플리케이션이 제대로 동작하며 요청을 처리할 수 있는지를 나타냅니다. 두 값은 독립적으로 변할 수 있습니다.

이제 첫 번째 Application을 생성하고 배포해 상태를 확인하는 전체 GitOps 루프를 한 바퀴 돌아 보았습니다. 다음 섹션부터는 UI·SSO·멀티테넌시가 필요 없을 때 선택하는 **Core(헤드리스) 모드** 사용법을 살펴봅니다.
</none></none>

## 13. Argo CD Core(헤드리스 모드) 사용하기

지금까지 「Application 동기화(Sync) 및 상태 확인」까지의 실습은 모두 **API Server와 Web UI**를 전제로 진행했습니다. 하지만 「핵심 아키텍처 이해」와 「Argo CD 설치 방법 선택」에서 예고했듯, Argo CD는 컴포넌트 기반 아키텍처 덕분에 **더 미니멀한 설치 — Argo CD Core** 를 선택할 수 있습니다. Core는 Argo CD를 **헤드리스(headless) 모드**로 실행하면서도, Git 저장소에서 원하는 상태를 가져와 Kubernetes에 적용하는 **완전한 기능의 GitOps 엔진**을 그대로 유지합니다.

### Core를 선택하는 이유

Argo CD Core는 **멀티테넌시 기능이 필요 없고 Argo CD를 단독으로 사용하는 클러스터 관리자**에게 가장 적합합니다. 공식 문서가 드는 대표적인 사용 시나리오는 다음과 같습니다.

- 클러스터 관리자로서 **Kubernetes RBAC에만 의존**하고 싶을 때.
- DevOps 엔지니어로서 새로운 API를 배우거나 별도 CLI에 의존하지 않고 **Kubernetes API만으로** 배포를 자동화하고 싶을 때.
- 클러스터 관리자로서 개발자에게 **Argo CD UI나 CLI를 제공하고 싶지 않을 때**.

대신 Core 설치에서는 다음 기능 그룹을 **사용할 수 없습니다.**

| 사용 불가 | 부분적으로만 사용 가능 |
|-----------|------------------------|
| Argo CD RBAC 모델 | Argo CD Web UI |
| Argo CD API | Argo CD CLI |
| Argo CD Notification Controller | 멀티테넌시 (git push 권한 기반의 엄격한 GitOps 한정) |
| OIDC 기반 인증 | — |

즉 Core 모드의 멀티테넌시는 Argo CD 자체의 RBAC가 아니라 **순수하게 Git push 권한**에 의해 결정됩니다.

### Core의 아키텍처

Core 설치는 「Argo CD 설치 방법 선택」에서 정리했듯 **API Server와 UI를 포함하지 않고, 각 컴포넌트의 경량(non-HA) 버전**만 설치합니다. 더 적은 컴포넌트로도 핵심 GitOps 기능은 그대로 동작합니다.

![Argo CD Core](https://argo-cd.readthedocs.io/en/stable/assets/argocd-core-components.png)

> **참고: Redis는 여전히 포함됩니다.** Argo CD 컨트롤러는 Redis 없이도 실행될 수 있지만 권장되지 않습니다. 컨트롤러는 Kube API와 Git에 대한 부하를 줄이는 중요한 캐싱 메커니즘으로 Redis를 사용하므로, Core 설치에도 Redis가 함께 포함됩니다.

### Core 설치하기

Core는 **필요한 모든 리소스를 담은 단일 매니페스트 파일** `core-install.yaml` 하나를 적용해 설치합니다. 「Argo CD 설치 실습」의 표준 설치와 동일하게 `--server-side --force-conflicts` 플래그를 사용합니다(이 플래그가 필요한 이유는 CRD 크기 제한 때문이며, 해당 섹션에서 자세히 설명했습니다).

```bash
export ARGOCD_VERSION=<원하는 Argo CD 릴리스 버전 (예: v2.7.0)>
kubectl create namespace argocd
kubectl apply -n argocd --server-side --force-conflicts \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/$ARGOCD_VERSION/manifests/core-install.yaml
```

> **버전 고정(pinned version).** 위 예시는 `ARGOCD_VERSION` 환경 변수로 특정 릴리스 버전을 지정합니다. 「Argo CD 설치 실습」에서 다룬 것과 같은 맥락으로, 적용 시점에 따라 내용이 달라지지 않도록 고정된 버전을 사용하는 것이 좋습니다.

### Core 사용하기: GitOps가 기본 인터페이스

Core가 설치되면 사용자는 **GitOps에 의존해 Argo CD와 상호작용**합니다. 이때 사용할 수 있는 Kubernetes 리소스는 「Argo CD 핵심 개념 사전」에서 강조한 대로 **`Application`과 `ApplicationSet` CRD**입니다. 이 두 CRD를 통해 애플리케이션을 배포하고 관리합니다.

#### `argocd login --core`로 CLI 사용하기

Core 모드에서도 **Argo CD CLI를 사용할 수 있습니다.** 다만 동작 방식이 다릅니다. CLI는 명령을 처리하기 위해 **로컬 API Server 프로세스를 띄우고(spawn)**, 명령이 끝나면 그 프로세스도 함께 종료됩니다. 이 과정은 추가 명령 없이 사용자에게 투명하게 일어납니다.

CLI를 Core 모드로 사용하려면 `login` 서브커맨드에 **`--core` 플래그**를 전달해야 합니다. 이 `--core` 플래그가 바로 CLI와 Web UI 요청을 처리하는 로컬 Argo CD API Server 프로세스를 띄우는 역할을 합니다.

```bash
# 현재 kube 컨텍스트를 argocd 네임스페이스로 변경
kubectl config set-context --current --namespace=argocd
argocd login --core
```

여기서 중요한 권한 모델을 기억해야 합니다. **Core 모드는 오직 Kubernetes RBAC에만 의존**하므로, CLI를 실행하는 사용자(또는 프로세스)는 주어진 명령을 수행하기 위해 **Argo CD 네임스페이스의 `Application`·`ApplicationSet` 리소스에 대한 적절한 권한**을 가지고 있어야 합니다.

> 「Argo CD 설치 실습」과 「CLI로 로그인하고 클러스터 등록하기」에서 안내한 대로, UI·SSO·멀티클러스터 기능이 필요 없다면 `argocd login --core`로 CLI 접근을 구성하고 접근·로그인·클러스터 등록 단계를 건너뛸 수 있습니다.

#### 로컬 대시보드 실행하기

UI 방식으로 상호작용하는 것을 선호한다면, **Web UI를 로컬에서 실행**할 수 있습니다. 다음 명령으로 시작합니다.

```bash
argocd admin dashboard -n argocd
```

실행 후 Argo CD Web UI는 다음 주소에서 접근할 수 있습니다.

```text
http://localhost:8080
```

### Multi-Tenant 모드와의 차이 요약

「Application 동기화(Sync) 및 상태 확인」까지 다룬 Multi-Tenant 흐름과 Core 모드의 핵심 차이를 정리하면 다음과 같습니다.

| 항목 | Multi-Tenant | Core (헤드리스) |
|------|--------------|-----------------|
| 설치 매니페스트 | `install.yaml` / `ha/install.yaml` | `core-install.yaml` (단일 파일) |
| API Server / UI | 상시 실행 | 미포함 (CLI가 로컬 API Server를 일시 기동) |
| 인증·인가 | Argo CD RBAC + SSO(OIDC 등) | Kubernetes RBAC만 사용 |
| CLI 로그인 | `argocd login <server>` | `argocd login --core` |
| UI 접근 | LoadBalancer·Ingress·Port Forwarding | `argocd admin dashboard` (로컬 `http://localhost:8080`) |
| 주 상호작용 수단 | UI·CLI·API | GitOps (`Application`·`ApplicationSet` CRD) |

이처럼 Core는 "Argo CD의 GitOps 엔진만 가볍게 운영하고 싶다"는 요구에 정확히 들어맞는 설치 유형입니다. 다음으로는 Multi-Tenant든 Core든 공통으로 적용되는, Argo CD가 매니페스트를 어떤 소스 유형으로 받아들이는지를 「지원되는 Config Management 도구」에서 살펴봅니다.
</server>

## 14. 지원되는 Config Management 도구

「Argo CD란 무엇인가?」와 「핵심 아키텍처 이해」에서 반복해 짚었듯, Argo CD가 Git에서 가져오는 매니페스트는 **하나의 고정된 형식이 아니라 여러 소스 유형**으로 표현될 수 있습니다. 그리고 「핵심 아키텍처 이해」에서 본 것처럼, 이 소스를 실제 Kubernetes 매니페스트로 **렌더링하는 주체는 Repository Server**입니다. 이 섹션에서는 Argo CD가 공식적으로 지원하는 매니페스트 소스 유형들을 정리합니다.

### 공식적으로 지원되는 소스 유형

Argo CD 공식 문서(Overview의 "How it works")는 Kubernetes 매니페스트를 다음과 같은 방식으로 지정할 수 있다고 명시합니다.

- [Kustomize](https://kustomize.io) 애플리케이션
- [Helm](https://helm.sh) 차트
- [Jsonnet](https://jsonnet.org) 파일
- 플레인(plain) YAML/JSON 매니페스트 디렉터리
- **config management plugin**으로 구성된 임의의 커스텀 config management 도구

이는 「Argo CD 핵심 개념 사전」에서 정의한 용어와 직접 연결됩니다. 거기서 **Tool**(= **Configuration management tool**)은 "파일 디렉터리로부터 매니페스트를 생성하는 도구"이고, **Application source type**은 "애플리케이션을 빌드하는 데 어떤 Tool이 사용되는지"를 가리키며, **Configuration management plugin**은 "커스텀 도구"를 의미한다고 정리했습니다.

| 소스 유형 | 용어상의 분류 | 설명 (출처 기준) |
|-----------|---------------|------------------|
| Kustomize | Tool / Configuration management tool | Kustomize 애플리케이션. 핵심 개념 사전에서 Tool의 대표 예시로 언급됩니다. |
| Helm | Tool / Configuration management tool | Helm 차트. Git 내에서 Helm 파라미터를 오버라이드할 수 있습니다(Features 목록의 "Parameter overrides for overriding helm parameters in Git"). |
| Jsonnet | Tool / Configuration management tool | Jsonnet 파일. |
| 플레인 YAML/JSON 디렉터리 | Directory | 별도의 템플릿 도구 없이 매니페스트가 담긴 디렉터리. |
| Config Management Plugin (CMP) | Configuration management plugin | 임의의 커스텀 config management 도구. |

> 위 표의 "Tool", "Configuration management tool", "Configuration management plugin" 구분은 모두 「Argo CD 핵심 개념 사전」의 정의를 그대로 따른 것입니다. 앞의 두 용어는 동일한 의미이며, plugin은 커스텀 도구를 가리킨다는 점을 다시 떠올려 두면 좋습니다.

### 각 소스 유형이 Application과 만나는 지점

「핵심 아키텍처 이해」에서 정리했듯 Repository Server는 다음 입력을 받아 매니페스트를 생성·반환합니다.

- repository URL
- revision (commit, tag, branch)
- application path
- template specific settings: parameters, helm `values.yaml`

여기서 마지막 항목인 **"template specific settings: parameters, helm values.yaml"** 이 바로 위에서 본 도구별 설정이 들어가는 자리입니다. 즉 Helm 차트라면 `values.yaml`과 파라미터가, Kustomize·Jsonnet이라면 각 도구가 요구하는 설정이 이 입력을 통해 전달됩니다.

이는 「첫 번째 Application 생성 — CLI 방식」에서 사용한 `--repo`, `--path`, 그리고 「첫 번째 Application 생성 — UI 방식」에서 입력한 repository URL / revision / path와도 대응합니다. 다시 말해, **어떤 소스 유형을 쓰든 Application을 정의하는 큰 틀(레포·리비전·경로)은 동일**하고, 도구별 차이는 그 위에 얹히는 설정에서 나타납니다.

### 소스 유형은 어떻게 결정되는가 (Tool Detection)

guestbook 예제처럼 플레인 디렉터리를 사용할 때는 별도 설정 없이도 동작했습니다. Argo CD 공식 문서에는 소스가 어떤 도구로 만들어졌는지에 관한 **Tool Detection**(도구 감지) 페이지가 별도로 마련되어 있습니다. 구체적인 감지 방식과 명시적으로 도구를 지정하는 방법은 Argo CD 공식 문서의 다음 User Guide 페이지들에서 다룹니다.

- "Tools" (Application Sources) — 지원 도구 전반에 대한 개요
- "Tool Detection" — 소스 유형 감지에 대한 설명
- "Kustomize" / "Helm" / "Jsonnet" / "Directory" — 도구별 상세 사용법
- "Plugins" / "Config Management Plugins" — 커스텀 도구(CMP) 구성 방법

> 이 가이드는 입문 범위를 다루므로, 각 도구의 세부 옵션까지는 들어가지 않습니다. 필요할 때 위 공식 문서를 참고하세요. 또한 「사전 지식 및 준비 사항」에서 안내했듯, Kustomize·Helm은 "애플리케이션을 템플릿화할 계획이 있다면" 미리 익혀 두면 도움이 되는 선택 사항입니다.

### 정리

Argo CD가 다중 config management/템플릿 도구(Kustomize, Helm, Jsonnet, plain-YAML)를 지원한다는 점은 공식 Features 목록에도 명시된 이점입니다. Kustomize·Helm·Jsonnet 같은 표준 도구를 쓰든, 플레인 디렉터리를 쓰든, 혹은 어느 쪽에도 맞지 않아 Config Management Plugin으로 커스텀 도구를 연결하든, 최종적으로는 모두 Repository Server에서 동일한 입력 모델을 거쳐 Kubernetes 매니페스트로 렌더링되고, 그 결과가 클러스터에 반영됩니다.

한편 Argo CD의 일부 기능은 안정(stable) 단계가 아닌 Alpha/Beta로 분류됩니다. 어떤 기능이 실험적이며 프로덕션에서 주의해야 하는지는 이어지는 「Feature Maturity 및 안정성 주의사항」에서 정리합니다.

## 15. Feature Maturity 및 안정성 주의사항

「Argo CD란 무엇인가?」 끝부분과 「지원되는 Config Management 도구」 마지막에서 예고했듯, Argo CD는 커뮤니티에 의해 활발히 개발되고 있으며 **모든 기능이 동일한 안정성 단계에 있는 것은 아닙니다.** 일부 기능은 `Stable`이 아니라 `Alpha` 또는 `Beta` 상태로 표시됩니다. 이 섹션에서는 그 상태 표시가 무엇을 의미하는지, 그리고 프로덕션에서 어떤 점을 주의해야 하는지를 공식 「Feature Maturity」 문서를 근거로 정리합니다.

### Alpha/Beta 기능을 쓸 때의 위험

Argo CD 기능에는 안정성과 성숙도를 나타내는 상태(status)가 표시될 수 있습니다. 공식 문서는 비(非)안정 기능에 대해 다음과 같이 **명시적으로 경고**합니다.

> **Caution — Alpha/Beta 기능 사용 위험**
> - Alpha 및 Beta 기능은 **하위 호환성(backward compatibility)을 보장하지 않으며**, 향후 릴리스에서 **호환성을 깨는 변경(breaking changes)** 의 대상이 됩니다.
> - Argo 사용자는 이러한 기능을 **프로덕션 환경에서 의존하지 않을 것이 강력히 권장됩니다.** 특히 **Argo CD 업그레이드를 직접 제어할 수 없는 경우** 더욱 그렇습니다.
> - 더 나아가, **Alpha 기능의 제거(removal)** 는 Argo CD 업그레이드 후 여러분의 리소스를 **예측 불가능한 상태로 바꿔 놓을 수 있습니다.**

따라서 운영상 두 가지 실천 사항이 따라옵니다. 공식 문서가 직접 권하는 것은 다음과 같습니다.

1. **사용 중인 기능을 문서화**해 두세요. 어떤 Alpha/Beta 기능에 의존하고 있는지 명확히 기록해야 합니다.
2. **업그레이드 전 릴리스 노트를 반드시 검토**하세요. 호환성을 깨는 변경이나 기능 제거가 포함되어 있을 수 있습니다.

요약하면, **`Stable`로 표시되지 않은 기능은 "동작하지만 변경·제거될 수 있는" 실험적 기능**으로 다루어야 하며, 통제 가능한 업그레이드 환경이 아니라면 프로덕션 의존을 피하는 것이 안전합니다.

### Alpha와 Beta는 어떻게 다른가

위 경고는 Alpha와 Beta를 함께 묶어 "비안정 기능"으로 다루지만, 둘은 별도의 상태로 표시됩니다. 특히 **Alpha 기능의 "제거"가 리소스를 예측 불가능한 상태로 만들 수 있다**는 점이 별도로 강조된 것에 주목하세요. 즉 Alpha 기능을 쓸 때는 변경뿐 아니라 **기능 자체가 사라질 가능성**까지 고려해야 합니다.

> 참고: 각 상태(status)의 정확한 정의 자체는 Argo CD 문서가 아니라 argoproj 커뮤니티의 feature-status 규약을 따릅니다. 이 가이드는 입문 범위이므로, 여기서는 공식 「Feature Maturity」 문서가 직접 경고하는 운영상의 함의에 집중합니다.

### 비안정 기능 목록 (출처 기준)

아래 표는 공식 「Feature Maturity」 문서의 Overview에 정리된, **현재 비안정(non-stable) 상태인 기능들**입니다. "Introduced"는 해당 기능이 도입된 Argo CD 버전입니다.

| 기능 | 도입 버전 | 상태 |
|------|-----------|------|
| AppSet Progressive Syncs | v2.6.0 | Beta |
| Proxy Extensions | v2.7.0 | Beta |
| Skip Application Reconcile | v2.7.0 | Alpha |
| AppSets in any Namespace | v2.8.0 | Beta |
| Cluster Sharding: round-robin | v2.8.0 | Alpha |
| Dynamic Cluster Distribution | v2.9.0 | Alpha |
| Cluster Sharding: consistent-hashing | v2.12.0 | Alpha |
| Service Account Impersonation | v2.13.0 | Alpha |
| Source Hydrator | v2.14.0 | Alpha |

이 가이드 도입부에서 언급한 **Progressive Syncs**(ApplicationSet의 점진적 동기화)는 `Beta`, **Dynamic Cluster Distribution**(컨트롤러의 동적 클러스터 분배)과 **Skip Reconcile**(특정 Application의 reconcile 건너뛰기)은 `Alpha`임을 위 표에서 확인할 수 있습니다.

### 어떤 설정이 "불안정 설정"인가

「Feature Maturity」 문서는 위 기능들이 **어떤 CRD 필드나 설정값을 통해 활성화되는지**까지 "Unstable Configurations"로 구체적으로 명시합니다. 즉, 특정 매니페스트 필드나 ConfigMap/Deployment 설정을 켜는 순간 해당 비안정 기능에 의존하게 된다는 뜻입니다. 대표적인 예를 출처에서 그대로 옮기면 다음과 같습니다.

| 기능 | 활성화 지점(리소스) | 속성 / 변수 | 상태 |
|------|---------------------|-------------|------|
| Skip Application Reconcile | Application CRD | `metadata.annotations[argocd.argoproj.io/skip-reconcile]` | Alpha |
| Service Account Impersonation | AppProject CRD | `spec.destinationServiceAccounts.*` | Alpha |
| AppSet Progressive Syncs | ApplicationSet CRD | `spec.strategy.*`, `status.applicationStatus.*` | Beta |
| AppSet Progressive Syncs | `ConfigMap/argocd-cmd-params-cm` | `applicationsetcontroller.enable.progressive.syncs` | Beta |
| AppSet Progressive Syncs | `Deployment/argocd-applicationset-controller` | `ARGOCD_APPLICATIONSET_CONTROLLER_ENABLE_PROGRESSIVE_SYNCS` | Beta |
| Proxy Extensions | `ConfigMap/argocd-cmd-params-cm` | `server.enable.proxy.extension` | Beta |
| Dynamic Cluster Distribution | `Deployment/argocd-application-controller` | `ARGOCD_ENABLE_DYNAMIC_CLUSTER_DISTRIBUTION` | Alpha |
| Cluster Sharding: round-robin | `ConfigMap/argocd-cmd-params-cm` | `controller.sharding.algorithm: round-robin` | Alpha |
| Cluster Sharding: consistent-hashing | `ConfigMap/argocd-cmd-params-cm` | `controller.sharding.algorithm: consistent-hashing` | Alpha |
| Service Account Impersonation | `ConfigMap/argocd-cm` | `application.sync.impersonation.enabled` | Alpha |

(위 표는 출처의 항목 일부를 요약한 것입니다. 전체 목록과 추가 변수는 공식 「Feature Maturity」 문서를 참고하세요.)

이 표가 실무에서 중요한 이유는, **예컨대 Application 매니페스트에 `argocd.argoproj.io/skip-reconcile` 어노테이션을 다는 행위 자체가 Alpha 기능에 의존하는 것**이기 때문입니다. 이런 필드를 쓰기로 결정했다면, 위에서 정리한 두 가지 실천 사항(사용 기능 문서화·업그레이드 전 릴리스 노트 검토)을 함께 적용해야 합니다.

### 안정적인 기능 vs. 실험적 기능 구분하기

지금까지 이 가이드에서 **실습으로 다룬 핵심 흐름** — 표준 설치, `argocd app create`, `argocd app sync`, 수동 Sync, Health/Sync 상태 확인, Core 설치(`core-install.yaml`)와 `argocd login --core` — 는 위의 비안정 목록에 등장하지 않습니다. 즉 입문자가 따라간 기본 GitOps 루프는 실험적 기능에 의존하지 않습니다.

반면, 위 표에 나열된 기능들(Progressive Syncs, Dynamic Cluster Distribution, Skip Reconcile, Cluster Sharding, Service Account Impersonation, Source Hydrator, AppSets/Proxy Extensions 관련 설정 등)은 **명시적으로 켜야 하는 비안정 기능**입니다. 따라서 구분의 기준은 단순합니다.

- **별도 설정 없이 기본 동작에 포함되는 기능** → 안정적으로 사용 가능.
- **공식 「Feature Maturity」 문서의 비안정 목록에 등장하는 기능** → `Alpha`/`Beta`로 간주하고, 프로덕션에서는 통제 가능한 업그레이드 환경과 문서화·릴리스 노트 검토를 전제로만 사용.

이러한 안정성 고려는 프로덕션 운영의 일부일 뿐입니다. 자동 동기화 정책, 초기 admin 비밀번호 보안, HA 설치 권장, 버전 고정(pinned version) 등 운영 전반의 권장 사항은 마지막 「모범 사례 및 다음 단계」에서 종합해 정리합니다.

## 16. 모범 사례 및 다음 단계

이 가이드에서 우리는 Argo CD의 개념부터 설치, 첫 Application의 생성·동기화, Core 모드, 지원 도구, 안정성 단계까지 한 바퀴를 돌았습니다. 마지막으로, 그 과정에서 곳곳에 흩어져 있던 **운영상의 핵심 권장 사항**을 한곳에 모아 정리합니다. 모든 항목은 앞선 섹션들에서 이미 근거를 제시한 것이며, 여기서는 "프로덕션으로 가기 전에 반드시 체크할 목록"으로 다시 묶습니다.

### 핵심 권장 사항 요약

| 권장 사항 | 무엇을 / 왜 | 관련 섹션 |
|-----------|-------------|-----------|
| **초기 admin 비밀번호 보안** | 비밀번호 변경 후 `argocd-initial-admin-secret` Secret을 삭제 | 「Argo CD CLI 설치 및 UI 접근」 |
| **프로덕션은 HA 설치** | 평가·데모·테스트는 Non-HA, 프로덕션은 HA 번들(`ha/install.yaml`) 권장 | 「Argo CD 설치 방법 선택」 |
| **버전 고정(pinned version)** | `stable` 대신 `v3.2.0` 같은 고정 버전 사용 | 「Argo CD 설치 실습」 |
| **자동 동기화 정책 고려** | 수동 Sync 단계를 정책으로 자동화 | 「Application 동기화(Sync) 및 상태 확인」 |
| **비안정 기능 신중 사용** | Alpha/Beta 기능은 통제 가능한 업그레이드 환경에서만 | 「Feature Maturity 및 안정성 주의사항」 |

### 1. 초기 admin 비밀번호를 반드시 정리하세요

「Argo CD CLI 설치 및 UI 접근」에서 보았듯, `admin` 계정의 초기 비밀번호는 자동 생성되어 `argocd-initial-admin-secret` Secret의 `password` 필드에 **평문(clear text)** 으로 저장됩니다. 공식 가이드는 명확히 경고합니다.

> 비밀번호를 변경한 후에는 Argo CD 네임스페이스에서 `argocd-initial-admin-secret`을 **삭제해야 합니다.** 이 Secret은 초기 생성 비밀번호를 평문으로 보관하는 것 외에 다른 용도가 없으며 언제든 안전하게 삭제할 수 있습니다. 새 admin 비밀번호를 다시 생성해야 할 경우 Argo CD가 필요에 따라 재생성합니다.

따라서 운영 순서는 다음과 같습니다.

```bash
# 1) 로그인 후 비밀번호 변경
argocd account update-password

# 2) 더 이상 필요 없는 초기 비밀번호 Secret 삭제
kubectl delete secret argocd-initial-admin-secret -n argocd
```

### 2. 프로덕션에는 HA 설치를 사용하세요

「Argo CD 설치 방법 선택」에서 정리한 대로, 설치 매니페스트에는 Non-HA와 HA 두 종류가 있습니다.

- **Non-HA 설치는 프로덕션 용도로 권장되지 않으며**, 일반적으로 평가·데모·테스트 기간에 사용됩니다.
- **HA 설치가 프로덕션 용도로 권장됩니다.** HA 번들은 동일한 컴포넌트를 포함하되 고가용성과 복원력(resiliency)을 위해 튜닝되어, 지원되는 컴포넌트에 대해 다중 복제본(replica)을 사용합니다.

즉 이 가이드의 실습은 `install.yaml`(Non-HA)로 진행했지만, 실제 운영 클러스터에서는 `ha/install.yaml`(또는 `ha/namespace-install.yaml`)을 사용하는 것이 출발점입니다.

### 3. 버전을 고정하세요 (pinned version)

「Argo CD 설치 실습」과 「Argo CD Core(헤드리스 모드) 사용하기」에서 강조했듯, `stable` 브랜치를 가리키는 URL은 적용 시점에 따라 내용이 달라질 수 있습니다. 공식 문서는 프로덕션에서 `v3.2.0`처럼 **고정된 버전을 사용할 것을 권장**합니다.

```bash
# stable 대신 명시적 버전을 가리키는 예
kubectl apply -n argocd --server-side --force-conflicts \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/v3.2.0/manifests/install.yaml
```

이렇게 하면 설치·업그레이드 시점이 명확해지고, 「Feature Maturity 및 안정성 주의사항」에서 다룬 "업그레이드 전 릴리스 노트 검토" 실천과도 자연스럽게 맞물립니다.

### 4. 수동 Sync를 자동 동기화 정책으로

「Application 동기화(Sync) 및 상태 확인」에서 우리는 동기화 정책을 `Manual`로 두고 명시적으로 `argocd app sync`를 실행했습니다. `SyncPolicy`가 설정되지 않은 동안에는 Git이 바뀌어도 `OutOfSync` 상태로 남아 있을 뿐 자동 배포되지 않습니다. 이 수동 단계를 자동화하려면 **자동 동기화 정책(Automated Sync Policy)** 을 구성합니다. 구체적인 설정은 Argo CD 공식 User Guide의 "Automated Sync Policy" 문서에서 다룹니다.

### 5. 비안정 기능은 통제된 환경에서만

「Feature Maturity 및 안정성 주의사항」에서 정리했듯, Progressive Syncs(Beta), Dynamic Cluster Distribution(Alpha), Skip Application Reconcile(Alpha) 같은 기능은 **하위 호환성을 보장하지 않으며 향후 릴리스에서 깨지는 변경의 대상**입니다. 프로덕션에서 의존할 경우 다음 두 가지를 반드시 함께 실천하세요.

1. 사용 중인 Alpha/Beta 기능을 **문서화**한다.
2. **업그레이드 전 릴리스 노트를 검토**한다.

### 다음 단계

입문 범위의 기본 GitOps 루프(설치 → Application 생성 → Sync → 상태 확인)를 익혔다면, 이제 운영을 깊게 다루는 다음 주제로 나아갈 수 있습니다. 공식 문서에서 이어 학습하기 좋은 영역은 다음과 같습니다.

- **선언적 설정(Declarative Setup)** — Application·AppProject 자체를 Git에 매니페스트로 관리하기.
- **High Availability** — 프로덕션 HA 구성의 세부 튜닝.
- **User Management / RBAC** — SSO(OIDC 등) 연동과 멀티테넌시 인가 정책.
- **ApplicationSet** — 여러 Application을 생성기(generator)로 일괄 생성·관리.
- **Ingress / TLS 구성** — 자체 서명 인증서를 대체하는 안전한 외부 접근.
- **Automation from CI Pipelines** — CLI와 액세스 토큰으로 CI와 통합.

여기까지가 "Getting Started with Argo CD: Declarative GitOps for Kubernetes"의 마지막 섹션입니다. 핵심을 다시 한 문장으로 요약하면 — **배포를 Git에 선언으로 담고, Argo CD가 그것을 클러스터에 자동으로 반영하며, 차이와 이력을 항상 드러내는** 일관된 GitOps 모델을, 안전한 비밀번호 관리·HA 설치·버전 고정·자동 동기화·비안정 기능 주의라는 운영 원칙 위에서 운용하는 것입니다.
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
