---
layout: post
title: "Getting Started with Argo CD: Declarative GitOps for Kubernetes"
date: 2026-07-06 01:25:20
author: "bits-bytes-nn"
categories: ["Tech Guides"]
tags: []
cover: /assets/images/tech-guides.jpg
use_math: true
---

## GitOps란 무엇인가: 원칙과 Argo CD의 위치

Kubernetes가 클라우드 네이티브 인프라의 핵심 오케스트레이션 계층으로 자리 잡으면서, 애플리케이션과 인프라를 어떻게 배포하고 운영할 것인가에 대한 접근 방식도 함께 진화했습니다. 그 결과 등장한 운영 모델이 **GitOps**입니다. GitOps는 소프트웨어 개발에서 이미 검증된 관행(버전 관리, 코드 리뷰, 자동화된 워크플로)을 인프라와 애플리케이션 전달에 그대로 적용하는 운영 프레임워크입니다. 핵심 발상은 단순합니다. Git 저장소를 시스템이 어떤 상태여야 하는지를 정의하는 **단일 진실 공급원(single source of truth)** 으로 삼고, 이 선언된 상태를 클러스터에 자동으로 반영하는 것입니다.

여기서 짚어야 할 개념적 차이가 있습니다. 전통적인 명령형(imperative) 접근에서는 사용자가 원하는 상태에 도달하기 위한 단계와 명령을 직접 지시합니다. 반면 GitOps는 선언적(declarative) 접근을 취합니다. 사용자는 "무엇이 되어야 하는가(what)"만 선언하고, "어떻게 그 상태에 도달할 것인가(how)"는 시스템이 판단하여 실행합니다. Kubernetes 자체가 이미 이러한 선언형 모델 위에 세워져 있기 때문에, GitOps는 Kubernetes와 자연스럽게 맞물립니다. YAML 매니페스트로 표현된 원하는 상태를 Git에 저장하면, 클러스터 안의 에이전트가 그 상태로 시스템을 수렴시킵니다.

### OpenGitOps의 네 가지 원칙

GitOps라는 용어는 원래 Git을 인프라 구성의 진실 공급원으로 사용하는 방식을 설명하기 위해 만들어졌고, 이후 벤더 중립적인 OpenGitOps 워킹 그룹이 이를 네 가지 명확한 원칙으로 정리했습니다. 이 원칙들은 "GitOps를 어떻게 구현하라"고 지시하지는 않지만, 어떤 시스템이 GitOps로 관리된다고 말할 수 있는지에 대한 기준을 제시합니다.

| 원칙 | 의미 | 실무적 함의 |
|------|------|-------------|
| **선언적(Declarative)** | 시스템의 원하는 상태가 선언적으로 표현되어야 합니다. | 인프라, 네트워킹, 정책, 애플리케이션 모두를 명령 스크립트가 아닌 선언형 명세(예: Kubernetes YAML)로 기술합니다. |
| **버전 관리 및 불변(Versioned and Immutable)** | 원하는 상태가 불변성과 버전 관리를 강제하는 방식으로 저장되며 완전한 이력을 보존합니다. | 모든 변경이 새로운 불변 버전을 만들고, 저자·타임스탬프·변경 이유가 담긴 감사 이력이 남습니다. 롤백은 이전 커밋으로 되돌리는 것으로 끝납니다. |
| **자동 풀(Pulled Automatically)** | 소프트웨어 에이전트가 소스로부터 원하는 상태 선언을 자동으로 가져옵니다. | 웹훅이나 이벤트에 의존하지 않고, 에이전트가 Git에서 매니페스트를 직접 당겨 옵니다. |
| **지속적 조정(Continuously Reconciled)** | 소프트웨어 에이전트가 실제 상태를 지속적으로 관찰하고 원하는 상태를 적용하려 시도합니다. | 실제 상태와 선언된 상태를 끊임없이 비교하여 드리프트(drift)를 감지하고 교정하는 자기 치유(self-healing) 루프가 형성됩니다. |

세 번째와 네 번째 원칙은 함께 작동하면서 GitOps의 가장 강력한 특성을 만들어 냅니다. 여기서 등장하는 **소프트웨어 에이전트**는 사용자나 조직을 대신해 작업을 지속적으로 수행하는 프로그램으로, 대표적으로 Kubernetes 컨트롤러가 이에 해당합니다. 컨트롤러는 조정 에이전트(reconciliation agent)처럼 동작하여 시스템 상태를 관찰하고 규제합니다. 이들은 Git을 유일한 진실 공급원으로 바라보며, 시스템 상태가 원하는 상태와 어긋나면 이를 다시 맞추는 것을 자기 임무로 삼습니다.

### 조정 루프를 개념적으로 이해하기

이 지속적 조정 루프는 다음과 같은 순환으로 요약할 수 있습니다. 에이전트는 (1) Git에서 원하는 상태를 읽고, (2) 클러스터에서 실제 상태를 읽은 뒤, (3) 두 상태를 비교합니다. 만약 두 상태가 일치하면 아무 조치 없이 다음 주기를 기다립니다. 일치하지 않으면(즉, 누군가 수동으로 리소스를 바꾸었거나 부분적 장애가 발생했다면) 에이전트는 드리프트를 알리거나 선언된 상태를 다시 적용하여 클러스터를 교정합니다. 이 폐루프(closed-loop) 동작이 바로 클러스터를 "항상 정의된 상태로 수렴하는 자기 치유 시스템"으로 만드는 핵심입니다.

### 푸시 모델 vs 풀 모델

GitOps를 다른 CI/CD 방식과 구별짓는 결정적 지점은 변경 사항이 어떻게 클러스터로 전파되느냐입니다. 이는 단순한 구현 취향의 문제가 아니라 보안 경계와 운영 복잡도를 근본적으로 바꾸는 아키텍처 선택입니다.

| 구분 | 푸시 모델(Push) | 풀 모델(Pull) |
|------|-----------------|----------------|
| 변경 전파 주체 | 외부 CI/CD 파이프라인이 클러스터로 직접 밀어 넣습니다. | 클러스터 내부 에이전트(오퍼레이터)가 Git에서 당겨 옵니다. |
| 자격 증명 위치 | 파이프라인이 클러스터를 수정할 자격 증명을 보관합니다. | 자격 증명이 클러스터 밖으로 나가지 않습니다. |
| Kubernetes API 노출 | 외부에서 접근 가능해야 하는 경우가 많습니다. | API를 공개적으로 노출할 필요가 없습니다. |
| 다중 클러스터 확장성 | 클러스터마다 추가 자격 증명과 네트워크 접근이 필요해 복잡도가 증가합니다. | 각 클러스터가 스스로 상태를 당겨 오므로 격리와 확장에 유리합니다. |
| 드리프트 교정 | 파이프라인이 실행될 때만 반영되며, 자동 교정이 기본이 아닙니다. | 지속적 조정으로 드리프트를 자동 감지·교정합니다. |

전통적인 파이프라인에서는 중앙 시스템이 저장된 자격 증명으로 클러스터를 직접 수정합니다. 이는 단일 실패 지점을 만듭니다. 파이프라인이 침해되면 공격자가 그 파이프라인이 닿는 모든 환경에 광범위한 접근 권한을 얻게 됩니다. 반면 풀 모델에서는 에이전트가 클러스터 안에 상주하며 Git에서 변경 사항을 당겨 옵니다. 자격 증명이 클러스터 내부에 머물기 때문에 중앙 파이프라인이 침해되더라도 전체 클러스터 집단의 침해로 곧바로 이어지지 않습니다. 이러한 이유로 보안과 규모를 중시하는 GitOps 구현에서는 풀 기반 모델이 기본 아키텍처로 채택됩니다.

### Argo CD의 위치

이러한 GitOps 원칙들이 실제 도구로 구현된 대표적인 사례가 **Argo CD**입니다. Argo CD는 공식적으로 "Kubernetes를 위한 선언적 GitOps 지속적 전달(continuous delivery) 도구"로 정의됩니다. 그 설계 철학은 GitOps 원칙과 정확히 맞아떨어집니다.

> - 애플리케이션 정의, 구성, 환경은 선언적이고 버전 관리되어야 합니다.
> - 애플리케이션 배포와 라이프사이클 관리는 자동화되고, 감사 가능하며, 이해하기 쉬워야 합니다.

Argo CD는 앞서 설명한 네 가지 원칙을 어떻게 구현할까요? 첫째, Git 저장소를 원하는 애플리케이션 상태를 정의하는 진실 공급원으로 사용합니다(선언적, 버전 관리 및 불변). 둘째, Argo CD는 하나의 **Kubernetes 컨트롤러**로 구현되어 실행 중인 애플리케이션을 지속적으로 모니터링하고, 현재의 라이브 상태(live state)를 Git 저장소에 명시된 원하는 목표 상태(target state)와 비교합니다(자동 풀, 지속적 조정). 이는 GitOps의 세 번째·네 번째 원칙을 그대로 코드로 옮긴 것입니다. Argo CD는 Kubernetes 컨트롤러가 사용하는 것과 동일한 조정 패턴 — 실제 상태를 관찰하고, 원하는 상태와 비교하고, 교정 조치를 취하는 — 위에서 동작하므로 별도의 외부 오케스트레이션 계층 없이 깔끔하게 통합됩니다.

이 조정 과정에서 Argo CD는 몇 가지 핵심 개념을 도입합니다. 라이브 상태가 목표 상태에서 벗어난 애플리케이션은 `OutOfSync` 상태로 간주됩니다. Argo CD는 이 차이를 보고하고 시각화하는 동시에, 라이브 상태를 원하는 목표 상태로 되돌리는 **동기화(sync)** 를 자동 또는 수동으로 수행할 수단을 제공합니다. Git 저장소에서 목표 상태를 수정하면 그 변경이 지정된 대상 환경에 자동으로 적용되어 반영될 수 있습니다.

다음 그림은 Argo CD의 전체 동작 흐름을 보여 줍니다.

![Argo CD 아키텍처](https://argo-cd.readthedocs.io/en/stable/assets/argocd_architecture.png)

Argo CD가 매니페스트의 출처로 삼을 수 있는 형식은 하나로 고정되어 있지 않습니다. Kustomize 애플리케이션, Helm 차트, Jsonnet 파일, 플레인 YAML/JSON 매니페스트 디렉터리, 그리고 구성 관리 플러그인으로 등록한 임의의 커스텀 도구를 모두 지원합니다. 또한 배포는 브랜치나 태그의 업데이트를 추적하거나 특정 Git 커밋의 매니페스트에 고정하는 방식으로 이루어질 수 있습니다. 이처럼 Argo CD는 GitOps의 추상적 원칙을 Kubernetes 위에서 실용적으로 구현하는 지점에 위치하며, 풀 기반·조정 중심 오퍼레이터 계열의 대표 도구로 자리매김하고 있습니다.

GitOps가 제공하는 이점 — 협업과 투명성 강화, 배포 주기 가속, 보안과 컴플라이언스 개선, 재해 복구 역량 강화 — 는 모두 이 네 가지 원칙에서 파생됩니다. 모든 변경이 커밋으로 기록되므로 무엇이, 누구에 의해, 왜 바뀌었는지가 감사 가능한 이력으로 남고, 문제가 생긴 배포는 특정 커밋으로 추적하여 즉시 되돌릴 수 있습니다. Git이 시스템의 완전한 선언적 정의를 담고 있기 때문에, 클러스터가 손실되거나 재구축되어야 할 때 저장소가 곧 복구 청사진이 됩니다. 이러한 특성들이 왜 그토록 많은 조직이 GitOps를 채택하는지를 설명하며, Argo CD는 바로 그 실현 수단으로서 이 가이드 전반에서 다루게 될 도구입니다.

## Argo CD 핵심 개념과 용어

Argo CD를 효과적으로 다루려면 이 도구가 도입하는 고유한 용어 체계를 먼저 익혀야 합니다. Git, Docker, Kubernetes, 지속적 전달과 GitOps의 일반 개념에 이미 익숙하다는 전제 위에서, 여기서는 Argo CD에만 해당하는 어휘를 정리합니다. 이 용어들은 이후 모든 장의 CLI 출력, UI 화면, 매니페스트 필드에서 반복적으로 등장하므로 정확히 구분해 두는 것이 중요합니다.

### 중심에 있는 것: Application

Argo CD에서 가장 근본적인 단위는 **Application**입니다. Application은 하나의 매니페스트로 정의되는 Kubernetes 리소스의 묶음이며, 그 자체가 Kubernetes의 커스텀 리소스 정의(Custom Resource Definition, CRD)로 구현되어 있습니다. Application이 Kubernetes 리소스라는 사실은 중요한 함의를 갖습니다. Argo CD의 설정 자체가 선언적으로 Git에 저장되고 관리될 수 있다는 뜻이기 때문입니다.

Argo CD Core 설치에서 사용할 수 있는 리소스가 `Application`과 `ApplicationSet` CRD라는 점에서 알 수 있듯, Argo CD는 몇 가지 커스텀 리소스를 중심으로 동작합니다. `ApplicationSet`은 별도의 장에서 자세히 다룹니다. 또한 `AppProject` 리소스도 존재하며(Argo CD 리소스에는 Applications, ApplicationSets, AppProjects가 포함됩니다), 이는 접근 제어를 다루는 장에서 살펴봅니다.

### 핵심 용어 정리

다음 표는 Argo CD 문서가 정의하는 핵심 개념들을 한자리에 모은 것입니다. 이후 서술에서는 이 중 혼동하기 쉬운 항목을 골라 더 깊이 설명합니다.

| 용어 | 정의 |
|------|------|
| **Application** | 매니페스트로 정의된 Kubernetes 리소스의 그룹. CRD로 구현됩니다. |
| **Application source type** | 애플리케이션을 빌드하는 데 사용되는 도구(Tool)의 종류. |
| **Target state (목표 상태)** | Git 저장소의 파일로 표현되는, 애플리케이션이 도달해야 할 원하는 상태. |
| **Live state (라이브 상태)** | 실제로 배포되어 있는 상태. 어떤 파드 등이 실제로 떠 있는가. |
| **Sync status (동기화 상태)** | 라이브 상태가 목표 상태와 일치하는지 여부. 배포된 애플리케이션이 Git이 명시한 것과 같은가. |
| **Sync (동기화)** | 애플리케이션을 목표 상태로 이동시키는 과정. 예를 들어 클러스터에 변경을 적용하는 것. |
| **Sync operation status** | 동기화 작업이 성공했는지 여부. |
| **Refresh (새로고침)** | Git의 최신 코드를 라이브 상태와 비교하여 무엇이 다른지 파악하는 것. |
| **Health (상태 건강도)** | 애플리케이션이 올바르게 실행되고 요청을 처리할 수 있는지에 대한 건강 상태. |
| **Tool (도구)** | 파일 디렉터리로부터 매니페스트를 생성하는 도구. 예: Kustomize. Application source type 참고. |
| **Configuration management tool** | Tool과 동일한 의미. |
| **Configuration management plugin** | 커스텀 도구(사용자가 등록한 임의의 도구). |

### 세 가지 상태와 sync status의 의미

앞선 장에서 라이브 상태가 목표 상태에서 벗어난 애플리케이션이 `OutOfSync`로 간주된다는 점을 소개했습니다. 이 판정의 논리를 용어로 정확히 표현하면 다음과 같습니다. **목표 상태**는 Git 저장소에 담긴 원하는 모습이고, **라이브 상태**는 실제 클러스터에 존재하는 모습이며, **sync status**는 이 둘이 일치하는지를 나타내는 값입니다. 즉 sync status는 "배포된 것이 Git이 말하는 것과 같은가?"라는 단 하나의 질문에 대한 답입니다. 둘이 같으면 동기화된 상태(Synced), 다르면 `OutOfSync`가 됩니다.

여기서 반드시 구분해야 할 것이 sync status와 **health**입니다. 이 둘은 전혀 다른 것을 측정합니다.

- **sync status**는 *일치 여부*를 봅니다. 배포된 매니페스트가 Git의 선언과 동일한가만을 판단합니다.
- **health**는 *동작 여부*를 봅니다. 애플리케이션이 올바르게 실행되고 있으며 요청을 처리할 수 있는 상태인지를 평가합니다.

두 지표는 서로 독립적으로 조합될 수 있습니다. 예컨대 Git의 최신 매니페스트가 클러스터에 적용되어 `Synced`이지만 health가 나쁠 수 있고, 반대로 아직 배포되지 않아 리소스가 존재하지 않는 초기 상태에서는 sync status가 `OutOfSync`이면서 health가 `Missing`으로 나타납니다. 실제로 첫 Application을 만들고 동기화하는 장에서 확인하게 될 `argocd app get` 출력이 정확히 이 상황을 보여 줍니다.

```text
Name:            guestbook
...
Sync Status:     OutOfSync from  (1ff8a67)
Health Status:   Missing

GROUP  KIND        NAMESPACE  NAME          STATUS     HEALTH
apps   Deployment  default    guestbook-ui  OutOfSync  Missing
       Service     default    guestbook-ui  OutOfSync  Missing
```

애플리케이션이 아직 배포되지 않아 Kubernetes 리소스가 생성되지 않은 초기 상태이므로 `OutOfSync`로 나타납니다. 여기서 `Sync Status`와 `Health Status`가 나란히, 그러나 별개의 열로 표시된다는 점에 주목하면 두 개념의 독립성이 분명해집니다.

### sync, sync operation status, refresh의 구분

**sync**는 상태를 나타내는 값이 아니라 *행위*입니다. 애플리케이션을 목표 상태로 이동시키기 위해 실제로 변경을 적용하는 과정 — 즉 저장소에서 매니페스트를 가져와 `kubectl apply`로 클러스터에 반영하는 작업 — 을 가리킵니다. 이 sync 작업이 성공적으로 끝났는지 여부를 별도로 추적하는 값이 **sync operation status**입니다. sync status(일치 여부)와 sync operation status(작업 성공 여부)는 이름이 비슷하지만 다른 것을 측정합니다. 전자는 "지금 일치하는가"라는 상태를, 후자는 "방금 실행한 동기화 작업이 성공했는가"라는 결과를 나타냅니다.

**refresh**는 sync와 흔히 혼동되지만 다릅니다. refresh는 Git의 최신 코드를 라이브 상태와 *비교*하여 무엇이 달라졌는지 파악하는 작업입니다. 즉 refresh는 차이를 관찰하는 행위이고, sync는 그 차이를 실제로 해소하는 적용 행위입니다. 이 구분은 자동 동기화를 다룰 때 특히 중요해집니다. Argo CD가 드리프트를 *감지*하는 것과 그것을 *교정*하는 것은 서로 다른 단계이기 때문입니다.

### Application source type, Tool, 그리고 플러그인

마지막 묶음은 매니페스트가 어떻게 만들어지는지에 관한 용어입니다. **Tool**(구성 관리 도구, configuration management tool)은 파일 디렉터리로부터 Kubernetes 매니페스트를 생성하는 도구를 뜻하며, 대표적으로 Kustomize가 있습니다. 어떤 애플리케이션이 어떤 Tool로 빌드되는지를 나타내는 값이 **Application source type**입니다. 앞선 장에서 Argo CD가 Kustomize, Helm, Jsonnet, 플레인 YAML/JSON 디렉터리 등 여러 형식을 지원한다고 소개했는데, 이때 "어떤 형식인가"를 분류하는 개념이 바로 source type입니다.

기본 제공되는 도구만으로 충분하지 않을 때는 **configuration management plugin**을 사용합니다. 이는 사용자가 직접 등록한 커스텀 도구입니다. 각 매니페스트 소스 도구의 실제 사용법은 매니페스트 소스 도구를 다루는 장에서 구체적으로 살펴봅니다.

## Argo CD 아키텍처: 컴포넌트가 함께 동작하는 방식

Argo CD는 하나의 거대한 단일 프로세스가 아니라, 각기 다른 책임을 지는 여러 컴포넌트가 협력하는 **컴포넌트 기반 아키텍처(component-based architecture)** 로 설계되어 있습니다. 이 설계 덕분에 필요에 따라 더 미니멀한 구성으로 설치하는 것도 가능합니다. Argo CD가 전체적으로 하나의 Kubernetes 컨트롤러처럼 동작한다는 점은 이미 짚었지만, 그 "컨트롤러"가 내부적으로 어떻게 나뉘어 실제 일을 처리하는지를 이해하는 것이 이 장의 목표입니다. 핵심 컴포넌트는 세 가지입니다. **API Server**, **Repository Server**, 그리고 **Application Controller**입니다. 여기에 캐싱을 담당하는 Redis가 더해집니다.

### 세 가지 핵심 컴포넌트

먼저 각 컴포넌트가 무엇을 책임지는지 표로 정리하면 다음과 같습니다.

| 컴포넌트 | 유형 | 주요 책임 |
|----------|------|-----------|
| **API Server** | gRPC/REST 서버 | 애플리케이션 관리 및 상태 보고, sync·rollback·사용자 정의 액션 등 작업 호출, 저장소·클러스터 자격 증명 관리(K8s 시크릿으로 저장), 인증 및 외부 아이덴티티 공급자로의 인증 위임, RBAC 강제, Git 웹훅 이벤트 수신·전달 |
| **Repository Server** | 내부 서비스 | 애플리케이션 매니페스트를 담은 Git 저장소의 로컬 캐시 유지, 입력이 주어지면 Kubernetes 매니페스트를 생성·반환 |
| **Application Controller** | Kubernetes 컨트롤러 | 실행 중인 애플리케이션을 지속적으로 모니터링하며 라이브 상태와 목표 상태를 비교, `OutOfSync` 상태 감지 및 선택적 교정, 라이프사이클 이벤트(PreSync, Sync, PostSync)에 대한 사용자 정의 훅 호출 |

이 세 컴포넌트가 각자 담당하는 영역이 명확히 분리되어 있다는 점이 Argo CD 아키텍처를 이해하는 열쇠입니다. 하나씩 깊이 들여다보겠습니다.

#### API Server

**API Server**는 gRPC/REST 서버로서, Web UI, CLI, 그리고 CI/CD 시스템이 소비하는 API를 노출합니다. 즉 사용자가 `argocd` CLI로 명령을 내리거나 브라우저에서 UI를 조작할 때, 그 요청을 실제로 받아 처리하는 진입점이 바로 이 컴포넌트입니다. API Server의 책임은 넓습니다. 애플리케이션의 관리와 상태 보고를 담당하고, sync·rollback·사용자 정의 액션 같은 애플리케이션 작업을 호출합니다. 또한 저장소와 클러스터의 자격 증명을 관리하는데, 이 자격 증명들은 Kubernetes 시크릿으로 저장됩니다. 인증을 처리하고 외부 아이덴티티 공급자로 인증을 위임하며, RBAC를 강제하고, Git 웹훅 이벤트를 수신·전달하는 리스너/포워더 역할까지 맡습니다.

여기서 주목할 점은 자격 증명의 위치입니다. 앞서 풀 모델의 보안 이점을 논하면서 "자격 증명이 클러스터 밖으로 나가지 않는다"고 설명했는데, 이것이 구체적으로 구현되는 지점이 바로 API Server입니다. 저장소 접근 자격 증명과 대상 클러스터 자격 증명이 모두 Kubernetes 시크릿으로 관리되기 때문입니다.

#### Repository Server

**Repository Server**는 애플리케이션 매니페스트를 담은 Git 저장소의 로컬 캐시를 유지하는 내부 서비스입니다. 이 컴포넌트의 핵심 임무는 단 하나로 요약됩니다 — 필요한 입력이 주어지면 Kubernetes 매니페스트를 **생성하여 반환**하는 것입니다. 이때 필요한 입력은 다음과 같습니다.

- 저장소 URL(repository URL)
- 리비전(revision) — 커밋, 태그, 또는 브랜치
- 애플리케이션 경로(application path)
- 템플릿 도구별 설정 — 파라미터, Helm의 `values.yaml` 등

여기서 앞서 다룬 개념들이 실제 컴포넌트와 연결됩니다. Argo CD가 Kustomize, Helm, Jsonnet, 플레인 YAML 같은 여러 소스 도구를 지원한다고 했는데, 이 매니페스트 생성 작업을 실제로 수행하는 곳이 Repository Server입니다. 예컨대 Helm 차트가 소스라면 Repository Server가 `values.yaml`을 입력받아 차트를 렌더링해 최종 Kubernetes 매니페스트를 만들어 냅니다. 즉 "목표 상태(target state)"라는 추상적 개념이 실제 YAML 매니페스트로 물질화되는 지점이 바로 이 컴포넌트입니다.

#### Application Controller

**Application Controller**는 실행 중인 애플리케이션을 지속적으로 모니터링하며, 현재의 라이브 상태를 저장소에 명시된 원하는 목표 상태와 비교하는 Kubernetes 컨트롤러입니다. Argo CD 전체를 "하나의 Kubernetes 컨트롤러"라고 부를 때, 그 조정 루프의 실질적 심장부가 바로 이 Application Controller입니다. 이 컴포넌트는 `OutOfSync` 애플리케이션 상태를 감지하고, 필요하다면 선택적으로 교정 조치를 취합니다. 또한 라이프사이클 이벤트에 대한 사용자 정의 훅 — PreSync, Sync, PostSync — 을 호출하는 책임도 맡습니다. 이 훅들은 블루/그린 배포나 카나리 업그레이드처럼 복잡한 애플리케이션 롤아웃을 지원하기 위한 장치이며, 동기화 정책과 라이프사이클 훅을 다루는 장에서 자세히 살펴봅니다.

정리하면, Application Controller가 조정 루프를 돌리는 두뇌라면, Repository Server는 "목표 상태가 무엇인지"를 계산해 공급하는 렌더링 엔진이고, API Server는 사람과 자동화 도구가 이 시스템과 대화하는 창구입니다.

### Redis: 컨트롤러를 떠받치는 캐시

세 핵심 컴포넌트 외에 Redis도 아키텍처의 중요한 구성 요소입니다. Argo CD 컨트롤러는 엄밀히 말하면 Redis 없이도 실행될 수 있지만, 그렇게 운영하는 것은 권장되지 않습니다. 컨트롤러가 Redis를 중요한 캐싱 메커니즘으로 사용하여 Kube API와 Git에 가해지는 부하를 줄이기 때문입니다. 이러한 이유로 Redis는 최소 구성의 설치 방식에도 포함됩니다.

기본 설치에서 Redis는 비밀번호 인증을 사용하며, 이 비밀번호는 Argo CD가 설치된 네임스페이스의 `argocd-redis`라는 Kubernetes 시크릿에 `auth` 키로 저장됩니다.

다음 그림은 이러한 컴포넌트 구성을 시각적으로 보여 줍니다. 그림의 `Core` 박스는 Argo CD Core를 선택할 때 설치되는 컴포넌트들을 나타냅니다.

![Argo CD Core 컴포넌트 구성](https://argo-cd.readthedocs.io/en/stable/assets/argocd-core-components.png)

이 컴포넌트 기반 설계 덕분에 API Server와 UI를 제외한 최소 구성(headless 모드, Argo CD Core)도 가능한데, 이러한 설치 방식의 선택지는 설치 방식을 다루는 장에서 구체적으로 비교합니다.

### 컴포넌트가 함께 동작하는 흐름: 하나의 sync를 따라가기

각 컴포넌트의 역할을 개별적으로 이해했다면, 이제 이들이 실제 작업 하나를 어떻게 나누어 처리하는지 따라가 보는 것이 이해를 완성시킵니다. 사용자가 어떤 애플리케이션을 동기화하는 상황을 예로 들어 보겠습니다.

```bash
argocd app sync guestbook
```

이 한 줄의 명령이 실행될 때 컴포넌트들 사이에서 벌어지는 일을 단계별로 정리하면 다음과 같습니다.

1. **CLI → API Server.** `argocd` CLI가 발행한 요청은 gRPC/REST 인터페이스를 통해 API Server에 도달합니다. API Server는 먼저 사용자를 인증하고 RBAC 정책에 따라 이 사용자가 해당 애플리케이션에 sync 작업을 호출할 권한이 있는지 확인합니다. 권한이 확인되면 sync라는 애플리케이션 작업을 호출합니다.

2. **목표 상태 계산(Repository Server).** 실제로 무엇을 적용해야 하는지 알려면 목표 상태를 확정해야 합니다. 이를 위해 Repository Server가 저장소 URL, 리비전, 애플리케이션 경로, 그리고 템플릿 설정을 입력받아 매니페스트를 렌더링합니다. 로컬 캐시를 유지하고 있으므로 매번 원격 Git을 전부 다시 받아오지 않아도 됩니다.

3. **비교와 교정(Application Controller).** Application Controller는 Repository Server가 생성한 목표 상태와 클러스터의 라이브 상태를 비교합니다. 라이브 상태가 목표 상태에서 벗어나 있으면 `OutOfSync`로 판정하고, sync 과정에서 그 차이를 해소하기 위해 매니페스트를 클러스터에 적용합니다. 이 과정에서 PreSync·Sync·PostSync 훅이 정의되어 있다면 적절한 시점에 호출됩니다.

4. **캐싱(Redis).** 이 전 과정에서 컨트롤러는 조회한 상태와 렌더링 결과를 Redis에 캐싱하여, 이후 조정 주기에서 Kube API와 Git에 대한 반복 부하를 줄입니다.

앞서 첫 Application 예제에서 `argocd app sync guestbook`이 "저장소에서 매니페스트를 가져와 `kubectl apply`를 수행한다"고 설명했는데, 그 한 문장 뒤에서 실제로는 이처럼 API Server·Repository Server·Application Controller가 각자의 책임을 나누어 협력하고 있는 것입니다. "매니페스트를 가져오는" 부분은 Repository Server가, "권한을 확인하고 작업을 호출하는" 부분은 API Server가, "비교하고 적용하는" 부분은 Application Controller가 담당합니다.

Git 웹훅이 설정된 환경이라면 흐름의 출발점만 달라집니다. 이 경우 CLI 대신 Git 웹훅 이벤트가 API Server의 리스너/포워더로 전달되며, 그 이후의 Repository Server·Application Controller의 협력 방식은 동일합니다. 다만 GitOps의 풀 모델에서 컨트롤러는 지속적으로 라이브 상태를 관찰하며 목표 상태를 적용하려 하므로, 웹훅은 어디까지나 반응 속도를 높이는 보조 수단이라는 점을 기억할 필요가 있습니다.

## 설치 방식 선택하기: Multi-Tenant vs Core vs HA

Argo CD 설치를 시작하기 전에 먼저 결정해야 할 것은 "어떤 형태로 설치할 것인가"입니다. Argo CD는 크게 **멀티테넌트(Multi-Tenant)** 와 **Core**라는 두 가지 설치 유형을 제공하며, 멀티테넌트 유형 안에서 다시 **비고가용성(Non-HA)** 과 **고가용성(High Availability, HA)** 매니페스트를 고를 수 있습니다. 이 선택은 단순한 설치 편의의 문제가 아니라, 누가 Argo CD를 사용할 것인지, 권한 경계를 어디에 둘 것인지, 그리고 어느 수준의 내결함성이 필요한지를 규정하는 아키텍처적 결정입니다. 앞서 컴포넌트 기반 설계 덕분에 더 미니멀한 구성이 가능하다고 언급했는데, 여기서 그 선택지를 구체적으로 비교합니다.

### 멀티테넌트 설치

멀티테넌트 설치는 Argo CD를 설치하는 가장 일반적인 방식입니다. 조직 내 여러 애플리케이션 개발 팀에게 서비스를 제공하고 플랫폼 팀이 이를 유지·관리하는 상황을 전형적인 대상으로 삼습니다. 최종 사용자는 API Server를 통해 Web UI나 `argocd` CLI로 Argo CD에 접근하며, CLI는 `argocd login <server-host>` 명령으로 구성합니다.

멀티테넌트에는 두 가지 성격의 매니페스트가 제공되며, 이 둘은 **권한 범위(cluster-admin vs 네임스페이스 수준)** 라는 축에서 갈라집니다.

- **`install.yaml` (표준 설치)** — cluster-admin 권한을 갖는 표준 Argo CD 설치입니다. Argo CD가 실행되는 바로 그 클러스터에 애플리케이션을 배포하려는 경우에 사용합니다. 물론 이 설치로도 자격 증명을 입력하면 외부 클러스터로 배포하는 것이 여전히 가능합니다.
- **`namespace-install.yaml` (네임스페이스 수준 설치)** — 클러스터 롤(cluster role) 없이 네임스페이스 수준 권한만으로 동작하는 설치입니다. Argo CD가 실행되는 클러스터에 직접 배포할 필요가 없고 오로지 입력받은 외부 클러스터 자격 증명에만 의존하려는 경우에 적합합니다. 예를 들어 팀마다 별도의 Argo CD 인스턴스를 운영하면서 각 인스턴스가 외부 클러스터로만 배포하는 구성이 대표적입니다. 기본 롤 구성으로는 같은 클러스터 안에서 Argo CD 리소스(Applications, ApplicationSets, AppProjects)만 배포할 수 있으며, 실제 워크로드 배포는 외부 클러스터로 이루어지는 GitOps 모드만 지원합니다. 이 동작은 `argocd-application-controller` 서비스 계정에 새로운 롤을 정의·바인딩하여 변경할 수 있습니다.

`namespace-install.yaml`을 쓸 때 반드시 유의할 점이 있습니다. Argo CD CRD가 이 매니페스트에 **포함되지 않으므로** 별도로 설치해야 합니다. CRD 매니페스트는 `manifests/crds` 디렉터리에 있으며 다음 명령으로 설치합니다.

```bash
kubectl apply --server-side --force-conflicts -k https://github.com/argoproj/argo-cd/manifests/crds\?ref\=stable
```

또한 표준 설치의 경우, 설치 매니페스트에 포함된 `ClusterRoleBinding`이 `argocd` 네임스페이스의 서비스 계정에 바인딩되어 있습니다. 따라서 다른 네임스페이스에 설치할 때는 이 참조를 그에 맞게 조정하지 않으면 권한 관련 오류가 발생할 수 있습니다.

### 고가용성(HA) 설치

멀티테넌트 매니페스트는 각각 HA 변형을 함께 제공합니다. HA 설치는 프로덕션 사용에 권장되며, 동일한 컴포넌트를 고가용성과 내결함성에 맞게 튜닝한 번들입니다. 핵심 차이는 **지원되는 컴포넌트에 대해 여러 개의 레플리카(replica)를 운영한다**는 점입니다.

- `ha/install.yaml` — `install.yaml`과 동일하되 지원 컴포넌트가 다중 레플리카로 구성됩니다.
- `ha/namespace-install.yaml` — `namespace-install.yaml`과 동일하되 지원 컴포넌트가 다중 레플리카로 구성됩니다.

반대로 비고가용성 매니페스트(`install.yaml`, `namespace-install.yaml`)는 프로덕션에는 권장되지 않으며, 평가 기간의 데모나 테스트 용도로 주로 사용됩니다.

### Core 설치

Core 설치는 Argo CD를 **헤드리스(headless) 모드**로 실행하는 별개의 설치 방식입니다. 이 방식으로도 Git 저장소에서 원하는 상태를 가져와 Kubernetes에 적용하는 완전한 기능의 GitOps 엔진을 갖게 됩니다. 앞선 아키텍처 논의에서 컴포넌트 기반 설계 덕분에 더 적은 컴포넌트만으로도 핵심 GitOps 기능이 유지된다고 설명했는데, Core가 바로 그 최소 구성 설치입니다. Core는 API Server와 UI를 포함하지 않으며, 각 컴포넌트의 경량(비HA) 버전을 설치합니다.

Core를 선택하면 다음 기능들이 **제공되지 않습니다**.

- Argo CD RBAC 모델
- Argo CD API
- Argo CD Notification Controller
- OIDC 기반 인증

그리고 다음 기능들은 **부분적으로만** 제공됩니다.

- Argo CD Web UI
- Argo CD CLI
- 멀티테넌시(오직 git push 권한에 기반한 엄격한 GitOps 방식)

이러한 제약은 Core가 겨냥하는 사용자상을 그대로 반영합니다. Core는 멀티테넌시 기능이 필요 없이 독립적으로 Argo CD를 사용하는 클러스터 관리자에게 가장 적합하며, 포함되는 컴포넌트가 적어 설정이 더 쉽습니다. Core가 정당화되는 대표적 상황은 다음과 같습니다.

- 클러스터 관리자로서 Kubernetes RBAC에만 의존하고 싶은 경우
- DevOps 엔지니어로서 새로운 API나 별도 CLI를 익히지 않고 Kubernetes API에만 의존해 배포를 자동화하고 싶은 경우
- 클러스터 관리자로서 개발자에게 Argo CD UI나 CLI를 제공하고 싶지 않은 경우

Core에서 사용할 수 있는 Kubernetes 리소스는 `Application`과 `ApplicationSet` CRD입니다. CLI는 Core 모드에서도 사용할 수 있는데, 이 경우 `login` 하위 명령에 `--core` 플래그를 전달합니다. 이 플래그는 CLI 및 Web UI 요청을 처리하기 위해 로컬 Argo CD API Server 프로세스를 띄우는 역할을 하며, 명령이 끝나면 이 로컬 프로세스도 함께 종료됩니다. Core 모드에서는 인증과 인가가 전적으로 Kubernetes RBAC에 의존하므로, CLI를 실행하는 사용자(또는 프로세스)는 Argo CD 네임스페이스와 `Application`·`ApplicationSet` 리소스에 대한 적절한 권한을 가지고 있어야 합니다.

```bash
kubectl config set-context --current --namespace=argocd   # 현재 kube 컨텍스트를 argocd 네임스페이스로 변경
argocd login --core
```

Core 설치는 필요한 모든 리소스를 담은 단일 매니페스트(`core-install.yaml`) 하나를 적용하는 것으로 끝납니다.

```bash
export ARGOCD_VERSION=<원하는 Argo CD 릴리스 버전 (예: v2.7.0)>
kubectl create namespace argocd
kubectl apply -n argocd --server-side --force-conflicts \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/$ARGOCD_VERSION/manifests/core-install.yaml
```

### 세 방식 비교

다음 표는 각 설치 방식의 성격과 적합한 상황을 한눈에 정리한 것입니다.

| 항목 | 멀티테넌트 (Non-HA) | 멀티테넌트 (HA) | Core |
|------|--------------------|------------------|------|
| 대표 매니페스트 | `install.yaml`, `namespace-install.yaml` | `ha/install.yaml`, `ha/namespace-install.yaml` | `core-install.yaml` |
| 주 대상 | 여러 개발 팀에 서비스하는 플랫폼 팀 | 프로덕션 멀티테넌트 운영 | 독립적으로 사용하는 클러스터 관리자 |
| API Server / Web UI | 포함 | 포함 | 미포함(헤드리스) |
| Argo CD RBAC / API / OIDC / 알림 | 제공 | 제공 | 미제공 |
| CLI | 제공 | 제공 | `--core` 플래그로 로컬 API Server를 띄워 사용 |
| 멀티테넌시 | 제공 | 제공 | git push 권한 기반의 제한적 방식만 |
| 컴포넌트 레플리카 | 단일(비HA) | 다중 레플리카(내결함성) | 경량(비HA) 단일 |
| 프로덕션 권장 여부 | 권장되지 않음(평가·데모·테스트용) | 프로덕션 권장 | 관리자 단독 사용 시나리오에 적합 |

`install.yaml`과 `namespace-install.yaml` 중 어느 쪽을 고를지는 권한 범위의 문제라는 점을 다시 강조할 필요가 있습니다. Argo CD가 실행되는 클러스터에 직접 배포해야 한다면 cluster-admin 권한을 갖는 표준 설치가 필요하고, 오직 외부 클러스터로만 배포하며 네임스페이스 수준 권한으로 격리하고 싶다면 네임스페이스 설치가 적합합니다.

### Kustomize와 Helm을 통한 설치

위의 매니페스트들은 Kustomize로도 설치할 수 있습니다. 권장되는 방식은 매니페스트를 원격 리소스로 포함한 뒤 Kustomize 패치로 추가 커스터마이징을 적용하는 것입니다.

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: argocd
resources:
  - https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

기본 `argocd`가 아닌 커스텀 네임스페이스에 설치하려면, `ClusterRoleBinding`이 해당 네임스페이스의 서비스 계정을 올바르게 참조하도록 패치를 적용해야 합니다.

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

이 외에 Helm으로 설치하는 방법도 있습니다. Argo CD용 Helm 차트는 현재 커뮤니티가 유지·관리하며 `argo-helm/charts/argo-cd`에서 제공됩니다.

마지막으로 버전 호환성도 설치 방식 선택 못지않게 중요합니다. 각 Argo CD 버전이 어떤 Kubernetes 버전과 함께 테스트되었는지는 다음과 같습니다.

| Argo CD 버전 | 테스트된 Kubernetes 버전 |
|--------------|--------------------------|
| 3.4 | v1.35, v1.34, v1.33, v1.32 |
| 3.3 | v1.35, v1.34, v1.33, v1.32 |
| 3.2 | v1.34, v1.33, v1.32, v1.31 |

프로덕션 설치에서는 `stable` 브랜치를 그대로 추적하기보다 `v3.2.0`처럼 고정된 버전을 사용하는 것이 권장되며, 대상 클러스터의 Kubernetes 버전이 위 표의 테스트 범위 안에 드는지 확인하는 것이 안전합니다.
&lt;/your-custom-namespace&gt;&lt;/your-custom-namespace&gt;&lt;/server-host&gt;

## 사전 준비 및 배경 지식 다지기

Argo CD를 실제로 설치하기 전에, 도구가 딛고 설 기반 환경과 학습 전제를 점검하는 단계가 필요합니다. Argo CD는 Kubernetes 위에서 동작하는 컨트롤러이므로, 클러스터와 그 클러스터에 접근할 로컬 도구가 준비되어 있어야 하고, 매니페스트를 다루는 데 필요한 최소한의 배경 지식도 갖춰져 있어야 합니다. 이 준비가 부실하면 설치 자체는 성공하더라도 이후 접근·등록·동기화 단계에서 원인을 찾기 어려운 문제로 이어지기 쉽습니다.

### 환경 요구 사항

공식 시작 가이드는 Argo CD를 설치하고 조작하기 위한 최소 요구 사항으로 다음 세 가지를 명시합니다.

| 요구 사항 | 설명 |
|-----------|------|
| **`kubectl` 설치** | Kubernetes 클러스터와 통신하는 명령줄 도구입니다. Argo CD 매니페스트 적용, 서비스 패치, 시크릿 조회 등 설치 전후의 거의 모든 작업이 `kubectl`을 거칩니다. |
| **`kubeconfig` 파일** | 클러스터 접속 정보를 담은 설정 파일입니다. 기본 위치는 `~/.kube/config`이며, 여러 클러스터의 컨텍스트를 이 파일에서 관리합니다. |
| **CoreDNS** | 클러스터 내부 DNS 해석을 담당합니다. microk8s 환경이라면 `microk8s enable dns && microk8s stop && microk8s start` 명령으로 활성화할 수 있습니다. |

`kubeconfig`는 특히 중요합니다. 배포 대상 클러스터를 Argo CD에 등록할 때 이 파일에 정의된 컨텍스트 목록을 그대로 활용하기 때문입니다. 현재 사용 가능한 컨텍스트를 확인하는 명령은 다음과 같습니다.

```bash
kubectl config get-contexts -o name
```

이 명령으로 나열되는 컨텍스트 이름들은 이후 외부 클러스터를 등록하는 단계에서 다시 쓰이므로, 지금 시점에 어떤 클러스터에 접근 가능한지를 미리 파악해 두면 좋습니다. 또한 로컬 Kubernetes 환경(Docker Desktop 등)에서 실습을 진행할 계획이라면, 로컬 클러스터에 특화된 별도의 설정 절차가 필요할 수 있다는 점도 염두에 두어야 합니다.

CoreDNS가 요구되는 이유는 앞서 아키텍처를 다루며 살펴본 것처럼 Argo CD가 여러 컴포넌트로 구성되고, 이들이 클러스터 내부 서비스 이름으로 서로를 찾기 때문입니다. 예컨대 배포 대상이 Argo CD와 같은 클러스터일 때 사용하는 주소 `https://kubernetes.default.svc` 역시 클러스터 내부 DNS 해석에 의존합니다.

### 갖춰야 할 배경 지식

공식 문서는 Argo CD를 효과적으로 사용하기에 앞서 "플랫폼이 기반하고 있는 기술"을 이해할 것을 명시적으로 권장합니다. Argo CD의 핵심 개념들은 이미 다른 도구에서 확립된 개념 위에 세워져 있으므로, 다음 영역에 대한 기초가 있으면 학습에 도움이 됩니다.

- **Docker와 Kubernetes의 기초** — 컨테이너, 파드, 디플로이먼트, 서비스 같은 기본 리소스와 선언형 매니페스트의 동작 방식에 대한 이해가 전제됩니다. Argo CD의 라이브 상태·목표 상태 비교는 결국 이러한 Kubernetes 리소스를 대상으로 이루어집니다.
- **지속적 전달(CD)과 GitOps 개념** — 이 가이드의 앞부분에서 다룬 GitOps 원칙과 조정 루프에 대한 이해가 여기에 해당합니다.
- **Git의 기본 워크플로** — Git 저장소가 진실 공급원 역할을 하므로 커밋, 브랜치, 태그 같은 개념에 익숙해야 합니다.

여기에 더해, 애플리케이션을 어떤 방식으로 템플릿화할 것인지에 따라 추가로 학습해 둘 도구가 갈립니다.

| 사용 계획 | 미리 익혀 둘 도구 |
|-----------|-------------------|
| 매니페스트를 패치·오버레이 방식으로 관리 | **Kustomize** |
| 매니페스트를 파라미터화하여 패키징 | **Helm** |
| CI 도구와 연동해 파이프라인을 구성 | GitHub Actions, Jenkins 등 사용 중인 CI 도구 |

Kustomize와 Helm은 Argo CD가 매니페스트 소스로 직접 지원하는 도구이며, 각각의 실제 활용법은 매니페스트 소스 도구를 다루는 장에서 구체적으로 살펴봅니다. 이 시점에서는 자신의 배포 워크플로가 어느 쪽에 가까운지를 가늠하고, 해당 도구의 기본 개념을 훑어 두는 정도면 충분합니다.

### 설치 형태에 대한 사전 결정

환경과 지식이 준비되었다면, 실제 설치에 앞서 어떤 설치 형태를 택할지도 미리 결정해 두는 것이 좋습니다. 설치 방식의 차이(멀티테넌트 표준·네임스페이스·HA·Core)는 이미 설치 방식을 다루는 장에서 비교했으므로, 여기서는 그 결정이 사전 준비 단계에서 어떤 실무적 확인으로 이어지는지만 짚어 둡니다.

- UI·SSO·멀티클러스터 기능이 필요 없다면 Core 컴포넌트만 설치하는 선택지가 있습니다.
- 어느 방식을 택하든, 대상 클러스터의 Kubernetes 버전이 사용하려는 Argo CD 버전의 테스트 범위 안에 드는지 확인해야 합니다.
- 프로덕션 설치라면 `stable` 브랜치를 그대로 추적하기보다 고정된 릴리스 버전을 사용하는 편이 권장됩니다.

이러한 결정과 확인이 끝나면, 실제 매니페스트를 클러스터에 적용하는 설치 작업으로 넘어갈 준비가 된 것입니다.

## Argo CD 설치하기

준비된 클러스터에 Argo CD를 올리는 가장 표준적인 절차는 전용 네임스페이스를 만들고 공식 매니페스트를 적용하는 두 단계로 끝납니다. 멀티테넌트 표준 설치를 기준으로 하면 다음 명령이 전부입니다.

```bash
kubectl create namespace argocd
kubectl apply -n argocd --server-side --force-conflicts \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

첫 번째 명령은 `argocd`라는 새 네임스페이스를 만들고, 두 번째 명령은 stable 브랜치의 공식 매니페스트를 적용합니다. 이 결과로 Argo CD의 모든 서비스와 애플리케이션 리소스가 이 `argocd` 네임스페이스 안에 자리 잡게 됩니다. 즉 API Server, Repository Server, Application Controller, Redis 같은 구성 요소가 모두 이 네임스페이스에 배포됩니다.

여기서 눈여겨봐야 할 부분은 `-f` 뒤의 URL이 아니라 그 앞에 붙은 두 플래그입니다. 이 `--server-side`와 `--force-conflicts`는 단순한 관례가 아니라 Argo CD 설치가 성공하기 위한 필수 조건이며, 왜 필요한지를 이해하면 설치 중에 마주치는 오류의 상당수를 미리 피할 수 있습니다.

### `--server-side --force-conflicts`가 왜 필요한가

**서버 사이드 적용(server-side apply)** 은 `last-applied-configuration` 애노테이션을 저장하지 않습니다. 반면 전통적인 클라이언트 사이드 적용은 이 애노테이션에 마지막으로 적용한 구성을 저장합니다.

`--server-side` 플래그가 요구되는 이유가 바로 여기에 있습니다. Argo CD의 일부 CRD, 대표적으로 ApplicationSet CRD는 클라이언트 사이드 `kubectl apply`가 강제하는 **262KB 애노테이션 크기 제한**을 초과합니다. 서버 사이드 적용은 `last-applied-configuration` 애노테이션 자체를 저장하지 않기 때문에 이 제한을 우회할 수 있습니다.

두 번째 플래그인 `--force-conflicts`는 이 적용 작업이, Helm이나 이전의 `kubectl apply`처럼 다른 도구가 이전에 관리하던 필드의 소유권을 넘겨받도록 허용합니다. 처음 설치하는 경우에는 안전하며, 업그레이드 시에는 반드시 필요합니다.

여기서 실무적으로 반드시 알아 두어야 할 부작용이 있습니다. Argo CD 매니페스트에 **정의되어 있는** 필드에 가한 커스텀 수정은 덮어써집니다. 반대로 매니페스트에 **명시되지 않은** 필드는 보존됩니다. 이 차이를 표로 정리하면 다음과 같습니다.

| 구분 | 예시 필드 | 적용 시 동작 |
|------|-----------|--------------|
| Argo CD 매니페스트에 정의된 필드 | `affinity`, `env`, `probes` | 커스텀 수정이 덮어써짐 |
| 매니페스트에 명시되지 않은 필드 | `resources`의 limits/requests, `tolerations` | 기존 설정이 보존됨 |

이 규칙을 이해하면, 예컨대 Argo CD 컴포넌트에 리소스 제한이나 톨러레이션을 직접 추가해 두었다면 재적용 후에도 그 값이 유지되지만, `affinity`나 프로브 설정을 손봤다면 다음 적용 때 덮어써진다는 점을 예측할 수 있습니다. 컴포넌트 동작을 세밀하게 조정해야 한다면 이런 필드는 명령형 패치가 아니라 Kustomize 패치 방식으로 선언적으로 관리하는 편이 훨씬 안전합니다.

### 버전 고정

위 명령은 `stable` 브랜치를 그대로 가리키므로 실행 시점의 최신 안정 매니페스트가 적용됩니다. 실습이나 평가 용도로는 편리하지만, 프로덕션 설치에서는 `v3.2.0`처럼 고정된 릴리스 버전을 사용하는 것이 권장됩니다. 버전을 고정하면 URL의 `stable` 부분을 원하는 태그로 바꾸면 됩니다.

```bash
kubectl create namespace argocd
kubectl apply -n argocd --server-side --force-conflicts \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/v3.2.0/manifests/install.yaml
```

버전을 고정할 때는 대상 클러스터의 Kubernetes 버전이 해당 Argo CD 버전의 테스트 범위 안에 드는지 함께 확인하는 것이 좋습니다.

### Core·네임스페이스·HA 설치의 명령 차이

설치 형태를 이미 결정했다면, 실제 적용 명령은 위 표준 설치에서 매니페스트 파일 이름만 바꾸는 방식으로 이어집니다.

| 설치 유형 | 적용할 매니페스트 | 유의점 |
|-----------|-------------------|--------|
| 멀티테넌트 표준 | `manifests/install.yaml` | cluster-admin 권한. CRD 포함 |
| 멀티테넌트 네임스페이스 | `manifests/namespace-install.yaml` | CRD 미포함, 별도 설치 필요 |
| 멀티테넌트 HA | `manifests/ha/install.yaml`, `manifests/ha/namespace-install.yaml` | 지원 컴포넌트가 다중 레플리카 |
| Core(헤드리스) | `manifests/core-install.yaml` | 단일 매니페스트, API Server·UI 미포함 |

네임스페이스 수준 설치를 택했다면 CRD가 매니페스트에 포함되지 않으므로, `manifests/crds` 디렉터리의 CRD를 먼저 별도로 적용해야 합니다. 다음 명령을 사용할 수 있습니다.

```bash
kubectl apply --server-side --force-conflicts -k https://github.com/argoproj/argo-cd/manifests/crds\?ref\=stable
```

Core 설치라면 `ARGOCD_VERSION` 환경 변수를 지정한 뒤 `core-install.yaml` 하나만 적용하는 것으로 끝납니다.

```bash
export ARGOCD_VERSION=<원하는 릴리스 버전 (예: v2.7.0)>
kubectl create namespace argocd
kubectl apply -n argocd --server-side --force-conflicts \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/$ARGOCD_VERSION/manifests/core-install.yaml
```

### 설치 직후의 상태 이해하기

매니페스트를 적용한 직후 Argo CD가 어떤 상태에 놓이는지를 정확히 알아 두면, 다음 단계인 접근·로그인에서 혼란을 줄일 수 있습니다. 두 가지 지점이 특히 중요합니다.

첫째, 기본 설치는 **자체 서명 인증서(self-signed certificate)** 를 사용합니다. 따라서 별도의 작업 없이는 브라우저나 CLI에서 바로 접근하기 어렵습니다. 이를 해결하는 방법은 세 가지가 있습니다 — 정식 인증서를 구성하거나, 클라이언트 OS가 자체 서명 인증서를 신뢰하도록 설정하거나, 모든 Argo CD CLI 조작에서 `--insecure` 플래그를 사용하는 것입니다.

둘째, 기본 설치에서 Argo CD는 외부에 노출되지 않습니다. 설치만으로는 클러스터 밖에서 API Server에 접근할 수 없으며, LoadBalancer 서비스 타입 변경·Ingress·포트 포워딩 중 하나를 구성해야 접근이 열립니다.

### 로컬 Kubernetes 환경에서 설치할 때

Docker Desktop이나 그 밖의 로컬 Kubernetes 환경에서 실습을 진행한다면, 위의 표준 설치 절차만으로는 부족할 수 있습니다. 로컬 클러스터에 맞춘 전체 설정과 구성 단계는 공식 문서의 "Running Argo CD Locally" 가이드에 따로 정리되어 있으므로, 로컬 환경이라면 그 지침을 함께 참고하는 것이 안전합니다.

설치가 끝나 `argocd` 네임스페이스에 컴포넌트들이 배포되었다면, 이제 이 인스턴스에 실제로 접근하여 CLI로 로그인하는 단계로 넘어갈 준비가 된 것입니다.

## Argo CD 접근 및 CLI 로그인

`argocd` 네임스페이스에 컴포넌트가 자리 잡았다면, 이제 그 인스턴스와 실제로 대화할 차례입니다. 대화의 통로는 두 가지 — 브라우저에서 여는 Web UI, 그리고 자동화와 CI 연동에 유리한 `argocd` CLI입니다. 둘 다 결국 API Server에 도달해야 하므로, 접근을 여는 작업과 CLI를 로그인 상태로 만드는 작업이 이 장의 두 축을 이룹니다.

### CLI 내려받기

CLI는 릴리스 페이지(`https://github.com/argoproj/argo-cd/releases/latest`)에서 최신 바이너리를 받거나, macOS·Linux·WSL 환경이라면 Homebrew로 설치할 수 있습니다.

```bash
brew install argocd
```

기본 설치 상태에서 API Server가 클러스터 밖으로 노출되어 있지 않기 때문에, CLI를 갖추었더라도 먼저 접근 경로를 열어야 합니다.

### API Server를 외부에 노출하기

접근을 여는 방법은 세 가지이며, 각각 성격과 적합한 상황이 다릅니다.

| 방법 | 어떻게 여는가 | 주된 용도 |
|------|---------------|-----------|
| **LoadBalancer** | `argocd-server` 서비스 타입을 `LoadBalancer`로 변경 | 클라우드 제공자가 외부 IP를 할당해 주는 환경 |
| **Ingress** | Ingress 리소스로 라우팅 구성 | Ingress를 통한 접근 |
| **포트 포워딩** | `kubectl port-forward`로 로컬 포트에 연결 | 서비스를 노출하지 않고 붙어 보는 경우 |

LoadBalancer 방식은 서비스 타입을 바꾸는 것으로 시작합니다.

```bash
kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "LoadBalancer"}}'
```

잠시 기다리면 클라우드 제공자가 외부 IP를 할당하며, 다음 명령으로 그 값을 조회할 수 있습니다.

```bash
kubectl get svc argocd-server -n argocd -o=jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

Ingress 구성은 공식 Ingress 문서의 절차를 따릅니다. 서비스를 노출하지 않고 API Server에 연결하려면 포트 포워딩을 사용할 수 있습니다.

```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

이후 API Server는 `https://localhost:8080`으로 접근할 수 있습니다.

### CLI로 로그인하기

접근 경로가 열렸다면 로그인에 필요한 것은 자격 증명입니다. `admin` 계정의 초기 비밀번호는 자동 생성되어, 설치 네임스페이스의 `argocd-initial-admin-secret` 시크릿 안 `password` 필드에 평문으로 보관됩니다. 이 값을 `argocd` CLI로 출력할 수 있습니다.

```bash
argocd admin initial-password -n argocd
```

얻은 비밀번호와 `admin` 사용자명으로 API Server의 IP나 호스트명을 지정해 로그인합니다.

```bash
argocd login <argocd_server>
```

여기서 두 가지 실무적 상황을 짚어야 합니다.

첫째, 기본 설치는 자체 서명 인증서를 사용합니다. 정식 인증서를 구성하거나 OS가 자체 서명 인증서를 신뢰하도록 설정하지 않았다면, 이 가이드의 모든 Argo CD CLI 조작에 `--insecure` 플래그를 붙일 수 있습니다. 즉 로그인 자체도 `argocd login <argocd_server> --insecure` 형태가 됩니다.

둘째, CLI 환경이 API Server와 통신할 수 있어야 합니다. 앞서 설명한 방식으로 서버가 직접 접근 가능하지 않다면, CLI에게 포트 포워딩을 통해 접근하도록 지시하는 두 가지 방법이 있습니다. 하나는 모든 CLI 명령에 `--port-forward-namespace argocd` 플래그를 붙이는 것이고, 다른 하나는 `ARGOCD_OPTS` 환경 변수를 설정해 두는 것입니다.

```bash
export ARGOCD_OPTS='--port-forward-namespace argocd'
```

로그인이 끝나면 초기 비밀번호를 교체합니다.

```bash
argocd account update-password
```

비밀번호를 변경한 뒤에는 `argocd-initial-admin-secret`을 삭제해도 무방합니다. 이 시크릿은 초기 생성 비밀번호를 평문으로 보관하는 것 외에는 다른 용도가 없으며, 새 admin 비밀번호를 재생성해야 할 때 Argo CD가 다시 만들어 냅니다.

### Core(헤드리스) 설치를 사용하는 경우

Core 설치는 API Server와 UI를 포함하지 않으므로, 이 장에서 설명한 노출·로그인 절차가 적용되지 않습니다. `argocd login --core`로 로컬 API Server 프로세스를 띄워 CLI를 사용하며, 이 장의 노출과 로그인 단계는 건너뜁니다.
&lt;/argocd_server&gt;&lt;/argocd_server&gt;

## 배포 대상 클러스터 등록하기

지금까지의 과정으로 Argo CD 인스턴스가 실행되고 CLI가 로그인 상태에 놓였다면, 남은 질문은 "어디에 배포할 것인가"입니다. Argo CD는 자신이 실행 중인 클러스터에 배포할 수도 있고, 완전히 분리된 외부 클러스터에 배포할 수도 있습니다. 이 두 경우는 준비 작업의 유무에서 확연히 갈립니다.

### 언제 클러스터 등록이 필요한가

클러스터를 별도로 등록하는 작업은 **오직 외부 클러스터에 배포할 때만 필요**합니다. Argo CD가 실행 중인 바로 그 클러스터(내부)에 배포하는 경우에는 등록 절차가 필요 없으며, 애플리케이션의 Kubernetes API 서버 주소로 `https://kubernetes.default.svc`를 사용하면 됩니다. 이 주소는 클러스터 내부에서 사용하는 인클러스터 Kubernetes API 서버 엔드포인트입니다.

| 배포 대상 | 클러스터 등록 필요 여부 | 사용하는 API 서버 주소 |
|-----------|------------------------|------------------------|
| Argo CD가 실행 중인 같은 클러스터(내부) | 불필요 | `https://kubernetes.default.svc` |
| 별도의 외부 클러스터 | 필요 (`argocd cluster add`) | 등록 시 kubeconfig 컨텍스트에서 결정 |

외부 클러스터에 배포하려면 Argo CD가 그 클러스터를 조작할 수 있는 자격 증명을 확보해야 하고, 클러스터 등록이란 바로 그 자격 증명을 Argo CD에 넘겨 주는 과정입니다.

### 등록 가능한 클러스터 확인하기

등록의 출발점은 로컬 `kubeconfig`에 정의된 컨텍스트입니다. 아래 명령으로 현재 접근 가능한 컨텍스트 목록을 확인할 수 있습니다.

```bash
kubectl config get-contexts -o name
```

이 목록에서 배포 대상으로 삼을 컨텍스트 이름을 하나 고르는 것이 첫 단계입니다.

### `argocd cluster add`로 클러스터 등록하기

선택한 컨텍스트 이름을 `argocd cluster add` 명령에 넘기면 등록이 진행됩니다. 예를 들어 `docker-desktop` 컨텍스트를 등록하는 명령은 다음과 같습니다.

```bash
argocd cluster add docker-desktop
```

이 명령은 대상 클러스터 안에 Argo CD 전용 자격 증명 구조를 구축합니다. 구체적으로 다음이 이루어집니다.

- 해당 kubectl 컨텍스트가 가리키는 클러스터의 `kube-system` 네임스페이스에 `argocd-manager`라는 **ServiceAccount**가 설치됩니다.
- 이 서비스 계정이 admin 수준의 **ClusterRole**에 바인딩됩니다.
- Argo CD는 이 서비스 계정의 토큰을 사용하여 배포·모니터링 같은 관리 작업을 수행합니다.

### 권한 범위를 좁히기

보안상 권한을 좁히고 싶다면, `argocd-manager-role` 롤의 규칙을 수정하여 제한된 네임스페이스·그룹·종류(kind)에 대해서만 `create`, `update`, `patch`, `delete` 권한을 갖도록 조정할 수 있습니다.

다만 여기에는 넘을 수 없는 최소 요건이 있습니다. Argo CD가 정상 동작하려면 **클러스터 범위(cluster-scope)에서 `get`, `list`, `watch` 권한이 반드시 필요**합니다.

| 권한 종류 | 축소 가능 여부 | 비고 |
|-----------|----------------|------|
| `create`, `update`, `patch`, `delete` | 특정 네임스페이스·그룹·kind로 제한 가능 | 배포 대상 범위에 맞춰 좁힐 수 있음 |
| `get`, `list`, `watch` | 클러스터 범위에서 필수 | Argo CD 동작에 필요 |

### 네임스페이스 수준 설치에서의 인클러스터 배포

`namespace-install.yaml`(네임스페이스 수준 설치)을 사용하는 경우, 같은 클러스터로의 배포도 입력받은 자격 증명에 의존합니다. 이때는 아래처럼 `--in-cluster` 플래그와 네임스페이스를 지정해 등록할 수 있습니다.

```bash
argocd cluster add <context> --in-cluster --namespace <your namespace="">
```

이 방식은 네임스페이스 수준 권한만으로 동작하는 설치가 클러스터 롤 없이도 대상 클러스터로 배포할 수 있도록 자격 증명을 입력받는 경로를 제공합니다.

클러스터 등록이 끝나면, 이제 그 대상을 향해 실제 Application을 정의하고 동기화할 준비가 된 것입니다. 첫 Application을 생성할 때 지정하는 목적지(destination)로 인클러스터 주소 `https://kubernetes.default.svc`를 쓸지, 방금 등록한 외부 클러스터를 쓸지가 여기서 결정한 내용에 따라 갈립니다.
&lt;/your&gt;&lt;/context&gt;

## 첫 Application 만들고 동기화하기

지금까지의 준비 — 설치, 로그인, 그리고 필요하다면 클러스터 등록 — 는 모두 이 한 단계를 위한 것이었습니다. Argo CD에서 실제로 무언가를 배포한다는 것은 곧 **Application을 하나 정의하고 그것을 동기화한다**는 뜻입니다. Application이 매니페스트로 정의되는 Kubernetes 리소스 그룹이며 그 자체가 CRD로 구현되어 있다는 점은 앞서 정리했으므로, 여기서는 그 추상적 정의가 어떻게 구체적인 명령과 필드로 물질화되는지를 손에 잡히게 다뤄 보겠습니다.

실습에는 Argo CD가 공식적으로 제공하는 예제 저장소를 사용합니다. `https://github.com/argoproj/argocd-example-apps.git` 저장소에는 guestbook이라는 데모 애플리케이션이 담겨 있으며, Argo CD가 어떻게 동작하는지를 보여 주기 위해 마련된 것입니다. 한 가지 주의할 점이 있습니다. 이 예제 애플리케이션은 AMD64 아키텍처에서만 호환될 수 있습니다. ARM64나 ARMv7 같은 다른 아키텍처에서 실행 중이라면 의존성이나 컨테이너 이미지가 해당 플랫폼용으로 빌드되지 않아 문제가 생길 수 있으므로, 호환성을 확인하거나 아키텍처에 맞는 이미지를 직접 빌드하는 것을 고려해야 합니다.

### CLI로 Application 만들기

Application을 만들기 전에 먼저 현재 네임스페이스를 `argocd`로 설정합니다. 이후에 이어지는 명령들이 Argo CD 리소스가 위치한 네임스페이스를 대상으로 동작해야 하기 때문입니다.

```bash
kubectl config set-context --current --namespace=argocd
```

그다음 guestbook 예제 애플리케이션을 생성하는 명령은 다음과 같습니다.

```bash
argocd app create guestbook \
  --repo https://github.com/argoproj/argocd-example-apps.git \
  --path guestbook \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace default
```

이 한 줄이 하는 일을 필드 단위로 뜯어보면, Application이라는 리소스가 결국 "**어떤 소스로부터** 매니페스트를 가져와 **어느 목적지에** 배포할 것인가"를 선언하는 명세라는 사실이 분명해집니다. 각 플래그가 답하는 질문을 표로 정리하면 다음과 같습니다.

| 플래그 | 답하는 질문 | 이 예제의 값 |
|--------|-------------|--------------|
| `guestbook` (위치 인자) | 이 Application의 이름은? | `guestbook` |
| `--repo` | 목표 상태를 담은 Git 저장소는 어디인가? | `https://github.com/argoproj/argocd-example-apps.git` |
| `--path` | 저장소 안에서 매니페스트가 있는 경로는? | `guestbook` |
| `--dest-server` | 어느 클러스터의 Kubernetes API로 배포하는가? | `https://kubernetes.default.svc` |
| `--dest-namespace` | 그 클러스터의 어느 네임스페이스에 배포하는가? | `default` |

여기서 `--repo`와 `--path`는 함께 **소스(source)** 를 구성합니다. 앞서 살펴본 대로 이 소스 정보는 Repository Server가 매니페스트를 렌더링할 때 입력으로 받는 바로 그 값들 — 저장소 URL과 애플리케이션 경로 — 입니다. 반면 `--dest-server`와 `--dest-namespace`는 **목적지(destination)** 를 구성합니다. `--dest-server`에 인클러스터 주소인 `https://kubernetes.default.svc`를 지정했다는 것은, 이 애플리케이션을 Argo CD가 실행 중인 바로 그 클러스터에 배포하겠다는 뜻입니다. 만약 외부 클러스터로 배포하려 했다면 배포 대상 클러스터 등록 단계에서 얻은 클러스터를 목적지로 지정했을 것입니다. 즉 클러스터 등록에서 결정한 "어디에 배포할 것인가"가 여기서 `--dest-server` 값으로 구체화되는 것입니다.

### 같은 정의를 선언적으로 표현하기

CLI로 Application을 만드는 것은 편리하지만, GitOps의 정신에 가장 잘 부합하는 방식은 Application 자체를 매니페스트로 선언하여 Git에 두는 것입니다. Application이 Kubernetes CRD로 구현되어 있다는 사실이 바로 이것을 가능하게 합니다. 위의 CLI 명령과 동일한 내용을 선언적 매니페스트로 표현하면 다음과 같은 형태가 됩니다.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: guestbook
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/argoproj/argocd-example-apps.git
    targetRevision: HEAD
    path: guestbook
  destination:
    server: https://kubernetes.default.svc
    namespace: default
```

이 매니페스트에서 CLI 플래그가 어떻게 `spec` 아래 필드로 대응되는지 주목할 만합니다. `--repo`는 `source.repoURL`, `--path`는 `source.path`, `--dest-server`는 `destination.server`, `--dest-namespace`는 `destination.namespace`가 됩니다. 여기에 더해 `source.targetRevision`이 등장하는데, 이는 추적할 리비전을 지정하는 필드입니다. `HEAD`로 두면 기본 브랜치의 최신 상태를 따라가며, 앞서 소개한 대로 특정 브랜치·태그·커밋으로 고정할 수도 있습니다. 이렇게 Application을 선언적 매니페스트로 관리하면, Argo CD의 설정 그 자체가 버전 관리되고 감사 가능한 GitOps 대상이 됩니다.

### UI로 Application 만들기

멀티테넌트 설치처럼 Web UI가 포함된 경우라면 브라우저에서도 같은 작업을 할 수 있습니다. UI로 접속해 로그인한 뒤 **+ New App** 버튼을 누르면 애플리케이션 정의 폼이 열립니다. 채워야 할 값은 CLI 플래그와 정확히 같은 정보이며, 대응 관계는 다음과 같습니다.

| UI 입력 항목 | 입력할 값 |
|--------------|-----------|
| Application Name | `guestbook` |
| Project | `default` |
| Sync Policy | `Manual`(기본값 그대로) |
| Repository URL | `https://github.com/argoproj/argocd-example-apps.git` |
| Revision | `HEAD` |
| Path | `guestbook` |
| Cluster URL (Destination) | `https://kubernetes.default.svc` (또는 클러스터 이름으로 `in-cluster`) |
| Namespace (Destination) | `default` |

여기서 sync policy를 `Manual`로 두었다는 점을 눈여겨봐 두면 좋습니다. 이는 애플리케이션이 생성되더라도 자동으로 배포되지 않고, 사용자가 명시적으로 동기화를 실행할 때까지 대기한다는 뜻입니다. 자동 동기화로 전환하는 방법과 그 함의는 동기화 정책과 라이프사이클 훅을 다루는 부분에서 별도로 살펴봅니다. 폼을 모두 채운 뒤 상단의 **Create** 버튼을 누르면 `guestbook` 애플리케이션이 생성됩니다.

### 생성 직후의 상태 확인하기

Application을 만든 직후, 그 상태를 CLI로 조회할 수 있습니다.

```bash
argocd app get guestbook
```

이 명령의 출력에는 서버 주소, 네임스페이스, 저장소 URL, 경로, sync policy 같은 메타데이터와 함께 각 리소스의 sync status·health가 나란히 표시됩니다. 방금 생성한 애플리케이션은 아직 배포되지 않았으므로 sync status가 `OutOfSync`, health가 `Missing`으로 나타납니다. 이 초기 상태의 출력과 그 의미 — sync status와 health가 서로 독립적인 별개의 열이라는 점 — 는 핵심 개념을 다룬 부분에서 이미 짚었으므로 여기서는 반복하지 않습니다. 핵심만 다시 확인하면, `guestbook-ui`라는 Deployment와 Service가 목표 상태에는 정의되어 있으나 클러스터에는 아직 존재하지 않기 때문에 두 리소스 모두 `OutOfSync`/`Missing`으로 보고된다는 것입니다.

### 동기화(sync)로 실제 배포하기

Application을 만드는 것은 "무엇을, 어디에 배포할지"를 선언한 것일 뿐, 실제 배포는 아직 일어나지 않았습니다. 라이브 상태를 목표 상태로 이동시키는 행위, 곧 **동기화**를 실행해야 리소스가 클러스터에 생성됩니다.

```bash
argocd app sync guestbook
```

이 명령은 저장소에서 매니페스트를 가져와 그것들에 대해 `kubectl apply`를 수행합니다. 이 한 문장 뒤에서 API Server가 권한을 확인하고, Repository Server가 매니페스트를 렌더링하며, Application Controller가 목표 상태와 라이브 상태를 비교해 적용하는 과정이 어떻게 분담되는지는 아키텍처를 다룬 부분에서 상세히 설명했습니다. 실질적으로는 이 명령이 완료된 뒤 guestbook 애플리케이션이 클러스터에서 실행되며, 그 리소스 구성 요소·로그·이벤트, 그리고 평가된 health status를 확인할 수 있게 됩니다.

UI에서 동기화하려면 Applications 페이지에서 guestbook 애플리케이션의 **Sync** 버튼을 누른 뒤, 열리는 패널에서 **Synchronize** 버튼을 클릭하면 됩니다. guestbook 애플리케이션을 클릭하면 리소스 트리 형태로 배포된 구성 요소들을 시각적으로 더 자세히 살펴볼 수 있습니다.

정리하면, 첫 Application의 라이프사이클은 **선언(create) → 관찰(get에서 `OutOfSync`/`Missing` 확인) → 적용(sync)** 이라는 세 박자로 완결됩니다. 이 흐름은 CLI로 하든 UI로 하든, 명령형으로 만들든 선언적 매니페스트로 관리하든 동일하며, 다만 관리 방식만 달라질 뿐입니다. 소스를 어떤 도구(Kustomize·Helm·Jsonnet·플레인 YAML)로 렌더링할지는 매니페스트 소스 도구를 다루는 부분에서 이어서 구체적으로 살펴봅니다.

## 매니페스트 소스 도구: Kustomize, Helm, Jsonnet, 플레인 YAML

Argo CD가 Git 저장소로부터 목표 상태를 뽑아낼 때, 그 저장소에 담긴 파일이 반드시 완성된 Kubernetes 매니페스트일 필요는 없습니다. 공식 문서가 밝히듯 Kubernetes 매니페스트는 여러 방식으로 지정될 수 있습니다 — Kustomize 애플리케이션, Helm 차트, Jsonnet 파일, 플레인 YAML/JSON 매니페스트 디렉터리, 그리고 구성 관리 플러그인으로 등록한 임의의 커스텀 도구가 그것입니다. 어떤 도구가 애플리케이션을 빌드하는 데 쓰이는지가 곧 그 애플리케이션의 source type이며, 실제 매니페스트 생성 작업은 Repository Server가 저장소 URL·리비전(커밋·태그·브랜치)·경로·템플릿 설정(파라미터, Helm values.yaml)을 입력받아 수행합니다. 여기서는 각 도구가 무엇을 해결하려는지, 그리고 어떤 상황에서 어느 쪽을 선택해야 하는지를 실무 관점에서 다룹니다.

이 선택이 왜 중요한지는 GitOps를 조금만 진행해 보면 곧바로 드러납니다. 환경(개발·스테이징·프로덕션), 클러스터, 규제 요건 등을 고려하기 시작하면 거의 같은 YAML이 미세한 차이만 두고 반복적으로 쌓이게 됩니다. "같은 것을 반복하지 말라(Don't Repeat Yourself)"는 원칙이 금세 무너지는 것입니다. Kustomize와 Helm은 이 중복 문제를 서로 다른 철학으로 해결합니다.

### 플레인 YAML/JSON 디렉터리

가장 단순한 소스는 완성된 Kubernetes 매니페스트가 담긴 디렉터리입니다. 저장소의 특정 경로에 `Deployment`, `Service` 같은 YAML(또는 JSON) 파일을 그대로 두면, Argo CD는 그 디렉터리의 매니페스트를 별도의 템플릿 처리 없이 클러스터에 적용합니다. 변형이 필요 없는 단일 환경 배포나, 이미 다른 도구로 렌더링해 둔 최종 산출물을 커밋해 두는 경우에 적합합니다. 다만 환경별 차이를 표현할 수단이 없으므로, 여러 환경으로 확장하는 순간 앞서 말한 YAML 중복 문제에 부딪히게 됩니다.

### Kustomize: 원본을 건드리지 않는 오버레이/패치

Kustomize는 Kubernetes에 기본 내장된 패치 프레임워크로, 원본 매니페스트를 그대로 둔 채 변경 사항을 겹쳐(overlay) 새로운 매니페스트를 렌더링합니다. 원본을 수정하는 대신 결과물을 새로 만들어 낸다는 점이 핵심입니다.

전형적인 디렉터리 구조는 공유 기반 구성을 담는 `base`와 환경별 변형을 담는 `overlays`로 나뉩니다.

```text
.
├── base
│   ├── deployment.yaml
│   └── kustomization.yaml
└── overlays
    └── dev
        └── kustomization.yaml
```

`base/kustomization.yaml`은 어떤 리소스를 읽어들일지 지정하는 단순한 파일입니다.

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
```

실제 "마법"은 오버레이에서 일어납니다. `overlays/dev/kustomization.yaml`은 `base`의 리소스를 읽어들인 뒤, 예컨대 특정 Deployment의 레플리카 수를 1에서 3으로 바꾸는 식의 패치를 적용합니다. 그 결과 Kustomize는 레플리카가 3으로 바뀐 새 Deployment 매니페스트를 렌더링해 냅니다. 명령줄에서는 `kubectl apply -k overlays/dev/`로 직접 클러스터에 적용할 수도 있는데, Argo CD를 사용할 때는 이 렌더링을 Repository Server가 대신 수행합니다.

이 방식이 유용한 이유는 환경 간 애플리케이션의 차이가 구조적으로는 크지 않기 때문입니다. 데이터베이스 자격 증명, 스케일, 컨테이너 이미지 같은 값은 환경마다 달라지지만, 선언 자체의 구조는 동일하게 유지됩니다. 따라서 공통 부분은 `base`에 두고 환경별 차이(delta)만 오버레이로 저장하면 중복을 크게 줄일 수 있습니다. 값을 미리 알고 있어 "패치"로 표현할 수 있는 경우에 특히 강력합니다.

### Helm: 값을 주입하는 템플릿·패키징

Helm은 Kubernetes 애플리케이션 배포를 위한 패키지 매니저이자 템플릿 엔진입니다. Linux의 `apt`나 `dnf/yum`이 애플리케이션의 설치·관리·라이프사이클을 돕듯, Helm도 같은 목표를 지향합니다. Helm은 **차트(chart)** 라는 패키지화·템플릿화된 YAML 매니페스트 묶음으로 구성되며, 사용자가 템플릿에 정의된 파라미터에 값을 주입하면 이를 매니페스트에 채워 넣어 **릴리스(release)** 를 만들어 냅니다. 릴리스는 클러스터에 배포되는 최종 YAML의 표현이며, 그 정보는 Kubernetes 클러스터에 시크릿으로 저장됩니다.

명령줄에서 차트를 저장소에 추가하고 값을 주입해 배포하는 흐름은 다음과 같습니다.

```bash
helm repo add akuity-demos https://akuity.github.io/demo-helm-charts/
helm install myapp --create-namespace --namespace example \
  --set replicaCount=3 akuity-demos/simple-go
```

이 예에서 `--set replicaCount=3`처럼 배포마다 달라지는 값만 넘기면 되고, 나머지 구조는 차트가 담고 있습니다. 즉 애플리케이션마다 저장할 것은 **각 배포에 해당하는 values 파일**뿐입니다. Argo CD에서 Helm 차트를 소스로 지정하면, Repository Server가 Helm values를 입력으로 받아 차트를 렌더링하고 최종 Kubernetes 매니페스트를 생성합니다.

Helm은 클러스터에 대해 미리 알 수 없는 값을 다뤄야 할 때 빛을 발합니다. 대표적인 예가 Ingress의 `host` 필드처럼 배포 대상에 따라 달라지는 FQDN입니다. 이런 값은 사전에 패치로 고정하기 어렵기 때문에, 구성을 파라미터화하여 배포 시점에 값만 공급하는 방식이 자연스럽습니다. 또한 ISV(독립 소프트웨어 벤더)가 제공하는 서드파티 애플리케이션 스택을 소비할 때도 Helm이 사실상 표준 배포 수단으로 쓰입니다.

### Jsonnet

Argo CD는 Jsonnet 파일도 매니페스트 소스로 지원합니다. 이 파일들이 Repository Server를 통해 Kubernetes 매니페스트로 렌더링됩니다. Kustomize의 패치나 Helm의 값 주입과는 다른 접근으로 매니페스트를 생성하려는 팀에 선택지가 됩니다.

### 커스텀 도구: 구성 관리 플러그인

기본 제공되는 Kustomize·Helm·Jsonnet·플레인 YAML만으로 요구가 충족되지 않을 때는, 임의의 커스텀 도구를 구성 관리 플러그인(configuration management plugin)으로 등록하여 사용할 수 있습니다. 이 경우에도 역할 분담은 동일합니다 — 플러그인은 파일로부터 매니페스트를 생성하는 역할을 하고, Argo CD는 그 결과를 목표 상태로 삼아 조정합니다.

### 어느 도구를 선택할 것인가

각 소스 도구의 성격과 적합한 상황을 정리하면 다음과 같습니다.

| 소스 도구 | 무엇을 하는가 | 잘 맞는 상황 |
|-----------|---------------|--------------|
| 플레인 YAML/JSON 디렉터리 | 완성된 매니페스트를 그대로 적용 | 변형이 필요 없는 단일 환경, 이미 렌더링된 산출물 |
| Kustomize | 원본을 유지한 채 오버레이/패치로 새 매니페스트 렌더링 | 값을 이미 아는 환경별 차이를 패치로 표현 |
| Helm | 파라미터화된 차트에 값을 주입해 릴리스 생성 | 배포 시점에야 정해지는 값, 서드파티 애플리케이션 소비 |
| Jsonnet | Jsonnet 파일을 매니페스트로 렌더링 | Kustomize·Helm과 다른 접근으로 구성 생성 |
| 구성 관리 플러그인 | 사용자가 등록한 커스텀 도구로 매니페스트 생성 | 위 도구로 표현되지 않는 조직 고유의 워크플로 |

주의할 점은, 이것이 "Kustomize 대(對) Helm"의 양자택일 문제가 아니라 대체로 "Kustomize 그리고 Helm"의 조합 문제라는 것입니다. 주로 원시 Kubernetes 매니페스트를 다룬다면 Kubernetes와 여러 GitOps 도구에 기본 내장된 Kustomize를 우선 활용하는 편이 자연스럽습니다. 미리 알 수 없는 값을 파라미터화해야 하거나 Helm 생태계에서 넘어온 경우라면 Helm 차트를 작성하거나 소비하는 방향이 낫습니다. 실무에서는 대부분 두 도구를 함께 사용하여, 구조적 공통점은 템플릿화하면서도 필요한 지점에는 커스터마이징 여지를 남깁니다. 각 도구로 렌더링된 매니페스트를 저장소에서 어떻게 조직하는지에 대한 구체적 지침은 GitOps 저장소 구조와 매니페스트 관리 모범 사례를 다루는 부분에서 이어집니다.

## 동기화 정책과 라이프사이클 훅

첫 Application을 만들 때 sync policy를 `Manual`로 두면 애플리케이션이 생성되어도 사용자가 명시적으로 동기화를 실행하기 전까지는 아무것도 배포되지 않는다는 점을 확인했습니다. 이 수동 방식은 실습과 이해에는 적합하지만, GitOps의 네 번째 원칙인 지속적 조정을 온전히 구현하려면 동기화를 자동화해야 합니다. 여기서는 동기화가 언제·어떻게 일어날지를 규정하는 **동기화 정책(sync policy)** 과, 복잡한 롤아웃을 제어하기 위해 동기화 과정에 개입하는 **라이프사이클 훅**을 다룹니다.

### 수동 동기화와 자동 동기화

동기화 정책은 크게 두 가지 모드로 나뉩니다. 수동(manual) 모드에서는 라이브 상태가 목표 상태에서 벗어나 `OutOfSync`로 감지되더라도 Argo CD가 스스로 교정하지 않고, 사용자가 `argocd app sync`를 실행하거나 UI에서 동기화를 누를 때까지 기다립니다. 반면 자동(automated) 모드에서는 Application Controller가 드리프트를 감지하는 즉시 목표 상태를 적용하려 시도합니다.

| 구분 | 수동(Manual) | 자동(Automated) |
|------|--------------|-----------------|
| 드리프트 감지 | 지속적으로 수행(refresh) | 지속적으로 수행(refresh) |
| 교정 시점 | 사용자가 sync를 실행할 때 | 감지 즉시 자동으로 |
| 기본값 | 신규 Application 생성 시 기본 | 명시적으로 활성화해야 함 |
| 적합한 상황 | 배포 시점을 사람이 통제하고 싶을 때 | 완전한 자기 치유 루프가 필요할 때 |

여기서 핵심 개념을 다시 짚어 둘 필요가 있습니다. 앞서 refresh와 sync를 구분했듯이, 드리프트를 *감지*하는 것(refresh)과 그것을 *교정*하는 것(sync)은 별개의 단계입니다. 수동 모드에서도 Argo CD는 계속해서 라이브 상태와 목표 상태를 비교하여 `OutOfSync`를 표시하지만, 실제 교정 행위는 하지 않습니다. 자동 모드는 이 감지 결과에 곧바로 교정을 이어 붙이는 것입니다.

자동 동기화 정책은 Application 매니페스트의 `spec.syncPolicy.automated` 아래에 선언합니다.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: frontend-app
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/example/infra-config.git
    targetRevision: main
    path: apps/frontend
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

### 자동 동기화의 두 스위치: prune와 selfHeal

`automated` 블록 안의 두 불리언 값은 서로 다른 종류의 드리프트에 대응하며, 각각을 독립적으로 켜고 끌 수 있습니다.

- **`selfHeal: true`** — 라이브 상태가 목표 상태에서 벗어났을 때 Git에 선언된 상태를 다시 적용하여 클러스터를 교정합니다. 예를 들어 누군가 `kubectl`로 레플리카 수를 손으로 바꿔 놓았다면, self-heal이 켜져 있는 경우 Argo CD가 그 변경을 감지해 Git이 말하는 값으로 되돌립니다. 이것이 GitOps가 약속하는 자기 치유 루프의 실제 구현체입니다.
- **`prune: true`** — Git에서 리소스가 제거되었을 때 클러스터에서도 해당 리소스를 삭제합니다. prune을 켜지 않으면 Git에서 매니페스트를 지워도 클러스터에 남아 있는 리소스는 그대로 방치되어, 목표 상태와 라이브 상태 사이에 지속적인 불일치가 생깁니다.

두 값을 구분해서 이해하는 것이 중요합니다. self-heal은 "바뀐 것을 되돌리는" 교정이고, prune은 "사라진 것을 지우는" 교정입니다. 둘 다 켜야 Git이 진정한 단일 진실 공급원으로서 클러스터를 완전하게 지배하게 됩니다.

### 동기화 옵션

`syncOptions`는 동기화 동작 자체를 세밀하게 조정하는 옵션 목록입니다. 대표적인 예가 위 매니페스트에 사용된 `CreateNamespace=true`로, 목적지 네임스페이스가 존재하지 않을 때 Argo CD가 이를 자동으로 생성하도록 지시합니다.

### 라이프사이클 훅: PreSync, Sync, PostSync

단순한 `kubectl apply`만으로는 표현하기 어려운 배포 시나리오가 있습니다. 데이터베이스 스키마 마이그레이션을 배포 직전에 실행해야 하거나, 새 버전을 띄운 뒤 헬스 체크를 돌려야 하거나, 블루/그린·카나리 업그레이드처럼 단계적으로 트래픽을 전환해야 하는 경우입니다. Argo CD는 이런 복잡한 롤아웃을 지원하기 위해 동기화 과정을 세 개의 **라이프사이클 이벤트**로 나누고, 각 시점에 사용자 정의 훅을 호출합니다. 앞서 아키텍처를 다루며 언급했듯, 이 훅들을 실제로 호출하는 책임은 Application Controller가 맡습니다.

| 훅(단계) | 실행 시점 | 대표적 용도 |
|----------|-----------|-------------|
| **PreSync** | 매니페스트가 적용되기 *전* | 데이터베이스 마이그레이션 등 사전 준비 작업 |
| **Sync** | 매니페스트 적용과 함께 | 복잡한 롤아웃 로직의 본 단계 |
| **PostSync** | 모든 매니페스트가 적용되어 정상 상태가 된 *후* | 배포 후 헬스 체크·검증·알림 |

세 단계는 순서를 갖습니다. PreSync가 먼저 완료되어야 Sync가 진행되고, Sync가 끝난 뒤 PostSync가 실행되는 식입니다. 이 순차적 단계 구분 덕분에 "적용 전에 무엇을 준비하고, 적용 후에 무엇을 검증할 것인가"를 선언적으로 표현할 수 있으며, 이것이 공식 기능 목록에서 훅을 "블루/그린 및 카나리 업그레이드 같은 복잡한 애플리케이션 롤아웃을 지원하는" 장치로 설명하는 이유입니다.

정책과 훅은 서로 보완적입니다. 동기화 정책은 "동기화를 언제, 어떤 강도로 수행할 것인가"를 규정하고, 라이프사이클 훅은 "그 한 번의 동기화 안에서 어떤 순서로 일을 처리할 것인가"를 규정합니다. 자동 동기화로 지속적 조정을 켜 두면서도, 훅을 통해 각 롤아웃이 안전하게 준비·검증되도록 만드는 것이 두 기능을 함께 쓰는 전형적인 방식입니다.

## 접근 제어와 멀티테넌시 개요

멀티테넌트 설치가 여러 애플리케이션 개발팀에게 하나의 Argo CD 인스턴스를 제공하고 플랫폼 팀이 이를 유지·운영하는 방식이라는 점은 앞서 설치 방식을 다루며 확인했습니다. 그렇다면 그 위에서 "누가 무엇을 할 수 있는가"는 어떻게 규정될까요? Argo CD의 접근 제어는 두 축으로 나뉩니다. **인증(authentication)은 외부 아이덴티티 공급자에게 위임**하고, **인가(authorization)는 Argo CD 자체의 RBAC 정책으로 규정**하는 구조입니다.

### 인증은 외부 IdP에 위임한다

아키텍처를 다루며 언급했듯, API Server의 책임 중 하나가 "인증을 처리하고 외부 아이덴티티 공급자로 인증을 위임"하는 것이며, 또한 API Server는 RBAC 시행을 담당합니다. Argo CD는 사용자 계정을 자체적으로 관리하기보다, 조직이 이미 운영 중인 SSO 시스템에 로그인 판단을 넘기는 방식을 취합니다. 공식 기능 목록이 밝히는 지원 대상은 다음과 같습니다.

| 범주 | 지원 공급자 |
|------|-------------|
| 표준 프로토콜 | OIDC, OAuth2, LDAP, SAML 2.0 |
| 서비스 연동 | GitHub, GitLab, Microsoft, LinkedIn |

즉 "이 사람이 누구인가"라는 질문에 대한 답은 Argo CD 바깥의 IdP가 내리고, Argo CD는 그 결과를 받아 인가를 판단합니다. 참고로 Argo CD Core 설치에서는 OIDC 기반 인증과 Argo CD RBAC 모델, Argo CD API가 제공되지 않으며, 오직 Kubernetes RBAC에만 의존합니다. Core에서 멀티테넌시는 git push 권한에 기반한 GitOps 방식으로만 부분적으로 제공되므로, IdP 연동을 통한 멀티테넌시는 멀티테넌트 설치에서 온전히 성립합니다.

### 인가와 AppProject

인증을 통과한 사용자에게 실제 권한을 부여하는 것은 Argo CD의 RBAC 정책입니다. 공식 기능 목록은 "인가를 위한 멀티테넌시와 RBAC 정책"을 명시하고 있습니다. Argo CD가 관리하는 커스텀 리소스에는 Applications, ApplicationSets와 더불어 **AppProjects**가 포함됩니다.

모든 Application은 어떤 프로젝트에 소속됩니다. 첫 Application을 만들 때 프로젝트로 `default`를 지정했던 것을 떠올리면, Application이 항상 어떤 프로젝트에 속한다는 사실이 드러납니다.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: guestbook
  namespace: argocd
spec:
  project: team-a   # default 대신 팀 전용 AppProject에 소속
  # ...
```

정리하면, 인증은 외부 IdP가 신원을 보증하고, 인가는 Argo CD RBAC가 판단합니다. 이 요소들이 맞물려 하나의 인스턴스 위에서 여러 팀이 공존하는 멀티테넌시가 성립합니다.

## ApplicationSet으로 다중 애플리케이션/클러스터 관리하기

개별 Application을 하나씩 정의하는 방식은 대상이 소수일 때는 충분합니다. 그러나 환경(개발·스테이징·프로덕션)과 클러스터가 늘어나고 팀마다 유사한 애플리케이션을 반복 배포하기 시작하면, 거의 같은 내용의 Application 매니페스트를 손으로 여러 벌 작성하고 유지해야 하는 부담이 생깁니다. 이는 매니페스트 소스 도구를 다룰 때 언급한 "같은 것을 반복하지 말라" 문제가 Application 리소스 자체의 수준에서 다시 나타나는 것입니다. **ApplicationSet**은 바로 이 문제를 겨냥한 상위 추상화로, 하나의 선언으로 여러 개의 Application을 자동으로 생성하고 관리할 수 있게 해 줍니다.

ApplicationSet 역시 Argo CD가 관리하는 커스텀 리소스 중 하나입니다. 앞서 정리했듯 Argo CD의 리소스에는 Applications, ApplicationSets, AppProjects가 포함되며, Argo CD Core 설치에서도 사용할 수 있는 CRD가 바로 `Application`과 `ApplicationSet`입니다. ApplicationSet 리소스를 조정하는 주체는 별도의 컨트롤러인 `argocd-applicationset-controller`이며, 이 컨트롤러가 ApplicationSet 명세를 읽어 실제 `Application` 리소스들을 만들어 냅니다.

### 제너레이터와 템플릿: 두 부분으로 이루어진 모델

ApplicationSet은 개념적으로 두 부분으로 구성됩니다.

- **제너레이터(generator)** — 어떤 Application들을 만들 것인지 결정하는 파라미터 집합을 생성합니다. 예를 들어 "등록된 클러스터마다 하나씩", "Git 저장소의 각 디렉터리마다 하나씩" 같은 규칙이 여기에 해당합니다.
- **템플릿(template)** — 생성될 `Application`의 틀입니다. 제너레이터가 만들어 낸 파라미터가 이 템플릿의 필드에 채워져 최종 Application 매니페스트가 완성됩니다.

즉 제너레이터가 "몇 개를, 어떤 값으로" 만들지를 정하고, 템플릿이 "각각이 어떤 모양의 Application인지"를 정합니다. 전체 골격은 다음과 같은 형태를 띱니다.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: guestbook
  namespace: argocd
spec:
  generators:
    - # 파라미터를 생성하는 제너레이터 (아래 표의 유형 중 선택)
  template:
    # Application 매니페스트의 틀 — 제너레이터가 만든 파라미터로 채워짐
```

템플릿 필드는 Go 템플릿 문법으로도 작성할 수 있어, 제너레이터가 공급하는 값을 유연하게 참조할 수 있습니다. 각 제너레이터가 노출하는 구체적인 필드는 Argo CD 문서의 제너레이터 레퍼런스에서 확인해야 합니다.

### 사용할 수 있는 제너레이터

Argo CD가 제공하는 제너레이터는 다음과 같습니다. 어떤 축(클러스터, Git, 외부 시스템 등)을 기준으로 Application을 펼칠 것인지에 따라 선택합니다.

| 제너레이터 | 무엇을 기준으로 Application을 생성하는가 |
|-----------|------------------------------------------|
| **List** | 직접 나열한 값 목록 |
| **Cluster** | Argo CD에 등록된 클러스터들 |
| **Git** | Git 저장소의 디렉터리 또는 파일 |
| **SCM Provider** | 소스 코드 관리 제공자(SCM)에서 조회한 저장소들 |
| **Pull Request** | 열려 있는 풀 리퀘스트 |
| **Cluster Decision Resource** | 클러스터 선택 결과를 담은 리소스 |
| **Matrix** | 두 제너레이터의 조합(곱) |
| **Merge** | 여러 제너레이터 결과의 병합 |
| **Plugin** | 사용자가 등록한 커스텀 제너레이터 |

여기서 **Cluster 제너레이터**는 배포 대상 클러스터 등록 단계에서 다룬, Argo CD에 등록된 클러스터 정보를 그대로 활용합니다. 클러스터를 한 번 등록해 두면 새 클러스터가 추가될 때 동일한 애플리케이션이 자동으로 그 클러스터에도 펼쳐지도록 만들 수 있어, 다중 클러스터 관리의 핵심 수단이 됩니다. **Git 제너레이터**는 GitOps 저장소 구조와 매니페스트 관리 모범 사례에서 논의할 저장소 레이아웃과 맞물려, 저장소의 디렉터리 구조를 그대로 Application 목록으로 환원합니다. **Matrix**와 **Merge**는 단일 제너레이터로 표현하기 어려운 조건을 조합하기 위한 결합용 제너레이터이며, **Post Selector**를 사용하면 생성 결과를 다시 걸러낼 수 있습니다.

### 점진적 동기화(Progressive Syncs)

ApplicationSet이 다수의 Application을 한꺼번에 만들어 낼 때, 이들을 동시에 모두 동기화하는 대신 단계적으로 굴리고 싶은 경우가 있습니다. 이를 위한 기능이 **Progressive Syncs**로, ApplicationSet의 `spec.strategy` 아래에 롤아웃 전략을 선언합니다. 다만 이 기능은 아직 **Beta** 성숙도(v2.6.0에서 도입)이며, 롤아웃 진행 상태는 `status.applicationStatus` 아래에 기록됩니다. Beta 기능은 향후 릴리스에서 하위 호환성이 보장되지 않을 수 있으므로, 프로덕션에서 활용하기 전에 릴리스 노트를 확인하는 것이 안전합니다. 또한 이 기능은 컨트롤러 설정으로 켜야 합니다.

| 설정 위치 | 키 / 변수 |
|-----------|-----------|
| `ConfigMap/argocd-cmd-params-cm` | `applicationsetcontroller.enable.progressive.syncs` |
| `Deployment/argocd-applicationset-controller` | `ARGOCD_APPLICATIONSET_CONTROLLER_ENABLE_PROGRESSIVE_SYNCS` |

### 여러 네임스페이스에서의 ApplicationSet

여러 팀이 각자의 네임스페이스에서 ApplicationSet을 정의하도록 허용하는 **ApplicationSet in any namespace** 기능도 제공됩니다. 이 역시 Beta 성숙도(v2.8.0에서 도입)이며, 허용할 네임스페이스 목록과 SCM 제공자 관련 동작을 컨트롤러 설정으로 지정합니다.

| 설정 위치 | 키 / 변수 |
|-----------|-----------|
| `ConfigMap/argocd-cmd-params-cm` | `applicationsetcontroller.namespaces` |
| `Deployment/argocd-applicationset-controller` | `ARGOCD_APPLICATIONSET_CONTROLLER_NAMESPACES` |
| `ConfigMap/argocd-cmd-params-cm` | `applicationsetcontroller.enable.scm.providers`, `applicationsetcontroller.allowed.scm.providers` |

한 가지 실무적 유의점이 있습니다. Argo CD 설치를 다룰 때 언급했듯, 일부 Argo CD CRD(예: ApplicationSet)는 크기가 커서 클라이언트 사이드 `kubectl apply`가 적용하는 262KB 애노테이션 크기 제한을 초과합니다. 따라서 이러한 매니페스트를 적용할 때 `--server-side` 방식이 요구되는 것이며, 이 점이 설치 절차에서 서버 사이드 적용을 필수로 삼는 이유 중 하나입니다.

## GitOps 저장소 구조와 매니페스트 관리 모범 사례

Argo CD가 어떤 도구로 매니페스트를 렌더링하는지는 앞서 다뤘지만, 그 파일들을 Git 저장소 안에서 어떻게 배치하고 운영할 것인가는 별개의 문제입니다. 저장소 레이아웃은 조직의 협업 구조와 배포 워크플로가 그대로 각인되는 지점이며, 여기서의 결정이 이후 저장소의 유지보수성과 확장성을 좌우합니다. 이 장에서는 매니페스트를 *만드는* 방법이 아니라 *조직하고 운영하는* 관점의 판단 기준을 정리합니다.

### 정적으로 둘 값과 파라미터화할 값의 경계

저장소를 설계할 때 가장 먼저 마주치는 질문은 "어떤 값을 파일로 고정해 두고, 어떤 값을 파라미터로 분리할 것인가"입니다. 이 경계를 잘못 그으면 저장소가 거의 같은 YAML의 사본으로 뒤덮이거나(중복 폭발), 반대로 지나치게 추상화되어 실제 배포되는 내용을 파일만 봐서는 알 수 없게 됩니다.

실무적 판단 기준은 **값이 언제 확정되느냐**에 있습니다. 앞서 다룬 이유로, 배포 대상을 미리 알아 정적 파일에 그대로 적을 수 있는 값(레플리카 수, 환경별 리소스 스케일, 컨테이너 이미지 등)은 `base`와 `overlays` 디렉터리에 정적으로 두는 편이 낫습니다. 파일을 열면 실제 배포되는 값이 그대로 보이므로 리뷰와 감사가 쉽기 때문입니다. 반대로 대상 클러스터의 FQDN처럼 배포 시점에야 정해져 사전에 파일로 고정하기 어려운 값은 values 파일로 분리해 파라미터화하는 것이 자연스럽습니다.

이 기준을 저장소 조직 관점의 규칙으로 옮기면 다음과 같이 정리됩니다.

| 값의 성격 | 저장소에서의 배치 | 이유 |
|-----------|-------------------|------|
| 환경 간 구조적으로 동일하고 미리 아는 델타 | `base`에 공통 구성, `overlays/<env>`에 패치 | 파일에서 최종 값이 직접 보여 리뷰·감사 용이 |
| 배포 시점에야 확정되는 값 | 배포 단위별 values 파일로 분리 | 사전 고정 불가, 값만 공급하는 편이 유연 |
| 환경마다 본질적으로 다른 값(Secret, ConfigMap 등) | 환경별 디렉터리에 개별 보관 | 병합 대상이 아니라 환경 고유 자산 |

핵심은 두 도구의 정의를 다시 고르는 문제가 아니라, **"이 값을 파일로 박아 둘 것인가, values로 뽑아낼 것인가"라는 파일 조직의 판단**이라는 점입니다. 대부분의 저장소는 공통 골격을 `base`로 유지하고 환경별 차이만 오버레이 델타로 저장하되, 사전에 알 수 없는 소수의 값만 파라미터로 남기는 조합으로 수렴합니다.

### 브랜치가 아니라 디렉터리로 환경을 나눈다

애플리케이션 코드에 익숙한 조직은 본능적으로 git-flow를 GitOps 저장소에도 적용하려 합니다. 그러나 코드를 관리하는 방식과 배포 매니페스트를 관리하는 방식 사이에는 근본적인 차이가 있으며, 그중 가장 중요한 것이 **환경을 브랜치가 아니라 디렉터리로 구분해야 한다**는 원칙입니다.

환경별 장기 브랜치를 두지 않는 이유는 프로모션(promotion)이 브랜치 간 단순 병합이 아니기 때문입니다. Git을 쓰다 보면 Kubernetes YAML을 코드처럼(이름부터 infrastructure *as code*입니다) 다루고 싶어지지만, 실제로 프로모션하는 것은 코드가 아니라 매니페스트입니다. 환경별 구성은 병합 대상이 아닙니다. 예컨대 Secret과 ConfigMap은 환경마다 본질적으로 다르므로 브랜치 병합으로 옮길 수 없고, 이미지 업데이트를 프로모션하려면 매 변경을 체리픽해야 하는데 이는 그 가치에 비해 과도한 부담을 낳습니다.

따라서 권장되는 방식은 트렁크 기반 개발(trunk based development)을 Kustomize·Helm과 결합하는 것입니다. 환경별 구성은 브랜치가 아니라 폴더로 분리하고, 공통점은 템플릿화하되 커스터마이징 여지를 남깁니다. 아울러 애플리케이션을 실행하는 코드와 그것을 배포하는 매니페스트는 **별도의 저장소로 분리**하는 것이 권장됩니다. 레플리카 수 변경처럼 코드가 전혀 바뀌지 않은 업데이트가 코드베이스의 재빌드·재테스트를 유발하는 상황을 막고, 환경 변경의 승인 절차가 개발자의 지속적 통합(CI) 과정을 방해하지 않도록 하기 위해서입니다.

다음은 GitOps에서 권장되는 모노레포 레이아웃의 한 예입니다.

```text
infra-config/
  base/                     # 공유 기반 구성
    frontend/
      deployment.yaml
      service.yaml
      kustomization.yaml
  overlays/
    staging/                # 스테이징 전용 패치
      frontend/
        replicas-patch.yaml
        kustomization.yaml
    production/             # 프로덕션 전용 패치
      frontend/
        replicas-patch.yaml
        hpa.yaml
        kustomization.yaml
  policies/                 # OPA 또는 Kyverno 정책
    require-labels.yaml
    restrict-registries.yaml
```

여기서 `overlays/production`에만 HPA(Horizontal Pod Autoscaler)를 추가하는 식으로, 환경 사이의 차이가 브랜치가 아니라 디렉터리 경계로 드러난다는 점에 주목할 만합니다. 이 구조는 앞서 ApplicationSet의 Git 제너레이터를 다룬 내용과 정확히 맞물립니다.

### 정답은 없다: Conway의 법칙과 저장소 구조

GitOps를 도입하는 조직이 초기에 가장 많이 부딪히는 질문이 "가장 좋은 저장소 레이아웃은 무엇인가"입니다. 그러나 여기에는 만능 해답이 없습니다. 그 근본 원인은 Conway의 법칙에 있습니다.

> "시스템을 설계하는 모든 조직은 그 조직의 의사소통 구조를 그대로 복제한 설계를 만들어 낸다." — Melvin E. Conway

즉 저장소 디렉터리 구조(그리고 워크플로 전반)는 조직이 어떻게 구성되고 소통하는지가 결정하는 것이지, 그 반대가 아닙니다. 조직의 경계와 책임 분리가 디렉터리 구조에 큰 영향을 미칩니다. 개발자가 플랫폼 구성을 손대지 않고 플랫폼 운영자가 개발자의 소스 코드를 바꾸지 않는 것처럼, 이러한 경계(silo보다는 "boundary"라 부르는 것이 정확합니다)가 저장소 형태로 투영됩니다.

따라서 "좋은 구조"는 조직마다 다를 수밖에 없으며, 다음과 같은 서로 다른 패턴들이 각기 다른 조직 형태에 대응합니다.

| 패턴 | 특징 |
|------|------|
| 저장소 1개 : 클러스터 1개(1:1) | ApplicationSet으로 부트스트래핑하는 단순 매핑 |
| 모노레포에서 다중 클러스터 관리 | 하나의 저장소에서 여러 클러스터 구성을 폴더로 분리 |
| 팀별 저장소(Repo per team) | 팀 경계를 저장소 경계로 삼음 |
| 애플리케이션별 저장소(Repo for Application) | 애플리케이션 단위로 저장소를 분리 |

어느 패턴을 택하든 앞서 정리한 원칙 — DRY 유지, 환경은 디렉터리로 분리, 사전에 알 수 없는 값만 파라미터화 — 이 조직 고유의 최적 구조로 안내하는 지침이 됩니다.

### 시크릿은 평문으로 저장하지 않는다

GitOps 저장소 운영에서 가장 까다로운 지점 중 하나가 시크릿입니다. Git이 단일 진실 공급원이라는 원칙과, 민감 정보를 버전 관리 저장소에 평문으로 두어서는 안 된다는 보안 요구가 정면으로 충돌하기 때문입니다.

해결책의 공통 원리는 **암호화된 형태로 저장소에 커밋하고, 복호화는 오직 클러스터 안에서만 수행**하는 것입니다. 이를 위해 Sealed Secrets, HashiCorp Vault, 또는 외부 시크릿 오퍼레이터 같은 도구가 쓰입니다. 예컨대 Sealed Secrets는 클러스터의 공개 키로 값을 암호화하며, 그 값은 클러스터 안에서 동작하는 컨트롤러만이 복호화할 수 있습니다. 따라서 암호화된 산출물을 Git에 안전하게 커밋하면서도, 복호화 키는 클러스터 밖으로 나가지 않습니다.

```yaml
# SealedSecret은 값을 암호화하므로 Git에 커밋해도 안전합니다.
# 클러스터 안의 컨트롤러만이 이를 복호화할 수 있습니다.
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: database-credentials
  namespace: production
spec:
  encryptedData:
    # 클러스터의 공개 키로 암호화된 값
    username: AgBghJ8s...encrypted...data==
    password: AgCkxP2r...encrypted...data==
  template:
    metadata:
      name: database-credentials
      namespace: production
    type: Opaque
```

이 접근의 핵심은, 시크릿 역시 버전 관리되는 GitOps 파이프라인을 그대로 통과하되 민감 데이터의 노출 없이 관리된다는 점입니다. 시크릿은 클러스터 내부 컨트롤러가 관리하는 자산으로 남고, 저장소에는 그 암호화된 표현만 남습니다.

### 프로덕션에 대한 수동 변경을 금지한다

GitOps 저장소 운영에서 가장 지키기 어렵지만 가장 중요한 규칙은 **프로덕션 환경에 대한 수동 변경을 없애는 것**입니다. 모든 수정은 Git 워크플로를 거쳐야 하며, 이렇게 해야 단일 진실 공급원과 감사 가능한 이력이 온전히 유지됩니다.

이 원칙은 앞서 다룬 자동 동기화의 self-heal 동작과 직결됩니다. 누군가 `kubectl`로 클러스터를 직접 손대면, Argo CD는 이를 드리프트로 감지하고 Git에 선언된 상태로 되돌립니다. 즉 수동 변경은 되돌려지는 것이 정상 동작입니다. 만약 긴급 상황("break-glass")에서 핫픽스를 직접 적용해야 했다면, 올바른 장기적 조치는 그 변경을 즉시 Git 저장소에 커밋하여 새로운 진실 공급원으로 만드는 것입니다. 변경이 필요 없는 단순 조사·트러블슈팅이라면 읽기 전용 도구로 리소스를 살펴보아 GitOps 워크플로를 위반하지 않도록 합니다.

도입을 시작할 때는 단일 비핵심 애플리케이션부터 GitOps 워크플로로 관리하여 팀의 신뢰를 쌓고, 저장소 구조와 PR 규칙을 저위험 환경에서 다듬은 뒤 점진적으로 확장하는 것이 권장됩니다.

### 롤백은 저장소 운영의 관점에서

GitOps에서 롤백은 별도의 배포 도구를 동원하는 작업이 아니라 Git 되돌리기로 귀결된다는 점은 앞서 원칙에서 짚었습니다. 저장소 운영 관점에서 실제로 유용한 것은, 이 되돌리기가 다른 모든 변경과 **동일한 PR·리뷰 절차와 감사 이력**을 그대로 남긴다는 사실입니다. 되돌리기 자체가 새로운 불변 커밋이 되어 "언제, 누가, 무엇을 왜 되돌렸는가"가 이력에 기록되며, 에이전트는 이 되돌린 상태를 감지해 이전의 정상 상태로 클러스터를 수렴시킵니다.

```bash
# 롤백은 Git에서 커밋을 되돌리는 것만으로 충분합니다.
# GitOps 에이전트가 되돌리기를 감지해 이전 상태를 적용합니다.
git revert HEAD --no-edit

# 되돌리기 커밋을 푸시하여 조정을 트리거합니다.
git push origin main
```

`git revert`는 대상 커밋을 이력에서 지우는 대신, 그 변경을 취소하는 **새로운 커밋을 추가**한다는 점이 저장소 운영에서 중요합니다. 원래의 변경 이력과 되돌리기 이력이 모두 보존되므로 사후 조사와 컴플라이언스 감사가 온전하게 유지되고, 이 되돌리기 커밋 역시 브랜치 보호와 코드 오너, 필수 승인 같은 저장소 수준의 정책을 그대로 통과합니다. 결국 롤백조차 다른 변경과 구별 없이 리뷰 가능한 하나의 이벤트로 남는 것이 GitOps 저장소가 제공하는 운영상의 안전망입니다.
&lt;/env&gt;

## Argo CD vs Flux: 도구 선택과 트레이드오프

GitOps 오퍼레이터를 도입하려는 팀이 거의 예외 없이 마주치는 갈림길이 Argo CD와 Flux 중 무엇을 쓸 것인가입니다. 여러 GitOps 개론 자료가 이 둘을 "가장 널리 채택된 오픈소스 오퍼레이터"로 나란히 꼽으며, 둘 다 견고한 조정 루프, 다중 애플리케이션 관리, 표준 CI 워크플로와의 연동을 제공한다고 설명합니다. 이 공통 기반을 먼저 분명히 해 두는 것이 비교의 출발점입니다.

### 두 도구가 공유하는 것

Argo CD와 Flux는 근본에서 같은 모델 위에 서 있습니다. 둘 다 이 가이드 전반에서 논의한 **풀 기반(pull-based) 조정 오퍼레이터**입니다. 즉 외부 파이프라인이 클러스터로 변경을 밀어 넣는 대신, 클러스터 내부의 에이전트가 Git에서 원하는 상태를 당겨 와 적용하고, 실제 상태가 어긋나면 이를 감지·교정합니다. 이 설계가 왜 보안 경계와 다중 클러스터 확장성 측면에서 유리한지는 앞서 푸시 모델과 풀 모델을 대비하며 다뤘습니다. 이 공통점 때문에, "GitOps를 할 것인가"라는 질문에는 두 도구가 동일한 답을 주며, 실제 선택은 GitOps를 *어떻게 조직하고 운영할 것인가*라는 상위 층위에서 갈립니다.

여기서 한 가지 솔직하게 짚어 둘 점이 있습니다. 이 가이드가 근거로 삼는 자료는 Argo CD를 중심으로 서술되어 있어, Flux 고유의 내부 컴포넌트나 구체적 CRD·플래그를 이 문서가 검증된 사실로 제시할 수는 없습니다. 따라서 아래의 비교는 두 도구의 **설계 지향점(design orientation)** 차이라는, 개념 수준에서 안전하게 말할 수 있는 축에 집중합니다. Flux의 구체적 기능 목록과 최신 API는 반드시 Flux 자체 문서에서 확인해야 합니다.

### 선택을 가르는 진짜 축은 설계 지향점이다

두 도구의 가장 큰 차이는 "무엇을 할 수 있는가"의 목록이 아니라 **어떤 형태의 시스템으로 GitOps를 실현하는가**에 있습니다. Argo CD는 이 가이드에서 상세히 살펴본 대로 하나의 **중앙 인스턴스(central instance)** 지향 플랫폼입니다. API Server를 통해 Web UI와 `argocd` CLI가 붙고, 그 위에서 외부 IdP 연동 인증과 자체 RBAC 기반 인가로 여러 팀이 하나의 인스턴스를 공유하는 멀티테넌시가 성립합니다. 즉 Argo CD는 "사람이 들여다보고 조작하는 대시보드형 컨트롤 플레인"을 기본값으로 삼습니다. 동시에 컴포넌트 기반 설계 덕분에 UI·API Server를 뺀 Core(헤드리스) 모드로 축소할 수도 있어, 운영 표면적을 상황에 맞게 조절할 여지를 남깁니다.

반면 Flux는 개념적으로 별도의 중앙 UI나 자체 API 서버·RBAC 계층을 기본으로 전제하기보다, Kubernetes 네이티브 도구로 다루는 지향에 가깝게 이해되곤 합니다. 이 지향점은 Argo CD Core가 겨냥하는 사용자상 — "새로운 API나 별도 CLI를 익히지 않고 Kubernetes API에만 의존하고 싶은" DevOps 엔지니어, "UI/CLI를 개발자에게 제공하고 싶지 않은" 클러스터 관리자 — 과 개념적으로 겹칩니다. 바꿔 말하면, Argo CD의 기본 형태가 지향하는 지점(중앙 UI·API)과 Argo CD Core가 지향하는 지점(Kubernetes RBAC 기반의 최소 구성)이 대략 이 두 도구의 성향 스펙트럼 양 끝을 보여 준다고 볼 수 있습니다.

### 의사결정 기준 축

앞서 이미 다룬 Argo CD의 기능(RBAC 기반 멀티테넌시, ApplicationSet, PreSync/Sync/PostSync 훅, Core 헤드리스 모드, health 평가 등)은 여기서 다시 나열하지 않습니다. 대신 도구를 고를 때 실제로 판단해야 하는 축을 정리하면 다음과 같습니다. 각 축은 "어느 쪽이 우월한가"가 아니라 "우리 조직의 형태와 요구가 어느 쪽에 더 잘 맞는가"를 묻는 질문입니다.

| 선택 축 | 이 축이 묻는 질문 | Argo CD의 기본 성향 | Flux의 성향(설계 지향점) |
|---------|-------------------|----------------------|--------------------------|
| **UI 필요성** | 배포 상태·리소스 트리를 시각적으로 보고 조작할 사람이 있는가 | 실시간 Web UI를 기본 제공 | 번들 UI를 기본 전제로 하지 않음(Kubernetes 네이티브 도구 중심) |
| **운영 표면적** | 유지·운영해야 할 컴포넌트와 인터페이스가 얼마나 되는가 | API Server·UI 포함 시 표면적이 넓음, Core로 축소 가능 | 상대적으로 얇은 표면적 지향 |
| **인증·인가 경계** | 인증/인가를 도구 자체 계층에서 규정할 것인가, Kubernetes에 맡길 것인가 | 외부 IdP 연동 + Argo CD 자체 RBAC 계층 | Kubernetes RBAC에 기대는 방식과 잘 어울림 |
| **조직 경계·멀티테넌시** | 하나의 인스턴스를 여러 팀이 공유해야 하는가 | RBAC로 인스턴스 내 멀티테넌시 지원 | git push 권한 등 Kubernetes 경계에 기반한 분리에 친화적 |
| **학습 곡선** | 팀이 새 API·CLI를 익힐 여력이 있는가 | 별도 API·CLI·UI를 익혀 얻는 가시성이 큼 | 이미 익숙한 Kubernetes API에만 의존하고 싶을 때 부담이 적음 |
| **최소 설치** | UI·SSO·멀티클러스터 기능 없이 순수 GitOps 엔진만 필요한가 | Core(헤드리스) 모드로 최소 구성 가능 | 애초에 최소·조합형 구성을 지향 |

이 표에서 반복적으로 드러나는 패턴은, Argo CD의 강점이 대체로 **"사람과 조직을 위한 층"**(UI, 중앙 인스턴스, 자체 RBAC, 인스턴스 내 멀티테넌시)에 있다는 점입니다. 여러 개발 팀에 GitOps를 서비스로 제공하고 플랫폼 팀이 그 위에서 접근 제어와 가시성을 통제해야 하는 상황이라면, 이 층이 곧바로 가치가 됩니다. 반대로 클러스터 관리자가 단독으로 운영하며 Kubernetes RBAC만으로 충분하고 별도의 UI·API 계층을 늘리고 싶지 않다면, 그 요구는 Flux의 지향점(또는 Argo CD Core)과 더 잘 맞습니다.

### Conway의 법칙, 이번에는 도구 표면적의 관점에서

저장소 구조를 다룰 때 인용한 Conway의 법칙 — 조직의 의사소통 구조가 시스템 설계에 그대로 투영된다 — 은 도구 선택에도 새로운 각도로 적용됩니다. 저장소 레이아웃이 조직 경계를 반영하듯, **감당할 수 있는 운영 표면적** 역시 조직 형태의 함수입니다. 전담 플랫폼 팀이 있어 중앙 컨트롤 플레인과 UI·RBAC를 운영·관리할 수 있는 조직은 Argo CD의 넓은 표면적을 자산으로 흡수하지만, 소수의 관리자가 여러 클러스터를 얇게 운영하는 조직에서는 그 표면적이 유지 부담으로 돌아옵니다. 따라서 "어느 도구가 더 강력한가"보다 "우리 조직이 어느 수준의 표면적을 지속적으로 감당할 수 있는가"를 먼저 물어야, 도구 선택이 조직 현실과 어긋나지 않습니다.

결국 Argo CD와 Flux는 같은 GitOps 원리를 서로 다른 형태로 구현한 도구이며, 선택의 본질은 기능 카탈로그의 우열이 아니라 UI 필요성, 운영 표면적, 인증·인가 경계, 그리고 조직이 감당할 수 있는 복잡도라는 축 위에서의 자기 진단입니다. 이 가이드가 Argo CD를 깊이 다뤄 온 만큼, 위 축들을 기준으로 자신의 조직이 Argo CD의 중앙 인스턴스 지향에 가까운지, 아니면 더 얇은 조합형 오퍼레이터 지향에 가까운지를 판단하는 것이 실질적인 선택의 첫걸음입니다.

## 운영 시 흔한 문제와 트러블슈팅

트러블슈팅의 핵심은 새로운 지식이 아니라 순서입니다. Argo CD는 세 가지 값을 각각 다른 컴포넌트가 관장하므로, 문제를 만나면 "지금 무엇이 잘못되었는가"를 그 값들에 대응하는 진단 명령으로 좁혀 가는 것이 가장 빠릅니다. sync status가 맞지 않는가, health가 나쁜가, 아니면 Argo CD 컴포넌트 자체가 일을 못 하고 있는가 — 이 세 질문에 답하는 도구가 각각 다릅니다.

### 진단의 출발점: `argocd app get`으로 상태를 읽어 내리기

거의 모든 진단은 애플리케이션 상태를 정확히 읽는 것에서 시작합니다.

```bash
argocd app get guestbook
```

이 출력에서 두 가지를 분리해서 봐야 합니다. 상단의 `Sync Status`와 `Health Status`는 애플리케이션 전체의 요약이고, 하단 표의 `STATUS`·`HEALTH` 열은 리소스별 세부값입니다. 여기서 문제의 성격이 곧바로 갈립니다.

- `Sync Status: OutOfSync`인데 특정 리소스만 `OutOfSync`라면, 그 리소스의 라이브 상태가 목표 상태(Git이 말하는 상태)와 다르다는 뜻입니다. 문제는 "무엇을 배포할지"의 계산 단계, 즉 목표 상태와 라이브 상태의 차이에 있습니다.
- `Sync Status: Synced`인데 `Health Status`가 나쁘게 나온다면, Git이 말하는 것은 이미 적용되었으나 실행이 정상이 아니라는 뜻입니다. 문제는 매니페스트가 아니라 배포된 워크로드 쪽에 있습니다.

`Sync Status`에 함께 표시되는 리비전 해시(예: `OutOfSync from (1ff8a67)`)는 Argo CD가 비교 기준으로 삼은 커밋을 알려 줍니다. 이 해시가 방금 푸시한 커밋과 다르다면 아직 최신 상태를 관찰하지 못한 것이므로, refresh를 유도해 Git 최신 코드와 라이브 상태를 다시 비교시켜야 합니다. refresh가 "Git 최신 코드와 라이브 상태를 비교해 차이를 파악"하는 단계이고 sync가 "애플리케이션을 목표 상태로 이동시키는" 단계라는 점은 서로 다른 동작이며, 진단에서는 이 구분이 그대로 두 갈래의 조치로 이어집니다.

### 문제 계층 좁히기: 워크로드인가, Argo CD 컴포넌트인가

`Health Status`가 나쁠 때는 그 원인이 배포된 애플리케이션 자체에 있으므로, 표준 Kubernetes 도구로 내려갑니다. `argocd app get`이 알려 준 문제 리소스의 네임스페이스에서 이벤트와 로그를 확인합니다.

```bash
# 배포 대상 네임스페이스에서 최근 이벤트를 시간순으로 확인
kubectl get events -n default --sort-by=.lastTimestamp

# 문제가 되는 파드의 로그 확인
kubectl logs -n default <pod-name>
```

이미지 풀 실패, 프로브 실패, 크래시 루프 같은 원인은 sync status로는 드러나지 않고 이벤트와 파드 로그에서만 보입니다. 즉 Argo CD가 리소스를 정확히 적용했더라도 워크로드가 뜨지 못하는 상황은 Kubernetes 계층의 문제이며, Argo CD는 그 결과를 health로 보고할 뿐입니다.

반대로 애플리케이션이 아니라 Argo CD 자신이 일을 못 하는 경우도 있습니다. 이때는 `argocd` 네임스페이스의 컴포넌트 상태부터 봅니다.

```bash
kubectl get pods -n argocd
```

컴포넌트가 `CrashLoopBackOff`나 `Pending`에 머물러 있다면, 어떤 컴포넌트가 무엇을 책임지는지에 따라 어디를 봐야 할지가 정해집니다. 각 컴포넌트의 역할은 아키텍처를 다룬 부분에서 정리한 그대로이며, 진단 시에는 그 역할을 로그 확인 대상으로 환원하면 됩니다.

```bash
# 매니페스트 생성이 실패할 때(렌더링 오류 등) — Repository Server
kubectl logs -n argocd deploy/argocd-repo-server

# sync가 시작되지 않거나 교정이 일어나지 않을 때 — Application Controller
kubectl logs -n argocd statefulset/argocd-application-controller

# CLI/UI 요청 처리·인증·RBAC 문제 — API Server
kubectl logs -n argocd deploy/argocd-server
```

예를 들어 Kustomize·Helm 렌더링에 실패하거나 저장소 접근 자격 증명이 잘못되었을 때 그 오류는 매니페스트 생성을 담당하는 Repository Server 쪽에서 드러나고, 대상 클러스터로의 적용이 권한 부족으로 막히면 그 실패는 Application Controller 쪽에서 확인됩니다. 후자의 경우 원인은 대개 대상 클러스터 자격 증명의 권한 범위 문제인데, Argo CD 동작에 클러스터 범위의 `get`·`list`·`watch`가 반드시 필요하다는 제약은 클러스터 등록을 다룬 부분에서 짚은 그대로입니다.

### 증상별 빠른 진단표

자주 마주치는 증상과, 그것을 곧바로 좁혀 갈 진단 절차를 정리하면 다음과 같습니다.

| 증상 | 먼저 실행할 진단 | 무엇을 보고 판단하는가 |
|------|------------------|------------------------|
| 애플리케이션이 계속 `OutOfSync` | `argocd app get <app>` | sync policy가 수동인지, 어느 리소스가 `OutOfSync`인지 확인. 수동이면 `argocd app sync <app>` 실행 |
| Git에 푸시했는데 반영이 안 됨 | `argocd app get <app>`의 리비전 해시 확인 | 표시된 해시가 최신 커밋과 다르면 아직 관찰 전. refresh로 최신 상태와 재비교 유도 |
| `Synced`인데 `Health`가 나쁨 | `kubectl get events`, `kubectl logs`(대상 네임스페이스) | 이미지 풀·프로브·크래시 등 워크로드 문제. Argo CD가 아니라 Kubernetes 계층 |
| 매니페스트가 생성되지 않음/렌더링 오류 | `kubectl logs deploy/argocd-repo-server -n argocd` | 저장소 접근·경로·values·템플릿 오류 메시지 |
| 대상 클러스터 적용이 권한으로 실패 | `kubectl logs statefulset/argocd-application-controller -n argocd` | 권한 관련 메시지. 대상 클러스터 롤 권한 범위 재확인 |
| 컴포넌트가 뜨지 못함 | `kubectl get pods -n argocd`, 해당 파드 `kubectl logs` | 어느 컴포넌트가 실패하는지에 따라 책임 영역으로 좁힘 |
| CLI가 서버에 붙지 못함 | 접근 경로·인증서 설정 재확인 | 노출·`--insecure`·포트 포워딩은 접근을 다룬 부분 참조 |

### 드리프트가 되돌려질 때와 되돌려지지 않을 때

운영 중 흔히 겪는 혼란 하나는 "손으로 바꿨는데 원래대로 돌아간다" 혹은 반대로 "손으로 바꿨는데 그대로 남아 있다"입니다. 이 두 상황은 모두 자동 동기화 정책의 두 스위치로 설명됩니다. 라이브 상태의 변경이 되돌려진다면 self-heal이 켜져 있는 것이고, Git에서 지운 리소스가 클러스터에 남아 있다면 prune이 꺼져 있는 것입니다. 어느 쪽이 설정되어 있는지는 애플리케이션 매니페스트의 `spec.syncPolicy.automated`를 확인하면 됩니다. 따라서 "왜 되돌아가는가"라는 질문의 진단은 새로운 명령이 아니라 정책 선언을 읽는 것으로 끝납니다.

### 롤백은 진단이 아니라 되돌리기다

배포가 문제를 일으켰을 때의 올바른 조치는 클러스터를 직접 손보는 것이 아니라 진실 공급원을 되돌리는 것입니다. GitOps에서 롤백은 이전 커밋으로 되돌리는 것으로 귀결되며, `git revert`로 되돌리기 커밋을 푸시하면 Argo CD가 그 상태를 감지해 클러스터를 이전의 정상 상태로 수렴시킵니다. 트러블슈팅 관점에서 기억할 것은 하나입니다 — 긴급 상황에서 클러스터를 직접 고쳤다면, 그 변경을 즉시 Git에 커밋해 새로운 진실 공급원으로 만들어야 self-heal과의 충돌을 피할 수 있습니다.
&lt;/app&gt;&lt;/app&gt;&lt;/app&gt;&lt;/pod-name&gt;
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
* <https://oneuptime.com/blog/post/2026-02-20-gitops-principles-guide/view>
* <https://www.plural.sh/blog/what-is-gitops>
* <https://akuity.io/blog/gitops-best-practices-whitepaper>
* <https://akuity.io/blog/getting-into-gitops>
