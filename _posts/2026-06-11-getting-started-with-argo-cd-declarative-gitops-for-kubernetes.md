---
layout: post
title: "Getting Started with Argo CD: Declarative GitOps for Kubernetes"
date: 2026-06-11 22:50:29
author: "bits-bytes-nn"
categories: ["Tech Guides"]
tags: []
cover: /assets/images/tech-guides.jpg
use_math: true
---

## GitOps란 무엇인가: 원칙과 사상적 기반

Argo CD를 제대로 다루기 위해서는, 그 도구가 구현하고자 하는 운영 철학인 GitOps를 먼저 이해해야 합니다. Argo CD는 스스로를 "Kubernetes를 위한 선언적(declarative) GitOps 지속적 전달(CD) 도구"라고 규정합니다. 즉 Argo CD의 모든 설계 결정은 GitOps라는 사상에서 출발하므로, 이 원칙들을 건너뛰면 이후의 동작 방식이 그저 임의의 규칙처럼 보이게 됩니다. 이 장에서는 GitOps가 무엇이고 어떤 문제의식에서 출발했으며, 어떤 원칙 위에 서 있는지를 차근차근 다룹니다.

### GitOps가 등장한 배경

소프트웨어 개발 수명 주기의 많은 부분이 자동화되었지만, 인프라는 오랫동안 전문 팀에 의존하는 수작업 영역으로 남아 있었습니다. 애플리케이션 코드 변경, 방화벽 업데이트, 시스템 아키텍처 현대화는 흔히 지연과 비효율에 부딪혔고, 전통적인 팀은 변경 관리와 표준화된 실천법 정착에 어려움을 겪었습니다. GitOps는 2017년 조직의 인프라 변경을 다루기 위한 운영 프레임워크로 만들어졌으며, 코드에 대한 단일 진실 공급원(single source of truth) 역할을 하여 운영 환경에 대한 통제력을 제공합니다.

핵심 통찰은 단순합니다. 애플리케이션 개발에서 이미 검증된 모범 사례(버전 관리, 협업, 코드 리뷰, CI/CD)를 인프라 자동화에도 그대로 적용하자는 것입니다. 애플리케이션 소스 코드가 빌드될 때마다 동일한 바이너리를 생성하듯, 코드로 저장된 구성 파일은 배포될 때마다 동일한 인프라 환경을 생성합니다. 이렇게 하면 애플리케이션 정의·구성·환경이 선언적이고 버전 관리되며, 배포와 수명 주기 관리가 자동화되고 감사 가능하며 이해하기 쉬워집니다. Argo CD가 존재하는 이유 역시 정확히 이 명제 위에 서 있습니다.

특히 Kubernetes의 등장은 GitOps를 결정적으로 가속했습니다. Kubernetes가 워낙 선언적인 성격을 띤 탓에, 기존의 Infrastructure as Code(IaC) 도구로는 부족했고, 그 선언적 본성을 적극적으로 활용하기 위해 Argo CD나 Flux 같은 관리 도구가 부상했습니다. 다시 말해 GitOps는 Kubernetes라는 토대 위에서 가장 자연스럽게 꽃피는 운영 모델입니다.

### 명령형 vs. 선언형: GitOps의 출발점

GitOps를 이해하는 가장 어려우면서도 가장 중요한 지점은 "선언형(declarative)"이라는 단어의 무게입니다. 인프라를 기술하는 방식은 명령형(imperative)과 선언형이라는 양극단으로 나눌 수 있는데, 둘의 차이는 본질적으로 하나로 압축됩니다. 명령형은 *어떻게(how)* 할 것인지를 다루고, 선언형은 *무엇을(what)* 원하는지를 다룹니다.

비유하자면, 명령형은 누군가에게 프로젝트를 맡기면서 단계별 절차와 결과물 제출 방식까지 일일이 지정하는 것이고, 선언형은 구체적인 지시 없이 원하는 최종 결과만 정의해 두고 그 과정은 위임하는 것입니다. 선언형 접근은 최종 상태(end state)를 정의하는 데 집중하므로 프로그래밍을 크게 단순화할 수 있다는 점에서 선호됩니다. Kubernetes는 이러한 선언형 모델의 대표적 사례로, 구성을 "명령"이 아니라 "사실(facts)"로 표현하게 해 줍니다.

| 구분 | 명령형(Imperative) | 선언형(Declarative) |
|------|--------------------|---------------------|
| 초점 | 어떻게 달성하는가 (절차) | 무엇을 원하는가 (목표 상태) |
| 표현 방식 | 단계별 지시 | 최종 상태의 사실 |
| 변경 후 상태 추적 | 절차 실행 이력에 의존 | 선언된 상태 자체가 곧 명세 |
| GitOps 적합성 | 낮음 | 높음 (Git에 저장·재현 용이) |

이 구분이 왜 결정적일까요? 선언형이어야만 "Git에 저장된 것이 곧 시스템의 desired state"라는 등식이 성립하기 때문입니다. 절차가 아니라 목표 상태가 파일로 남으면, 그 파일을 다시 적용(reconcile)하는 것만으로 언제든 동일한 시스템을 재현할 수 있고, 문제가 생겼을 때 신속하게 재현·관리할 수 있습니다.

### OpenGitOps 원칙 1.0

오늘날 GitOps는 클라우드 네이티브 기술의 기반으로 자리 잡으면서, 벤더 중립적 그룹이 정립한 OpenGitOps 원칙(v1.0.0)이라는 산업 표준을 갖추게 되었습니다. GitOps로 관리되는 시스템은 다음 네 가지 속성을 만족합니다.

| 원칙 | 설명 |
|------|------|
| **선언적(Declarative)** | GitOps로 관리되는 시스템은 그 desired state가 반드시 선언적으로 표현되어야 한다. |
| **버전 관리·불변(Versioned and Immutable)** | desired state는 불변성과 버전 관리를 강제하고, 완전한 버전 이력을 보존하는 방식으로 저장된다. |
| **자동 풀(Pulled Automatically)** | 소프트웨어 에이전트가 소스로부터 desired state 선언을 자동으로 가져온다. |
| **지속적 조정(Continuously Reconciled)** | 소프트웨어 에이전트가 실제 시스템 상태를 지속적으로 관찰하며 desired state를 적용하려 시도한다. |

이 네 원칙은 GitOps의 "목표"를 정의하지만, *어떻게* 구현할지를 일러 주지는 않는다는 점을 유념해야 합니다. 즉 원칙은 구현을 안내하는 나침반이지, 구현 매뉴얼이 아닙니다. Argo CD는 바로 이 네 원칙을 Kubernetes 위에서 구체적인 기계로 옮긴 결과물입니다.

각 원칙을 Argo CD의 관점에서 풀어 보면 다음과 같습니다.

- **선언적**: 매니페스트는 Kustomize 애플리케이션, Helm 차트, Jsonnet 파일, 평이한 YAML/JSON 디렉터리 등 여러 방식으로 표현할 수 있으나, 어떤 경우든 "원하는 상태"를 선언합니다.
- **버전 관리·불변**: Git 저장소를 desired state의 단일 진실 공급원으로 사용합니다. 모든 코드 요소를 추적할 수 있어 롤백이 단순해지고, 변경은 풀 리퀘스트(PR)/머지 리퀘스트(MR)를 통해 추적됩니다. 머지 커밋 자체가 감사 로그(audit trail) 역할을 합니다.
- **자동 풀**: Argo CD는 Git 저장소를 desired application state를 정의하는 단일 진실 공급원으로 사용하며, 변경을 가져와 대상 환경에 자동으로 반영할 수 있습니다.
- **지속적 조정**: Argo CD는 Kubernetes 컨트롤러로 구현되어 실행 중인 애플리케이션을 지속적으로 모니터링하고, 현재의 라이브 상태(live state)를 Git에 명시된 desired target state와 비교합니다.

### 소프트웨어 에이전트와 자가 치유

GitOps에서 "지속적 조정"을 실제로 수행하는 주체가 소프트웨어 에이전트입니다. 이 에이전트는 소스로부터 desired state 선언을 자동으로 풀해 와 시스템의 실제 상태와 비교하고, 둘 사이에 차이(divergence)가 생기면 즉각적으로 알립니다. 노드나 파드가 실패하거나 단순한 사람의 실수가 발생하더라도, 에이전트는 운영 팀에게 즉각적인 피드백과 제어 루프를 제공합니다.

이 동작을 좀 더 형식적으로 이해해 봅시다. desired state를 \( S\_{desired} \), 클러스터의 라이브 상태를 \( S\_{live} \)라 하면, 에이전트의 목표는 다음 관계가 유지되도록 끊임없이 작동하는 것입니다.

$$
S\_{live} \;\longrightarrow\; S\_{desired} \quad \text{whenever} \quad S\_{live} \neq S\_{desired}
$$

Git에 정의되지 않은 수동 변경이나 오류, 즉 구성 드리프트(configuration drift)는 이 자동화에 의해 덮어쓰여, 환경은 Git에 정의된 desired state로 수렴합니다. Argo CD에서는 라이브 상태가 target state에서 벗어난 애플리케이션을 `OutOfSync` 상태로 간주하고, 그 차이를 보고·시각화하며, 자동 또는 수동으로 라이브 상태를 desired state로 되돌리는 수단을 제공합니다. 이때 동작하는 동기화 메커니즘과 드리프트 감지, 자가 치유(self-healing)의 내부 동작은 이후의 동기화 메커니즘을 다루는 장에서 깊이 살펴봅니다.

### GitOps가 가져오는 변화와 그 대가

GitOps의 이점은 분명합니다. 단일 도구로 인프라와 애플리케이션 수명 주기를 통합 관리하여 팀 간 협업과 조율이 향상되고, 오류가 줄며, 문제 해결이 빨라집니다. desired state가 Git에 선언되면 추가 변경은 시스템 전반에 자동으로 적용되고, 변경을 위해 클러스터 자격 증명을 따로 사용할 필요가 없어 배포 시간이 줄어듭니다. 또한 Kubernetes 클러스터 구성, Docker 이미지, 클라우드 인스턴스, 온프레미스 자원에 이르기까지 모든 인프라의 일관성을 유지하는 데 도움을 줍니다.

다만 GitOps는 공짜가 아닙니다. 협업을 전제로 한 모든 변화가 그렇듯, GitOps는 참여자 전원의 규율과 새로운 방식에 대한 헌신을 요구합니다. 승인 과정은 인프라에 "위원회식 변경(change by committee)"이라는 요소를 끌어들여, 빠른 수동 변경에 익숙한 엔지니어에게는 번거롭고 시간이 걸리는 것처럼 느껴질 수 있습니다. 운영 환경을 직접 수정하거나 무언가를 손으로 바꾸고 싶은 유혹은 강하지만, 그런 "카우보이 엔지니어링"이 줄어들수록 GitOps는 더 잘 작동합니다. 모든 변경 내역을 머지 리퀘스트와 이슈에 기록하는 습관이 GitOps 성공의 토대입니다.

정리하면, GitOps는 "선언적으로 desired state를 기술하고, 이를 버전 관리되는 불변의 저장소에 두며, 소프트웨어 에이전트가 이를 자동으로 가져와 실제 상태를 지속적으로 desired state에 수렴시키는" 운영 모델입니다. 이어지는 장에서는 이 모델을 구체적으로 구현하는 GitOps 엔진으로서 Argo CD가 어떤 역할을 맡고 어떻게 작동하는지를 살펴봅니다.

## Argo CD 개요: GitOps Engine으로서의 역할과 작동 방식

앞서 살펴본 네 가지 OpenGitOps 원칙은 "무엇을 달성해야 하는가"를 규정할 뿐, "어떻게 구현하는가"는 일러 주지 않습니다. Argo CD는 바로 그 빈칸을 Kubernetes 위에서 메우는 구체적인 기계입니다. 이 장에서는 추상적 원칙이 아니라, Argo CD가 실제로 무엇을 추상화 단위로 삼고, 내부 컴포넌트들이 어떤 순서로 메시지를 주고받으며 그 추상화를 실현하는지를 들여다봅니다.

### Application: GitOps를 담는 단일 추상 단위

Argo CD의 모든 동작은 `Application`이라는 하나의 객체로 수렴합니다. `Application`은 매니페스트로 정의되는 Kubernetes 리소스 그룹으로, Custom Resource Definition(CRD)입니다. 이를 통해 "어떤 Git 저장소의 어떤 경로를, 어떤 리비전으로, 어느 클러스터의 어느 네임스페이스에, 어떤 동기화 정책으로 배포할 것인가"를 단 하나의 선언으로 묶습니다. 다음은 전형적인 `Application` 매니페스트입니다.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: guestbook
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/argoproj/argocd-example-apps
    targetRevision: HEAD
    path: guestbook
  destination:
    server: https://kubernetes.default.svc
    namespace: guestbook
  syncPolicy:
    automated:
      selfHeal: true
      prune: true
```

이 한 장의 YAML이 곧 GitOps 그 자체입니다. `source.repoURL`과 `path`는 desired state가 어디에 선언되어 있는지를 가리키는 단일 진실 공급원의 좌표이고, `targetRevision: HEAD`는 추적할 리비전(브랜치·태그·커밋)을 지정합니다. `destination.server`에 들어간 `https://kubernetes.default.svc`는 Argo CD가 실행 중인 바로 그 클러스터 내부로 배포한다는 뜻이며, 외부 클러스터로 보낼 때는 등록된 클러스터의 API 서버 주소가 들어갑니다(외부 클러스터 등록은 별도의 장에서 다룹니다). `syncPolicy.automated` 아래의 `selfHeal`과 `prune`은 자동 동기화 정책을 제어하는 필드이며, 이 정책 필드들이 실제로 어떤 동작을 유발하는지는 동기화 메커니즘을 다루는 장에서 깊이 파고듭니다.

핵심은, Git URL·대상 클러스터·경로·동기화 정책이 흩어진 설정이 아니라 하나의 Kubernetes 객체로 묶인다는 점입니다. `Application` 자체가 클러스터에 저장되는 리소스이므로, Argo CD를 구동하는 행위 역시 선언적 객체를 적용하는 행위가 됩니다.

### 세 컴포넌트가 하나의 사이클을 도는 방식

Argo CD는 컴포넌트 기반 아키텍처로 설계되어 있으며, 그 중심에는 세 가지 핵심 컴포넌트가 있습니다. 아래 아키텍처 다이어그램은 이들이 Git·Kubernetes API와 어떻게 연결되는지를 한눈에 보여 줍니다.

![Argo CD Architecture](https://argo-cd.readthedocs.io/en/stable/assets/argocd_architecture.png)

각 컴포넌트의 책임은 다음과 같이 명확히 나뉩니다.

| 컴포넌트 | 성격 | 주요 책임 |
|---|---|---|
| **API Server** | gRPC/REST 서버 | Web UI·CLI·CI/CD가 소비하는 API 노출, 애플리케이션 관리 및 상태 보고, sync·rollback·사용자 정의 작업 호출, 저장소·클러스터 자격 증명 관리(K8s 시크릿으로 저장), 외부 ID 공급자로의 인증·인가 위임, RBAC 강제, Git 웹훅 이벤트 수신·전달 |
| **Repository Server** | 내부 서비스 | 애플리케이션 매니페스트를 담은 Git 저장소의 로컬 캐시 유지, 저장소 URL·리비전(커밋·태그·브랜치)·경로·템플릿 설정(파라미터, Helm `values.yaml`)을 입력받아 Kubernetes 매니페스트를 생성·반환 |
| **Application Controller** | Kubernetes 컨트롤러 | 실행 중인 애플리케이션을 지속적으로 모니터링하며 라이브 상태와 target state를 비교, `OutOfSync` 상태를 감지하고 선택적으로 시정 조치 수행, 수명 주기 이벤트용 사용자 정의 훅(PreSync·Sync·PostSync) 호출 |

이 표만으로는 "그래서 어떻게 연결되는가"가 보이지 않습니다. 기능 목록을 넘어, 하나의 변경이 흘러가는 실제 경로를 따라가 보면 컴포넌트들의 협업이 비로소 드러납니다. 개발자가 Git 저장소에 매니페스트 변경을 머지했다고 가정해 봅시다.

1. **이벤트 수신** — 설정된 Git 웹훅이 발생하면 **API Server**가 그 이벤트를 수신·전달하는 리스너 역할을 합니다. 웹훅이 없더라도 이후 단계의 주기적 비교를 통해 같은 변경이 결국 포착됩니다.
2. **매니페스트 생성** — 비교를 수행하려면 "Git이 말하는 desired state"를 구체적인 Kubernetes 매니페스트로 풀어내야 합니다. 이 작업은 **Repository Server**가 담당합니다. 저장소 URL·리비전·경로와 템플릿 설정을 입력받아, 로컬 캐시를 갱신하고 Kustomize·Helm·Jsonnet·평이한 YAML 등을 렌더링한 최종 매니페스트를 반환합니다.
3. **비교와 판정** — **Application Controller**가 Repository Server로부터 받은 target state(rendered manifest)를 클러스터의 라이브 상태와 비교합니다. 둘이 일치하지 않으면 해당 `Application`을 `OutOfSync`로 판정하고, 일치하면 `Synced`로 둡니다.
4. **시정 조치** — 자동 동기화 정책이 켜져 있거나 사용자가 수동으로 sync를 호출하면, 컨트롤러는 저장소에서 매니페스트를 가져와 클러스터에 `kubectl apply`를 수행하여 라이브 상태를 desired state로 끌어옵니다. 이 과정에서 PreSync·Sync·PostSync 훅이 정의되어 있으면 단계별로 호출됩니다.

여기서 주목할 점은 역할의 비대칭입니다. Repository Server는 "Git이 무엇을 원하는가"만 계산하고 클러스터를 건드리지 않으며, Application Controller는 "클러스터가 지금 어떤 상태인가"를 감시하고 시정하되 매니페스트를 직접 렌더링하지는 않습니다. API Server는 이 둘 위에서 사용자·자동화·인증·자격 증명을 중개합니다. 이 분리 덕분에 일부 컴포넌트만 떼어 낸 경량 설치(Argo CD Core)나 고가용성 구성이 가능해지며, 각 설치 유형의 차이는 설치 유형을 다루는 장에서 살펴봅니다.

### 컨트롤러는 얼마나 자주, 어떻게 들여다보는가

`Application` 상태 분류의 본질은 결국 비교 연산입니다. Repository Server가 산출한 target state를 \( M\_{target} \), 클러스터에서 관측된 라이브 상태를 \( M\_{live} \)라 하면, Application Controller가 부여하는 동기화 상태는 다음과 같이 정의됩니다.

$$
\text{SyncStatus} =
\begin{cases}
\texttt{Synced} & \text{if } M\_{live} = M\_{target} \\[4pt]
\texttt{OutOfSync} & \text{if } M\_{live} \neq M\_{target}
\end{cases}
$$

이 판정은 일회성이 아니라 컨트롤러가 지속적으로 반복하는 모니터링의 산물입니다. 이때 두 가지 트리거가 비교를 유발합니다. 하나는 앞서 설명한 **Git 웹훅** 이벤트로, GitHub·BitBucket·GitLab의 변경 알림을 API Server가 받아 처리합니다. 다른 하나는 **새로고침(refresh)** 으로, 이는 Git의 최신 코드와 라이브 상태를 비교해 무엇이 다른지를 다시 계산하는 동작입니다. 웹훅이 없거나 누락된 환경에서도 이 비교 덕분에 변경은 결국 포착됩니다.

이 캐싱과 비교가 효율적으로 돌아가도록, 컨트롤러는 Redis를 중요한 캐싱 메커니즘으로 사용해 Kube API와 Git에 가해지는 부하를 줄입니다. 그래서 경량 설치 방식인 Argo CD Core에서도 Redis는 함께 포함됩니다. 결과적으로 라이브 상태가 desired state에서 벗어난 애플리케이션은 `OutOfSync`로 보고·시각화되며, 자동 또는 수동으로 라이브 상태를 desired state로 되돌릴 수 있는 수단이 제공됩니다. 이 판정 이후에 작동하는 구성 드리프트 자동 감지의 내부 동작은 동기화 메커니즘을 다루는 장에서 본격적으로 분해합니다.

### 원칙이 기능으로 번역되는 지점

이렇게 작동하는 Argo CD의 풍부한 기능들은 임의로 모인 것이 아니라, 각각이 GitOps 원칙의 구체적 실현입니다. 원칙별로 묶어 보면 그 대응 관계가 분명해집니다.

| GitOps 원칙 | 대응하는 Argo CD 기능 |
|---|---|
| 선언적(Declarative) | 다중 config 관리·템플릿 도구 지원(Kustomize·Helm·Jsonnet·plain-YAML), Helm 파라미터 오버라이드 |
| 버전 관리·불변(Versioned and Immutable) | Git에 커밋된 구성으로의 Rollback/Roll-anywhere, 애플리케이션 이벤트·API 호출에 대한 감사 추적(audit trail) |
| 자동 풀(Pulled Automatically) | 지정 환경으로의 자동 배포, 웹훅 연동(GitHub·BitBucket·GitLab), 자동화용 액세스 토큰, CI 통합용 CLI |
| 지속적 조정(Continuously Reconciled) | 애플리케이션 리소스의 헬스 상태 분석, 구성 드리프트 자동 감지·시각화, 자동/수동 동기화, 실시간 활동을 보여 주는 Web UI |

이 표는 기능을 단순 나열하는 대신, 각 기능이 어떤 사상적 약속을 지키기 위해 존재하는지를 보여 줍니다. 예컨대 멀티 클러스터 관리·SSO 통합(OIDC·OAuth2·LDAP·SAML 2.0 등)·멀티 테넌시와 RBAC·Prometheus 메트릭 같은 운영 기능들은 원칙 자체라기보다 이 네 원칙을 조직 규모에서 안전하게 운용하기 위한 토대이며, 각각 멀티 테넌시 접근 제어를 다루는 장 등에서 다시 등장합니다.

정리하면, Argo CD는 `Application`이라는 단일 선언으로 GitOps의 desired state를 캡슐화하고, Repository Server가 그 선언을 매니페스트로 렌더링하면, Application Controller가 라이브 상태와 지속적으로 비교해 `Synced`/`OutOfSync`를 판정하며, API Server가 그 모든 흐름을 사용자·자동화와 잇는 GitOps 엔진입니다. 이어지는 장에서는 이 작동의 어휘—`Application`, `Sync`, `Health`, `OutOfSync`—를 하나씩 정밀하게 정의해, 이후 실습에서 마주칠 상태 표시를 정확히 읽어 낼 수 있도록 합니다.

## Argo CD 핵심 개념 용어 사전: Application, Sync, Health, OutOfSync

앞선 장들에서 `Application`이 GitOps의 desired state를 캡슐화하는 단일 객체임을, 그리고 Application Controller가 라이브 상태와 target state를 비교해 `OutOfSync` 상태를 판정한다는 큰 그림을 살펴봤습니다. 그러나 실제 실습에서 CLI나 Web UI가 쏟아내는 상태 표시를 정확히 읽어 내려면, 그 그림을 구성하는 어휘 하나하나를 정밀하게 분리해서 이해해야 합니다. 특히 초보자가 가장 자주 혼동하는 지점은 "Sync 상태와 Health 상태가 별개의 축"이라는 사실입니다. 이 장은 그 혼동을 해소하는 데 초점을 둡니다.

### 공식 용어집을 하나의 좌표계로 묶기

Argo CD 공식 문서의 Core Concepts는 Argo CD에 특화된 용어를 다음과 같이 정의합니다. 이 정의들은 흩어진 낱말이 아니라, 서로 맞물려 하나의 좌표계를 이룹니다.

| 용어 | 정의 | 비유적으로 답하는 질문 |
|------|------|------------------------|
| **Application** | 매니페스트로 정의되는 Kubernetes 리소스 그룹(CRD) | "무엇을 하나의 묶음으로 관리하는가?" |
| **Application source type** | 애플리케이션을 빌드하는 데 사용되는 Tool(도구) | "이 매니페스트는 무엇으로 만들어지는가?" |
| **Target state** | Git 저장소의 파일로 표현된 애플리케이션의 desired state | "Git은 무엇이 배포되어야 한다고 말하는가?" |
| **Live state** | 애플리케이션의 라이브 상태. 어떤 파드 등이 실제로 배포되어 있는가 | "지금 클러스터에는 무엇이 떠 있는가?" |
| **Sync status** | 라이브 상태가 target state와 일치하는지 여부 | "배포된 것이 Git이 말하는 것과 같은가?" |
| **Sync** | 애플리케이션을 target state로 이동시키는 과정(예: 클러스터에 변경 적용) | "어긋난 것을 어떻게 되돌리는가?" |
| **Sync operation status** | sync가 성공했는지 여부 | "방금 실행한 sync 작업은 성공했는가?" |
| **Refresh** | Git의 최신 코드를 라이브 상태와 비교해 무엇이 다른지 계산 | "지금 다시 비교하면 무엇이 달라졌는가?" |
| **Health** | 애플리케이션의 건강 상태. 올바르게 실행 중이며 요청을 처리할 수 있는가 | "배포된 것이 제대로 동작하는가?" |
| **Tool** | 파일 디렉터리로부터 매니페스트를 생성하는 도구(예: Kustomize) | (Application source type과 동일) |

이 표를 관통하는 핵심은 **target state와 live state라는 두 상태**, 그리고 그 둘을 잇는 **두 동사 — Refresh와 Sync** 입니다. Refresh는 "비교를 다시 한다"는 행위이고, Sync는 "라이브를 target으로 옮긴다"는 행위입니다. 이 둘은 결코 같은 것이 아닙니다. Refresh는 Git의 최신 코드와 라이브 상태를 비교해 차이를 계산하며, Sync는 실제로 `kubectl apply`에 해당하는 변경을 가합니다. 새로고침으로 `OutOfSync`임을 확인하는 것과, 동기화로 그것을 `Synced`로 만드는 것은 별개의 단계라는 점을 분명히 해 두어야 합니다.

### 두 개의 직교하는 축: Sync status vs. Health

가장 중요한 통찰은, **Sync status와 Health가 서로 독립적인 두 축**이라는 것입니다. 하나는 "Git과 일치하는가"를 묻고, 다른 하나는 "실제로 잘 돌아가는가"를 묻습니다. 이 둘은 논리적으로 별개의 질문이므로, 네 가지 조합이 모두 가능합니다.

- Git과 일치하면서(`Synced`) 잘 돌아가는 상태 — 정상 운영의 목표 지점.
- Git과 일치하지만(`Synced`) 동작은 망가진 상태 — 매니페스트는 그대로 적용됐으나 이미지 버그 등으로 파드가 비정상일 수 있습니다.
- Git과 어긋났지만(`OutOfSync`) 현재 동작 자체는 멀쩡한 상태 — 누군가 클러스터를 손으로 바꿔 드리프트가 생겼으나 아직 서비스는 살아 있는 경우.
- Git과 어긋나고(`OutOfSync`) 동작도 안 되는 상태 — 아직 배포 자체가 이뤄지지 않은 초기 상태가 대표적입니다.

이 직교성을 식으로 정리하면, 동기화 판정과는 별개로, 헬스 판정이 라이브 리소스 자체의 동작 가능성에 대한 함수로 독립적으로 매겨진다는 뜻입니다.

$$
\text{ApplicationState} = \big(\; \underbrace{\text{SyncStatus}}\_{\text{Git과 일치하는가}},\; \underbrace{\text{HealthStatus}}\_{\text{제대로 동작하는가}} \;\big)
$$

Argo CD가 애플리케이션 리소스의 헬스 상태를 분석한다는 것은, 단지 "Git과 같은가"를 넘어 "리소스가 올바르게 실행되며 요청을 처리할 수 있는가"라는, Git만 봐서는 알 수 없는 정보를 클러스터로부터 직접 평가한다는 의미입니다.

### 첫 배포에서 두 축을 함께 읽기

이 직교성은 실습에서 곧장 눈에 들어옵니다. guestbook 애플리케이션을 생성한 직후, 아직 동기화하기 전에 `argocd app get`으로 상태를 조회하면 다음과 같은 출력을 보게 됩니다.

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

GROUP  KIND        NAMESPACE  NAME           STATUS     HEALTH
apps   Deployment  default    guestbook-ui   OutOfSync  Missing
       Service     default    guestbook-ui   OutOfSync  Missing
```

이 출력은 두 축이 동시에, 그러나 별도의 열로 보고된다는 사실을 그대로 드러냅니다. `SyncStatus`가 `OutOfSync`인 이유는 명확합니다. 애플리케이션이 아직 배포되지 않아 Kubernetes 리소스가 하나도 생성되지 않았으므로, 라이브 상태가 Git이 말하는 target state와 어긋나 있는 것입니다. 동시에 `HealthStatus`는 `Missing`으로 표시되는데, 이는 평가할 라이브 리소스 자체가 아직 존재하지 않는다는 사실을 헬스 축에서 표현한 것입니다. 같은 정보가 리소스 수준(`Deployment`, `Service`)에서도 `STATUS`와 `HEALTH` 두 열로 각각 반복됩니다. `SyncStatus` 옆 괄호 안의 `1ff8a67`은 비교 기준이 된 Git 리비전을 가리킵니다.

이제 동기화를 수행하면 두 축이 어떻게 움직이는지 관찰할 수 있습니다.

```bash
argocd app sync guestbook
```

이 명령은 저장소에서 매니페스트를 가져와 `kubectl apply`를 수행하는 **Sync** 동작이며, 이때 그 작업의 성공 여부가 바로 **Sync operation status**입니다. 동기화로 애플리케이션이 배포되고 나면, 리소스 컴포넌트와 로그, 이벤트, 그리고 평가된 헬스 상태를 확인할 수 있습니다. 즉 "Git과 같아짐"과 "제대로 돌아감"은 서로 다른 시점에 도달할 수 있으며, 이 차이야말로 두 축을 분리해서 보아야 하는 실천적 이유입니다.

### 용어들이 컴포넌트와 맞물리는 방식

이 어휘들은 앞 장에서 본 세 컴포넌트의 역할 분담과 정확히 대응합니다. Target state는 Repository Server가 매니페스트로 렌더링한 결과이고, Live state는 Application Controller가 클러스터에서 관측한 현재 모습입니다. Refresh는 이 둘을 다시 비교해 Sync status를 산출하는 동작이며, Sync는 그 차이를 해소하는 동작으로 Sync operation status라는 별도의 결과를 남깁니다. Health는 Sync 축과 무관하게 라이브 리소스의 동작 가능성을 평가하는 독립 축입니다.

정리하면, Argo CD의 상태 화면을 읽는다는 것은 곧 **"Git과 일치하는가(Sync status)"와 "제대로 동작하는가(Health)"라는 두 질문을 동시에, 그러나 따로 던지는 일**입니다. 이 두 축을 분리해 읽는 습관이 갖춰지면, `OutOfSync`가 곧 장애를 뜻하지 않으며 `Synced`가 곧 정상 동작을 보장하지도 않는다는 점이 자연스럽게 이해됩니다. 이렇게 정의한 어휘를 바탕으로, 이어지는 장들에서는 실제 Application을 생성·동기화하고, 드리프트 감지와 자가 치유가 내부적으로 어떻게 작동하는지를 본격적으로 파고듭니다.
</none>

## Argo CD 아키텍처: 세 가지 핵심 컴포넌트의 역할과 상호작용

앞선 장에서 API Server·Repository Server·Application Controller가 각각 어떤 책임을 지며, 하나의 변경이 이 셋을 거쳐 어떻게 흘러가는지를 큰 흐름으로 살펴봤습니다. 이 장은 그 흐름을 다시 반복하지 않고, 한 걸음 더 들어갑니다. 즉 각 컴포넌트가 *어떤 종류의 소프트웨어인지*(상태를 갖는가, 클러스터를 건드리는가, 누구를 신뢰하는가)를 해부하고, 이 성격 차이가 왜 Argo CD를 여러 조각으로 나누어 배포할 수 있게 만드는지를 규명합니다. 컴포넌트의 책임 목록을 외우는 것과, 그 컴포넌트가 장애·확장·보안 측면에서 어떻게 행동하는지를 이해하는 것은 전혀 다른 일이며, 이후의 설치 유형 선택과 고가용성 구성은 바로 이 후자의 이해 위에서만 합리적으로 결정할 수 있습니다.

### 추상적 책임을 구체적 워크로드로 내리기

세 컴포넌트는 개념적 역할이 아니라 클러스터에 실제로 떠 있는 Kubernetes 워크로드입니다. Argo CD를 `argocd` 네임스페이스에 설치하면, API Server는 `argocd-server`라는 Deployment로, Application Controller는 `argocd-application-controller`라는 StatefulSet으로 배치됩니다. 여기에 더해, 컨트롤러가 캐싱 메커니즘으로 의존하는 Redis가 함께 설치되며, 그 접근 암호는 설치 네임스페이스의 `argocd-redis` 시크릿의 `auth` 키에 저장됩니다. Repository Server는 매니페스트를 렌더링하는 내부 서비스로서 별도의 워크로드로 동작합니다.

컨트롤러가 Deployment가 아니라 StatefulSet으로 배포된다는 사실은 사소한 디테일이 아닙니다. 이는 컨트롤러가 클러스터 상태에 대한 모니터링·조정 책임을 안정적인 정체성으로 수행해야 하는 컴포넌트임을 드러내며, 고가용성 구성에서 다수 클러스터를 여러 컨트롤러 인스턴스로 나누어 맡기는 샤딩(sharding) 같은 고급 동작이 이 워크로드를 중심으로 정의되는 이유이기도 합니다. 반대로 API Server가 Deployment라는 점은, 그것이 본질적으로 요청을 처리하는 프런트엔드여서 복제본을 늘려 수평 확장할 수 있는 성격임을 시사합니다.

이 차이를 한눈에 정리하면 다음과 같습니다.

| 컴포넌트 | 클러스터 상태를 직접 변경하는가 | 외부에 노출되는가 | 핵심 의존 |
|---|---|---|---|
| API Server (`argocd-server`) | 아니오 (사용자 요청을 중개·위임) | 예 (Web UI·CLI·CI/CD가 소비) | 외부 ID 공급자, K8s 시크릿 |
| Repository Server | 아니오 (매니페스트만 생성·반환) | 아니오 (내부 서비스) | Git 저장소, 로컬 캐시 |
| Application Controller (`argocd-application-controller`) | 예 (`kubectl apply` 수행, 훅 호출) | 아니오 | Kube API, Redis 캐시 |

이 표가 드러내는 핵심은 "클러스터를 실제로 바꾸는 권한이 오직 Application Controller 한 곳에 집중된다"는 점입니다. API Server와 Repository Server는 각각 사용자 중개와 매니페스트 계산만 담당할 뿐, 클러스터의 라이브 리소스를 직접 손대지 않습니다. 이 권한 집중이 보안 추론과 장애 격리의 출발점이 됩니다.

### API Server: 신뢰 경계의 관문

API Server는 gRPC/REST 서버로서, Web UI·CLI·CI/CD 시스템이 소비하는 API를 노출하는 컴포넌트입니다. 앞서 그 책임 목록은 살펴봤으므로, 여기서는 그 책임들이 왜 *하나의 컴포넌트에 모여 있는가*에 주목합니다. API Server가 맡는 일들 — 애플리케이션 관리와 상태 보고, `sync`·`rollback`·사용자 정의 작업의 호출, 저장소·클러스터 자격 증명 관리, 외부 ID 공급자로의 인증·인가 위임, RBAC 강제, Git 웹훅 이벤트의 수신·전달 — 은 모두 "외부 세계와 Argo CD 내부 사이에 놓인 단 하나의 신뢰 경계"라는 한 가지 성격을 공유합니다.

특히 자격 증명의 보관 방식이 이 경계의 본질을 잘 보여 줍니다. 저장소와 클러스터의 자격 증명은 Kubernetes 시크릿으로 저장되며, API Server가 이를 관리합니다. 외부 클러스터를 등록할 때(`argocd cluster add`) 설치되는 `argocd-manager` 서비스 계정의 토큰 역시 이 자격 증명 체계의 일부로, Argo CD가 대상 클러스터에 배포·모니터링을 수행할 때 사용됩니다. 이처럼 민감한 정보의 접근과 인증·인가 판정이 API Server 한 곳에 모이기 때문에, 멀티 테넌시와 RBAC 같은 조직 규모의 접근 제어를 이 컴포넌트 위에서 일관되게 강제할 수 있습니다. 그 구체적인 RBAC 정책과 Projects 기반 격리는 멀티 테넌시 접근 제어를 다루는 장에서 본격적으로 다룹니다.

### Repository Server: 부수 효과 없는 렌더링 엔진

Repository Server의 성격은 "순수 함수(pure function)"라는 비유로 가장 정확하게 포착됩니다. 이 컴포넌트는 애플리케이션 매니페스트를 담은 Git 저장소의 로컬 캐시를 유지하면서, 다음 입력이 주어지면 그에 대응하는 Kubernetes 매니페스트를 생성해 반환하는 일을 수행합니다.

- 저장소 URL
- 리비전(커밋·태그·브랜치)
- 애플리케이션 경로(path)
- 템플릿별 설정(파라미터, Helm `values.yaml` 등)

이 동작을 형식화하면, Repository Server는 입력의 함수로서 매니페스트를 산출하는 변환으로 볼 수 있습니다.

$$
M\_{target} = f\_{repo}\big(\text{repoURL},\ \text{revision},\ \text{path},\ \text{templateSettings}\big)
$$

이 함수성에는 두 가지 실천적 함의가 있습니다. 첫째, Repository Server는 클러스터의 라이브 상태를 알지도, 변경하지도 않습니다. 그것은 오로지 "Git이 무엇을 원하는가"만 계산할 뿐, "지금 클러스터가 어떤 상태인가"에는 관여하지 않습니다. 둘째, 같은 입력에는 같은 출력이 대응하므로 결과를 캐싱하기에 자연스럽고, 그래서 로컬 캐시 유지가 이 컴포넌트의 명시적 책임으로 포함됩니다. Kustomize 애플리케이션, Helm 차트, Jsonnet 파일, 평이한 YAML/JSON 디렉터리 등 다양한 소스 타입을 동일한 인터페이스 뒤에서 렌더링할 수 있는 것도, 이 컴포넌트가 "어떤 도구로 만들었든 결국 매니페스트라는 단일 산출물로 환원한다"는 단순한 계약을 지키기 때문입니다. Kustomize와 Helm을 실제로 연결하는 방법은 Config Management Tool을 다루는 장에서 구체화합니다.

### Application Controller: 클러스터를 향한 유일한 손

Application Controller는 Kubernetes 컨트롤러로 구현되어, 실행 중인 애플리케이션을 지속적으로 모니터링하며 라이브 상태와 target state를 비교하고, `OutOfSync` 상태를 감지하면 선택적으로 시정 조치를 수행합니다. 앞 장에서 다룬 동기화 상태 판정과 Redis 캐싱을 여기서 반복하지는 않되, 컨트롤러가 가진 두 가지 고유 권능을 짚어 둘 필요가 있습니다.

첫째, 클러스터에 실제 변경을 가하는 권한은 이 컨트롤러에 집중되어 있습니다. 동기화가 일어날 때 매니페스트를 클러스터에 적용하는 행위가 일어나며, 이 권한이 한 컴포넌트에 격리되어 있다는 사실이 "어떤 변경이 클러스터에 들어갈 수 있는가"를 추론하기 쉽게 만듭니다. 둘째, 컨트롤러는 수명 주기 이벤트를 위한 사용자 정의 훅 — PreSync·Sync·PostSync — 을 호출하는 책임을 집니다. 이 훅 단계 덕분에 블루/그린이나 카나리 같은 복잡한 롤아웃을 동기화 과정에 끼워 넣을 수 있으며, 그 내부 동작은 동기화 메커니즘을 다루는 장에서 깊이 분해합니다.

여기서 다시 강조할 비대칭이 있습니다. 컨트롤러는 매니페스트를 *직접 렌더링하지 않습니다*. 비교에 필요한 target state는 Repository Server가 계산하며, 컨트롤러 자신은 그 결과를 라이브 상태와 견주어 판정하고 시정하는 데 집중합니다. 즉 "무엇을 배포해야 하는가(계산)"와 "지금 무엇이 떠 있으며 어떻게 맞출 것인가(조정)"라는 두 관심사가 Repository Server와 Application Controller로 깔끔하게 분리되어 있습니다.

### 분리가 만들어 내는 자유: Core와 HA

이러한 역할 분리는 단순한 설계 미학이 아니라, 설치 형태를 유연하게 변주할 수 있게 만드는 실질적 토대입니다. Argo CD가 컴포넌트 기반 아키텍처로 설계되었기에, 더 미니멀한 설치 — 더 적은 컴포넌트로도 핵심 GitOps 기능이 그대로 동작하는 구성 — 가 가능합니다. 아래 다이어그램은 헤드리스 설치인 Argo CD Core에서 어떤 컴포넌트가 남는지를 보여 줍니다.

![Argo CD Core 컴포넌트 구성](https://argo-cd.readthedocs.io/en/stable/assets/argocd-core-components.png)

Core 모드는 API Server와 UI를 포함하지 않고 각 컴포넌트의 경량(non-HA) 버전만 설치하지만, 그럼에도 desired state를 Git에서 가져와 Kubernetes에 적용하는 GitOps 엔진의 본질은 온전히 유지됩니다. 이것이 가능한 이유는 앞서 본 비대칭 때문입니다 — 클러스터를 조정하는 손(Application Controller)과 매니페스트를 계산하는 머리(Repository Server)가 사용자 중개 관문(API Server)과 분리되어 있으므로, 관문을 떼어 내도 핵심 루프는 멈추지 않습니다. 다만 컨트롤러가 Redis를 캐싱 메커니즘으로 의존하는 만큼, Redis 없이 컨트롤러를 돌리는 것은 권장되지 않으며, 그래서 Core 설치에도 Redis는 함께 포함됩니다.

반대 방향으로는, 같은 분리 덕분에 외부에 노출되는 API Server나 다수 클러스터를 책임지는 Application Controller에 복제본을 늘리고 자원을 더 배분하는 고가용성 구성을 컴포넌트별로 독립적으로 튜닝할 수 있습니다. Multi-Tenant·Core 설치 유형과 고가용성 구성이 각각 어떤 트레이드오프를 갖는지는 이어지는 설치 유형 선택 장에서 본격적으로 비교합니다.

정리하면, Argo CD의 세 컴포넌트는 책임의 목록으로뿐 아니라 *성격*으로 구분됩니다. API Server는 외부와 내부를 잇는 신뢰 경계의 관문이고, Repository Server는 부수 효과 없이 "Git이 원하는 것"만 계산하는 렌더링 엔진이며, Application Controller는 클러스터에 실제로 손을 대는 조정자입니다. 클러스터 변경 권한이 한 컴포넌트에 집중되고 계산·중개·조정의 관심사가 깔끔히 나뉘어 있다는 이 구조가, 경량 설치부터 고가용성 운용까지 다양한 형태를 자연스럽게 허용하는 근본 이유입니다.

## 설치 유형 선택: Multi-Tenant vs. Core vs. High Availability

앞 장에서 살펴본 컴포넌트 분리—클러스터를 조정하는 Application Controller, 매니페스트를 계산하는 Repository Server, 외부와의 관문인 API Server—는 단순한 설계 미학이 아니라 *설치 형태를 변주할 수 있는 실질적 토대*입니다. 이 장에서는 그 변주가 실제로 어떤 선택지로 구체화되는지를 다룹니다. 설치를 실행하기 *전에* 자신의 운영 맥락에 맞는 형태를 고르는 것이 중요하며, 그 선택은 곧 "어떤 매니페스트를 적용할 것인가"라는 결정으로 귀결됩니다.

### 두 갈래의 큰 분기: Multi-Tenant와 Core

Argo CD의 설치는 크게 **multi-tenant**와 **core** 두 유형으로 나뉩니다.

**Multi-Tenant 설치**는 가장 일반적인 설치 방식입니다. 조직 내 여러 애플리케이션 개발 팀에게 서비스를 제공하기 위한 형태로, 보통 플랫폼 팀이 유지·관리합니다. 최종 사용자는 API Server를 통해 Web UI나 `argocd` CLI로 Argo CD에 접근하며, CLI는 `argocd login <server-host>` 명령으로 구성해야 합니다.

**Core 설치**는 Argo CD를 헤드리스(headless) 모드로 구동하는 다른 형태의 설치입니다. 멀티 테넌시 기능이 필요 없이 클러스터 관리자가 독립적으로 Argo CD를 사용하는 경우에 가장 적합합니다. 더 적은 컴포넌트를 포함하므로 설치가 더 쉽고, API Server와 UI를 포함하지 않으며 각 컴포넌트의 경량(non-HA) 버전만 설치합니다. 그럼에도 Git에서 desired state를 가져와 Kubernetes에 적용하는 GitOps 엔진의 본질은 온전히 유지됩니다. Core 모드에서 무엇이 빠지고 무엇이 부분적으로 남는지는 Core 모드를 다루는 장에서 본격적으로 파고듭니다.

Multi-Tenant 유형에서는 다시 **고가용성 여부**라는 축으로 Non-HA와 HA 두 가지 매니페스트 묶음 중 하나를 골라야 합니다.

### Multi-Tenant 안의 선택: Non-HA와 HA, install과 namespace-install

Multi-Tenant 유형에서는 두 종류의 설치 매니페스트가 제공됩니다.

먼저 **Non-High Availability** 묶음은 프로덕션 사용에는 권장되지 않으며, 평가 기간 동안의 데모와 테스트에 주로 쓰입니다. 여기에는 두 가지 매니페스트가 있습니다.

- **`install.yaml`** — cluster-admin 권한을 갖는 표준 Argo CD 설치입니다. Argo CD가 실행되는 바로 그 클러스터에 애플리케이션을 배포하려는 경우에 사용합니다. 입력받은 자격 증명을 통해 외부 클러스터에도 여전히 배포할 수 있습니다.
- **`namespace-install.yaml`** — 클러스터 롤(cluster roles)을 필요로 하지 않고 네임스페이스 수준 권한만으로 동작하는 설치입니다. Argo CD가 실행되는 클러스터에 직접 배포할 필요가 없고 오로지 입력된 외부 클러스터 자격 증명에만 의존하려는 경우에 사용합니다. 예컨대 팀마다 별도의 Argo CD 인스턴스를 운영하고 각 인스턴스가 외부 클러스터에 배포하는 구성이 대표적입니다. 기본 롤만으로는 같은 클러스터 안에 Argo CD 리소스(`Application`, `ApplicationSet`, `AppProject`)만 배포할 수 있으며, 필요하면 새 롤을 정의해 `argocd-application-controller` 서비스 계정에 바인딩하여 확장할 수 있습니다.

`namespace-install.yaml`을 사용할 때 한 가지 주의할 점이 있습니다. Argo CD CRD가 이 매니페스트에는 포함되어 있지 않으므로 별도로 설치해야 합니다.

```bash
kubectl apply --server-side --force-conflicts \
  -k https://github.com/argoproj/argo-cd/manifests/crds\?ref\=stable
```

다음으로 **High Availability** 묶음은 프로덕션 사용에 권장되는 형태입니다. 동일한 컴포넌트를 포함하되 고가용성과 회복성(resiliency)을 위해 튜닝되어 있으며, 지원되는 컴포넌트에 대해 다중 복제본(multiple replicas)을 갖습니다. HA 묶음 역시 두 가지로 나뉩니다.

- **`ha/install.yaml`** — `install.yaml`과 같되 지원 컴포넌트가 다중 복제본으로 구성된 버전입니다.
- **`ha/namespace-install.yaml`** — `namespace-install.yaml`과 같되 다중 복제본으로 구성된 버전입니다.

HA 묶음은 이처럼 지원되는 컴포넌트별로 복제본을 늘려 고가용성과 회복성을 확보합니다.

### 네 가지 매니페스트를 하나의 표로

지금까지의 분기를 한눈에 정리하면 다음과 같습니다. 선택은 결국 "프로덕션인가", "같은 클러스터에 배포하는가", "멀티 테넌시 기능이 필요한가"라는 세 질문의 조합입니다.

| 설치 유형 | 매니페스트 | 권한 범위 | 프로덕션 권장 | 주된 용도 |
|---|---|---|---|---|
| Multi-Tenant / Non-HA | `install.yaml` | cluster-admin (클러스터 롤 포함) | 아니오 (평가·데모·테스트용) | Argo CD가 도는 클러스터 자체에 배포, 외부 클러스터도 자격 증명으로 가능 |
| Multi-Tenant / Non-HA | `namespace-install.yaml` | 네임스페이스 수준 (클러스터 롤 불필요) | 아니오 | 외부 클러스터 전용 배포, 팀별 다중 인스턴스 운용 (CRD 별도 설치 필요) |
| Multi-Tenant / HA | `ha/install.yaml` | cluster-admin | 예 | 프로덕션, 같은 클러스터 배포 + 다중 복제본 |
| Multi-Tenant / HA | `ha/namespace-install.yaml` | 네임스페이스 수준 | 예 | 프로덕션, 외부 클러스터 전용 + 다중 복제본 |
| Core | `core-install.yaml` | — (헤드리스, 경량 컴포넌트) | — | 멀티 테넌시 불필요한 클러스터 관리자, GitOps 엔진만 필요 |

`install.yaml` 계열을 사용할 때 한 가지 더 유의할 점은, 설치 매니페스트에 포함된 `ClusterRoleBinding`이 `argocd` 네임스페이스의 ServiceAccount에 바인딩되어 있다는 사실입니다. 따라서 기본값과 다른 네임스페이스에 설치하려면 그 네임스페이스 참조를 올바르게 조정해야 하며, 그렇지 않으면 권한 관련 오류가 발생할 수 있습니다. 사용자 정의 네임스페이스 설치를 포함한 실제 설치 절차는 이어지는 설치 및 초기 접근 장에서 다룹니다.

### Kustomize로 원격 매니페스트를 감싸기

위의 매니페스트들은 Kustomize를 통해서도 설치할 수 있습니다. 권장 방식은 매니페스트를 원격 리소스(remote resource)로 포함시키고, Kustomize 패치로 추가 커스터마이징을 적용하는 것입니다.

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: argocd
resources:
  - https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

특히 기본값인 `argocd`가 아닌 사용자 정의 네임스페이스에 설치하려면, `ClusterRoleBinding`이 올바른 네임스페이스의 ServiceAccount를 참조하도록 패치를 적용할 수 있습니다.

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

이 패치는 `ClusterRoleBinding`이 사용자 정의 네임스페이스의 ServiceAccount에 정확히 매핑되도록 보장하여, 배포 중 발생할 수 있는 권한 관련 문제를 예방합니다. 이 밖에 커뮤니티가 유지·관리하는 Helm 차트(`argo-helm/charts/argo-cd`)를 통한 설치도 가능합니다.

### 어떤 유형을 고를 것인가

선택의 논리는 단순한 의사결정 흐름으로 압축됩니다. UI·SSO·멀티 클러스터 기능이 필요하다면 Multi-Tenant를 택하고, 그렇지 않고 클러스터 관리자가 Kubernetes RBAC만으로 GitOps 엔진을 독립적으로 돌리고 싶다면 Core를 택합니다. Multi-Tenant를 택했다면 프로덕션 여부가 Non-HA와 HA를 가르며, 배포 대상이 외부 클러스터 전용인지 여부가 `install`과 `namespace-install`을 가릅니다.

마지막으로, 어떤 유형을 고르든 Kubernetes 버전 호환성을 확인해 두는 것이 좋습니다. Argo CD 각 버전은 특정 Kubernetes 버전들에 대해 테스트됩니다.

| Argo CD 버전 | 테스트된 Kubernetes 버전 |
|---|---|
| 3.4 | v1.35, v1.34, v1.33, v1.32 |
| 3.3 | v1.35, v1.34, v1.33, v1.32 |
| 3.2 | v1.34, v1.33, v1.32, v1.31 |

정리하면, Argo CD의 설치는 "Multi-Tenant 대 Core"라는 운영 모델의 분기와, Multi-Tenant 내부의 "Non-HA 대 HA" 및 "클러스터 권한 대 네임스페이스 권한"이라는 두 축으로 구성됩니다. 자신의 운영 맥락—사용자 구성, 프로덕션 여부, 배포 대상 클러스터—을 이 축 위에 올려놓으면 적용해야 할 매니페스트가 하나로 좁혀집니다. 이렇게 유형을 결정했다면, 이어지는 장에서 실제 초기 설치 절차를 따라가게 됩니다.
</your-custom-namespace></your-custom-namespace></server-host>

## Argo CD 설치 및 초기 접근: namespace 생성부터 CLI 로그인까지

앞 장에서 자신의 운영 맥락에 맞는 설치 유형을 골랐다면, 이제 실제로 클러스터에 Argo CD를 올리고 거기에 접속하기까지의 절차를 따라갈 차례입니다. 이 장은 가장 표준적인 Multi-Tenant 설치를 기준으로, namespace 생성 → 매니페스트 적용 → CLI 설치 → 외부 접근 경로 확보 → CLI 로그인이라는 다섯 단계를 차례로 다룹니다. 각 단계는 사소해 보이지만, 특히 "왜 `--server-side`인가", "왜 처음엔 접속이 안 되는가", "초기 비밀번호는 어디서 오는가" 같은 지점에서 초보자가 자주 막히므로 그 이유까지 함께 짚습니다.

### 사전 준비 사항

설치를 시작하기 전에 갖춰야 할 것은 단출합니다.

- `kubectl` 명령줄 도구가 설치되어 있을 것.
- kubeconfig 파일이 있을 것(기본 위치는 `~/.kube/config`).
- 클러스터에 CoreDNS가 활성화되어 있을 것. microk8s의 경우 `microk8s enable dns && microk8s stop && microk8s start`로 켤 수 있습니다.

Docker Desktop이나 그 밖의 로컬 Kubernetes 환경에서 실행한다면, 로컬 클러스터에 맞춘 별도의 설정 단계가 필요할 수 있습니다.

### namespace 생성과 매니페스트 적용

설치는 `argocd`라는 전용 네임스페이스를 만들고 그 안에 공식 매니페스트를 적용하는 것으로 시작합니다.

```bash
kubectl create namespace argocd
kubectl apply -n argocd --server-side --force-conflicts \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

이 두 명령은 모든 Argo CD 서비스와 애플리케이션 리소스가 거주할 새 `argocd` 네임스페이스를 만들고, `stable` 브랜치의 공식 매니페스트를 적용해 Argo CD를 설치합니다. 프로덕션에서는 `stable` 대신 `v3.2.0`처럼 고정된(pinned) 버전을 쓰는 것이 권장됩니다.

여기서 `--server-side --force-conflicts` 두 플래그는 장식이 아니라 필수입니다. 그 이유는 다음과 같습니다.

- `--server-side`가 필요한 이유는 일부 Argo CD CRD(예: ApplicationSet)가 클라이언트 측 `kubectl apply`가 강제하는 262KB 어노테이션 크기 한도를 초과하기 때문입니다. 서버 측 적용(server-side apply)은 `last-applied-configuration` 어노테이션을 저장하지 않으므로 이 한도 문제를 우회합니다.
- `--force-conflicts`는 이전에 다른 도구(Helm이나 과거의 `kubectl apply` 등)가 관리하던 필드의 소유권을 이번 적용이 가져올 수 있게 합니다. 새로 설치할 때는 안전하며, 업그레이드 시에는 필요합니다.

한 가지 주의할 점은 서버 측 적용의 부수 효과입니다. Argo CD 매니페스트에 정의된 필드(예: `affinity`, `env`, `probes`)에 가했던 커스텀 수정은 덮어쓰입니다. 다만 매니페스트에 명시되지 않은 필드(예: `resources` 한도/요청, `tolerations`)는 보존됩니다. 또한 앞 장에서 언급했듯 설치 매니페스트의 `ClusterRoleBinding`은 `argocd` 네임스페이스를 참조하므로, 다른 네임스페이스에 설치한다면 그 참조를 알맞게 조정해야 합니다.

UI·SSO·멀티 클러스터 기능이 필요 없다면 여기서 Core 컴포넌트만 설치하는 길로 빠질 수도 있는데, 그 경로는 Core 모드를 다루는 장에서 본격적으로 다룹니다.

설치 직후 한 가지 더 기억해 둘 점은 Redis 인증입니다. 기본 설치의 Redis는 비밀번호 인증을 사용하며, 그 비밀번호는 설치 네임스페이스의 `argocd-redis` 시크릿의 `auth` 키에 저장됩니다.

### Argo CD CLI 내려받기

다음으로 명령줄 도구를 설치합니다. 최신 버전은 릴리스 페이지에서 직접 내려받을 수 있고, macOS·Linux·WSL에서는 Homebrew로 간단히 설치할 수 있습니다.

```bash
brew install argocd
```

CLI는 이후 로그인·애플리케이션 생성·동기화 등 자동화와 CI 통합의 진입점이 됩니다.

### Argo CD 접근 경로 확보

설치를 마쳐도 곧바로 브라우저나 CLI로 접속되지 않습니다. 기본적으로 Argo CD는 클러스터 외부로 노출되지 않기 때문입니다. 외부에서 접근하려면 다음 세 가지 방법 중 하나로 `argocd-server`를 노출해야 합니다. 각 방법은 운영 맥락에 따라 트레이드오프가 다릅니다.

| 방법 | 특징 | 주된 용도 |
|---|---|---|
| Service Type LoadBalancer | 클라우드 제공자가 외부 IP를 할당 | 클라우드 환경에서의 상시 외부 노출 |
| Ingress | Ingress 리소스를 통한 라우팅(별도 구성 문서 참조) | 도메인·TLS를 갖춘 정식 노출 |
| Port Forwarding | 서비스를 노출하지 않고 로컬에서 터널링 | 로컬 테스트·평가 |

**Service Type Load Balancer** 방식은 `argocd-server` 서비스의 타입을 `LoadBalancer`로 바꿉니다.

```bash
kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "LoadBalancer"}}'
```

잠시 기다리면 클라우드 제공자가 외부 IP를 할당하며, 다음 명령으로 그 IP를 조회할 수 있습니다.

```bash
kubectl get svc argocd-server -n argocd \
  -o=jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

**Ingress** 방식은 Ingress를 통해 Argo CD를 노출하는 것으로, 구체적인 구성은 Ingress 관련 공식 문서를 따릅니다.

**Port Forwarding** 방식은 서비스를 외부로 노출하지 않고 API 서버에 연결합니다.

```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

이렇게 하면 API 서버에 `https://localhost:8080`으로 접근할 수 있습니다.

여기서 주의할 점은 인증서입니다. 기본 설치는 자체 서명 인증서(self-signed certificate)를 사용하므로, 약간의 추가 작업 없이는 깔끔하게 접근되지 않습니다. 다음 셋 중 하나로 해결합니다. (1) 정식 인증서를 구성하고 클라이언트 OS가 신뢰하도록 하거나, (2) 클라이언트 OS가 자체 서명 인증서를 신뢰하도록 설정하거나, (3) 모든 Argo CD CLI 작업에 `--insecure` 플래그를 붙입니다.

### CLI 로그인과 초기 비밀번호

이제 CLI로 로그인할 차례입니다. 먼저 초기 `admin` 계정의 비밀번호를 확보해야 합니다. 이 비밀번호는 자동 생성되어, 설치 네임스페이스의 `argocd-initial-admin-secret` 시크릿의 `password` 필드에 평문으로 저장됩니다. CLI로 다음과 같이 조회합니다.

```bash
argocd admin initial-password -n argocd
```

조회한 비밀번호와 사용자 이름 `admin`으로 Argo CD의 IP 또는 호스트명에 로그인합니다.

```bash
argocd login <argocd_server>
```

CLI 환경이 Argo CD API 서버와 통신할 수 있어야 한다는 점에 유의해야 합니다. 만약 위에서 외부 노출을 하지 않고 포트 포워딩만 쓰는 상황이라면, CLI에 포워딩 경로를 알려 줘야 합니다. 두 가지 방법이 있습니다. (1) 모든 CLI 명령에 `--port-forward-namespace argocd` 플래그를 추가하거나, (2) 환경 변수를 설정합니다.

```bash
export ARGOCD_OPTS='--port-forward-namespace argocd'
```

로그인에 성공했다면 즉시 비밀번호를 바꾸는 것이 좋습니다.

```bash
argocd account update-password
```

비밀번호를 바꾼 뒤에는 `argocd-initial-admin-secret`을 네임스페이스에서 삭제하는 것이 권장됩니다. 이 시크릿은 최초 생성된 비밀번호를 평문으로 보관하는 것 외에 다른 용도가 없으며, 언제든 안전하게 삭제할 수 있습니다. 만약 새 admin 비밀번호를 다시 생성해야 하는 상황이 오면 Argo CD가 필요에 따라 이 시크릿을 다시 만들어 줍니다.

마지막으로, 이후 단계에서 실행할 여러 명령은 현재 네임스페이스가 `argocd`로 설정되어 있다고 가정합니다. 따라서 컨텍스트의 기본 네임스페이스를 미리 맞춰 두면 편리합니다.

```bash
kubectl config set-context --current --namespace=argocd
```

이로써 클러스터에 Argo CD가 올라가고, 외부에서 접근할 경로가 열렸으며, CLI로 인증까지 마친 상태가 되었습니다. 다음으로는 (필요하다면) 외부 클러스터를 등록하고, 첫 번째 `Application`을 생성·동기화하는 단계로 넘어갑니다. 클러스터 등록은 외부 클러스터를 다루는 장에서, 첫 Application 생성과 Sync는 이어지는 장에서 CLI와 UI 두 방법으로 살펴봅니다.
</argocd_server>

## 첫 번째 Application 생성 및 Sync: CLI와 UI 두 가지 방법

설치와 로그인을 마쳤다면, 이제 실제로 무언가를 배포해 GitOps 루프가 도는 모습을 눈으로 확인할 차례입니다. 핵심은 **"Application을 생성하는 일"과 "Application을 Sync하는 일"이 분리된 두 동작**이라는 사실을 체감하는 데 있습니다. 생성은 "무엇을 배포할지 선언"하는 것이고, Sync는 "그 선언을 실제로 클러스터에 적용"하는 것입니다.

예제로는 공식 가이드가 사용하는 guestbook 애플리케이션을 그대로 따라갑니다. 이 애플리케이션은 `https://github.com/argoproj/argocd-example-apps.git` 저장소의 `guestbook` 경로에 들어 있으며, Argo CD가 어떻게 작동하는지를 시연하기 위한 용도입니다. 한 가지 주의할 점은, 이 예제 애플리케이션이 AMD64 아키텍처에서만 호환될 수 있다는 사실입니다. ARM64나 ARMv7 같은 다른 아키텍처에서 실행한다면 의존성이나 컨테이너 이미지가 해당 플랫폼용으로 빌드되지 않아 문제가 생길 수 있으므로, 호환성을 확인하거나 필요 시 아키텍처별 이미지를 직접 빌드하는 것이 좋습니다.

본격적으로 들어가기 전에, 이후 명령들이 현재 네임스페이스를 `argocd`로 가정한다는 점을 다시 한번 확인해 둡니다. 설치 단계에서 이미 설정했다면 넘어가도 좋지만, 새 셸을 열었다면 다음을 실행해 둡니다.

```bash
kubectl config set-context --current --namespace=argocd
```

또한 Argo CD가 실행 중인 그 클러스터 내부로 배포한다면 destination의 K8s API 서버 주소로 `https://kubernetes.default.svc`를 사용한다는 점을 기억해 둡니다. 외부 클러스터로 배포할 때 필요한 클러스터 등록은 별도로 다룹니다.

### CLI로 Application 생성하기

CLI에서는 `argocd app create` 명령 하나로 Application을 생성합니다. guestbook 예제의 경우 다음과 같습니다.

```bash
argocd app create guestbook \
  --repo https://github.com/argoproj/argocd-example-apps.git \
  --path guestbook \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace default
```

이 명령에서 가장 중요한 학습 포인트는, **각 플래그가 결국 `Application` 매니페스트로 번역된다**는 사실입니다. Application은 매니페스트로 정의되는 Kubernetes 리소스(CRD)이며, CLI는 매니페스트를 직접 타이핑하는 수고를 덜어 주는 편의 계층일 뿐입니다. 플래그와 그것이 답하는 질문을 나란히 두면 의미가 분명해집니다.

| CLI 플래그 | 의미 |
|---|---|
| (위치 인자) `guestbook` | Application의 이름 |
| `--repo` | desired state가 선언된 Git 저장소 URL |
| `--path` | 저장소 내에서 매니페스트가 위치한 경로 |
| `--dest-server` | 배포 대상 클러스터의 K8s API 서버 주소 |
| `--dest-namespace` | 배포 대상 네임스페이스 |

여기서 주목할 점은 이 명령이 클러스터에 무언가를 배포하지 *않는다*는 것입니다. `argocd app create`는 단지 추적할 desired state를 등록할 뿐이며, 실제 Kubernetes 리소스는 아직 만들어지지 않습니다. 그래서 생성 직후 상태를 조회하면 `SyncStatus`는 `OutOfSync`이고 `HealthStatus`는 `Missing`으로 나타납니다 — 애플리케이션이 아직 배포되지 않았고 아무 Kubernetes 리소스도 생성되지 않았기 때문입니다.

또 한 가지 눈여겨볼 부분은 위 예제에서 동기화 정책을 지정하지 않았다는 점입니다. 실제로 생성 후 `argocd app get`을 실행하면 `Sync Policy`가 `<none>`으로 표시됩니다. 따라서 이 Application은 수동(Manual) 동기화 모드로 동작하며, 사람이 명시적으로 Sync를 호출하기 전까지는 클러스터에 변경이 가해지지 않습니다.

### UI로 Application 생성하기

같은 일을 Web UI에서도 할 수 있습니다. UI는 CLI에서 입력한 것과 같은 정보를 폼 형태로 입력받습니다. 브라우저로 Argo CD의 IP/호스트명에 접속해 로그인한 뒤, Applications 페이지에서 **+ New App** 버튼을 눌러 생성 폼을 엽니다.

폼은 크게 세 묶음의 정보를 요구합니다.

- **General(일반)**: 앱 이름을 `guestbook`으로, 프로젝트를 `default`로 지정하고, 동기화 정책(Sync Policy)은 `Manual`로 그대로 둡니다.
- **Source(소스)**: 저장소 URL을 `https://github.com/argoproj/argocd-example-apps.git`로 설정하고, Revision은 `HEAD`로 두며, Path는 `guestbook`으로 지정합니다.
- **Destination(대상)**: 클러스터 URL을 `https://kubernetes.default.svc`로 설정합니다(클러스터 이름으로는 `in-cluster`를 사용할 수도 있습니다). 네임스페이스는 `default`로 지정합니다.

이 정보를 모두 채운 뒤 화면 상단의 **Create** 버튼을 누르면 `guestbook` Application이 생성됩니다. 어느 경로를 쓰든 클러스터에는 같은 `Application` 객체가 남습니다.

### CLI와 UI, 무엇을 언제 쓰는가

두 방법은 기능적으로 동등하지만, 사용 맥락이 다릅니다.

| 구분 | CLI (`argocd app create`) | Web UI (+ New App) |
|---|---|---|
| 입력 방식 | 플래그 기반 단일 명령 | 폼 기반 단계별 입력 |
| 자동화·스크립트화 | 용이 (CI 통합에 적합) | 부적합 (수동 조작) |
| 학습·탐색 | 플래그를 알아야 함 | 필드가 시각적으로 안내됨 |
| 결과물 | 동일한 `Application` 객체 | 동일한 `Application` 객체 |

처음 개념을 익히거나 무엇이 어디로 매핑되는지 탐색할 때는 UI가 직관적이고, 반복 가능한 배포나 자동화에는 CLI가 적합합니다. Argo CD CLI는 자동화와 CI 통합을 위해 제공됩니다.

### Application을 Sync(배포)하기

Application을 생성하기만 해서는 아무것도 배포되지 않습니다. desired state를 실제 클러스터로 옮기는 별도의 동작, 즉 **Sync**가 필요합니다. CLI에서는 다음 한 줄로 수행합니다.

```bash
argocd app sync guestbook
```

이 명령은 저장소에서 매니페스트를 가져와 `kubectl apply`를 수행하는 동작이며, 실행 후에는 guestbook 앱이 클러스터에서 동작하기 시작합니다. 이제 리소스 컴포넌트, 로그, 이벤트, 그리고 평가된 헬스 상태를 확인할 수 있습니다.

UI에서도 동일한 동작이 제공됩니다. Applications 페이지에서 guestbook 애플리케이션의 **Sync** 버튼을 누르면 패널이 열리고, 거기서 **Synchronize** 버튼을 누르면 동기화가 시작됩니다. guestbook 애플리케이션을 클릭하면 더 자세한 진행 상황을 살펴볼 수 있습니다.

정리하면, 첫 Application 배포는 "생성(선언 등록) → Sync(클러스터 적용)"라는 두 박자로 이뤄집니다. 생성 직후의 `OutOfSync`/`Missing`은 오류가 아니라 아직 배포되기 전이라는 정상적인 신호이며, Sync를 거쳐야 비로소 라이브 리소스가 만들어지고 헬스 평가가 가능해집니다.
</none>

## Sync 메커니즘 심층 이해: Reconciliation Loop, Drift Detection, Self-Healing

앞선 장들에서 동기화 상태(`Synced`/`OutOfSync`)가 어떻게 판정되는지, 그리고 `Application`에 자동(Automated) 또는 수동(Manual) 동기화 정책을 지정할 수 있다는 사실까지는 확인했습니다. 그러나 그 정책이 *언제 어떤 메커니즘으로* 동작하는지는 미뤄 두었습니다. 이 장은 바로 그 빈칸을 채웁니다. Argo CD가 OpenGitOps의 네 번째 원칙인 "지속적 조정(Continuously Reconciled)" — 소프트웨어 에이전트가 실제 시스템 상태를 지속적으로 관측하고 desired state를 적용하려 시도한다 — 을 실제 기계로 어떻게 구현하는지를, 조정 루프(reconciliation loop)·드리프트 감지(drift detection)·자가 치유(self-healing)라는 세 겹의 동작으로 분해해 들여다봅니다. 이 세 가지는 별개의 기능이 아니라, 하나의 제어 루프를 서로 다른 깊이에서 바라본 단면이라는 점을 먼저 염두에 두면 좋습니다.

### 조정 루프: 멈추지 않는 제어 회로

Argo CD가 Kubernetes 컨트롤러로 구현되어 실행 중인 애플리케이션을 *지속적으로* 모니터링하며 현재의 라이브 상태를 desired target state와 비교한다는 것은, 동기화가 일회성 명령이 아니라 끊임없이 도는 회로 위에서 일어난다는 뜻입니다. 이 회로를 한 바퀴 단위로 풀면 다음과 같은 단계가 반복됩니다.

1. **관측(observe)** — Application Controller가 클러스터의 라이브 리소스를 읽어 현재 상태 \( M\_{live} \)를 파악합니다.
2. **목표 계산(compute target)** — Repository Server가 Git의 선언을 렌더링한 target state \( M\_{target} \)를 산출합니다.
3. **비교(diff)** — 두 상태를 견주어 차이가 있으면 `OutOfSync`, 없으면 `Synced`로 판정합니다.
4. **조정(reconcile)** — 동기화 정책에 따라 차이를 해소할지 결정하고, 필요하면 클러스터에 변경을 적용합니다.

이 루프가 한 번 돌고 끝나지 않고 계속 반복된다는 사실이야말로 "지속적 조정"의 본질입니다. 형식적으로 표현하면, 컨트롤러는 매 사이클마다 라이브 상태를 desired state 쪽으로 밀어 넣는 연산을 적용하여 환경이 결국 다음 고정점으로 수렴하도록 만듭니다.

$$
M\_{live}^{(t+1)} =
\begin{cases}
\text{reconcile}\big(M\_{live}^{(t)},\, M\_{target}\big) & \text{if } M\_{live}^{(t)} \neq M\_{target} \ \text{and policy permits} \\[4pt]
M\_{live}^{(t)} & \text{otherwise}
\end{cases}
$$

여기서 핵심은 비교 자체(3단계)와 조정 행위(4단계)가 분리되어 있다는 점입니다. 비교는 항상 일어나 차이를 *보고·시각화*하지만, 실제로 클러스터에 손을 대는 조정은 동기화 정책이 허락할 때만 발생합니다. 그래서 "차이를 발견했다"와 "차이를 고쳤다"는 결코 같은 사건이 아닙니다 — 이 분리가 이후 수동 동기화와 자동 동기화를 가르는 토대가 됩니다.

한 가지만 덧붙입니다. Git이 바뀌지 않아도 *라이브 쪽이* 바뀌면 비교 결과가 달라질 수 있습니다. 즉 루프는 "Git의 변경"뿐 아니라 "클러스터의 변경"도 포착하는데, 바로 이 후자의 경로가 드리프트 감지의 출발점입니다.

### 드리프트 감지: Git에 없는 변경을 잡아내기

구성 드리프트(configuration drift)란 Git에 선언되지 않은 변경, 즉 누군가 클러스터를 직접 수정하거나 오류로 라이브 상태가 target state에서 벗어난 상황을 가리킵니다. Argo CD의 비교 연산은 Git의 변경만이 아니라 이 라이브 측 이탈도 동일하게 \( M\_{live} \neq M\_{target} \)로 환원하므로, 드리프트는 자연스럽게 `OutOfSync`로 드러납니다. 이것이 Argo CD가 제공하는 "자동 구성 드리프트 감지·시각화" 기능의 정체입니다.

구체적인 시나리오로 감을 잡아 봅시다. guestbook 애플리케이션이 이미 `Synced`로 배포된 상태에서, 운영자가 급한 마음에 클러스터를 직접 손봤다고 가정합니다.

```bash
# Git에는 없는, 사람이 직접 가한 변경 (드리프트 유발)
kubectl scale deployment guestbook-ui --replicas=5 -n default
```

Git의 매니페스트는 여전히 원래의 복제본 수를 선언하고 있으므로, 다음 비교 사이클에서 라이브 상태와 target state는 어긋나게 됩니다. 이때 `argocd app get`으로 조회하면 해당 `Deployment`가 `OutOfSync`로 표시되고, 그 차이가 보고·시각화됩니다.

```bash
argocd app get guestbook
```

주목할 점은, 이 시점에서 애플리케이션은 멀쩡히 동작할 수 있다는 것입니다. 즉 `OutOfSync`이면서도 Health는 정상일 수 있는데, 이는 Sync 상태(라이브가 Git과 같은가)와 Health(애플리케이션이 제대로 동작하는가)가 별개의 개념이라는 사실의 실전 사례입니다. 드리프트 감지는 어디까지나 "Git과 다르다"를 알릴 뿐, 그 자체로 장애를 의미하지 않습니다. 그 차이를 실제로 되돌릴지는 다음 단계인 동기화 정책의 몫입니다.

### 수동 동기화와 자동 동기화

드리프트를 감지한 뒤 컨트롤러가 *무엇을 할 것인가*는 `Application`의 동기화 정책이 결정합니다. 앞서 첫 Application을 만들 때 정책을 지정하지 않으면 `Sync Policy`가 `<none>`으로 표시되며 수동(Manual) 모드로 동작한다는 점을 보았습니다. 두 모드의 차이를 정리하면 다음과 같습니다.

| 구분 | 수동(Manual) 동기화 | 자동(Automated) 동기화 |
|---|---|---|
| 드리프트·Git 변경 감지 | 동일하게 감지·보고 (`OutOfSync` 표시) | 동일하게 감지·보고 |
| 차이 해소 시점 | 사람이 `argocd app sync` 또는 UI에서 Synchronize를 호출할 때만 | 컨트롤러가 정책에 따라 자동으로 |
| 적합한 맥락 | 변경 적용을 사람이 승인·통제하려는 경우 | desired state로의 자동 수렴을 원하는 경우 |

수동 모드에서는 비교 루프가 계속 돌며 `OutOfSync`를 알려 주지만, 실제 조정은 사람이 명시적으로 동기화를 호출하기 전까지 일어나지 않습니다. `argocd app sync` 명령은 저장소에서 매니페스트를 가져와 `kubectl apply`를 수행합니다. 반대로 자동 모드에서는 Git의 desired target state에 가해진 변경이 자동으로 대상 환경에 반영될 수 있습니다.

### 자가 치유(self-heal): 라이브 드리프트를 되돌리기

자동 동기화가 켜져 있으면, 앞의 `kubectl scale`로 만든 것 같은 라이브 변경이 자동화에 의해 덮어쓰여 환경이 다시 desired state로 수렴할 수 있습니다. 이는 GitOps 모델이 약속하는 바 — "Git에 정의되지 않은 수동 변경이나 오류, 즉 구성 드리프트는 자동화에 의해 덮어쓰여 환경이 Git에 정의된 desired state로 수렴한다" — 가 구체화되는 지점입니다. 소프트웨어 에이전트가 발산(divergence)을 감지해 자가 치유(self-healing) 방식으로 실제 상태를 선언된 상태와 비교·수렴시키는 제어 루프가 바로 이것입니다.

여기서 운영상의 함의가 분명해집니다. 자동 수렴이 동작하면 "운영 환경을 손으로 빠르게 고치는" 카우보이식 변경(cowboy engineering)은 사실상 무력화됩니다. 사람이 클러스터를 바꿔도 자동화가 Git의 값으로 되돌리기 때문입니다. 이것은 번거로움이 아니라 의도된 규율입니다 — 모든 변경이 Git을 거치도록 강제함으로써, Git을 단일 진실 공급원(single source of truth)으로 유지하는 것입니다.

### 동기화 단계와 훅: 조정 과정에 절차를 끼워 넣기

지금까지는 동기화를 "차이를 한 번에 적용하는 행위"로 단순화해 왔지만, 실제 조정은 여러 단계로 나뉠 수 있습니다. Argo CD는 수명 주기 이벤트를 위한 사용자 정의 훅을 제공하며, 그 단계는 **PreSync · Sync · PostSync**로 구분됩니다. Application Controller가 이 훅들을 호출하는 책임을 지며, 덕분에 블루/그린이나 카나리(canary) 같은 복잡한 애플리케이션 롤아웃을 조정 과정 안에 절차적으로 끼워 넣을 수 있습니다.

직관적으로, PreSync는 본 동기화 이전의 사전 작업, Sync는 매니페스트의 실제 적용, PostSync는 적용 이후의 후처리에 대응하는 지점입니다. 즉 "라이브를 target으로 옮긴다"는 단일 동작이, 단계가 정의되어 있으면 순서를 갖는 일련의 작업으로 확장됩니다. 이런 단계와 훅, 그리고 동기화의 세부 옵션들은 그 자체로 깊은 주제이므로, 이후 실제 구성 도구·선언적 설정을 다루는 장들에서 다시 마주치게 됩니다.

### 세 겹의 동작을 하나로 잇기

정리하면, 조정 루프·드리프트 감지·자가 치유는 분리된 기능이 아니라 하나의 제어 회로를 깊이별로 본 것입니다. 조정 루프는 \( M\_{live} \)와 \( M\_{target} \)를 끊임없이 관측·계산·비교하는 골격이고, 드리프트 감지는 그 비교가 Git 변경뿐 아니라 라이브 측 이탈까지 `OutOfSync`로 포착한다는 사실이며, 자가 치유는 그렇게 감지된 차이를 동기화 정책이 허락하는 한 능동적으로 desired state로 수렴시키는 마지막 단계입니다. 그리고 이 모든 비교가 효율적으로 반복될 수 있는 것은, 컨트롤러가 Redis를 중요한 캐싱 메커니즘으로 활용해 Kube API와 Git에 가해지는 부하를 줄이기 때문입니다. 이 루프가 어느 클러스터를 향해 도는지 — Argo CD가 실행 중인 내부 클러스터(`https://kubernetes.default.svc`)인지, 아니면 별도로 등록된 외부 클러스터인지 — 는 이어지는 장에서 살펴봅니다.
</none>

## 외부 클러스터 등록과 내부 클러스터 배포: argocd cluster add 사용법

앞선 동기화 메커니즘을 다루는 장의 끝에서, 조정 루프가 "어느 클러스터를 향해 도는가"라는 질문을 남겨 두었습니다. 지금까지의 모든 실습은 `destination.server`에 `https://kubernetes.default.svc`를 넣는, 즉 Argo CD가 실행 중인 바로 그 클러스터로 배포하는 경우만 다뤘습니다. 이 장은 그 좌표가 *외부 클러스터*를 가리킬 때 무엇이 필요한지를 채웁니다. 핵심은 단순합니다 — 외부 클러스터에 배포하려면 그 클러스터의 자격 증명을 먼저 Argo CD에 등록해야 하며, 내부 클러스터에는 그 등록이 필요 없습니다.

### 내부 배포 vs. 외부 배포: 자격 증명이 갈림길

배포 대상이 내부냐 외부냐에 따라 사전 준비가 완전히 달라집니다. 이 차이를 먼저 명확히 해 두면 이후 절차의 이유가 분명해집니다.

| 구분 | 내부 클러스터 배포 | 외부 클러스터 배포 |
|---|---|---|
| 대상 지정(`destination`) | `https://kubernetes.default.svc` (이름으로는 `in-cluster`) | 등록된 클러스터의 API 서버 주소 |
| 자격 증명 등록 | 불필요 | `argocd cluster add`로 사전 등록 필요 |
| 자격 증명의 출처 | Argo CD가 도는 클러스터 자체 | 대상 클러스터에 설치되는 `argocd-manager` 서비스 계정 토큰 |

다시 말해, 내부 배포에서는 Argo CD가 이미 자기 클러스터에 대한 권한을 갖고 있으므로 별도 등록 단계가 없습니다. 반면 외부 클러스터로 배포하려는 경우에만 클러스터의 자격 증명을 Argo CD에 등록하는 작업이 필요합니다. 앞서 살펴봤듯 이 자격 증명은 Kubernetes 시크릿으로 저장되며 API Server가 관리합니다.

### 등록 절차: 컨텍스트를 골라 등록하기

외부 클러스터 등록은 두 단계로 이뤄집니다. 먼저 현재 kubeconfig에 들어 있는 모든 클러스터 컨텍스트를 나열합니다.

```bash
kubectl config get-contexts -o name
```

그다음 목록에서 컨텍스트 이름 하나를 골라 `argocd cluster add CONTEXTNAME`에 넘깁니다. 예를 들어 `docker-desktop` 컨텍스트라면 다음과 같습니다.

```bash
argocd cluster add docker-desktop
```

이 한 줄이 외부 클러스터를 Argo CD의 배포 대상으로 편입시키는 전부입니다. 등록이 끝나면 그 클러스터의 API 서버 주소를 `Application`의 `destination.server`에 지정해 배포할 수 있게 됩니다.

### `argocd cluster add`가 실제로 하는 일

이 명령은 단순히 주소를 등록하는 데 그치지 않습니다. 내부적으로 다음 작업을 수행합니다.

- 대상 kubectl 컨텍스트가 가리키는 클러스터의 `kube-system` 네임스페이스에 `argocd-manager`라는 **ServiceAccount**를 설치합니다.
- 이 서비스 계정을 admin 수준의 **ClusterRole**에 바인딩합니다.
- Argo CD는 이 서비스 계정의 토큰을 사용해 대상 클러스터에 대한 관리 작업(배포·모니터링)을 수행합니다.

여기서 주목할 점은, 등록 이후 Argo CD가 대상 클러스터와 통신할 때 사용하는 것이 새로 만들어진 `argocd-manager` 토큰이라는 사실입니다. 즉 등록 시점에 사용한 kubeconfig 컨텍스트는 "어디에 서비스 계정을 심을 것인가"를 결정하는 도구이며, 지속적인 조정 루프는 그 클러스터에 심긴 전용 서비스 계정 토큰을 통해 동작합니다.

### 권한 범위 조정: admin 권한이 부담스러울 때

기본값으로 `argocd cluster add`는 admin 수준의 ClusterRole을 바인딩하므로, 보안 관점에서 권한이 과도하다고 느껴질 수 있습니다. 이때 `argocd-manager-role` 역할의 규칙을 수정해 권한을 좁힐 수 있습니다. 구체적으로는, 제한된 네임스페이스·그룹·종류(kind)에 대해서만 `create`, `update`, `patch`, `delete` 권한을 갖도록 줄이는 것이 가능합니다.

다만 한 가지 제약이 있습니다. **`get`, `list`, `watch` 권한은 클러스터 스코프(cluster-scope)에서 반드시 보장되어야 합니다.** Argo CD가 정상적으로 동작하려면 이 읽기·감시 권한이 클러스터 전역에서 필요하기 때문입니다.

### 같은 클러스터에 네임스페이스 권한만으로 배포하기

설치 유형을 다루는 장에서 소개한 `namespace-install.yaml` 구성, 즉 클러스터 롤 없이 네임스페이스 수준 권한만으로 동작하는 설치를 떠올려 봅시다. 이 경우에도 Argo CD가 도는 바로 그 클러스터(`kubernetes.svc.default`)에 배포하는 것이 가능한데, 이때는 제공된 자격 증명을 통해 클러스터를 등록하는 형태를 취합니다.

```bash
argocd cluster add <context> --in-cluster --namespace <your namespace="">
```

`--in-cluster` 플래그는 등록 대상이 Argo CD가 실행 중인 클러스터 자체임을 나타내고, `--namespace`로 배포 범위를 특정 네임스페이스로 한정합니다. 이는 팀별로 별도의 Argo CD 인스턴스를 운영하면서 각 인스턴스가 외부 클러스터에 배포하는 구성에서, 필요에 따라 자기 클러스터로도 배포 경로를 여는 방식에 해당합니다. 다만 기본 제공 롤로는 같은 클러스터에 Argo CD 리소스(Applications, ApplicationSets, AppProjects)만 배포할 수 있습니다.

정리하면, 클러스터 등록의 본질은 "대상 클러스터에 Argo CD 전용 서비스 계정을 심고 그 토큰을 자격 증명으로 보관하는 일"입니다. 내부 배포에는 이 단계가 필요 없고, 외부 배포에는 반드시 선행되어야 합니다. 이렇게 등록된 클러스터들은 다수 환경·다수 클러스터로 Application을 자동 생성하는 ApplicationSet을 다루는 장에서 다시 핵심 재료로 등장합니다.
</your></context>

## Config Management Tool: Kustomize와 Helm을 Argo CD와 함께 사용하기

앞서 Repository Server를 "부수 효과 없는 렌더링 엔진"으로 규정하면서, 이 컴포넌트가 Kustomize 애플리케이션·Helm 차트·Jsonnet 파일·평이한 YAML/JSON 디렉터리를 동일한 인터페이스 뒤에서 렌더링한다고 짚었습니다. 이 장은 그 추상적 설명을 두 가지 가장 널리 쓰이는 도구—Kustomize와 Helm—의 실제 동작으로 구체화합니다. Argo CD는 "어떤 도구로 만들었든 결국 Kubernetes 매니페스트라는 단일 산출물로 환원한다"는 계약을 지키므로, 이 도구들을 이해하는 일은 곧 Repository Server에 무엇을 입력하면 어떤 매니페스트가 나오는지를 이해하는 일입니다.

### 왜 템플릿·패칭 도구가 필요한가

GitOps를 처음 시작한 많은 팀이 똑같은 벽에 부딪힙니다 — "다뤄야 할 YAML이 너무 많다"는 문제입니다. 환경·클러스터·규제 제약 등을 고려하다 보면 아주 미세한 차이만 있는 거의 동일한 YAML을 수없이 복제하게 되고, 이는 "반복하지 말라(DRY, Don't Repeat Yourself)"는 원칙을 정면으로 위반합니다. 환경마다 애플리케이션의 구조적 선언은 거의 같고, 데이터베이스 자격 증명·복제본 수·컨테이너 이미지 같은 *값*만 달라지기 때문입니다. Kustomize와 Helm은 바로 이 "구조는 같고 값만 다르다"는 상황을 각기 다른 방식—패칭과 파라미터화—으로 해소합니다.

Argo CD가 "다중 config 관리·템플릿 도구 지원"을 핵심 기능으로 내세우는 것도 이 맥락에서입니다. 즉 사용자는 자신에게 익숙한 도구로 desired state를 표현하면 되고, Argo CD는 그것을 받아 클러스터에 적용할 최종 매니페스트로 풀어냅니다.

### Kustomize: 원본을 건드리지 않고 덧입히기

Kustomize는 Kubernetes 매니페스트에 변경 사항을 오버레이(overlay)하는 패칭 프레임워크입니다. 원본 매니페스트는 그대로 둔 채, 결과로서 변경이 반영된 새 매니페스트를 렌더링한다는 점이 핵심입니다. 전형적인 디렉터리 구조는 `base`와 `overlays` 두 갈래로 나뉩니다.

```text
$ tree
├── base
│   ├── deployment.yaml
│   └── kustomization.yaml
└── overlays
    └── dev
        └── kustomization.yaml
```

`base` 디렉터리에는 평범한 Deployment 매니페스트와 함께, 그것을 "읽어 들이라"고 지시하는 `kustomization.yaml`이 놓입니다.

```yaml
# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
```

진짜 마법은 `overlays/dev`의 `kustomization.yaml`에서 일어납니다. 여기서는 base를 자원으로 끌어들인 뒤, 특정 Deployment의 복제본 수를 3으로 바꾸는 패치를 적용합니다.

```yaml
# overlays/dev/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
patches:
  - target:
      kind: Deployment
      name: nginx
    patch: |-
      - op: replace
        path: /spec/replicas
        value: 3
```

이렇게 구성한 뒤 다음 명령으로 최종 매니페스트를 렌더링할 수 있습니다.

```bash
kustomize build overlays/dev
```

`kubectl apply -k overlays/dev/`로 클러스터에 직접 적용할 수도 있습니다. 렌더링된 결과는 복제본 수가 3으로 치환된 새 Deployment 매니페스트입니다. 환경 간 차이가 크지 않은 애플리케이션이라면, 공통 부분은 base에 두고 환경별 차이(delta)만 오버레이로 저장하는 이 방식이 YAML 중복을 크게 줄여 줍니다. 이미 설치 유형을 다루는 장에서 원격 매니페스트를 `resources`로 감싸 패치하는 Kustomize 활용을 본 바 있는데, 애플리케이션 배포에서도 같은 원리가 그대로 작동합니다.

### Helm: 값을 주입해 릴리스를 찍어 내기

Helm은 Kubernetes의 패키지 매니저로, 본질적으로는 애플리케이션 배포를 위한 "템플릿 엔진"으로 볼 수 있습니다. Helm은 차트(chart)라는 패키지로 구성되는데, 차트는 YAML 매니페스트를 템플릿화·패키징한 것입니다. 사용자가 템플릿에 정의된 파라미터에 값(values)을 주입하면, Helm이 그 값을 매니페스트에 삽입해 릴리스(release)를 생성합니다. 릴리스는 클러스터에 배포되는 YAML의 최종 상태 표현이며, 그 정보는 Kubernetes 클러스터에 시크릿으로 저장됩니다.

다음은 저장소를 추가하고 릴리스를 배포하는 예입니다.

```bash
helm repo add akuity-demos https://akuity.github.io/demo-helm-charts/

helm install myapp \
  --create-namespace --namespace example \
  --set replicaCount=3 \
  akuity-demos/simple-go
```

여기서 `--set replicaCount=3`처럼 파라미터에 값을 넘기면, 차트가 제공하는 템플릿이 그 값으로 채워져 배포됩니다. 결과적으로 애플리케이션을 여러 환경에 배포할 때 각 배포마다 달라지는 것은 Values 파일뿐이며, 차트 자체는 그대로 재사용됩니다. Argo CD의 관점에서 보면, Repository Server가 받는 입력 중 "템플릿별 설정(파라미터, Helm `values.yaml`)"이 바로 이 값들에 해당합니다. Argo CD가 기능 목록에서 "Git에 커밋된 Helm 파라미터를 오버라이드하는 파라미터 오버라이드"를 제공하는 것도, Helm의 이 값 주입 모델을 GitOps 워크플로 안으로 끌어들인 결과입니다.

### 무엇을, 언제 쓰는가

두 도구는 경쟁 관계가 아니라 보완 관계입니다. 실무에서는 "Kustomize 대 Helm"이 아니라 "Kustomize와 Helm"으로 함께 쓰이는 경우가 많습니다. 선택의 기준을 정리하면 다음과 같습니다.

| 구분 | Kustomize | Helm |
|---|---|---|
| 동작 방식 | 원본 매니페스트에 패치를 오버레이해 새 매니페스트 렌더링 | 템플릿에 값을 주입해 릴리스 생성 |
| 적합한 상황 | 주로 원시(raw) Kubernetes 매니페스트를 다룰 때 | 값을 파라미터화해야 할 때, 서드파티(ISV) 애플리케이션 스택을 소비할 때 |
| 값을 미리 아는가 | 패칭 대상 값을 사전에 알고 있을 때 자연스러움 | 클러스터에 적용되기 전까지 값을 알 수 없을 때(예: Ingress의 `host` FQDN) 유용 |
| 생태계 | Kubernetes·다수 GitOps 도구에 내장 | 대규모 차트 저장소 생태계, 다수 ISV가 배포 수단으로 채택 |

원시 매니페스트가 중심이라면 Kubernetes와 여러 GitOps 도구에 내장된 Kustomize를 우선 고려하고, 값을 사전에 알 수 없거나 서드파티 차트를 소비해야 한다면 Helm으로 파라미터화하는 것이 적절합니다. 대표적으로 여러 클러스터에 배포할 때 각 클러스터의 FQDN을 미리 알 수 없는 Ingress 같은 경우가 Helm이 빛나는 지점입니다.

### Argo CD와 맞물리는 지점

이 두 도구가 Argo CD에서 의미를 갖는 방식은 핵심 개념을 다루는 장에서 정의한 "Application source type", 즉 "애플리케이션을 빌드하는 데 사용되는 Tool"이라는 어휘로 정확히 설명됩니다. `Application`이 가리키는 저장소 경로에 Kustomize의 `kustomization.yaml`이 있으면 Kustomize로, Helm 차트가 있으면 Helm으로 렌더링됩니다. 어느 쪽이든 Repository Server가 저장소 URL·리비전·경로와 템플릿별 설정(파라미터, `values.yaml`)을 입력받아 최종 Kubernetes 매니페스트를 산출하고, 그 산출물을 Application Controller가 라이브 상태와 비교해 동기화 여부를 판정합니다.

다시 말해, Kustomize와 Helm을 "Argo CD와 함께 쓴다"는 것은 별도의 통합 작업이 필요한 일이 아니라, Git 저장소에 어떤 형식으로 desired state를 표현해 두느냐의 선택입니다. 도구를 무엇으로 고르든 GitOps 루프의 본질—Git을 단일 진실 공급원으로 삼아 desired state를 자동으로 수렴시키는—은 동일하게 유지됩니다. 이렇게 단일 애플리케이션을 다루는 수준을 넘어, 여러 애플리케이션과 환경을 선언적으로 묶어 관리하는 방법은 이어지는 선언적 설정과 App of Apps 패턴, 그리고 ApplicationSet을 다루는 장에서 살펴봅니다.

## 선언적 설정(Declarative Setup)과 App of Apps 패턴 소개

첫 Application을 다루는 장에서는 `argocd app create`나 Web UI의 **+ New App** 폼으로 Application을 만들었습니다. 그러나 잠시 멈춰 생각해 보면, 이 방식에는 미묘한 모순이 있습니다. GitOps의 첫 번째 원칙은 desired state를 *선언적으로* 표현하라는 것이고 두 번째 원칙은 그것을 *버전 관리되는 불변의 저장소*에 두라는 것인데, CLI 명령이나 UI 클릭으로 Application을 만드는 행위 자체는 명령형이며 Git에 기록되지 않습니다. 다시 말해, 배포되는 애플리케이션은 GitOps로 관리되지만 "무엇을 배포하라는 지시" 그 자체는 GitOps 바깥에 머물러 있는 셈입니다. 이 장은 그 마지막 빈틈을 메우는 두 가지 방법—선언적 설정과 App of Apps 패턴—을 소개합니다.

### Application을 YAML로 선언하고 kubectl로 적용하기

핵심 통찰은 앞서 여러 차례 확인한 사실에서 곧장 따라 나옵니다. `Application`은 매니페스트로 정의되는 Kubernetes 리소스(CRD)입니다. 즉 Application은 그 자체로 하나의 Kubernetes 객체이므로, CLI를 거치지 않고도 평범한 YAML 파일로 작성해 `kubectl apply`로 클러스터에 적용할 수 있습니다. 이것이 선언적 설정(Declarative Setup)의 출발점입니다.

guestbook 예제를 선언적으로 표현하면 다음과 같은 매니페스트가 됩니다.

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

이 파일을 Git 저장소에 커밋해 두고 다음처럼 적용하면, `argocd app create`로 만든 것과 동일한 `Application` 객체가 클러스터에 생성됩니다.

```bash
kubectl apply -f guestbook-app.yaml
```

CLI 방식과 선언적 방식의 차이를 정리하면 다음과 같습니다. 결과물(클러스터에 남는 `Application` 객체)은 같지만, "지시"가 어디에 기록되는가가 결정적으로 다릅니다.

| 구분 | 명령형 생성 (`argocd app create`, UI) | 선언적 설정 (`kubectl apply -f`) |
|---|---|---|
| Application 정의의 출처 | 사람이 입력한 명령·폼 | Git에 커밋된 YAML 파일 |
| 버전 관리·감사 추적 | 명령 자체는 Git에 남지 않음 | 정의가 곧 Git 이력에 남음 |
| 재현성 | 같은 명령을 다시 입력해야 함 | 파일을 다시 적용하면 동일 |
| GitOps 원칙 부합 | 부분적 (대상은 GitOps, 지시는 외부) | 완전 (지시까지 선언적·버전 관리) |

이 방식은 Application뿐 아니라 멀티 테넌시 접근 제어를 다루는 장에서 등장할 `AppProject` 같은 Argo CD 자체의 설정 리소스에도 동일하게 적용됩니다. 즉 Argo CD를 운영하는 데 필요한 구성 객체들 역시 Git에 선언으로 두고 관리할 수 있으며, 이 운영 방식이 공식 문서가 "Declarative Setup"이라는 이름으로 다루는 영역입니다.

### 규모의 문제: 수백 개의 Application 매니페스트

선언적 설정은 한 가지 새로운 고민을 낳습니다. 애플리케이션이 수십·수백 개로 늘어나면, 그만큼의 `Application` YAML 파일을 각각 `kubectl apply`로 적용해야 하는 부담이 생깁니다. 새 클러스터를 처음부터 구성(bootstrapping)하는 상황을 떠올리면 문제가 분명해집니다. 모니터링 스택, 인그레스 컨트롤러, 여러 마이크로서비스 등 클러스터에 올라가야 할 모든 것을 일일이 손으로 적용하는 것은 GitOps가 줄이려던 바로 그 수작업입니다.

여기서 한 단계 더 나아간 발상이 필요합니다. `Application` 자체가 Git에 저장 가능한 Kubernetes 매니페스트라면, *Argo CD가 Application 매니페스트들을 배포 대상으로 삼는 것*도 가능하지 않을까요? 실제로 그렇습니다. namespace 수준 권한 설치를 다룰 때 언급했듯, 기본 롤만으로도 같은 클러스터에 `Application`, `ApplicationSet`, `AppProject` 같은 Argo CD 리소스를 배포할 수 있습니다. 이 사실이 App of Apps 패턴을 가능하게 하는 토대입니다.

### App of Apps 패턴

App of Apps 패턴은 *다른 Application들을 자식으로 거느리는 하나의 부모 Application*을 두는 구성입니다. 부모 Application의 `source`는 일반적인 워크로드 매니페스트가 아니라, **여러 개의 자식 `Application` 매니페스트가 들어 있는 Git 디렉터리**를 가리킵니다.

작동 원리는 지금까지 배운 조각들의 자연스러운 조합입니다.

1. Git 저장소의 한 디렉터리(예: `apps/`)에 여러 자식 `Application` YAML을 모아 둡니다.
2. 그 디렉터리를 `source.path`로 가리키는 부모 `Application`을 하나 만듭니다.
3. 부모를 동기화하면, Argo CD는 그 디렉터리의 매니페스트들—즉 자식 `Application` 객체들—을 클러스터에 적용합니다.
4. 자식 `Application` 객체들이 생성되면, Application Controller가 이번에는 각 자식을 자신의 desired state로 인식하고 그에 해당하는 실제 워크로드를 동기화합니다.

부모 Application은 다음과 같은 모습이 됩니다. `path`가 개별 워크로드가 아니라 자식 앱 정의들이 모인 디렉터리를 가리킨다는 점이 핵심입니다.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: bootstrap
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/example/gitops-repo.git
    targetRevision: HEAD
    path: apps        # 자식 Application 매니페스트들이 들어 있는 디렉터리
  destination:
    server: https://kubernetes.default.svc
    namespace: argocd
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

`apps/` 디렉터리 안에는 앞서 본 형식의 자식 Application들이 각각 하나의 파일로 놓입니다. 예컨대 `apps/guestbook.yaml`은 guestbook 워크로드를 가리키는 `Application`이고, `apps/monitoring.yaml`은 모니터링 스택을 가리키는 `Application`이 되는 식입니다. 부모 하나만 클러스터에 등록하면, 그 뒤로는 디렉터리에 자식 매니페스트를 추가·수정하는 것만으로 전체 애플리케이션 집합이 선언적으로 관리됩니다.

이 패턴의 가치는 명확합니다. 클러스터를 부트스트랩할 때 단 하나의 매니페스트만 적용하면 그것이 나머지 모든 애플리케이션을 끌어오므로, 새 클러스터 구성이 "부모 하나 적용"으로 압축됩니다. 또한 부모 Application에 위처럼 자동 동기화 정책을 걸어 두면, Git 저장소의 `apps/` 디렉터리에 새 자식 정의를 커밋하는 것만으로 새 애플리케이션이 자동으로 클러스터에 등장하게 됩니다. 변경의 단위가 "명령 실행"이 아니라 "Git 커밋"이 되는 것입니다.

다만 한 가지 유념할 점이 있습니다. App of Apps에서 부모가 관리하는 desired state는 *자식 Application 객체 그 자체*이지, 자식이 배포하는 워크로드가 아닙니다. 즉 동기화는 두 층위에서 일어납니다. 부모의 동기화는 자식 `Application` 리소스를 생성·갱신하고, 각 자식의 동기화는 그 자식이 가리키는 실제 매니페스트를 배포합니다. 이 두 층위를 분리해서 이해해야 부모를 동기화했는데 워크로드가 곧바로 뜨지 않는 상황 등을 정확히 읽어 낼 수 있습니다.

### 다음으로: 자동 생성과의 차이

App of Apps는 "자식 Application들을 *직접 작성해* 디렉터리에 모아 둔다"는 점에서, 여전히 각 자식 매니페스트를 손으로 관리해야 합니다. 환경이나 클러스터마다 거의 동일한 자식 정의를 반복 작성하게 되면, Config Management Tool을 다루는 장에서 본 "구조는 같고 값만 다르다"는 중복 문제가 다시 고개를 듭니다. 이 반복을 템플릿화하여 Application을 *자동으로 생성*하는 메커니즘이 ApplicationSet이며, 이는 이어지는 장에서 본격적으로 다룹니다. App of Apps가 "Application들을 모아 한 번에 배포하는" 패턴이라면, ApplicationSet은 "Application들을 규칙에 따라 찍어 내는" 컨트롤러라는 점에서 서로 보완적입니다.

## ApplicationSet: 다수 클러스터·환경에 Application 자동 생성

선언적 설정과 App of Apps 패턴을 다루는 장의 끝에서, App of Apps가 "자식 `Application` 매니페스트들을 *직접 작성해* 디렉터리에 모아 두는" 방식이라 환경·클러스터마다 거의 동일한 정의를 반복 작성하게 된다는 한계를 짚었습니다. 구조는 같고 값만 다른 이 중복은 Config Management Tool을 다루는 장에서 마주친 문제와 본질적으로 같습니다. ApplicationSet은 바로 이 반복을 컨트롤러 수준에서 해소합니다. App of Apps가 "이미 작성된 Application들을 모아 한 번에 배포하는" 패턴이라면, ApplicationSet은 "하나의 템플릿과 입력 집합으로 Application들을 규칙에 따라 *찍어 내는*" 컨트롤러입니다.

### 두 부분으로 이루어진 모델: 생성기와 템플릿

ApplicationSet은 `ApplicationSet`이라는 별도의 CRD로 정의되며, 그 동작은 두 부분의 조합으로 이해하면 명확합니다.

- **생성기(generator)**: "몇 개의 Application을, 어떤 파라미터로 만들 것인가"를 결정하는 입력원입니다. 생성기는 일련의 파라미터 집합을 산출하며, 각 파라미터 집합 하나가 하나의 Application으로 이어집니다.
- **템플릿(template)**: 생성기가 만들어 낸 각 파라미터 집합을 끼워 넣어 실제 `Application` 객체로 렌더링하는 틀입니다.

즉 생성기가 \( N \)개의 파라미터 집합을 내놓으면 ApplicationSet 컨트롤러는 템플릿에 그 값을 채워 \( N \)개의 `Application`을 생성·관리합니다. 클러스터 목록이나 Git 디렉터리 구조가 바뀌어 생성기의 출력이 달라지면, 그에 맞춰 Application이 자동으로 추가·갱신·삭제됩니다. App of Apps에서는 새 애플리케이션마다 자식 매니페스트를 손으로 추가해야 했지만, ApplicationSet에서는 생성기의 입력만 늘어나면 Application이 따라서 늘어납니다.

구조를 골격으로만 표현하면 다음과 같은 모습입니다. 생성기 블록과 템플릿 블록이 명확히 분리된다는 점이 핵심입니다.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: guestbook
  namespace: argocd
spec:
  generators:
    # 여기에 사용할 생성기를 지정합니다 (List, Cluster, Git 등)
    - ...
  template:
    # 생성기가 만든 파라미터로 채워질 Application 템플릿
    metadata:
      name: ...
    spec:
      project: default
      source: ...
      destination: ...
```

각 생성기의 구체적 필드 문법은 Argo CD 공식 문서의 생성기별 레퍼런스(List Generator, Cluster Generator, Git Generator 등)에 정의되어 있으며, 템플릿 안에서 파라미터를 치환하는 방식은 Go Template 기능으로도 작성할 수 있습니다.

### 생성기 카탈로그

Argo CD는 서로 다른 입력원에 대응하는 여러 종류의 생성기를 제공합니다. "Application을 어디서 끌어낼 것인가"라는 질문에 대한 답이 곧 생성기 선택입니다.

| 생성기 | 입력원 |
|---|---|
| **List** | 고정된 값(element) 목록을 직접 나열 |
| **Cluster** | Argo CD에 등록된 클러스터들 |
| **Git** | Git 저장소의 디렉터리 구조 또는 파일 |
| **SCM Provider** | SCM(소스 코드 관리) 공급자에서 발견한 저장소들 |
| **Pull Request** | 열려 있는 풀 리퀘스트들 |
| **Cluster Decision Resource** | 클러스터 결정 리소스 |
| **Matrix** | 두 생성기의 조합 |
| **Merge** | 여러 생성기 결과의 병합 |
| **Plugin** | 사용자 정의 플러그인 |

여기에 더해, 어떤 생성기를 쓰든 그 결과를 한 번 더 걸러 내는 **Post Selector** 가 모든 생성기에 적용될 수 있습니다.

이 중 다수 클러스터·환경 시나리오에서 가장 직접적으로 빛나는 것이 **Cluster 생성기**입니다. 외부 클러스터 등록을 다루는 장에서 `argocd cluster add`로 등록한 클러스터들이 여기서 다시 핵심 재료로 등장합니다. Cluster 생성기는 등록된 클러스터 각각에 대해 파라미터 집합을 만들어 내므로, 새 클러스터를 하나 등록하면 동일한 애플리케이션이 그 클러스터에 자동으로 배포되도록 만들 수 있습니다. 한편 환경별 디렉터리 구조를 가진 GitOps 저장소라면 **Git 생성기**가, 명시적으로 몇 개의 환경만 다루고 싶다면 **List 생성기**가 자연스럽습니다.

### 점진적 동기화와 네임스페이스 확장

ApplicationSet으로 다수의 Application을 한꺼번에 다루기 시작하면, "모든 Application을 동시에 동기화할 것인가, 아니면 단계적으로 진행할 것인가"라는 새로운 운영 문제가 생깁니다. 이를 위해 ApplicationSet은 **Progressive Syncs(점진적 동기화)** 기능을 제공하며, 이는 `spec.strategy.*` 및 `status.applicationStatus.*` 필드를 통해 동작합니다. 다만 한 가지 주의가 필요합니다. Progressive Syncs는 v2.6.0에서 도입된 **Beta** 단계 기능입니다.

또한 기본적으로 ApplicationSet은 Argo CD가 설치된 네임스페이스에서 동작하지만, **AppSets in any Namespace** 기능(v2.8.0 도입, **Beta**)을 통해 다른 네임스페이스에서도 ApplicationSet을 운용할 수 있습니다. Alpha/Beta 기능은 향후 릴리스에서 호환성이 깨지는 변경이 있을 수 있으므로, 프로덕션 환경에서 의존하기 전에 사용 중인 기능을 문서화하고 업그레이드 전 릴리스 노트를 검토하는 것이 권장됩니다. 기능 성숙도(Feature Maturity)를 확인하는 일반적인 방법과 그 의미는 다음 단계 학습 경로를 다루는 장에서 더 살펴봅니다.

### 설치와 Core 모드에서의 가용성

ApplicationSet은 별도로 설치할 필요 없이 표준 Argo CD 설치에 함께 포함됩니다. 설치 및 초기 접근을 다루는 장에서 `--server-side --force-conflicts` 플래그가 필요한 이유를 설명하면서 언급했듯, 일부 Argo CD CRD가 클라이언트 측 적용의 262KB 어노테이션 한도를 초과하는 대표적 사례가 바로 ApplicationSet CRD입니다. 즉 서버 측 적용을 사용하는 표준 설치 절차를 따랐다면 ApplicationSet 컨트롤러와 CRD가 이미 준비되어 있습니다.

주목할 점은 Core 모드를 다루는 장에서 다룰 헤드리스 설치에서도 ApplicationSet을 쓸 수 있다는 사실입니다. Argo CD Core 설치에서 사용 가능한 Kubernetes 리소스는 `Application`과 `ApplicationSet` CRD이며, 사용자는 이 두 리소스를 통해 GitOps 기반으로 애플리케이션을 배포·관리합니다. 또한 네임스페이스 수준 권한 설치(`namespace-install.yaml`)에서도 기본 롤만으로 같은 클러스터에 `Application`·`ApplicationSet`·`AppProject` 같은 Argo CD 리소스를 배포할 수 있습니다.

정리하면, ApplicationSet은 "생성기가 입력 집합을 산출하고, 템플릿이 그것을 Application으로 렌더링한다"는 단순한 모델로, 다수 클러스터와 환경에 걸친 Application 생성을 자동화하는 컨트롤러입니다. 클러스터 목록·Git 구조·풀 리퀘스트 같은 동적인 입력에 Application 집합을 묶어 둠으로써, 환경이 늘어나도 손으로 매니페스트를 복제하지 않고 선언적으로 규모를 키울 수 있습니다. 이렇게 자동 생성된 Application들이 실제로 어떤 저장소 구조 위에서 관리되어야 하는지는 이어지는 GitOps 저장소 구조 베스트 프랙티스를 다루는 장에서, 그리고 이들을 누가 어떤 권한으로 다룰 수 있는지는 RBAC와 Projects를 다루는 장에서 살펴봅니다.

## Argo CD Core 모드: 경량 헤드리스 설치와 활용

설치 유형을 다루는 장에서 Core가 "헤드리스(headless) 모드로 동작하는 경량 설치"이며 API Server와 UI 없이 각 컴포넌트의 non-HA 버전만 설치한다는 점, 그리고 아키텍처를 다루는 장에서 그 분리가 어떻게 가능한지를 이미 살펴봤습니다. 이 장은 그 빈칸을 채웁니다 — 즉 Core 모드에서 *무엇이 빠지고 무엇이 부분적으로만 남는지*를 정밀하게 구분하고, 그렇게 단출해진 환경에서 실제로 애플리케이션을 어떻게 설치·조작하는지를 다룹니다.

### 무엇이 사라지고 무엇이 부분적으로 남는가

Core 모드의 본질을 이해하는 가장 정확한 방법은 "어떤 기능 묶음이 완전히 빠지고, 어떤 것이 제약된 형태로만 살아남는가"를 표로 분리해 보는 것입니다. 헤드리스 설치인 만큼 다음 기능 그룹은 아예 사용할 수 없습니다.

| 사용 불가 (완전히 빠짐) | 부분적으로 사용 가능 |
|---|---|
| Argo CD RBAC 모델 | Argo CD Web UI |
| Argo CD API | Argo CD CLI |
| Argo CD Notification Controller | 멀티 테넌시 (Git push 권한에 기반) |
| OIDC 기반 인증 | |

왼쪽 열에서 RBAC 강제, 외부 ID 공급자로의 인증 위임(OIDC), API 노출은 본래 API Server의 책임이었으므로, 그 컴포넌트를 떼어 낸 Core 설치에서는 통째로 사라집니다. 이것이 "클러스터 관리자가 Kubernetes RBAC만으로 GitOps 엔진을 독립적으로 돌리고 싶을 때" Core가 적합한 이유입니다 — 별도의 인증·인가 계층 없이, 클러스터의 기본 RBAC가 곧 접근 통제 수단이 됩니다.

오른쪽 열의 "부분적"이라는 단서는 좀 더 풀어서 읽어야 합니다. 특히 **멀티 테넌시가 "Git push 권한에 엄격히 기반한 GitOps 방식으로 좁혀진다"**는 표현은 다음을 뜻합니다. Core에는 Argo CD 자체의 RBAC 모델이 없으므로, 팀 간 격리는 오로지 *Git 저장소에 대한 push 권한*과 *Argo CD 네임스페이스 리소스에 대한 Kubernetes 권한*으로만 이루어집니다. 즉 "desired state를 바꾸려면 해당 Git 경로에 커밋을 push할 수 있어야 한다"는 사실 자체가 테넌트 경계가 되는 것이며, Argo CD가 사용자별로 권한을 판정해 주지는 않습니다. 이렇게 통제 지점이 Argo CD 바깥(Git과 Kubernetes RBAC)으로 옮겨 가기 때문에 "부분적"이라고 분류됩니다.

한편, 앞서 컨트롤러가 캐싱을 위해 Redis에 의존한다고 설명했듯, Redis는 Core 설치에도 함께 포함됩니다.

### 설치: 단일 매니페스트 적용

Core는 필요한 모든 리소스를 담은 단일 매니페스트 파일(`core-install.yaml`)을 적용하는 것으로 설치됩니다. 표준 설치와 마찬가지로 `--server-side --force-conflicts` 플래그가 필요하며, 그 이유는 설치 및 초기 접근을 다루는 장에서 설명한 CRD 크기 한도 문제와 동일합니다.

```bash
kubectl create namespace argocd
kubectl apply -n argocd --server-side --force-conflicts \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/core-install.yaml
```

위 예시는 `stable` 브랜치를 직접 참조합니다. 프로덕션에서는 `stable` 대신 고정된 버전(예: `v3.2.0`)을 쓰는 것이 권장되며, 그 경우 URL의 `stable` 부분을 원하는 릴리스 태그로 교체하면 됩니다.

### 활용: GitOps와 로컬에서 깨어나는 CLI/UI

Core가 설치되면 사용자는 전적으로 GitOps에 의존해 Argo CD와 상호작용합니다. 이때 다룰 수 있는 Kubernetes 리소스는 `Application`과 `ApplicationSet` CRD이며, 사용자는 이 두 리소스를 통해 애플리케이션을 배포·관리합니다. 선언적 설정과 ApplicationSet을 다루는 장에서 본 매니페스트들이 그대로 Core에서도 동작한다는 의미입니다.

흥미로운 지점은, API Server가 상주하지 않음에도 CLI와 Web UI를 *쓸 수 있다*는 사실입니다. 비결은 "필요할 때 로컬에서 API Server 프로세스를 띄운다"는 데 있습니다. CLI를 Core 모드로 쓰려면 `login` 서브커맨드에 `--core` 플래그를 넘깁니다. 이 플래그가 바로 CLI·Web UI 요청을 처리할 로컬 Argo CD API Server 프로세스를 띄우는 역할을 합니다.

```bash
kubectl config set-context --current --namespace=argocd   # 현재 컨텍스트를 argocd 네임스페이스로
argocd login --core
```

로그인 후에는 일반적인 CLI 명령이 그대로 동작합니다. 예컨대 다음 한 줄로 특정 Application을 동기화할 수 있습니다.

```bash
argocd app sync guestbook
```

이때 CLI는 명령을 처리하기 위해 로컬 API Server 프로세스를 잠깐 띄웠다가, 명령이 끝나면 그 프로세스를 함께 종료합니다. 이 과정은 추가 명령 없이 사용자에게 투명하게 일어납니다. 다만 Core는 Kubernetes RBAC에만 의존하므로, CLI를 실행하는 사용자(또는 프로세스)는 Argo CD 네임스페이스에 접근할 수 있고 `Application`·`ApplicationSet` 리소스에 대한 적절한 권한을 가지고 있어야 합니다 — 앞서 본 RBAC·OIDC의 부재가 여기서 실질적 제약으로 나타나는 셈입니다.

Web UI 역시 같은 원리로 로컬에서 띄울 수 있습니다.

```bash
argocd admin dashboard -n argocd
```

이 명령을 실행하면 Argo CD Web UI가 `http://localhost:8080`에서 제공됩니다.

정리하면, Argo CD Core는 "API Server·RBAC·OIDC·Notification Controller를 덜어 내고, Git push와 Kubernetes RBAC만으로 경계를 세우는" 헤드리스 GitOps 엔진입니다. UI·SSO·멀티 테넌시 기능이 필요 없는 클러스터 관리자에게 가장 단출한 선택지이며, 그럼에도 `--core` 로그인이나 `argocd admin dashboard`를 통해 필요할 때마다 CLI와 UI를 로컬에서 임시로 깨워 쓸 수 있다는 점에서, 경량성과 편의성 사이의 균형을 갖춥니다.

## GitOps 저장소 구조 베스트 프랙티스: 코드 저장소 분리, 환경별 디렉터리 vs. 브랜치

지금까지 `Application`을 선언적으로 작성하고, App of Apps와 ApplicationSet으로 규모를 키우는 방법을 살펴봤습니다. 그러나 정작 그 매니페스트들이 *어떤 Git 저장소에, 어떤 구조로* 놓여야 하는지는 다루지 않았습니다. 이 질문은 GitOps를 처음 도입하는 조직이 가장 먼저 부딪히는 난관이며, 흥미롭게도 단 하나의 정답이 없는 영역입니다. 그럼에도 검증된 원칙들이 존재하므로, 이 장은 "왜 그렇게 구조를 잡아야 하는가"라는 이유에 초점을 맞춰 핵심 실천법을 정리합니다.

### 왜 "정답이 없는가": Conway의 법칙

저장소 구조에 만능 해법이 없는 근본 원인은 Conway의 법칙으로 설명됩니다.

> "시스템을 설계하는 모든 조직은 그 조직의 의사소통 구조를 복제한 설계를 만들어 낸다." — Melvin E. Conway

즉 조직이 어떻게 구성되고 소통하는지가 디렉터리 구조와 워크플로 형태를 결정하지, 그 반대가 아닙니다. 조직의 경계와 책임 분담이 저장소 레이아웃에 그대로 투영되므로, "좋은 구조"의 구체적 모습은 조직마다 다를 수밖에 없습니다. 따라서 이 장에서 제시하는 것은 고정된 템플릿이 아니라, 조직에 맞는 최적 구조를 찾도록 안내하는 상위 수준의 원칙입니다.

### 원칙 1: 코드 저장소와 배포 저장소를 분리하라

조직이 가장 먼저 마주하는 결정은 "애플리케이션을 *실행*하는 코드와, 그것을 *배포*하는 매니페스트를 어떻게 다룰 것인가"입니다. 권장되는 답은 단순합니다 — **둘을 분리하라**는 것입니다.

이 분리가 필요한 이유는 두 자산의 변경 성격이 근본적으로 다르기 때문입니다. 예컨대 Deployment의 복제본 수를 바꾸는 일은 애플리케이션 코드 자체는 그대로인데도, 코드 저장소와 배포 저장소가 합쳐져 있으면 변경되지 않은(그리고 이미 프로덕션에 있는) 코드베이스를 다시 빌드하고 테스트하게 만들 수 있습니다. 또한 환경 변경에 대한 승인 과정은 코드 변경에 대한 승인과 성격이 다르며, 개발자의 지속적 통합(CI) 흐름을 방해해서는 안 됩니다. 두 흐름을 분리하면 이런 불필요한 마찰이 사라집니다.

여기서 한 걸음 더 들어가면, 두 저장소는 흔히 *서로 다른 Git 워크플로*를 채택하게 됩니다. 애플리케이션 개발에는 오랫동안 사실상 표준이었던 git-flow를 쓰는 조직이 많지만, GitOps 저장소에는 다수의 DevOps 엔지니어가 트렁크 기반 개발(trunk based development)을 채택합니다. 이 둘은 근본적으로 다른 종류의 워크플로이며, 본능적으로 코드에 쓰던 git-flow를 그대로 매니페스트에 적용하려는 시도가 바로 다음에 다룰 문제의 출발점이 됩니다.

### 원칙 2: 환경은 브랜치가 아니라 디렉터리로 구분하라

대부분의 조직이 거쳐야 하는 가장 큰 사고방식의 전환은, **환경별 설정을 브랜치가 아니라 별도의 폴더(디렉터리)에 두는 것**입니다. Git을 쓰다 보니 Kubernetes YAML을 코드처럼 다루려는 본능("infrastructure *as code*"라는 이름 자체가 그렇게 유도합니다)이 작동하지만, 여기서 다루는 것은 코드가 아니라 *매니페스트의 승격(promotion)*이라는 점을 명심해야 합니다.

환경을 장기 존속 브랜치(long-lived branch)로 관리하지 않는 이유를 구체적으로 풀어 보면 다음과 같습니다.

- **승격은 단순한 머지가 아니다.** 한 환경에서 다음 환경으로의 승격은 브랜치 간 머지처럼 간단하지 않습니다. 매니페스트를 승격하는 것이지 코드를 승격하는 것이 아니기 때문입니다.
- **환경별 설정은 본질적으로 다르다.** Secret이나 ConfigMap은 환경마다 근본적으로 달라서, 그대로 머지되어서는 안 되는 종류의 차이입니다.
- **이미지 업데이트 승격이 악몽이 된다.** git-flow로 환경 브랜치를 운영하면 갱신된 이미지를 승격할 때 모든 변경을 일일이 cherry-pick 해야 하며, 이는 부담만 크고 득이 없습니다.

대신, 앞서 Config Management Tool을 다루는 장에서 본 Kustomize의 `base`/`overlays` 구조처럼 환경별 차이를 폴더에 담아 두면, 이 모든 마찰이 사라집니다. 트렁크 기반 개발과 Kustomize·Helm을 결합해 공통점을 템플릿화하고 환경별 차이만 별도로 두는 것이 GitOps 워크플로를 단순화하는 이상적 방식입니다.

두 접근의 차이를 정리하면 다음과 같습니다.

| 구분 | 환경별 브랜치 (지양) | 환경별 디렉터리 (권장) |
|---|---|---|
| 승격 방식 | 브랜치 간 머지·cherry-pick | 디렉터리(오버레이)에 변경 반영 |
| 환경별 설정(Secret·ConfigMap) | 머지 시 충돌·혼선 유발 | 폴더별로 명확히 분리 보존 |
| 이미지 업데이트 전파 | 변경마다 cherry-pick 필요 | 해당 오버레이만 수정 |
| 적합한 워크플로 | git-flow의 관성 | 트렁크 기반 개발 |

### 원칙 3: DRY — "같은 YAML을 반복하지 마라"

프로그래밍의 DRY(Don't Repeat Yourself) 원칙은 GitOps 저장소에서 "Don't Repeat YAML"로 변주됩니다. 모든 것을 Git에 저장하다 보면 거의 동일한 YAML이 수없이 쌓이게 되는데, Kustomize와 Helm을 활용하면 이 중복을 피할 수 있습니다. 구체적으로는 Kustomize로 배포의 기반(base) 설정을 두고 필요한 차이(delta)만 패치된 오버레이로 저장하는 방식이 저장소를 깔끔하고 이해하기 쉽게 유지해 줍니다.

다만 패칭으로 해결되지 않는 상황도 있습니다. 값을 사전에 알 수 있을 때는 Kustomize 패칭이 쉽지만, 대상 클러스터에 적용되기 전까지 값을 알 수 없는 경우가 존재합니다. 대표적인 예가 Ingress 객체의 `host` 필드로, 다수 클러스터에 배포할 때 각 클러스터의 정규화된 도메인 이름(FQDN)을 미리 알 수 없습니다. 이런 시나리오에서는 설정을 파라미터화하는 것이 합리적이며, 특히 Helm의 lookup 기능이 빛을 발합니다. 결국 실무에서는 Kustomize와 Helm을 조합해 GitOps 저장소의 YAML 중복을 최소화하게 됩니다.

### 조직에 맞는 구조를 찾는 출발점

앞서 강조했듯 단 하나의 "참된" 레이아웃은 없습니다. 저장소 구조는 조직이 서로 어떻게 소통하고 현재의 배포 워크플로가 어떻게 표현되는지에 크게 의존합니다. 서로 다른 워크플로를 가진 조직 단위는 흔히 사일로(silo)라 불리지만, 더 정확히는 "경계(boundary)"라 부르는 편이 맞습니다. 개발자가 플랫폼 설정을 건드리지 않듯, 플랫폼을 담당하는 운영자도 보통 개발자의 소스 코드를 바꾸지 않기 때문입니다.

이러한 경계 위에서 선택할 수 있는 대표적 패턴으로는, 저장소 하나가 클러스터 하나에 대응하는 1:1 레이아웃(ApplicationSet으로 부트스트래핑), 모노레포(monorepo)에서 다수 클러스터를 다루는 방식, 그리고 "팀별 저장소" 대 "애플리케이션별 저장소"의 트레이드오프 등이 있습니다. 어느 쪽을 택하든, 코드와 배포의 분리·환경의 디렉터리 분리·DRY라는 세 원칙을 토대로 삼으면 조직에 맞는 최적 구조로 수렴하기가 한결 쉬워집니다.

정리하면, GitOps 저장소 구조에 만능 답은 없지만, 검증된 나침반은 분명히 존재합니다. 애플리케이션 코드와 배포 매니페스트를 분리하고, 환경을 브랜치가 아닌 디렉터리로 나누며, Kustomize와 Helm으로 YAML 중복을 줄이는 것입니다. 이 구조 위에서 누가 어떤 권한으로 각 경계를 다룰 수 있는지를 통제하는 일은, 이어지는 RBAC와 Projects를 다루는 장에서 살펴봅니다.

## RBAC와 Projects를 이용한 멀티 테넌시 접근 제어

여러 팀이 하나의 Argo CD 인스턴스를 공유하는 Multi-Tenant 설치에서는 "누가 무엇을 할 수 있는가"라는 질문이 중요해집니다. Argo CD는 멀티 테넌시와 인가를 위한 RBAC 정책을 기능으로 제공합니다. 아키텍처 관점에서 보면, 외부 ID 공급자로의 인증·인가 위임과 RBAC 강제는 모두 API Server의 책임으로 모입니다.

### RBAC와 API Server

API Server는 Web UI·CLI·CI/CD 시스템이 사용하는 API를 노출하는 컴포넌트로, 다음과 같은 책임을 가집니다.

- 애플리케이션 관리 및 상태 보고
- 동기화·롤백 등 애플리케이션 작업 호출
- 저장소·클러스터 자격 증명 관리(Kubernetes Secret으로 저장)
- 외부 ID 공급자로의 인증 및 인가 위임
- RBAC 강제
- Git 웹훅 이벤트의 리스너/포워더

인증은 외부 ID 공급자(OIDC·OAuth2·LDAP·SAML 2.0 등)로 위임될 수 있으며, 인증으로 "당신이 누구인가"가 확인되면 그다음 "당신이 무엇을 할 수 있는가"를 판정하는 것이 RBAC입니다.

### AppProject: 핵심 선언 리소스의 하나

Argo CD에서 `AppProject`는 `Application`·`ApplicationSet`과 함께 선언적으로 관리되는 핵심 리소스입니다. 예를 들어 namespace 수준 권한으로 설치하는 경우, 기본 역할만으로는 같은 클러스터에 `Application`·`ApplicationSet`·`AppProject` 같은 Argo CD 리소스만 배포할 수 있으며, 새 역할을 정의해 `argocd-application-controller` 서비스 계정에 바인딩함으로써 이를 조정할 수 있습니다.

### 대상 클러스터 쪽 권한과의 관계

외부 클러스터를 등록할 때 `argocd cluster add CONTEXTNAME` 명령은 해당 컨텍스트의 `kube-system` 네임스페이스에 `argocd-manager` 서비스 계정을 설치하고, 이 계정을 admin 수준 ClusterRole에 바인딩합니다. Argo CD는 이 서비스 계정 토큰으로 배포·모니터링 같은 관리 작업을 수행합니다. `argocd-manager-role` 역할의 규칙은 제한된 네임스페이스·그룹·종류에 대해서만 `create`·`update`·`patch`·`delete` 권한을 갖도록 수정할 수 있으나, `get`·`list`·`watch` 권한은 Argo CD가 동작하기 위해 클러스터 범위에서 필요합니다.

내부 클러스터(즉 Argo CD가 실행되는 같은 클러스터)에 배포할 때는 `https://kubernetes.default.svc`를 애플리케이션의 K8s API 서버 주소로 사용합니다.

### Core 모드에서의 인가: 경계가 Git과 Kubernetes로 옮겨 간다

Core 모드는 Argo CD를 headless로 실행하는 설치 방식으로, 다음 기능들이 제공되지 않습니다.

- Argo CD RBAC 모델
- Argo CD API
- Argo CD Notification Controller
- OIDC 기반 인증

따라서 위에서 설명한 RBAC 강제는 동작하지 않습니다. Core 모드의 멀티 테넌시는 엄격히 Git push 권한에 기반한 GitOps로 성립하며, Argo CD는 오직 Kubernetes RBAC에만 의존합니다. CLI나 프로세스가 명령을 실행하려면 Argo CD 네임스페이스에 접근할 수 있어야 하고, `Application`·`ApplicationSet` 리소스에 대한 적절한 권한을 가져야 합니다. 즉 통제 지점이 Argo CD 바깥(Git push 권한과 Kubernetes RBAC)으로 이동합니다.

요컨대 Multi-Tenant 설치에서 멀티 테넌시는 API Server가 강제하는 RBAC 정책으로 구현되고, 이 설정 역시 선언적으로 Git에 두고 버전 관리할 수 있습니다. 각 정책 필드의 전체 레퍼런스와 사용 가능한 자원·액션 목록은 공식 문서의 RBAC Configuration과 Project Specification Reference에서 확인할 수 있습니다.

## Feature Maturity 확인과 다음 단계 학습 경로

지금까지 Argo CD의 핵심 동작 원리와 실천법을 모두 살펴봤습니다. 마지막으로, 새로운 기능을 도입할 때 반드시 거쳐야 할 안전 점검 단계인 **기능 성숙도(Feature Maturity)** 확인을 정리하고, 이후 깊이 파고들 만한 학습 경로를 안내합니다. ApplicationSet을 다루는 장에서 Progressive Syncs와 AppSets in any Namespace가 각각 Beta 단계라고 언급했는데, 그 "단계"가 무엇을 뜻하고 어떻게 확인하는지가 이 장의 주제입니다.

### 성숙도 등급과 그 위험

Argo CD의 기능에는 안정성과 성숙도를 나타내는 상태가 부여될 수 있습니다. 안정(Stable) 단계에 도달하지 못한 기능은 **Alpha** 또는 **Beta**로 표시되며, 공식 문서의 Feature Maturity 페이지에서 도입 버전과 함께 정리되어 있습니다.

핵심 주의 사항은 분명합니다. **Alpha와 Beta 기능은 하위 호환성을 보장하지 않으며, 향후 릴리스에서 호환성이 깨지는 변경(breaking change)이 발생할 수 있습니다.** 특히 Argo CD 업그레이드를 직접 통제할 수 없는 환경이라면, 프로덕션에서 이러한 기능에 의존하지 않는 것이 강력히 권장됩니다. 더 나아가, Alpha 기능이 제거될 경우 업그레이드 이후 리소스가 예측 불가능한 상태로 바뀔 수도 있습니다. 따라서 사용 중인 기능을 문서화하고, 업그레이드 전에 릴리스 노트를 반드시 검토해야 합니다.

공식 문서에 정리된 비안정 기능 일부를 도입 버전·상태와 함께 보면 다음과 같습니다.

| 기능 | 도입 버전 | 상태 |
|---|---|---|
| AppSet Progressive Syncs | v2.6.0 | Beta |
| Proxy Extensions | v2.7.0 | Beta |
| Skip Application Reconcile | v2.7.0 | Alpha |
| AppSets in any Namespace | v2.8.0 | Beta |
| Cluster Sharding: round-robin | v2.8.0 | Alpha |
| Dynamic Cluster Distribution | v2.9.0 | Alpha |
| Cluster Sharding: consistent-hashing | v2.12.0 | Alpha |
| Service Account Impersonation | v2.13.0 | Alpha |
| Source Hydrator | v2.14.0 | Alpha |

문서는 이 외에도 각 기능을 활성화하는 구체적인 위치 — `Application`·`AppProject`·`ApplicationSet` CRD의 특정 속성, 그리고 `argocd-cmd-params-cm` ConfigMap이나 각 컴포넌트 Deployment의 환경 변수 — 까지 표로 제시합니다. 예컨대 Progressive Syncs는 `ApplicationSet` CRD의 `spec.strategy.*`·`status.applicationStatus.*` 속성과 `applicationsetcontroller.enable.progressive.syncs` 설정으로 제어되며, 모두 Beta로 표시되어 있습니다. 어떤 기능을 켜기 전에 이 표에서 상태를 확인하는 습관이, 예기치 못한 업그레이드 사고를 막는 가장 단순한 방어책입니다.

### 다음 단계 학습 경로

이 가이드는 Argo CD를 시작하는 데 필요한 토대를 다뤘지만, 운영 규모로 나아가면 더 깊이 파고들 영역이 많습니다. 공식 문서를 기준으로 다음 경로를 권합니다.

- **동기화 세부 제어**: Sync Phases and Waves, Sync Windows, Selective Sync, Sync Options 등은 PreSync·Sync·PostSync 훅을 넘어 동기화 순서와 시점을 정밀하게 통제하는 방법을 다룹니다.
- **사용자 관리와 보안**: OIDC·SAML 등 외부 ID 공급자 연동(User Management), RBAC Configuration, TLS 구성, Security Overview는 멀티 테넌시 접근 제어를 다루는 장에서 소개한 내용을 실제 설정 수준으로 확장합니다.
- **고가용성과 운영**: High Availability, Dynamic Cluster Distribution, Reconcile Optimization, Metrics, Disaster Recovery는 프로덕션 규모 운영의 핵심 주제입니다.
- **확장과 통합**: Config Management Plugins, Notifications, Resource Health 커스터마이징, CI 파이프라인 자동화는 Argo CD를 조직의 워크플로에 맞춰 확장하는 길을 제시합니다.

기초를 다지는 데에는 공식 문서의 Understand The Basics가 권하듯 Docker·Kubernetes·Kustomize·Helm에 대한 이해를 함께 다져 두는 것이 도움이 됩니다. 그리고 Argo CD는 커뮤니티가 활발히 개발 중인 프로젝트이므로, GitHub의 릴리스 목록과 공식 블로그를 통해 변화하는 기능 성숙도와 새로운 기능을 꾸준히 확인하는 것이 안전하고 효과적인 운영의 마지막 열쇠입니다.
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
* <https://www.harness.io/blog/gitops-principles>
* <https://opengitops.dev>
* <https://akuity.io/blog/gitops-best-practices-whitepaper>
* <https://about.gitlab.com/topics/gitops>
