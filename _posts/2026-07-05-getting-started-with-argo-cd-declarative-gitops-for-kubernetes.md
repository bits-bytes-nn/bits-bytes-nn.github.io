---
layout: post
title: "Getting Started with Argo CD: Declarative GitOps for Kubernetes"
date: 2026-07-05 00:53:54
author: "bits-bytes-nn"
categories: ["Tech Guides"]
tags: []
cover: /assets/images/tech-guides.jpg
use_math: true
---

## GitOps란 무엇인가: 원칙과 사고 모델

GitOps는 인프라와 애플리케이션 구성을 관리하기 위한 실천 방식의 집합입니다. 그 핵심에는 하나의 단순하지만 강력한 발상이 자리합니다. 시스템이 "어떻게 되어야 하는가"에 대한 선언을 Git 저장소에 저장하고, 이 저장소를 유일한 진실의 원천(single source of truth)으로 삼아 실제 시스템이 항상 그 선언과 일치하도록 만드는 것입니다. GitOps라는 용어 자체가 "Git"과 "Operations"의 결합이며, 버전 관리 시스템인 Git을 운영의 중심에 놓는다는 지향을 담고 있습니다.

이 접근은 갑자기 등장한 것이 아니라 Infrastructure as Code(IaC)의 연장선 위에 있습니다. IaC는 인프라의 원하는 상태를 명시한 구성 파일을 받아 인프라를 자동으로 그 상태로 만드는 기술이며, 이를 선언적 구성(declarative configuration)이라 부릅니다. GitOps는 여기서 한 걸음 더 나아가 그 선언적 구성 전체를 Git 저장소에 담습니다. 결과적으로 Git 저장소는 시스템 상태 전부를 보관하게 되고, 상태 변화의 추적·검토가 모두 Git의 이력을 통해 가능해집니다.

### 왜 선언적인가: 명령형과의 근본적 차이

GitOps를 이해하기 위해 가장 먼저 넘어야 할 개념적 장벽은 선언적(declarative) 사고와 명령형(imperative) 사고의 차이입니다. 두 방식의 구분은 하나의 단순한 축으로 압축됩니다. 명령형은 "어떻게(how)" 하는가를 다루고, 선언적은 "무엇을(what)" 원하는가를 다룹니다.

비유하자면 명령형은 어떤 사람에게 프로젝트를 맡기면서 밟아야 할 구체적인 단계와 최종 결과물을 어떻게 제시할지까지 일일이 지시하는 것과 같습니다. 반대로 선언적 방식은 구체적인 지시 없이 완성해야 할 목표만 건네고, 그 목표에 도달하는 경로는 시스템에 맡깁니다. 목표가 어떻게 달성되는지는 신경 쓰지 않고 단지 완성되기만을 원하는 것입니다.

Kubernetes는 본질적으로 선언적입니다. 사용자는 YAML 파일에 원하는 상태를 기술하고, 시스템은 그 파일을 한 줄씩 읽어 지시와 규칙을 시스템 상태에 적용합니다. 다음과 같은 매니페스트를 보면 어떤 API 버전을 사용하는지, 어떤 Kubernetes 리소스인지, 복제본(replica)을 몇 개 실행할지, 파드에서 어떤 컨테이너를 구동할지가 모두 사실(fact)로 표현되어 있습니다.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  replicas: 3          # 원하는 복제본 수 — 명령이 아니라 사실의 선언
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
        - image: nginx
          name: nginx
```

여기서 결정적인 성질이 드러납니다. 시스템 상태가 원하는 상태와 어긋나면, 시스템은 그 모순을 자동으로 바로잡아 시스템 상태가 항상 원하는 상태와 일치하도록 시도합니다. 예컨대 위 매니페스트가 복제본 3개를 선언했는데 실제로는 1개만 떠 있다면, 선언적 시스템은 나머지 2개를 스스로 만들어 냅니다. "복제본을 2개 더 생성하라"는 명령을 내리지 않아도 되는 것입니다. 이 자기 수복(self-healing)적 성질이 GitOps가 선언적 방식을 택하는 핵심 이유입니다.

두 접근 방식의 성격 차이를 정리하면 다음과 같습니다.

| 관점 | 명령형(Imperative) | 선언적(Declarative) |
| --- | --- | --- |
| 초점 | 결과에 도달하는 절차(how) | 원하는 최종 상태(what) |
| 지시 내용 | 실행할 단계의 나열 | 목표 상태의 기술 |
| 상태 불일치 시 | 사용자가 교정 단계를 직접 수행 | 시스템이 자동으로 원하는 상태로 조정 |
| 롤백 | 역순 절차를 새로 작성/실행 | 이전 버전 선언을 다시 적용 |
| GitOps 적합성 | 낮음 | 높음(전제 조건) |

### GitOps의 네 가지 원칙

GitOps는 특정 도구가 아니라 원칙의 집합으로 정의됩니다. 벤더 중립적으로 결성된 OpenGitOps 프로젝트는 GitOps로 관리되는 시스템이 만족해야 할 네 가지 근본 원칙을 정립했습니다. 이 네 원칙은 GitOps의 사고 모델을 이루는 골격이므로 하나씩 짚어 보겠습니다.

**첫째, 선언적(Declarative).** GitOps로 관리되는 시스템은 원하는 상태를 반드시 선언적으로 표현해야 합니다. 앞서 살펴본 대로, 시스템의 청사진을 절차가 아닌 목표 상태로 기술하고 이를 유일한 진실의 원천으로 삼는 것이 출발점입니다.

**둘째, 버전 관리되고 불변임(Versioned and Immutable).** 원하는 상태는 불변성, 버전 관리, 완전한 이력 보존을 보장하는 방식으로 저장됩니다. Git은 특정 시점의 파일 스냅샷을 찍고 그 스냅샷에 대한 참조를 저장하는데, 이 스냅샷은 불변(immutable)입니다. 변경이 발생하면 원본을 고치는 것이 아니라 새로운 버전의 스냅샷으로 남고, 원래 스냅샷은 그대로 보존됩니다. 이 성질 덕분에 이전 버전으로 되돌리는 롤백(rollback)이 쉬워집니다. 새 버전에 문제가 생겼을 때 마지막 안정 빌드로 손쉽게 복귀할 수 있고, 시간 순으로 정렬된 모든 인프라 변경 이력은 규정 준수(compliance)와 보안을 위한 감사(auditing), 그리고 실패한 배포의 문제 해결을 수월하게 만듭니다.

**셋째, 자동으로 가져옴(Pulled Automatically).** 소프트웨어 에이전트가 원하는 상태 선언을 소스로부터 자동으로 가져옵니다. 여기서 소프트웨어 에이전트란 사용자나 조직을 위해 작업을 지속적으로 수행하는 프로그램으로, 대표적인 예가 Kubernetes 컨트롤러입니다. 이 원칙의 미묘한 점은 GitOps가 웹훅(webhook)이나 이벤트 기반 방식에 의존하지 않고 매니페스트를 스스로 끌어온다(pull)는 데 있습니다. 클러스터가 여럿이면 매니페스트도 여럿인데, 이를 각 클러스터에 사람이 일일이 적용하는 것은 비현실적입니다. 에이전트는 Git 저장소 같은 소스에서 매니페스트를 자동으로 가져와 클러스터에 적용함으로써 이 수고를 대신합니다.

**넷째, 지속적으로 조정됨(Continuously Reconciled).** 소프트웨어 에이전트는 실제 시스템 상태를 지속적으로 관찰하고 원하는 상태를 적용하려 시도합니다. 에이전트는 일종의 조정(reconciliation) 에이전트로서 Git을 유일한 진실의 원천으로 삼아, 시스템 상태를 원하는 상태에 맞추어 고치는 일을 끊임없이 반복합니다.

이 네 원칙은 목적지를 정의할 뿐 구현 방법이나 모범 사례를 지시하지는 않는다는 점에 유의해야 합니다. 즉 원칙은 구현을 안내하는 지침이지 청사진 자체가 아닙니다. 구체적인 저장소 구조나 워크플로우 전략은 이후의 내용에서 별도로 다룹니다.

### 조정 루프라는 사고 모델

네 원칙을 하나의 동작하는 그림으로 합치면 조정 루프(reconciliation loop)라는 사고 모델이 됩니다. GitOps 도구는 두 가지 상태를 끊임없이 비교합니다. 하나는 Git에 선언된 원하는 상태(target state)이고, 다른 하나는 클러스터에서 실제로 돌아가는 라이브 상태(live state)입니다. 개념적으로는 다음과 같이 표현할 수 있습니다.

$$
\text{drift} = \text{live state} \ne \text{target state}
$$

두 상태가 어긋나면(drift가 감지되면) 에이전트는 원하는 상태를 다시 적용하여 격차를 좁힙니다. 바로 이 지점에서 GitOps의 실질적 이점이 나옵니다. 원하는 상태가 Git에 선언되고 승인되면 이후의 변경은 시스템 전체에 자동으로 반영되며, 변경을 위해 클러스터 자격 증명을 직접 다룰 필요가 없어 배포까지의 시간을 줄일 수 있습니다. 또한 어떤 환경에서 문제가 발생하더라도 그 문제를 특정 pull request로 추적할 수 있어, 해당 pull request가 도입한 변경에 집중해 원인을 찾을 수 있습니다.

이 사고 모델은 곧 살펴볼 Argo CD의 동작 방식과 정확히 맞닿아 있습니다. Argo CD는 Kubernetes 컨트롤러로 구현되어 실행 중인 애플리케이션을 지속적으로 감시하고, 라이브 상태를 Git에 명시된 원하는 상태와 비교하며, 두 상태가 어긋난 애플리케이션을 `OutOfSync`로 표시합니다. 여기서 다룬 네 원칙과 조정 루프의 사고 모델은, 이어지는 내용에서 Argo CD의 개념과 아키텍처를 이해하기 위한 공통의 어휘가 됩니다.

## Argo CD 핵심 개념

Argo CD는 Kubernetes를 위한 선언적 GitOps 지속적 전달(continuous delivery) 도구입니다. 앞서 다룬 조정 루프의 사고 모델을 실제 제품으로 구현할 때, Argo CD는 몇 가지 고유한 용어와 개념 위에서 동작합니다. 이 어휘들을 정확히 구분해 두면 이후의 설치·운영·동기화 과정에서 화면에 표시되는 상태 문구와 CLI 출력이 무엇을 뜻하는지 곧바로 해석할 수 있습니다. 여기서는 GitOps 일반 원칙이 아니라 Argo CD에 특화된 개념들만 골라 하나씩 다룹니다.

### Application: 관리의 기본 단위

Argo CD에서 가장 핵심이 되는 대상은 **Application**입니다. Application은 하나의 매니페스트에 정의된 Kubernetes 리소스의 묶음(a group of Kubernetes resources)이며, 그 자체가 Kubernetes의 사용자 정의 리소스(Custom Resource Definition, CRD)로 구현됩니다. 이 점이 Argo CD의 성격을 잘 드러냅니다. 즉 "무엇을 배포할지"에 대한 정의조차 또 하나의 Kubernetes 리소스로 표현되며, 따라서 Application 자체도 선언적으로 다룰 수 있습니다.

하나의 Application은 대략 세 가지 질문에 답하는 정보로 구성됩니다. 어디에서 매니페스트를 가져오는가(소스 저장소와 경로), 어디에 배포하는가(대상 클러스터와 네임스페이스), 그리고 어떻게 동기화할 것인가(동기화 정책)입니다. Application을 CLI로 생성할 때 넘기는 인자들이 이 구조를 그대로 보여 줍니다.

```bash
argocd app create guestbook \
  --repo https://github.com/argoproj/argocd-example-apps.git \
  --path guestbook \
  --dest-server https://kubernetes.default.svc \
  --dest-namespace default
```

여기서 `--repo`와 `--path`는 원하는 상태가 담긴 소스의 위치를, `--dest-server`와 `--dest-namespace`는 리소스가 실제로 적용될 대상을 지정합니다. Argo CD가 실행 중인 것과 같은 클러스터(내부)에 배포하는 경우에는 대상 서버 주소로 `https://kubernetes.default.svc`를 사용합니다.

### 상태를 이루는 두 축: Target state와 Live state

앞서 조정 루프를 설명하면서 Git에 선언된 원하는 상태와 클러스터의 실제 상태를 비교한다고 했는데, Argo CD는 이 두 상태에 각각 고유한 이름을 붙입니다.

- **Target state(대상 상태)**는 애플리케이션의 원하는 상태로, Git 저장소의 파일들로 표현됩니다.
- **Live state(라이브 상태)**는 그 애플리케이션의 실제 상태로, 실제로 어떤 파드 등이 배포되어 있는가를 가리킵니다.

이 두 상태의 이름을 별도로 기억해 두어야 하는 이유는, Argo CD의 거의 모든 판단이 "Target state와 Live state를 비교한 결과"로 환원되기 때문입니다.

### Sync status와 Sync: 일치 여부와 일치시키는 행위

두 상태를 놓고 Argo CD가 내리는 첫 번째 판단이 **Sync status(동기화 상태)**입니다. Sync status는 라이브 상태가 대상 상태와 일치하는지 여부, 즉 "배포된 애플리케이션이 Git이 명시한 것과 같은가?"에 대한 답입니다. 두 상태가 어긋난 애플리케이션은 `OutOfSync`로 표시됩니다. 예를 들어 방금 Application을 만들었지만 아직 어떤 Kubernetes 리소스도 생성되지 않았다면, 대상 상태에는 리소스가 있으나 라이브 상태에는 아무것도 없으므로 애플리케이션은 처음에 `OutOfSync` 상태가 됩니다.

여기서 반드시 구분해야 할 것은 상태(status)와 행위(action)입니다. **Sync(동기화)**는 애플리케이션을 대상 상태로 옮기는 *과정*을 뜻합니다. 즉 Kubernetes 클러스터에 변경을 적용하는 행위입니다. Sync status가 "지금 일치하는가?"라는 정적인 질문이라면, Sync는 "일치하도록 만들어라"라는 동적인 명령입니다. `argocd app sync`를 실행하면 Argo CD는 저장소에서 매니페스트를 가져와 사실상 `kubectl apply`를 수행합니다.

그리고 이 Sync가 성공했는지 실패했는지를 별도로 추적하는 것이 **Sync operation status(동기화 작업 상태)**입니다. Sync가 어긋남을 없앨 목적으로 시도되더라도 그 시도 자체가 성공한다는 보장은 없기 때문에, "일치 여부"와 "적용 시도의 성공 여부"는 서로 다른 정보로 분리되어 있습니다.

### Health: 리소스가 올바로 동작하는가

Sync status와 나란히, 그러나 완전히 다른 축으로 존재하는 개념이 **Health(상태 건강도)**입니다. Health는 애플리케이션이 올바르게 실행되고 있는가, 요청을 처리할 수 있는가를 나타냅니다. 이 구분은 초보자가 가장 자주 혼동하는 지점이므로 강조할 가치가 있습니다.

Sync와 Health는 서로 독립적입니다. 애플리케이션이 Git과 완벽히 일치(`Synced`)하더라도 컨테이너가 기동에 실패해 건강하지 않을 수 있고, 반대로 Git과 어긋나 있(`OutOfSync`)지만 이전에 배포된 버전이 여전히 정상 동작 중일 수도 있습니다. 즉 하나는 "선언과 실제가 같은가"를, 다른 하나는 "실제가 잘 돌아가는가"를 묻습니다. 실제로 `argocd app get`의 출력에서도 이 둘은 각각 `Sync Status`와 `Health Status`라는 별개의 필드로 나타나며, 리소스 단위로도 나뉘어 표시됩니다.

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

위 출력은 아직 동기화 전이라 애플리케이션 전체가 `OutOfSync`이며, 아무것도 배포되지 않았으므로 Health가 `Missing`으로 나타난 예입니다. `SyncStatus` 옆의 `(1ff8a67)`는 비교 기준이 된 Git 커밋을 가리킵니다.

### Refresh: 비교를 다시 수행하기

**Refresh(새로 고침)**는 Git의 최신 코드를 라이브 상태와 다시 비교하여 무엇이 달라졌는지 계산하는 작업입니다. Sync가 실제 클러스터에 변경을 *적용*하는 행위였다면, Refresh는 그런 적용 없이 대상 상태와 라이브 상태의 차이를 *재계산*하기만 하는 읽기 성격의 작업입니다. 이 구분 덕분에, 클러스터에 손을 대지 않고도 "지금 Git과 무엇이 어긋나 있는지"만 확인할 수 있습니다.

### Tool과 Application source type: 매니페스트를 만드는 방법

Argo CD는 Git 저장소의 파일을 그대로 배포하는 것이 아니라, 그 파일들로부터 최종 Kubernetes 매니페스트를 생성한 뒤 적용합니다. 이 생성 작업을 담당하는 것이 **Tool(도구)**입니다. Tool은 파일들의 디렉터리로부터 매니페스트를 만들어 내는 도구를 뜻하며, Kustomize가 대표적인 예입니다. 문서에서는 같은 대상을 **Configuration management tool(구성 관리 도구)**이라고도 부르는데, 둘은 같은 개념을 가리킵니다.

**Application source type(애플리케이션 소스 타입)**은 특정 애플리케이션을 빌드하는 데 어떤 Tool이 사용되는가를 나타냅니다. Argo CD는 여러 방식으로 매니페스트를 지정할 수 있습니다.

- Kustomize 애플리케이션
- Helm 차트
- Jsonnet 파일
- 평문 YAML/JSON 매니페스트의 디렉터리
- 구성 관리 플러그인으로 설정된 임의의 커스텀 구성 관리 도구

여기서 마지막 항목이 **Configuration management plugin(구성 관리 플러그인)**으로, 커스텀 도구를 뜻합니다. 이러한 소스 도구들의 상세한 활용과 선택 전략은 이 가이드의 뒤쪽에서 별도로 다룹니다.

### 핵심 개념 정리 표

지금까지의 개념들은 서로 짝을 이루거나 쉽게 혼동되므로, 하나의 표로 대비해 두면 이후 장에서 상태 화면을 해석할 때 참조점이 됩니다.

| 개념 | 무엇을 가리키는가 | 성격 | 자주 혼동되는 대상 |
| --- | --- | --- | --- |
| Application | 매니페스트로 정의된 Kubernetes 리소스 묶음(CRD) | 관리 단위(객체) | — |
| Target state | Git 저장소 파일로 표현된 원하는 상태 | 상태(선언) | Live state |
| Live state | 실제로 배포된 파드 등 실제 상태 | 상태(실제) | Target state |
| Sync status | 라이브 상태가 대상 상태와 일치하는지 여부(`OutOfSync` 등) | 상태(판단) | Sync / Health |
| Sync | 애플리케이션을 대상 상태로 옮기는 과정(예: `kubectl apply`) | 행위 | Sync status |
| Sync operation status | 그 Sync가 성공했는지 여부 | 상태(결과) | Sync status |
| Refresh | Git 최신 코드와 라이브 상태를 다시 비교해 차이를 계산 | 행위(읽기) | Sync |
| Health | 애플리케이션이 올바로 동작하며 요청을 처리하는가 | 상태(건강도) | Sync status |
| Tool / 구성 관리 도구 | 파일 디렉터리로부터 매니페스트를 생성하는 도구(예: Kustomize) | 도구 | Application source type |
| Application source type | 해당 애플리케이션을 빌드하는 데 쓰이는 Tool의 종류 | 분류 | Tool |
| 구성 관리 플러그인 | 커스텀 도구 | 도구(확장) | Tool |

이 표에서 특히 눈여겨볼 축은 **상태와 행위의 구분**, 그리고 **Sync(일치 여부)와 Health(동작 건강도)의 분리**입니다. Argo CD의 UI와 CLI는 이 두 축을 병렬로 노출하므로, 어떤 애플리케이션이 문제를 겪을 때 "Git과 어긋난 것인지", "적용에 실패한 것인지", "적용은 됐지만 동작이 불건강한 것인지"를 세 갈래로 나누어 진단할 수 있습니다. 이 개념들이 실제 컴포넌트를 통해 어떻게 산출되는지는 아키텍처를 다루는 다음 장에서 이어집니다.
&lt;/none&gt;

## Argo CD 아키텍처: 내부 동작 원리

Argo CD가 조정 루프를 실제로 구현하는 방식은 하나의 거대한 프로그램이 아니라 역할이 분리된 여러 컴포넌트의 협업입니다. Argo CD는 컴포넌트 기반 아키텍처를 염두에 두고 설계되었기 때문에, 필요에 따라 일부 컴포넌트만 설치해 더 최소한의 형태로 운영할 수도 있습니다. 이 유연성이 이후에 다룰 설치 방식(멀티테넌트, Core 등)의 차이를 만들어 내는 근원이므로, 각 컴포넌트가 무엇을 책임지는지 정확히 파악해 두면 설치·운영·문제 진단의 지도가 됩니다.

핵심 컴포넌트는 세 가지입니다. 요청을 받아들이는 **API Server**, 매니페스트를 생성하는 **Repository Server**, 그리고 상태를 비교하고 조정하는 **Application Controller**입니다. 여기에 캐싱을 담당하는 **Redis**가 더해집니다. 공식 문서가 제시하는 아키텍처 그림은 이 컴포넌트들이 어떻게 배치되는지를 한눈에 보여 줍니다.

![Argo CD Architecture](https://argo-cd.readthedocs.io/en/stable/assets/argocd_architecture.png)

### Application Controller: 조정의 심장

Application Controller는 앞서 사고 모델로 다룬 조정 루프를 코드로 옮긴 컴포넌트입니다. 이것은 Kubernetes 컨트롤러로 구현되어 실행 중인 애플리케이션을 지속적으로 감시하고, 현재의 라이브 상태를 Git 저장소에 명시된 원하는 대상 상태와 비교합니다. 이 비교 결과 두 상태가 어긋난 애플리케이션은 `OutOfSync`로 판정됩니다.

컨트롤러의 책임은 단순히 차이를 감지하는 데서 그치지 않습니다. 감지된 `OutOfSync` 상태에 대해 선택적으로 교정 조치(corrective action)를 취하며, 이 교정이 바로 라이브 상태를 대상 상태로 되돌리는 동기화입니다. 또한 애플리케이션 수명 주기 이벤트에 대응하는 사용자 정의 훅(PreSync, Sync, PostSync)을 호출하는 것도 이 컨트롤러의 몫입니다. 즉 "Sync status를 계산하고, 필요하면 Sync를 실행하며, 그 과정에서 훅을 발동한다"는 일련의 판단과 행위가 모두 Application Controller에 집중되어 있습니다.

여기서 짚어야 할 미묘한 분업이 있습니다. Application Controller는 스스로 매니페스트를 렌더링하지 않습니다. "Git이 명시한 원하는 상태가 구체적으로 어떤 Kubernetes 매니페스트인가"라는 질문의 답은 Repository Server가 만들어 냅니다. 컨트롤러는 그 결과를 받아 라이브 상태와 비교하는 역할을 맡습니다.

### Repository Server: 매니페스트를 생성하는 내부 서비스

Repository Server는 애플리케이션 매니페스트를 담고 있는 Git 저장소의 로컬 캐시를 유지하는 내부 서비스입니다. 이 컴포넌트의 유일한 임무는 다음 입력들을 받아 최종 Kubernetes 매니페스트를 생성해 반환하는 것입니다.

- 저장소 URL(repository URL)
- 리비전(revision) — 커밋, 태그, 또는 브랜치
- 애플리케이션 경로(application path)
- 템플릿별 설정(template specific settings) — 예를 들어 파라미터, Helm의 `values.yaml`

이 목록을 보면 핵심 개념에서 다룬 Tool(구성 관리 도구)이 어디에서 실제로 작동하는지가 드러납니다. Kustomize나 Helm 같은 도구가 파일 디렉터리로부터 매니페스트를 빌드하는 작업이 바로 이 Repository Server 내부에서 일어납니다. 사용자가 Application에 지정한 소스 저장소·경로·파라미터가 여기로 전달되고, Repository Server는 그에 해당하는 리비전을 캐시에서 꺼내거나 갱신한 뒤 매니페스트를 렌더링합니다.

이 분리가 중요한 이유는, "무엇을 배포해야 하는가"(대상 상태의 계산)와 "지금 무엇이 배포되어 있는가"(라이브 상태의 관찰)가 서로 다른 컴포넌트의 관심사로 나뉘어 있기 때문입니다. Refresh가 클러스터를 건드리지 않고도 차이를 재계산할 수 있는 것도 이 구조 덕분입니다. Repository Server가 최신 리비전으로 대상 상태를 다시 생성하고, Application Controller가 그것을 라이브 상태와 비교하기만 하면 되기 때문입니다.

### API Server: 외부 세계와의 접점

API Server는 gRPC/REST 서버로, Web UI와 CLI, 그리고 CI/CD 시스템이 소비하는 API를 노출합니다. 사용자가 `argocd` CLI로 명령을 내리거나 브라우저에서 UI를 조작할 때 그 요청이 가장 먼저 도달하는 지점이 이곳입니다. API Server의 책임은 다음과 같이 여러 갈래로 나뉩니다.

- 애플리케이션 관리 및 상태 보고(application management and status reporting)
- 애플리케이션 작업의 호출(sync, rollback, 사용자 정의 액션 등)
- 저장소 및 클러스터 자격 증명 관리 — Kubernetes 시크릿(K8s secrets)으로 저장됨
- 인증, 그리고 외부 신원 공급자로의 인증 위임(auth delegation)
- RBAC 정책 시행(enforcement)
- Git 웹훅 이벤트의 수신·전달(listener/forwarder)

주목할 점은 저장소와 클러스터의 자격 증명이 Kubernetes 시크릿으로 저장된다는 사실입니다. 이것은 "선언조차 Kubernetes 리소스로 표현한다"는 Argo CD의 일관된 성격과 맞닿아 있으며, 외부 클러스터 등록이나 사설 저장소 연결이 어떻게 안전하게 관리되는지의 밑바탕이 됩니다. 또한 인증·RBAC·SSO 위임이 모두 API Server에 모여 있다는 점은, 뒤에서 다룰 Argo CD Core(헤드리스 모드)가 왜 이 컴포넌트를 생략하고 Kubernetes RBAC만으로 동작하는지를 이해하는 열쇠가 됩니다. API Server가 없으면 RBAC 모델·API·OIDC 기반 인증 같은 기능이 함께 사라지기 때문입니다.

### Redis: 부하를 줄이는 캐시

Redis는 위 세 컴포넌트와 나란히 설치되는 캐싱 계층입니다. Application Controller는 Redis 없이도 실행될 수는 있으나 권장되지 않습니다. 컨트롤러가 Redis를 중요한 캐싱 메커니즘으로 사용해 Kube API와 Git에 걸리는 부하를 줄이기 때문입니다. 매 조정 주기마다 클러스터 API를 전부 다시 조회하거나 저장소를 매번 새로 당겨오는 대신, 캐시된 상태를 활용함으로써 대규모 환경에서도 조정 루프가 감당 가능한 비용으로 반복될 수 있습니다. 이 때문에 Core 설치처럼 컴포넌트를 최소화한 방식에서도 Redis는 함께 포함됩니다.

### 컴포넌트를 관통하는 요청 흐름

세 컴포넌트가 어떻게 협력하는지는, 앞서 핵심 개념에서 살펴본 `argocd app sync guestbook` 한 줄이 내부적으로 유발하는 흐름을 따라가 보면 선명해집니다. 이 명령은 저장소에서 매니페스트를 가져와 `kubectl apply`를 수행한다고 설명했는데, 그 "사이"에서 벌어지는 일이 곧 아키텍처의 실제 동작입니다.

```text
argocd CLI
   │  (1) sync 요청 (gRPC/REST)
   ▼
API Server ──(2) 인증·RBAC 검사, sync 작업 호출──┐
   │                                            │
   │  (3) 대상 상태 매니페스트 요청              │
   ▼                                            │
Repository Server ──(4) Git 리비전 렌더링────────┘
   │      (repo URL, revision, path, values 입력)
   │      → 최종 Kubernetes 매니페스트 반환
   ▼
Application Controller
   │  (5) 라이브 상태(클러스터) vs 대상 상태 비교
   │  (6) OutOfSync면 kubectl apply 수행, 훅 실행
   ▼
대상 Kubernetes 클러스터
```

이 흐름을 단계별로 풀면 다음과 같습니다. CLI나 UI가 낸 요청은 (1)~(2) API Server에 도달해 인증·RBAC 검사를 거친 뒤 sync라는 애플리케이션 작업으로 변환됩니다. (3)~(4) 대상 상태를 확정하기 위해 Repository Server가 해당 저장소·리비전·경로·파라미터로부터 매니페스트를 렌더링합니다. (5)~(6) Application Controller가 이 대상 상태를 클러스터의 라이브 상태와 비교하고, 어긋난 부분이 있으면 클러스터에 변경을 적용하며 정의된 훅을 호출합니다. Git의 대상 상태에 가해진 어떤 수정이든 이 경로를 통해 지정된 대상 환경에 자동으로 반영될 수 있습니다.

Refresh는 이 흐름에서 (6)의 적용 단계만 빠진 형태로 이해할 수 있습니다. Repository Server가 최신 코드로 대상 상태를 다시 만들고 Application Controller가 라이브 상태와 비교해 차이를 계산하지만, 클러스터에 변경을 가하지는 않는 것입니다. 이렇게 보면 핵심 개념에서 구분했던 "행위(Sync)"와 "읽기(Refresh)"의 차이가 어느 컴포넌트가 어디까지 관여하느냐의 차이로 자연스럽게 환원됩니다.

### 컴포넌트별 책임 정리

세 컴포넌트의 역할을 하나의 표로 대비해 두면, 이후 설치 방식과 문제 진단을 논할 때 어떤 컴포넌트를 살펴야 하는지 판단하는 기준이 됩니다.

| 컴포넌트 | 유형 | 핵심 책임 | 조정 루프에서의 위치 |
| --- | --- | --- | --- |
| API Server | gRPC/REST 서버 | 인증·RBAC·SSO 위임, 애플리케이션 작업 호출(sync·rollback 등), 저장소/클러스터 자격 증명 관리(K8s 시크릿), Git 웹훅 수신·전달 | 외부 요청의 진입점 |
| Repository Server | 내부 서비스 | Git 저장소의 로컬 캐시 유지, 입력(URL·리비전·경로·파라미터·values)으로부터 최종 매니페스트 생성 | 대상 상태(target state) 산출 |
| Application Controller | Kubernetes 컨트롤러 | 라이브 상태와 대상 상태 비교, `OutOfSync` 감지, 교정 조치(동기화) 수행, PreSync/Sync/PostSync 훅 호출 | 비교·판정·조정의 실행 |
| Redis | 캐시 | 컨트롤러의 상태 캐싱으로 Kube API·Git 부하 경감(권장 구성) | 조정 루프의 성능 지지대 |

이 분업 구조에서 특히 되새길 것은, Argo CD가 "선언을 관찰하고 실제와 대조해 고친다"는 하나의 사고 모델을 세 갈래의 책임으로 나누어 구현했다는 점입니다. 어떤 애플리케이션이 문제를 겪을 때, 자격 증명이나 권한 문제라면 API Server를, 매니페스트가 의도대로 렌더링되지 않는 문제라면 Repository Server를, 상태 비교나 동기화·훅 실행 문제라면 Application Controller를 의심하는 식으로 진단의 초점을 좁힐 수 있습니다.

## 왜 Argo CD인가: GitOps의 필요성과 위치

GitOps 원칙은 목적지를 정의할 뿐 특정 도구를 지정하지 않습니다. 그렇다면 그 목적지에 도달하기 위한 여러 구현체 가운데 왜 Argo CD를 택하는가라는 질문이 남습니다. Argo CD의 공식 문서는 자신이 존재하는 이유를 두 문장으로 압축합니다. 첫째, 애플리케이션 정의·구성·환경은 선언적이고 버전 관리되어야 한다는 것입니다. 둘째, 애플리케이션의 배포와 수명 주기 관리는 자동화되고, 감사 가능하며, 이해하기 쉬워야 한다는 것입니다. 앞의 명제가 GitOps 원칙과 겹치는 "전제"라면, 뒤의 명제—특히 "감사 가능"과 "이해하기 쉬움"—가 Argo CD를 단순한 조정 루프 이상의 제품으로 만드는 지점입니다.

### 두 가지 설계 지향을 실제 기능으로 환산하기

Argo CD가 내세우는 두 지향은 추상적인 구호가 아니라 각각 구체적인 기능 집합으로 뒷받침됩니다. 공식 기능 목록을 이 두 지향에 대응시키면, "왜 손수 만든 파이프라인 대신 Argo CD인가"라는 물음의 답이 드러납니다.

| Argo CD의 설계 지향 | 이를 뒷받침하는 구체 기능 |
| --- | --- |
| 선언적·버전 관리 | Application CRD, 다중 구성 관리 도구 지원(Kustomize·Helm·Jsonnet·평문 YAML), 브랜치·태그·특정 커밋으로의 추적, Git에 커밋된 임의 구성으로의 롤백(roll-anywhere) |
| 자동화 | 지정 대상 환경으로의 자동 배포, 자동/수동 동기화, 자동 구성 드리프트 감지 |
| 감사 가능 | 애플리케이션 이벤트·API 호출에 대한 감사 추적(audit trail), 멀티테넌시와 RBAC 정책, 자동화용 액세스 토큰 |
| 이해하기 쉬움 | 애플리케이션 활동을 실시간으로 보여 주는 Web UI, 드리프트 시각화, 리소스 헬스 상태 분석 |

이 표에서 특히 눈여겨볼 것은 마지막 두 행입니다. 순수한 GitOps 조정 루프만으로도 "선언적·자동화"는 달성할 수 있습니다. 그러나 변경 이력을 사람이 검토 가능한 형태로 남기는 감사 추적, 그리고 무엇이 왜 어긋났는지를 한눈에 보여 주는 드리프트 시각화와 헬스 분석은 원칙이 요구하는 바를 넘어선 제품 수준의 부가가치입니다. Argo CD가 조정 엔진에 더해 gRPC/REST API Server와 Web UI를 갖춘 이유가 여기에 있습니다.

### 원칙 위에 더해지는 플랫폼 기능

Argo CD를 채택하는 실질적 동기는, GitOps의 최소 요건을 넘어서는 운영 기능들입니다. 다음은 손수 만든 스크립트로는 확보하기 어려운, Argo CD가 기본 제공하는 대표적 능력입니다.

- **다중 클러스터 관리**: 여러 클러스터로의 배포와 관리를 수행합니다.
- **인증과 권한 위임**: OIDC, OAuth2, LDAP, SAML 2.0, GitHub, GitLab, Microsoft, LinkedIn 등 SSO를 통합하고, 멀티테넌시와 RBAC로 조직 단위 권한을 시행합니다.
- **정교한 배포 전략 지원**: PreSync·Sync·PostSync 훅을 통해 블루/그린이나 카나리아 같은 복잡한 롤아웃을 구성할 수 있습니다.
- **관측 가능성과 자동화 접점**: Prometheus 메트릭을 노출하고, CI 통합용 CLI와 웹훅(GitHub·BitBucket·GitLab) 연동을 제공합니다.

이 목록은 Argo CD가 단일 팀의 조정 도구를 넘어, 플랫폼 팀이 여러 개발 팀에게 서비스하는 멀티테넌트 배포 플랫폼으로 설계되었음을 보여 줍니다. 실제로 Argo CD는 이 목적에 맞춘 멀티테넌트 설치와, 더 최소한의 Core(헤드리스) 설치를 모두 제공하는데, 이 선택지 자체가 "왜 Argo CD인가"의 한 근거입니다. 설치 방식의 구체적 비교는 이 가이드의 뒤에서 별도로 다룹니다.

### 도구 지형 속 Argo CD의 위치

Kubernetes가 선언적 성격을 지니면서, 기존의 Infrastructure as Code 도구만으로는 부족한 지점을 메우기 위해 Kubernetes의 선언적 성격을 활용하는 관리 도구들이 등장했습니다. 그 대표적인 두 축이 Argo CD와 Flux입니다. GitOps는 벤더 중립적으로 정립된 OpenGitOps 원칙 위에 세워진 접근이므로, 조정 루프라는 근본 사고 모델은 공유됩니다. 따라서 실질적 선택은 원칙의 차이가 아니라 제품 형태의 차이에서 갈립니다.

Argo CD 쪽의 문서화된 차별점은 앞서 정리한 기능 표에 그대로 드러납니다. 애플리케이션 활동을 실시간으로 보여 주는 Web UI, 드리프트 시각화, 멀티테넌시와 RBAC를 갖춘 API Server 중심 구조는 Argo CD가 UI 주도의, 여러 팀이 공유하는 플랫폼을 지향한다는 신호입니다. 반대로 UI·SSO·멀티클러스터 기능이 필요 없는 클러스터 관리자라면 Core 모드처럼 컴포넌트를 최소화한 경량 설치를 택할 수 있습니다. 즉 "UI와 다중 테넌트 거버넌스를 통합 관리하고자 하는가"가 Argo CD 선택의 핵심 판단 기준이 됩니다.

거버넌스와 지속성 측면에서도 근거가 있습니다. Argo CD는 커뮤니티에 의해 활발히 개발되고 있으며, 공식적으로 도입한 조직 목록이 지속적으로 늘고 있습니다. Argo CD가 Argo Project의 산물이라는 점, 그리고 그 릴리스가 GitHub에서 공개적으로 관리된다는 점은, 장기적으로 의존할 도구를 고를 때의 안정성 지표가 됩니다.

마지막으로 GitOps와 DevOps의 관계를 짚어 둘 필요가 있습니다. GitOps는 DevOps를 대체하는 것이 아니라, DevOps 팀이 인프라 자동화 전략의 한 축으로 채택하는 운영 프레임워크입니다. Argo CD는 이 프레임워크를 Kubernetes 위에서 구체화한 도구로서, 개발 팀이 이미 익숙한 Git 워크플로우와 CLI·CI 통합을 그대로 활용하도록 함으로써 기존 DevOps 관행과 자연스럽게 맞물립니다.

## 설치 방식 선택하기

Argo CD의 설치는 단일한 절차가 아니라 몇 갈래의 갈림길로 이루어져 있습니다. 공식 문서는 설치를 크게 두 유형으로 구분합니다. 여러 팀에게 서비스하는 것을 전제로 한 **멀티테넌트(multi-tenant)** 설치와, 컴포넌트를 최소화한 **Core** 설치입니다. 그리고 멀티테넌트 안에서 다시 고가용성(High Availability, HA) 여부에 따라 매니페스트가 갈리며, 클러스터 전체 권한이 필요한지 네임스페이스 권한만으로 충분한지에 따라 또 한 번 나뉩니다. 어떤 조합을 고르느냐는 앞서 아키텍처에서 다룬 컴포넌트 구성과 직결되므로, 각 선택지가 무엇을 포함하고 무엇을 전제하는지 정확히 짚어 두는 것이 이 단계의 핵심입니다.

### 멀티테넌트 설치: 가장 일반적인 방식

멀티테넌트 설치는 Argo CD를 설치하는 가장 흔한 방식입니다. 이 유형은 조직 내 여러 애플리케이션 개발 팀에게 서비스를 제공하기 위해 사용되며, 보통 플랫폼 팀이 유지·관리합니다. 최종 사용자는 API Server를 통해 Web UI나 `argocd` CLI로 Argo CD에 접근하는데, CLI는 `argocd login <server-host>` 명령으로 구성해야 합니다. 즉 이 방식은 앞서 다룬 API Server 중심 구조, 다시 말해 인증·RBAC·SSO 위임과 UI를 전부 갖춘 완전한 형태입니다.

멀티테넌트 설치에는 다시 고가용성 여부라는 축이 존재합니다.

**비고가용성(Non-HA)** 매니페스트는 프로덕션 사용에는 권장되지 않으며, 평가 기간의 데모나 테스트에 주로 쓰입니다. 여기에는 두 가지 매니페스트가 제공됩니다.

- **`install.yaml`** — 클러스터 관리자(cluster-admin) 권한을 갖는 표준 설치입니다. Argo CD가 실행되는 것과 같은 클러스터에 애플리케이션을 배포할 계획이라면 이 매니페스트를 사용합니다. 물론 자격 증명을 입력하면 외부 클러스터에도 배포할 수 있습니다.
- **`namespace-install.yaml`** — 네임스페이스 수준의 권한만 요구하며 클러스터 롤(cluster role)을 필요로 하지 않는 설치입니다. Argo CD가 자신이 실행되는 클러스터에 배포할 필요가 없고 오로지 입력된 외부 클러스터 자격 증명에만 의존할 경우에 적합합니다. 예를 들어 팀별로 여러 Argo CD 인스턴스를 운영하면서 각 인스턴스가 외부 클러스터에 배포하도록 하는 구성이 이에 해당합니다. 다만 이 매니페스트에는 Argo CD CRD가 포함되어 있지 않아 별도로 설치해야 합니다.

CRD를 별도로 설치하려면 다음 명령을 사용합니다.

```bash
kubectl apply --server-side --force-conflicts \
  -k https://github.com/argoproj/argo-cd/manifests/crds\?ref\=stable
```

**고가용성(HA)** 매니페스트는 프로덕션 사용에 권장되는 방식으로, 동일한 컴포넌트를 포함하되 고가용성과 복원력을 위해 튜닝된 번들입니다. 지원되는 컴포넌트를 다중 복제본으로 실행한다는 점이 비고가용성 버전과의 결정적 차이입니다. 여기에도 클러스터 권한 여부에 대응하는 두 매니페스트가 있습니다. `ha/install.yaml`은 `install.yaml`과 같되 복제본이 다중화된 형태이며, `ha/namespace-install.yaml`은 `namespace-install.yaml`의 다중 복제본 버전입니다.

### Core 설치: 헤드리스 모드

Core 설치는 Argo CD를 헤드리스(headless) 모드로 배포하기 위한, 성격이 다른 설치입니다. 이 방식으로도 Git 저장소에서 원하는 상태를 가져와 Kubernetes에 적용하는, 완전히 동작하는 GitOps 엔진을 얻을 수 있습니다. 멀티테넌시 기능이 필요 없이 독립적으로 Argo CD를 사용하는 클러스터 관리자에게 가장 적합하며, 더 적은 컴포넌트를 포함하기에 설치가 더 쉽습니다. 핵심적인 차이는 이 번들에 **API Server와 UI가 포함되지 않고**, 각 컴포넌트의 경량(비고가용성) 버전을 설치한다는 점입니다. 설치 매니페스트는 `core-install.yaml`로 제공됩니다. Core 모드의 아키텍처와 사용법은 이어지는 **Argo CD Core (Headless 모드)** 장에서 자세히 다룹니다.

### 설치 방식 비교

지금까지의 갈래를 하나의 표로 정리하면, 어떤 상황에서 어떤 매니페스트를 골라야 하는지가 선명해집니다.

| 매니페스트 | 유형 | 권한 범위 | 자체 클러스터 배포 | API Server·UI | 주 용도 |
| --- | --- | --- | --- | --- | --- |
| `install.yaml` | 멀티테넌트 · Non-HA | cluster-admin | 가능 | 포함 | 평가·데모·테스트 |
| `namespace-install.yaml` | 멀티테넌트 · Non-HA | 네임스페이스 수준 | 기본적으로 외부 클러스터 대상(입력 자격 증명 시 자체 클러스터도 가능) | 포함 | 팀별 인스턴스, 외부 클러스터 배포 |
| `ha/install.yaml` | 멀티테넌트 · HA | cluster-admin | 가능 | 포함(다중 복제본) | 프로덕션 |
| `ha/namespace-install.yaml` | 멀티테넌트 · HA | 네임스페이스 수준 | 외부 클러스터 대상 | 포함(다중 복제본) | 프로덕션, 팀별 인스턴스 |
| `core-install.yaml` | Core(헤드리스) | — | 가능 | **미포함** | 단일 관리자, Kubernetes RBAC만 사용 |

`namespace-install.yaml` 계열은 CRD를 별도로 설치해야 한다는 점을 다시 상기할 필요가 있습니다. 또한 이 매니페스트에 포함된 기본 롤로는 같은 클러스터에 Argo CD 리소스(Application, ApplicationSet, AppProject)만 배포할 수 있고, 실제 배포는 외부 클러스터로 이루어지는 GitOps 모드만 지원됩니다. 이를 바꾸려면 새 롤을 정의해 `argocd-application-controller` 서비스 어카운트에 바인딩해야 합니다.

### 어떤 방식을 고를 것인가

선택의 첫 갈림길은 "UI·SSO·멀티클러스터 기능이 필요한가"입니다. 공식 안내에서도 이 기능들이 필요 없다면 Core 컴포넌트만 설치할 수 있다고 명시합니다. 클러스터 관리자로서 Kubernetes RBAC에만 의존하고 싶거나, 새로운 API·CLI를 도입하지 않고 Kubernetes API만으로 배포를 자동화하려는 경우, 또는 개발자에게 Argo CD UI나 CLI를 제공하고 싶지 않은 경우가 Core를 정당화하는 대표적인 상황입니다.

멀티테넌트를 택했다면 두 번째 갈림길은 고가용성 여부입니다. 프로덕션 환경이라면 다중 복제본으로 튜닝된 HA 번들이 권장되고, 데모나 평가 목적이라면 비고가용성 번들로 충분합니다.

세 번째 갈림길은 권한 범위입니다. Argo CD가 자신이 실행되는 클러스터에도 배포해야 한다면 cluster-admin 권한을 갖는 `install.yaml` 계열이, 오직 외부 클러스터에만 배포하며 최소 권한을 지향한다면 네임스페이스 수준 권한만 요구하는 `namespace-install.yaml` 계열이 적합합니다.

한편 어떤 매니페스트를 고르든 설치 대상 Kubernetes 버전이 해당 Argo CD 버전과 호환되는지 확인해야 합니다. 공식 문서는 각 Argo CD 버전이 테스트된 Kubernetes 버전을 다음과 같이 제시합니다.

| Argo CD 버전 | 테스트된 Kubernetes 버전 |
| --- | --- |
| 3.4 | v1.35, v1.34, v1.33, v1.32 |
| 3.3 | v1.35, v1.34, v1.33, v1.32 |
| 3.2 | v1.34, v1.33, v1.32, v1.31 |

이렇게 유형(멀티테넌트/Core), 가용성(HA/Non-HA), 권한 범위라는 세 축으로 조합이 정해지고 나면, 실제로 이 매니페스트들을 어떻게 적용하는지가 남습니다. 원격 리소스로 포함해 Kustomize 패치로 커스터마이즈하거나 커뮤니티가 관리하는 Helm 차트를 사용하는 구체적 절차는 **설치 방법: Kustomize와 Helm 활용** 장에서, 실제 첫 설치와 접근 구성은 **첫 설치와 클러스터 접근 설정** 장에서 다룹니다.
&lt;/server-host&gt;

## Argo CD Core (Headless 모드)

Core 설치가 `core-install.yaml` 하나로 API Server와 UI를 뺀 경량 번들을 배포한다는 점은 앞서 설치 방식을 비교하며 확인했습니다. 실제로 이 번들에는 API Server나 UI가 포함되지 않고, 각 컴포넌트의 경량(non-HA) 버전이 설치됩니다. 여기서 파고들 질문은 그 다음입니다. API Server를 통째로 들어냈을 때 **정확히 어떤 기능이 사라지고, 어떤 기능이 반쪽만 남으며, 사라진 UI와 CLI 접근이 헤드리스 모드에서는 어떤 원리로 되살아나는가**입니다.

### 무엇이 빠지고, 무엇이 반쪽으로 남는가

Core 설치에서 완전히 사라지는 기능과 부분적으로만 동작하는 기능은 명확히 구분되어 있습니다. API Server가 담당하던 책임(인증·RBAC 시행·외부 신원 공급자 위임·API 노출)이 없어지므로, 그 위에 얹혀 있던 기능들이 함께 제거됩니다.

| 구분 | 대상 | Core에서의 상태 |
| --- | --- | --- |
| 완전히 사라짐 | Argo CD RBAC 모델 | 사용 불가 |
| 완전히 사라짐 | Argo CD API | 사용 불가 |
| 완전히 사라짐 | Notification Controller(알림) | 사용 불가 |
| 완전히 사라짐 | OIDC 기반 인증 | 사용 불가 |
| 부분적으로 동작 | Argo CD Web UI | 로컬에서만 실행(아래 참조) |
| 부분적으로 동작 | Argo CD CLI | 로컬 API Server를 띄워 동작 |
| 부분적으로 동작 | 멀티테넌시 | Git push 권한에 기반한 순수 GitOps 방식만 |

"완전히 사라짐"에 속한 네 항목이 모두 API Server의 책임과 맞닿아 있다는 사실에 주목할 만합니다. 인증·RBAC 시행·외부 신원 공급자 위임은 앞서 아키텍처에서 API Server의 몫으로 정리되었고, 그 컴포넌트가 없으므로 이 기능들도 존재할 자리가 없습니다. 그 결과 Core 모드의 멀티테넌시는 Argo CD 자체의 권한 모델이 아니라 오로지 Git 저장소에 대한 push 권한으로만 구획됩니다. 즉 누가 무엇을 배포할 수 있는지를 통제하는 지점이 Argo CD가 아니라 Git 쪽으로 옮겨 갑니다.

한편 Core 모드에서 사용자가 다룰 수 있는 Kubernetes 리소스는 `Application`과 `ApplicationSet` CRD입니다. 이 CRD들을 통해 사용자는 Kubernetes에 애플리케이션을 배포·관리할 수 있습니다. GitOps 엔진으로서의 기능 자체는 그대로 살아 있고, 그 위에 있던 접근 제어·API 계층만 벗겨진 셈입니다.

### 헤드리스 CLI는 어떻게 동작하는가

여기서 초보자가 가장 의아해하는 지점이 등장합니다. API Server를 설치하지 않았는데 어떻게 `argocd` CLI를 쓸 수 있는가? 답은 CLI가 **로컬 API Server 프로세스를 스스로 띄운다**는 데 있습니다.

Core 모드에서 CLI 명령을 실행하면, CLI는 로컬에 Argo CD API Server 프로세스를 생성해 그 프로세스로 명령을 처리합니다. 명령이 끝나면 이 로컬 API Server 프로세스도 함께 종료됩니다. 이 과정은 추가 명령 없이 사용자에게 투명하게 일어납니다. 즉 클러스터 안에 상주하는 API Server 대신, 명령을 실행하는 그 순간에만 로컬에서 잠깐 살아 있는 API Server가 그 역할을 대신하는 구조입니다.

이 흐름을 그려 보면, 멀티테넌트 설치에서 원격 API Server에 로그인해 통신하던 것과 근본적으로 다른 경로임이 드러납니다.

```text
[멀티테넌트 모드]
argocd CLI ──(gRPC/REST, 원격)──▶ 클러스터 내 상주 API Server ──▶ Repo/Controller

[Core(헤드리스) 모드]
argocd CLI
   │  (1) argocd login --core  → 로컬 API Server 프로세스 spawn
   ▼
로컬 임시 API Server (사용자 머신에서 실행)
   │  (2) Kubernetes 자격 증명으로 Kubernetes API에 접근
   ▼
Kubernetes API ──▶ Application / ApplicationSet CRD 조작
   │
   ▼
   (3) 명령 종료 시 로컬 API Server 프로세스도 종료
```

이 로컬 API Server를 띄우는 스위치가 바로 `login` 하위 명령에 붙이는 `--core` 플래그입니다. `--core` 플래그는 CLI와 Web UI 요청을 처리할 로컬 Argo CD API Server 프로세스를 생성하는 역할을 맡습니다. Core 모드에서 CLI를 쓰려면 다음처럼 현재 kube 컨텍스트를 Argo CD 네임스페이스로 바꾼 뒤 `--core`로 로그인합니다.

```bash
# 현재 kube 컨텍스트를 argocd 네임스페이스로 변경
kubectl config set-context --current --namespace=argocd

# 헤드리스(코어) 모드로 CLI 로그인
argocd login --core
```

멀티테넌트 설치에서 `argocd login <server-host>`로 원격 서버 주소를 지정하던 것과 대비됩니다. `--core`에는 서버 호스트가 필요 없으며, 대신 Argo CD 네임스페이스에 접근할 수 있는 Kubernetes 자격 증명에 의존합니다.

### 로컬에서 Web UI 띄우기

UI가 번들에서 빠졌더라도, 필요할 때 로컬에서 Web UI를 실행할 수 있습니다. 이 역시 `--core`가 띄우는 로컬 API Server 프로세스가 CLI 요청뿐 아니라 Web UI 요청도 함께 처리하기 때문에 가능합니다. 다음 명령으로 로컬 대시보드를 실행합니다.

```bash
argocd admin dashboard -n argocd
```

이후 Web UI는 `http://localhost:8080`에서 접근할 수 있습니다. 이 대시보드는 사용자의 머신에서 로컬로 구동되는 것이므로, 클러스터 밖으로 UI를 노출하거나 Ingress·LoadBalancer를 구성하는 멀티테넌트 방식과는 성격이 다릅니다. 여러 팀에게 상시 제공되는 공용 UI가 아니라, 관리자가 필요할 때 자기 머신에서 띄우는 진단 창에 가깝습니다.

### 접근 제어는 전적으로 Kubernetes RBAC로

Core 모드의 접근 제어를 이해하는 핵심은, Argo CD 고유의 RBAC 모델이 사라진 자리를 Kubernetes RBAC가 온전히 메운다는 점입니다. Core 모드는 오직 Kubernetes RBAC에만 의존하며, CLI를 실행하는 사용자(또는 프로세스)는 Argo CD 네임스페이스에 접근할 수 있어야 하고, 실행하려는 명령에 맞게 `Application`·`ApplicationSet` 리소스에 대한 적절한 권한을 가지고 있어야 합니다.

이 사실은 실무에서 곧바로 문제 진단의 출발점이 됩니다. `argocd login --core` 이후의 명령이 권한 오류로 실패한다면, 원격 서버 연결이나 Argo CD 계정 문제가 아니라 **명령을 실행하는 주체의 Kubernetes RBAC 권한**을 먼저 살펴야 합니다. Core 모드는 Kubernetes RBAC에만 의존하기 때문에, 그 주체가 대상 리소스에 대한 권한을 갖지 못하면 명령도 실패합니다.

예를 들어 특정 사용자에게 Argo CD 네임스페이스에서 `Application`과 `ApplicationSet`을 다룰 권한을 부여하려면 다음과 같은 Role과 RoleBinding을 정의할 수 있습니다.

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: argocd
  name: argocd-core-app-manager
rules:
  - apiGroups: ["argoproj.io"]
    resources: ["applications", "applicationsets"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  namespace: argocd
  name: argocd-core-app-manager-binding
subjects:
  - kind: User
    name: dev-user            # 인증되는 주체
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: argocd-core-app-manager
  apiGroup: rbac.authorization.k8s.io
```

여기서 `verbs`에 어떤 동작을 포함하느냐가 곧 그 사용자가 CLI로 수행할 수 있는 명령의 범위를 결정합니다. 조회만 허용하려면 `get`·`list`·`watch`만 남기고, 애플리케이션 생성·삭제까지 허용하려면 `create`·`delete` 등을 더하는 식입니다. Argo CD 자체 RBAC 정책 파일을 편집하는 대신 이렇게 표준 Kubernetes RBAC로 권한을 조율한다는 점이 Core 모드의 가장 뚜렷한 운영상 특징입니다.

이처럼 Core 모드에서 겪는 접근 관련 문제는 대부분 세 갈래로 좁혀집니다. 첫째, kube 컨텍스트의 네임스페이스가 `argocd`로 설정되어 있는지(`kubectl config set-context --current --namespace=argocd`), 둘째, `login` 시 `--core` 플래그를 실제로 붙였는지, 셋째, 명령을 실행하는 주체가 `Application`·`ApplicationSet`에 대한 RBAC 권한을 갖는지입니다. 이 세 가지가 갖춰지면, API Server가 없는 최소 구성으로도 CLI와 로컬 UI를 통한 GitOps 운영이 온전히 이루어집니다.
&lt;/server-host&gt;

## 설치 방법: Kustomize와 Helm 활용

원격 매니페스트를 클러스터에 그대로 적용하는 것만으로도 Argo CD는 동작하지만, 실무에서는 네임스페이스를 바꾸거나 특정 필드를 조정해야 하는 경우가 잦습니다. 공식 문서는 이런 커스터마이즈를 위해 두 가지 경로를 제시합니다. 하나는 매니페스트를 원격 리소스로 포함해 패치를 얹는 Kustomize 방식이고, 다른 하나는 커뮤니티가 관리하는 Helm 차트를 파라미터화해 설치하는 방식입니다.

### Kustomize로 설치하기

Kustomize를 쓸 때 권장되는 형태는 공식 매니페스트를 **원격 리소스로 포함**한 뒤 Kustomize 패치로 추가 커스터마이즈를 적용하는 것입니다. 가장 단순한 `kustomization.yaml`은 다음과 같습니다.

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: argocd
resources:
  - https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

여기서 `resources`에 넣은 URL은 앞선 설치 방식 논의에서 살펴본 매니페스트 계열(`install.yaml`, `ha/install.yaml`, `core-install.yaml` 등) 중 목적에 맞는 것으로 바꿔 지정하면 됩니다. `namespace` 필드는 이 Kustomization이 렌더링하는 모든 리소스를 지정한 네임스페이스에 배치하도록 강제합니다. 이렇게 작성한 디렉터리는 다음처럼 렌더링해 확인하거나 클러스터에 바로 적용할 수 있습니다.

```bash
# 렌더링 결과만 확인
kustomize build .

# 클러스터에 직접 적용 (CRD 크기 제한·필드 소유권 문제 회피)
kubectl apply -k . --server-side --force-conflicts
```

여기서 `--server-side --force-conflicts` 두 플래그가 왜 필요한지는 짚어 둘 가치가 있습니다. `--server-side` 플래그는 ApplicationSet 같은 일부 Argo CD CRD가 클라이언트 사이드 `kubectl apply`가 부과하는 262KB 어노테이션 크기 제한을 초과하기 때문에 요구됩니다. 서버 사이드 적용(server-side apply)은 `last-applied-configuration` 어노테이션을 저장하지 않으므로 이 한계를 우회합니다. `--force-conflicts` 플래그는 이전에 다른 도구(예: Helm이나 과거의 `kubectl apply`)가 관리하던 필드의 소유권을 이번 적용이 넘겨받도록 허용합니다. 새 설치에서는 안전하고 업그레이드에서는 필수적인데, 다만 Argo CD 매니페스트에 정의된 필드(`affinity`, `env`, `probes` 등)에 가한 사용자 지정 수정은 이 과정에서 덮어써진다는 점을 기억해야 합니다. 반대로 매니페스트에 명시되지 않은 필드(예: `resources` 리밋/리퀘스트나 `tolerations`)는 보존됩니다.

한 가지 주의할 함정은 네임스페이스입니다. 공식 설치 매니페스트에는 `argocd` 네임스페이스를 참조하는 `ClusterRoleBinding` 리소스가 포함되어 있어, 이를 다른 네임스페이스에 설치하면서 네임스페이스 참조를 갱신하지 않으면 권한 관련 오류가 발생할 수 있습니다. 사용자 지정 네임스페이스에 설치할 때는 이 바인딩을 명시적으로 패치해야 합니다.

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

이 패치는 `ClusterRoleBinding`이 사용자 지정 네임스페이스의 ServiceAccount에 올바르게 매핑되도록 보장하여, 배포 중 권한 관련 문제를 방지합니다. `namespace-install.yaml` 계열을 원격 리소스로 쓸 때는 여기에 더해 CRD를 별도로 설치해야 하는데, 그 명령은 앞서 설치 방식을 다루며 소개한 CRD 설치 명령을 그대로 사용하면 됩니다. 만약 그 적용이 실패한다면, `--server-side --force-conflicts`가 함께 붙었는지부터 확인하는 것이 진단의 출발점이 됩니다.

### Helm으로 설치하기

Argo CD는 Helm으로도 설치할 수 있습니다. 다만 이 Helm 차트는 현재 **커뮤니티가 관리**하며 `argo-helm/charts/argo-cd` 저장소에서 제공된다는 점을 분명히 알아 둘 필요가 있습니다. 공식 매니페스트와 별개로 유지되므로, 파라미터 이름과 기본값은 반드시 해당 차트의 문서를 기준으로 확인해야 합니다.

Helm 설치의 일반적인 흐름은 차트 저장소를 등록하고, 네임스페이스를 생성하며 설치하는 것입니다. Helm은 설치 시 값을 주입할 수 있으므로, 환경마다 달라지는 부분만 오버라이드해 같은 차트를 "도장 찍듯" 재사용할 수 있습니다.

```bash
# 커뮤니티 차트 저장소 등록 (저장소 URL은 argo-helm 차트 문서에서 확인)
helm repo add <repo-name> <chart-repository-url>
helm repo update

# argocd 네임스페이스를 생성하며 설치
helm install argocd <repo-name>/argo-cd \
  --namespace argocd --create-namespace
```

`--create-namespace`는 지정한 `--namespace`가 없을 때 함께 생성하도록 하는 플래그이고, `install`의 첫 인자(`argocd`)는 릴리스 이름입니다. 설치 후에도 Helm 프레임워크를 통해 이 릴리스를 버전 관리·업그레이드·수정할 수 있습니다.

파라미터를 조정하는 방법은 두 가지입니다. 값이 몇 개뿐이라면 명령줄에서 `--set`으로 즉석 오버라이드하고, 값이 많거나 형상 관리 대상이라면 `values` 파일로 분리해 넘깁니다.

```bash
# 개별 값을 즉석에서 오버라이드
helm install argocd <repo-name>/argo-cd \
  --namespace argocd --create-namespace \
  --set <parameter>=<value>

# 오버라이드 값을 파일로 분리해 관리
helm install argocd <repo-name>/argo-cd \
  --namespace argocd --create-namespace \
  -f values-prod.yaml
```

여기서 `<parameter>`나 `values-prod.yaml`에 들어갈 구체적인 키 이름은 커뮤니티 차트가 노출하는 값에 따라 달라지므로, 그대로 옮겨 쓰기보다는 차트가 제공하는 기본 `values.yaml`을 열람해 확인한 뒤 필요한 항목만 덮어쓰는 것이 안전합니다. GitOps 관점에서 보면, 각 배포마다 달라지는 부분을 별도의 `values` 파일로 저장해 두는 것이 곧 환경 간 차이를 최소한의 형태로 버전 관리하는 방식이 됩니다.

### Kustomize와 Helm 중 무엇을 쓸 것인가

두 방식은 배타적 선택지라기보다 서로 다른 상황에 맞는 도구입니다. 개념적 차이를 정리하면 다음과 같습니다.

| 관점 | Kustomize | Helm |
| --- | --- | --- |
| 매니페스트 조작 방식 | 원본 매니페스트를 유지한 채 패치(오버레이)로 변경분만 얹음 | 템플릿에 값을 주입해 렌더링된 릴리스를 생성 |
| Argo CD 제공 형태 | 공식 매니페스트를 원격 리소스로 포함(권장) | 커뮤니티가 관리하는 `argo-helm/charts/argo-cd` 차트 |
| 값을 미리 알아야 하는가 | 패치할 값을 사전에 알고 있을 때 적합 | 클러스터에 적용되기 전까지 값을 모를 때 파라미터화에 유리 |
| 커스터마이즈 단위 | 리소스 필드 단위 패치(예: `ClusterRoleBinding` subject) | 차트가 노출한 파라미터 단위 |
| 적용 명령 | `kubectl apply -k` (서버 사이드 적용 권장) | `helm install` / `helm upgrade` |

원본이 순수 Kubernetes 매니페스트이고 특정 필드만 바꾸면 되는 상황에서는 Kubernetes와 여러 GitOps 도구에 내장된 Kustomize가 자연스럽고, 배포 시점에야 값이 정해지는 파라미터(예: Ingress의 호스트명 같은)를 다루거나 서드파티 스택을 값으로 조립해야 할 때는 Helm의 파라미터화가 더 잘 맞습니다. 실제로는 두 도구를 함께 사용하는 경우가 많으며, 각 도구의 템플릿·오버레이 전략과 선택 기준은 이 가이드의 뒤쪽에서 더 깊이 다룹니다.

어느 방식을 택하든, 매니페스트를 실제로 클러스터에 올린 뒤에는 Argo CD 서버에 접근해 초기 관리자 비밀번호를 확인하고 CLI나 UI로 로그인하는 절차가 이어집니다. 그 실제 첫 설치와 클러스터 접근 구성은 바로 다음에서 살펴봅니다.
&lt;/parameter&gt;&lt;/repo-name&gt;&lt;/value&gt;&lt;/parameter&gt;&lt;/repo-name&gt;&lt;/repo-name&gt;&lt;/chart-repository-url&gt;&lt;/repo-name&gt;&lt;/your-custom-namespace&gt;&lt;/your-custom-namespace&gt;

## 첫 설치와 클러스터 접근 설정

매니페스트를 렌더링하고 커스터마이즈하는 방법은 앞서 정리했으므로, 이제 실제로 클러스터에 Argo CD를 올리고 그 서버에 접근 가능한 상태로 만드는 절차를 다룹니다. 이 단계에서 초보자가 가장 자주 막히는 지점은 설치 자체가 아니라 그 다음입니다. 설치를 마쳐도 Argo CD는 기본적으로 클러스터 외부로 노출되지 않으며, 게다가 자체 서명 인증서(self-signed certificate)를 사용하기 때문에 별도의 조치 없이는 브라우저나 CLI로 곧바로 접근할 수 없습니다. 따라서 이 절의 핵심은 "설치 → 접근 경로 확보"라는 두 걸음을 정확히 밟는 데 있습니다.

### 사전 요건 확인

설치를 시작하기 전에 로컬 환경이 몇 가지 조건을 갖추고 있어야 합니다. 공식 안내가 명시하는 요건은 다음과 같습니다.

- `kubectl` 커맨드라인 도구가 설치되어 있을 것
- 클러스터에 접근할 수 있는 kubeconfig 파일이 있을 것(기본 위치는 `~/.kube/config`)
- 클러스터에 CoreDNS가 활성화되어 있을 것

CoreDNS는 microk8s 환경이라면 `microk8s enable dns && microk8s stop && microk8s start`로 활성화할 수 있습니다. 또한 설치 대상 Kubernetes 버전이 사용할 Argo CD 버전과 호환되는지는 앞서 다룬 테스트된 버전 표를 기준으로 미리 확인해 두어야 합니다.

### Argo CD 설치

가장 기본이 되는 설치는 `argocd` 네임스페이스를 만들고 stable 브랜치의 공식 매니페스트를 적용하는 두 명령으로 끝납니다.

```bash
kubectl create namespace argocd
kubectl apply -n argocd --server-side --force-conflicts \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

이 명령은 새 `argocd` 네임스페이스를 생성하고, 그 안에 모든 Argo CD 서비스와 애플리케이션 리소스가 자리하도록 stable 브랜치의 공식 매니페스트를 적용합니다. 프로덕션에서는 `stable` 대신 `v3.2.0` 같은 고정 버전(pinned version)을 사용하는 것이 권장됩니다. 여기 붙은 `--server-side --force-conflicts` 두 플래그가 왜 필요한지는 앞서 설치 방법을 다루며 상세히 설명했으므로, 요점만 상기하면 일부 CRD(예: ApplicationSet)가 클라이언트 사이드 `kubectl apply`의 262KB 어노테이션 크기 제한을 넘기기 때문에 서버 사이드 적용이 요구된다는 점입니다.

한편 이 기본 설치에는 Redis에 대한 한 가지 세부 사항이 딸려 있습니다. 기본 설치의 Redis는 패스워드 인증을 사용하며, 그 패스워드는 Argo CD가 설치된 네임스페이스의 `argocd-redis`라는 Kubernetes 시크릿에 `auth` 키로 저장됩니다. 통상적인 사용에서 직접 만질 일은 없지만, 캐시 계층의 자격 증명이 어디에 있는지 알아 두면 문제 진단 시 참조점이 됩니다.

UI·SSO·멀티클러스터 기능이 필요 없다면 여기서 방향을 틀어 Core 컴포넌트만 설치할 수 있습니다. 그 경우의 설치와 접근 방식은 앞서 다룬 Argo CD Core(Headless 모드) 설명대로 `argocd login --core`로 서버 노출·로그인 절차(3~5단계) 자체를 건너뛰게 됩니다. 이 절의 나머지 내용은 API Server와 UI를 갖춘 멀티테넌트 설치를 전제로 합니다.

### CLI 내려받기

설치가 끝나면 서버와 통신할 `argocd` CLI를 로컬에 준비합니다. 최신 버전은 릴리스 페이지(`https://github.com/argoproj/argo-cd/releases/latest`)에서 내려받을 수 있으며, macOS·Linux·WSL에서는 Homebrew로 설치하는 것이 간편합니다.

```bash
brew install argocd
```

CLI는 이후 로그인·애플리케이션 생성·동기화 등 대부분의 조작에 쓰이므로, 서버 접근 경로를 확보하기 전에 미리 갖춰 두는 편이 좋습니다.

### 왜 접근 경로를 따로 열어야 하는가

기본 상태에서 Argo CD는 클러스터 외부로 노출되지 않습니다. 즉 `argocd-server`라는 서비스는 존재하지만, 브라우저나 로컬 CLI가 그 서비스에 도달할 통로가 아직 없습니다. 브라우저나 CLI로 접근하려면 다음 세 가지 방법 중 하나로 통로를 열어야 합니다. 각 방법은 노출 범위와 영속성이 다르므로, 목적에 맞게 선택하는 것이 중요합니다.

| 방법 | 노출 범위 | 성격 | 대표 용도 |
| --- | --- | --- | --- |
| Service Type LoadBalancer | 클러스터 외부(외부 IP) | 클라우드 제공자가 외부 IP를 할당하는 상시 노출 | 클라우드 환경의 상시 접근 |
| Ingress | 클러스터 외부(호스트명 기반) | Ingress 컨트롤러를 통한 상시 라우팅 | 도메인·TLS를 갖춘 프로덕션 노출 |
| Port Forwarding | 로컬 머신에 한정 | 명령 실행 동안만 유지되는 임시 터널 | 노출 없이 빠른 로컬 확인·테스트 |

#### Service Type Load Balancer

`argocd-server` 서비스의 타입을 `LoadBalancer`로 바꾸면 클라우드 제공자가 서비스에 외부 IP를 할당합니다.

```bash
kubectl patch svc argocd-server -n argocd \
  -p '{"spec": {"type": "LoadBalancer"}}'
```

잠시 기다린 뒤 할당된 외부 IP는 다음처럼 조회할 수 있습니다.

```bash
kubectl get svc argocd-server -n argocd \
  -o=jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

이 방식은 외부 IP가 배정되어 상시 접근이 가능하다는 장점이 있으나, 로드밸런서를 프로비저닝할 수 있는 클라우드 환경을 전제로 합니다.

#### Ingress

도메인명과 TLS를 갖춘 노출이 필요하다면 Ingress를 구성합니다. 구체적인 설정 방법은 공식 Ingress 문서를 따르며, Ingress 컨트롤러를 통해 호스트명 기반으로 `argocd-server`에 트래픽을 라우팅하게 됩니다. 프로덕션에서 여러 팀에게 상시 UI를 제공하는 경우에 자연스러운 선택입니다.

#### Port Forwarding

서비스를 외부로 노출하지 않고도 API Server에 접속하려면 `kubectl port-forward`를 사용합니다. 이 방법은 로컬 머신에서만 유효한 임시 터널을 열기 때문에, 클러스터를 외부에 노출하지 않고 빠르게 확인하고 싶을 때 적합합니다.

```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

이후 API Server는 `https://localhost:8080`에서 접근할 수 있습니다. 여기서 로컬 포트 `8080`이 서비스의 `443` 포트로 연결된다는 점에 유의해야 합니다. 즉 접속 대상은 HTTPS입니다.

### 자체 서명 인증서 다루기

기본 설치는 자체 서명 인증서를 사용하므로, 별도 조치 없이는 접근이 곧바로 되지 않습니다. 이를 해소하는 방법은 세 갈래 중 하나입니다.

- 정식 인증서를 구성하고, 클라이언트 OS가 그 인증서를 신뢰하도록 설정한다(공식 TLS 설정 문서 참고).
- 클라이언트 OS가 자체 서명 인증서를 신뢰하도록 설정한다.
- 모든 `argocd` CLI 작업에 `--insecure` 플래그를 붙인다.

빠른 평가나 로컬 테스트라면 `--insecure`가 간편하지만, 상시 운영 환경에서는 정식 인증서를 구성하는 편이 바람직합니다.

### 네임스페이스 컨텍스트와 CLI 통신 경로

접근 경로를 열고 나면, CLI가 명령을 올바른 네임스페이스로 보내도록 현재 kube 컨텍스트의 기본 네임스페이스를 `argocd`로 맞춰 두는 것이 편리합니다.

```bash
kubectl config set-context --current --namespace=argocd
```

앞선 설치·패치 명령들에는 이미 `-n argocd`가 붙어 있었으므로 이 설정이 필요 없었지만, 이후 이어지는 로그인·애플리케이션 생성 명령들은 기본 네임스페이스에 의존하므로 이 시점에 한 번 맞춰 두면 반복 입력을 줄일 수 있습니다.

또한 CLI 환경은 Argo CD API Server와 통신할 수 있어야 합니다. 만약 위에서 설명한 방식으로 서버가 직접 접근 가능하지 않다면, 포트 포워딩을 통해 CLI가 서버에 도달하도록 지시할 수 있습니다. 두 가지 방법이 있습니다.

```bash
# 방법 1: 개별 명령마다 플래그를 붙인다
argocd app list --port-forward-namespace argocd

# 방법 2: 환경 변수로 한 번에 설정한다
export ARGOCD_OPTS='--port-forward-namespace argocd'
```

`--port-forward-namespace argocd`는 CLI가 해당 네임스페이스를 통해 서버에 접근하도록 하며, 이를 매번 붙이기 번거롭다면 `ARGOCD_OPTS` 환경 변수에 지정해 두면 됩니다.

여기까지 마치면 Argo CD가 클러스터에 설치되고 브라우저·CLI가 도달할 수 있는 상태가 갖춰집니다. 남은 것은 초기 관리자 자격 증명을 확인해 CLI로 로그인하고 계정을 관리하는 일이며, 이는 **CLI 로그인과 관리자 계정 관리**에서 이어집니다.

## CLI 로그인과 관리자 계정 관리

접근 경로를 확보하고 나면, 브라우저나 CLI가 서버에 도달할 수는 있지만 아직 신원을 증명하지 못한 상태입니다. 여기서 사용할 자격 증명이 바로 초기 관리자(`admin`) 계정이며, 그 비밀번호는 설치 과정에서 자동으로 생성됩니다. 이 절은 초기 비밀번호가 어디에 어떤 형태로 저장되는지, 그것으로 어떻게 로그인하는지, 그리고 로그인 이후 비밀번호를 바꾸고 임시 시크릿을 정리하는 데까지의 흐름을 다룹니다.

### 초기 관리자 비밀번호는 어디에 있는가

`admin` 계정의 초기 비밀번호는 자동 생성되며, Argo CD 설치 네임스페이스에 있는 `argocd-initial-admin-secret`이라는 이름의 시크릿에 저장됩니다. 주목할 점은 이 비밀번호가 해당 시크릿의 `password` 필드에 **평문(clear text)**으로 담긴다는 사실입니다. 즉 별도의 복호화 과정 없이 그대로 읽어 사용할 수 있습니다.

이 값을 직접 꺼내는 가장 간단한 방법은 `argocd` CLI가 제공하는 전용 명령입니다.

```bash
argocd admin initial-password -n argocd
```

이 명령은 지정한 네임스페이스(`-n argocd`)의 `argocd-initial-admin-secret`에서 초기 비밀번호를 조회해 출력합니다.

### CLI로 로그인하기

초기 비밀번호를 확보했다면, 사용자명 `admin`과 함께 Argo CD의 IP 또는 호스트명을 대상으로 로그인합니다. 로그인 명령의 형태는 다음과 같습니다.

```bash
argocd login <argocd_server>
```

여기서 `<argocd_server>`에는 접근 경로를 확보하며 얻은 주소(로드밸런서가 배정한 외부 IP, Ingress 호스트명, 또는 포트 포워딩 시 `localhost:8080`)를 넣습니다. 기본 설치가 자체 서명 인증서를 사용하므로 곧바로 접속이 되지 않는다는 점, 그리고 CLI가 서버에 직접 도달하지 못할 때 포트 포워딩으로 통로를 여는 방법은 접근 경로 설정에서 이미 다뤘으므로 여기서는 그 방식을 그대로 따르면 됩니다.

또한 헤드리스 구성으로 설치했다면 서버 주소 대신 `argocd login --core`로 로그인한다는 점을 상기하면 됩니다. 이 경우의 원리와 접근 제어는 Argo CD Core(Headless 모드)에서 설명한 그대로입니다.

### 비밀번호 변경하기

로그인에 성공했다면 자동 생성된 초기 비밀번호를 계속 쓰기보다는 곧바로 변경하는 것이 바람직합니다. 비밀번호는 다음 명령으로 갱신합니다.

```bash
argocd account update-password
```

이 명령은 현재 로그인된 계정의 비밀번호를 새 값으로 바꿉니다. 초기 비밀번호는 어디까지나 최초 접근을 위한 임시 자격 증명이므로, 이 단계를 거쳐 운영에 사용할 비밀번호로 교체하는 것이 자연스러운 순서입니다.

### 초기 비밀번호 시크릿 정리하기

비밀번호를 변경했다면 `argocd-initial-admin-secret`은 더 이상 유지할 이유가 없습니다. 공식 안내는 비밀번호를 바꾼 뒤 이 시크릿을 Argo CD 네임스페이스에서 삭제할 것을 권고합니다.

```bash
kubectl delete secret argocd-initial-admin-secret -n argocd
```

여기서 초보자가 오해하기 쉬운 지점을 분명히 해 둘 필요가 있습니다. 이 시크릿의 유일한 목적은 최초에 생성된 비밀번호를 평문으로 보관하는 것뿐이며, 그 외의 기능은 없습니다. 따라서 언제든 안전하게 삭제할 수 있습니다. 게다가 만약 이후에 새로운 관리자 비밀번호를 다시 생성해야 하는 상황이 오면, Argo CD가 필요에 따라 이 시크릿을 다시 만들어 냅니다.

### 전체 흐름 요약

지금까지의 네 단계를 하나의 순서로 이으면 다음과 같습니다. 접근 경로를 열어 둔 상태에서 아래를 차례로 실행하면 관리자 계정 초기 설정이 마무리됩니다.

```bash
# 1) 초기 관리자 비밀번호 확인
argocd admin initial-password -n argocd

# 2) admin 계정과 위 비밀번호로 로그인 (<argocd_server>는 확보한 접근 주소)
argocd login <argocd_server>

# 3) 비밀번호를 운영용 값으로 변경
argocd account update-password

# 4) 더 이상 필요 없는 초기 비밀번호 시크릿 삭제
kubectl delete secret argocd-initial-admin-secret -n argocd
```

이 네 단계를 마치면 Argo CD에 인증된 상태로 접근할 수 있게 되며, 이후 애플리케이션을 배포할 대상 클러스터를 등록하거나 첫 Application을 생성하는 작업으로 나아갈 수 있습니다.
&lt;/argocd_server&gt;&lt;/argocd_server&gt;&lt;/argocd_server&gt;&lt;/argocd_server&gt;

## 외부 클러스터 등록하기

지금까지는 Argo CD가 설치된 클러스터 안에서 인증된 상태로 접근하는 데까지 다루었습니다. 그런데 실무에서 자주 마주치는 구성은 "Argo CD는 관리용 클러스터 한 곳에서 돌고, 실제 애플리케이션은 여러 대상 클러스터에 배포한다"는 형태입니다. 이때 필요한 것이 외부 클러스터를 Argo CD에 등록하는 작업입니다. 이 등록 절차는 어디까지나 **외부 클러스터에 배포할 때만 필요한 선택적 단계**라는 점을 먼저 분명히 해 둘 필요가 있습니다.

### 내부 배포와 외부 배포의 갈림길

핵심 개념에서 Application을 만들 때 대상 서버 주소로 `--dest-server https://kubernetes.default.svc`를 사용한다고 언급했는데, 이 주소가 바로 "내부 배포"의 표식입니다. Argo CD가 실행되는 것과 같은 클러스터에 배포하는 경우에는 애플리케이션의 Kubernetes API 서버 주소로 `https://kubernetes.default.svc`를 지정하면 되고, 별도의 클러스터 등록이 필요하지 않습니다.

반면 다른 클러스터에 배포하려면 그 클러스터의 자격 증명을 Argo CD가 알고 있어야 합니다. 앞서 아키텍처에서 짚었듯이 저장소·클러스터 자격 증명은 API Server가 Kubernetes 시크릿으로 저장·관리합니다. 외부 클러스터 등록이란 결국 그 대상 클러스터에 접근할 수 있는 자격 증명을 Argo CD가 사용할 수 있는 형태로 등록해 두는 일입니다.

### 등록 절차

등록의 출발점은 현재 kubeconfig에 어떤 클러스터 컨텍스트들이 있는지 확인하는 것입니다. Argo CD는 로컬 kubeconfig에 정의된 컨텍스트를 기반으로 대상 클러스터를 식별합니다.

```bash
# 현재 kubeconfig에 등록된 모든 클러스터 컨텍스트 이름을 나열
kubectl config get-contexts -o name
```

나열된 컨텍스트 중 배포 대상으로 삼을 이름을 골라 `argocd cluster add`에 넘깁니다. 예를 들어 `docker-desktop` 컨텍스트를 등록한다면 다음과 같습니다.

```bash
argocd cluster add docker-desktop
```

이 한 줄이 실제로는 대상 클러스터 쪽에 여러 리소스를 만들어 냅니다. 위 명령은 해당 kubectl 컨텍스트가 가리키는 클러스터의 `kube-system` 네임스페이스에 `argocd-manager`라는 이름의 ServiceAccount를 설치하고, 그 ServiceAccount를 관리자(admin) 수준의 ClusterRole에 바인딩합니다. Argo CD는 이 ServiceAccount의 토큰을 사용해 배포와 모니터링 같은 관리 작업을 수행합니다. 즉 등록 이후 Argo CD가 그 외부 클러스터를 조작하는 근거가 되는 신원이 바로 `argocd-manager`입니다.

### 등록되는 권한과 최소 권한으로 좁히기

`argocd cluster add`가 기본적으로 부여하는 것은 admin 수준의 ClusterRole이지만, 이 권한 범위는 필요에 따라 좁힐 수 있습니다. `argocd-manager-role` 롤의 규칙(rules)을 수정하여, 제한된 네임스페이스·그룹·종류(kind)의 집합에 대해서만 `create`, `update`, `patch`, `delete` 권한을 갖도록 조정할 수 있습니다.

다만 여기에는 한 가지 넘을 수 없는 조건이 있습니다. Argo CD가 정상 동작하려면 `get`, `list`, `watch` 권한은 **클러스터 범위(cluster-scope)에서** 요구된다는 점입니다. 이는 조정 루프의 성격을 떠올리면 자연스럽습니다. Argo CD는 라이브 상태를 지속적으로 관찰해 대상 상태와 비교해야 하므로, 클러스터 전반의 리소스를 읽고(`get`·`list`) 변화를 감시하는(`watch`) 능력은 필수적이기 때문입니다. 반대로 실제로 리소스를 만들고 바꾸는 쓰기 권한은 대상 범위를 좁혀 최소 권한 원칙에 맞출 수 있습니다.

정리하면 다음과 같습니다.

| 권한 | 조정 루프에서의 역할 | 범위를 좁힐 수 있는가 |
| --- | --- | --- |
| `get`, `list`, `watch` | 라이브 상태 관찰·비교(필수) | 아니오 — 클러스터 범위에서 필요 |
| `create`, `update`, `patch`, `delete` | 교정 조치(동기화) 시 리소스 적용 | 예 — 특정 네임스페이스·그룹·kind로 제한 가능 |

### 네임스페이스 수준 설치와의 연계

설치 방식을 고르며 다룬 `namespace-install.yaml` 계열은 네임스페이스 수준 권한만으로 동작하며, 기본적으로 외부 클러스터에 배포하는 GitOps 모드를 전제로 한다고 언급했습니다. 이 경우에도 같은 클러스터(`kubernetes.svc.default`)에 배포하는 것이 완전히 불가능한 것은 아니며, 자격 증명을 명시적으로 제공하면 됩니다. 이때 사용하는 형태가 다음과 같은 `--in-cluster` 옵션을 동반한 등록입니다.

```bash
argocd cluster add <context> --in-cluster --namespace <your namespace="">
```

이렇게 등록해 두면 이후 Application을 생성할 때 `--dest-server`에 등록된 외부 클러스터의 API 서버 주소를 지정함으로써, 그 클러스터를 배포 대상으로 삼을 수 있습니다. 실제로 이 등록된 클러스터를 대상으로 첫 Application을 만들고 동기화하는 흐름은 이어지는 **첫 Application 생성 및 동기화**에서 다룹니다.
&lt;/your&gt;&lt;/context&gt;

## 첫 Application 생성 및 동기화

인증을 마치고 배포 대상까지 정해졌다면, 이제 실제로 Application 하나를 만들어 클러스터에 올려 볼 차례입니다. 예제로 쓰이는 것은 Argo CD가 공식으로 제공하는 guestbook 애플리케이션으로, `https://github.com/argoproj/argocd-example-apps.git` 저장소의 `guestbook` 경로에 들어 있습니다. Application을 CLI로 만드는 명령과 그 인자 구조는 앞서 다룬 대로이며, 동기화가 사실상 `kubectl apply`로 귀결된다는 점도 이미 설명했습니다. 여기서는 그 사실을 전제로, **UI에서 Application이 어떤 화면 흐름으로 만들어지는가**, **동기화 직후 무엇이 관찰되는가**, 그리고 실전에서 초보자를 가장 자주 넘어뜨리는 **아키텍처 호환성 함정**에 집중합니다.

### 시작하기 전에: guestbook 예제의 아키텍처 함정

명령을 실행하기 전에 반드시 짚어야 할 실전 함정이 하나 있습니다. 공식 문서는 이 예제 애플리케이션이 **AMD64 아키텍처에서만 호환될 수 있다**고 명시적으로 경고합니다. ARM64나 ARMv7 같은 다른 아키텍처에서 실행 중이라면, 해당 플랫폼용으로 빌드되지 않은 의존성이나 컨테이너 이미지 때문에 문제를 겪을 수 있습니다.

이 경고가 중요한 이유는, 문제가 발생하는 지점이 Argo CD 자체가 아니라 **배포된 파드의 런타임**이라는 데 있습니다. ARM 기반 아키텍처에서 guestbook을 동기화하면, Argo CD의 Sync status는 `Synced`로 정상 표시되어 — 즉 Git과 클러스터의 선언이 일치한다는 판정은 통과하지만 — 해당 플랫폼용으로 빌드되지 않은 이미지나 의존성 때문에 파드가 정상적으로 동작하지 못해 Health가 정상으로 올라오지 않을 수 있습니다. 이는 앞서 강조한 **Sync와 Health가 독립적인 축**이라는 사실이 실제로 드러나는 대표적 사례입니다. "적용은 성공했으나 동작이 불건강한" 상황인 것입니다. 이런 아키텍처에서는 애플리케이션의 호환성을 검증하거나, 필요하다면 자신의 플랫폼용 이미지를 별도로 빌드하는 것을 고려해야 합니다.

### UI를 통한 Application 생성 화면 흐름

CLI 한 줄로 Application을 만들 수도 있지만, Argo CD를 처음 익힐 때는 UI의 폼을 따라가 보는 편이 각 필드가 무엇을 요구하는지 감각적으로 이해하는 데 도움이 됩니다. 브라우저로 Argo CD 외부 UI에 접속해 관리자 자격 증명으로 로그인한 뒤, Applications 페이지에서 **+ New App** 버튼을 누르는 것으로 시작합니다.

![+ New App 버튼](https://argo-cd.readthedocs.io/en/stable/assets/new-app.png)

폼이 열리면 애플리케이션의 기본 정보부터 채웁니다. 이름은 `guestbook`, 프로젝트(project)는 `default`, 그리고 동기화 정책(sync policy)은 `Manual`로 그대로 둡니다. 동기화 정책을 수동으로 두는 것은 처음 배포 흐름을 눈으로 관찰하기에 알맞습니다. 자동 동기화가 아니므로, 리소스가 실제로 만들어지는 순간을 사용자가 직접 트리거하고 관찰할 수 있기 때문입니다.

![애플리케이션 정보 입력](https://argo-cd.readthedocs.io/en/stable/assets/app-ui-information.png)

다음은 소스, 즉 원하는 상태가 어디에 있는지를 지정하는 단계입니다. 저장소 URL에 예제 저장소 주소를 넣고, 리비전(revision)은 `HEAD`로, 경로(path)는 `guestbook`으로 설정합니다. 이 세 필드가 결합되어 Repository Server가 렌더링할 대상이 확정됩니다.

![저장소 연결](https://argo-cd.readthedocs.io/en/stable/assets/connect-repo.png)

마지막으로 Destination(대상)을 지정합니다. Argo CD가 실행 중인 클러스터에 배포한다면 클러스터 URL로 `https://kubernetes.default.svc`를 지정하거나, 클러스터 이름으로 `in-cluster`를 선택합니다. 네임스페이스는 `default`로 둡니다.

![대상 설정](https://argo-cd.readthedocs.io/en/stable/assets/destination.png)

모든 정보를 채운 뒤 UI 상단의 **Create** 버튼을 누르면 `guestbook` Application이 생성됩니다.

![애플리케이션 생성](https://argo-cd.readthedocs.io/en/stable/assets/create-app.png)

### 동기화 실행과 화면상의 흐름

생성 직후의 상태는 앞서 핵심 개념에서 본 그대로입니다. 아직 아무 리소스도 배포되지 않았으므로 애플리케이션은 `OutOfSync`이고 Health는 `Missing`입니다. 이제 동기화를 트리거하면, UI에서는 다음과 같은 상호작용이 이어집니다. Applications 페이지에서 guestbook 애플리케이션의 **Sync** 버튼을 누릅니다.

![guestbook 애플리케이션의 Sync 버튼](https://argo-cd.readthedocs.io/en/stable/assets/guestbook-app.png)

버튼을 누르면 패널이 열리고, 거기서 **Synchronize** 버튼을 눌러 실제 동기화를 확정합니다. CLI에서 동기화를 트리거하려는 경우에도 결과는 같습니다 — 동기화가 완료되면 guestbook 앱이 실행되기 시작합니다.

### 동기화 이후에 무엇이 보이는가: 리소스 트리와 관찰 대상

동기화가 이 섹션에서 진짜로 중요한 이유는 명령의 의미가 아니라 **그 이후에 관찰할 수 있는 산출물** 때문입니다. 동기화가 완료되면 guestbook 앱이 실행되며, 이제 그 **리소스 구성 요소(resource components), 로그(logs), 이벤트(events), 그리고 평가된 헬스 상태(assessed health status)**를 함께 확인할 수 있습니다. 이 네 가지가 Argo CD가 제공하는 관측 창구이며, 각각이 서로 다른 질문에 답합니다.

| 관찰 대상 | 어디서 보는가 | 무엇을 알려주는가 |
| --- | --- | --- |
| 리소스 구성 요소(리소스 트리) | guestbook 애플리케이션을 클릭해 진입 | Application이 어떤 하위 Kubernetes 리소스(Deployment, Service 등)로 펼쳐졌는가 |
| 헬스 상태 | 각 리소스 노드와 애플리케이션 전체 | 실제 리소스가 올바로 동작하며 요청을 처리하는가 |
| 로그 | 개별 파드/컨테이너 | 컨테이너 내부에서 무슨 일이 일어나는가 |
| 이벤트 | 리소스별 | 리소스가 생성·스케줄·재시작되며 남긴 사건의 기록 |

애플리케이션 항목을 클릭하면 더 자세한 내용을 볼 수 있는데, 이때 나타나는 것이 리소스 트리(resource tree) 화면입니다.

![guestbook 리소스 트리](https://argo-cd.readthedocs.io/en/stable/assets/guestbook-tree.png)

이 트리는 단순한 목록이 아니라, 하나의 Application이 실제로 어떤 하위 리소스들로 전개되었는지를 시각적으로 펼쳐 보여 줍니다. guestbook의 경우 `guestbook-ui`라는 Deployment와 Service가 트리에 나타나고, 각 노드마다 Sync 상태와 Health가 병렬로 표시됩니다. 여기서 앞서 다룬 개념들이 하나의 화면으로 수렴합니다. 동기화 전에 `OutOfSync / Missing`이던 각 리소스가 동기화 후 `Synced` 상태로, 그리고 헬스가 정상으로 전환되는 과정을, 리소스 단위로 나뉘어 관찰할 수 있는 것입니다.

### 무엇이 잘못됐을 때 어디를 봐야 하는가

동기화 후 기대한 대로 앱이 뜨지 않았을 때, 이 관측 창구들과 아키텍처 컴포넌트의 분업을 결합하면 진단의 초점을 좁힐 수 있습니다. 앞서 아키텍처에서 세 컴포넌트의 책임을 정리했는데, 그 지도가 여기서 실전 진단 흐름으로 이어집니다.

- **UI/CLI가 아예 애플리케이션 상태를 못 보여 주거나 권한 오류가 난다면** → 요청의 진입점이자 인증·RBAC를 시행하는 **API Server** 쪽 문제일 가능성이 큽니다.
- **리소스 트리에 기대한 리소스가 나타나지 않거나 매니페스트가 의도와 다르게 렌더링됐다면** → 대상 상태를 산출하는 **Repository Server**를 의심합니다. 저장소 URL·리비전·경로가 올바른지, guestbook 경로의 매니페스트가 제대로 빌드됐는지를 확인하는 것이 순서입니다.
- **트리에는 리소스가 정상적으로 나타나고 Sync는 `Synced`인데 파드가 뜨지 않는다면** → 상태 비교와 조정을 수행하는 **Application Controller**의 판정 결과, 즉 Health를 봅니다. 이 경우 진단의 무게는 Argo CD에서 파드 내부로 옮겨가며, 개별 파드의 **로그와 이벤트**가 핵심 단서가 됩니다.

앞서 언급한 ARM 아키텍처의 guestbook 문제가 정확히 세 번째 갈래에 해당합니다. Sync status는 `Synced`로 아무 문제가 없다고 표시되지만, 리소스 트리에서 `guestbook-ui` Deployment의 Health가 정상으로 올라오지 않고, 해당 이미지나 의존성이 그 플랫폼용으로 빌드되지 않아 발생하는 문제가 드러납니다. "Git과 일치하는가"와 "실제로 잘 돌아가는가"가 별개의 질문이라는 원칙을, 로그와 이벤트가 구체적인 증거로 확인해 주는 셈입니다.

## 매니페스트 소스 도구 이해하기

Argo CD가 지원하는 소스 타입의 목록과 각 타입의 정의는 이미 정리되었으므로, 여기서는 한 걸음 더 들어가 실전에서 부딪히는 질문에 답합니다. 하나의 Git 경로를 앞에 두었을 때 Argo CD는 그 경로를 어떤 도구로 렌더링하며, 각 소스 타입은 저장소 안에서 실제로 어떤 디렉터리 모습을 띠는가, 그리고 네 가지 기본 도구로도 부족할 때 무엇을 선택해야 하는가입니다.

### 소스의 생김새가 도구를 가른다

Application이 가리키는 소스는 결국 저장소 URL·리비전·경로의 조합이 지목하는 **하나의 디렉터리**입니다. Repository Server가 이 디렉터리를 받아 매니페스트를 렌더링한다는 점은 앞서 아키텍처에서 다뤘는데, 실무에서 중요한 것은 그 디렉터리 안에 무엇이 들어 있느냐가 어떤 Tool을 쓸지를 사실상 결정한다는 사실입니다.

| 경로에 존재하는 것 | 대응하는 소스 타입 | 렌더링 방식 |
| --- | --- | --- |
| `kustomization.yaml` | Kustomize | Kustomize 빌드 |
| `Chart.yaml` / `values.yaml` | Helm 차트 | Helm 템플릿 렌더링 |
| `.jsonnet` 파일 | Jsonnet | Jsonnet 평가 |
| 표식 파일 없는 평문 `.yaml`/`.json` 묶음 | 디렉터리 | 그대로 적용(템플릿 없음) |
| 위 어디에도 해당하지 않는 커스텀 렌더링 | 구성 관리 플러그인 | 사용자 정의 도구 |

이 대응 관계를 이해하면, "왜 내 Application이 Helm으로 인식되지 않을까" 같은 문제를 진단할 때 곧바로 경로 안의 파일 구성을 먼저 살피게 됩니다. 예컨대 Helm 차트를 배포하려는데 지정한 경로에 `Chart.yaml`이 없다면, 그 디렉터리는 Helm 차트로 취급되기 어렵습니다.

#### Kustomize 소스

Kustomize 소스의 최소 형태는 매니페스트 몇 개와 그것을 묶는 `kustomization.yaml` 한 장입니다.

```text
guestbook/
├── deployment.yaml
└── kustomization.yaml
```

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
```

이 `kustomization.yaml`은 지정한 리소스를 "읽어 들이라"는 지시일 뿐이며, 여기에 패치를 얹으면 원본 매니페스트를 건드리지 않고 변경분만 덧씌운 새 매니페스트가 렌더링됩니다. Argo CD의 Application이 이 디렉터리를 경로로 지정하면 Repository Server가 Kustomize로 빌드한 결과가 대상 상태가 됩니다. 여러 환경을 오버레이로 나눠 관리하는 구체적 전략은 이 가이드의 뒤에서 별도로 다룹니다.

#### Helm 소스

Helm 소스는 차트를 정의하는 `Chart.yaml`, 파라미터 기본값을 담은 `values.yaml`, 그리고 값이 주입될 템플릿들로 이루어집니다.

```text
mychart/
├── Chart.yaml
├── values.yaml
└── templates/
    └── deployment.yaml
```

`values.yaml`에 정의된 파라미터가 `templates` 아래 매니페스트에 주입되어 하나의 릴리스가 렌더링됩니다. Argo CD에서는 이 값들을 Application 수준에서 오버라이드할 수 있는데, 이것이 앞서 Repository Server의 입력으로 언급된 "템플릿별 설정(파라미터, `values.yaml`)"이 실제로 작동하는 지점입니다. 즉 같은 차트를 두고 배포마다 달라지는 값만 넘겨 "도장 찍듯" 재사용하는 구조가 여기서 성립합니다.

#### 평문 디렉터리와 Jsonnet

표식 파일 없이 순수한 `.yaml`/`.json` 매니페스트만 모여 있는 디렉터리는 어떤 템플릿 처리도 거치지 않고 그대로 적용됩니다. 템플릿화나 오버레이가 전혀 필요 없는 단순한 애플리케이션이라면 이 형태가 가장 투명합니다.

```text
manifests/
├── deployment.yaml
└── service.yaml
```

Jsonnet 소스는 `.jsonnet` 파일로 매니페스트를 프로그래밍적으로 생성하려는 경우에 쓰이며, 반복과 조건이 많은 구성을 코드처럼 표현할 때 선택지가 됩니다.

### 판단의 진짜 갈림길: 기본 도구 대 구성 관리 플러그인

Kustomize와 Helm 사이의 선택 기준은 앞서 다뤘으므로, 여기서 새롭게 짚어야 할 실전 판단은 다른 축입니다. 바로 **네 가지 기본 도구(Kustomize·Helm·Jsonnet·평문 디렉터리)로 충분한가, 아니면 구성 관리 플러그인이 필요한가**입니다.

핵심 개념에서 Tool을 "파일 디렉터리로부터 매니페스트를 생성하는 도구"로, 구성 관리 플러그인을 "커스텀 도구"로 정의한 바 있습니다. 이 정의를 실전 판단으로 옮기면 명확한 원칙이 하나 나옵니다. 매니페스트를 만들어 내는 방식이 네 가지 기본 도구 중 하나로 표현될 수 있다면 플러그인은 필요 없습니다. 플러그인이 정당화되는 경우는 오직 렌더링 과정 자체가 기본 도구의 범위 밖에 있을 때, 예컨대 사내에서 만든 자체 템플릿 스크립트나 별도의 매니페스트 생성 도구를 통해 원하는 상태를 만들어야 할 때입니다.

구성 관리 플러그인은 결국 기본 도구와 같은 역할을 하는 **커스텀 Tool**입니다. 즉 소스 디렉터리를 입력으로 받아 최종 Kubernetes 매니페스트를 산출하고, Argo CD는 그 산출물을 대상 상태로 삼습니다. 다른 소스 타입과 다른 점은 이 도구가 Argo CD에 내장되어 있지 않아 별도로 설정해 두어야 한다는 것입니다.

이 선택에서 실질적으로 고려할 점은 운영 부담입니다. 기본 도구는 Argo CD가 이미 지원하여 별도 준비 없이 동작하지만, 플러그인은 그 커스텀 도구를 호출할 수 있도록 설정하고 유지해야 하는 추가 표면을 만듭니다. 따라서 실전에서의 순서는 다음과 같습니다.

1. 순수 Kubernetes 매니페스트를 필드 단위로만 조정하면 되는가 → 평문 디렉터리 또는 Kustomize.
2. 배포 시점에 정해지는 값을 파라미터화하거나 서드파티 스택을 값으로 조립해야 하는가 → Helm.
3. 반복·조건이 많아 매니페스트를 코드처럼 생성하고 싶은가 → Jsonnet.
4. 위 어느 방식으로도 렌더링 과정을 표현할 수 없는가 → 구성 관리 플러그인.

구성 관리 플러그인의 구체적인 등록·설정 방식은 Argo CD의 Config Management Plugins 문서에서 다루는 별도 주제이며, 이 가이드의 범위를 벗어납니다. 여기서 기억해 둘 실전 지침은, 플러그인은 "마지막 선택지"라는 점입니다. 네 가지 기본 도구 중 하나로 표현 가능한 구성이라면 그것을 택하는 편이 설치와 유지 모두에서 단순합니다.

## Kustomize vs Helm: 템플릿과 오버레이 전략

GitOps를 처음 도입할 때 거의 모두가 같은 벽에 부딪힙니다. "다뤄야 할 YAML이 너무 많다"는 것입니다. 환경, 클러스터, 규제 요건 같은 변수를 고려하기 시작하면, 파일 사이의 차이는 미미한데도 거의 같은 YAML을 몇 번이고 복제하게 됩니다. 이는 소프트웨어 공학의 DRY(Don't Repeat Yourself) 원칙을 정면으로 거스르는 상황이며, GitOps 맥락에서는 "Don't Repeat YAML"로 바꿔 부를 만합니다. Kustomize와 Helm은 이 중복을 서로 다른 방식으로 해소하는 두 도구이며, 앞서 소스 타입 수준에서 각 도구가 어떤 파일 구조를 요구하는지는 짚었으므로, 여기서는 실제로 여러 환경을 관리할 때의 **오버레이 전략**과 **파라미터화 전략**을 파고듭니다.

### Kustomize: 원본을 건드리지 않는 오버레이

Kustomize의 핵심은 패치(patch)입니다. Kubernetes 매니페스트를 받아 그 위에 변경을 덧씌우되 원본 매니페스트는 그대로 두고, 변경이 반영된 새 매니페스트를 렌더링해 냅니다. 이 성질을 여러 환경에 확장하는 관용적 구조가 base와 overlays의 분리입니다.

```text
.
├── base
│   ├── deployment.yaml
│   └── kustomization.yaml
└── overlays
    └── dev
        └── kustomization.yaml
```

최상위에 공통 정의를 담는 `base`가 있고, 그 아래 `overlays/dev`처럼 환경별 디렉터리가 놓입니다. 디렉터리 이름이나 구조가 강제되는 것은 아니지만, 위 형태가 사실상의 표준으로 통용됩니다. `base/deployment.yaml`은 평범한 Deployment 매니페스트이고, `base/kustomization.yaml`은 그것을 "읽어 들이라"는 지시만 담습니다.

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
```

진짜 마법은 `overlays/dev` 쪽에서 일어납니다. 이 디렉터리의 `kustomization.yaml`은 두 가지를 지시합니다. 하나는 `../../base`가 제공하는 리소스를 읽어 들이라는 것이고, 다른 하나는 그중 특정 Deployment의 복제본 수를 3으로 바꾸라는 것입니다. 예컨대 base의 복제본이 1일 때, dev 오버레이가 이를 3으로 치환하도록 패치를 얹으면 다음과 같은 결과 매니페스트가 렌더링됩니다.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx
  name: nginx
spec:
  replicas: 3          # base의 값을 overlay가 덮어쓴 결과
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
        - image: nginx
          name: nginx
```

렌더링 결과만 확인하거나 클러스터에 바로 적용하는 명령은 앞서 소개한 방식과 동일합니다.

```bash
# dev 오버레이의 렌더링 결과 확인
kustomize build overlays/dev/

# dev 오버레이를 클러스터에 직접 적용
kubectl apply -k overlays/dev/
```

이 구조가 힘을 발휘하는 이유는, 환경 간 애플리케이션 배포를 한 걸음 물러나 바라보면 구조적으로는 거의 같기 때문입니다. 데이터베이스 자격 증명, 스케일, 컨테이너 이미지 같은 값들은 애플리케이션의 동작을 크게 바꾸지만, 선언의 골격 자체는 환경이 달라져도 그대로 유지되고 오직 값만 달라집니다. 따라서 공통 골격은 `base`에 한 번만 두고, 환경마다 달라지는 델타만 오버레이로 저장하면 됩니다. 이것이 Kustomize를 "같은 애플리케이션을 여러 환경에 배포하되 환경 간 차이만 보관하는" 강력한 도구로 만드는 지점입니다.

### Helm: 값을 주입해 릴리스를 찍어 내기

Helm은 접근이 다릅니다. Helm은 Kubernetes 애플리케이션 배포를 위한 "템플릿 엔진"으로, 앞서 소스 구조에서 본 차트를 입력받아 사용자가 공급한 값을 주입하고, 그 결과로 하나의 **릴리스(release)**를 생성합니다. 릴리스는 클러스터에 실제로 배포되는 YAML의 최종 표현이며, 그 정보는 클러스터에 시크릿으로 저장됩니다.

차트 저장소를 등록하고 값을 주입해 릴리스를 배포하는 흐름은 다음과 같습니다. `--set`으로 파라미터를 즉석에서 오버라이드하면, 같은 차트를 서로 다른 값으로 반복 배포할 수 있습니다.

```bash
$ helm repo add akuity-demos https://akuity.github.io/demo-helm-charts/
"akuity-demos" has been added to your repositories

$ helm install myapp --create-namespace --namespace example \
    --set replicaCount=3 akuity-demos/simple-go
NAME: myapp
NAMESPACE: example
STATUS: deployed
REVISION: 1

$ kubectl get pods -n example
NAME                               READY   STATUS    RESTARTS   AGE
myapp-simple-go-64c587f7b4-6htcs   1/1     Running   0          8s
myapp-simple-go-64c587f7b4-f2bmc   1/1     Running   0          8s
myapp-simple-go-64c587f7b4-r6tdg   1/1     Running   0          8s
```

여기서 요점은 Helm이 애플리케이션을 패키징하고 YAML 매니페스트를 파라미터화하는 방법이자 동시에 배포를 위한 템플릿 엔진이라는 점입니다. 애플리케이션을 하나의 차트로 묶어 두면, 각 배포마다 필요한 값만 조정하면서 "도장 찍듯" 배포할 수 있습니다. 이때 환경별로 보관해야 하는 것은 각 배포용 값 파일(values 파일)뿐입니다. Kustomize가 환경 간 델타를 오버레이 패치로 저장한다면, Helm은 그것을 values 파일로 저장하는 셈입니다.

### 패치인가 파라미터화인가: 선택의 실질적 기준

두 도구의 차이를 오버레이 전략의 관점에서 압축하면, "이미 알고 있는 값을 원본에 얹느냐"와 "적용 시점까지 모르는 값을 밖에서 채워 넣느냐"의 대비로 정리됩니다.

| 관점 | Kustomize (패치·오버레이) | Helm (파라미터화·템플릿) |
| --- | --- | --- |
| 다루는 단위 | 원본 매니페스트에 얹는 리소스 필드 패치 | 템플릿에 주입하는 파라미터 |
| 환경 간 차이의 저장 형태 | 환경별 오버레이 디렉터리(예: `overlays/dev`) | 환경별 values 파일 |
| 값을 언제 아는가 | 패치할 값을 사전에 알고 있을 때 적합 | 클러스터에 적용되기 전까지 값을 모를 때 유리 |
| 원본 취급 | 원본을 그대로 두고 델타만 렌더링 | 차트를 하나의 릴리스로 렌더링 |
| Kubernetes 기본 내장 | 예(kubectl·여러 GitOps 도구에 내장) | 아니오(별도 패키지 매니저) |

패치가 쉬운 것은 값을 미리 알고 있을 때뿐입니다. 반대로 대상 클러스터에 적용되기 전까지는 값을 알 수 없는 상황도 있습니다. 대표적인 예가 Ingress 오브젝트입니다. Ingress에는 애플리케이션의 정규화된 도메인명(FQDN)을 채워야 하는 `host` 필드가 있는데, 여러 클러스터에 배포할 때 각 클러스터의 FQDN을 사전에 알 수 없는 경우가 많습니다. 이런 시나리오에서는 구성을 파라미터화하는 Helm이 빛을 발합니다.

### "Kustomize 대 Helm"이 아니라 "Kustomize 와 Helm"

선택의 원칙을 단순화하면 이렇습니다. 주로 순수 Kubernetes 매니페스트를 다루고 특정 필드만 바꾸면 되는 경우라면 Kustomize를 최대한 활용하는 편이 자연스럽습니다. Kustomize는 Kubernetes에 내장되어 있을 뿐 아니라 여러 GitOps 도구에도 내장되어 있기 때문입니다. 반대로 값을 파라미터화해야 하거나 ISV(독립 소프트웨어 벤더)가 제공하는 서드파티 애플리케이션 스택을 소비해야 한다면 Helm 쪽으로 기울거나 아예 Helm 차트를 작성하는 편이 낫습니다.

그러나 실전에서 이것은 배타적 선택이 아닙니다. 대부분의 경우 두 도구를 함께(tandem) 사용하게 됩니다. 두 도구를 병행하면 YAML 중복을 최소화하면서도 각 환경에 필요한 만큼의 커스터마이즈 여지를 남길 수 있으며, 이는 곧 GitOps 저장소를 깨끗하고 이해하기 쉽게 유지하는 길이 됩니다. 저장소 자체를 어떻게 구조화하고 어떤 워크플로우로 환경을 승격할 것인가는 이어지는 저장소 구조·워크플로우 논의에서 별도로 다룹니다.

## GitOps 저장소 구조와 워크플로우 모범 사례

여러 환경 간 차이를 오버레이나 values 파일로 흡수하는 방법은 앞서 다룬 대로입니다. 그러나 그렇게 정리된 매니페스트를 실제로 어떤 저장소에, 어떤 디렉터리·브랜치 구조로 배치하고, 변경을 어떤 절차로 승격(promotion)할 것인가는 별개의 문제입니다. 이 물음에는 만병통치식 정답이 없습니다. 오히려 조직의 형태가 답을 상당 부분 규정한다는 점이 이 주제의 핵심 어려움입니다.

### 코드 저장소와 배포 저장소를 분리하라

GitOps를 도입하는 조직이 가장 먼저 부딪히는 결정은 "애플리케이션을 구동하는 코드"와 "그것을 배포하는 매니페스트"를 같은 저장소에 둘 것인가입니다. 권장되는 답은 분리입니다.

이유는 두 종류의 산출물이 서로 다른 워크플로우를 가지기 때문입니다. 대표적인 사례가 복제본 수를 바꾸는 변경입니다. 이 변경은 애플리케이션의 코드를 전혀 건드리지 않는데, 매니페스트가 코드와 같은 저장소에 얽혀 있으면 이미 프로덕션에서 잘 돌고 있고 변경되지도 않은 코드베이스를 불필요하게 다시 빌드하고 다시 테스트하게 만들 수 있습니다. 게다가 환경 구성 변경에 대한 승인 절차는 코드 변경에 대한 승인 절차와 성격이 다르며, 배포 매니페스트의 리뷰가 개발자의 지속적 통합(CI) 흐름을 가로막아서도 안 됩니다. 따라서 애플리케이션 개발에는 git-flow 같은 분기 모델을 계속 쓰더라도, GitOps 저장소에는 트렁크 기반 개발(trunk-based development)을 채택하는 조직이 많습니다.

### 환경은 브랜치가 아니라 디렉터리로

가장 반직관적이면서도 실무에서 큰 차이를 만드는 원칙이 "환경별 구성을 브랜치가 아니라 디렉터리(폴더)로 관리하라"입니다. Git에 익숙한 사람일수록 매니페스트도 코드처럼 다루고 싶어져, 환경별 장기 브랜치(long-lived branch)를 두고 브랜치 간 병합으로 승격을 처리하려는 본능을 갖습니다. 그러나 GitOps에서 승격되는 것은 코드가 아니라 **매니페스트**이며, 이 차이가 브랜치 모델을 무너뜨립니다.

문제의 본질은 환경 간 구성이 단순히 "같은 것의 최신·구버전" 관계가 아니라는 데 있습니다. Secret이나 ConfigMap처럼 환경마다 근본적으로 달라야 하는 값들이 존재하기 때문에, 환경 구성을 브랜치 간에 병합하는 것은 적절하지 않습니다. 그래서 실무에서는 "이미지 태그만 올리고 나머지 환경 구성은 그대로 두는" 부분 승격을 하게 되는데, 브랜치 모델에서 이는 변경마다 커밋을 하나하나 골라 옮기는 체리픽(cherry-pick)으로 귀결됩니다. 이미지 업데이트를 승격하려고 모든 변경을 체리픽해야 하는 것은 관리 부담을 키우고 득보다 실이 많습니다.

구체적인 시나리오로 두 방식을 대비해 보면 차이가 선명해집니다. 새 이미지 태그 `v1.4.0`을 `dev`에서 검증한 뒤 `prod`로 올린다고 합시다.

**환경=브랜치** 모델에서는 `dev` 브랜치의 여러 커밋 가운데 오직 이미지 태그를 바꾼 커밋만 골라 `prod`로 체리픽해야 합니다.

```bash
# 환경=브랜치: 이미지 태그 커밋만 골라 prod 브랜치로 옮긴다
git checkout prod
git cherry-pick 9f3c1a2   # "bump image to v1.4.0" 커밋 하나만
# 그 커밋에 dev 전용 변경이 섞여 있으면 충돌·오염 위험
git push origin prod
```

체리픽할 커밋이 깔끔하게 이미지 태그 하나만 담고 있으리라는 보장이 없고, 승격할 변경이 여러 커밋에 흩어져 있으면 이 과정은 곧 관리 부담과 오류의 원천이 됩니다.

**환경=디렉터리** 모델에서는 승격이 해당 환경 디렉터리의 파일 하나를 수정하는 평범한 커밋으로 환원됩니다.

```bash
# 환경=디렉터리: prod 오버레이의 이미지 태그만 갱신하는 명시적 커밋
#   overlays/prod/kustomization.yaml 안의 태그를 v1.4.0으로 수정
git commit -am "promote app to v1.4.0 in prod"
git push
```

무엇을 승격하는지가 diff에 그대로 드러나고, 환경 고유 구성은 각 디렉터리에 격리된 채 유지됩니다. 두 모델의 성격을 정리하면 다음과 같습니다.

| 관점 | 환경 = 브랜치 | 환경 = 디렉터리 |
| --- | --- | --- |
| 승격 방식 | 브랜치 간 병합/체리픽 | 대상 환경 디렉터리 파일 수정 커밋 |
| 부분 승격(이미지만 올리기) | 커밋 단위 체리픽 필요, 오염 위험 | 해당 필드만 수정하는 명시적 diff |
| 환경 고유 구성(Secret·ConfigMap) | 병합에 부적합 | 디렉터리별로 자연스럽게 격리 |
| 어울리는 개발 모델 | git-flow(애플리케이션 코드에 적합) | 트렁크 기반 개발 |

즉 트렁크 기반 개발을 앞서 다룬 Kustomize·Helm과 결합해, 공통 골격은 재사용하고 환경별 차이는 디렉터리로 격리하는 것이 GitOps 워크플로우를 단순하게 유지하는 길입니다.

### Conway의 법칙: 저장소 구조는 조직 구조를 따른다

"최선의" 또는 "표준" 저장소 레이아웃이 하나로 정해지지 않는 근본 이유는 Conway의 법칙에 있습니다.

> "시스템을 설계하는 어떤 조직이든, 그 설계는 필연적으로 조직의 커뮤니케이션 구조를 그대로 본뜬 구조를 낳는다." — Melvin E. Conway

다시 말해 디렉터리 구조가 조직을 규정하는 것이 아니라, 조직이 소통하는 방식과 책임의 경계가 디렉터리 구조를 규정합니다. 서로 다른 워크플로우를 가진 팀 간의 경계(흔히 "사일로"라 불리지만 "경계"라 부르는 편이 정확합니다)가 저장소를 나누는 실제 선이 됩니다. 개발자가 플랫폼 구성을 건드리지 않고 플랫폼 운영자가 개발자의 소스 코드를 건드리지 않듯, 저장소와 디렉터리 경계는 이 책임 분리를 반영해야 합니다.

### 저장소 패턴들

이 원칙을 실제 레이아웃으로 옮기는 방식에는 몇 가지 정착된 패턴이 있으며, 각각은 조직 형태에 따라 장단이 갈립니다.

**1:1 저장소–클러스터 레이아웃.** 하나의 클러스터에 하나의 배포 저장소를 대응시키는 방식입니다. 이 패턴에서는 클러스터를 부트스트랩할 때 Argo CD의 **ApplicationSet**을 활용하는 경우가 많습니다. 예를 들어 저장소가 다음과 같이 구성될 수 있습니다.

```text
cluster-repo/
├── bootstrap/
│   └── applicationset.yaml   # 저장소 내 앱 디렉터리들을 순회해 Application 자동 생성
└── apps/
    ├── guestbook/
    │   └── kustomization.yaml
    └── billing/
        └── kustomization.yaml
```

ApplicationSet은 이런 자동 생성을 위해 여러 제너레이터(예: Git 파일/디렉터리를 순회하는 Git Generator, 등록된 클러스터 목록을 순회하는 Cluster Generator)를 제공하며, 구체적인 필드 사양은 ApplicationSet 문서에서 별도로 다루는 주제입니다.

**모노레포(Monorepo).** 하나의 저장소에서 여러 클러스터를 함께 다루는 방식입니다. 클러스터·환경별 디렉터리를 한 저장소 안에 나누어 두고 전체를 일관된 표준으로 관리합니다. 조직의 표준을 강하게 통일하기 좋은 반면, 저장소가 커질수록 경계 관리가 관건이 됩니다.

**팀별 저장소 대 애플리케이션별 저장소.** "팀당 하나의 저장소(Repo per team)"로 나눌지 "애플리케이션당 하나의 저장소(Repo for Application)"로 나눌지도 흔한 갈림길입니다. 어느 쪽이 옳은지는 다시 조직의 커뮤니케이션 구조가 결정합니다.

이러한 패턴들은 배타적이지 않으며, 실제 사례로는 1:1 레이아웃을 ApplicationSet으로 부트스트랩하는 예, 모노레포에서 다중 클러스터를 다루는 표준 예, 팀별과 애플리케이션별 접근의 장단을 비교한 논의 등이 공개되어 있어 조직에 맞는 출발점을 고르는 참고가 됩니다.

### Pull Request 워크플로우와 승인 경계

GitOps에서 Git 저장소는 단순한 저장 장소가 아니라 환경을 관리하는 인터페이스입니다. 따라서 pull request는 곧 "환경 변경을 제안하고 검토하고 승인하는" 통제 지점이 됩니다. pull request가 열리면 제안된 변경의 개요를 저장소 브랜치 단위로 살펴볼 수 있고, 변경 요약을 덧붙이고, 태그를 달고, 다른 기여자를 언급할 수 있으며, 토픽 브랜치에 커밋을 더해 실제 제안 내용을 드러낼 수 있습니다. 모두가 변경에 동의하면 병합됩니다. 병합된 변경은 즉시 자동으로 배포하는 것이 GitOps의 모범으로, 이렇게 해야 Git 구성과 실제 인프라가 항상 일치하게 됩니다.

여기서 저장소 구조가 승인 정책과 직접 맞물린다는 점이 이 워크플로우의 핵심 실무 통찰입니다. 앞서 본 대로 환경을 디렉터리로 격리했다면, 리뷰와 승인의 경계도 디렉터리를 따라 자연스럽게 그을 수 있습니다. 예컨대 프로덕션 환경 디렉터리(`overlays/prod`)를 대상으로 하는 pull request에는 플랫폼 팀의 승인을 요구하고, 개발 환경 디렉터리의 변경에는 개발 팀의 승인만으로 충분하도록 정책을 나눌 수 있습니다. 이렇게 하면 코드 변경과 성격이 다른 배포 변경의 승인 절차를 분리하라는 원칙이, 저장소의 물리적 구조 위에서 실제 리뷰어 배정으로 구체화됩니다.

또한 문제가 발생했을 때의 추적 이점도 여기서 배가됩니다. 어떤 환경에서 이상이 생기면 그것을 도입한 특정 pull request로 원인을 좁힐 수 있는데, 환경이 디렉터리로 격리되어 있으면 "어느 환경의 어느 애플리케이션에 무엇이 바뀌었는가"가 그 pull request의 diff에 그대로 드러나 조사 범위가 한층 명확해집니다. 결국 코드와 배포의 분리, 환경의 디렉터리 격리, 조직 경계를 반영한 저장소 패턴, 그리고 그 경계 위에 얹힌 pull request 승인 정책이 서로 맞물릴 때, GitOps 저장소는 깨끗하고 감사 가능하며 이해하기 쉬운 상태로 유지됩니다.

## 학습 로드맵과 사전 지식

Argo CD를 효과적으로 다루려면 그것이 딛고 선 기반 기술을 먼저 이해해야 한다는 점은 공식 안내가 분명히 밝히는 전제입니다. 실제로 핵심 개념 문서는 "Git, Docker, Kubernetes, 지속적 전달(Continuous Delivery), GitOps에 이미 익숙하다고 가정한다"는 문장으로 시작합니다. 즉 Argo CD 고유의 어휘와 동작은 이 가이드에서 다루었지만, 그 밑바탕이 되는 기초 기술은 별도로 갖추고 있어야 각 장의 내용이 온전히 소화됩니다. 이 절은 그 사전 지식이 무엇인지 진단하고, 지금까지 다룬 내용을 각자의 처지에 맞게 되짚어 실전으로 넘어가는 발판을 마련하는 데 초점을 둡니다.

### 무엇을 먼저 알고 있어야 하는가

공식 문서가 요구하는 기반 기술은 대체로 정해져 있으며, 각각이 이 가이드의 어느 지점에서 실제로 쓰였는지를 대응시켜 보면 자신의 준비 상태를 가늠할 수 있습니다.

| 사전 지식 | 왜 필요한가 (짧게) | 이 가이드에서 부딪히는 지점 |
| --- | --- | --- |
| Git | 유일한 진실의 원천·커밋·브랜치 이해 | GitOps 원칙, 저장소 구조·워크플로우 |
| Docker/컨테이너 | 이미지·아키텍처 호환성 판단 | guestbook의 AMD64 함정, Health 진단 |
| Kubernetes | 매니페스트·RBAC·네임스페이스 조작 | Application, 설치, 클러스터 등록 |
| 지속적 전달(CD) | 자동 배포·롤백의 맥락 | Sync 정책, 배포 전략 |
| GitOps 원칙 | 선언·조정 루프의 사고 모델 | 전 장에 걸친 공통 어휘 |

이 표의 앞 세 항목이 특히 결정적입니다. Kubernetes의 선언적 성격을 모르면 Sync와 Health의 구분이 추상적으로 남고, Git의 스냅샷·불변성 개념이 없으면 롤백과 감사의 이점이 와닿지 않으며, 컨테이너 이미지의 아키텍처 개념이 없으면 "Synced인데 파드가 안 뜨는" 문제를 진단할 수 없습니다.

### 조건부로만 필요한 지식

모든 사전 지식이 필수는 아닙니다. 공식 안내는 두 가지를 **선택적(조건부)** 학습으로 명시합니다. 첫째는 템플릿 도구로, "애플리케이션을 어떻게 템플릿화할 계획인가에 따라" Kustomize 또는 Helm을 익히라고 안내합니다. 둘째는 CI 도구로, "CI 도구와 통합한다면" GitHub Actions나 Jenkins 문서를 참고하라고 권합니다. 즉 평문 YAML만 다룰 예정이라면 템플릿 도구 학습을 미뤄도 되고, 수동·자동 동기화만으로 충분하다면 CI 통합 지식은 나중에 확보해도 됩니다.

공식 문서가 제시하는 학습 자료를 정리하면 다음과 같습니다.

| 영역 | 성격 | 참고 자료(공식 안내) |
| --- | --- | --- |
| 컨테이너·Docker | 필수 기초 | 컨테이너·VM·Docker 입문 글 |
| Kubernetes | 필수 기초 | Kubernetes 입문 강의, 공식 튜토리얼 |
| 템플릿 도구 | 조건부 | Kustomize, Helm 공식 사이트 |
| CI 통합 | 조건부 | GitHub Actions 문서, Jenkins User Guide |

### 자신의 위치에서 되짚기: 자가 점검

이 가이드를 처음부터 끝까지 통과했다면, 이제 필요한 것은 각 장을 다시 읽는 것이 아니라 이해가 실제로 굳었는지 확인하는 일입니다. 아래 질문들은 앞선 장들의 핵심 판단력을 스스로 진단하는 체크리스트입니다.

- [ ] `OutOfSync`와 `Missing`이 왜 다른 축의 상태인지 즉시 설명할 수 있는가? (핵심 개념)
- [ ] 어떤 애플리케이션이 문제를 겪을 때 API Server·Repository Server·Application Controller 중 어디를 의심할지 가를 수 있는가? (아키텍처)
- [ ] `install.yaml`·`namespace-install.yaml`·`core-install.yaml`을 상황에 맞게 고를 수 있는가? (설치 방식)
- [ ] `argocd login`과 `argocd login --core`의 접근 제어 차이를 아는가? (Core 모드)
- [ ] 환경을 브랜치가 아니라 디렉터리로 나누는 이유를 말할 수 있는가? (저장소 구조)

한 항목이라도 막힌다면, 전체를 다시 훑기보다 괄호 안에 적힌 해당 장만 골라 되짚는 편이 효율적입니다.

### 역할에 따라 무엇을 우선할 것인가

같은 가이드라도 독자의 역할에 따라 무게를 두어야 할 장이 다릅니다. 공식 설치 안내가 구분하는 두 전형—Kubernetes RBAC에만 의존하는 클러스터 관리자와, 여러 팀에 서비스하는 플랫폼 팀—을 기준으로 읽기 우선순위를 나눌 수 있습니다.

```text
사전 지식이 갖춰졌다면, 역할로 분기한다.

당신은 ...?
├─ 클러스터 관리자 (혼자 쓰고, K8s RBAC만 원한다)
│     → Core 모드 + Kubernetes RBAC에 집중
│     → UI·SSO·멀티클러스터 장은 가볍게 훑어도 됨
│     → 첫 관문: argocd login --core, Application/ApplicationSet 권한
│
└─ 플랫폼 팀 (여러 개발 팀에 서비스한다)
      → 멀티테넌트 설치 + HA + 저장소 구조·PR 승인 경계에 집중
      → 첫 관문: API Server 접근 경로, 관리자 계정, 외부 클러스터 등록

두 경우 모두 공통 필수:
  GitOps 원칙 · Argo CD 핵심 개념 · 아키텍처 · 첫 Application
```

역할별로 우선순위를 표로 압축하면 다음과 같습니다.

| 장(주제) | 클러스터 관리자 | 플랫폼 팀 |
| --- | --- | --- |
| 핵심 개념·아키텍처 | 필수 | 필수 |
| Core (Headless) 모드 | 높음 | 낮음 |
| CLI 로그인·관리자 계정 관리 | 낮음(로컬 RBAC 중심) | 높음 |
| 외부 클러스터 등록 | 상황에 따라 | 높음 |
| 저장소 구조·워크플로우 | 중간 | 높음 |

### 다음 단계

사전 지식을 갖추고 자가 점검을 통과했다면, 남은 것은 실제 조직의 저장소와 클러스터 위에서 이 가이드의 절차를 밟아 보는 일입니다. 조건부 지식(템플릿 도구·CI 통합)은 지금 당장 전부 익히기보다, 평문 YAML로 첫 Application을 성공시킨 뒤 필요가 생기는 시점에 Kustomize→Helm→CI 순으로 확장하는 편이 부담이 적습니다. 그리고 어떤 기능을 프로덕션에서 신뢰할 수 있는지—즉 각 기능의 성숙도와 안정성 등급—는 실전 도입 직전에 반드시 확인해야 할 마지막 관문이며, 이는 이 가이드의 마지막에서 다룹니다.

## Argo CD 기능 개요와 성숙도

프로덕션 도입 직전에 반드시 통과해야 할 마지막 관문은 "지금 쓰려는 기능이 얼마나 안정적인가"를 확인하는 일입니다. Argo CD의 기능 목록 자체와 각 기능이 어떤 설계 지향을 뒷받침하는지는 앞서 다루었으므로, 여기서는 그 기능들이 저마다 다른 **성숙도 등급(feature maturity)**을 가진다는 사실과, 그 등급이 실제 운영 결정에 어떤 제약을 부과하는지에 초점을 둡니다.

### 성숙도 등급이라는 개념

Argo CD의 기능은 안정성과 성숙도를 나타내기 위해 특정 상태(status)로 표시될 수 있습니다. 이 등급 체계에서 명시적으로 표시되는 것은 아직 안정화되지 않은 기능들이며, 그 등급은 **Alpha**와 **Beta** 두 가지입니다. 바꿔 말하면, 별도의 등급 표시 없이 제공되는 기능은 안정(stable) 기능으로 간주됩니다. 이 가이드 전반에서 다룬 핵심 기능들—Application CRD 기반의 배포, Sync와 Health 판정, Kustomize·Helm 소스, CLI 로그인과 클러스터 등록 등—은 모두 이 안정 범주에 속하므로, 성숙도 표시를 별도로 신경 쓰지 않고 신뢰할 수 있습니다.

성숙도 등급이 중요한 이유는 그것이 곧 **하위 호환성 보장 여부**를 가르기 때문입니다. 공식 문서는 Alpha와 Beta 기능에 대해 다음과 같은 경고를 명확히 제시합니다.

- Alpha 및 Beta 기능은 하위 호환성을 보장하지 않으며, 향후 릴리스에서 **호환성을 깨는 변경(breaking changes)**이 가해질 수 있습니다.
- 특히 Argo CD 업그레이드를 직접 통제할 수 없는 환경이라면, 이러한 기능을 프로덕션에서 의존하지 않는 것이 강력히 권고됩니다.
- 더 나아가 Alpha 기능의 제거는 업그레이드 이후 리소스를 **예측 불가능한 상태**로 바꿀 수 있습니다.

이 경고가 실무에 남기는 지침은 구체적입니다. 어떤 Alpha/Beta 기능을 사용 중인지 문서로 남겨 두고, 업그레이드 전에 반드시 릴리스 노트를 검토해야 한다는 것입니다. 즉 성숙도 등급은 단순한 라벨이 아니라, 업그레이드 시 무엇이 깨질 수 있는지를 미리 알려 주는 위험 신호로 읽어야 합니다.

### 안정화되지 않은 기능들

공식 문서가 비안정(non-stable) 상태로 분류하는 기능들은 도입 시점과 등급이 함께 명시되어 있습니다. 아래 표는 그 목록으로, 각 기능이 어느 버전에서 도입되었고 현재 어떤 등급에 있는지를 보여 줍니다.

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

이 표에서 눈여겨볼 점은, 대규모·고가용성 운영과 직결되는 기능들이 상당수 아직 Alpha에 머물러 있다는 사실입니다. 예컨대 컨트롤러의 부하를 여러 인스턴스로 나누는 클러스터 샤딩(round-robin, consistent-hashing)이나 Dynamic Cluster Distribution은 모두 Alpha 등급입니다. 이는 프로덕션에서 이런 스케일링 기법에 의존하려는 플랫폼 팀이 업그레이드 시 특히 주의해야 함을 의미합니다. 반대로 여러 팀에게 서비스하기 위해 자주 거론되는 AppSets in any Namespace나 AppSet Progressive Syncs는 Beta까지 올라와 있어, Alpha보다는 안정적이지만 여전히 하위 호환성 보장 밖에 있습니다.

### CRD 필드 수준의 비안정 설정

성숙도 등급은 기능 단위뿐 아니라 특정 CRD의 개별 필드나 설정 항목 수준에서도 표시됩니다. 어떤 애플리케이션 매니페스트나 프로젝트 정의에 특정 필드를 넣는 순간, 그 필드가 비안정 기능에 속할 수 있다는 뜻입니다. 공식 문서가 정리한 대표적인 항목은 다음과 같습니다.

| CRD | 속성(property) | 상태 |
| --- | --- | --- |
| Application | `metadata.annotations[argocd.argoproj.io/skip-reconcile]` | Alpha |
| AppProject | `spec.destinationServiceAccounts.*` | Alpha |
| ApplicationSet | `spec.strategy.*` | Beta |
| ApplicationSet | `status.applicationStatus.*` | Beta |

이 구분이 실무에서 유용한 이유는, 매니페스트를 작성하다가 위와 같은 필드를 쓰게 될 때 "이 필드는 아직 안정화 전"이라는 사실을 곧바로 인지할 수 있기 때문입니다. 예를 들어 Application에 `argocd.argoproj.io/skip-reconcile` 어노테이션을 붙여 특정 애플리케이션의 조정을 건너뛰게 하는 것은 Alpha 기능이므로, Git에 커밋해 여러 환경에 퍼뜨리기 전에 그 위험을 문서화해 두는 편이 안전합니다.

마찬가지로 비안정 기능을 켜고 끄는 구성 항목—대개 `ConfigMap/argocd-cmd-params-cm`의 키나 각 컴포넌트 Deployment의 환경 변수—역시 같은 등급을 물려받습니다. 가령 ApplicationSet 컨트롤러에서 SCM Provider 관련 설정(`applicationsetcontroller.enable.scm.providers`)이나 Progressive Syncs 활성화 스위치(`applicationsetcontroller.enable.progressive.syncs`)를 켜는 것은 각각 Beta 기능을 활성화하는 행위이며, 클러스터 샤딩 알고리즘을 지정하는 설정(`controller.sharding.algorithm`)은 Alpha입니다. 즉 어떤 기능을 활성화하는 설정을 만지는 것 자체가 그 기능의 성숙도 위험을 떠안는 결정이 됩니다.

### 성숙도 정보를 운영 판단에 녹이기

이상의 등급 체계는 프로덕션 도입 여부를 가르는 실용적 체크리스트로 정리할 수 있습니다.

- **등급을 먼저 확인한다.** 도입하려는 기능이 표에 없으면 안정 기능으로 보아도 되고, Beta라면 신중하게, Alpha라면 매우 보수적으로 접근합니다.
- **사용 중인 비안정 기능을 문서화한다.** 어떤 Alpha/Beta 기능이나 필드를 활성화했는지 기록해 두면, 업그레이드 시 무엇을 점검해야 하는지가 명확해집니다.
- **업그레이드 전 릴리스 노트를 반드시 검토한다.** Alpha/Beta 기능은 호환성을 깨는 변경이나 제거의 대상이 될 수 있으므로, 업그레이드가 곧 위험 지점이 됩니다. 특히 업그레이드를 직접 통제할 수 없는 환경에서는 이런 기능에 대한 의존을 피하는 것이 권고됩니다.

Argo CD는 커뮤니티에 의해 활발히 개발되고 있고 릴리스가 GitHub에서 공개적으로 관리되며 도입 조직 목록도 지속적으로 늘고 있다는 점—앞서 거버넌스 근거로 언급한 사실들—은 장기적 신뢰의 토대가 됩니다. 그러나 그 신뢰를 프로덕션 배포로 옮길 때는, 제품 전체의 성숙도가 아니라 **실제로 활성화하는 개별 기능의 등급**을 기준으로 판단해야 합니다. 안정 기능만으로 이루어진 기본 GitOps 워크플로우는 지금 바로 신뢰할 수 있으며, 그 위에 Alpha/Beta 기능을 얹을지는 각 기능의 등급과 업그레이드 통제 가능성을 견주어 결정하면 됩니다.
- - -

### Sources
* &lt;https://argo-cd.readthedocs.io/en/stable/&gt;
* &lt;https://argo-cd.readthedocs.io/en/stable/understand_the_basics/&gt;
* &lt;https://argo-cd.readthedocs.io/en/stable/core_concepts/&gt;
* &lt;https://argo-cd.readthedocs.io/en/stable/getting_started/&gt;
* &lt;https://argo-cd.readthedocs.io/en/stable/operator-manual/&gt;
* &lt;https://argo-cd.readthedocs.io/en/stable/operator-manual/architecture/&gt;
* &lt;https://argo-cd.readthedocs.io/en/stable/operator-manual/installation/&gt;
* &lt;https://argo-cd.readthedocs.io/en/stable/operator-manual/feature-maturity/&gt;
* &lt;https://argo-cd.readthedocs.io/en/stable/operator-manual/core/&gt;
* &lt;https://www.harness.io/blog/gitops-principles&gt;
* &lt;https://www.flexera.com/blog/finops/understanding-gitops-principles-workflows-deployment-types&gt;
* &lt;https://akuity.io/blog/gitops-best-practices-whitepaper&gt;
* &lt;https://akuity.io/blog/getting-into-gitops&gt;
