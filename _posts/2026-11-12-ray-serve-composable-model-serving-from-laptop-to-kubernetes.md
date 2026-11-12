---
layout: post
title: "Ray Serve: Composable Model Serving from Laptop to Kubernetes"
date: 2026-11-12 04:19:57
description: "Ray Serve로 번역 모델을 로컬 HTTP 엔드포인트로 띄우고, 요약·번역 파이프라인으로 조합한 뒤 설정 파일과 KubeRay로 Kubernetes에 운영하는 과정을 정리합니다. 배포와 레플리카 구조, 요청 라우팅, 배포 파라미터, 대시보드와 메트릭까지 차례로 살펴봅니다."
categories: ["Tech Guides", "Data-Architecture"]
tags: ["KubeRay", "Model-Composition", "Request-Routing", "Deployment-Handle", "Model-Serving"]
cover: /assets/images/data-architecture.jpg
---
## TL;DR

Ray Serve는 모델을 온라인 추론 API로 내보내고, 트래픽이 늘면 그 API를 여러 머신으로 넓혀 주는 서빙 라이브러리입니다. 특정 프레임워크에 묶이지 않고, 응답 스트리밍과 동적 요청 배칭, 멀티 노드·멀티 GPU 서빙을 지원합니다. Serve 인스턴스를 이루는 구성 요소는 모두 Ray 액터(상태를 가진 채 별도 프로세스로 실행되는 Ray의 원격 객체)입니다. 모델이나 비즈니스 로직을 담은 배포(deployment)는 실행 시점에 별도 Ray 액터 프로세스로 뜨는 여러 레플리카가 되고, 레플리카 수는 배포마다 늘리거나 줄이거나 오토스케일링할 수 있습니다. `bind`는 배포에 생성자 인자를 묶어 애플리케이션을 만듭니다. 이때 다른 배포를 인자로 넘기면 런타임에 그 자리에는 `DeploymentHandle`이 들어갑니다. `DeploymentHandle`은 그 배포의 레플리카로 요청을 보내는 객체입니다. 그래서 여러 모델을 YAML 같은 정적 그래프 대신 함수 호출처럼 보이는 Python 코드로 조합하고, 모델마다 다른 자원과 머신을 줄 수 있습니다. 내부 구조에서는 컨트롤러가 다른 액터를 만들고 고치고 지우는 제어 평면만 맡습니다. 요청은 HTTP·gRPC 요청을 받아 레플리카로 전달하는 프록시 액터를 거쳐 레플리카로 흐릅니다. 컨트롤러가 죽어도 HTTP·gRPC·핸들 요청은 계속 보낼 수 있지만, 그동안 오토스케일링은 멈춥니다. 임의의 Python 코드를 그대로 실행하므로 FastAPI, MLflow, ONNX 같은 기존 도구와도 자연스럽게 어울립니다.

이 가이드는 번역 모델 하나를 `serve run`으로 로컬에 띄우는 데서 시작해 요약·번역 파이프라인으로 조합하고, 설정 파일과 Kubernetes(KubeRay)로 운영하는 과정을 따라갑니다. KubeRay 없이 Ray 클러스터 자체가 장애를 겪으면 Serve는 스스로 복구하지 못합니다. 프로덕션에서 권장되는 실행 방식은 KubeRay RayService를 쓰는 Kubernetes이고, Kubernetes가 아니어도 Serve CLI로 Ray 클러스터에서 직접 실행할 수 있습니다. Ray Serve가 가장 잘 맞는 경우는 전처리, 검색, 여러 모델 호출, 후처리가 하나의 Python 파이프라인을 이루고 단계마다 부하와 자원 요구가 달라 따로 확장해야 할 때입니다. 반대로 모델 하나를 최대한 빠르게 돌리는 일이나 모델 수명주기 관리가 핵심 요구라면 전용 추론 엔진이나 모델 레지스트리를 함께 붙여야 합니다.

## 한 프로세스로는 따로 확장할 수 없는 다중 모델 추론 파이프라인

단계마다 필요한 자원과 부하가 다른 추론 파이프라인을 웹 서버 하나에 묶으면 모든 단계를 함께 늘리고 줄여야 합니다. Ray Serve는 Ray 위에서 각 단계를 함수 호출처럼 이어 붙이면서도 단계별로 따로 확장하게 해 줍니다.

### 모델 하나짜리 엔드포인트가 무너지는 지점

많은 서빙 솔루션은 "텐서를 받아 텐서를 돌려주는" 방식에 집중합니다. 모델을 미리 정해 둔 구조화된 엔드포인트 뒤에 감싸는 방식입니다. 하지만 실제 서비스에서는 머신러닝을 비즈니스 로직이나 데이터베이스 조회 같은 전통적인 웹 서빙 로직과 결합해야 하는 경우가 많습니다. LLM 애플리케이션이 대표적인 예입니다. 많은 LLM 애플리케이션은 프롬프트 전처리, 벡터 DB 조회, LLM API 호출, 응답 검증을 하나로 묶습니다. 이미지 처리도 전처리, 분할, 필터링으로 이어지는 다단계 파이프라인이 흔합니다. 이때 단계마다 아키텍처나 프레임워크가 다르고 CPU나 GPU처럼 필요한 자원도 다를 수 있습니다.

예를 들어 아래와 같은 가상의 파이프라인을 한 프로세스에 올렸다고 가정해 보겠습니다.

```text
[예시: 가상의 LLM 파이프라인과 단계별 요구 자원]

요청 → 전처리(CPU, 가벼움) → 벡터 DB 조회(I/O 대기) → LLM 호출(GPU, 무거움) → 검증(CPU) → 응답
```

이 구조에서 LLM 단계가 병목이 되면 GPU가 붙은 프로세스 전체를 복제해야 합니다. 그러면 할 일이 거의 없는 전처리와 검증 코드도 GPU 노드를 그만큼 차지하게 됩니다. Serve에서는 각 단계를 따로 오토스케일링(autoscaling)할 수 있어서, 사용자 트래픽에 맞추면서도 자원을 낭비하지 않습니다.

### Ray 위에 얹힌 범용 서빙 계층

Ray Serve는 온라인 추론 API를 만드는 확장 가능한 모델 서빙 라이브러리입니다. 특정 프레임워크에 묶이지 않아서 PyTorch, TensorFlow, Keras로 만든 딥러닝 모델부터 Scikit-Learn 모델, 임의의 Python 비즈니스 로직까지 같은 도구로 서빙할 수 있습니다. LLM 서빙에 필요한 응답 스트리밍, 동적 요청 배칭, 멀티 노드·멀티 GPU 서빙 기능도 갖추고 있습니다.

Serve는 Ray 위에 구축되어 있어 여러 머신으로 쉽게 확장됩니다. 분수 GPU 같은 유연한 스케줄링도 지원하므로, 자원을 나눠 쓰면서 여러 모델을 낮은 비용으로 서빙할 수 있습니다. 확장 수단도 기본으로 들어 있습니다. 레플리카(replica) 수를 조정해 모델 자원을 늘리거나 줄이고, 요청을 배칭해 벡터화 연산의 이점을 살리며, 유연한 자원 할당으로 제한된 하드웨어에서도 많은 모델을 올립니다. HTTP 처리는 FastAPI 통합으로 작성합니다.

Ray가 실행되는 곳이면 노트북, Kubernetes, 주요 클라우드, 온프레미스 어디서든 Serve를 실행할 수 있습니다. 로컬 머신에서 배포 그래프 전체를 시험한 뒤 그대로 클러스터로 옮기면 되고, 그 과정에서 무거운 Kubernetes 개념이나 클라우드 설정을 알 필요는 없습니다. 임의의 Python 코드를 지원하므로 ONNX·TVM 같은 모델 최적화 도구, MLflow·Weights and Biases 같은 모델 레지스트리, Gradio·Streamlit, FastAPI·gRPC와도 자연스럽게 엮입니다.

### 정적 그래프 대신 프로그래밍 방식의 조합

여러 모델을 잇는 방식에서 Serve는 다른 솔루션과 갈라집니다. Serve는 다른 모델을 호출하는 코드가 평범한 함수 호출처럼 보이는 프로그래밍 API로 다중 모델 조합(model composition)을 지원합니다. 아래 표는 두 방식의 차이를 정리한 것입니다.

| 구분 | 정적 그래프 방식 | Ray Serve 방식 |
|---|---|---|
| 파이프라인 정의 | YAML 등 설정 언어로 그래프를 기술합니다 | Python 코드에서 함수 호출처럼 모델을 부릅니다 |
| 작성 방식 | 제약이 많고 다루기 어렵습니다 | 모델마다 다른 자원을 쓰고 클러스터의 다른 머신에서 실행되더라도 일반 프로그램처럼 작성합니다 |

예를 들어 영어 문장을 프랑스어로 옮기는 번역 모델 Translator와, 그 결과를 받아 요약하는 Summarizer를 잇는다고 해 봅시다. 코드에서는 Summarizer가 Translator를 부르는 한 줄로 보이지만, 실행될 때는 두 모델이 각자의 자원과 레플리카 수로 따로 확장됩니다. 이렇게 여러 ML 모델과 비즈니스 로직을 하나의 Python 프로그램으로 묶어 복잡한 추론 서비스를 만드는 일에 Serve가 특히 잘 맞습니다.

## 확장 단위가 되는 배포, 레플리카, 인그레스, 애플리케이션

Serve에서 확장의 기본 단위는 배포(deployment)입니다. 배포는 실행될 때 여러 레플리카로 복제되고, 배포 여럿을 묶은 애플리케이션(application)으로 클라이언트 앞에 나섭니다. 애플리케이션의 진입점은 최상위 인그레스 배포(ingress deployment) 하나가 맡습니다.

### 레플리카 여러 개로 실행되는 배포

배포는 Serve의 핵심 개념입니다. 들어오는 요청을 처리할 비즈니스 로직이나 ML 모델을 담고 있으며, Ray 클러스터 전체로 확장됩니다. 배포 하나에는 사용자가 작성한 Python 함수나 클래스 하나가 들어가고, `@serve.deployment` 데코레이터가 그 클래스를 Serve의 `Deployment` 객체로 바꿉니다. 데코레이터 파라미터는 배포마다 따로 지정하므로, 배포끼리 독립적으로 확장하고 구성할 수 있습니다.

실행 중인 배포는 레플리카 여러 개로 이루어집니다. 레플리카는 클래스나 함수의 개별 복사본이며, 각자 별도의 Ray 액터, 즉 별도 프로세스에서 시작됩니다. 레플리카 수는 들어오는 요청 부하에 맞춰 늘리거나 줄일 수 있고 오토스케일링에 맡길 수도 있습니다. 예를 들어 Translator 배포는 레플리카 넷, Summarizer 배포는 레플리카 하나로 따로 운영하는 식입니다. 레플리카마다 CPU·GPU를 얼마나 예약할지 정하는 방법은 Translator를 배포로 바꾸는 과정에서 함께 다룹니다.

### bind로 정의하고 serve.run으로 올리는 애플리케이션

배포를 정의했다고 바로 서빙이 시작되지는 않습니다. 정의부터 실행까지는 세 단계를 거칩니다.

1. Python 클래스(단순한 경우에는 함수)에 `@serve.deployment`를 붙여 배포를 정의합니다.
2. 생성자에 넘길 인자와 함께 배포를 `bind`해서 애플리케이션을 정의합니다.
3. 그 결과를 `serve.run`이나 같은 역할을 하는 `serve run` CLI로 배포합니다.

다음 코드에서는 Translator를 생성자 인자 하나만 받는 가장 작은 형태로 줄여 이 흐름을 보여 줍니다.

```python
from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment
class Translator:
    # Take the target language as an argument to the constructor.
    def __init__(self, language):
        self.language = language

    def __call__(self):
        return self.language

translator_app = Translator.bind("French")
handle: DeploymentHandle = serve.run(translator_app)
assert handle.remote().result() == "French"
```

`serve.run`은 `DeploymentHandle`을 돌려주므로 Python 코드에서 애플리케이션을 바로 호출할 수 있습니다. 핸들 호출을 체이닝하고 비동기로 다루는 방법은 모델 조합을 설명할 때 이어서 봅니다.

애플리케이션은 Serve 클러스터에서 업그레이드하는 단위이며, 배포 하나 이상으로 구성됩니다. 호출 경로는 두 가지입니다. 지정한 `route_prefix`에서 HTTP로 부르거나, Python에서 `DeploymentHandle`로 부를 수 있습니다.

### 모든 트래픽을 받는 인그레스 배포

애플리케이션 안에 배포가 여럿 있어도 `serve.run`에 넘기는 "최상위" 배포는 언제나 하나뿐입니다. 이 배포가 애플리케이션으로 들어오는 모든 트래픽의 진입점이 되므로 인그레스 배포라고 부릅니다. 인그레스 배포는 보통 다른 배포로 요청을 보내거나 `DeploymentHandle` API로 다른 배포를 호출하고, 그 결과를 조합해 사용자에게 돌려줍니다. 예를 들어 번역·요약 애플리케이션을 그림으로 나타내면 아래와 같습니다.

```text
[예시: 번역·요약 애플리케이션의 배포 그래프]

           HTTP 요청
               │
               ▼
 ┌───────────────────────────┐
 │ Summarizer (인그레스 배포)   │  ← serve.run에 넘기는 최상위 배포
 │  레플리카 ×1                │
 └─────────────┬─────────────┘
               │ DeploymentHandle 호출
               ▼
 ┌───────────────────────────┐
 │ Translator                │
 │  레플리카 ×4                │
 └───────────────────────────┘
```

애플리케이션의 HTTP 처리 로직도 인그레스 배포에 둡니다. 기본 동작에서는 클래스의 `__call__` 메서드가 Python ASGI 웹 프레임워크인 `Starlette`의 요청 객체를 받아 호출되고, 반환값은 JSON으로 직렬화됩니다. JSON이 아닌 응답이 필요하면 다른 `Starlette` 응답 객체를 직접 반환해도 됩니다. 그래서 Translator를 단독으로 인그레스 배포로 서빙한다면 HTTP 요청을 받는 `__call__`이 있어야 합니다.

## 제어 평면만 맡는 컨트롤러와 프록시·레플리카로 이어지는 데이터 경로

Serve 인스턴스에서 컨트롤러는 다른 액터를 만들고 고치고 지우는 제어 평면이고, 실제 요청은 프록시에서 레플리카로 곧장 흐릅니다. 컨트롤러가 멈춰도 HTTP·gRPC·핸들 요청은 계속 보낼 수 있지만, 오토스케일링은 멈춥니다.

### Serve 인스턴스를 이루는 네 종류의 액터

Serve는 Ray 액터 위에서 동작합니다. Serve 인스턴스 하나는 컨트롤러(controller), HTTP 프록시(proxy), gRPC 프록시, 레플리카 액터로 구성됩니다. Serve 애플리케이션을 이루는 액터 종류와 각자의 역할은 아래 구성도에 정리되어 있습니다.

![Ray Serve 아키텍처 구성도: 컨트롤러, 프록시, 레플리카 액터의 배치와 역할](https://docs.ray.io/en/latest/_images/architecture-2.0.svg)

- **컨트롤러**: Serve 인스턴스마다 하나만 있는 전역 액터로, 제어 평면을 관리합니다. 다른 액터를 생성·갱신·삭제하는 일이 모두 여기서 이뤄지며, 배포를 만들거나 조회하는 Serve API 호출도 컨트롤러에 대한 원격 호출입니다.
- **HTTP 프록시**: Uvicorn HTTP 서버를 실행합니다. 들어온 요청을 레플리카로 넘기고, 처리가 끝나면 응답을 돌려줍니다.
- **gRPC 프록시**: 유효한 `port`와 `grpc_servicer_functions`를 지정해 Serve를 시작한 경우에만 HTTP 프록시와 함께 뜹니다. grpcio 서버를 실행하며 하는 일은 HTTP 프록시와 같습니다.
- **레플리카**: 요청을 받아 실제 코드를 실행하는 액터입니다. ML 모델 인스턴스를 담을 수 있고, `@serve.batch`로 요청을 묶어 처리할 수 있습니다.

이와 별도로 Ray 클러스터의 노드마다 Serve REST API 서버가 있어서 Serve에 접속해 REST 요청에 응답합니다. Serve를 관리하는 CLI도 함께 제공됩니다.

서드파티 분석 글인 "Ray Serve Architecture Deep Dive"는 이 구조를 조정(reconcile) 루프로 설명합니다. 이 글의 설명으로는 애플리케이션 실행 호출이나 설정 파일이 목표 상태를 detached 컨트롤러에 넘기고, 컨트롤러가 애플리케이션·배포·프록시·레플리카 액터를 그 상태에 맞춰 조정합니다. 같은 글은 일반 요청의 데이터 경로가 컨트롤러를 거치지 않는다고 봅니다.

### 프록시에서 레플리카까지 가는 요청 경로

HTTP나 gRPC 요청이 프록시에 도착하면 세 단계를 거칩니다.

1. 프록시가 요청을 받아 파싱합니다.
2. HTTP URL 경로나 애플리케이션 이름 메타데이터로 해당 배포를 찾고, 요청을 큐에 넣습니다.
3. 큐에 있는 요청마다 사용 가능한 레플리카를 찾아 보냅니다. 모든 레플리카에 진행 중인 요청이 `max_ongoing_requests`보다 많으면, 레플리카가 빌 때까지 요청은 큐에서 기다립니다.

레플리카도 자체 요청 큐를 두고 요청을 하나씩 실행합니다. `asyncio`를 쓰면 동시에 처리할 수도 있습니다. 핸들러(배포 함수 또는 배포 클래스의 `__call__`)를 `async def`로 선언하면 레플리카는 핸들러 실행이 끝나기를 기다리지 않습니다. 일반 `def`로 선언하면 핸들러가 반환할 때까지 블록됩니다. Translator처럼 모델 추론이 무거운 핸들러라면 이 선언 방식에 따라 레플리카 하나가 요청을 겹쳐 받을 수 있는지가 갈립니다.

모델 조합에서 HTTP나 gRPC가 아니라 `DeploymentHandle`로 요청을 보내면, 요청은 핸들 안의 큐에 들어가고 위의 3단계부터 진행됩니다. `DeploymentHandle`은 같은 노드에 있는 라우터의 핸들을 감싼 것입니다. 그래서 레플리카끼리 핸들로 주고받는 요청도 HTTP·gRPC 요청과 같은 데이터 경로를 지나고, 같은 배포 선택과 배칭 절차가 적용됩니다. 번역·요약 애플리케이션에 대입해 보면 경로는 대략 아래와 같습니다.

```text
[예시: Summarizer → Translator 요청 경로]

클라이언트 ──HTTP──▶ 프록시 ──(1 파싱, 2 경로로 배포 찾기·큐)──▶ 라우터
                                                                 │ 3
                                                                 ▼
                                                    Summarizer 레플리카
                                                                 │ handle 호출
                                                                 ▼
                                                    핸들 큐 → 라우터 (3부터)
                                                                 │
                                                                 ▼
                                                    Translator 레플리카 ×4
```

수치를 넣어 보면 3단계의 대기 조건이 분명해집니다. 가상의 예로 Translator 레플리카가 넷이고 각 레플리카의 `max_ongoing_requests`가 5라면, 진행 중 요청이 20개를 넘는 순간부터 추가 요청은 배포 큐에서 기다립니다. 이 상한과 큐 길이를 조정하는 파라미터는 배포 파라미터를 다룰 때 따로 설명합니다.

요청 본문의 크기에 따라 전달 방식도 달라집니다. 작은 요청 객체는 액터 사이의 네트워크 호출로 바로 전송됩니다. 100KiB 이상인 큰 요청은 Ray의 공유 메모리 객체 저장소(object store)에 쓰이고, 레플리카가 zero-copy로 읽습니다. 긴 문서를 통째로 번역 요청으로 보내면 후자의 경로를 탑니다.

모델 코드에서 예외가 나면 Serve가 잡아서 감싼 뒤 트레이스백과 함께 500 상태 코드로 돌려줍니다. 레플리카는 그 뒤에도 계속 요청을 처리합니다.

### 컨트롤러 안에서 도는 오토스케일러

오토스케일러는 별도 서비스가 아니라 Serve 컨트롤러 액터 안에서 실행됩니다. 아래 다이어그램은 부하에 따라 배포의 레플리카 수가 늘거나 줄 때 메트릭이 어떻게 흐르는지 보여 줍니다. 핸들과 레플리카에서 나온 화살표가 컨트롤러 쪽으로 모이는 부분을 보면 됩니다.

![오토스케일링 흐름도: 핸들과 레플리카의 메트릭이 컨트롤러의 오토스케일러로 전달되어 레플리카 수를 조정하는 과정](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/autoscaling.svg)

각 `DeploymentHandle`과 레플리카는 주기적으로 메트릭을 오토스케일러에 보냅니다. 오토스케일러는 배포마다 핸들 큐와 레플리카의 진행 중 쿼리를 주기적으로 확인해 레플리카 수를 바꿀지 결정합니다. 반대 방향의 정보 흐름도 있습니다. 핸들은 컨트롤러를 계속 폴링해 새 레플리카를 찾고, 새 레플리카가 보이면 버퍼에 쌓인 쿼리나 새 쿼리를 `max_ongoing_requests`에 이를 때까지 보냅니다. 레플리카 선택에는 power of two choices 전략을 씁니다. 이때 어느 레플리카도 동시에 `max_ongoing_requests`보다 많은 요청을 처리하지 않도록 제약합니다.

### 장애 종류마다 다른 복구 주체

장애는 누가 복구하느냐로 나눠 보면 정리가 쉽습니다.

| 장애 대상 | 복구 주체와 결과 |
|---|---|
| 레플리카 액터 | 컨트롤러가 새 레플리카로 교체합니다. |
| 프록시 액터 | 컨트롤러가 재시작합니다. |
| 컨트롤러 액터 | Ray가 재시작합니다. |
| 노드 또는 클러스터 (KubeRay RayService 사용) | KubeRay가 노드나 클러스터를 복구하며, GCS FT(GCS 장애 내성) 기능으로 클러스터 크래시를 피할 수 있습니다. |
| Ray 클러스터 (KubeRay 미사용) | Ray Serve는 복구하지 못합니다. |

액터를 호스팅하던 머신이 죽으면 그 액터들은 다른 머신에서 자동으로 다시 시작됩니다. 라우팅 정책과 배포 설정 같은 컨트롤러 데이터는 헤드 노드(head node)의 Ray GCS(Global Control Store)에 체크포인트되므로 이때도 남습니다. 반면 라우터와 레플리카가 들고 있던 일시 데이터, 즉 네트워크 연결과 내부 요청 큐는 사라집니다. 머신 장애가 났을 때 Translator 레플리카 큐에 쌓여 있던 요청이 이 일시 데이터에 해당합니다.

컨트롤러 장애는 영향 범위가 따로 정해져 있습니다. 컨트롤러가 죽어 있는 동안에도 HTTP, gRPC, `DeploymentHandle` 요청은 계속 보낼 수 있지만 오토스케일링은 멈춥니다. 컨트롤러가 복구되면 오토스케일링도 다시 시작되지만, 그동안 모은 메트릭은 모두 잃습니다. 

> **운영 포인트:** 컨트롤러가 재시작된 직후에는 오토스케일러가 메트릭을 처음부터 다시 모으므로, 이 구간에 트래픽이 몰릴 것 같으면 배포 큐 길이와 레플리카 수를 직접 지켜보고 필요하면 `num_replicas`를 넉넉히 잡아 두세요.

### 노드마다 하나씩 띄울 수 있는 프록시

HTTP 프록시는 기본적으로 헤드 노드에 하나 있고, `serve.start()`나 설정 파일의 `proxy_location` 필드로 클러스터의 모든 노드에서 프록시를 실행할 수 있습니다. 노드마다 프록시를 하나씩 두면 모든 프록시가 같은 포트에 바인딩되므로, 어느 노드로 요청을 보내도 모든 모델에 닿습니다. 그 앞에 자체 로드 밸런서를 둘 수도 있습니다. 이렇게 하면 HTTP·gRPC 인그레스는 노드를 추가해서, 모델 추론은 배포의 `num_replicas`로 레플리카를 늘려서 각각 수평 확장합니다. `proxy_location`을 설정 파일에 적는 방법은 Serve 설정 파일 구조를 설명할 때 함께 봅니다.

## 무작위 두 후보 중 덜 바쁜 쪽을 고르는 기본 요청 라우터와 교체 가능한 라우팅 정책

배포 안에서 요청을 받을 레플리카는 요청 라우터(request router)가 고릅니다. 기본 정책은 레플리카 둘을 무작위로 뽑아 덜 바쁜 쪽으로 보내는 Power of Two Choices이고, `RequestRouterConfig`로 배포마다 다른 라우터로 바꿀 수 있습니다.

### 모델을 고르는 인그레스 라우팅과 레플리카를 고르는 요청 라우팅

"라우팅"이라는 말은 두 수준에서 쓰입니다. 인그레스 라우팅은 모델 수준으로, 요청의 `model_id`를 보고 어느 배포로 보낼지 정합니다. 요청 라우팅은 레플리카 수준으로, 정해진 배포의 여러 레플리카 중 어디로 보낼지 정합니다. Summarizer 애플리케이션으로 보면, HTTP 경로로 Summarizer를 찾는 일이 앞의 수준에 해당합니다. 아래 예시는 Translator가 레플리카 네 개(`num_replicas=4`)로 실행된다고 가정합니다. 이때 Summarizer 레플리카 안의 Translator 핸들이 네 레플리카 중 하나를 고르는 일이 뒤의 수준입니다. 아래 내용은 모두 이 레플리카 선택에 관한 것입니다.

### 조율 비용이 거의 없는 Power of Two Choices

앞서 핸들이 power of two choices로 레플리카를 고른다고 했는데, 기본 라우터의 동작은 두 단계뿐입니다. 레플리카 두 개를 무작위로 샘플링하고, 그중 진행 중인 요청이 더 적은 쪽으로 보냅니다. 조율 오버헤드를 최소로 유지하면서도 부하가 고르게 퍼집니다.

가상의 숫자로 따라가 보면 이렇습니다. Translator 레플리카 A, B, C, D의 진행 중 요청이 각각 5, 2, 4, 1개인 순간에 새 요청이 들어왔고, 라우터가 A와 C를 뽑았다고 가정합니다. 이 경우 요청은 4개를 처리 중인 C로 갑니다. 가장 한가한 D를 놓치더라도, 가장 바쁜 레플리카가 이길 일은 없습니다. 전체 레플리카 상태를 모으거나 다른 라우터와 합의하지 않고도 이 결정이 나옵니다.

라우터는 레플리카를 고르는 일 말고도 주기적인 헬스 체크와 레플리카의 스케줄링 통계 수집을 맡고, 이 정보를 라우팅 결정에 씁니다. 라우팅 정책은 배포 단위로 정해지며, 그 배포로 만든 모든 핸들이 같은 정책을 씁니다.

### 레플리카가 늘수록 흩어지는 접두사 캐시

부하만 보는 정책은 LLM 서빙에서 약점을 드러냅니다. vLLM의 자동 접두사 캐싱(Automatic Prefix Caching, APC)은 이미 계산한 KV 캐시를 재사용하므로, 같은 시스템 프롬프트나 대화 이력을 가진 요청이 같은 레플리카에 모여야 이득을 봅니다. Anyscale 블로그 글은 8xL4 머신 8대(GPU 64개)에서 Anyscale RayTurbo로 DeepSeek-R1-Distill-Qwen-32B를 서빙하며 공유 접두사 데이터셋으로 측정했습니다. 이 측정에서 power of two 라우터는 레플리카 수가 늘수록 무작위 선택 때문에 KV 캐시 적중률이 빠르게 떨어졌고, 접두사 인식 라우터는 적중률을 일정하게 유지했습니다.

접두사 인식 라우팅(prefix-aware routing)을 맡는 Serve LLM의 라우터는 `PrefixCacheAffinityRouter`입니다. 비슷한 접두사를 가진 요청을 같은 레플리카로 보내 APC의 KV 캐시 적중률을 높이며, 다음 순서로 레플리카를 고릅니다.

1. 레플리카 간 큐 길이 차이가 임계값보다 작으면 부하가 균형이라고 보고 접두사 매칭을 씁니다.
2. 접두사 일치율이 10% 이상이면 가장 많이 일치하는 레플리카로 보냅니다.
3. 일치율이 10% 미만이면 캐시 사용률이 가장 낮은 레플리카로 보냅니다.
4. 부하가 불균형하면 Power of Two Choices로 돌아갑니다.

앞의 Anyscale 글은 이 방식의 내부를 이렇게 설명합니다. 라우터는 문자 단위 접두사 트리로 각 레플리카의 접두사 캐시를 근사합니다. 이후 요청이 오면 그 내용으로 트리를 조회해 공통 접두사가 가장 긴 레플리카로 보냅니다. 4단계의 폴백은 캐시를 위해 한 레플리카에 요청이 쏠리는 상황을 막는 안전장치입니다. Translator처럼 요청끼리 공유하는 접두사가 거의 없는 모델이라면 기본 라우터로 충분합니다.

### `RequestRouterConfig`로 바꾸는 라우터와 통계·재시도 주기

`RequestRouterConfig`는 Serve가 요청을 배포의 레플리카로 보내는 방식을 설정하는 pydantic 모델입니다. 이 API는 alpha 단계입니다. Python SDK에서는 배포 스키마에 `request_router_config` 필드가 있어서 이 설정을 배포마다 따로 둘 수 있습니다. 아래 코드는 두 가지 쓰임을 함께 보여 줍니다. 첫 번째는 기본 라우터를 그대로 두고 통계 수집 주기와 타임아웃만 바꾸는 설정입니다. 두 번째는 LLM 배포에 쓸 설정으로, `PrefixCacheAffinityRouter`로 교체하면서 불균형 임계값을 넘깁니다. Translator 배포에는 첫 번째 설정만 붙입니다.

```bash
pip install "ray[serve,llm]"
```

```python
from ray.serve.config import RequestRouterConfig

# 기본 라우터 유지, 통계 수집 주기와 타임아웃만 조정
stats_router_config = RequestRouterConfig(
    request_routing_stats_period_s=5.0,
    request_routing_stats_timeout_s=15.0,
)

# LLM 배포용 예시: 접두사 캐시 친화 라우터로 교체
prefix_router_config = RequestRouterConfig(
    request_router_class="ray.serve.llm.request_router.PrefixCacheAffinityRouter",
    request_router_kwargs={"imbalanced_threshold": 20},
)

# Translator 배포는 기본 라우터를 유지하고 통계 주기만 조정
translator_app = Translator.options(
    num_replicas=4,
    request_router_config=stats_router_config,
).bind()
```

`request_router_class`에는 문자열 경로나 클래스를 직접 넣을 수 있습니다. `request_router_kwargs`에 넣은 값은 라우터 클래스의 `initialize_state` 메서드로 전달되므로, 두 번째 설정의 `imbalanced_threshold`도 그 경로로 라우터에 들어갑니다. 나머지 필드와 기본값은 다음과 같습니다.

| 필드 | 기본값과 의미 |
|---|---|
| `request_router_class` | `PowerOfTwoChoicesRequestRouter`입니다. |
| `request_routing_stats_period_s` | 10초입니다. 레플리카의 라우팅 통계를 기록하는 호출 간격이며, 기본 기록은 아무 일도 하지 않는 액터 호출입니다. |
| `request_routing_stats_timeout_s` | 30초입니다. 통계 메서드가 이 시간 안에 반환하지 않으면 실패로 봅니다. |
| `initial_backoff_s` | 0.025초입니다. 레플리카로 라우팅을 다시 시도하기 전 첫 대기 시간입니다. |
| `backoff_multiplier` | 2입니다. 재시도마다 대기 시간에 곱하는 배수입니다. |
| `max_backoff_s` | 0.5초입니다. 재시도 사이 대기 시간의 상한입니다. |

기본값으로 직접 곱해 보면 0.025, 0.05, 0.1, 0.2, 0.4초이고, 그다음부터는 0.5초입니다. 통계 기록의 내용은 배포에 `record_routing_stats` 메서드를 정의해 직접 채울 수 있습니다. 사용자 정의 라우터가 레플리카별 지표를 써야 한다면 이 메서드로 그 지표를 내보내면 됩니다.

### `RequestRouter`를 확장한 사용자 정의 라우터

직접 정책을 만들려면 `RequestRouter` 기반 클래스를 확장합니다. 구현할 핵심 메서드는 셋입니다. `choose_replicas()`는 요청을 처리할 레플리카를 고르고, `on_request_routed()`는 라우팅이 끝난 뒤 라우터 상태를 갱신하며, `on_replica_actor_died()`는 레플리카가 죽었을 때 관련 상태를 정리합니다. 이 메서드들이 불리는 순서는 라우터 수명 주기를 따릅니다. 라우터는 레플리카 목록과 함께 만들어지고, 요청마다 `choose_replicas()`가 호출되며, 라우팅에 성공하면 `on_request_routed()`가 호출됩니다. 레플리카가 죽으면 `on_replica_actor_died()`가 호출되고, 배포가 삭제되면 라우터도 정리됩니다. 구현에 쓰는 타입으로 `ReplicaID`, `PendingRequest`, `RunningReplica`가 함께 공개되어 있습니다.

라우터 메서드는 비동기로 작성해야 성능이 나옵니다. 여기서 상태 액터(`state_actor`)는 라우터가 메트릭을 조회하는 공유 액터를 말하며, 아래에서 다룰 중앙 저장소 방식의 싱글턴 액터가 그 예입니다. 예를 들어 상태 액터에서 값을 읽을 때는 `await self.state_actor.get.remote()`처럼 기다려야 합니다. `ray.get(...)`으로 감싸면 그 호출이 라우터를 블록하므로 피해야 합니다. 라우터가 내부 상태를 가진다면 `__init__`에서 `asyncio.Lock()`을 만들어 두고, `choose_replicas()` 안에서 `async with self.lock:`으로 공유 상태 갱신을 감싸는 패턴을 씁니다. 세 메서드와 락 패턴을 한 클래스에 모으면 골격은 이렇게 됩니다.

```python
# excerpt: 사용자 정의 라우터 골격
import asyncio

from ray.serve.request_router import RequestRouter


class TranslatorRouter(RequestRouter):
    def __init__(self):
        self.lock = asyncio.Lock()  # 비동기 코드용 락
        self.state = {}

    async def choose_replicas(self, *args, **kwargs):
        async with self.lock:  # 공유 상태 보호
            # 상태를 갱신하고 요청을 받을 레플리카를 골라 반환
            raise NotImplementedError

    async def on_request_routed(self, *args, **kwargs):
        # 라우팅 성공 후 라우터 상태 갱신
        raise NotImplementedError

    async def on_replica_actor_died(self, *args, **kwargs):
        # 죽은 레플리카의 상태 정리
        raise NotImplementedError
```

흔한 기능은 믹스인으로 붙일 수 있습니다. `LocalityMixin`은 같은 노드의 레플리카를 선호해 네트워크 지연을 줄입니다. `MultiplexMixin`은 LoRA 배포에서 레플리카마다 어떤 모델이 올라가 있는지 추적하고, `FIFOMixin`은 요청의 FIFO 순서를 보장합니다.

### 여러 라우터가 상태를 나누는 두 방식

라우터 로직은 배포 핸들을 소유한 프로세스에서 실행됩니다. 그래서 Summarizer 레플리카가 셋이면 Translator용 라우터도 셋이 각자 결정을 내립니다. 이 라우터들이 같은 정보를 보게 하는 방식은 두 가지입니다. 첫 번째는 중앙 저장소 방식입니다. 싱글턴 액터 하나를 메트릭 저장소로 두고, 각 라우터가 `choose_replicas()`와 `on_request_routed()`에서 그 액터의 원자적 `get()`/`set()`을 호출합니다. 두 번째는 브로드캐스트 방식입니다. Serve 컨트롤러가 각 레플리카의 로컬 통계를 폴링해 모든 라우터에 뿌리고, 레플리카는 요청을 받을 때마다 자기 통계를 갱신해 다음 폴링 때 돌려줍니다.

| 구분 | 중앙 싱글턴 저장소 | 컨트롤러 브로드캐스트 |
|---|---|---|
| 일관성 | 강한 일관성이며 메트릭이 즉시 반영됩니다. | 최종적 일관성이라 조금 오래된 데이터로 결정할 수 있습니다. |
| 요청당 비용 | 라우팅 결정마다 네트워크 홉이 하나 더 생깁니다. | 결정당 추가 RPC가 없습니다. |
| 처리량 한계 | 초당 수천 요청 규모에서는 단일 액터가 병목이 되어 time to first token(TTFT, 첫 토큰까지의 지연)에 영향을 줄 수 있습니다. | 더 높은 처리량으로 확장됩니다. |
| 구현 | 배포 로직을 고치지 않아도 되어 단순합니다. | 컨트롤러와 조율해야 해서 복잡합니다. |
| 맞는 상황 | 강한 일관성이 필요하고 처리량이 중간 수준일 때입니다. | 매우 높은 처리량이 필요하고 최종적 일관성을 감수할 수 있을 때입니다. |

처음에는 중앙 저장소로 시작해서 정책이 효과를 내는지 확인하는 편이 안전합니다. 라우팅 RPC가 TTFT에 드러나기 시작하면 그때 `record_routing_stats` 기반의 브로드캐스트 방식으로 옮기면 됩니다.

## 세 군데만 고쳐 로컬 HTTP로 여는 번역 모델 배포

이미 있는 모델 클래스에 Serve 모듈 가져오기, 데코레이터, `bind`를 더하고 HTTP 요청을 받을 진입점(`__call__` 메서드나 FastAPI 라우트)을 두면 Serve 배포가 되고, `serve run` 한 줄로 로컬 HTTP 엔드포인트가 열립니다. Serve와 함께 번역 예제가 쓰는 라이브러리도 설치합니다.

```bash
pip install "ray[serve]" transformers torch requests
```

### 일반 Python 클래스에서 배포로 바뀌는 지점

출발점은 t5-small 토크나이저와 모델을 생성자에서 불러오고, `translate` 메서드에서 입력 앞에 `"translate English to French: "`를 붙여 번역하는 평범한 `Translator` 클래스입니다. 여기에 Serve 모듈을 가져오고, `@serve.deployment` 데코레이터를 붙이고, 생성자 인자에 `bind`하면 배포로 바뀝니다. HTTP 요청을 받는 `__call__`도 하나 추가합니다. 앞서 본 인그레스 배포의 규칙대로 이 메서드가 Starlette 요청 객체를 받고, 반환값은 HTTP 응답 본문으로 나갑니다. 아래 스크립트를 `serve_quickstart.py`로 저장합니다.

```python
# File name: serve_quickstart.py
from starlette.requests import Request

import ray
from ray import serve

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class Translator:
    def __init__(self):
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    def translate(self, text: str) -> str:
        # Run inference
        input_ids = self.tokenizer(
            f"translate English to French: {text}", return_tensors="pt"
        ).input_ids
        output_ids = self.model.generate(
            input_ids, num_beams=4, early_stopping=True, max_length=300
        )

        # Post-process output to return only the translation text
        translation = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return translation

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        return self.translate(english_text)

translator_app = Translator.bind()
```

데코레이터 인자 `num_replicas`와 `ray_actor_options`는 배포의 복사본 수와 복사본마다 필요한 자원을 정합니다. `num_cpus`는 레플리카가 예약할 논리 CPU 수입니다. 분수로 주면 CPU 수보다 많은 레플리카를 한 머신에 올릴 수 있습니다. 생성자가 인자를 받지 않으므로 `bind()`는 빈 호출로 끝납니다. 이 인자들의 전체 목록과 적용 우선순위는 배포 파라미터를 다루는 부분에서 정리합니다.

### `serve run`으로 띄우고 POST로 확인하는 로컬 실행

`serve run`은 `module:application` 형식의 import 경로를 받습니다. 그래서 스크립트가 있는 디렉터리에서 실행해야 합니다.

```bash
$ serve run serve_quickstart:translator_app
```

이 명령은 애플리케이션을 띄운 뒤 콘솔로 로그를 계속 보여 주며 터미널을 붙잡고 있습니다. `Ctrl-C`를 누르면 애플리케이션이 내려갑니다. 기본 접근 주소는 `http://127.0.0.1:8000/`입니다. 다른 터미널에서 영어 문장을 JSON 본문에 담아 POST로 보내면, 앞의 `__call__`이 본문을 풀어 `translate`에 넘깁니다.

```python
# File name: model_client.py
import requests

english_text = "Hello world!"

response = requests.post("http://127.0.0.1:8000/", json=english_text)
french_text = response.text

print(french_text)
```

클라이언트를 실행하면 이런 응답이 돌아옵니다.

```text
$ python model_client.py

Bonjour monde!
```

### 같은 배포 위에 얹는 FastAPI 라우팅

원시 Starlette 요청을 직접 다루지 않으려면 FastAPI 앱을 `@serve.ingress(app)`로 배포에 감쌉니다. 이렇게 하면 `__call__` 대신 경로 데코레이터를 붙인 메서드가 요청을 받습니다. FastAPI 인그레스로 옮긴 전체 파일은 아래에 있습니다.

```python
# File name: model.py
import ray
from ray import serve
from fastapi import FastAPI

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI()

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
@serve.ingress(app)
class Translator:
    def __init__(self):
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    @app.post("/")
    def translate(self, text: str) -> str:
        # Run inference
        input_ids = self.tokenizer(
            f"translate English to French: {text}", return_tensors="pt"
        ).input_ids
        output_ids = self.model.generate(
            input_ids, num_beams=4, early_stopping=True, max_length=300
        )

        # Post-process output to return only the translation text
        translation = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return translation

translator_app = Translator.bind()
```

이 형태에서 `text: str`은 FastAPI가 파싱하는 쿼리 파라미터가 됩니다. 그래서 클라이언트는 본문 대신 `params`로 값을 보내고, 응답은 `.json()`으로 읽습니다. 파일을 `model.py`로 저장했다면 `serve run model:translator_app`으로 띄웁니다.

```python
import requests

response = requests.post("http://127.0.0.1:8000/", params={"text": "Hello world!"})
french_text = response.json()

print(french_text)
```

### 레지스트리에서 한 번 읽어 두는 워밍업 로드

Serve 배포는 임의의 Python 코드를 실행합니다. 따라서 모델 레지스트리의 표준 Python 클라이언트로 모델을 직접 불러오고, 로드와 추론을 한 배포 안에 함께 둘 수 있습니다. 프레임워크 전용 어댑터나 별도 직렬화 형식도 필요 없습니다. 아래 MLflow 예제는 번역 모델이 아닌 별도의 scikit-learn 회귀 모델(랜덤 포레스트)로 레지스트리 로드 패턴을 보여 줍니다. `production` 태그가 붙은 최신 모델을 `__init__`에서 한 번만 불러오고, 더미 입력으로 예측을 한 번 돌려 둡니다. 이런 워밍업 로드(warm start)는 첫 요청의 지연 급증을 없앱니다. 이 예제를 돌리려면 MLflow와 scikit-learn을 더 설치합니다.

```bash
pip install mlflow scikit-learn
```

```python
from ray import serve
import mlflow.pyfunc
import numpy as np

@serve.deployment
class MLflowModelDeployment:
    def __init__(self):
        # Search for models with production tag
        models = mlflow.search_logged_models(
            filter_string="tags.production='true' AND name='sk-learn-random-forest-reg-model'",
            order_by=[{"field_name": "creation_time", "ascending": False}],
        )
        if models.empty:
            raise ValueError("No model with production tag found")

        # Get the most recent production model
        model_row = models.iloc[0]
        artifact_location = model_row["artifact_location"]

        # Load model once during initialization (warm-start)
        self.model = mlflow.pyfunc.load_model(artifact_location)

        # Pre-warm the model with a dummy prediction
        dummy_input = np.zeros((1, 4))
        _ = self.model.predict(dummy_input)

    async def __call__(self, request):
        data = await request.json()
        features = np.array(data["features"])

        # MLflow validates input against the logged signature automatically
        prediction = self.model.predict(features)
        return {"prediction": prediction.tolist()}

app = MLflowModelDeployment.bind()
```

이 검색이 결과를 찾으려면 학습 쪽에서 미리 준비해 둘 것이 있습니다. 학습을 마친 뒤 `mlflow.set_logged_model_tags`로 `production` 태그를 `"true"`로 달아야 합니다. 모델을 로깅할 때는 `mlflow.models.infer_signature`로 만든 시그니처를 함께 남깁니다. 그러면 서빙 중에 입력이 이 시그니처로 자동 검증됩니다. `pip_requirements`에는 핵심 라이브러리 버전을 고정해 넣습니다. 그래야 학습, 평가, 서빙 환경에서 모델이 똑같이 동작합니다. scikit-learn이라면 전처리 단계까지 포함한 `Pipeline` 전체를 로깅해 학습과 서빙의 변환을 맞춥니다.

`Translator`에 이 패턴을 옮기려면 생성자의 `from_pretrained` 호출을 레지스트리 로드로 바꾸면 됩니다. 로드는 생성자에서 한 번만 하고, 요청 처리 경로에는 `translate`의 추론만 남깁니다.

## 핸들 호출로 이어 붙이고 따로 늘리는 요약·번역 파이프라인

배포를 다른 배포의 `bind` 인자로 넘기면 런타임에 그 인자가 `DeploymentHandle`로 바뀝니다. 핸들 호출은 결과 대신 `DeploymentResponse`를 즉시 돌려줍니다. 그래서 배포 안에서는 이 응답을 `await`로 기다리거나, 기다리지 않고 다음 핸들 호출에 그대로 넘겨야 합니다.

### 생성자 인자에서 핸들로 바뀌는 하위 배포

번역기 앞에 요약 단계를 붙여 보겠습니다. 영어 텍스트를 받아 두 번째 t5-small 모델로 요약하고, 요약문을 `Translator`에 넘겨 프랑스어로 바꾸는 두 단계 파이프라인입니다. 요청을 먼저 받는 쪽이 `Summarizer`이므로 `__call__`은 `Summarizer`에 둡니다. `Translator` 핸들은 생성자 인자로 받습니다. 발췌에서 생략한 `Translator`는 앞의 `Translator` 클래스를 조금 고친 버전입니다. 데코레이터는 인자 없는 `@serve.deployment`이고, 번역은 `translate(self, text: str)` 메서드가 맡습니다.

```python
# excerpt: serve_quickstart_composed.py 중 Summarizer와 애플리케이션 정의 (translate 메서드를 둔 Translator 배포는 생략)
from starlette.requests import Request

from ray import serve
from ray.serve.handle import DeploymentHandle

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

@serve.deployment
class Summarizer:
    def __init__(self, translator: DeploymentHandle):
        self.translator = translator

        # Load model.
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    def summarize(self, text: str) -> str:
        # Run inference
        input_ids = self.tokenizer(f"summarize: {text}", return_tensors="pt").input_ids
        output_ids = self.model.generate(
            input_ids, num_beams=4, early_stopping=True, max_length=15
        )

        # Post-process output to return only the summary text
        summary = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return summary

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        summary = self.summarize(english_text)

        translation = await self.translator.translate.remote(summary)
        return translation

app = Summarizer.bind(Translator.bind())
```

마지막 줄을 안쪽부터 읽으면 구조가 보입니다. `Translator`를 빈 생성자 인자에 바인딩하고, 바인딩된 `Translator`를 `Summarizer`의 생성자 인자로 넘깁니다. 코드에서는 바인딩된 배포 객체를 넘기지만, `Summarizer.__init__`이 실제로 받는 값은 핸들입니다. 실행 방법은 단일 배포 때와 같습니다. 스크립트가 있는 디렉터리에서 import 경로를 지정합니다.

```bash
$ serve run serve_quickstart_composed:app
```

클라이언트도 앞서 쓴 `model_client.py`에서 입력 문장만 길어집니다.

```python
# File name: composed_client.py
import requests

english_text = (
    "It was the best of times, it was the worst of times, it was the age "
    "of wisdom, it was the age of foolishness, it was the epoch of belief"
)
response = requests.post("http://127.0.0.1:8000/", json=english_text)
french_text = response.text

print(french_text)
```

`serve run`이 떠 있는 동안 다른 터미널에서 실행하면 요약을 거친 번역문이 돌아옵니다.

```text
$ python composed_client.py

C'était le meilleur des temps, c'était le pire des temps,
```

Spheron 블로그(서드파티)는 클러스터 안에서 끝나는 호출이 HTTP 왕복을 거치지 않으므로 직렬화 오버헤드와 서비스 간 지연이 줄어든다고 설명합니다.

### 결과 대신 즉시 돌아오는 `DeploymentResponse`

`self.translator.translate.remote(summary)`는 네 부분으로 나뉩니다. `self.translator`는 생성자로 받은 핸들이고, `translate`는 호출할 대상 배포의 메서드입니다. `.remote`는 다른 배포를 부르는 핸들 호출이라는 표시이고, `summary`는 메서드 인자입니다. 이 자리에는 위치 인자와 키워드 인자를 몇 개든 넣을 수 있습니다. 호출은 `Translator`의 `translate`를 비동기로 실행시키고 `DeploymentResponse`를 곧바로 반환합니다. 이 응답에 `await`를 걸어야 원격 호출이 끝난 뒤 반환값을 받습니다.

호출과 기다림이 분리되어 있으므로, 서로 독립인 하위 호출은 모두 먼저 걸어 두고 나중에 기다리는 식으로 쓸 수 있습니다. 공식 예제의 `Ingress`가 그렇게 합니다. `Hello`와 `World` 핸들을 차례로 호출해 두 응답을 받아 두고, 그다음에 두 응답을 `await`해 이어 붙입니다.

```python
# excerpt: 두 하위 호출을 먼저 걸고 나중에 기다리는 Ingress.__call__
async def __call__(self) -> str:
    hello_response = self._hello_handle.remote()
    world_response = self._world_handle.remote()
    return (await hello_response) + (await world_response)
```

응답의 `.result()` 메서드로도 반환값을 얻을 수 있습니다. 하지만 배포 안에서 `.result()`를 부르면 원격 호출이 끝날 때까지 그 배포는 다른 코드를 실행하지 못합니다. `await`를 쓰면 기다리는 동안에도 다른 요청을 처리합니다. 배포 바깥은 사정이 다릅니다. `serve.run(app)`이 돌려주는 핸들로 인그레스 배포를 부르는 테스트 코드라면 `handle.remote().result()`처럼 동기로 결과를 받아도 됩니다.

### `await` 없이 다음 단계로 넘기는 체이닝

`DeploymentResponse`는 다른 핸들 호출의 인자로 그대로 넘길 수 있습니다. 첫 응답을 직접 `await`할 필요는 없고, 기다리는 일은 Serve가 내부에서 처리합니다. 첫 호출이 끝나면 두 번째 호출이 받는 값은 응답 객체가 아니라 첫 호출의 출력입니다. 아래 `chain.py`에서 `Multiplier`의 `val`은 `DeploymentResponse`가 아니라 `Adder`가 계산한 정수입니다.

```python
# File name: chain.py
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse

@serve.deployment
class Adder:
    def __init__(self, increment: int):
        self._increment = increment

    def __call__(self, val: int) -> int:
        return val + self._increment

@serve.deployment
class Multiplier:
    def __init__(self, multiple: int):
        self._multiple = multiple

    def __call__(self, val: int) -> int:
        return val * self._multiple

@serve.deployment
class Ingress:
    def __init__(self, adder: DeploymentHandle, multiplier: DeploymentHandle):
        self._adder = adder
        self._multiplier = multiplier

    async def __call__(self, input: int) -> int:
        adder_response: DeploymentResponse = self._adder.remote(input)
        # Pass the adder response directly into the multiplier (no `await` needed).
        multiplier_response: DeploymentResponse = self._multiplier.remote(
            adder_response
        )
        # `await` the final chained response.
        return await multiplier_response

app = Ingress.bind(
    Adder.bind(increment=1),
    Multiplier.bind(multiple=2),
)

handle: DeploymentHandle = serve.run(app)
response = handle.remote(5)
assert response.result() == 12, "(5 + 1) * 2 = 12"
```

이렇게 하면 인그레스가 손에 쥐고 기다리는 응답은 마지막 하나뿐입니다. 응답 여러 개를 한 호출에 함께 넘길 수도 있습니다. 공식 예제는 `Adder.bind(increment=1)`과 `Adder.bind(increment=2)` 두 응답을 `Combiner.average`에 함께 넘겨, 입력 100.0에 대해 `{"result": 101.5}`를 받습니다.

요약·번역 파이프라인에서는 요약이 `Summarizer` 안의 지역 메서드라 넘길 응답이 없습니다. 요약을 별도 배포로 떼어 내면 같은 방식으로 요약 응답을 번역 핸들에 바로 넘길 수 있습니다.

### 제너레이터를 끝까지 소비하는 스트리밍 호출

출력을 여러 번에 나눠 돌려주는 메서드도 핸들로 호출할 수 있습니다. 조건은 두 가지입니다. 대상 메서드가 제너레이터여야 하고, 핸들에 `handle.options(stream=True)`를 설정해야 합니다. 이때 호출은 단일 `DeploymentResponse` 대신 `DeploymentResponseGenerator`를 반환합니다.

```python
# File name: stream.py
from typing import AsyncGenerator, Generator

from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator

@serve.deployment
class Streamer:
    def __call__(self, limit: int) -> Generator[int, None, None]:
        for i in range(limit):
            yield i

@serve.deployment
class Caller:
    def __init__(self, streamer: DeploymentHandle):
        self._streamer = streamer.options(
            # Must set `stream=True` on the handle, then the output will be a
            # response generator.
            stream=True,
        )

    async def __call__(self, limit: int) -> AsyncGenerator[int, None]:
        # Response generator can be used in an `async for` block.
        r: DeploymentResponseGenerator = self._streamer.remote(limit)
        async for i in r:
            yield i

app = Caller.bind(Streamer.bind())

handle: DeploymentHandle = serve.run(app).options(
    stream=True,
)

# Response generator can also be used as a regular generator in a sync context.
r: DeploymentResponseGenerator = handle.remote(10)
assert list(r) == list(range(10))
```

`DeploymentResponseGenerator`는 동기 제너레이터로도, 비동기 제너레이터로도 쓸 수 있습니다. 그래도 배포 안에서는 `async for`로 소비해야 합니다. 동기 제너레이터로 쓰면 `.result()`처럼 같은 레플리카의 다른 요청이 동시에 실행되지 못합니다. 위 코드도 이 구분을 따릅니다. 배포 바깥의 테스트 코드만 `list(r)`로 동기 소비합니다. 그리고 단일 응답과 달리 `DeploymentResponseGenerator`는 다른 핸들 호출의 인자로 넘길 수 없습니다. 스트리밍 단계 뒤에 다른 배포를 이어 붙이려면 호출하는 쪽에서 항목을 받아 직접 넘겨야 합니다.

### Ray 태스크로 넘기는 `ObjectRef`

핸들 호출을 일반 Ray 액터나 태스크와 섞어야 할 때는 응답을 Ray `ObjectRef`로 바꿉니다. 이때 쓰는 함수는 개발자 API `DeploymentResponse._to_object_ref`와 `DeploymentResponse._to_object_ref_sync`입니다. 아래 `response_to_object_ref.py`의 `Ingress`는 `SayHi` 배포의 응답을 `ObjectRef`로 바꿔 `@ray.remote` 태스크 `say_hi_task`에 넘기고, 태스크의 결과를 `await`합니다.

```python
# File name: response_to_object_ref.py
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse

@ray.remote
def say_hi_task(inp: str):
    return f"Ray task got message: '{inp}'"

@serve.deployment
class SayHi:
    def __call__(self) -> str:
        return "Hi from Serve deployment"

@serve.deployment
class Ingress:
    def __init__(self, say_hi: DeploymentHandle):
        self._say_hi = say_hi

    async def __call__(self):
        # Make a call to the SayHi deployment and pass the result ref to
        # a downstream Ray task.
        response: DeploymentResponse = self._say_hi.remote()
        response_obj_ref: ray.ObjectRef = await response._to_object_ref()
        final_obj_ref: ray.ObjectRef = say_hi_task.remote(response_obj_ref)
        return await final_obj_ref

app = Ingress.bind(SayHi.bind())
handle: DeploymentHandle = serve.run(app)
assert handle.remote().result() == "Ray task got message: 'Hi from Serve deployment'"
```

배포 안에서는 여기서도 비동기 버전에 `await`를 붙여 씁니다.

### 단계마다 따로 정하는 레플리카 수와 자원

조합의 실익은 확장에서 나옵니다. 요약과 번역이 별도 배포이므로 `autoscaling_config`, `num_replicas`, `num_cpus`, `num_gpus`를 배포마다 따로 정해 각자 확장할 수 있습니다. 공식 예제의 언어 분류 애플리케이션으로 보면 이렇습니다. 요청의 75%가 스페인어이고 25%가 프랑스어라면 `SpanishResponder`는 레플리카 3개, `FrenchResponder`는 1개로 둘 수 있습니다. 이 예제도 같은 방식으로 띄웁니다. `hello.py`가 있는 디렉터리에서 `serve run hello:language_classifier`를 실행하고, `{"language": "spanish", "name": "Dora"}`를 POST로 보내면 `Hola Dora`가 출력됩니다.

요약·번역 파이프라인에서도 요약 모델과 번역 모델의 처리 시간이나 GPU 필요량이 다르면 두 데코레이터에 서로 다른 값을 주면 됩니다. 한쪽을 늘리려고 다른 쪽까지 복제할 필요가 없습니다. 이 파라미터들의 기본값과 우선순위는 배포 파라미터를 다루는 부분에서 정리합니다.

## 설정 파일이 코드를 덮어쓰는 배포 파라미터와 처리량을 정하는 기본값

배포 파라미터를 지정하는 곳은 세 군데입니다. 두 곳 이상에 값이 있으면 파라미터마다 Serve 설정 파일, 애플리케이션 코드(`@serve.deployment` 또는 `.options()`), Serve 기본값 순으로 우선합니다.

### 세 곳에서 지정하는 같은 파라미터

배포를 처음 정의할 때는 데코레이터에 파라미터를 적습니다. 아래는 `Translator`에 레플리카 수, 자원, 동시성, 헬스 체크, 종료 파라미터를 모두 적은 예입니다.

```python
# excerpt: 데코레이터에 배포 파라미터를 모은 Translator 정의 (클래스 본문은 앞서 정의한 그대로)
from ray import serve

@serve.deployment(
    name="Translator",
    num_replicas=2,
    ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
    max_ongoing_requests=100,
    health_check_period_s=10,
    health_check_timeout_s=30,
    graceful_shutdown_timeout_s=20,
    graceful_shutdown_wait_loop_s=2,
)
class Translator:
    # __init__과 __call__은 앞서 정의한 번역 로직
```

이미 정의한 배포는 `.options()`로 일부 값만 바꿔 다시 쓸 수 있습니다.

```python
# excerpt: 정의를 재사용하면서 자원만 바꾸는 바인딩
translator_app = Translator.options(
    ray_actor_options={"num_cpus": 0.2, "num_gpus": 0.0}
).bind()
```

프로덕션에서는 배포 설정을 설정 파일에 두는 편이 좋습니다. 그러면 코드를 고치지 않고 값을 바꿀 수 있습니다. 앞의 `Translator` 코드를 `configure_serve.py`로 저장했다고 하면, 같은 값을 YAML로 옮긴 파일은 이런 모양입니다.

```yaml
applications:
- name: app1
  import_path: configure_serve:translator_app
  deployments:
  - name: Translator
    num_replicas: 2
    max_ongoing_requests: 100
    graceful_shutdown_wait_loop_s: 2.0
    graceful_shutdown_timeout_s: 20.0
    health_check_period_s: 10.0
    health_check_timeout_s: 30.0
    ray_actor_options:
      num_cpus: 0.2
      num_gpus: 0.0
```

`deployments` 항목은 코드에 지정한 배포 옵션을 덮어쓰려고 둡니다.

### 파라미터 단위로 적용되는 우선순위

우선순위는 파라미터 하나하나에 따로 적용됩니다. 코드에 `@serve.deployment(num_replicas=2, graceful_shutdown_timeout_s=6)`을 두고 설정 파일에 `num_replicas: 5`만 적었다고 하면, 적용되는 값은 다음 표와 같습니다.

| 파라미터 | 코드 | 설정 파일 | 적용 값 |
|---|---|---|---|
| `num_replicas` | 2 | 5 | 5 |
| `graceful_shutdown_timeout_s` | 6 | 없음 | 6 |
| `health_check_period_s` | 없음 | 없음 | 10 (기본값) |

딕셔너리 파라미터는 다르게 동작합니다. `ray_actor_options`는 설정 하나로 취급되므로, 설정 파일에 이 딕셔너리가 있으면 코드의 딕셔너리 전체가 무시됩니다. 코드에만 적은 `num_gpus`나 `runtime_env`는 지정하지 않은 값으로 보고 기본값을 씁니다. `user_config`와 `autoscaling_config`도 같은 방식으로 덮어씁니다. 그래서 설정 파일에 `ray_actor_options`를 쓸 때는 필요한 키를 빠짐없이 다시 적어야 합니다.

### 레플리카 수와 자원 배치

`num_replicas`의 기본값은 1입니다. 양의 정수로 주면 레플리카 수가 고정되고, `auto`로 주면 기본 설정으로 오토스케일링합니다. `autoscaling_config`를 설정하면 `num_replicas`를 숫자로 줄 수 없습니다. 설정 파일 스키마의 검증기도 이 조합에 "Manually setting num_replicas is not allowed when autoscaling_config is provided." 오류를 냅니다. 반대로 `auto`나 null은 `autoscaling_config`와 함께 쓸 수 있습니다.

`ray_actor_options`에 넣을 수 있는 키는 `accelerator_type`, `memory`, `num_cpus`, `num_gpus`, `object_store_memory`, `resources`, `runtime_env`입니다. 레플리카 하나가 여러 자원 묶음을 차지해야 하면 `placement_group_bundles`를 씁니다. 레플리카 액터는 첫 번째 번들에 스케줄되므로 `ray_actor_options`의 자원은 첫 번들 자원의 부분집합이어야 합니다. 이 필드를 쓰면 `max_replicas_per_node`는 함께 설정할 수 없습니다.

### 동시성 상한과 대기열이 만드는 백프레셔

`max_ongoing_requests`는 응답을 받지 않은 채 레플리카 하나에 보낼 수 있는 요청 수이며, 기본값은 앞서 데이터 경로에서 본 그대로입니다. 이 상한은 HTTP 요청과 `DeploymentHandle` 호출자를 모두 합쳐 적용됩니다. 상한을 넘은 요청은 앞서 본 요청 수명 주기대로 큐에서 기다립니다.

이 큐의 크기를 정하는 값이 `max_queued_requests`입니다. 호출자(프록시 또는 `DeploymentHandle`)마다 쌓을 수 있는 요청 수이고, 기본값은 -1로 제한이 없습니다. 한도에 도달하면 이후 요청에 대해 핸들은 `BackPressureError`를 내고, HTTP는 기본적으로 503을 반환합니다. `backpressure_config`로 상태 코드를 429로 바꾸고 `Retry-After` 헤더를 붙일 수 있습니다.

예를 들어 레플리카 2개에 기본값 5를 쓰면 동시에 처리되는 요청은 많아야 10개입니다. 11번째 요청부터는 호출자 큐에서 기다리므로, 큐 한도가 없으면 부하가 몰릴수록 대기 시간이 계속 늘어날 수 있습니다. 응답 시간 목표가 있는 서비스라면 큐 한도를 정해 빨리 거절하는 편이 낫습니다. 반대로 번역처럼 요청 하나가 레플리카를 오래 붙잡지 않는 모델은 `max_ongoing_requests`를 올려 처리량을 늘릴 수 있습니다. 앞서 데코레이터 예시에서 `max_ongoing_requests`를 기본값보다 크게 준 것이 이런 선택입니다.

### 헬스 체크와 정상 종료

컨트롤러는 `health_check_period_s`(기본 10초)마다 레플리카의 상태를 확인합니다. 기본 헬스 체크는 아무 일도 하지 않는 액터 호출이고, 배포에 `check_health` 메서드를 정의해 비정상일 때 예외를 던지게 만들 수 있습니다. 이 메서드가 `health_check_timeout_s`(기본 30초) 안에 반환하지 않으면 실패로 봅니다. 종료할 때 두 값은 함께 동작합니다. 레플리카는 `graceful_shutdown_wait_loop_s`(기본 2초) 동안 처리 중인 작업이 없어지기를 기다리고, 컨트롤러는 `graceful_shutdown_timeout_s`가 지나면 남은 작업과 상관없이 레플리카를 강제로 종료합니다.

### 클러스터 단위로 조절하는 용량

설정 파일의 `target_capacity`는 클러스터 전체 레플리카의 목표 용량을 백분율로 지정합니다. 각 배포의 `num_replicas`, `min_replicas`, `max_replicas`, `initial_replicas`가 이 비율만큼 조정됩니다. 레플리카 수를 Serve 밖에서 관리하려면 애플리케이션에 `external_scaler_enabled`를 켭니다. 켜면 그 애플리케이션의 어떤 배포에서도 Serve 내장 오토스케일링을 쓸 수 없습니다.

### 같은 옵션을 받는 Ray Serve LLM의 `LLMConfig`

Ray Serve LLM의 `LLMConfig`에서 필수 필드는 `model_loading_config` 하나뿐이고, 나머지 필드에는 모두 기본값이 있습니다. 그 안의 `model_id`는 사용자가 API 요청의 `model`로 보내는 값입니다. 배포 옵션은 `deployment_config`에 담습니다. 이 필드는 표준 `@serve.deployment` 옵션을 받습니다. 지원하는 키는 `name`, `num_replicas`, `ray_actor_options`, `max_ongoing_requests`, `autoscaling_config`, `max_queued_requests`, `user_config` 등입니다. 아래는 오토스케일링과 동시성 상한을 함께 준 LLM 배포 설정입니다. `accelerator_type`은 `deployment_config` 밖에 있는 최상위 필드입니다.

```python
from ray.serve.llm import LLMConfig

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-0.5b",
        model_source="Qwen/Qwen2.5-0.5B-Instruct",
    ),
    accelerator_type="A10G",
    deployment_config=dict(
        autoscaling_config=dict(min_replicas=1, max_replicas=4),
        max_ongoing_requests=64,
    ),
)
```

YAML 설정에서도 이 필드들이 키와 그대로 대응합니다. `engine_kwargs`의 `tensor_parallel_size` 같은 엔진 인자도 같은 파일에 함께 적어 `serve run config.yaml`로 배포합니다.

## 코드에서 만들어 클러스터 목표 상태로 적용하는 Serve 설정 파일

개발 중에는 `serve run`으로 고치고 다시 띄우기를 반복합니다. 프로덕션으로 넘어갈 때는 애플리케이션의 단일 진실 공급원(single source of truth)이 될 구조화된 설정 파일을 만듭니다. 이 파일 하나에 프록시 같은 시스템 수준 설정과 개별 배포 파라미터를 함께 적습니다. 이후에는 파일만 고쳐 다시 적용하면 재배포나 재시작 없이 배포 파라미터가 바뀝니다.

### `serve build`가 만드는 초안

앞서 만든 요약·번역 조합 애플리케이션을 `text_ml.py`에 `app = Summarizer.bind(Translator.bind())`로 저장했다고 하겠습니다. `serve build`는 이 import path를 받아 코드에 있는 모든 배포와 그 파라미터를 담은 설정 파일을 생성합니다. `-o`로 출력 경로를 지정하고, 생략하면 결과가 STDOUT으로 나옵니다.

```bash
serve build text_ml:app -o serve_config.yaml
```

생성된 파일의 최상위 키는 `proxy_location`, `http_options`, `grpc_options`, `logging_config`, `applications` 다섯 개입니다. 애플리케이션 항목에는 `import_path`, `runtime_env`, 배포별 옵션이 들어갑니다. `torch`와 `transformers`가 전역으로 설치되어 있지 않다면 `runtime_env`에 직접 넣어야 합니다. `runtime_env`에 두 패키지를 넣은 `serve_config.yaml`은 대략 이런 모양이 됩니다.

```yaml
proxy_location: EveryNode
http_options:
  host: 0.0.0.0
  port: 8000
applications:
- name: default
  route_prefix: /
  import_path: text_ml:app
  runtime_env:
    pip:
    - torch
    - transformers
  deployments:
  - name: Translator
    num_replicas: 1
    user_config:
      language: french
  - name: Summarizer
    num_replicas: 1
```

`runtime_env`의 `working_dir`와 `py_modules`에는 원격 URI만 쓸 수 있고, 로컬 zip 파일이나 디렉터리는 쓸 수 없습니다.

### 클러스터 전역인 프록시·HTTP 설정

`proxy_location`에 넣을 수 있는 값은 셋입니다.

- `EveryNode`: 레플리카가 하나 이상 있는 모든 노드에 프록시를 띄웁니다. 기본값입니다.
- `HeadOnly`: 헤드 노드에만 프록시를 하나 띄웁니다.
- `Disabled`: 프록시를 띄우지 않습니다. 배포 핸들로만 호출할 때 고릅니다.

아키텍처 설명은 기본 HTTP 프록시가 헤드 노드에 하나라고 적지만, 설정 파일과 `serve start` CLI의 기본값은 `EveryNode`입니다.

`http_options`는 Ray 클러스터 전체에 적용되고 런타임에는 바꿀 수 없습니다. `grpc_servicer_functions`가 기본값인 빈 목록이면 gRPC 서버는 시작되지 않습니다.

### 재시작 없이 바뀌는 `user_config`

`user_config`는 배포의 `reconfigure` 메서드로 전달됩니다. 레플리카를 재시작하지 않고 동적으로 갱신할 수 있으며, 값은 JSON으로 직렬화할 수 있어야 합니다. 생성 시점에 `user_config`를 지정하면 Serve가 `__init__` 직후 `reconfigure`를 호출하고, 설정 파일의 값을 바꿔 다시 적용할 때도 같은 메서드를 호출합니다. 번역 언어는 `Translator`의 다음 메서드가 정하는 prefix로 바뀝니다.

```python
# excerpt: Translator에 추가하는 reconfigure 메서드
from typing import Dict

def reconfigure(self, config: Dict):
    self.language = config.get("language", "french")
    if self.language.lower() == "french":
        self.prefix = "translate English to French: "
    elif self.language.lower() == "german":
        self.prefix = "translate English to German: "
    elif self.language.lower() == "romanian":
        self.prefix = "translate English to Romanian: "
```

### `serve deploy`와 REST API로 적용하는 선언적 배포

로컬 Ray 클러스터를 띄우고 설정을 제출한 뒤 `serve status`로 상태를 확인합니다.

```bash
ray start --head
serve deploy serve_config.yaml
serve status
```

정상이면 `default`가 `RUNNING`, 두 배포가 `HEALTHY`로 보입니다.

갱신 절차는 설정 파일을 고친 뒤 `serve deploy`를 다시 실행하는 것뿐입니다. 예를 들어 `language: french`를 `german`으로 바꿔 재적용하면 `reconfigure`가 호출되어 레플리카는 그대로 둔 채 번역 대상 언어만 바뀝니다.

`serve build` 출력은 `serve deploy`에도, REST API에도 그대로 쓸 수 있습니다. REST API는 선언적으로 배포하므로, Serve가 이미 실행 중이면 새 설정에 없는 애플리케이션을 모두 제거합니다.

VM에서는 이 파일을 `serve deploy`와 함께 쓰고, Kubernetes에서는 RayService 커스텀 리소스에 넣어 씁니다. 같은 파일을 `serve run serve_config.yaml`로 로컬에서 시험할 수도 있습니다. 이때 `--route-prefix` 플래그는 무시됩니다. 명령은 터미널을 점유하다가 Ctrl-C를 누르면 클러스터의 Serve를 종료합니다.

### 따로 업그레이드하는 멀티 애플리케이션

업그레이드 단위는 애플리케이션입니다. 그래서 서로 독립적인 모델이나 모델 그룹을 별도 애플리케이션으로 나누면 각각 다른 엔드포인트 뒤에 두고 따로 업그레이드할 수 있습니다. 반대로 한 요청에서 함께 쓰이고 함께 업그레이드되는 배포는 한 애플리케이션에 둡니다. 요약과 번역은 한 요청 안에서 이어지므로 `text_ml:app` 하나로 묶어 두는 편이 맞습니다. 여기에 이미지 분류 애플리케이션을 나란히 올리려면 두 import path를 한꺼번에 넘겨 설정을 생성합니다.

```bash
serve build image_classifier:app text_ml:app -o config.yaml
```

멀티 애플리케이션 설정에서는 개별 애플리케이션 항목에 `host`와 `port`를 둘 수 없으므로 반드시 지워야 합니다. 생성된 애플리케이션 이름은 바꿀 수 있으므로, 텍스트 애플리케이션 항목의 `name`을 `text_ml`로 고쳐 둡니다. `applications` 아래 항목을 추가하거나 지우거나 고쳐도 클러스터의 다른 애플리케이션에는 영향이 없고, 재제출했을 때의 제자리 업데이트(in-place update) 동작은 단일 애플리케이션과 같습니다. 애플리케이션끼리는 HTTP를 거치지 않고도 호출할 수 있습니다. `serve.get_app_handle`로 실행 중인 다른 애플리케이션의 핸들을 얻습니다. 호출하는 쪽 코드는 아래 분기처럼 짧습니다.

```python
# excerpt: 다른 애플리케이션의 핸들을 얻어 번역을 위임하는 분기
if req.get("should_summarize") is True:
    handle: DeploymentHandle = serve.get_app_handle("text_ml")
    return await handle.summarize.remote(result)
```

### `serve config`의 목표 상태와 `serve status`의 현재 상태

두 명령은 보는 대상이 다릅니다. `serve config`는 클러스터가 마지막으로 받은 설정, 곧 목표 상태를 보여 주고, `serve status`는 프록시와 애플리케이션의 현재 상태를 보고합니다. 원격 클러스터에서는 두 명령 모두 `--address/-a`로 대상을 지정합니다.

## 한 매니페스트로 묶는 RayService와 대시보드·로그·메트릭으로 읽는 운영 상태

프로덕션에서 Ray Serve를 실행할 때 권장하는 방법은 Kubernetes에서 KubeRay의 RayService 커스텀 리소스를 쓰는 것입니다. 헬스 체크, 상태 보고, 장애 복구, 업그레이드를 RayService가 자동으로 처리합니다. Kubernetes를 쓰지 않는다면 앞서 본 Serve CLI로 Ray 클러스터에서 직접 실행합니다.

### 클러스터와 Serve 설정을 함께 담는 RayService CR

RayService CR 하나에는 멀티 노드 Ray 클러스터와 그 위에서 도는 Serve 애플리케이션이 함께 들어갑니다. 배포, 업그레이드, 상태 조회는 모두 표준 `kubectl` 명령으로 합니다. `spec` 아래에 넣는 필드는 둘입니다. `serveConfigV2`에는 `serve build`가 출력한 Serve 설정을 그대로 복사해 넣고, `rayClusterConfig`에는 `RayCluster` CR YAML의 `spec` 내용을 넣습니다. 구성은 이렇습니다.

```text
RayService (spec)
├── serveConfigV2     ← serve build 출력 (applications: default …)
└── rayClusterConfig  ← RayCluster CR의 spec (헤드·워커 파드, 이미지, 자원)
```

Serve 설정의 `runtime_env`에서 `working_dir`는 원격 URI만 받습니다. 그래서 파드가 `text_ml:app`을 불러오려면 `text_ml.py`를 원격 zip에 올려 그 URI를 넣습니다. 앞서 만든 설정을 이름 `default` 그대로 `serveConfigV2`에 넣으면 이런 모양이 됩니다.

```yaml
# excerpt: RayService spec의 serveConfigV2 부분 (rayClusterConfig는 예제 CR ray-service.text-ml.yaml의 것을 그대로 둠)
spec:
  serveConfigV2: |
    applications:
      - name: default
        route_prefix: /
        import_path: text_ml:app
        runtime_env:
          working_dir: "https://github.com/<repo>/archive/<commit>.zip"
```

이렇게 고친 예제 CR 파일을 적용합니다.

```bash
kubectl apply -f ray-service.text-ml.yaml
```

매니페스트를 `kubectl apply`로 적용하면 KubeRay 컨트롤러가 먼저 Ray 클러스터를 만듭니다. 클러스터가 실행되면 컨트롤러는 REST API로 Serve 애플리케이션을 배포하고, 트래픽을 받을 Kubernetes Service도 만듭니다. 예제에서 쿼리를 보낼 대상은 `rayservice-sample-serve-svc`입니다. 예제 RayService는 시연용이라 `numCpus`를 아주 낮게 잡았으므로, 프로덕션에서는 자원을 더 줘야 합니다. 내려받는 데 오래 걸리는 큰 의존성은 `runtime_env` 대신 이미지의 Dockerfile에 넣어 두면, 파드가 뜨자마자 쓸 수 있어 신뢰성이 높아집니다. 헤드 노드 장애 허용(fault tolerance)도 함께 설정하기를 권장합니다.

실행 중인 상태는 `kubectl describe`로 봅니다. 클러스터 상태, 헬스 체크 실패나 재시작 같은 이벤트, `serve status`가 보고하는 애플리케이션 수준 상태가 한 화면에 나옵니다. `Application Statuses` 아래에는 `serveConfigV2`의 애플리케이션 이름(여기서는 `default`)마다 배포별 상태가 나옵니다. `Summarizer`와 `Translator`가 `HEALTHY`이면 정상입니다.

### 바뀐 필드에 따라 갈리는 두 가지 업데이트

매니페스트를 고쳐 다시 `kubectl apply`하면, 고친 필드에 따라 동작이 갈립니다.

| 바뀐 필드 | 동작 | 주의할 점 |
|---|---|---|
| `serveConfigV2`만 | 같은 Ray 클러스터에서 제자리 업데이트를 합니다 | 레플리카 수 조정이나 오토스케일링 파라미터 변경 같은 가벼운 갱신에 맞습니다 |
| `rayClusterConfig` (예: 컨테이너 이미지) | 새 클러스터를 띄워 배포하고, 준비되면 Service를 새 클러스터로 돌린 뒤 이전 클러스터를 종료합니다 | 다운타임은 없지만 두 클러스터를 동시에 스케줄할 만큼 Kubernetes 클러스터가 커야 합니다 |

`Translator`의 `language` 변경은 위쪽 행, 이미지 교체는 아래쪽 행입니다.

KubeRay 클러스터에서 오토스케일링을 쓰려면 Serve 설정의 오토스케일링 필드와 함께 `enableInTreeAutoscaling`을 True로 설정해야 하며, 대부분의 경우 Kubernetes 오토스케일링도 함께 켜기를 권장합니다.

로드 밸런서나 인그레스를 둘 때, Serve는 모든 노드에서 HTTP 프록시를 실행하므로 `/-/routes`를 노드 헬스 체크 엔드포인트로 쓸 수 있습니다.

### 대시보드의 Serve 뷰

Ray 대시보드는 로컬 테스트에서도, 원격 프로덕션 클러스터에서도 클러스터와 Serve 애플리케이션 상태를 보여 줍니다. 대시보드를 쓰려면 `ray[default]`처럼 대시보드 구성 요소가 포함된 설치가 필요합니다. KubeRay 같은 원격 클러스터에서는 대시보드가 헤드 노드에서 뜨지만, 포트가 외부로 노출되지 않을 수 있습니다.

`http://localhost:8265/#/serve`에는 클러스터에 배포된 애플리케이션 목록이 먼저 보입니다.

![Ray Serve 대시보드의 애플리케이션 목록 화면](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/multi-app/applications-dashboard.png)

애플리케이션 상세 페이지에서는 설정과 메타데이터, 배포와 레플리카 목록을 보고, 배포마다 배포 설정을, 레플리카마다 로그를 볼 수 있습니다. Grafana와 Prometheus를 구성했다면 배포와 레플리카마다 Grafana 대시보드도 열 수 있습니다. 각 애플리케이션 아래에는 배포가 나열됩니다.

![애플리케이션 아래에 나열된 배포 목록 화면](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/multi-app/deployments-dashboard.png)

배포를 다시 펼치면 레플리카가 하나씩 나옵니다. 레플리카 상세 페이지에는 메타데이터와 완료된 태스크 이력이 나오고, Grafana와 Prometheus를 구성했다면 상위 수준 메트릭도 함께 나옵니다.

![배포별 레플리카 목록 화면](https://raw.githubusercontent.com/ray-project/images/master/docs/serve/multi-app/replica-dashboard.png)

### JSON으로 바꾸는 Serve 로그

Kubernetes에서 컨트롤러 로그와 배포 로그는 헤드 파드와 워커 파드 모두의 `/tmp/ray/session_latest/logs/serve/` 아래에 쌓입니다. 배포 실패 원인과 오토스케일링 이벤트가 여기 남습니다.

로그 인코딩 기본값은 `TEXT`입니다. `RAY_SERVE_LOG_ENCODING` 환경 변수로 기본값을 바꿀 수 있고, 구조화 로그용 `JSON`도 지원합니다. Python에서는 `serve.run`이나 `@serve.deployment`의 `logging_config`에 `encoding="JSON"`을 넘기면 JSON 형식이 됩니다. 요약·번역 애플리케이션에 적용하고, 요청 ID를 직접 붙여 호출하는 코드는 다음과 같습니다.

```python
# excerpt: 앞서 정의한 Summarizer·Translator 클래스가 있는 파일 끝에 덧붙이는 부분
import requests
from ray import serve
from ray.serve.schema import LoggingConfig

app = Summarizer.bind(Translator.bind())
serve.run(app, logging_config=LoggingConfig(encoding="JSON"))

english_text = "It was the best of times, it was the worst of times."
response = requests.post(
    "http://127.0.0.1:8000/",
    json=english_text,
    headers={"X-Request-ID": "summarize-0001"},
)
french_text = response.text
print(french_text)
print(response.headers)
```

`X-Request-ID` 헤더로 지정한 요청 ID는 응답에서 돌려받을 수 있습니다. 같은 ID를 여러 요청에 쓰면 디버깅이 헷갈리므로 피합니다. 요청 처리 중에 남는 JSON 로그 레코드에는 `request_id` 필드가 들어 있어, 이 ID로 레플리카 로그를 찾을 수 있습니다. `serve.run`에 넘긴 애플리케이션 수준 설정은 모든 배포에 적용되고, 배포 수준 설정이 있으면 배포 수준 설정이 우선합니다.

분산 트레이싱은 `RAY_SERVE_TRACING_EXPORTER_IMPORT_PATH` 환경 변수를 설정하면 켜집니다.

### Prometheus와 Grafana로 보는 Serve 메트릭

상태별 태스크·액터 수, 리소스 사용량, 오토스케일러 상태 같은 Ray 기본 시계열을 보여 주는 메트릭 뷰는 Prometheus와 Grafana 설정이 있어야 합니다. Serve 메트릭은 대시보드의 Serve 페이지와 레플리카 상세 페이지에서 볼 수 있고, Grafana에서는 route, deployment, replica로 걸러 볼 수 있습니다. 프로덕션에서는 로그와 메트릭을 사용 중인 관측 플랫폼으로 내보내기를 권장합니다. 지연 문제를 추적할 때 먼저 볼 메트릭은 셋입니다.

| 메트릭 | 측정 대상 | 해석 |
|---|---|---|
| `ray_serve_http_request_latency_ms` | 프록시가 요청을 받은 시점부터 응답을 보낸 시점까지의 종단 HTTP 지연 히스토그램입니다 | 사용자가 체감하는 지연입니다 |
| `ray_serve_deployment_processing_latency_ms` | 큐 대기 시간을 뺀 요청 처리 시간 히스토그램입니다 | 앞의 값과 차이가 크면 대기 시간이 길다는 뜻입니다 |
| `ray_serve_autoscaling_replica_metrics_delay_ms` | 레플리카 메트릭이 컨트롤러에 도달하기까지 걸린 시간입니다 | 값이 크면 컨트롤러 과부하를 의심합니다 |

예를 들어 `Summarizer`의 종단 지연이 900ms인데 처리 시간이 200ms라면, 나머지 700ms는 대부분 큐 대기에서 생긴 것으로 읽을 수 있습니다(가상의 수치입니다). 이런 경우에는 앞서 다룬 `max_ongoing_requests`와 레플리카 수부터 다시 봅니다.

지연이 가장 큰 버킷(기본 600,000ms)을 넘으면 모두 `+Inf` 버킷에 들어가 백분위 추정이 부정확해집니다. 버킷은 `RAY_SERVE_REQUEST_LATENCY_BUCKETS_MS="10,50,100,500,1000,5000"`처럼 쉼표로 구분한 환경 변수로 지정합니다.

## 추론이 더 큰 Python 파이프라인의 일부일 때 맞는 Ray Serve의 자리

Ray Serve는 여러 모델과 비즈니스 로직을 하나의 Python 애플리케이션으로 엮어 단계마다 따로 확장해야 할 때 가장 잘 맞습니다. 반대로 모델 하나를 최대한 빠르게 돌리는 일이나 모델의 수명주기를 관리하는 일은 직접 맡지 않으므로, 그런 요구가 있으면 다른 도구를 함께 써야 합니다.

### Serve가 하지 않는 일

Serve는 TFServing, TorchServe, ONNXRuntime 같은 프레임워크 전용 서빙 도구와 달리, 모델을 빠르게 만드는 모델별 최적화를 하지 않습니다. 최적화는 사용자가 직접 하고, 그 결과물을 Serve에 올립니다. PyTorch JIT로 컴파일한 모델이나 ONNXRuntime 모델을 배포 안에서 실행하는 식입니다.

Serve는 완전한 ML 플랫폼도 아닙니다. SageMaker, Azure ML, Vertex AI 같은 호스팅 서비스에 있는 모델 수명주기 관리나 성능 시각화 기능이 없습니다. 대신 그 위에 자체 ML 플랫폼을 쌓을 수 있는 기본 요소를 제공하는 데 집중합니다. BentoML이나 MLflow 같은 도구가 모델을 하나씩 서빙하고 확장하는 데 초점을 둔다면, Serve는 모델 조합에 초점을 두고 어떤 모델 패키징·레지스트리 형식과도 함께 동작합니다.

### MLflow는 레지스트리로, 서빙은 Serve로

앞서 본 MLflow 레지스트리 워밍업 로드 패턴에서는 MLflow가 모델을 보관하고 Serve가 서빙을 맡았습니다. 이렇게 역할을 나누는 데는 이유가 있습니다. MLflow 내장 REST 서버는 동시성이 높은 GPU 워크로드에 최적화되어 있지 않습니다. 그래서 LLM, 확산 모델처럼 무거운 Transformer 계열 모델은 Hugging Face Hub나 Weights & Biases를 레지스트리로 쓰고, GPU 배칭·오토스케일링·스케줄링은 Ray Serve에 맡기도록 권장합니다.

### Triton과 비교하고 섞어 쓰기

Suhas Bhairav의 블로그 글 "Triton vs Ray Serve"는 두 도구의 쓰임새를 이렇게 나눕니다. Ray Serve는 동적 라우팅, 액터 형태의 상태, 데이터 처리 단계와의 긴밀한 결합이 필요한 Python 파이프라인에 맞고, Triton은 여러 프레임워크의 모델을 GPU에서 배칭하고 동시에 실행해 서빙하는 데 맞습니다. 같은 글의 비교표는 Ray Serve가 코드 우선 라우팅, 유연한 모델 조합, Ray 클러스터 상태에 연동된 동적 스케일링을 제공하는 반면, 기본 제공 GPU 배칭은 Triton보다 약하다고 적습니다. 이 글은 핵심 모델은 Triton에서 서빙하고, 데이터 전처리와 후처리 오케스트레이션은 Ray가 맡는 하이브리드 구성도 제안합니다.

요약·번역 애플리케이션에 대입해 보겠습니다. `Translator`가 GPU 한 장을 꽉 채우는 대형 번역 모델로 커지고 커널 수준 배칭이 처리량을 좌우하게 됐다고 가정합니다. 이때는 번역 모델 자체는 Triton에 두고, `Summarizer` 쪽의 요청 정리와 후처리, 단계별 확장은 Serve 배포로 남기는 구성이 앞의 제안에 해당합니다.

### 운영에서 만나는 한계와 장애 양상

컨트롤러·클러스터 장애와 백프레셔, 모델 예외의 동작은 앞의 아키텍처와 배포 파라미터 절에서 다뤘습니다.

GPU 용량 문제는 따로 짚어 둘 만합니다. Spheron 블로그(Ray Serve LLM 서빙을 다룬 서드파티 글)의 보고에 따르면, 오토스케일링은 레플리카를 추가할 뿐 GPU 노드를 새로 프로비저닝하지 않습니다. GPU가 모자라면 추가된 레플리카는 PENDING 상태로 기다립니다. 노드 수준 확장은 앞서 본 KubeRay의 `enableInTreeAutoscaling`과 Kubernetes 오토스케일링 쪽에서 챙겨야 합니다.

### 이 글의 판단

추론 앞뒤로 Python 로직이 붙고 모델이 둘 이상이며 단계마다 부하가 다르다면, Ray Serve를 기본 선택으로 둡니다. 노트북에서 만든 코드를 거의 그대로 Kubernetes까지 옮길 수 있다는 점이 가장 큰 이점입니다. 모델 하나의 GPU 처리량이 비용을 좌우한다면 그 모델만 Triton이나 vLLM 같은 전용 엔진에 맡기고, Serve는 그 바깥을 조율하게 합니다. 모델 버전 관리, 승인, 성능 추적은 처음부터 레지스트리나 실험 추적 도구에 맡기고, Serve 쪽에서 직접 만들려고 하지 않는 편이 낫습니다. 모델이 하나뿐이고 전후처리가 거의 없는 서비스라면, Serve가 주는 조합과 독립 확장의 이점은 잘 드러나지 않습니다.

- - -

### Sources
* [Ray Serve: Scalable and Programmable Serving — Ray 2.60.0](https://docs.ray.io/en/latest/serve/index.html)
* [Key Concepts — Ray 2.60.0](https://docs.ray.io/en/latest/serve/key-concepts.html)
* [Architecture — Ray 2.60.0](https://docs.ray.io/en/latest/serve/architecture.html)
* [Production Guide — Ray 2.59.0](https://docs.ray.io/en/latest/serve/production-guide/index.html)
* [Getting Started — Ray 2.60.0](https://docs.ray.io/en/latest/serve/getting_started.html)
* [Request routing — Ray 2.58.0](https://docs.ray.io/en/latest/serve/llm/architecture/routing-policies.html)
* [RequestRouterConfig — Ray 2.59.0](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.config.RequestRouterConfig.html)
* [ray.serve.schema — Ray 2.59.0](https://docs.ray.io/en/latest/_modules/ray/serve/schema.html)
* [Ray Serve API — Ray 2.59.0](https://docs.ray.io/en/latest/serve/api/index.html)
* [Develop and Deploy an ML Application — Ray 2.60.0](https://docs.ray.io/en/latest/serve/develop-and-deploy.html)
* [Model Registry Integration — Ray 2.60.0](https://docs.ray.io/en/latest/serve/model-registries.html)
* [Deploy Compositions of Models — Ray 2.60.0](https://docs.ray.io/en/latest/serve/model_composition.html)
* [Configure Ray Serve deployments — Ray 2.60.0](https://docs.ray.io/en/latest/serve/configure-serve-deployment.html)
* [Configuration reference — Ray 2.60.0](https://docs.ray.io/en/latest/serve/llm/user-guides/configuration.html)
* [Serve Config Files — Ray 2.59.0](https://docs.ray.io/en/latest/serve/production-guide/config.html)
* [Deploy Multiple Applications — Ray 2.60.0](https://docs.ray.io/en/latest/serve/multi-app.html)
* [Monitor Your Application — Ray 2.59.0](https://docs.ray.io/en/latest/serve/monitoring.html)
* [Deploy on Kubernetes — Ray 2.59.0](https://docs.ray.io/en/latest/serve/production-guide/kubernetes.html)
* [Ray Dashboard — Ray 2.59.0](https://docs.ray.io/en/latest/ray-observability/getting-started.html)
* [Ray Serve Architecture Deep Dive](https://ccinthepunkin.github.io/claw-research-pages/reports/ray-serve-architecture)
* [Ray Serve: Reduce LLM Inference Latency by 60% with Custom Request Routing](https://www.anyscale.com/blog/ray-serve-faster-first-token-custom-routing)
* [Ray Serve on GPU Cloud: Production LLM Serving Guide (2026) &#124; Spheron](https://www.spheron.network/blog/ray-serve-gpu-cloud-llm-deployment)
* [Triton vs Ray Serve: GPU Inference for Production AI &#124; Suhas Bhairav](https://suhasbhairav.com/blog/triton-inference-server-vs-ray-serve-gpu-model-serving-standard-vs-python-native-scaling)
