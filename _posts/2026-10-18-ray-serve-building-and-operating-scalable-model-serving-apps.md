---
layout: post
title: "Ray Serve: Building and Operating Scalable Model-Serving Apps"
date: 2026-10-18 04:15:17
description: "Ray Serve로 모델을 HTTP 추론 API로 배포하는 방법을 다룹니다. 디플로이먼트와 레플리카 설정, 스트리밍 응답, DeploymentHandle로 여러 모델을 잇는 구조, 설정 파일 배포, KubeRay 운영과 모니터링까지 정리합니다."
categories: ["Tech Guides", "Data-Architecture"]
tags: ["Deployment-Handle", "Replica-Scaling", "Streaming-Responses", "KubeRay", "Prometheus-Metrics", "Model-Serving"]
cover: /assets/images/data-architecture.jpg
use_math: true
---
## Ray Serve가 해결하는 문제

학습을 마친 모델은 요청을 받아 응답을 돌려주는 온라인 추론 API가 되어야 쓸모가 생깁니다. Ray Serve는 이런 온라인 추론 API를 만드는 확장 가능한 모델 서빙 라이브러리입니다. 실제 서비스에서는 모델 하나만 엔드포인트 뒤에 두고 끝나는 경우가 드뭅니다. 입력을 전처리하고, 데이터베이스를 조회하고, 여러 모델의 결과를 합치고, 응답을 검증해야 합니다. 이미지 처리만 해도 전처리, 분할, 필터링으로 이어지는 다단계 파이프라인이 필요하고, 단계마다 아키텍처나 프레임워크가 다르며 CPU와 GPU처럼 요구하는 자원도 다릅니다.

기존 서빙 도구 상당수는 "텐서 입력, 텐서 출력" 방식에 집중합니다. 모델을 미리 정해진 구조의 엔드포인트 뒤에 감싸는 방식입니다. 하지만 머신러닝은 단독으로는 쓸모가 적어서, 비즈니스 로직이나 데이터베이스 조회 같은 웹 서빙 로직과 결합해야 할 때가 많습니다. 여러 모델을 잇는 경우에도 YAML 같은 설정 언어로 정적 그래프를 정의하게 하는 도구가 많은데, 이 방식은 제약이 많고 다루기 어렵습니다.

### 핵심 아이디어: 서비스 전체를 하나의 파이썬 프로그램으로

Ray Serve는 여러 ML 모델, 비즈니스 로직, HTTP 처리를 한 프레임워크 안에서 하나의 파이썬 프로그램으로 작성하고 배포하게 합니다. HTTP 파싱과 검증은 FastAPI 연동으로 처리하고, 모델 간 호출은 프로그래밍 가능한 API로 표현합니다. 이 방식을 모델 컴포지션(model composition)이라고 하며, 다른 모델을 호출하는 코드가 평범한 함수 호출처럼 보입니다. 각 모델은 서로 다른 자원을 쓰면서 클러스터의 여러 머신에 흩어져 실행되지만, 개발자는 일반 프로그램을 쓰듯 코드를 작성합니다.

이 설계가 가능한 이유는 Ray Serve가 Ray 위에 구축되어 있어서입니다. 덕분에 여러 머신으로 쉽게 확장되고, 분할 GPU 같은 유연한 스케줄링을 지원해 여러 모델이 자원을 나눠 쓰며 낮은 비용으로 서빙됩니다. 운영 비용 측면에서 제공하는 기본 요소는 셋입니다.

- 레플리카(replica) 수를 늘리고 줄여 모델별 자원을 동적으로 조정하는 기능
- 요청을 묶어 벡터화 연산을 활용하는 배칭(GPU에서 특히 중요합니다)
- 제한된 하드웨어에 많은 모델을 올리는 유연한 자원 할당 모델

LLM 서빙을 위해서는 스트리밍 응답(streaming response), 동적 요청 배칭(dynamic request batching), 멀티 노드·멀티 GPU 서빙 같은 기능과 성능 최적화도 갖추고 있습니다. 프롬프트 전처리, 벡터 데이터베이스 조회, LLM API 호출, 응답 검증을 한 파이썬 모듈에 작성하고, 각 단계가 트래픽에 맞춰 독립적으로 오토스케일링(autoscaling)됩니다.

### 프레임워크에 묶이지 않는 설계

Ray Serve는 특정 ML 라이브러리에 종속되지 않는 범용 서빙 계층입니다. PyTorch, TensorFlow, Keras로 만든 딥러닝 모델부터 scikit-learn 모델, 임의의 파이썬 비즈니스 로직까지 같은 도구로 서빙합니다. 새 라이브러리와 아키텍처가 계속 등장하는 상황에서 서빙 인프라를 바꾸는 일은 시간과 비용이 들고 위험하기 때문에, 이 설계는 프레임워크 교체 부담을 줄여 줍니다.

대가도 있습니다. Ray Serve는 모델을 빠르게 만드는 모델별 최적화를 직접 하지 않습니다. 최적화는 사용자가 직접 하고, 그 결과물을 Ray Serve에서 실행하는 구조입니다. 예를 들어 PyTorch JIT로 컴파일한 모델이나 ONNXRuntime 모델을 그대로 올릴 수 있습니다. 임의의 파이썬 코드를 지원하므로 MLOps 도구와도 잘 맞물립니다. 모델 최적화기(ONNX, TVM), 모델 모니터링(Seldon Alibi, Arize), 모델 레지스트리(MLFlow, Weights and Biases), 데이터 앱 UI(Gradio, Streamlit), 웹 API 프레임워크(FastAPI, gRPC)와 함께 쓰는 구성이 가능합니다. 실행 위치도 Ray가 도는 곳이면 어디든 됩니다. 노트북, Kubernetes, 주요 퍼블릭 클라우드, 온프레미스 장비 모두 해당합니다.

### 다른 서빙 도구와 비교

| 비교 대상 | 해당 도구의 초점 | Ray Serve와의 차이 |
|---|---|---|
| TFServing, TorchServe, ONNXRuntime | 특정 프레임워크 전용 서빙과 모델별 최적화 | 프레임워크 비종속 서빙 계층, 모델별 최적화는 사용자 몫 |
| AWS SageMaker, Azure ML, Google Vertex AI | 관리형 서비스 형태의 ML 플랫폼 | 자체 인프라에서 쓰는 오픈소스 서빙 라이브러리, 모델 수명주기 관리와 성능 시각화 기능 없음 |
| Seldon, KServe, Cortex | Kubernetes 기반 모델 배포 | 노트북에서 개발 후 코드 변경 없이 또는 최소 변경으로 클러스터 확장, 초기 단계에 Kubernetes 클러스터 불필요 |
| BentoML, Comet.ml, MLflow | 개별 모델의 서빙과 확장 | 모델 컴포지션 중심 설계, 모든 모델 패키징·레지스트리 형식과 호환 |

표에서 눈여겨볼 차이는 범위입니다. Ray Serve는 완전한 ML 플랫폼이 아니라 모델 서빙에 집중하고, 그 위에 자체 ML 플랫폼을 쌓을 수 있는 기본 요소를 제공합니다. 모델 수명주기 관리나 성능 시각화까지 한곳에서 원한다면 관리형 플랫폼이 맞고, 여러 모델과 비즈니스 로직을 묶은 서비스를 자기 인프라에서 세밀하게 제어하고 싶다면 Ray Serve가 맞습니다. 공식 문서는 다른 도구로 같은 기능을 얻으려면 TensorFlow Serving과 SageMaker 같은 여러 프레임워크를 이어 붙이거나 마이크로 배칭 구성 요소를 직접 만들어야 한다고 주장합니다.

Kubernetes 개념을 깊이 몰라도 시작할 수 있다는 점도 실용적인 차이입니다. 로컬 머신에서 모델과 전체 디플로이먼트(deployment) 그래프를 테스트한 뒤, 프로덕션 시점에는 Kubernetes 오퍼레이터인 KubeRay로 같은 애플리케이션(application)을 클러스터에 배포합니다.

## 설치와 예제 모델 준비

예제를 실행하려면 Ray Serve와 함께 Hugging Face `transformers`, PyTorch, HTTP 클라이언트용 `requests`를 설치합니다. 셸에서 대괄호가 해석되지 않도록 따옴표로 감쌉니다.

```bash
pip install "ray[serve]" transformers requests torch
```

예제 모델은 Hugging Face의 t5-small입니다. t5-small은 텍스트를 입력받아 텍스트를 내놓는 모델로, 입력 앞에 작업 설명을 접두어로 붙이면 그 작업을 수행합니다. 아래 `Translator` 클래스는 Ray Serve를 전혀 쓰지 않는 평범한 파이썬 코드입니다. `__init__`에서 토크나이저와 모델을 불러오고, `translate`에서 입력에 `"translate English to French: "`를 붙여 `model.generate()`로 출력 토큰을 만든 뒤 번역 문자열로 디코딩합니다.

```python
# File name: model.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

translator = Translator()

translation = translator.translate("Hello world!")
print(translation)
```

실행하면 `"Hello world!"`가 프랑스어 `"Bonjour Monde!"`로 번역됩니다.

```console
$ python model.py

Bonjour Monde!
```

모델 적재는 생성자에 두고 추론은 메서드로 분리한 구조라서, 클래스 본문을 거의 고치지 않고 서빙 코드로 옮길 수 있습니다. t5-small은 설명용 예시일 뿐이며, 어떤 파이썬 프레임워크의 모델이든 같은 방식으로 따라 할 수 있습니다.

## 디플로이먼트, 레플리카, 자원 설정

앞서 만든 `Translator`는 한 프로세스 안에서 한 번 실행되고 끝나는 평범한 클래스입니다. 서빙에서는 이 클래스가 요청을 기다리며 계속 살아 있어야 하고, 트래픽이 늘면 사본을 여러 개 띄워야 하며, 사본마다 CPU나 GPU를 얼마나 쓸지 정해 두어야 합니다. Ray Serve에서 이 세 가지를 한꺼번에 담는 단위가 디플로이먼트입니다.

### 클래스를 디플로이먼트로 바꾸기

디플로이먼트는 Ray Serve의 중심 개념입니다. 사용자가 작성한 파이썬 클래스 하나(간단한 경우에는 함수 하나)를 담고, 그 코드로 들어오는 요청을 처리하며, Ray 클러스터 전체로 확장해 실행됩니다. 클래스를 디플로이먼트로 바꾸는 것은 `@serve.deployment` 데코레이터입니다. 아래는 `Translator`를 Ray Serve 애플리케이션으로 옮긴 전체 스크립트입니다.

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

원래 코드와 비교하면 바뀐 곳은 셋입니다. 클래스 위에 `@serve.deployment` 데코레이터가 붙었고, `__call__` 메서드가 새로 생겼으며, 맨 끝에서 `bind()`를 호출합니다. `__init__`과 `translate`는 한 글자도 바뀌지 않았습니다.

데코레이터는 `Translator`를 파이썬 클래스에서 Ray Serve의 `Deployment` 객체로 바꿉니다. 이후 `Translator`는 직접 인스턴스를 만드는 대상이 아니라, Ray Serve가 클러스터에 사본을 띄울 때 쓰는 설계도가 됩니다. `__call__`은 HTTP 요청이 들어왔을 때 기본으로 호출되는 메서드로, 요청 본문의 JSON을 읽어 `translate`에 넘기고 그 반환값을 응답 본문으로 돌려줍니다. 외부 트래픽을 받는 이런 디플로이먼트를 인그레스 디플로이먼트(ingress deployment)라고 하며, HTTP 처리 방식은 별도로 자세히 다룰 주제라서 여기서는 "요청을 받아 `translate`로 넘기는 얇은 진입점" 정도로 이해하면 충분합니다.

데코레이터에 넘긴 인자는 모두 선택 사항입니다. 레플리카 수나 자원을 정하지 않아도 되고, 그럴 때는 인자 없이 씁니다.

```python
@serve.deployment
class Translator:
    ...
```

### bind로 애플리케이션 정의하기

디플로이먼트만으로는 아직 실행할 수 있는 단위가 아닙니다. `bind`를 호출해 디플로이먼트 생성자에 넘길 인자를 묶어야 Ray Serve 애플리케이션이 정의됩니다. `Translator`의 생성자는 인자를 받지 않으므로 `Translator.bind()`처럼 빈 괄호로 호출했습니다. 이렇게 만든 `translator_app`은 로컬에서 실행할 수도, 프로덕션에 배포할 수도 있는 애플리케이션 객체입니다.

`bind`는 애플리케이션을 정의할 뿐이고, 실제 배포는 그 결과를 `serve.run`(또는 `serve run` CLI 명령)에 넘길 때 일어납니다. 생성자에서 모델 가중치를 읽어 들이는 무거운 작업은 레플리카마다 실행됩니다. `bind`는 "이 디플로이먼트를 이 인자로 만들어라"라는 지시를 기록하는 단계입니다. 생성자에 인자가 있다면 그 값을 `bind`에 그대로 넘깁니다.

```python
from ray import serve

@serve.deployment
class MyFirstDeployment:
    # Take the message to return as an argument to the constructor.
    def __init__(self, msg):
        self.msg = msg

    def __call__(self):
        return self.msg

my_first_deployment = MyFirstDeployment.bind("Hello world!")
```

`bind`의 인자로 다른 디플로이먼트를 `bind`한 결과를 넘길 수도 있습니다. 그러면 런타임에 그 인자가 DeploymentHandle로 바뀌어 생성자에 들어가고, 한 애플리케이션이 여러 디플로이먼트로 구성됩니다. 이 구조 덕분에 디플로이먼트 사이의 연결 관계를 평범한 파이썬 코드로 적습니다.

### num_replicas: 레플리카 수로 수평 확장

런타임에 디플로이먼트는 여러 개의 레플리카로 이루어집니다. 레플리카는 디플로이먼트에 담긴 클래스의 개별 사본으로, 각각 별도의 Ray 액터(Ray Actor), 즉 별도 프로세스에서 시작됩니다. `num_replicas`는 이 사본을 몇 개 띄울지 정하는 정수이고, 들어오는 요청은 레플리카들에 부하 분산됩니다. 예제의 `num_replicas=2`는 `Translator` 프로세스 두 개가 각자 t5-small을 메모리에 올리고 요청을 나눠 받는다는 뜻입니다.

레플리카가 프로세스 단위로 분리되어 있어서 얻는 이점이 있습니다. 한 레플리카가 긴 빔 서치 생성으로 바쁜 동안 다른 레플리카가 다음 요청을 처리하므로, 단일 프로세스의 처리량 한계를 사본 수로 넘어섭니다. 대가는 메모리입니다. 레플리카마다 생성자가 따로 실행되므로 모델 가중치도 레플리카 수만큼 복제됩니다. 작은 t5-small에서는 문제가 안 되지만, 큰 모델일수록 레플리카 하나를 늘리는 비용이 커집니다.

레플리카 수는 고정값으로 둘 수도 있고, 부하에 맞춰 늘리고 줄일 수도 있습니다. 오토스케일링은 `autoscaling_config` 파라미터로 설정합니다.

### ray_actor_options: 레플리카마다 자원 예약

`ray_actor_options`는 레플리카 각각에 적용할 설정을 담는 딕셔너리입니다. 자원 요구량이 대표적이며, 디플로이먼트 전체가 아니라 레플리카 하나 기준입니다.

| 옵션 | 타입 | 의미 |
|---|---|---|
| `num_cpus` | 실수 | 레플리카 하나가 예약할 CPU의 논리적 개수입니다. 분수로 지정할 수 있습니다. |
| `num_gpus` | 실수 | 레플리카 하나가 예약할 GPU의 논리적 개수입니다. 분수로 지정할 수 있습니다. |
| `resources` | 딕셔너리 | HPU나 TPU 같은 GPU 외 가속기 등 그 밖의 자원 요구량입니다. |

예제 설정 `{"num_cpus": 0.2, "num_gpus": 0}`은 레플리카마다 CPU 0.2개를 예약하고 GPU는 쓰지 않는다는 뜻입니다. 레플리카가 둘이므로 `Translator` 디플로이먼트 전체로는 CPU 0.4개를 예약합니다. 레플리카마다 GPU를 예약하려면 `num_gpus`에 값을 지정합니다.

여기서 "논리적"이라는 표현에 주의해야 합니다. 이 숫자는 Ray가 레플리카를 머신에 배치할 때 기준으로 삼는 예약량입니다. 그래서 실제 모델이 쓰는 연산량을 대략 반영해 정해야 하고, 너무 작게 잡으면 한 머신에 레플리카가 과하게 몰리며, 너무 크게 잡으면 자원이 남는데도 레플리카를 배치하지 못합니다.

### 분할 CPU·GPU로 레플리카 채워 넣기

`num_cpus`와 `num_gpus`가 실수인 이유는 분수 값을 허용하기 위해서입니다. 분수를 쓰면 레플리카 수보다 CPU나 GPU가 적은 머신에도 여러 레플리카를 함께 올릴 수 있습니다. 작은 모델 여러 개가 GPU 한 장을 나눠 쓰게 만드는 분할 GPU가 바로 이 기능이고, 한정된 하드웨어에서 많은 모델을 낮은 비용으로 서빙할 때 핵심 수단이 됩니다.

다음은 배치 가능한 레플리카 수를 손으로 따라가 본 예시로, 설명을 위해 만든 숫자입니다(예시). CPU 1개짜리 노트북에 예제 설정 그대로 `num_cpus=0.2`인 레플리카를 띄운다고 하면, 논리적으로 들어가는 레플리카 수는 다음과 같습니다.

$$
\left\lfloor \frac{1}{0.2} \right\rfloor = 5
$$

`num_replicas=2`라면 CPU 0.4개만 예약하고 0.6개가 남으므로, 같은 머신에 다른 디플로이먼트를 함께 올릴 여유가 생깁니다. 반대로 `num_cpus=1`로 두었다면 레플리카 하나가 CPU 전체를 예약해 두 번째 레플리카는 배치될 자리가 없습니다. GPU도 계산이 같습니다. `num_gpus=0.5`로 두면 GPU 한 장에 레플리카 두 개가 올라가고, 두 레플리카는 GPU 메모리를 나눠 쓰게 되므로 두 모델이 실제로 한 장의 메모리 안에 들어가는지는 따로 확인해야 합니다.

### 디플로이먼트마다 따로 설정하는 이유

레플리카 수와 자원을 데코레이터 인자로 둔 설계는 디플로이먼트 단위의 독립 확장으로 이어집니다. 한 애플리케이션 안의 디플로이먼트들은 `autoscaling_config`, `num_replicas`, `num_cpus`, `num_gpus`를 각자 다르게 설정합니다. 예를 들어 요청의 75%가 스페인어, 25%가 프랑스어인 서비스라면 스페인어 응답 디플로이먼트는 레플리카 3개, 프랑스어 쪽은 1개로 나눠 수요에 맞춥니다. CPU만 필요한 전처리 단계와 GPU가 필요한 추론 단계를 서로 다른 디플로이먼트로 나누면, 자원 종류와 양이 다른 단계들이 한 덩어리로 묶여 생기는 애플리케이션 수준의 병목도 피합니다.

실무에서는 먼저 데코레이터 인자 없이 동작을 확인하고, 모델 하나가 실제로 쓰는 CPU·GPU 양을 보고 `ray_actor_options`를 정한 뒤, 처리량이 부족한 디플로이먼트만 `num_replicas`를 늘리는 순서를 권합니다. 레플리카 하나의 자원 크기와 레플리카 수를 따로 조정할 수 있어서, 수직 방향(사본 하나의 크기)과 수평 방향(사본 개수)의 조정을 섞지 않고 하나씩 맞춰 나갈 수 있습니다.

## 로컬에서 실행하고 요청 보내기

앞서 만든 `serve_quickstart.py`는 맨 끝에서 `Translator.bind()`로 `translator_app`이라는 애플리케이션을 정의합니다. 로컬에서 이 애플리케이션을 실행해 HTTP 요청을 받게 하려면 `serve run` CLI 명령을 씁니다.

### serve run과 import 경로

`serve run`은 실행할 애플리케이션을 `module:application` 형식의 import 경로로 받습니다. 콜론 앞은 파이썬 모듈 이름이고, 콜론 뒤는 그 모듈 안에서 애플리케이션 객체를 담은 변수 이름입니다. `serve_quickstart.py` 파일의 `translator_app` 변수라면 다음과 같이 적습니다.

```bash
serve run serve_quickstart:translator_app
```

Serve는 import 경로로 모듈을 불러오므로 명령을 실행하는 위치가 중요합니다. `serve_quickstart.py`가 있는 디렉터리에서 실행해야 모듈을 찾습니다. 모듈을 찾는 기준 디렉터리는 `-d`(`--app-dir`) 옵션으로 바꿉니다. 이 디렉터리는 `PYTHONPATH`에 추가되며, 기본값은 현재 디렉터리를 뜻하는 `.`입니다. 기본값이면 `./main.py`에 있는 객체를 `main.object`라는 경로로 불러옵니다. 설치된 패키지에서 애플리케이션을 불러올 때는 이 옵션이 필요 없습니다.

import 경로는 `Application` 객체를 가리키거나, `Application`을 반환하는 함수를 가리켜야 합니다. 함수를 가리키면 import 경로 뒤에 `key=val` 형식으로 인자를 덧붙여 그 함수에 넘깁니다. 모델 경로나 레플리카 수처럼 실행할 때마다 바꾸고 싶은 값을 코드 수정 없이 명령줄에서 넘길 때 유용합니다.

```bash
serve run my_script:app model_path='/path/to/model.pkl' num_replicas=5
```

### 블로킹 실행과 종료

`serve run`은 기본적으로 블로킹 방식으로 동작합니다. 애플리케이션을 실행한 뒤 터미널을 점유한 채 로그를 콘솔로 계속 출력합니다.

`Ctrl-C`를 누르면 애플리케이션을 정리하는 데서 그치지 않고 클러스터의 Serve 자체를 종료합니다. Serve가 종료되면 클러스터의 모든 애플리케이션이 삭제되므로, 같은 클러스터에 다른 Serve 애플리케이션을 띄워 두었다면 그 애플리케이션도 함께 내려갑니다. 로컬에서 여러 애플리케이션을 동시에 시험할 때는 주의해야 합니다.

`--non-blocking`을 주면 명령을 블로킹하지 않고 실행합니다. `Ctrl-C`로 애플리케이션을 정리하는 동작은 블로킹으로 실행했을 때의 동작입니다. 자주 쓰는 옵션은 다음과 같습니다.

| 옵션 | 동작 |
|---|---|
| `--blocking`, `--non-blocking` | 명령을 블로킹으로 실행할지 정합니다. 블로킹이면 `Ctrl-C`를 누를 때까지 상태를 로그로 출력하다가 애플리케이션을 정리합니다. |
| `-r`, `--reload` | 작업 디렉터리의 파일 변경을 감지해 애플리케이션을 자동으로 다시 배포합니다. 실험적 기능이며, `Ctrl-C`를 누를 때까지 블로킹합니다. |
| `--name` | 애플리케이션 이름을 지정합니다. import 경로로 실행할 때만 쓰며, 설정 파일로 실행하면 무시됩니다. |
| `-a`, `--address` | `ray.init()`에 넘길 주소를 지정합니다. `RAY_ADDRESS` 환경 변수로도 지정합니다. |

모델 코드를 자주 고치는 단계라면 `--reload`가 편합니다.

### 기본 HTTP 엔드포인트

애플리케이션이 실행되면 기본적으로 다음 주소로 요청을 보냅니다.

```text
http://127.0.0.1:8000/
```

HTTP 프록시가 요청을 받는 호스트의 기본값은 localhost이고 포트의 기본값은 8000입니다. 그래서 별도 설정 없이도 같은 머신의 클라이언트가 `127.0.0.1:8000`으로 접속합니다. 이 주소로 들어온 요청은 앞서 정의한 인그레스 디플로이먼트인 `Translator`의 `__call__`로 전달되고, 요청은 레플리카 두 개에 부하 분산됩니다.

### 클라이언트 스크립트로 요청 보내기

`Translator`의 `__call__`은 요청 본문을 JSON으로 읽으므로, 클라이언트는 영어 문장을 JSON 데이터로 담아 POST 요청을 보냅니다. `requests.post`의 `json=` 인자에 문자열을 그대로 넘기면 그 문자열이 JSON으로 직렬화되어 본문에 들어갑니다. `__call__`이 반환한 번역 문자열은 응답 본문으로 돌아오므로 `response.text`로 읽습니다.

```python
# File name: model_client.py
import requests

english_text = "Hello world!"

response = requests.post("http://127.0.0.1:8000/", json=english_text)
french_text = response.text

print(french_text)
```

`serve run`이 터미널 하나를 점유하고 있으므로, 애플리케이션이 실행 중인 상태에서 다른 터미널 창을 열어 클라이언트를 실행합니다.

```console
$ python model_client.py

Bonjour monde!
```

번역 결과가 HTTP 응답으로 돌아왔다면, Ray Serve를 쓰지 않던 `Translator` 클래스가 레플리카 두 개로 확장된 HTTP 서비스로 동작하고 있습니다. 클라이언트와 서버가 주고받는 형식은 `__call__`이 정합니다. 요청 본문을 JSON으로 읽고 문자열을 반환하도록 작성했기 때문에 클라이언트도 JSON을 보내고 텍스트를 받습니다. `__call__`이 다른 형식으로 입력을 읽으면 클라이언트 코드도 그 형식에 맞춰 바꿔야 합니다.

## Starlette와 FastAPI로 인그레스와 HTTP 처리하기

애플리케이션 하나는 디플로이먼트 여러 개로 구성될 수 있지만, 외부 트래픽을 받는 디플로이먼트는 언제나 하나입니다. `serve run`이나 `serve.run`에 넘긴 최상위 디플로이먼트가 그 역할을 맡는 인그레스 디플로이먼트이고, 애플리케이션의 HTTP 처리 로직은 이 디플로이먼트에 정의합니다. 요청을 어떤 형식으로 읽을지, 어떤 상태 코드와 헤더로 응답할지, 경로를 몇 개 둘지가 모두 여기서 결정됩니다. Ray Serve는 HTTP API를 만드는 방법을 두 단계로 제공합니다. 원시 요청 객체를 직접 다루는 Starlette 방식과, 검증과 문서 생성까지 갖춘 API 서버를 만드는 FastAPI 통합입니다.

| 구분 | Starlette `Request`와 `__call__` | FastAPI 통합(`@serve.ingress`) |
|---|---|---|
| 진입점 | `__call__` 메서드 하나입니다. | `@app.get`, `@app.post` 등으로 등록한 여러 메서드입니다. |
| 요청 해석 | `request.json()`, `request.query_params`로 직접 읽습니다. | 함수 시그니처를 보고 FastAPI가 파싱하고 타입을 검증합니다. |
| 라우팅 | 경로 접두사 하나로 들어온 요청을 모두 받습니다. | 경로 접두사 아래에 경로를 여러 개 둡니다. |
| 부가 기능 | 직접 구현해야 합니다. | 미들웨어, OpenAPI 문서 자동 생성, 의존성 주입, WebSocket을 그대로 씁니다. |
| 적합한 경우 | 단순한 엔드포인트 하나를 빠르게 노출할 때입니다. | 여러 경로와 입력 검증이 필요한 API 서버를 만들 때입니다. |

### `__call__`과 Starlette 요청 객체

앞서 만든 `Translator`의 `__call__`이 가장 기본적인 형태입니다. 경로 접두사 `/`로 들어온 HTTP 요청은 디플로이먼트의 `__call__` 메서드로 전달되며, 이때 유일한 인자가 Starlette의 `Request` 객체입니다. 이 객체는 HTTP 요청 그 자체이므로 본문을 JSON으로 읽을 수도(`await request.json()`), 바이트로 읽을 수도(`await request.body()`), 쿼리 문자열만 꺼낼 수도(`request.query_params`) 있습니다. 쿼리 문자열을 그대로 돌려주는 디플로이먼트라면 다음과 같습니다.

```python
import starlette.requests
import requests
from ray import serve

@serve.deployment
class Counter:
    def __call__(self, request: starlette.requests.Request):
        return request.query_params

serve.run(Counter.bind())
resp = requests.get("http://localhost:8000?a=b&c=d")
assert resp.json() == {"a": "b", "c": "d"}
```

반환값은 두 종류입니다. JSON으로 직렬화되는 객체를 반환하면 Serve가 JSON으로 직렬화해 응답 본문에 담습니다. 상태 코드나 헤더를 직접 정하고 싶다면 Starlette `Response` 객체를 반환합니다. 입력이 잘못되었을 때 400 계열 상태 코드를 돌려주는 식의 처리가 이 경로로 들어갑니다.

이 방식은 HTTP 처리 코드가 거의 없어 가볍습니다. 대신 입력 검증, 경로 분기, 오류 응답을 모두 직접 써야 합니다. `Translator`처럼 "문자열 하나 받아 문자열 하나 돌려주는" 엔드포인트에는 충분하지만, 경로가 늘어나거나 입력 스키마가 복잡해지면 `__call__` 안에 `if` 분기가 쌓이기 시작합니다. 그 시점이 FastAPI로 넘어갈 때입니다.

### 경로 접두사

애플리케이션은 `route_prefix`로 지정한 경로에서 HTTP 요청을 받으며, 기본값은 `/`입니다. 이 값은 `serve.run()`의 인자나 Serve 설정 파일(Serve config)에서 다른 문자열로 바꿉니다. 예를 들어 번역 애플리케이션을 `/translate` 아래에 두려면 다음과 같이 실행합니다.

```python
serve.run(translator_app, route_prefix="/translate")
```

### `@serve.ingress`로 FastAPI 앱 감싸기

FastAPI 통합은 `@serve.ingress` 데코레이터로 FastAPI 앱을 디플로이먼트 클래스에 감싸는 방식입니다. FastAPI의 라우트 데코레이터를 클래스 메서드에 붙이면, 그 메서드가 해당 경로와 HTTP 메서드의 처리기가 됩니다. `Translator`를 이 방식으로 옮기면 `__call__`이 사라지고 `translate` 자체가 엔드포인트가 됩니다.

```python
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

데코레이터 순서를 지켜야 합니다. `@serve.deployment`가 바깥, `@serve.ingress(app)`가 안쪽입니다. 그리고 요청 형식이 바뀌었습니다. FastAPI는 시그니처의 `text: str`을 보고 요청에서 `text`라는 값을 찾아 파싱하고 타입을 검증하며, 이렇게 선언한 단순 타입 인자는 쿼리 파라미터로 해석됩니다. 그래서 클라이언트도 본문 JSON이 아니라 `params`로 값을 보내고, 반환된 문자열은 JSON으로 직렬화되어 오므로 `response.json()`으로 읽습니다.

```python
# File name: model_client.py
import requests

response = requests.post("http://127.0.0.1:8000/", params={"text": "Hello world!"})
french_text = response.json()

print(french_text)
```

`__call__` 방식에서는 `await http_request.json()` 한 줄이 입력 형식을 정했다면, FastAPI 방식에서는 함수 시그니처가 입력 형식을 정합니다. `text`를 빠뜨린 요청은 모델 코드에 닿기 전에 FastAPI가 거절하므로, 잘못된 입력 때문에 레플리카가 비싼 `generate`를 실행하는 일이 없습니다.

FastAPI 라우트는 Serve의 경로 접두사 위에 얹힙니다. 애플리케이션의 `route_prefix`가 `/my_app`이고 메서드에 `@app.get("/fetch_data")`를 붙였다면, 실제 호출 경로는 `/my_app/fetch_data`입니다. 경로 변수와 여러 HTTP 메서드를 함께 쓰는 예는 다음과 같습니다.

```python
import requests
from fastapi import FastAPI
from ray import serve

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class MyFastAPIDeployment:
    @app.get("/")
    def root(self):
        return "Hello, world!"

    @app.post("/{subpath}")
    def root(self, subpath: str):
        return f"Hello from {subpath}!"

serve.run(MyFastAPIDeployment.bind(), route_prefix="/hello")
resp = requests.post("http://localhost:8000/hello/Serve")
assert resp.json() == "Hello from Serve!"
```

이미 운영 중인 FastAPI 앱이 있다면 코드를 고치지 않고 그대로 감쌀 수도 있습니다. 모듈 수준에서 라우트를 등록한 `app`을 `@serve.ingress(app)`에 넘기고 클래스 본문은 `pass`로 비워 두면, 기존 미들웨어와 OpenAPI 문서 자동 생성 같은 기능이 그대로 동작한 채 레플리카 단위로 확장됩니다. HTTP 미들웨어는 `@serve.ingress`에 넘기는 FastAPI 앱에 `app.add_middleware(...)`로 등록합니다. `HTTPOptions.middlewares` 필드는 더 이상 권장하지 않습니다(deprecated). WebSocket도 FastAPI를 통해 지원하며, 처리 함수에 `@app.websocket`을 붙입니다.

### FastAPI 팩토리 패턴

앞의 방식은 FastAPI 객체를 만든 뒤 데코레이터에 넘기는 객체 기반 패턴입니다. 이 패턴에서는 FastAPI 객체를 cloudpickle로 직렬화할 수 있어야 하므로, 스레드 락처럼 직렬화할 수 없는 구성 요소에 의존하는 라이브러리는 쓸 수 없습니다. OpenTelemetry의 `FastAPIInstrumentor`가 대표적입니다. 팩토리 패턴은 FastAPI 객체 대신 그 객체를 만드는 함수를 `@serve.ingress`에 넘깁니다. 그러면 각 레플리카에서 FastAPI 앱을 직접 만들므로 직렬화가 필요 없습니다.

```python
from fastapi import FastAPI
from ray import serve
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

def fastapi_factory():
    app = FastAPI()
    FastAPIInstrumentor.instrument_app(app)

    @app.get("/")
    async def root():
        return "ok"

    return app

@serve.deployment
@serve.ingress(fastapi_factory)
class Ingress:
    pass

app = Ingress.bind()
```

대신 `self`를 쓸 수 없습니다. 팩토리 안에서 정의한 처리기는 디플로이먼트 인스턴스에 접근하지 못하므로, 생성자에서 적재한 모델이나 다른 디플로이먼트 참조를 `self`로 꺼내 쓰지 못합니다. 다른 디플로이먼트를 호출해야 한다면 `serve.get_deployment_handle`로 이름을 지정해 가져와야 합니다.

### 스트리밍 응답

LLM 텍스트 생성이나 영상 처리처럼 순전파 한 번에 몇 초씩 걸리는 작업은 결과를 한꺼번에 보내기보다, 만들어지는 대로 조금씩 보내는 편이 사용자 경험이 훨씬 낫습니다. HTTP 스트리밍 응답은 처리기에서 제너레이터를 감싼 Starlette `StreamingResponse`를 반환하면 됩니다. `__call__` 방식과 FastAPI 통합 양쪽에서 지원합니다.

클라이언트가 스트림을 끝까지 받지 않고 연결을 끊는 경우도 고려해야 합니다. 생성이 무한히 이어지거나 비싼 작업이라면, 아무도 받지 않는 토큰을 계속 만드는 것은 자원 낭비입니다. `StreamingResponse`에 비동기 제너레이터를 넘기면, 클라이언트 연결이 끊길 때 제너레이터가 취소되며 `asyncio.CancelledError`가 발생합니다. 다만 취소가 일어나려면 제너레이터 안 어딘가에서 `await`해야 합니다. 다음 예는 숫자를 끝없이 스트리밍하다가 클라이언트가 11번째 조각을 받은 뒤 연결을 끊으면 멈춥니다.

```python
import asyncio
import time
from typing import AsyncGenerator

import requests
from starlette.responses import StreamingResponse
from starlette.requests import Request

from ray import serve

@serve.deployment
class StreamingResponder:
    async def generate_forever(self) -> AsyncGenerator[str, None]:
        try:
            i = 0
            while True:
                yield str(i)
                i += 1
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print("Cancelled! Exiting.")

    def __call__(self, request: Request) -> StreamingResponse:
        gen = self.generate_forever()
        return StreamingResponse(gen, status_code=200, media_type="text/plain")

serve.run(StreamingResponder.bind())

r = requests.get("http://localhost:8000?max=10", stream=True)
start = time.time()
r.raise_for_status()
for i, chunk in enumerate(r.iter_content(chunk_size=None, decode_unicode=True)):
    print(f"Got result {round(time.time()-start, 1)}s after start: '{chunk}'")
    if i == 10:
        print("Client disconnecting")
        break
```

클라이언트 쪽은 `requests`의 `stream=True` 옵션으로 응답을 받고, `iter_content`로 도착한 조각을 하나씩 읽습니다. 실행하면 조각이 약 0.1초 간격으로 도착하고, 클라이언트가 루프를 빠져나가면 레플리카 로그에 `Cancelled! Exiting.`이 찍힙니다. 서버의 `await asyncio.sleep(0.1)`이 취소 지점 역할을 합니다. 취소 신호(`asyncio.CancelledError`)를 받아 정리 코드를 실행하려면 비동기 제너레이터를 넘기고 그 안에서 `await`해야 합니다.

### 요청 취소

스트리밍이 아니어도 취소는 일어납니다. 처리 시간이 엔드투엔드 타임아웃을 넘거나, 응답을 받기 전에 HTTP 클라이언트가 연결을 끊으면 Serve는 처리 중인 요청을 취소합니다. 요청이 어디까지 갔는지에 따라 동작이 다릅니다.

- 프록시가 아직 요청을 레플리카에 보내지 않았다면, 요청을 그냥 버립니다.
- 이미 레플리카에 보냈다면, 레플리카에서 처리기를 실행하는 `asyncio.Task`를 취소합니다. 그러면 처리기가 다음번 `await` 문에 들어설 때 `asyncio.CancelledError`가 발생합니다.

여기서 핵심은 취소가 `await` 지점에서만 효력을 내는 것입니다. 요청이 끝날 때까지 남은 `await`이 없으면 레플리카는 요청을 평소대로 끝까지 처리해 프록시에 응답을 보내고, 프록시는 그 응답을 버립니다. 앞서 만든 `Translator`가 바로 이 경우입니다. `translate` 안에는 `await`이 없으므로, 클라이언트가 이미 떠난 뒤에도 `model.generate`가 끝까지 실행된 다음 결과가 버려집니다. 블로킹 작업을 `await`할 수 있는 형태로 작성해야 Serve가 그 작업이 끝나기를 기다리지 않고 취소합니다.

> **핵심:** 취소는 프로세스를 강제로 멈추는 신호가 아니라 asyncio 작업에 전달되는 예외입니다. `await` 없이 이어지는 동기 추론 코드는 취소 요청을 받아도 끝까지 실행되므로, 끊긴 요청이 자원을 계속 차지합니다.

취소 시 정리 작업이 필요하면 `try`/`except asyncio.CancelledError`로 감쌉니다. 반대로 취소되면 안 되는 비동기 호출은 `asyncio.shield()`로 보호합니다. 다음 예에서 요청이 취소되면 `__call__`에는 취소 오류가 발생하지만 `snore()`는 끝까지 실행되어 `ZZZ`를 출력합니다.

```python
import asyncio
from ray import serve

@serve.deployment
class SnoringSleeper:
    async def snore(self):
        await asyncio.sleep(1)
        print("ZZZ")

    async def __call__(self):
        try:
            print("SnoringSleeper received request!")

            # Prevent the snore() method from being cancelled
            await asyncio.shield(self.snore())

        except asyncio.CancelledError:
            print("SnoringSleeper's request was cancelled!")

app = SnoringSleeper.bind()
```

취소는 처리 메서드가 띄운 하위 호출로도 전파됩니다. 인그레스가 호출한 다른 디플로이먼트, Ray 태스크, 액터 호출도 함께 취소되며, 각 처리기는 인그레스와 같은 방식으로 `asyncio.CancelledError`를 처리합니다. 단, `DeploymentHandle` 호출에는 `asyncio.shield`를 써도 하위 처리기의 취소를 막지 못합니다. 하위 처리기에서도 취소 오류를 직접 처리해야 합니다.

### keep-alive 타임아웃

Serve는 내부적으로 Uvicorn HTTP 서버로 요청을 처리하며, Uvicorn은 기본적으로 요청과 요청 사이에 HTTP 연결을 5초 동안 유지합니다. 클라이언트나 앞단 로드 밸런서가 연결을 더 오래 재사용하려 한다면, 서버가 먼저 연결을 닫아 요청이 실패할 수 있습니다. 이 값은 Serve 설정 파일의 `http_options` 필드에 `keep_alive_timeout_s`를 넣어 바꿉니다. 이 설정은 Ray 클러스터 전체에 적용되며 런타임 중에는 갱신할 수 없으므로, 앞단 구성 요소의 유휴 연결 타임아웃을 확인한 뒤 미리 정해 두기를 권합니다.

## DeploymentHandle로 여러 모델 조합하기

실제 서비스에서는 요청 하나를 모델 하나로 처리하지 못하는 경우가 많습니다. 긴 영어 문서를 받아 프랑스어 요약을 돌려주는 서비스라면 요약 모델이 먼저 문장을 줄이고, 번역 모델이 그 결과를 프랑스어로 옮겨야 합니다. 두 모델을 한 디플로이먼트 클래스에 함께 넣어도 동작은 합니다. 하지만 그러면 요약과 번역이 늘 같은 레플리카 수와 같은 자원 설정을 공유합니다. Ray Serve의 모델 컴포지션은 전처리, 모델 추론, 후처리 같은 단계를 각각 별도의 디플로이먼트로 나누고, 디플로이먼트끼리 DeploymentHandle로 서로를 호출하게 합니다.

### 요약 모델과 번역 모델을 잇는 애플리케이션

예제 파이프라인은 단계가 둘입니다. 영어 텍스트를 요약하고, 그 요약을 프랑스어로 번역합니다. 번역은 이미 `Translator`가 맡고 있으므로, 같은 t5-small 모델에 `"summarize: "` 접두사를 붙여 요약하는 `Summarizer`를 추가합니다. 두 모델을 한 애플리케이션으로 묶은 전체 코드는 다음과 같습니다.

```python
# File name: serve_quickstart_composed.py
from starlette.requests import Request

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

@serve.deployment
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

단일 모델 버전과 비교하면 역할이 바뀌었습니다. 요청이 가장 먼저 도착하는 곳이 `Summarizer`이므로 `__call__`은 이제 `Summarizer`에 있고, `Summarizer`가 인그레스 디플로이먼트가 됩니다. `Translator`는 HTTP 요청을 직접 받지 않으므로 `__call__`이 빠졌고, 대신 다른 디플로이먼트가 `translate` 메서드를 호출합니다. 데코레이터 인자는 모두 생략했으며, 필요해지면 두 디플로이먼트에 각각 다른 값을 줍니다.

마지막 줄 `Summarizer.bind(Translator.bind())`가 애플리케이션의 구조를 정합니다. 빈 생성자 인자로 `Translator`를 `bind`하고, 그 결과를 `Summarizer`의 생성자 인자로 넘깁니다. 실행은 단일 모델 때와 같이 파일이 있는 디렉터리에서 합니다.

```bash
serve run serve_quickstart_composed:app
```

클라이언트도 영어 텍스트를 JSON 본문에 담아 보냅니다.

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

애플리케이션이 실행 중일 때 다른 터미널에서 실행하면, 『두 도시 이야기』의 첫 구절이 요약된 뒤 프랑스어로 번역되어 돌아옵니다.

```console
$ python composed_client.py

C'était le meilleur des temps, c'était le pire des temps,
```

### bind 인자가 핸들로 바뀌는 과정

`Summarizer`의 생성자 시그니처는 `translator: DeploymentHandle`인데, `bind`에 넘긴 값은 `Translator.bind()`의 결과였습니다. 둘이 다른 객체인데도 동작하는 이유는 Ray Serve가 런타임에 인자를 바꿔 넣기 때문입니다. 다른 디플로이먼트를 `bind`한 결과를 생성자 인자로 넘기면, Serve는 런타임에 그 인자를 해당 디플로이먼트를 가리키는 DeploymentHandle로 대체해 생성자에 전달합니다. 그래서 `Summarizer` 레플리카 안의 `self.translator`는 `Translator` 디플로이먼트를 호출하는 핸들입니다.

이 방식에서는 디플로이먼트 사이의 연결을 설정 파일이 아니라 코드로 씁니다. 다른 모델 호출이 함수 호출처럼 보이고, 두 모델이 다른 자원을 쓰며 다른 머신에서 실행되더라도 어느 모델을 부를지 고르는 조건문이나 여러 번 부르는 반복문을 평범한 파이썬으로 씁니다.

핸들은 디플로이먼트 안에서만 쓰지 않습니다. `serve.run`도 애플리케이션 인그레스 디플로이먼트의 DeploymentHandle을 반환하므로, HTTP를 거치지 않고 파이썬 코드에서 애플리케이션을 직접 호출할 수 있어 테스트 코드에서 편리합니다. 지원이 중단되었던 `RayServeHandle`과 `RayServeSyncHandle` API는 Ray 2.10부터 완전히 제거되었습니다.

### `.remote()`와 DeploymentResponse

`self.translator.translate.remote(summary)`는 네 부분으로 나뉩니다. `self.translator`는 생성자로 받은 핸들이고, `translate`는 호출할 `Translator`의 메서드이며, `remote`는 이것이 다른 디플로이먼트를 향한 핸들 호출임을 나타냅니다. 괄호 안의 `summary`는 `translate`에 전달할 인자로, 위치 인자와 키워드 인자를 원하는 만큼 넘기면 핸들이 그대로 메서드에 전달합니다. 메서드 이름 없이 `handle.remote(...)`로 호출하면 그 디플로이먼트의 `__call__`이 실행됩니다.

`.remote()`는 결과를 기다리지 않고 즉시 반환됩니다. 반환값은 번역 문자열이 아니라 `DeploymentResponse` 객체로, 결과를 가리키는 퓨처(future)입니다. 이 응답을 `await`하면 원격 호출이 끝날 때까지 기다렸다가 반환값을 돌려줍니다. 호출과 `await`를 떼어 놓으면 호출을 먼저 여러 개 보내 두고 나중에 결과를 모읍니다.

```python
from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment
class Hello:
    def __call__(self) -> str:
        return "Hello"

@serve.deployment
class World:
    def __call__(self) -> str:
        return " world!"

@serve.deployment
class Ingress:
    def __init__(self, hello_handle: DeploymentHandle, world_handle: DeploymentHandle):
        self._hello_handle = hello_handle
        self._world_handle = world_handle

    async def __call__(self) -> str:
        hello_response = self._hello_handle.remote()
        world_response = self._world_handle.remote()
        return (await hello_response) + (await world_response)

app = Ingress.bind(Hello.bind(), World.bind())

handle: DeploymentHandle = serve.run(app)
assert handle.remote().result() == "Hello world!"
```

`Ingress.__call__`은 두 디플로이먼트에 요청을 모두 보낸 뒤에 결과를 기다립니다. 첫 번째 `await` 동안 두 번째 호출도 이미 실행 중이므로, 서로 의존하지 않는 모델 여러 개를 부르는 앙상블 같은 구조에서는 이 순서가 지연 시간을 줄입니다. 호출마다 바로 `await`하면 두 호출이 차례로 실행됩니다.

### await와 `.result()`의 차이

`DeploymentResponse`에서 값을 꺼내는 방법은 둘입니다. 비동기 코드에서는 `await`를, 동기 코드에서는 `.result()`를 씁니다. 위 예제의 마지막 줄처럼 디플로이먼트 바깥의 스크립트에서는 `.result()`가 자연스럽습니다. 문제는 디플로이먼트 안에서 `.result()`를 부를 때입니다.

`await`는 결과를 기다리는 동안 레플리카가 다른 요청을 처리하게 합니다. 반면 `.result()`는 원격 호출이 끝날 때까지 디플로이먼트가 다른 코드를 전혀 실행하지 못하게 막습니다. 인그레스 레플리카가 하위 모델의 응답을 기다리며 멈춰 있으면, 그 레플리카로 들어온 다른 요청은 하위 모델이 한가해도 처리되지 못합니다. 그래서 디플로이먼트 안에서는 `.result()` 대신 `await`를 써야 합니다.

| 방법 | 쓰는 위치 | 기다리는 동안 레플리카의 동작 |
|---|---|---|
| `await response` | 디플로이먼트의 `async` 메서드 안에서 씁니다. | 같은 레플리카가 다른 요청을 계속 처리합니다. |
| `response.result()` | 디플로이먼트 바깥의 동기 코드(테스트 스크립트 등)에서 씁니다. | 디플로이먼트 바깥에서는 문제가 없고, 안에서 부르면 호출이 끝날 때까지 레플리카가 멈춥니다. |
| 다른 핸들 호출에 `response`를 인자로 전달 | 디플로이먼트 안에서 파이프라인 단계를 이을 때 씁니다. | 직접 기다리지 않으며, Serve가 내부에서 결과를 기다려 다음 단계에 넘깁니다. |

같은 이유로 `Summarizer.__call__`의 `self.summarize(english_text)` 호출에도 주의해야 합니다. 핸들을 거치지 않는 동기 메서드라서 asyncio의 일반 동작대로 `generate`가 끝날 때까지 이벤트 루프를 붙잡습니다. 번역은 `await`로 기다리므로 그 동안에는 `Summarizer` 레플리카가 다른 요청을 받아들이지만, 현재 예제에서 요약 단계는 여전히 동기로 실행됩니다. 요약 단계까지 별도 디플로이먼트로 분리하면, 그 무거운 연산을 기다리는 지점도 `await` 가능한 핸들 호출로 바뀝니다.

### 응답을 그대로 넘겨 호출 연결하기

여러 단계의 파이프라인에서는 앞 단계의 응답을 `await`하지 않고, 한 핸들 호출이 반환한 `DeploymentResponse`를 다른 핸들 호출의 인자로 그대로 넘겨도 됩니다. 기다리는 일은 Serve가 내부에서 맡습니다. 첫 번째 호출이 끝나면 Serve는 `DeploymentResponse` 객체가 아니라 그 호출의 출력값을 두 번째 호출에 전달하므로, 받는 쪽 메서드는 평범한 값을 인자로 받습니다.

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

`Multiplier.__call__`의 `val`은 `Adder`가 반환한 정수 6이며, `Ingress`가 `await`하는 것은 마지막 응답 하나뿐입니다. 앞 단계 응답을 `await`하지 않아도 되므로 코드가 짧아지고, 인그레스는 파이프라인의 연결만 정의합니다. 요약 단계를 별도 디플로이먼트로 두면 요약-번역 파이프라인도 `self._translator.translate.remote(self._summarizer.summarize.remote(text))` 형태가 됩니다. 응답 여러 개를 한 호출의 인자로 함께 넘길 수도 있습니다. 두 `Adder`의 응답을 `Combiner.average.remote(...)`에 함께 넘기면 Serve가 두 출력값을 `average`에 전달합니다.

응답은 최상위 위치 인자나 키워드 인자로만 넘길 수 있습니다. 리스트나 딕셔너리 같은 중첩 객체 안에 넣어 넘기는 방식은 더 이상 지원하지 않으며, 그럴 때는 `DeploymentResponse._to_object_ref()`로 대응하는 Ray `ObjectRef`를 꺼내 넘깁니다. 각 `DeploymentResponse`는 내부적으로 Ray `ObjectRef`(스트리밍 호출이면 `ObjectRefGenerator`)에 대응하므로, 핸들 호출 결과를 Ray 태스크나 액터에 넘길 때도 같은 방법을 씁니다. 비동기 코드에서는 `await response._to_object_ref()`를, 동기 코드에서는 `_to_object_ref_sync`를 쓰며, 둘 다 개발자 API입니다.

### 스트리밍 핸들 호출

LLM 토큰 생성처럼 결과가 조금씩 만들어지는 디플로이먼트를 다른 디플로이먼트가 호출할 때는 스트리밍 핸들 호출을 씁니다. 호출되는 메서드가 제너레이터여야 하고, 핸들에 `handle.options(stream=True)`를 설정해야 합니다. 그러면 핸들 호출은 단일 `DeploymentResponse` 대신 `DeploymentResponseGenerator`를 반환하고, 이 객체를 동기 또는 비동기 제너레이터처럼 순회합니다.

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

`Caller`는 생성자에서 `options(stream=True)`를 적용한 핸들을 저장해 두고, `__call__`에서 `async for`로 `Streamer`의 출력을 받는 대로 다시 내보냅니다. 스크립트 끝의 테스트 코드는 디플로이먼트 바깥이라 `list(r)`처럼 동기 제너레이터로 써도 됩니다. 디플로이먼트 안에서는 `.result()`와 같은 이유로 동기 제너레이터로 쓰지 말아야 하며, 그러면 그 레플리카에서 다른 요청이 동시에 실행되지 못합니다. `DeploymentResponseGenerator`는 다른 핸들 호출의 인자로 넘길 수 없으므로, 스트림을 다음 단계로 넘기려면 위 예제처럼 직접 순회하며 전달합니다. HTTP 클라이언트까지 흘려보내려면 앞서 설명한 `StreamingResponse`에 비동기 제너레이터를 넘기는 방식과 조합합니다.

### 단계별로 따로 확장하기

언어별 응답 디플로이먼트 두 개로 요청을 나누는 애플리케이션에서 요청의 75%가 스페인어이고 25%가 프랑스어라면, 스페인어 디플로이먼트는 레플리카 3개로, 프랑스어 디플로이먼트는 1개로 두어 부하에 맞춥니다. 종류와 양이 서로 다른 자원을 쓰는 모델과 비즈니스 로직을 디플로이먼트로 나누면 애플리케이션 수준의 병목을 피할 수 있습니다.

## TensorFlow, PyTorch, scikit-learn 모델 서빙하기

앞서 t5-small `Translator`를 디플로이먼트로 바꾼 과정에는 Hugging Face나 PyTorch에만 해당하는 부분이 없었습니다. 생성자에서 모델을 적재하고, `__call__`에서 HTTP 요청을 모델 입력으로 바꿔 추론한 뒤, JSON으로 직렬화할 수 있는 값을 돌려줍니다. Ray Serve는 특정 프레임워크에 묶이지 않으므로 이 구조는 TensorFlow, PyTorch, scikit-learn의 어떤 버전에도 그대로 적용됩니다. 프레임워크마다 달라지는 부분은 두 가지입니다. 모델을 어디서 불러오는지, 요청 본문을 어떤 형식으로 받는지입니다.

| 구분 | TensorFlow(Keras) | PyTorch(torchvision) | scikit-learn |
|---|---|---|---|
| 예제 모델 | MNIST 분류용 신경망 | 사전 학습된 ResNet18 | Iris 데이터 `GradientBoostingClassifier` |
| 모델 적재 위치 | 파일 시스템의 `.h5` 파일 | torchvision 모델 저장소 | 파일 시스템의 pickle 파일 |
| `bind` 인자 | 모델 경로 | 없음 | 모델 경로와 레이블 경로 |
| 요청 해석 | `await request.json()` | `await request.body()` | `await request.json()` |
| 응답 | 로짓 배열과 모델 파일 경로 | 예측 클래스 인덱스 | 사람이 읽는 클래스 이름 |

각 예제에 필요한 라이브러리는 `"ray[serve]"`, `requests`와 함께 설치합니다.

```bash
pip install "tensorflow>=2.0" requests "ray[serve]"
pip install torch torchvision requests "ray[serve]"
pip install scikit-learn requests "ray[serve]"
```

### TensorFlow: 저장한 Keras 모델을 경로로 받기

`tutorial_tensorflow.py`는 Keras로 간단한 MNIST 모델을 학습해 임시 디렉터리에 저장합니다. 파일이 이미 있으면 학습을 건너뛰므로, 모듈을 다시 불러와도 학습이 반복되지 않습니다.

```python
from ray import serve

import os
import tempfile
import numpy as np
from starlette.requests import Request
from typing import Dict

import tensorflow as tf

TRAINED_MODEL_PATH = os.path.join(tempfile.gettempdir(), "mnist_model.h5")

def train_and_save_model():
    # Load mnist dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Train a simple neural net model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=1)

    model.evaluate(x_test, y_test, verbose=2)
    model.summary()

    # Save the model in h5 format in local file system
    model.save(TRAINED_MODEL_PATH)

if not os.path.exists(TRAINED_MODEL_PATH):
    train_and_save_model()

@serve.deployment
class TFMnistModel:
    def __init__(self, model_path: str):
        import tensorflow as tf

        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)

    async def __call__(self, starlette_request: Request) -> Dict:
        # Step 1: transform HTTP request -> tensorflow input
        # Here we define the request schema to be a json array.
        input_array = np.array((await starlette_request.json())["array"])
        reshaped_array = input_array.reshape((1, 28, 28))

        # Step 2: tensorflow input -> tensorflow output
        prediction = self.model(reshaped_array)

        # Step 3: tensorflow output -> web output
        return {"prediction": prediction.numpy().tolist(), "file": self.model_path}

mnist_model = TFMnistModel.bind(TRAINED_MODEL_PATH)
```

`__call__`의 주석대로 요청을 TensorFlow 입력으로 바꾸고, 추론하고, 출력을 웹 응답으로 바꾸는 세 단계입니다. 요청 스키마를 `{"array": [...]}` 형태의 JSON으로 정하는 것도 디플로이먼트 코드의 몫입니다. 모델 경로는 `bind`의 생성자 인자로 넘기므로, 다른 체크포인트를 서빙하려면 `bind`에 넘기는 값만 바꾸면 됩니다. 응답에 `file` 필드가 있어서 클라이언트가 응답한 모델 파일을 바로 알 수 있습니다.

디플로이먼트를 배포하면 Ray Serve가 클래스를 인스턴스화하면서 파일 시스템에서 모델을 적재합니다.

```bash
serve run tutorial_tensorflow:mnist_model
```

```python
import requests
import numpy as np

resp = requests.get(
    "http://localhost:8000/", json={"array": np.random.randn(28 * 28).tolist()}
)
print(resp.json())
```

디플로이먼트가 HTTP 요청을 받으면 Ray Serve가 기본적으로 `__call__`을 호출하므로, 이 예제처럼 본문을 담은 GET 요청도 처리합니다. 응답에는 열 개 클래스의 로짓 배열과 모델 파일 경로가 담기며, 예측값은 실행마다 달라집니다.

### PyTorch: 바이트 본문으로 이미지 받기

PyTorch 예제는 학습 없이 torchvision의 사전 학습 ResNet18을 생성자에서 직접 불러오므로 `bind`에 넘길 인자가 없습니다. 요청 본문은 `request.body()`로 원시 바이트 그대로 읽습니다.

```python
# File name: tutorial_pytorch.py
from ray import serve

from io import BytesIO
from PIL import Image
from starlette.requests import Request
from typing import Dict

import torch
from torchvision import transforms
from torchvision.models import resnet18

@serve.deployment
class ImageModel:
    def __init__(self):
        self.model = resnet18(pretrained=True).eval()
        self.preprocessor = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t[:3, ...]),  # remove alpha channel
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    async def __call__(self, starlette_request: Request) -> Dict:
        image_payload_bytes = await starlette_request.body()
        pil_image = Image.open(BytesIO(image_payload_bytes))
        print("[1/3] Parsed image data: {}".format(pil_image))

        pil_images = [pil_image]  # Our current batch size is one
        input_tensor = torch.cat(
            [self.preprocessor(i).unsqueeze(0) for i in pil_images]
        )
        print("[2/3] Images transformed, tensor shape {}".format(input_tensor.shape))

        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        print("[3/3] Inference done!")
        return {"class_index": int(torch.argmax(output_tensor[0]))}

image_model = ImageModel.bind()
```

전처리 파이프라인도 생성자에서 한 번만 만듭니다. `eval()`로 배치 정규화가 학습 때 쌓은 통계를 쓰게 하고, `torch.no_grad()`로 그래디언트 계산을 끕니다. 이미지를 리스트에 담아 `torch.cat`으로 합치는 구조는 지금은 배치 크기 1이지만 요청을 묶어 처리할 때 그대로 이어집니다. 텐서는 JSON으로 직렬화되지 않으므로 `int(...)`로 파이썬 기본 타입으로 바꿔 돌려줍니다.

```bash
serve run tutorial_pytorch:image_model
```

```python
import requests

ray_logo_bytes = requests.get(
    "https://raw.githubusercontent.com/ray-project/"
    "ray/master/doc/source/images/ray_header_logo.png"
).content

resp = requests.post("http://localhost:8000/", data=ray_logo_bytes)
print(resp.json())
```

클라이언트는 `json=` 대신 `data=`로 PNG 바이트를 본문에 싣습니다. 응답은 `{'class_index': ...}` 형태이며 숫자는 환경에 따라 다를 수 있습니다.

### scikit-learn: 모델과 레이블 파일을 함께 적재하기

scikit-learn 예제는 Iris 데이터로 `GradientBoostingClassifier`를 학습해 모델은 pickle로, 클래스 이름 목록은 JSON으로 따로 저장합니다. `data`와 `target`을 따로 섞으면 특성과 레이블의 짝이 어긋나므로, 아래 코드는 순열 하나로 두 배열을 함께 섞습니다.

```python
# File name: tutorial_sklearn.py
from ray import serve

import pickle
import json
import numpy as np
import os
import tempfile
from starlette.requests import Request
from typing import Dict

from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error

model = GradientBoostingClassifier()

iris_dataset = load_iris()
data, target, target_names = (
    iris_dataset["data"],
    iris_dataset["target"],
    iris_dataset["target_names"],
)

# Shuffle features and labels with the same permutation
idx = np.random.permutation(len(data))
data, target = data[idx], target[idx]
train_x, train_y = data[:100], target[:100]
val_x, val_y = data[100:], target[100:]

model.fit(train_x, train_y)
print("MSE:", mean_squared_error(model.predict(val_x), val_y))

# Save the model and label to file
MODEL_PATH = os.path.join(
    tempfile.gettempdir(), "iris_model_gradient_boosting_classifier.pkl"
)
LABEL_PATH = os.path.join(tempfile.gettempdir(), "iris_labels.json")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
with open(LABEL_PATH, "w") as f:
    json.dump(target_names.tolist(), f)

@serve.deployment
class BoostingModel:
    def __init__(self, model_path: str, label_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(label_path) as f:
            self.label_list = json.load(f)

    async def __call__(self, starlette_request: Request) -> Dict:
        payload = await starlette_request.json()
        print("Worker: received starlette request with data", payload)

        input_vector = [
            payload["sepal length"],
            payload["sepal width"],
            payload["petal length"],
            payload["petal width"],
        ]
        prediction = self.model.predict([input_vector])[0]
        human_name = self.label_list[prediction]
        return {"result": human_name}

boosting_model = BoostingModel.bind(MODEL_PATH, LABEL_PATH)
```

여기서는 추론 앞뒤의 코드가 핵심입니다. 디플로이먼트가 이름 붙은 필드 네 개를 모델이 기대하는 순서의 특성 벡터로 조립하고, 정수 클래스 인덱스를 레이블 목록으로 `"versicolor"` 같은 이름으로 바꿉니다. "텐서 입력, 텐서 출력" 방식이라면 클라이언트나 별도 서비스가 맡았을 변환을 모델과 같은 클래스에 둡니다. 필드가 빠지면 `payload[...]`에서 `KeyError`가 나므로, 필드가 많아지면 앞서 설명한 FastAPI 통합으로 HTTP 파싱과 검증을 맡기는 편이 낫습니다.

학습 코드가 모듈 최상위에 있어서 `serve run`이 모듈을 불러올 때마다 학습과 저장이 다시 실행됩니다. 실제 서비스에서는 학습 스크립트와 서빙 모듈을 나누고, 서빙 모듈은 저장된 파일만 읽게 하기를 권합니다.

```bash
serve run tutorial_sklearn:boosting_model
```

```python
import requests

sample_request_input = {
    "sepal length": 1.2,
    "sepal width": 1.0,
    "petal length": 1.1,
    "petal width": 0.9,
}
response = requests.get("http://localhost:8000/", json=sample_request_input)
print(response.text)
```

응답은 `{"result": ...}` 형태이고 값은 Iris 품종 이름 가운데 하나입니다.

세 예제 모두 디플로이먼트 안의 프레임워크 코드는 생성자와 `__call__`에만 있고, 데코레이터, `bind`, `serve run`, 클라이언트 호출 방식은 같습니다. 그래서 각 모델을 디플로이먼트로 만든 뒤 DeploymentHandle로 한 애플리케이션에 묶을 수 있습니다. Ray Serve는 모델별 최적화를 하지 않으므로 추론 속도 최적화는 직접 해야 하며, PyTorch JIT으로 컴파일한 모델이나 ONNXRuntime 모델도 Ray Serve에서 실행할 수 있습니다.

## 아키텍처: 컨트롤러, 프록시, 요청 흐름

`serve run` 한 줄로 띄운 애플리케이션은 내부에서 여러 Ray 액터로 나뉘어 동작합니다. 액터마다 맡은 일을 알면 `max_ongoing_requests`나 프록시 위치 같은 설정이 무엇을 바꾸는지, 장애가 나면 어디까지 복구되는지 판단할 수 있습니다.

### Serve 인스턴스를 이루는 액터

아래 구조도에서는 두 경로가 갈라지는 부분을 눈여겨보면 됩니다. 하나는 컨트롤러가 프록시와 레플리카를 관리하는 제어 경로이고, 다른 하나는 클라이언트 요청이 프록시나 핸들을 거쳐 레플리카로 가는 데이터 경로입니다.

![Serve 인스턴스의 컨트롤러, 프록시, 레플리카와 요청 경로를 보여 주는 Ray Serve 아키텍처 구조도](https://docs.ray.io/en/latest/_images/architecture-2.0.svg)

| 액터 | 개수 | 맡은 일 |
|---|---|---|
| 컨트롤러(controller) | Serve 인스턴스마다 하나인 전역 액터입니다. | 제어 평면을 관리하며 다른 액터를 생성·갱신·삭제합니다. 디플로이먼트를 생성하거나 조회하는 Serve API를 호출하면 컨트롤러를 원격 호출합니다. |
| HTTP 프록시 | `proxy_location` 설정에 따라 정해집니다. | Uvicorn HTTP 서버로 요청을 받아 레플리카로 전달하고, 처리가 끝나면 응답합니다. |
| gRPC 프록시 | 유효한 gRPC 설정(`port`, `grpc_servicer_functions`)이 있을 때 HTTP 프록시와 함께 실행됩니다. | grpcio 서버로 요청을 받아 레플리카로 전달하고 응답합니다. |
| 레플리카 | 디플로이먼트마다 여러 개를 둘 수 있으며, `num_replicas`로 개수를 정합니다. | 요청에 대응하는 코드를 실제로 실행하며, `@serve.batch`로 요청을 묶어 처리하기도 합니다. |

컨트롤러는 요청 경로에 끼지 않습니다. 컨트롤러는 다른 액터의 생성·갱신·삭제를 맡고, 실제 요청은 프록시나 DeploymentHandle에서 레플리카로 곧장 갑니다. 제어 평면(control plane)과 데이터 평면(data plane)이 나뉘어 있어서 컨트롤러가 잠시 멈춰도 트래픽은 계속 처리됩니다.

### 요청 하나의 수명

HTTP나 gRPC 요청이 프록시에 도착하면 다음 순서로 처리됩니다.

1. 프록시가 요청을 받아 파싱합니다.
2. HTTP URL 경로나 애플리케이션 이름 메타데이터로 해당 디플로이먼트를 찾고, 요청을 큐에 넣습니다.
3. 큐의 요청마다 받을 수 있는 레플리카를 찾아 보냅니다. 모든 레플리카에 처리 중인 요청이 `max_ongoing_requests`개를 넘으면, 여유가 생길 때까지 요청은 큐에서 기다립니다.

레플리카도 자체 큐를 두고 요청을 하나씩 실행하며, `asyncio`로 여러 요청을 동시에 처리하기도 합니다. 핸들러(디플로이먼트 함수나 `__call__`)를 `async def`로 선언하면 레플리카는 핸들러 실행을 기다리지 않습니다. 일반 `def`라면 핸들러가 반환할 때까지 블로킹됩니다.

DeploymentHandle로 보내는 요청은 1, 2단계를 건너뛰고 핸들 내부 큐에 들어가 3단계부터 진행합니다. 핸들은 같은 노드의 "라우터"를 감싸고 있습니다. 그래서 레플리카 사이의 호출도 HTTP·gRPC 요청과 같은 데이터 경로를 지나며, 디플로이먼트 선택과 배칭 절차가 똑같이 적용됩니다. 작은 요청 객체는 액터 사이에 네트워크 호출로 직접 전달됩니다. 100KiB 이상인 큰 객체는 Ray의 공유 메모리 오브젝트 스토어에 기록되고, 레플리카가 제로 카피(zero-copy)로 읽습니다.

### power of two choices 라우팅

DeploymentHandle은 컨트롤러를 계속 폴링해 새 레플리카를 확인합니다. 새 레플리카를 발견하면 쌓여 있던 요청과 새 요청을 `max_ongoing_requests`에 이를 때까지 그 레플리카로 보냅니다. 레플리카는 power of two choices 방식으로 고르며, 어떤 레플리카도 동시에 `max_ongoing_requests`개를 넘는 요청을 처리하지 않습니다. 무작위로 고른 후보 둘만 비교해 덜 바쁜 쪽에 보내므로, 라우터가 전역 상태 없이도 특정 레플리카로 요청이 쏠리는 현상을 줄입니다.

### 오토스케일링이 동작하는 위치

오토스케일러는 컨트롤러 액터 안에서 실행됩니다. 각 DeploymentHandle과 레플리카는 주기적으로 지표를 오토스케일러에 보냅니다. 오토스케일러는 디플로이먼트마다 핸들 큐와 레플리카에서 처리 중인 요청을 확인해 레플리카 수를 조정할지 정합니다.

오토스케일러가 컨트롤러 안에 있어서, 컨트롤러가 죽어도 HTTP, gRPC, DeploymentHandle 요청은 계속 처리되지만 오토스케일링은 멈춥니다. 컨트롤러가 복구되면 오토스케일링도 다시 시작되지만, 그 전에 모은 지표는 모두 사라집니다.

### 장애 허용

모델 평가 코드의 예외 같은 애플리케이션 오류는 Serve가 잡아서 감쌉니다. 클라이언트는 트레이스백과 함께 500 상태 코드를 받고, 레플리카는 계속 요청을 처리합니다. 머신 장애는 액터 계층을 따라 복구됩니다.

| 장애 대상 | 복구 동작 |
|---|---|
| 레플리카 액터 | 컨트롤러가 새 레플리카로 교체합니다. |
| 프록시 액터 | 컨트롤러가 다시 시작합니다. |
| 컨트롤러 액터 | Ray가 다시 시작합니다. |
| 노드 또는 클러스터(KubeRay RayService 사용 시) | KubeRay가 복구합니다. |
| Ray 클러스터 전체(KubeRay 미사용 시) | Ray Serve가 복구하지 못합니다. |

죽은 머신의 액터는 다른 머신에서 자동으로 다시 시작됩니다. 라우팅 정책, 디플로이먼트 설정 같은 컨트롤러 데이터는 헤드 노드의 Ray GCS(Global Control Store)에 체크포인트로 저장되어 남습니다. 하지만 라우터와 레플리카의 네트워크 연결이나 내부 요청 큐 같은 일시적 데이터는 사라집니다. 처리 중이던 요청을 잃을 수 있으므로 중요한 요청은 클라이언트 쪽에 재시도를 두는 편이 안전합니다. 클러스터 장애를 피하려면 KubeRay의 GCS 장애 허용 기능을 씁니다.

### 프록시 배치와 수평 확장

프록시 위치는 `serve.start()`나 Serve 설정 파일의 `proxy_location`으로 정합니다. 기본값 `EveryNode`는 레플리카가 하나 이상 있는 모든 노드에 프록시를 띄우고, `HeadOnly`는 헤드 노드에만 하나를 둡니다. DeploymentHandle로만 호출한다면 `Disabled`로 프록시를 끕니다.

```python
from ray import serve
from ray.serve.config import ProxyLocation

serve.start(proxy_location=ProxyLocation.EveryNode)
```

노드마다 있는 프록시는 모두 같은 포트에 바인딩되므로 어느 노드로 보내도 모든 모델에 도달합니다. 그 위에 직접 로드 밸런서를 둡니다. HTTP·gRPC 인그레스는 노드를 추가해, 모델 추론은 `num_replicas`를 늘려 서로 독립적으로 확장합니다.

## 설정 파일 기반 배포

`serve run model:translator_app`처럼 import 경로로 실행하는 방식은 개발 단계에 편합니다. 하지만 운영 환경에서는 레플리카 수, 자원, 프록시 위치, 모델 언어 같은 값이 데코레이터 인자와 명령행 인자에 흩어져 있으면 지금 클러스터에서 무엇이 어떤 설정으로 돌고 있는지 한눈에 알기 어렵습니다. Ray Serve는 이 값들을 YAML 파일 하나에 모으는 방식을 운영 배포의 권장 방법으로 둡니다. Serve 설정 파일은 클러스터의 단일 진실 공급원(single source of truth)입니다. 프록시 같은 시스템 수준 구성과 디플로이먼트별 매개변수를 한곳에 적고, 파일을 고쳐 다시 적용하면 클러스터가 그 상태를 따라가는 선언적 방식으로 동작합니다. 개별 디플로이먼트 매개변수는 애플리케이션을 재배포하거나 재시작하지 않고 설정 파일만 수정해서 바꿀 수 있습니다.

VM에서 운영한다면 이 설정 파일을 `serve deploy` 명령으로 적용하고, Kubernetes에서 운영한다면 같은 내용을 RayService 리소스에 넣습니다. 어느 쪽이든 출발점은 같은 YAML 파일입니다.

### 예제 애플리케이션: 언어와 요약 길이를 바꿀 수 있는 파이프라인

예제는 앞서 만든 요약·번역 파이프라인과 구조가 같고, 두 디플로이먼트에 외부 설정으로 바꿀 상태와 `reconfigure` 메서드를 더했습니다. `Translator`는 번역 대상 언어를, `Summarizer`는 요약문의 최소·최대 길이를 설정으로 받습니다. 기존 파일에서 바뀌는 부분만 보이면 다음과 같으며, 완성한 파일을 `text_ml.py`로 저장합니다.

```python
# File name: text_ml.py (changes to the composed pipeline)
from typing import Dict

@serve.deployment
class Translator:
    def __init__(self):
        self.language = "french"
        self.prefix = "translate English to French: "
        # ... load the t5-small tokenizer and model as before ...

    def translate(self, text: str) -> str:
        input_ids = self.tokenizer(
            f"{self.prefix}{text}", return_tensors="pt"
        ).input_ids
        # ... generate and decode as before ...

    def reconfigure(self, config: Dict):
        self.language = config.get("language", "french")

        if self.language.lower() == "french":
            self.prefix = "translate English to French: "
        elif self.language.lower() == "german":
            self.prefix = "translate English to German: "
        elif self.language.lower() == "romanian":
            self.prefix = "translate English to Romanian: "
        else:
            pass

@serve.deployment
class Summarizer:
    def __init__(self, translator: DeploymentHandle):
        # ... load the t5-small tokenizer and model as before ...
        self.translator = translator
        self.min_length = 5
        self.max_length = 15

    def summarize(self, text: str) -> str:
        input_ids = self.tokenizer(f"summarize: {text}", return_tensors="pt").input_ids
        output_ids = self.model.generate(
            input_ids,
            num_beams=4,
            early_stopping=True,
            min_length=self.min_length,
            max_length=self.max_length,
        )
        # ... decode as before ...

    def reconfigure(self, config: Dict):
        self.min_length = config.get("min_length", 5)
        self.max_length = config.get("max_length", 15)

app = Summarizer.bind(Translator.bind())
```

생성자는 기본값(프랑스어, 길이 5~15)으로 상태를 초기화하고, `reconfigure`는 설정으로 받은 값으로 그 상태를 덮어씁니다. 모델 가중치 적재처럼 무거운 일은 생성자에만 두고 `reconfigure`에는 가벼운 속성 갱신만 둡니다. 그래야 설정을 바꿀 때 모델을 다시 적재하지 않습니다.

### serve build로 설정 파일 생성하기

설정 파일을 처음부터 손으로 쓸 필요는 없습니다. `serve build`는 애플리케이션의 import 경로를 받아 코드를 불러온 뒤, 애플리케이션에 들어 있는 디플로이먼트와 그 매개변수를 모두 담은 설정 파일을 만듭니다.

```bash
serve build text_ml:app -o serve_config.yaml
```

`-o`(`--output-path`)를 생략하면 YAML이 표준 출력으로 나옵니다. 모듈을 찾을 디렉터리는 `-d`(`--app-dir`)로 지정하며 기본값은 현재 디렉터리 `.`입니다. import 경로를 여러 개 넘기면 애플리케이션 여러 개를 담은 설정이 만들어집니다. 생성된 파일은 다음과 같습니다.

```yaml
proxy_location: EveryNode

http_options:
  host: 0.0.0.0
  port: 8000

grpc_options:
  port: 9000
  grpc_servicer_functions: []

logging_config:
  encoding: TEXT
  log_level: INFO
  logs_dir: null
  enable_access_log: true

applications:
- name: default
  route_prefix: /
  import_path: text_ml:app
  runtime_env: {}
  deployments:
  - name: Translator
    num_replicas: 1
    user_config:
      language: french
  - name: Summarizer
```

`serve build`는 애플리케이션 이름을 자동으로 붙이며, 이 이름은 바꿔도 됩니다. HTTP·gRPC 옵션도 기본값으로 채워 넣으므로 필요하면 수정합니다. 생성된 파일은 출발점이고, 운영에 맞게 손보는 것이 원래 쓰임새입니다.

### 설정 파일의 구조

최상위 필드는 다섯 개입니다. 앞의 넷은 클러스터 전체에, 마지막 `applications`는 애플리케이션별로 적용됩니다.

| 필드 | 적용 범위와 내용 |
|---|---|
| `proxy_location` | 프록시를 띄울 위치이며 기본값은 `EveryNode`입니다. |
| `http_options` | HTTP 프록시의 `host`, `port`, `request_timeout_s`, `keep_alive_timeout_s`입니다. 클러스터 전역 설정이라 런타임에 바꿀 수 없습니다. |
| `grpc_options` | gRPC 프록시의 `port`, `grpc_servicer_functions`, `request_timeout_s`입니다. 역시 전역 설정이며 런타임에 바꿀 수 없습니다. |
| `logging_config` | 컨트롤러, 프록시, 레플리카 로그의 전역 설정입니다. 애플리케이션·디플로이먼트 수준 설정이 있으면 그쪽이 우선합니다. |
| `applications` | 배포할 애플리케이션 목록입니다. |

설정 파일에서 `host`의 기본값은 `0.0.0.0`이라 디플로이먼트가 외부에 공개되며, Kubernetes에서 클러스터 밖으로 노출하려면 반드시 `0.0.0.0`이어야 합니다. `port` 기본값은 HTTP가 `8000`, gRPC가 `9000`입니다. `request_timeout_s`는 요청을 끝내고 다른 레플리카에서 재시도하기 전까지의 종단 간 타임아웃이며, 기본적으로는 타임아웃이 없습니다. `grpc_servicer_functions`가 빈 목록이면 gRPC 서버는 시작되지 않습니다.

`applications`의 각 항목에는 다음 필드가 들어갑니다.

| 필드 | 의미 |
|---|---|
| `name` | 애플리케이션 이름입니다. 애플리케이션마다 고유해야 합니다. |
| `route_prefix` | HTTP로 호출할 경로 접두사입니다. 기본값은 `/`이고 애플리케이션마다 고유해야 합니다. |
| `import_path` | 최상위 디플로이먼트의 경로이며 `serve run`에 넘기던 값과 같습니다. 가장 작은 설정 파일은 이 필드 하나로 이루어집니다. |
| `runtime_env` | 애플리케이션이 실행될 런타임 환경입니다. |
| `external_scaler_enabled` | 클러스터 밖에서 REST API로 디플로이먼트를 확장하는 외부 확장 API를 켭니다. 켜면 내장 오토스케일링(`autoscaling_config`)을 쓸 수 없습니다. 기본값은 `False`입니다. |
| `deployments` | `@serve.deployment`에 적은 설정을 덮어쓰는 디플로이먼트별 옵션 목록입니다. 선택 사항입니다. |
| `args` | 애플리케이션 빌더에 넘기는 인자입니다. |

`deployments` 목록은 디플로이먼트를 정의하지 않고 코드의 설정을 덮어씁니다. 각 항목의 `name`은 코드의 디플로이먼트 이름과 일치해야 하고, 항목 하나하나가 선택 사항입니다. 앞의 설정에서 `Summarizer` 항목을 통째로 지워도 파일은 유효하며, 배포하면 `Summarizer`는 데코레이터에 적힌 값으로 올라갑니다. `deployments` 섹션 전체를 생략하면 모든 디플로이먼트가 코드에 적힌 매개변수로 실행됩니다. 코드의 데코레이터가 기본값이고 설정 파일의 값이 그 위를 덮으므로, 운영에서 자주 바꾸는 값(레플리카 수, `user_config`)만 설정 파일에 두고 나머지는 코드에 맡겨도 됩니다. 앞서 만든 `num_replicas=2`, `ray_actor_options` 버전을 `serve build`로 변환하면 그 값이 다음처럼 `deployments` 항목으로 옮겨집니다.

```yaml
  deployments:
  - name: Translator
    num_replicas: 2
    ray_actor_options:
      num_cpus: 0.2
      num_gpus: 0.0
```

### runtime_env로 의존성 묶기

`serve build`가 만든 파일에서 `runtime_env`는 항상 `{}`로 비어 있으므로 직접 채워야 합니다. 런타임 환경(runtime environment)은 애플리케이션이 실행될 환경을 정의하며, `pip` 패키지 같은 의존성을 애플리케이션 단위로 묶습니다. `text_ml.py`는 `torch`와 `transformers`가 필요하므로, 클러스터에 이 패키지가 전역으로 설치되어 있지 않다면 다음처럼 적습니다.

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

제약이 두 가지 있습니다. 첫째, `runtime_env`를 지정했다면 그 런타임 환경 안에서 `import_path`를 불러올 수 있어야 합니다. 둘째, 설정 파일의 `runtime_env`는 `working_dir`와 `py_modules`에 원격 URI만 쓸 수 있고 로컬 zip 파일이나 디렉터리는 쓰지 못합니다. 설정 파일은 원격 클러스터에 그대로 전달되므로, 배포 명령을 실행한 머신의 로컬 경로는 클러스터에서 찾을 수 없다고 이해하면 됩니다. 코드를 클러스터에 전달하려면 S3 버킷이나 GitHub 아카이브 같은 원격 zip 파일의 URI를 `working_dir`에 적습니다.

```yaml
  runtime_env:
    working_dir: "https://github.com/ray-project/serve_config_examples/archive/HEAD.zip"
```

`serve deploy`에도 `--runtime-env`, `--runtime-env-json`, `--working-dir` 옵션이 있지만, import 경로로 배포할 때만 쓰이고 설정 파일로 배포할 때는 무시됩니다. 설정 파일을 쓴다면 런타임 환경도 파일 안에 적습니다.

### serve deploy로 적용하고 serve status로 확인하기

로컬에서 Ray 클러스터를 띄우고 설정 파일을 적용합니다.

```bash
# Start a local Ray cluster.
ray start --head

# Deploy the Text ML application to the local Ray cluster.
serve deploy serve_config.yaml
```

```text
2022-08-16 12:51:22,043 SUCC scripts.py:180 --
Sent deploy request successfully!
 * Use `serve status` to check deployments' statuses.
 * Use `serve config` to see the running app's config.
```

`serve run`과 달리 `serve deploy`는 터미널을 점유하지 않습니다. 실행 중인 Ray 클러스터의 대시보드 주소로 REST API 요청을 보내고 바로 끝나며, 주소 기본값은 `http://localhost:8265`입니다. 원격 클러스터라면 `-a`(`--address`)나 `RAY_DASHBOARD_ADDRESS` 환경 변수로 주소를 지정합니다. 요청을 보냈다는 메시지가 배포 완료를 뜻하지는 않으므로 실제 상태는 `serve status`로 확인합니다.

```text
$ serve status
proxies:
  cef533a072b0f03bf92a6b98cb4eb9153b7b7c7b7f15954feb2f38ec: HEALTHY
applications:
  default:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1694041157.2211847
    deployments:
      Translator:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
      Summarizer:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
```

애플리케이션이 `RUNNING`이고 디플로이먼트가 `HEALTHY`이면 앞서 쓰던 요약·번역 클라이언트로 `http://127.0.0.1:8000/`에 요청을 보낼 수 있습니다. 애플리케이션 하나만 보려면 `-n`(`--name`)으로 이름을 지정합니다.

`serve deploy`는 선언적으로 동작합니다. 클러스터에 Serve가 이미 실행 중이면 새 설정에 없는 애플리케이션을 모두 제거하고, 실행 중이 아니면 Serve를 시작합니다. 설정 파일에서 애플리케이션 하나를 빼고 다시 적용하면 그 애플리케이션이 내려가므로, 여러 애플리케이션을 운영할 때는 파일 하나에 전부 적어 둡니다. 업데이트도 같은 방식으로, 파일을 고친 뒤 `serve deploy`를 다시 실행합니다.

로컬 테스트에서는 같은 파일을 `serve run serve_config.yaml`로 실행해도 됩니다. YAML을 넘기면 코드가 바뀌지 않은 기존 애플리케이션은 갱신하지 않으며, `--route-prefix`와 `--name` 옵션은 무시됩니다. 개발 중에는 `serve run`으로 반복하고, 운영에 넘길 때 `serve build`로 파일을 만들어 다듬은 뒤 `serve deploy`로 적용하는 흐름을 권합니다.

### user_config와 reconfigure: 재시작 없이 설정 바꾸기

`user_config`는 디플로이먼트에 구조화된 설정을 넘기는 필드입니다. JSON으로 직렬화할 수 있는 객체라면 무엇이든 YAML에 적을 수 있고, Serve는 이 값을 실행 중인 레플리카와 앞으로 만들어질 레플리카에 모두 적용합니다. 적용할 때 레플리카를 재시작하지 않으므로 모델을 다시 적재하는 비용도, 그동안의 처리 공백도 생기지 않습니다.

이 기능을 쓰려면 디플로이먼트에 `reconfigure` 메서드를 구현합니다. 인자는 딕셔너리, 리스트, 문자열 같은 JSON 직렬화 가능 객체 하나뿐입니다. 호출 시점은 둘입니다. 디플로이먼트를 만들 때(데코레이터나 설정 파일에서) `user_config`가 지정되어 있으면 Serve는 `__init__` 직후에 `reconfigure`를 호출해 그 값을 넘깁니다. 이후 설정 파일의 `user_config`를 새 값으로 고쳐 클러스터에 다시 적용하면, 레플리카는 그대로 두고 `reconfigure`만 다시 호출합니다.

번역 언어를 독일어로 바꾸려면 설정 파일에서 한 줄만 고쳐 다시 적용합니다.

```yaml
  deployments:
  - name: Translator
    num_replicas: 1
    user_config:
      language: german
  - name: Summarizer
    num_replicas: 1
```

```bash
serve deploy serve_config.yaml
```

다음은 이 과정에서 레플리카 하나가 겪는 일을 손으로 따라가 본 것입니다(예시).

1. 처음 배포할 때 `__init__`이 t5-small을 적재하고 `self.prefix`를 프랑스어 접두사로 둡니다.
2. 바로 이어서 `reconfigure({"language": "french"})`가 호출되어 같은 값을 다시 씁니다.
3. 설정을 `german`으로 고쳐 다시 배포하면 같은 레플리카에서 `reconfigure({"language": "german"})`만 호출되어 `self.prefix`가 `"translate English to German: "`로 바뀝니다. 모델은 다시 적재하지 않습니다.
4. 이후 요청부터 요약문이 독일어로 번역됩니다.

`Summarizer`도 `user_config`에 `min_length`, `max_length`를 넣으면 요약 길이가 바뀝니다. 이 필드는 클러스터를 재시작하지 않고 모델 가중치와 버전을 바꾸거나, 모델 컴포지션 그래프의 트래픽 분할 비율을 조정하거나, 기능 플래그·A/B 테스트·하이퍼파라미터를 바꾸는 데 씁니다.

대신 무엇을 바꿀 수 있는지는 전적으로 `reconfigure` 구현에 달려 있습니다. Serve는 값을 전달할 뿐이고, 그 값으로 상태를 갱신하는 코드는 직접 써야 합니다. 예제의 `Translator.reconfigure`는 지원하지 않는 언어가 오면 `pass`로 넘어가므로 `self.language`만 바뀌고 접두사는 이전 값이 남습니다. 운영 코드라면 이런 경우를 명시적으로 처리하기를 권합니다. `user_config`로 모델 가중치를 바꾸려면 `reconfigure` 안에서 새 가중치를 불러와야 하고, 그 작업이 길면 갱신도 그만큼 오래 걸립니다. 자주 바꾸는 가벼운 값은 `user_config`로, 생성 시점에 한 번 정하면 되는 무거운 값은 `bind`의 생성자 인자로 나누는 편이 무난합니다.

## KubeRay로 Kubernetes에서 운영하기

`serve deploy`로 설정 파일을 적용하는 방식은 Ray 클러스터가 이미 떠 있다고 가정합니다. 클러스터를 누가 띄우는지, 노드가 죽으면 누가 다시 세우는지, 새 버전을 올리는 동안 트래픽이 어떻게 되는지는 Serve 바깥 계층이 정합니다. 앞서 장애 허용을 정리한 표에서 보았듯이 Ray Serve는 레플리카, 프록시, 컨트롤러 액터까지는 스스로 복구하지만, KubeRay 없이 Ray 클러스터 자체가 실패하면 복구하지 못합니다. 그래서 Ray Serve는 운영 환경의 권장 방식으로 Kubernetes 위에서 KubeRay의 RayService 커스텀 리소스를 쓰는 방법을 꼽습니다. Kubernetes를 쓰지 않는다면 Serve CLI로 Ray 클러스터에 직접 배포하는 방법도 여전히 유효합니다.

### RayService가 자동으로 처리하는 일

KubeRay는 Kubernetes에서 Ray 클러스터를 다루는 오퍼레이터이고, RayService는 Serve 애플리케이션을 올리기 위한 커스텀 리소스입니다. RayService는 운영에 꼭 필요한 요구 사항 넷을 자동으로 처리합니다.

- 헬스 체크
- 상태 보고
- 장애 복구
- 업그레이드

장애 복구는 Serve 내부 복구와 이어집니다. 복구는 계층별로 나뉩니다. 레플리카와 프록시는 컨트롤러가, 컨트롤러는 Ray가 다시 세우고, RayService를 쓰면 KubeRay가 죽은 노드나 무너진 클러스터를 복구합니다. 앞서 표에서 "Serve가 복구하지 못합니다"로 남아 있던 칸을 KubeRay가 맡습니다.

Kubernetes로 옮긴다고 개발 방식이 바뀌지는 않습니다. 흐름은 다음과 같습니다.

1. 노트북에서 `serve run`으로 실행하고 고치기를 반복합니다.
2. 운영에 넘길 준비가 되면 `serve build`로 Serve 설정 파일을 만듭니다.
3. 생성된 파일에 `runtime_env`를 채우고 레플리카 수, `user_config` 같은 값을 다듬습니다.
4. VM이라면 이 파일을 `serve deploy`로 적용하고, Kubernetes라면 같은 YAML을 Kubernetes 설정, 즉 RayService 리소스에 그대로 복사해 넣습니다.

마지막 단계만 갈라질 뿐, 어느 쪽이든 클러스터 설정의 기준은 같은 파일입니다. Serve를 쓰는 데 무거운 Kubernetes 개념이나 클라우드 설정을 알 필요는 없고, Kubernetes 지식은 RayService 리소스를 다루는 단계에서야 필요합니다.

### Serve 설정 파일을 RayService에 넣기

RayService에 넣는 내용은 VM에서 `serve deploy`에 넘기던 Serve 설정과 같습니다. 앞서 설명한 `http_options.host`와 `runtime_env` 제약이 Kubernetes에서 그대로 적용되며, 특히 `host`는 `0.0.0.0`이어야 클러스터 밖에서 디플로이먼트에 닿고, 코드는 `working_dir`에 적은 원격 zip URI로 전달됩니다. Serve 설정 파일을 Kubernetes 매니페스트와 같은 저장소에서 버전 관리해 두면 어떤 커밋이 어떤 설정으로 배포되었는지 추적하기 쉽습니다.

의존성을 전달하는 방법은 둘입니다. 하나는 지금까지처럼 `runtime_env`의 `pip`에 `torch`, `transformers`를 적는 방법이고, 다른 하나는 KubeRay와 함께 쓸 커스텀 Docker 이미지를 만들어 패키지를 미리 넣어 두는 방법입니다. `runtime_env` 방식은 이미지를 다시 빌드하지 않고 YAML만 고치면 되므로 가볍습니다. 반면 PyTorch처럼 큰 패키지는 커스텀 이미지에 넣어 두는 편이 무난합니다. 런타임 설치를 거치지 않으니 환경이 이미지 단위로 고정되고, 대신 이미지 빌드 파이프라인을 따로 관리해야 합니다.

### 무중단 업그레이드와 프록시 드레이닝

KubeRay는 운영 중인 애플리케이션의 무중단 업그레이드(zero-downtime upgrade)를 지원합니다. 업그레이드를 구체적으로 어떤 절차로 진행하는지는 KubeRay 문서가 다루며, 여기서는 Serve 쪽에서 처리 공백을 줄이는 수단 두 가지를 따로 봅니다.

첫째는 변경 범위를 작게 두는 설계입니다. 앞서 다룬 `user_config` 갱신은 레플리카를 재시작하지 않고 `reconfigure`만 다시 호출하므로, 번역 언어나 요약 길이처럼 자주 바꾸는 값은 모델을 다시 적재하지 않고 바뀝니다.

둘째는 프록시 드레이닝(draining)입니다. 이는 업그레이드 전용 기능이 아니라 프록시가 종료될 때 늘 거치는 수명 주기 동작입니다. 노드마다 프록시를 두고 그 앞에 로드 밸런서를 세운 구성에서 프록시 하나를 내리면, 로드 밸런서가 그 사실을 알기 전까지 요청이 계속 그 프록시로 향합니다. 드레이닝은 이 간극을 메웁니다. 종료 예정인 프록시는 드레이닝 상태에 들어가 일부러 헬스 체크에 실패하고, 로드 밸런서는 실패한 헬스 체크를 보고 그 프록시로 새 트래픽을 보내지 않습니다. 이미 받은 요청은 정상적으로 끝까지 처리되고, 프록시는 최소 드레이닝 기간이 지난 뒤에야 종료됩니다.

프록시 헬스 체크와 수명 주기는 다음 환경 변수로 조정합니다.

| 환경 변수 | 의미 | 기본값 |
|---|---|---|
| `RAY_SERVE_PROXY_HEALTH_CHECK_PERIOD_S` | 컨트롤러가 각 프록시의 상태를 확인하는 주기(초)입니다. | `10.0` |
| `RAY_SERVE_PROXY_HEALTH_CHECK_TIMEOUT_S` | 컨트롤러가 헬스 체크 응답을 기다리는 시간(초)입니다. 연속 3회 실패하면 프록시를 비정상으로 표시하고 다시 시작합니다. | `10.0` |
| `RAY_SERVE_PROXY_READY_CHECK_TIMEOUT_S` | 시작할 때 프록시가 준비되기를 기다리는 시간(초)입니다. | `5.0` |
| `RAY_SERVE_PROXY_MIN_DRAINING_PERIOD_S` | 프록시가 종료 전 드레이닝 상태에 머무는 최소 시간(초)입니다. | `30.0` |

최소 드레이닝 기간은 로드 밸런서의 헬스 체크 주기와 맞물려 정해야 합니다. 다음은 로드 밸런서가 15초마다 헬스 체크를 한다고 가정하고 기본값 30초 동안 요청 흐름을 손으로 따라가 본 것입니다(예시, 숫자는 설명용입니다).

1. 0초에 프록시 A가 드레이닝 상태에 들어가 헬스 체크에 실패하기 시작합니다. 로드 밸런서는 아직 A가 정상이라고 알고 있습니다.
2. 0초에서 다음 헬스 체크 사이에 들어온 새 요청은 여전히 A로 갑니다. A는 아직 종료되지 않았으므로 요청을 받아 처리합니다.
3. 늦어도 15초 무렵 로드 밸런서의 헬스 체크가 실패를 확인하면, 이후 새 요청은 다른 노드의 프록시로 갑니다. 실패 한 번에 바로 제외하는 로드 밸런서라고 가정한 경우입니다.
4. 15초부터 30초까지 A는 새 트래픽 없이 이미 받은 요청만 마칩니다.
5. 30초가 지나면 A가 종료됩니다. 이 시점에 A로 향하는 요청은 없습니다.

같은 로드 밸런서가 연속 세 번 실패해야 대상을 제외한다면 감지까지 최대 45초 가까이 걸리고, 30초에 종료된 A로 그 차이만큼 요청이 갈 수 있습니다. 로드 밸런서가 비정상 대상을 빼는 데 걸리는 시간을 먼저 계산하고, 최소 드레이닝 기간을 그보다 넉넉하게 두기를 권합니다. 너무 길게 잡으면 프록시를 내릴 때마다 그만큼 기다려야 합니다.

환경 변수를 지정하는 위치와 방법은 KubeRay·Ray 클러스터 문서를 따릅니다.

### 헤드 노드 장애 허용

앞서 장애 허용에서 보았듯이 컨트롤러는 라우팅 정책과 디플로이먼트 설정을 헤드 노드의 GCS에 체크포인트로 남기고, 다시 시작된 컨트롤러는 그 데이터로 상태를 되찾습니다. 이 구조에서 헤드 노드는 복구의 근거가 놓인 자리라서, 헤드 노드를 잃으면 액터 하나의 장애가 아니라 클러스터 장애가 됩니다. KubeRay는 무너진 클러스터를 다시 세우지만, KubeRay의 GCS 장애 허용(GCS fault tolerance) 기능을 켜면 애초에 클러스터가 무너지는 상황을 피합니다. 운영 환경에서는 RayService와 함께 헤드 노드 장애 허용을 설정하는 것이 권장 관행입니다.

## 모니터링과 디버깅

운영 중에 문제가 생기면 확인할 것은 셋입니다. 어떤 액터가 살아 있는지, 디플로이먼트가 목표 상태에 도달했는지, 특정 요청이 어디서 느려지거나 실패했는지입니다. Ray Serve는 Ray Dashboard, `serve status`, 로그, Prometheus 메트릭으로 이 질문에 답합니다. 운영 환경에서는 로그와 메트릭을 사용 중인 관측 플랫폼으로 내보내기를 권합니다.

### Ray Dashboard로 클러스터 전체 보기

대시보드는 로컬 테스트와 원격 클러스터 모두에서 클러스터 URI의 8265 포트로 열리며, 로컬이라면 `http://localhost:8265`입니다. Serve 페이지에서는 디플로이먼트별 실행 중인 레플리카 수를 보고, 레플리카 상세 페이지에서 메타데이터와 로그를 확인합니다. 레플리카 프로세스의 `logging`과 `print` 출력이 모두 여기에 남습니다.

Actors 화면에서는 액터별 PID, 로그 링크, 생존 여부를 봅니다. `num_replicas=2`인 `Translator` 애플리케이션이라면 컨트롤러 1개, HTTP 프록시 1개, 레플리카 2개로 액터가 넷이며, 클러스터 장애가 의심되면 어떤 액터가 죽었는지부터 확인합니다.

### serve status와 serve config로 상태 읽기

`serve config`는 클러스터가 마지막으로 받은 설정 파일, 즉 목표 상태를 보여 주고, `serve status`는 현재 상태를 보여 줍니다. 클러스터는 디플로이먼트를 배포하고 실패한 레플리카를 복구하며 목표 상태로 수렴하려 하므로, 둘을 나란히 보면 차이가 드러납니다. 원격 클러스터에서는 두 명령 모두 `--address/-a` 인자로 클러스터에 접속합니다.

| 대상 | 상태 | 의미 |
|---|---|---|
| 프록시 | `STARTING` | 시작 중이며 아직 요청을 받을 준비가 되지 않은 상태입니다. |
| 프록시 | `HEALTHY` | 정상적으로 요청을 처리하는 상태입니다. |
| 프록시 | `UNHEALTHY` | 헬스 체크에 실패한 상태입니다. Serve가 이 프록시를 종료하고 같은 노드에 새로 띄웁니다. |
| 프록시 | `DRAINING` | 정상이지만 새 요청을 받지 않는 상태입니다. 처리 중인 요청이 남아 있을 수 있습니다. |
| 프록시 | `DRAINED` | 새 요청을 받지 않고 남은 요청도 없는 상태입니다. |
| 애플리케이션 | `NOT_STARTED` | 클러스터에 배포된 애플리케이션이 없는 상태입니다. |
| 애플리케이션 | `DEPLOYING` | `serve deploy` 요청을 처리하며 디플로이먼트를 만들거나 갱신하는 상태입니다. |
| 애플리케이션 | `RUNNING` | 이전 배포 요청을 모두 마치고 최신 목표 상태를 유지하는 안정 상태입니다. |
| 애플리케이션 | `DEPLOY_FAILED` | 최신 `serve deploy` 요청이 실패한 상태입니다. |
| 디플로이먼트 | `UPDATING` | 이전 배포 요청의 목표 상태로 갱신 중인 상태입니다. |
| 디플로이먼트 | `HEALTHY` | 목표 레플리카 수로 정상 실행 중인 상태입니다. |
| 디플로이먼트 | `UNHEALTHY` | 갱신 후 비정상이 된 상태입니다. 레플리카 확장 실패, 헬스 체크 실패, 시스템·머신 오류가 원인일 수 있습니다. |
| 디플로이먼트 | `DEPLOY_FAILED` | 시작이나 갱신에 실패한 상태입니다. 대개 생성자 오류가 원인입니다. |
| 디플로이먼트 | `UPSCALING` / `DOWNSCALING` | 오토스케일링이 켜진 디플로이먼트가 레플리카를 늘리거나 줄이는 상태입니다. |
| 레플리카 | `STARTING` | 시작 중이며 아직 요청을 받지 못하는 상태입니다. |
| 레플리카 | `UPDATING` | `reconfigure` 갱신을 진행 중인 상태입니다. |
| 레플리카 | `RECOVERING` | 레플리카가 자기 상태를 복구하고 있는 상태입니다. |
| 레플리카 | `RUNNING` | 정상적으로 요청을 처리하는 상태입니다. |
| 레플리카 | `STOPPING` | 종료 중인 상태입니다. |

디플로이먼트가 `DEPLOY_FAILED`라면 생성자 오류부터 의심합니다. `message` 필드에 상황 설명이 담깁니다.

파이썬에서는 `serve.status()`가 CLI와 같은 정보를 `dataclass`로 돌려주므로, 디플로이먼트나 드라이버 스크립트 안에서 상태를 코드로 판단합니다. 다음은 `RUNNING`인 애플리케이션 이름을 모아 돌려주는 예입니다.

```python
from typing import List, Dict

from ray import serve
from ray.serve.schema import ServeStatus, ApplicationStatusOverview


@serve.deployment
def get_healthy_apps() -> List[str]:
    serve_status: ServeStatus = serve.status()
    app_statuses: Dict[str, ApplicationStatusOverview] = serve_status.applications

    running_apps = []
    for app_name, app_status in app_statuses.items():
        if app_status.status == "RUNNING":
            running_apps.append(app_name)

    return running_apps


monitoring_app = get_healthy_apps.bind()
```

### 로그와 요청 ID

Serve는 표준 `logging` 모듈의 `"ray.serve"` 로거를 씁니다. 로그는 기본적으로 `stderr`와 각 노드의 `/tmp/ray/session_latest/logs/serve/`에 함께 기록되며, 컨트롤러·프록시의 시스템 로그, 액세스 로그, 레플리카 안의 사용자 로그가 모두 포함됩니다. 개발 중에는 로그가 드라이버 프로그램으로 스트리밍되므로 디버깅하는 동안 `serve run`을 띄워 둡니다.

```python
# File name: monitoring.py
from ray import serve
import logging
from starlette.requests import Request

logger = logging.getLogger("ray.serve")


@serve.deployment
class SayHello:
    async def __call__(self, request: Request) -> str:
        logger.info("Hello world!")
        return "hi"


say_hello = SayHello.bind()
```

`serve run monitoring:say_hello`로 띄우고 다른 터미널에서 `curl -X GET http://localhost:8000/`을 보내면 다음과 같은 줄이 찍힙니다.

```text
(ServeReplica:SayHello pid=63520) INFO 2023-04-10 15:59:45,403 SayHello SayHello#kTBlTj HzIYOzaEgN / monitoring.py:16 - Hello world!
(ServeReplica:SayHello pid=63520) INFO 2023-04-10 15:59:45,403 SayHello SayHello#kTBlTj HzIYOzaEgN / replica.py:527 - __CALL__ OK 0.5ms
```

괄호 안은 액터 종류와 PID이고, 이어서 로그 수준, 타임스탬프, 디플로이먼트 이름, 레플리카 태그, 요청 ID, 경로, 파일 이름과 줄 번호가 나옵니다. 두 번째 줄이 액세스 로그입니다. 디스크의 로그는 ELK나 Loki로 수집해 디플로이먼트·레플리카 기준으로 검색하고, 파일 회전은 `RAY_ROTATION_MAX_BYTES`와 `RAY_ROTATION_BACKUP_COUNT`로 설정합니다.

로그 수준은 디플로이먼트 생성자 안에서 `"ray.serve"` 로거로 바꿉니다. JSON 형식이 필요하면 `serve.run`이나 `@serve.deployment`의 `logging_config`에 `encoding="JSON"`을, 액세스 로그를 끄려면 `@serve.deployment`의 `logging_config`에 `{"enable_access_log": False}`를 넘깁니다. `serve.run`에 준 `logging_config`는 애플리케이션의 모든 디플로이먼트에 적용되며, 디플로이먼트 수준에서도 설정하면 디플로이먼트 수준 설정이 우선합니다.

```python
# File name: json_logging.py
from ray import serve
from ray.serve.schema import LoggingConfig
import requests


@serve.deployment(logging_config=LoggingConfig(encoding="JSON"))
class Model:
    def __call__(self) -> int:
        return "hello world"


serve.run(Model.bind())
resp = requests.get("http://localhost:8000/")
```

요약·번역 파이프라인처럼 요청 하나가 여러 액터를 거치면 요청 ID로 로그를 잇습니다. 요청 헤더에 `X-Request-ID`를 넣으면 그 값이 프록시 로그와 디플로이먼트 로그에 찍히고 응답 헤더로도 돌아오며, ID가 중복되면 로그가 섞이므로 고유한 값을 씁니다.

```python
# File name: request_id.py
from ray import serve
import requests


@serve.deployment
class Model:
    def __call__(self) -> int:
        return 1


serve.run(Model.bind())
resp = requests.get("http://localhost:8000", headers={"X-Request-ID": "123-234"})
print(resp.headers["X-Request-ID"])
```

### Prometheus 메트릭

Serve는 성공·실패 요청 수 같은 시스템 메트릭을 Ray 메트릭 인프라로 내보내며, 기본적으로 각 노드에서 Prometheus 형식으로 노출합니다. HTTP/gRPC 프록시 호출과 `DeploymentHandle` 호출은 수집되는 메트릭이 다릅니다. 처음에는 다음부터 보기를 권합니다.

- 처리량과 지연: `ray_serve_num_http_requests_total`, `ray_serve_http_request_latency_ms`
- 오류를 낸 디플로이먼트: `ray_serve_num_deployment_http_error_requests_total`
- 역압(backpressure): `ray_serve_deployment_queued_queries`
- 콜드 스타트: `ray_serve_replica_startup_latency_ms`와 그 일부인 `ray_serve_replica_initialization_latency_ms`
- 상태 시계열: `ray_serve_deployment_status`, `ray_serve_proxy_status`

p95·p99가 부정확해 보이면 `RAY_SERVE_REQUEST_LATENCY_BUCKETS_MS="10,50,100,500,1000,5000"`처럼 히스토그램 경계를 바꾸고 Serve 액터를 다시 시작합니다. 메트릭이 Prometheus에 늦게 보이면 Ray Core의 `metrics_report_interval_ms`(기본값 10000밀리초)를 줄입니다. 이 설정은 안정된 공개 API가 아니어서 예고 없이 바뀔 수 있습니다.

```bash
ray start --head --system-config='{"metrics_report_interval_ms": 1000}'
```

`async` 핸들러 안의 동기 코드가 이벤트 루프를 막는지는 `ray_serve_event_loop_scheduling_latency_ms`로 확인합니다. 10밀리초 미만이 정상이고, 100~500밀리초면 블로킹 I/O나 CPU 연산을 의심합니다. 고급 디버깅용 내부 메트릭이라 이후 릴리스에서 바뀔 수 있습니다.

애플리케이션 고유의 지표는 `ray.serve.metrics`로 정의하며, Serve가 디플로이먼트·레플리카 태그를 덧붙여 내보냅니다.

```python
# File name: custom_metric.py
from ray import serve
from ray.serve import metrics


@serve.deployment
class MyDeployment:
    def __init__(self):
        self.num_requests = 0
        self.my_counter = metrics.Counter(
            "my_counter",
            description=("The number of odd-numbered requests to this deployment."),
            tag_keys=("model",),
        )
        self.my_counter.set_default_tags({"model": "123"})

    def __call__(self):
        self.num_requests += 1
        if self.num_requests % 2 == 1:
            self.my_counter.inc()


my_deployment = MyDeployment.bind()
```

Prometheus 수집 설정은 Ray Metrics 문서를 따릅니다. 모델 품질이나 데이터 드리프트처럼 시스템 메트릭 밖의 영역은 Arize 같은 ML 관측 플랫폼의 클라이언트 코드를 디플로이먼트 안에 넣어 연동합니다.

## 다음 단계

번역기 하나를 디플로이먼트로 바꾸는 데서 시작해 운영과 관측까지 왔다면, 다음 공부는 실제로 부딪힌 문제를 기준으로 고르면 됩니다.

자원을 더 세밀하게 나눠야 한다면 Resource allocation 가이드를 읽습니다. 앞서 쓴 `num_cpus`, `num_gpus` 외에도 `ray_actor_options`의 `resources`로 HPU나 TPU 같은 GPU 외 가속기를 요청합니다. 레플리카 수를 고정값 대신 부하에 맞춰 조절하려면 Autoscaling guide에서 `autoscaling_config` 설정을 익힙니다.

프레임워크별 튜토리얼은 Examples 페이지에 모여 있습니다. 요청 배칭을 쓰는 텍스트 생성기, 요청·응답 스트리밍 챗봇, Gradio·Triton·MLflow 연동, AWS NeuronCore나 Intel Gaudi 서빙, 모델 멀티플렉싱 같은 배포 패턴 예제가 실려 있습니다.

LLM을 서빙하려면 Ray Serve LLM을 봅니다. 이 프레임워크는 vLLM 같은 추론 엔진 하나의 성능을 수평 확장과 분산 전략으로 넓힙니다. 핵심 구성 요소는 둘입니다. 엔진 인스턴스 하나를 관리하는 디플로이먼트 `LLMServer`가 있고, `/v1/chat/completions` 같은 OpenAI 호환 엔드포인트를 제공하는 인그레스 디플로이먼트 `OpenAiIngress`가 있습니다. 소형·중형·대형 모델부터 비전, 추론 모델까지 크기와 유형별 배포 튜토리얼이 있어서 출발점으로 삼기 좋습니다.

- - -

### Sources
* [Getting Started — Ray 2.59.0](https://docs.ray.io/en/latest/serve/getting_started.html)
* [Ray Serve: Scalable and Programmable Serving — Ray 2.59.0](https://docs.ray.io/en/latest/serve/index.html)
* [Serve ML Models (Tensorflow, PyTorch, Scikit-Learn, others) — Ray 2.59.0](https://docs.ray.io/en/latest/serve/tutorials/serve-ml-models.html)
* [Set Up FastAPI and HTTP — Ray 2.59.0](https://docs.ray.io/en/latest/serve/http-guide.html)
* [Deploy Compositions of Models — Ray 2.59.0](https://docs.ray.io/en/latest/serve/model_composition.html)
* [Key Concepts — Ray 2.59.0](https://docs.ray.io/en/latest/serve/key-concepts.html)
* [Production Guide — Ray 2.59.0](https://docs.ray.io/en/latest/serve/production-guide/index.html)
* [Examples — Ray 2.59.0](https://docs.ray.io/en/latest/serve/examples.html)
* [Develop and Deploy an ML Application — Ray 2.59.0](https://docs.ray.io/en/latest/serve/develop-and-deploy.html)
* [Serve Config Files — Ray 2.59.0](https://docs.ray.io/en/latest/serve/production-guide/config.html)
* [Monitor Your Application — Ray 2.58.0](https://docs.ray.io/en/latest/serve/monitoring.html)
* [Architecture — Ray 2.59.0](https://docs.ray.io/en/latest/serve/architecture.html)
* [Paper Reading: Ray: A Distributed Framework for Emerging AI Applications &#124; Kevin Hu's Blog](https://blog.kevinhu.me/2021/08/02/01-ray-first-impressions)
* [Top 8 Machine Learning Model Deployment Tools in 2026](https://www.truefoundry.com/blog/model-deployment-tools)
* [Serve Config Files — Ray 2.55.0](https://docs.ray.io/en/releases-2.55.0/serve/production-guide/config.html)
* [Ray Serve API — Ray 2.59.0](https://docs.ray.io/en/latest/serve/api/index.html)
* [Architecture overview — Ray 2.58.0](https://docs.ray.io/en/latest/serve/llm/architecture/overview.html)
* [Observability and monitoring — Ray 2.59.0](https://docs.ray.io/en/latest/serve/llm/user-guides/observability.html)
* [Cluster Management CLI — Ray 2.59.0](https://docs.ray.io/en/latest/cluster/cli.html)
* [Index — Ray 2.58.0](https://docs.ray.io/en/latest/genindex.html)
