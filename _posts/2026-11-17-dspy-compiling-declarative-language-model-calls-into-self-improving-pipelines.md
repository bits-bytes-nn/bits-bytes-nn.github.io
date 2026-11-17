---
layout: post
title: "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"
seo_title: "DSPy: Compiling Declarative LM Calls into Self-Improving Pipelines"
subtitle: "DSPy: 선언적 LM 호출을 스스로 개선하는 파이프라인으로 컴파일하기"
date: 2023-10-05 17:37:25
paper_author: "Stanford University"
paper_doi: "10.48550/arXiv.2310.03714"
description: "DSPy는 손으로 쓴 프롬프트 템플릿 대신 시그니처와 모듈로 LM 파이프라인을 선언하고, 텔레프롬프터가 데모를 부트스트래핑해 컴파일합니다. GSM8K에서 GPT-3.5 81.6% test, HotPotQA에서 770M T5-Large 39.3% Ans 결과를 정리합니다."
categories: ["Paper Reviews", "Language-Models"]
tags: ["Declarative-Language-Model-Programming", "Teleprompter-Optimization", "Bootstrapped-Few-Shot-Demonstrations", "Modular-Prompting-Techniques", "Text-Transformation-Graphs", "Agentic-AI", "Retrieval-Augmented-Generation"]
cover: /assets/images/language-models.jpg
use_math: true
---
## TL;DR

### 이 연구를 시작하게 된 배경과 동기는 무엇입니까?
LM 호출 여러 개를 엮는 파이프라인은 프롬프트에 민감합니다. 그런데도 기존 파이프라인과 LangChain·LlamaIndex 같은 프레임워크는 시행착오로 다듬은 긴 프롬프트 템플릿에 의존합니다. LangChain 코드베이스에만 1000자를 넘는 문자열이 50개 있었습니다. 저자들은 이런 문자열이 다른 LM, 다른 도메인, 다른 파이프라인으로 옮겨 간다는 보장이 없다고 보고, 분류기 가중치를 손으로 조정하는 일에 빗댑니다.

### 이 연구에서 제시하는 새로운 해결 방법은 무엇입니까?
딥러닝 프레임워크처럼 "레이어 조립"과 "옵티마이저 학습"을 프롬프팅에 옮겼습니다.
- **시그니처**: `question -> answer`처럼 모듈이 "무엇을" 하는지만 자연어 타입으로 선언합니다.
- **모듈**: 체인 오브 소트·ReAct 같은 기법을 LM·지시문·데모를 파라미터로 갖는 모듈로 바꾸고, Python 제어 흐름으로 조립합니다.
- **텔레프롬프터**: 프로그램, 소량의 학습 입력, 평가 지표를 받아 데모를 부트스트래핑합니다. 결과물은 퓨 샷 프롬프트가 되거나 작은 LM의 파인튜닝 데이터가 됩니다.

### 제안된 방법은 어떻게 구현되었습니까?
컴파일 모드에서 Predict 호출마다 트레이스를 남깁니다. BootstrapFewShot은 교사 프로그램을 학습 입력에 실행한 뒤, 평가 지표를 통과한 실행의 모든 단계 입출력을 해당 모듈의 데모로 넣습니다. 그래서 라벨은 최종 답에만 있으면 됩니다. 그다음 무작위 탐색(또는 Optuna의 TPE)으로 데모 조합을 고르거나, BootstrapFinetune으로 가중치를 학습합니다. 마지막으로 앙상블로 프로그램을 확장합니다. 교사 프로그램을 지정하면 텔레프롬프터끼리 합성할 수 있습니다.

### 이 연구의 결과가 가지는 의미는 무엇입니까?
- GSM8K에서 GPT-3.5용 CoT를 bootstrap 후 +ensemble로 컴파일하면 88.3 dev / 81.6 test를 냅니다. 사람 추론 체인 없이도 bootstrap(72.9 test)이 +human_CoT(72.4 test)와 맞먹습니다.
- HotPotQA에서 react bootstrap×2는 전문가 데모(+human_r)를 앞섭니다. GPT-3.5에서 39.0 대 33.0, llama2-13b-chat에서 40.0 대 28.3입니다. multihop ensemble은 GPT-3.5 dev 54.7 Ans로 가장 높습니다.
- llama2-13b-chat 앙상블 교사로 파인튜닝한 770M T5-Large multihop은 dev 39.3% Ans / 46.0% Psg를 냅니다. 작은 오픈 모델로 파이프라인을 돌리려는 팀이 가장 큰 이득을 봅니다.

- - -

## 손으로 다듬은 프롬프트 템플릿의 한계와 DSPy의 발상

### 여러 번의 LM 호출이 엮일수록 커지는 프롬프트 민감도

GPT 계열 같은 파운데이션 모델은 대규모 데이터로 사전 학습된 범용 모델입니다. 과제별 모델을 따로 학습하지 않고 프롬프트만 바꿔 여러 과제에 적용할 수 있어서, 이전보다 적은 데이터로 더 높은 추상화 수준에서 NLP 시스템을 만들 수 있습니다([Bommasani et al.](https://arxiv.org/pdf/2108.07258)). 이를 바탕으로 복잡한 과제를 여러 번의 LM 호출로 쪼개는 LM 파이프라인과 에이전트가 빠르게 늘고 있습니다. 질문을 검색 질의로 바꾸고, 문서를 찾고, 근거를 따져 답하는 단계마다 LM을 한 번씩 부르는 구조입니다.

LM은 프롬프트를 어떻게 쓰느냐에 민감합니다. 호출 여러 개가 서로 맞물려야 하는 파이프라인에서는 이 민감도가 더 커집니다. 예를 들어 앞 단계 출력 형식이 달라지면 뒤 단계가 흔들릴 수 있습니다. 그래서 기존 파이프라인과 인기 있는 개발 프레임워크는 대부분 하드코딩된 프롬프트 템플릿에 의존합니다. 프롬프트 템플릿은 지시문과 예시를 담은 긴 문자열이며, 시행착오를 거쳐 손으로 다듬어집니다. 저자들은 이 방식을 분류기의 가중치를 손으로 조정하는 일에 빗댑니다. 깨지기 쉽고 확장하기 어려울 수 있으며, 어떤 프롬프트 문자열이 다른 파이프라인, 다른 LM, 다른 데이터 도메인, 심지어 다른 입력에서도 통한다는 보장이 없다는 비판입니다.

### 신경망 프레임워크에서 빌려 온 두 가지 원칙

DSPy(디-에스-파이로 읽습니다)는 딥러닝 프레임워크가 정착시킨 두 가지 합의에서 출발합니다. 첫째, 범용 레이어를 모듈처럼 조립해 어떤 복잡한 구조든 만들 수 있습니다. 둘째, 가중치는 사람이 정하지 않고 옵티마이저가 학습합니다. PyTorch에서 `nn.Linear`나 어텐션 레이어를 쌓은 뒤 Adam으로 학습시키는 방식이 바로 이 두 원칙을 따릅니다.

DSPy는 이 구도를 프롬프팅에 옮깁니다.

- **레이어 → 모듈**: 체인 오브 소트나 ReAct 같은 문자열 기반 프롬프팅 기법을 선언적 모듈로 바꿉니다. 각 모듈에는 입력과 출력을 자연어로 타입처럼 선언한 시그니처가 붙습니다.
- **조립 방식 → 실행 시 정의(define-by-run)**: [PyTorch](https://doi.org/10.48550/arXiv.1912.01703)를 직접 본떠, 모듈을 선언한 뒤 if 문, for 루프, 예외 처리 같은 일반 Python 제어 흐름으로 연결합니다.
- **옵티마이저 → 컴파일러와 텔레프롬프터**: 컴파일러는 프로그램, 소량의 학습 입력(라벨은 선택), 검증용 평가 지표를 받습니다. 이것으로 각 모듈의 데모를 부트스트래핑해 퓨 샷 프롬프트를 만들거나 작은 LM을 파인튜닝합니다. 어떤 방식으로 학습할지는 범용 최적화 전략인 텔레프롬프터가 정합니다.

이렇게 하면 LM 파이프라인은 텍스트 변환 그래프가 됩니다. 텍스트 변환 그래프란 선언적 모듈을 통해 LM을 호출하는 명령형 계산 그래프입니다. 저자들은 이 계열 작업을 **파운데이션 모델 프로그래밍**이라 부르며, Torch·Theano·Chainer가 딥러닝에 제공한 추상화를 LM 파이프라인에 제공하려 합니다.

> **핵심:** 프롬프트 문자열을 사람이 직접 쓰는 대상이 아니라 학습되는 파라미터로 다룹니다. 사람은 "무엇을 할지"만 선언하고, "어떻게 프롬프트할지"는 컴파일러가 데이터와 평가 지표로 찾습니다.

### 기여와 계보

저자들이 내세우는 기여는 두 가지입니다. 하나는 프롬프팅 기법을 파라미터화된 선언적 모듈로 옮긴 최초의 프로그래밍 모델이고, 다른 하나는 이런 모듈로 이루어진 임의의 파이프라인을 최적화하는 범용 전략(텔레프롬프터)을 갖춘 컴파일러입니다. 실험에서는 아주 짧은 프로그램만으로 스스로 개선하는 다단계 시스템을 부트스트래핑할 수 있었습니다. 이는 llama2-13b-chat이나 7억 7천만 파라미터 T5-Large 같은 작은 모델에서도 마찬가지였습니다. DSPy는 같은 저자들이 앞서 만든 Demonstrate–Search–Predict(DSP) 프레임워크의 두 번째 버전입니다.

### LangChain·LlamaIndex와 비교한 위치

LangChain과 LlamaIndex는 특정 에이전트, 정해진 검색 파이프라인, 데이터베이스 연결, 에이전트 메모리처럼 바로 가져다 쓸 수 있는 체인과 도구를 제공합니다. 반면 DSPy가 제공하는 것은 시그니처, 모듈, 텔레프롬프터라는 핵심 연산자와 옵티마이저입니다. 목표는 새 파이프라인을 빠르게 짜고 품질은 컴파일로 끌어올리는 것입니다.

저자들은 비공식 조사로 이 차이를 보여 줍니다. 2023년 9월 말 기준으로 LangChain 코드베이스에는 1000자를 넘는 문자열이 50개 있었고, 대부분 프롬프트였습니다. 프롬프트 전용 파일도 `prompts.py` 12개, `prompt.py` 42개에 달했습니다. DSPy에는 1000자 넘는 문자열이 하나도 없고, 과제별로 손으로 쓴 데모도 전혀 없습니다. LangChain은 Oracle, GoogleSQL, DuckDB, Crate, MySQL 등 SQL 데이터베이스 제공자마다 프롬프트를 따로 두며, 이 프롬프트들의 평균 길이는 1058자입니다. 저자들은 LM과 과제에 맞춰 자동으로 부트스트래핑한 프롬프트가 제공자별 프롬프트를 하드코딩하는 방식보다 훨씬 강력하고 모듈적일 것이라고 봅니다.

수작업 프롬프트의 규모는 저자들이 공개 논문과 저장소에서 모은 예시에서 드러납니다.

| 프롬프트 | 출처 | 단어 수 | 문자 수 |
| --- | --- | --- | --- |
| Text-evidence checker | Gao et al. (2023a) | 818 | 4964 |
| Math word problems (PAL) | LangChain & Gao et al. (2023b) | 566 | 3957 |
| ReAct | Yao et al. (2022) | 593 | 3889 |
| Zero-shot ReAct | LangChain | 101 | 600 |
| QA with sources | LangChain | 992 | 6197 |
| SQL MyScale querying | LangChain | 343 | 2239 |
| Relevant docs retrieval | LlamaIndex | 129 | 719 |
| IRS chatbot | LlamaIndex | 389 | 2258 |

PAL 프롬프트를 예로 들면, 수학 문장제 8개와 그에 대응하는 Python 풀이 함수를 통째로 넣은 퓨 샷 템플릿입니다. QA with sources 프롬프트는 계약서 조항과 연설문 발췌까지 예시로 담아 6000자를 넘깁니다. MyScale SQL 프롬프트는 "세미콜론을 붙이지 말 것", "존재하지 않는 열을 조회하지 말 것" 같은 규칙을 일일이 나열합니다. 이런 문자열 하나하나가 특정 LM과 특정 과제에 맞춰진 수작업 결과물이고, DSPy가 없애려는 대상이 바로 이것입니다.

## 배경: 인컨텍스트 학습에서 검색·도구 결합 파이프라인과 프롬프트 최적화까지

### 프롬프트로 끌어내는 정교한 행동

파운데이션 모델 프로그래밍의 핵심 메커니즘은 인컨텍스트 학습(in-context learning)입니다. 가중치를 바꾸지 않고 프롬프트 안에 지시문과 몇 개의 입출력 예시만 넣으면 모델이 새 과제를 수행합니다. 예시를 몇 개 넣으면 퓨 샷, 하나도 넣지 않으면 제로 샷이라고 부릅니다. 지시문 튜닝(instruction tuning)을 거친 모델이 늘면서, 프롬프트만 잘 짜도 꽤 복잡한 행동을 끌어낼 수 있다는 결과가 쌓였습니다. 나중에 DSPy가 모듈로 옮기는 대표 기법은 다음과 같습니다.

- **체인 오브 소트(chain-of-thought)**: [Wei et al.](https://arxiv.org/pdf/2201.11903)이 제안한 기법입니다. 퓨 샷 예시에 질문과 답만 넣지 않고, 그 사이의 중간 추론 단계인 추론 근거(rationale)를 함께 넣습니다. 그러면 모델도 근거를 먼저 쓰고 나서 답을 냅니다.
- **자기 일관성(self-consistency)**: [Wang et al.](https://arxiv.org/pdf/2203.11171)이 제안했습니다. 추론 경로를 하나만 탐욕적으로 디코딩하지 않고, 샘플링으로 $$m$$개를 뽑은 뒤 최종 답에 다수결 투표를 합니다. 맞는 추론 경로들은 같은 답으로 모이고, 틀린 경로는 여러 답으로 흩어진다는 직관에 기댑니다.
- **ReAct**: [Yao et al.](https://arxiv.org/pdf/2210.03629)이 제안했습니다. "생각"과 "행동(예: 위키백과 검색)"을 번갈아 생성하고, 행동이 돌려준 관찰 결과를 다음 생각에 반영하는 에이전트 루프입니다.
- **Program of Thought**: 자연어로 추론하는 대신 문제를 푸는 코드를 생성하고, 그 코드를 실행해 답을 얻습니다.
- **다중 체인 비교(multi-chain comparison)**: [Yoran et al.](https://arxiv.org/pdf/2304.13007)의 방식입니다. 여러 체인을 뽑는 데까지는 자기 일관성과 같지만, 답만 세지 않습니다. 체인 전체를 한 번의 LM 호출에 넣어 비교·종합하는 메타 추론으로 새 근거와 답을 만듭니다.

이 기법들은 모두 손으로 쓴 예시와 지시문에 의존합니다.

### LM이 직접 만드는 약지도

예전에는 과제별 휴리스틱이나 손으로 짠 규칙이 있어야 하던 약지도(weak supervision)를 이제는 LM이 직접 수행합니다. 대표 사례가 추론 근거 자기 생성입니다. [STaR](https://arxiv.org/pdf/2203.14465)는 소수의 근거 예시로 시작합니다. 모델이 질문마다 근거와 답을 생성하면, 최종 답이 정답과 일치한 근거만 남겨 그것으로 파인튜닝합니다. 중간 단계에 라벨을 달지 않고, 최종 답의 정답 여부를 근거 품질의 필터로 씁니다. LM의 자기 개선, 체인 오브 소트 예시 자동 생성 같은 연구도 같은 흐름입니다. DSPy의 부트스트래핑은 이 아이디어를 이어받아, 파이프라인 전체를 실행하고 평가 지표를 통과한 결과만 데모로 남깁니다.

### 도구를 엮는 파이프라인과 툴킷

인컨텍스트 학습은 이제 도구 호출과 자주 결합합니다. 검색 모델, 멀티모달 파운데이션 모델, API, 계산기 등이 그런 도구입니다. 검색 증강 생성(RAG)은 그중 대표적인 형태입니다. 질의에 맞는 문서를 먼저 찾고, 그 문서를 조건으로 붙여 답을 생성합니다([Lewis et al.](https://doi.org/10.48550/arXiv.2005.11401)). LangChain, Semantic Kernel, LlamaIndex는 이런 체인과 에이전트를 미리 묶어 제공합니다. 하지만 과제별 행동은 여전히 손으로 쓴 프롬프트 템플릿으로 표현합니다.

### 단일 호출 프롬프트 최적화를 넘어서

프롬프트를 자동으로 찾으려는 시도도 있습니다. 저자들은 이산 최적화와 강화학습을 꼽습니다. 인용된 연구에는 진화 알고리즘을 결합한 탐색이나 LLM 자체를 옵티마이저로 쓰는 방법도 포함됩니다. 다만 대부분 논리적으로 LM 호출 하나에 해당하는 프롬프트만 최적화 대상으로 삼습니다. DSPy는 이 범위를 임의의 다단계 파이프라인으로 넓힙니다. 고수준의 선언적 시그니처에서 출발해, 제약 조건을 만족하는 다단계 데모를 부트스트래핑하는 방식입니다. 텔레프롬프터는 교차 검증 같은 모델 선택 기법으로 최적화할 수 있습니다. 저자들은 원리상 강화학습이나 LM 피드백, 학습 기반 또는 베이지안 하이퍼파라미터 최적화도 쓸 수 있다고 말합니다.

### 접근법의 계보

앞서 말한 딥러닝 프레임워크의 추상화 외에도, DSPy는 미분 가능 프로그래밍의 발상을 신경망이 아닌 LM 호출에 적용하고 문법 요소는 PyTorch에서 빌려 왔습니다. 논증 방식은 Bergstra et al.(Theano, Hyperopt), Paszke et al.(PyTorch), Wolf et al.(Hugging Face Transformers)의 선행 연구에서 착안했습니다. 그래서 벤치마크 수치와 일부 정성적 논거를 함께 제시합니다.

## 시그니처: 프롬프트 대신 자연어로 타입을 선언하는 인터페이스

DSPy는 LM을 텍스트를 생성하는 추상적인 장치로 다룹니다. 프롬프트 문자열을 받아 텍스트 완성을 돌려주기만 하면 됩니다. 인컨텍스트 학습이 가능한 GPT-3.5나 Llama2-7b 같은 모델이어도 되고, T5-base처럼 파인튜닝할 수 있는 작은 모델이어도 됩니다. 기본 LM을 하나 지정해 두면, 따로 설정하지 않는 한 모든 연산이 그 모델을 씁니다. DSPy 프로그램은 평범한 Python 함수입니다. 질문이나 요약할 논문 같은 과제 입력을 받아 여러 단계를 거친 뒤 답이나 요약을 돌려줍니다. 자동 최적화를 위한 추상화는 세 가지이고, 각각 맡는 일이 다릅니다. 시그니처는 모듈의 입출력 동작을 추상화합니다. 모듈은 손으로 쓰던 프롬프팅 기법을 대체하며 임의의 파이프라인으로 조립됩니다. 텔레프롬프터는 파이프라인의 모든 모듈을 최적화해 평가 지표를 최대화합니다.

### "어떻게"가 아닌 "무엇을" 적는 선언

시그니처는 자연어로 적은 함수의 타입 선언입니다. "질문을 받아 답을 돌려준다"처럼 텍스트 변환이 **무엇을** 해야 하는지만 짧게 적습니다. 특정 LM에 그 동작을 시키려면 **어떻게** 프롬프트해야 하는지는 적지 않습니다. 형식적으로는 입력 필드들과 출력 필드들의 튜플이고, 지시문을 선택적으로 덧붙일 수 있습니다. 각 필드는 이름과 선택적 메타데이터로 이루어집니다. 메타데이터로는 프롬프트에서 필드 앞에 붙는 필드 접두사와 설명(description)이 있습니다. 과제 설명이나 필드 설명은 보통 생략하고, 필드 값은 기본적으로 자유 형식 문자열로 취급합니다.

설명을 생략할 수 있는 이유는 필드의 역할을 DSPy가 **필드 이름에서 추론**하기 때문입니다. 컴파일러는 인컨텍스트 학습을 통해 `question`과 `answer`를 서로 다른 역할로 해석하고, 이 필드들을 프롬프트에서 어떻게 쓸지를 반복적으로 다듬어 갑니다. 이름은 사람이 정하지만, 그 이름을 LM에 전달하는 방식은 학습 대상입니다.

### 단축 표기: 한 줄짜리 질의응답 프로그램

실제로는 `question -> answer` 같은 단축 표기를 주로 씁니다. 다음 코드의 첫 줄만으로 기본 질의응답 시스템이 완성됩니다. 둘째 줄은 사용 예이고, 셋째 줄은 GPT-3.5를 LM으로 썼을 때의 응답입니다.

```
qa = dspy.Predict("question -> answer")
qa(question="Where is Guaraní spoken?")
# Out: Prediction(answer='Guaraní is spoken mainly in South America.')
```

단축 표기에서 필드 이름은 그 필드가 변환에서 맡는 의미상 역할을 나타냅니다. DSPy는 이 표기를 파싱한 뒤 필드 이름을 LM에 줄 지시문으로 풀어 씁니다. 예를 들어 `english_document -> french_translation`라고 쓰면 영어를 프랑스어로 번역하라는 프롬프트가 만들어집니다. 프로그래밍 언어의 타입 시그니처처럼 시그니처는 인터페이스를 정의하고 기대 동작을 힌트로 줄 뿐입니다. 실제로 실행하려면 위의 Predict처럼 그 시그니처를 가진 모듈을 선언해야 합니다.

### 프롬프트 문자열보다 나은 두 가지

첫째, 시그니처는 컴파일할 수 있습니다. 같은 선언이 스스로 개선되면서 파이프라인에 맞춰지는 프롬프트가 될 수도 있고, 파인튜닝된 모델이 될 수도 있습니다. 이 과정의 중심은 시그니처마다 쓸모 있는 데모를 부트스트래핑하는 일입니다. 하드코딩된 문자열에는 고칠 여지가 없습니다. 반면 시그니처는 "무엇을"만 고정해 두기 때문에, 컴파일러가 "어떻게"를 LM과 파이프라인에 맞게 채울 수 있습니다. 둘째, 구조화된 형식 지정과 파싱을 DSPy가 맡습니다. 그래서 사용자 프로그램에서 깨지기 쉬운 문자열 조작을 줄일 수 있고, 이상적으로는 아예 없앨 수 있습니다.

### 클래스 기반 시그니처

변환을 더 세밀하게 제어하고 싶으면 시그니처를 Python 클래스로 정의합니다. 이때 docstring이 지시문이 되고, `InputField`와 `OutputField`의 `desc`로 각 필드의 형식이나 역할을 직접 설명합니다. 다음 시그니처는 문맥과 선택적인 질문으로 검색 질의를 만듭니다.

```
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField(dtype=dspy.SearchQuery)
```

이것을 `dspy.Predict(GenerateSearchQuery)`로 감싸면 합성 정보 검색(IR) 데이터셋을 만드는 시스템이 됩니다. `context="Language typology"`만 넣으면 LM이 질문을 먼저 생성하고, 그 질문을 거쳐 질의를 만듭니다. 출력 예는 `'"language classification" OR "language typology" -wikipedia'`입니다. 질문이 이미 있다면 `question=`으로 함께 넘기면 됩니다.

출력 필드의 타입을 기본 문자열 대신 bool, int, float, list, dict로 지정하는 기능도 있습니다. `contexts, question -> answer_found: bool`처럼 씁니다. 다만 논문 작성 시점에는 개발 중인 실험적 기능이었고, DSPy의 핵심 기능은 아닙니다.

## 모듈: 프롬프팅 기법을 파라미터화된 레이어로 바꾸기

모듈 선언은 해당 시그니처를 가진 함수를 돌려줍니다. PyTorch의 레이어처럼 DSPy 모듈도 호출 가능한 객체입니다. 차이는 내부에 담긴 "가중치"가 숫자 텐서가 아니라 LM 선택, 지시문, 데모라는 점입니다.

### Predict: 모든 시그니처를 실행하는 기본 모듈

시그니처를 다루는 핵심 모듈은 Predict입니다. 내부 상태는 세 가지입니다. 첫째는 전달받은 시그니처입니다. 둘째는 선택적으로 지정하는 LM으로, 처음 값은 None이고 지정하면 이 모듈에서만 기본 LM 대신 쓰입니다. 셋째는 프롬프트에 넣을 데모 목록으로, 처음에는 비어 있습니다. 논문 부록의 간략화된 의사코드는 다음과 같습니다.

```
class Predict(dspy.Module):
    def __init__(self, signature, **config):
        self.signature = dspy.Signature(signature)
        self.config = config

        # Module Parameters.
        self.lm = dspy.ParameterLM(None) # use the default LM
        self.demonstrations = dspy.ParameterDemonstrations([])

    def forward(self, **kwargs):
        lm = get_the_right_lm(self.lm, kwargs)
        signature = get_the_right_signature(self.signature, kwargs)
        demonstrations = get_the_right_demonstrations(self.demonstrations, kwargs)

        prompt = signature(demos=self.demos, **kwargs)
        completions = lm.generate(prompt, **self.config)
        prediction = Prediction.from_completions(completions, signature=signature)

        if dsp.settings.compiling is not None:
            trace = dict(predictor=self, inputs=kwargs, outputs=prediction)
            dspy.settings.traces.append(trace)

        return prediction
```

`lm`과 `demonstrations`는 `ParameterLM`, `ParameterDemonstrations`로 감싸여 있어 "모듈 파라미터"로 명시됩니다. 호출 흐름은 다음과 같습니다. `question`처럼 시그니처의 입력 필드에 해당하는 키워드 인자를 받습니다. 시그니처와 데모로 프롬프트를 구성하고, LM을 호출한 뒤, 완성 텍스트에서 출력 필드를 파싱해 `Prediction`으로 돌려줍니다.

마지막 `if` 블록이 컴파일 모드(compile mode)입니다. 컴파일 중이라는 설정이 켜져 있으면 Predict는 호출할 때마다 `(predictor, inputs, outputs)` 형태의 트레이스를 전역 목록에 추가합니다. 프로그램 안에 Predict가 여러 개 있어도, 실행이 끝나면 어느 모듈이 무엇을 받아 무엇을 냈는지 모두 남습니다. 텔레프롬프터는 이 기록을 재료로 데모를 부트스트래핑합니다.

### 내장 모듈: 기법을 시그니처에 독립적인 함수로

기존 방식은 손으로 쓴 퓨 샷 예시처럼 과제에 특화된 내용을 프롬프트에 직접 넣습니다. DSPy 모듈은 이와 달리 프롬프팅 기법을 어떤 시그니처에도 적용되는 모듈형 함수로 옮깁니다. 내장 모듈로는 ChainOfThought, ProgramOfThought, MultiChainComparison, ReAct가 있습니다. 각각 체인 오브 소트, Program of Thought, 다중 체인 비교, ReAct를 일반화한 것입니다. 저자들은 이 과정에서 제로 샷 프롬프팅과 추론 근거 자기 생성의 아이디어도 함께 일반화했다고 설명합니다. 그 결과 임의의 다단계 파이프라인을 부트스트래핑할 수 있는 파라미터화된 모듈이 됩니다.

내장 모듈은 모두 같은 시그니처를 구현하므로 서로 바꿔 끼울 수 있습니다. 앞서 본 `dspy.Predict("question -> answer")`에서 `Predict`를 `ChainOfThought`로 바꾸기만 하면, 출력 필드를 확정하기 전에 단계별로 추론하는 시스템이 됩니다.

### 몇 줄짜리 ChainOfThought

내장 모듈은 몇 줄의 코드로 구현됩니다. 사용자 시그니처를 확장한 뒤, 새 시그니처로 Predict를 한 번 이상 호출하는 방식입니다. ChainOfThought의 간략화된 구현은 다음과 같습니다.

```
class ChainOfThought(dspy.Module):
    def __init__(self, signature):
        # Modify signature from ‘*inputs -> *outputs‘ to ‘*inputs -> rationale, *outputs‘.
        rationale_field = dspy.OutputField(prefix="Reasoning: Let’s think step by step.")
        signature = dspy.Signature(signature).prepend_output_field(rationale_field)

        # Declare a sub-module with the modified signature.
        self.predict = dspy.Predict(signature)

    def forward(self, **kwargs):
        # Just forward the inputs to the sub-module.
        return self.predict(**kwargs)
```

이 모듈이 하는 일은 하나입니다. `*inputs -> *outputs` 형태의 시그니처 앞쪽 출력에 추론 근거 필드를 끼워 넣어 `*inputs -> rationale, *outputs`로 바꿉니다. 이 필드의 필드 접두사는 `"Reasoning: Let's think step by step."`입니다. 나머지 일은 Predict에 맡깁니다. LM은 출력 필드를 순서대로 채우므로 근거를 먼저 쓰고 답을 나중에 씁니다.

이 코드에는 추론 예시가 하나도 없습니다. 기존 체인 오브 소트 프롬프트는 연구 논문이나 인기 프롬프팅 라이브러리 할 것 없이 길게 손으로 쓴 추론 예시를 통째로 담고 있습니다. 앞서 표로 정리한 손으로 쓴 추론 프롬프트들이 그런 예입니다. 그런데도 저자들은 이 몇 줄짜리 클래스가 어떤 LM이나 과제에서든 효과적인 퓨 샷 프롬프팅을 학습할 수 있는 완전한 모듈이라고 말합니다. 예시가 코드에 없어도 되는 이유는 데모가 파라미터이기 때문입니다.

### 파라미터화: 프롬프트를 이루는 요소를 학습 대상으로

DSPy가 프롬프팅 기법을 **파라미터화**한다는 말은 다음과 같은 관찰에서 나옵니다. 특정 시그니처를 구현하려는 LM 호출은 어떤 형태든 아래 세 가지를 정해야 합니다.

1. **호출할 LM**: 어떤 모델에 보낼지 정합니다.
2. **지시문과 각 필드의 필드 접두사**: 과제를 설명하는 문장과, 프롬프트에서 각 필드 앞에 붙는 문자열입니다.
3. **데모**: 고정된 LM에서는 퓨 샷 프롬프트의 예시로 쓰이고, 파인튜닝할 때는 학습 데이터로 쓰입니다. 저자들은 이것을 가장 중요한 요소로 꼽습니다.

손으로 쓴 프롬프트 템플릿은 이 세 가지를 문자열 하나에 한꺼번에 굳혀 버립니다. DSPy는 이들을 모듈의 상태로 분리합니다. 앞서 본 Predict의 `self.lm`과 `self.demonstrations`가 그 상태입니다. 신경망에 비유하면 시그니처와 모듈 구조는 아키텍처이고, 이 세 요소는 가중치입니다. 아키텍처를 그대로 두고 가중치만 바꾸면 행동이 달라지듯, 같은 `ChainOfThought("question -> answer")`도 담긴 데모에 따라 다른 프롬프트가 됩니다.

> **핵심:** 데모가 모듈 파라미터라서 같은 데모 집합이 두 가지로 쓰입니다. 고정된 대형 LM에서는 프롬프트에 붙는 퓨 샷 예시가 되고, 작은 LM에서는 파인튜닝 데이터가 됩니다. 그래서 "프롬프트 최적화"와 "파인튜닝"이 데모를 만들고 고르는 하나의 문제로 묶입니다.

이 관점이 미묘한 이유가 있습니다. 숫자 가중치와 달리 데모에는 기울기가 없습니다. 다단계 파이프라인에서는 중간 모듈의 "정답" 데모, 예컨대 좋은 검색 질의가 무엇인지 아무도 라벨을 달아 주지 않습니다. 그래서 파라미터를 채우는 방법이 따로 필요합니다. 앞서 본 컴파일 모드의 트레이스가 그 출발점입니다. 프로그램을 실행해 최종 결과가 좋은 경우의 각 모듈 입출력을 그 모듈의 데모로 쓰면 됩니다. 논문은 세 요소 가운데 주로 데모를 자동으로 생성하고 선택하는 일에 집중합니다. 저자들은 사례 연구에서 좋은 데모를 부트스트래핑하는 것이 정교한 LM 파이프라인에 새로운 행동을 체계적으로 가르치는 강력한 방법이었다고 보고합니다.

### 도구: 계산을 실행하는 모듈

도구는 LM을 호출하지 않고 계산을 실행하는 모듈입니다. 검색 모델은 `dspy.Retrieve` 모듈로 쓰며, 논문 작성 시점 기준으로 ColBERTv2, Pyserini, Pinecone 검색기를 기본 지원합니다. 실험에서 기본 검색기로 쓰는 [ColBERTv2](https://arxiv.org/pdf/2112.01488)는 질의와 문서를 토큰 단위 임베딩으로 표현하는 후기 상호작용(late interaction) 신경망 검색기입니다. 질의 토큰마다 문서 토큰과의 최대 유사도를 구해 합산하는 방식으로 점수를 매기고, 잔차 압축으로 인덱스 크기를 줄였습니다. DSPy에서는 고정된 인터페이스 뒤에 숨은 백엔드일 뿐이고, 검색기 가중치는 최적화 대상이 아닙니다. 이 밖에 SQL 질의를 실행하는 `dspy.SQL`과 샌드박스에서 Python 코드를 실행하는 `dspy.PythonInterpreter`도 실험적으로 탐색했습니다.

### 실행 시 정의(define-by-run) 프로그램과 RAG

모듈은 실행 시 정의 인터페이스로 임의의 파이프라인에 조립됩니다. 실행 시 정의는 Chainer와 PyTorch가 대중화한 방식입니다. 계산 그래프를 미리 정적으로 선언하지 않고, 평범한 코드가 실행되는 동안 그래프가 만들어집니다. 그래서 반복문이나 조건문 같은 제어 흐름을 그대로 쓸 수 있습니다. DSPy는 이 구조를 그대로 따릅니다. 필요한 모듈은 `__init__`에서 선언합니다. 그래야 DSPy가 최적화 대상 모듈을 추적할 수 있습니다. 파이프라인 자체는 `forward` 메서드 안에 임의의 코드로 적습니다. 다음은 단순하지만 완전한 RAG 시스템입니다.

```
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        # ‘Retrieve‘ will use the user’s default retrieval settings unless overriden.
        self.retrieve = dspy.Retrieve(k=num_passages)
        # ‘ChainOfThought‘ with signature that generates answers given retrieval & question.
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate_answer(context=context, question=question)
```

질문으로 패시지 $$k$$개(기본값 3)를 검색합니다. 그 패시지를 `context`로 삼아 `context, question -> answer` 시그니처의 ChainOfThought가 답을 생성합니다. 모듈성을 보여 주려고 기본 Predict 자리에 ChainOfThought를 그대로 끼워 넣었습니다. 사용할 때는 `RAG()("Where is Guaraní spoken?")`처럼 호출하면 됩니다. 시그니처 한 줄을 `"context, question -> search_query"`로 바꾸면 같은 구조가 답 대신 검색 질의를 만드는 시스템이 됩니다. 이 RAG 프로그램에는 프롬프트 문자열이 한 줄도 없습니다. 학습 가능한 파라미터는 `generate_answer` 안 Predict의 LM·지시문·데모이며, 이 파라미터를 채우는 일은 텔레프롬프터가 맡습니다.

## 텔레프롬프터: 소량 데이터와 평가 지표로 프로그램을 컴파일하는 방식

DSPy 프로그램을 컴파일할 때는 보통 텔레프롬프터(teleprompter)를 호출합니다. 텔레프롬프터는 프로그램, 학습 집합, 평가 지표를 받아 최적화된 **새 프로그램**을 돌려주는 옵티마이저입니다. 최적화 전략은 텔레프롬프터마다 다릅니다.

### 중간 단계 라벨이 필요 없는 학습 집합

신경망 학습과 비교하면 필요한 데이터가 매우 적습니다. DSPy의 학습 집합은 예시 몇 개로도 충분하고, 데이터가 많으면 더 강한 최적화를 할 수 있습니다. 예시가 불완전해도 됩니다. 입력값만 있어도 되고, 파이프라인 중간 단계의 라벨은 평가 지표가 직접 쓰지 않는 한 필요 없습니다. 저자들은 실제로 라벨을 많아야 프로그램의 **최종 출력**에만 둔다고 가정합니다.

이 라벨 효율성은 모듈성과 직결됩니다. 새 파이프라인을 만들 때는 그 코드를 다시 컴파일하기만 하면 되고, 새 파이프라인 전용 데이터에 주석을 달 필요가 없습니다. 예를 들어 앞서 본 RAG에 검색 질의를 생성하는 단계를 하나 더 끼워 넣는 경우를 생각해 볼 수 있습니다. 이때도 "좋은 검색 질의"라는 새 라벨을 만들지 않고 다시 컴파일하면 됩니다. 새 파이프라인마다 그 구조에 맞춘 주석 데이터를 만들어야 했다면, 모듈을 바꿔 끼울 때마다 데이터 구축 비용이 들었을 것입니다.

### 평가 지표: EM에서 DSPy 프로그램까지

평가 지표는 정확 일치(EM)나 F1처럼 단순할 수도 있고, 여러 기준을 함께 따지는 DSPy 프로그램 전체일 수도 있습니다.

가장 단순한 사용 예는 앞서 정의한 RAG를 질문–답 쌍과 EM으로 컴파일하는 것입니다. 이때 최적화의 목표는 퓨 샷 데모를 효과적으로 부트스트래핑하는 일입니다.

```
# Small training set with only questions and final answers.
qa_trainset = [dspy.Example(question="What is the capital of France?", answer="Paris")]

# The teleprompter will bootstrap missing labels: reasoning chains and retrieval contexts.
teleprompter = dspy.BootstrapFewShot(metric=dspy.evaluate.answer_exact_match)
compiled_rag = teleprompter.compile(RAG(), trainset=qa_trainset)
```

학습 예시에는 질문과 최종 답만 있고, ChainOfThought가 쓸 추론 근거나 검색 문맥은 없습니다. BootstrapFewShot은 학습 예시로 RAG를 직접 실행해 봅니다. 그리고 모든 모듈의 출력이 모여 유효한 결과, 즉 시그니처를 지키고 평가 지표를 통과하는 결과로 이어진 실행에서 각 모듈의 입출력 사례를 그 모듈의 데모로 모읍니다. 비어 있던 추론 근거와 검색 문맥은 이렇게 프로그램이 스스로 채웁니다. 중간 단계가 옳았는지는 최종 결과가 평가 지표를 통과했는지로 간접 판정하므로, 라벨은 최종 답에만 있으면 됩니다.

평가 지표를 바꾸면 프로그램이 배우는 행동도 달라집니다. 검색된 문맥에서 답을 그대로 뽑아 오는 추출형 답변을 유도하고 싶다면, `answer_exact_match` 대신 다음과 같은 지표를 씁니다.

```
def answer_and_context_match(example, pred, trace=None):
    answer_match = dspy.evaluate.answer_exact_match(example, pred)

    # Is the prediction a substring of some passage?
    context_match = any((pred.answer.lower() in c) for c in pred.context)

    return answer_match and context_match
```

이 지표는 답이 정답과 일치하면서 검색된 패시지 가운데 하나의 부분 문자열이기도 해야 통과시킵니다. 맞는 답이라도 문맥에 없는 표현으로 쓰면 그 실행은 데모로 채택되지 않습니다. 그래서 컴파일된 프로그램은 문맥에서 답을 가져오는 쪽으로 기웁니다. 다만 부분 문자열 검사는 근거 확인 방법으로 거친 편입니다. 저자들은 답이 문맥에 충실하게 근거하는지를 별도의 DSPy 프로그램이 판정하면 더 정확할 수 있다고 보고, 이런 지표를 완전히 지원하며 권장합니다. 평가 지표 자체를 LM 파이프라인으로 만들 수도 있습니다.

### 교사 프로그램으로 텔레프롬프터 합성하기

텔레프롬프터는 교사 프로그램(teacher program)을 지정하는 방식으로 서로 합성할 수 있습니다. 교사를 지정하면 DSPy는 프롬프트 최적화에 쓸 데모를 그 교사 프로그램에서 샘플링합니다.

이 구조는 지식 증류(knowledge distillation)와 닮았습니다. 지식 증류는 크고 비싼 교사 모델의 출력을 작은 학생 모델의 학습 신호로 삼아, 학생이 교사의 행동을 적은 비용으로 흉내 내게 하는 기법입니다. DSPy에서는 단일 모델이 아니라 **프로그램 전체**가 교사와 학생이 됩니다. 대형 LM을 쓰는 복잡한 앙상블 같은 비싼 프로그램이 교사가 되고, 작은 LM을 쓰는 단순한 파이프라인이 학생 프로그램으로서 감독을 받습니다.

논문의 예는 앞서 만든 `compiled_rag`에서 출발합니다. 이 프로그램이 llama2-13b-chat을 쓰도록 컴파일되어 있다고 가정하고, 이를 교사로 삼아 Flan-T5-large를 파인튜닝해 효율적인 프로그램을 만듭니다.

```
# Larger set of questions with *no labels*. Labels for all steps will be bootstrapped.
unlabeled_questions = [dspy.Example(question="What is the capital of Germany?"), ...]

# As we assumes no answer, we use ‘answer_passage_match‘ to filter ungrounded answers.
finetuning_teleprompter = BootstrapFinetune(metric=dspy.evaluate.answer_passage_match)

# We set ‘teacher=compiled_rag‘ to compose. Bootstrapping will now use ‘compiled_rag‘.
compiled_rag_via_finetune = finetuning_teleprompter.compile(RAG(), teacher=compiled_rag, trainset=unlabeled_questions, target=’google/flan-t5-large’)
```

여기서 학생은 `compile`에 넘긴 빈 `RAG()`이고, 교사는 `teacher=compiled_rag`입니다. DSPy는 이 교사 프로그램에서 데모를 샘플링합니다. 두 프로그램이 같은 RAG 구조이므로, 교사가 단계마다 남긴 입출력이 학생의 대응 단계에 들어간다고 볼 수 있습니다. 의도는 학생이 이미 데모로 다듬어진 대형 LM 파이프라인의 행동을 물려받게 하는 것입니다.

이 예가 앞의 예와 다른 점은 셋입니다. 첫째, 학습 집합이 더 크지만 **답 라벨이 전혀 없는** 질문 목록입니다. 모든 단계의 라벨을 부트스트래핑으로 만듭니다. 둘째, 정답이 없으므로 EM을 쓸 수 없습니다. 대신 `answer_passage_match`로 답이 검색된 패시지에 근거하는지 확인해 근거 없는 답을 걸러 냅니다. 정답 대신 근거만 확인하므로, 이 필터를 통과한 데모도 틀릴 수 있습니다. 셋째, 데모의 용도가 다릅니다. BootstrapFinetune은 데모를 프롬프트에 붙이지 않고, `target`으로 지정한 Flan-T5-large의 파인튜닝 데이터로 씁니다.

전체 흐름은 다음과 같습니다. 먼저 질문–답 몇 쌍으로 대형 LM 프로그램을 컴파일합니다. 다음으로 그 프로그램이 라벨 없는 질문들에 대해 모든 단계의 입출력을 만들어 냅니다. 마지막으로 근거 확인을 통과한 출력만 골라 작은 모델을 학습시킵니다. 사람이 단 라벨은 처음의 최종 답 몇 개뿐인데, 학생 프로그램은 모든 중간 단계의 학습 데이터를 얻게 됩니다. 첫 번째 컴파일의 결과물이 두 번째 컴파일의 입력이 되므로, 텔레프롬프터도 모듈처럼 쌓아 올릴 수 있습니다.

> **핵심:** 교사 프로그램이 라벨을 만들어 냅니다. 라벨 효율성과 데모의 이중 용도가 합쳐지면, 비싼 파이프라인의 행동을 싼 모델로 옮기는 과정을 텔레프롬프터 두 개를 이어 붙이는 코드 몇 줄로 표현할 수 있습니다.

## 컴파일러 내부: 데모 부트스트래핑, 후보 탐색, 앙상블

DSPy 프로그램은 컴파일 덕분에 수작업 프롬프트 없이 동작합니다. 텔레프롬프터는 프롬프팅이나 파인튜닝으로 모듈의 품질(또는 비용)을 개선하고, DSPy는 이 둘을 같은 최적화의 두 수단으로 다룹니다. 새 텔레프롬프터를 만들 때 DSPy가 형식을 강제하지는 않습니다. 다만 전형적인 텔레프롬프터는 다음 세 단계를 거칩니다.

1. **후보 생성**: 각 Predict 모듈의 파라미터에 들어갈 후보 값을 만듭니다.
2. **파라미터 최적화**: 후보 가운데 무엇을 쓸지 고르거나, 후보로 LM 가중치를 학습합니다.
3. **고차 프로그램 최적화**: 프로그램의 제어 흐름 자체를 바꿉니다.

### 1단계: 트레이스를 걸러 데모로 만드는 부트스트래핑

컴파일러는 먼저 프로그램 안의 고유한 Predict 모듈(예측기)을 재귀적으로 모두 찾습니다. ChainOfThought 같은 내장 모듈도 내부에서 Predict를 호출하므로, 다른 모듈 아래에 중첩된 Predict도 빠짐없이 모읍니다. 텔레프롬프터는 이렇게 찾은 예측기 $$p$$마다 파라미터 후보를 생성합니다. 대상은 지시문, 필드 설명, 그리고 가장 중요한 데모(입력–출력 예시 쌍)입니다. 현재 버전의 DSPy는 데모에 집중합니다. 저자들은 기각 샘플링과 비슷한 단순한 방법이 매우 효과적인 다단계 시스템을 부트스트래핑하는 데 도움이 된다고 보고합니다.

기각 샘플링(rejection sampling)은 확률론의 기법입니다. 다루기 쉬운 제안 분포에서 샘플을 뽑고, 수락 판정을 통과한 샘플만 남깁니다. 고전적 기각 샘플링에서는 수락 확률을 목표 분포와 제안 분포의 비율로 계산하므로, 남은 샘플이 정확히 목표 분포를 따릅니다. DSPy의 대응은 느슨합니다. 제안 분포 자리에는 교사 프로그램이 입력마다 만들어 내는 실행 결과의 분포가 옵니다. 수락 판정 자리에는 확률이 아니라 평가 지표의 통과 여부라는 결정적 판정이 옵니다. 따라서 남은 데모가 어떤 분포를 따른다는 보장은 없습니다. 논문이 "rejection-sampling-*like*"라고 쓰는 것도 이 때문입니다. 남는 것은 "생성하고, 시험을 통과한 것만 남긴다"는 구조입니다.

가장 단순하면서도 의미 있는 텔레프롬프터인 BootstrapFewShot이 이 방식으로 동작합니다. 먼저 교사 프로그램을 학습 입력에 대해 실행합니다. 교사를 지정하지 않으면 컴파일 중인 프로그램의 제로 샷 버전, 즉 데모가 비어 있는 학생 자신이 교사가 됩니다. 입력마다 한 번 실행할 수도 있고, 높은 샘플링 온도로 여러 번 실행해 다양한 출력을 얻을 수도 있습니다. 실행은 컴파일 모드에서 이루어지므로 다단계 트레이스가 실행 내내, 사용자가 신경 쓰지 않아도 스레드 안전하게 기록됩니다. 트레이스는 앞서 본 Predict의 컴파일 모드가 남기는 호출 기록입니다.

다음으로 프로그램의 평가 지표를 써서, 여러 단계가 합쳐져 파이프라인 전체가 지표를 통과한 트레이스만 남깁니다. 나쁜 실행은 버리고 좋은 실행의 각 단계를 데모 후보로 삼으면 프로그램 안의 **모든 시그니처**에 대한 잠정 라벨이 생깁니다. 무엇을 버리고 무엇을 쓸지 같은 설계 결정은 사용자가 바꿀 수 있습니다.

설명용 수치로 작은 예를 들어 보겠습니다. 질문 → 검색 질의 생성 → 검색 → 답 생성으로 이어지는 2단계 프로그램이 있고, 학습 질문 10개를 교사로 실행했더니 4개에서 최종 답이 EM을 통과했다고 합시다. 이 4번의 실행에는 각각 질의 생성 모듈의 (질문 → 질의) 기록과 답 생성 모듈의 (문맥·질문 → 근거·답) 기록이 있습니다. BootstrapFewShot은 앞의 기록 4개를 질의 생성 모듈의 데모로, 뒤의 기록 4개를 답 생성 모듈의 데모로 넣습니다. 나머지 6번의 실행은 어느 단계가 잘못됐는지와 상관없이 통째로 버립니다. 사람이 "좋은 검색 질의"에 라벨을 단 적이 없는데도, 최종 결과로 간접 검증된 질의 예시가 생깁니다.

미분 가능한 신경망이라면 최종 손실의 기울기가 각 레이어에 책임을 나눠 줍니다. LM 파이프라인에는 기울기가 없어서 이 신용 할당이 어렵습니다. BootstrapFewShot은 "끝까지 성공한 실행의 모든 단계는 좋은 예"라고 가정해 이 문제를 우회하고, 판정 기준을 평가 지표 하나로 단순하게 만듭니다. 일반적으로 보면 이런 방식에서는 중간 단계가 틀렸는데 우연히 최종 답이 맞은 실행도 데모로 들어갈 수 있으며, 논문은 이 한계를 따로 다루지 않습니다.

#### 부록 의사코드로 본 BootstrapFewShot

논문 부록의 간략화된 의사코드는 다음과 같습니다.

```
class SimplifiedBootstrapFewShot(Teleprompter):
    def __init__(self, metric=None):
        self.metric = metric

    def compile(self, student, trainset, teacher=None):
        teacher = teacher if teacher is not None else student
        compiled_program = student.deepcopy()

        # Step 1. Prepare mappings between student and teacher Predict modules.
        # Note: other modules will rely on Predict internally.
        assert student_and_teacher_have_compatible_predict_modules(student, teacher)
        name2predictor, predictor2name = map_predictors_recursively(student, teacher)

        # Step 2. Bootstrap traces for each Predict module.
        # We’ll loop over the training set. We’ll try each example once for simplicity.
        for example in trainset:
            if we_found_enough_bootstrapped_demos(): break

            # turn on compiling mode which will allow us to keep track of the traces
            with dspy.setting.context(compiling=True):
                # run the teacher program on the example, and get its final prediction
                # note that compiling=True may affect the internal behavior here
                prediction = teacher(**example.inputs())

                # get the trace of the all interal Predict calls from teacher program
                predicted_traces = dspy.settings.trace

            # if the prediction is valid, add the example to the traces
            if self.metric(example, prediction, predicted_traces):
                for predictor, inputs, outputs in predicted_traces:
                    d = dspy.Example(automated=True, **inputs, **outputs)
                    predictor_name = self.predictor2name[id(predictor)]
                    compiled_program[predictor_name].demonstrations.append(d)

        return compiled_program
```

코드만 봐서는 잘 드러나지 않는 점이 셋 있습니다. 첫째, 예측기 대응표가 교사와 학생을 잇습니다. 트레이스는 교사의 Predict 객체를 가리키지만, 데모는 학생 사본의 같은 이름 Predict에 들어가야 합니다. 그래서 교사와 학생의 Predict 구조가 호환되어야 하고, 코드는 이를 먼저 확인합니다. 둘째, 데모는 원래 학생이 아니라 깊은 복사본에 쌓입니다. 텔레프롬프터가 입력을 건드리지 않고 새 프로그램을 돌려주는 이유입니다. 셋째, 평가 지표는 `metric(example, prediction, trace)`처럼 트레이스도 인자로 받습니다. 그래서 최종 예측만 볼 수도 있고, 원하면 중간 단계 출력에 "검색 질의가 너무 길면 탈락" 같은 조건을 걸 수도 있습니다. 판정을 통과하면 트레이스의 각 `(inputs, outputs)`는 `automated=True` 표시가 붙은 예시가 되어, 대응하는 학생 예측기의 데모 목록에 추가됩니다. 데모가 충분히 모이면 루프는 학습 집합을 다 돌기 전에 멈춥니다.

#### LM이 불안정한데도 통하는 이유

LM은 매우 불안정할 수 있습니다. 그래도 저자들은 다단계 설계의 해 공간을 탐색하는 데에는 LM이 꽤 효율적이라고 보고합니다. 잘 분해된 프로그램이라면 학습 예시 가운데 적어도 몇 개에서는 보통 LM이 시그니처와 평가 지표의 제약을 통과합니다. 그 몇 개가 데모가 되므로, 전체 성공률이 낮아도 출발할 수 있습니다.

저자들은 이 덕분에 필요하면 부트스트래핑을 반복할 수 있다고 말합니다. 반복의 한 가지 자연스러운 형태는 1차에서 얻은 데모를 가진 컴파일된 프로그램을 교사로 지정해 다시 BootstrapFewShot을 돌리는 것입니다. 앞서 본 교사 프로그램 합성 구조를 그대로 쓰는 방식입니다. 데모가 붙은 교사가 더 많은 예시를 통과시킬 것이라는 기대는 일반적인 추론이고, 논문이 이 단계에서 직접 보인 결과는 아닙니다.

### 2단계: 후보 집합에서 고르는 하이퍼파라미터 탐색

1단계를 마치면 각 파라미터에 데모, 지시문 같은 이산적인 후보 집합이 생깁니다. 남은 문제는 어떤 조합을 쓸지이고, 저자들은 이를 하이퍼파라미터 튜닝으로 다룹니다. 보통은 학습 집합이나 검증 집합에 대한 교차 검증으로 평가 지표의 평균 품질을 최적화합니다. 지표의 성격에 따라서는 어떤 단계에도 라벨이 없어도 됩니다.

형식적으로 쓰면 다음과 같습니다. $$P$$는 프로그램의 예측기 집합이고, $$\mathcal{D}_p$$는 예측기 $$p$$의 부트스트래핑된 데모 후보 집합입니다. $$\mathbf{d} = (d_p)_{p \in P}$$는 예측기마다 고른 후보이고, $$\Pi(\mathbf{d})$$는 그 선택으로 만든 프로그램입니다. $$V$$는 검증 집합, $$\mu$$는 평가 지표입니다. 목표는 다음 식입니다.

$$
\mathbf{d}^* = \arg\max_{\mathbf{d}} \; \frac{1}{\lvert V \rvert}\sum_{(x,y)\in V} \mu\big(\Pi(\mathbf{d})(x),\, y\big)
$$

이 함수는 기울기가 없고, 한 번 평가하는 데 LM 호출이 여러 번 필요한 비싼 블랙박스 함수입니다. 그래서 일반적인 하이퍼파라미터 최적화 알고리즘을 그대로 쓸 수 있습니다. 논문은 무작위 탐색(random search)과, HyperOpt·Optuna에 구현된 트리 구조 파젠 추정기(TPE)를 예로 듭니다.

#### 무작위 탐색: BootstrapFewShotWithRandomSearch

무작위 탐색은 설정을 무작위로 여러 개 뽑아 각각 평가하고 가장 좋은 것을 남기는, 가장 단순한 하이퍼파라미터 최적화 방법입니다. BootstrapFewShotWithRandomSearch는 미리 모은 데모를 조합하지 않고, BootstrapFewShot 자체를 무작위 조건을 바꿔 가며 여러 번 돌립니다.

```
class SimplifiedBootstrapFewShotWithRandomSearch(Teleprompter):
    def __init__(self, metric = None, trials=16):
        self.metric = metric
        self.trials = trials

    def compile(self, student, *, teacher=None, trainset, valset=None):
        # we can do forms of cross-validation if valset is unset.
        valset = trainset if valset is None else valset

        candidates = []
        for seed in range(self.trials):
            # Create a new basic bootstrap few-shot program.
            shuffled_trainset = shuffle(trainset, seed=seed)
            tp = BootstrapFewShot(metric=metric, max_bootstrap_demos=random_size())
            candidate_program = tp.compile(student, shuffled_trainset, teacher)

            # Step 2: Evaluate the generated candidate program.
            score = evaluate_program(candidate_program, self.metric, valset)
            candidates.append((score, candidate_program))

        # return the best candidate program.
        return max(candidates, key=lambda x: x[0])[1]
```

무작위성의 출처는 둘입니다. 하나는 시드별로 섞은 학습 집합입니다. BootstrapFewShot은 데모가 충분히 모이면 멈추므로, 순서가 바뀌면 데모가 되는 예시도 바뀝니다. 다른 하나는 시도마다 무작위로 정하는 데모 수 상한입니다. 각 후보 프로그램은 검증 집합에서 평가되고, 최고 점수 프로그램이 결과가 됩니다. 검증 집합을 주지 않으면 학습 집합을 검증에도 씁니다. 이 방식에서는 1단계와 2단계가 한 루프 안에서 함께 돌고, 비교 단위는 데모 집합 전체입니다.

#### TPE와 Optuna: BootstrapFewShotWithOptuna

배경 지식으로 정리하면, TPE는 이전 시도 결과를 활용하는 순차적 모델 기반 탐색입니다. [Bergstra et al.](https://openalex.org/W2113207845)은 이를 수백 차원의 대규모 조건부 탐색 공간에 적용했습니다. TPE는 지금까지 관측한 설정을 점수 분위수 $$\gamma$$를 기준으로 "좋은" 쪽과 "나쁜" 쪽으로 나누고, 각각의 밀도 $$\ell(x)$$, $$g(x)$$를 파젠(커널 밀도) 추정으로 모델링합니다. 다음 후보는 $$\ell(x)/g(x)$$가 큰 곳, 즉 좋은 설정 근처에서는 흔하고 나쁜 설정 근처에서는 드문 곳에서 고릅니다. 표준 유도에서 기대 개선량은 다음에 비례합니다.

$$
\mathrm{EI}(x) \propto \left(\gamma + (1-\gamma)\,\frac{g(x)}{\ell(x)}\right)^{-1}
$$

[Optuna](https://doi.org/10.1145/3292500.3330701)는 TPE를 기본 샘플러로 쓰는 하이퍼파라미터 최적화 프레임워크로, 탐색 공간을 목적 함수가 실행되는 동안 정의하는 define-by-run API가 특징입니다. Optuna 논문의 설명에 따르면 TPE는 차원마다 독립적으로 모델링하므로 이산·범주형 변수가 섞인 공간에도 적용하기 쉽습니다. 이는 DSPy 논문이 아니라 Optuna 쪽 배경입니다.

BootstrapFewShotWithOptuna는 1단계와 2단계를 명확히 나눕니다.

```
class SimplifiedBootstrapFewShotWithOptuna(Teleprompter):
    def __init__(self, metric, trials=16):
        self.metric = metric
        self.trials = trials

    def objective(self, trial):
        pool = self.pool

        # Step 1: Create copy of student program.
        candidate_program = self.student.reset_copy()

        # Step 2: Based on trial, select demos for each predictor in program.
        # Note. For simplicity, we can just select a single demo for each predictor.
        # But we can easily tune the number of demonstrations to select here.
        for (name, predictor1), (_, predictor2) in \
        zip(pool.named_predictors(), candidate_program.named_predictors()):
            all_demos = predictor1.demos
            demo_index = trial.suggest_int(f"demo_index_for_{name}", 0, len(all_demos) - 1)
            predictor2.demos = [all_demos[demo_index]]

        # Step 3: Evaluate the modified candidate program.
        score = evaluate_program(candidate_program, self.metric, self.valset)

        # Step 4: Store the candidate for Optuna to select highest-scoring program.
        trial.set_user_attr("program", candidate_program)
        return score

    def compile(self, student, trainset, teacher=None, valset=None):
        self.trainset = trainset
        self.valset = trainset if valset is None else valset

        self.student = student.deepcopy()
        self.teacher = teacher.deepcopy() if teacher else student.deepcopy()

        # Leverage BootstrapFewshot to create a large number of potential demonstrations.
        tp = BootstrapFewShot()
        self.pool = tp.compile(self.student, self.teacher, self.trainset, self.metric)

        # Use Optuna to find the best program by optimizing the objective function.
        best_program = optimize_with_optuna(self.objective)

        print(’Best score:’, best_program.score)
        print(’Best program:’, best_program)
        return best_program
```

`compile`은 먼저 BootstrapFewShot으로 데모를 넉넉히 모아 풀을 만듭니다. 풀 안에서 각 예측기의 데모 목록이 그 예측기의 후보 집합 $$\mathcal{D}_p$$입니다. 이후 Optuna가 목적 함수를 반복 호출합니다. 목적 함수는 학생의 빈 사본을 만들고, 예측기마다 `trial.suggest_int`로 풀 안의 데모 인덱스 하나를 고릅니다. 이 인덱스가 앞선 목표식의 $$d_p$$입니다. 인덱스 범위는 예측기별 풀 길이에 따라 실행 중에 정해지는데, define-by-run 방식이 바로 이런 경우에 맞습니다. 의사코드는 단순화를 위해 예측기마다 데모를 하나만 고르지만, 주석에 따르면 고를 데모 수도 쉽게 튜닝 대상으로 만들 수 있습니다. 각 후보의 검증 점수가 목적 함수 값이 되고, 후보 프로그램은 시도에 저장되어 최고 점수의 것이 반환됩니다.

두 탐색기의 차이는 다음과 같습니다. 무작위 탐색은 매 시도마다 부트스트래핑을 새로 돌려 데모 집합 전체를 독립적으로 비교합니다. Optuna 버전은 부트스트래핑을 한 번만 하고, 그 풀 안에서 예측기별 선택을 탐색합니다. 기본 샘플러인 TPE가 이전 시도 결과를 활용하므로, 이 선택은 앞선 점수를 반영해 이루어집니다. 두 의사코드의 기본 시도 횟수는 모두 16입니다.

#### 데모로 가중치를 바꾸는 BootstrapFinetune

2단계의 또 다른 옵티마이저는 BootstrapFinetune입니다. 이 옵티마이저는 후보 가운데 고르지 않고, 부트스트래핑한 데모로 각 예측기 LM의 가중치를 업데이트합니다. 적용이 끝나면 각 모듈의 LM 파라미터가 파인튜닝된 새 모델로 바뀝니다. 앞서 본 데모의 이중 용도가 여기서 실제 코드 경로가 됩니다. 1단계의 트레이스 수집은 같고, 2단계에서 그 결과를 고를 후보로 쓰는지 학습 데이터로 쓰는지만 다릅니다.

### 3단계: 제어 흐름을 바꾸는 고차 프로그램 최적화

DSPy 컴파일러는 프로그램의 제어 흐름 자체를 바꾸는 최적화도 지원합니다. 1·2단계가 모듈 안의 파라미터를 채운다면, 3단계는 모듈들을 부르는 방식을 바꿉니다. 가장 단순한 형태가 앙상블이며, 사례 연구에서도 이를 씁니다.

앙상블은 같은 프로그램의 사본 여러 개를 각각 부트스트래핑하므로, 사본마다 데모가 다를 수 있습니다. 그런 다음 원래 프로그램을 새 프로그램으로 바꿉니다. 새 프로그램은 사본들을 병렬로 실행하고, 사용자 정의 함수로 예측들을 하나로 줄입니다. 대표적인 축약 함수는 다수결 투표입니다. 예를 들어 사본 다섯 개 가운데 셋이 "Paris", 둘이 "Lyon"을 내면 최종 답은 "Paris"입니다. 자기 일관성은 한 프롬프트에서 추론 경로를 여러 개 샘플링해 투표합니다. 앙상블에서는 따로 컴파일된 **프로그램들**이 투표합니다. 저자들이 교사 프로그램 합성을 설명하며 든 예처럼, 대형 LM을 쓰는 비싼 앙상블이 싼 프로그램을 감독하는 교사가 될 수도 있습니다.

저자들은 앞으로 이 단계에 더 동적인 기법을 쉽게 담을 수 있다고 봅니다. 테스트 시점에 부트스트래핑하는 방식이나 자동 백트래킹 같은 논리가 그 예입니다.

## GSM8K 사례 연구: 범용 모듈 조합으로 수학 문장제 풀기

### 평가 가설: 수작업 프롬프트는 정말 필요한가

프로그래밍 프레임워크는 계산 효율, 개발 생산성, 코드와 개념의 직관성 등 여러 축으로 평가할 수 있습니다. 저자들은 이런 축은 제쳐 두고, 현재 LM 파이프라인에서 가장 시급하다고 보는 문제에 집중합니다. 성능 좋은 시스템을 만드는 데 과제별로 손으로 쓴 프롬프트가 얼마나 필요한지입니다. 검증할 가설은 셋입니다.

- **H1**: DSPy에서는 손으로 다듬은 프롬프트 문자열을 간결하고 잘 정의된 모듈로 바꿀 수 있고, 품질이나 표현력은 줄지 않습니다.
- **H2**: 모듈을 파라미터화하고 프롬프팅을 최적화 문제로 다루면 DSPy가 여러 LM에 더 잘 적응하며, 전문가가 쓴 프롬프트를 능가할 수도 있습니다.
- **H3**: 이렇게 얻은 모듈성 덕분에, 쓸모 있는 성능 특성을 보이거나 미묘한 평가 지표에 맞는 복잡한 파이프라인을 더 철저히 탐색할 수 있습니다.

저자들은 질문의 형태도 바꾸자고 제안합니다. "GSM8K에서 여러 LM을 비교하면 어떤가"는 조건이 덜 정해진 질문입니다. "프로그램 P를 전략 S로 컴파일했을 때 GSM8K에서 어떤가"는 잘 정의되어 재현할 수 있는 실행입니다. 아래 결과표의 각 행이 바로 이런 (프로그램, 컴파일 전략, 학습 데이터) 조합입니다.

### 실험 설정: 200개 학습, 300개 개발, 1.3k 테스트

GSM8K는 [Cobbe et al.](https://arxiv.org/pdf/2110.14168)이 공개한 초등학교 수준의 수학 문장제 데이터셋으로, 여러 단계의 산술 추론이 필요한 문제들로 구성됩니다. 저자들은 공식 학습 분할에서 질문–답 쌍 200개를 학습용, 300개를 개발(dev)용으로 뽑았습니다. 최종 평가에는 공식 테스트셋 1.3k개를 씁니다. 테스트셋에 과적합하지 않도록 폭넓은 비교는 dev에서 하고, 각 접근법에서 유망한 대표 설정만 테스트셋으로 평가했습니다. 그래서 결과표의 테스트 칸 상당수가 비어 있습니다. 평가 지표는 기존 GSM8K 연구를 따라, LM 출력에 나타난 **최종 수치**가 정답과 같은지 보는 정확도입니다. 이 지표는 최종 숫자만 봅니다.

### 세 프로그램: vanilla, CoT, reflection

비교하는 프로그램은 셋이고, 코드는 다음이 전부입니다.

```
vanilla = dspy.Predict("question -> answer")  # GSM8K Program 'vanilla'

CoT = dspy.ChainOfThought("question -> answer")  # GSM8K Program 'CoT'
```

```
class ThoughtReflection(dspy.Module):
    def __init__(self, num_attempts):
        self.predict = dspy.ChainOfThought("question -> answer", n=num_attempts)
        self.compare = dspy.MultiChainComparison('question -> answer', M=num_attempts)

    def forward(self, question):
        completions = self.predict(question=question).completions
        return self.compare(question=question, completions=completions)

reflection = ThoughtReflection(num_attempts=5) # GSM8K Program 'reflection'
```

vanilla는 질문에서 바로 답을 내는 1단계 Predict입니다. CoT는 추론 근거를 먼저 생성하는 2단계 ChainOfThought입니다. reflection은 ChainOfThought로 추론 체인 5개를 답과 함께 샘플링하고, 내장 모듈 MultiChainComparison이 다섯 시도를 나란히 비교해 그 패턴을 반영한 새 답을 냅니다. 세 프로그램 모두 범용 모듈만 씁니다. 수학 문제 전용 모듈이나 특정 LM 전용 모듈은 하나도 없습니다.

### 반복해서 쓰이는 컴파일 설정

결과표의 "컴파일" 열은 다음 설정들을 뜻합니다.

- **none**: 컴파일하지 않은 제로 샷입니다. 시그니처에서 나온 지시문만으로 실행합니다.
- **fewshot**: 가장 단순한 컴파일러인 LabeledFewShot을 씁니다. `dspy.LabeledFewShot(k=8).compile(program, trainset=trainset)`는 학습 집합에서 데모 $$k=8$$개를 무작위로 뽑는데, 학습 예시와 시그니처에 공통인 필드, 즉 question과 answer만 담습니다. 추론 근거는 없습니다. 무작위 샘플링이라 설정에 따라 3–5회 실행의 평균을 보고합니다.
- **bootstrap**: BootstrapFewShotWithRandomSearch를 GSM8K 정확도를 평가 지표로 삼아 실행하고, devset을 검증 집합(`valset`)으로 씁니다. 학습 예시에 대한 데모 체인을 부트스트래핑하고, 그 가운데 어떤 데모를 쓸지를 무작위 탐색으로 고릅니다.
- **bootstrap×2**: 부트스트래핑을 중첩합니다. `tp.compile(program, teacher=bootstrap, ...)`처럼, 이미 최적화된 bootstrap 프로그램을 교사로 삼아 다시 부트스트래핑합니다. 원래 제로 샷 프로그램이 약할 때 특히 쓸모 있습니다.
- **+ensemble**: 부트스트래핑 컴파일러가 만든 후보 프로그램 가운데 상위 7개를 다수결 투표로 앙상블합니다(`Ensemble(reduce_fn=dspy.majority).compile(bootstrap.programs[:7])`). 결과표에서 이 행은 **바로 위 행**의 컴파일 결과 위에 쌓입니다.
- **+human_CoT**: "학습 데이터" 열의 변형입니다. 기본 trainset에는 GSM8K가 제공하는 사람의 추론 체인이 없고, +human_CoT는 각 예시에 그 추론 문자열을 덧붙인 학습 집합입니다. 두 집합은 어느 컴파일러에든 바꿔 넣을 수 있습니다.

비용도 크지 않습니다. 컴파일은 보통 몇 분에서 수십 분 걸립니다. 비싼 설정도 프로그램을 수천 번(예: 검증 예시 150–300개에 대해 10–20회 시도) 실행하는 정도이고, 이 실행은 병렬로 돌릴 수 있습니다.

### 결과표 읽기

각 행은 별개의 파이프라인입니다. 프로그램 열의 모듈을 학습 데이터 열의 예시로 컴파일했다는 뜻입니다. 정확도 단위는 %입니다.

| 프로그램 | 컴파일 | 학습 데이터 | GPT-3.5 Dev | GPT-3.5 Test | llama2-13b-chat Dev | llama2-13b-chat Test |
| --- | --- | --- | --- | --- | --- | --- |
| vanilla | none | n/a | 24.0 | 25.2 | 7.0 | 9.4 |
| vanilla | fewshot | trainset | 33.1 | – | 4.3 | – |
| vanilla | bootstrap | trainset | 44.0 | – | 28.0 | – |
| vanilla | bootstrap×2 | trainset | 64.7 | 61.7 | 37.3 | 36.5 |
| vanilla | +ensemble | trainset | 62.7 | 61.9 | 39.0 | 34.6 |
| CoT | none | n/a | 50.0 | – | 26.7 | – |
| CoT | fewshot | trainset | 63.0 | – | 27.3 | – |
| CoT | fewshot | +human_CoT | 78.6 | 72.4 | 34.3 | 33.7 |
| CoT | bootstrap | trainset | 80.3 | 72.9 | 43.3 | – |
| CoT | +ensemble | trainset | 88.3 | 81.6 | 43.7 | – |
| reflection | none | n/a | 65.0 | – | 36.7 | – |
| reflection | fewshot | trainset | 71.7 | – | 36.3 | – |
| reflection | bootstrap | trainset | 83.0 | 76.0 | 44.3 | 40.2 |
| reflection | +ensemble | trainset | 86.7 | – | 49.0 | 46.9 |

표는 세 블록으로 나눠 읽으면 쉽습니다. 블록마다 프로그램은 고정하고 컴파일 전략만 바꿉니다.

#### vanilla: 답 필드 안에서 추론을 배우는 현상

추론 없이 답을 바로 내야 하는 vanilla는 두 LM 모두 약합니다. 좋은 데모가 없을 때 특히 그렇습니다. none에서 GPT-3.5는 dev 24.0, llama2-13b-chat은 dev 7.0 / test 9.4에 그칩니다. 무작위 질문–답 쌍 8개를 붙이는 fewshot은 GPT-3.5를 33.1로 올리지만, llama2-13b-chat은 4.3으로 오히려 떨어집니다.

그런데 bootstrap을 쓰면 dev가 44.0 / 28.0으로 오르고, bootstrap×2에서는 GPT-3.5가 64.7 dev / 61.7 test, llama2-13b-chat이 37.3 / 36.5에 이릅니다. llama2-13b-chat dev 기준으로는 none의 7.0에서 37.3으로 올랐고, 두 값으로 계산하면 약 5.3배입니다. 프로그램은 여전히 `question -> answer` 한 줄짜리 Predict입니다. 차이는 컴파일된 프롬프트에서 드러납니다. 다음은 llama2-13b-chat용 vanilla를 bootstrap×2로 컴파일해 DSPy가 자동 생성한 프롬프트의 데모 한 개입니다.

```
Question: Jimmy and Irene go shopping for clothes on a Tuesday, where senior citizens get a 10% discount on their purchases. Jimmy picks out 3 shorts from the $15 rack. Irene grabs 5 shirts from the $17 rack. How much money do they give to the cashier?
Answer: Jimmy picks out 3 shorts at $15 each = $45. Irene grabs 5 shirts at $17 each = $85. Total cost = $45 + $85 = $130. Since senior citizens get a 10% discount, they will pay 10% of $130 = $13. So, they will give the cashier $130 - $13 = $117.
```

시그니처에는 Reasoning 필드가 없지만, 데모의 `Answer:` 필드 자체가 단계별 계산으로 채워져 있습니다. 저자들의 설명은 이렇습니다. 부트스트래핑된 프롬프트가 LM이 answer 필드를 먼저 추론에 쓰도록 허용하고, 평가 지표가 출력에서 최종 수치만 뽑아 비교하므로 이런 출력도 정답으로 인정됩니다. 그래서 이런 실행이 지표를 통과해 데모로 남을 수 있고, 데모를 본 LM은 answer 칸에서 먼저 풀이를 씁니다. bootstrap×2에서는 bootstrap 프로그램이 교사가 됩니다.

사람은 vanilla에 체인 오브 소트를 가르친 적이 없습니다. 평가 지표를 통과한 실행만 데모로 남기자, 출력 형식 안에서 추론하는 데모가 자동으로 들어왔습니다. 프로그램 구조는 그대로인데 행동이 바뀌었고, 이 행동을 정한 것은 평가 지표와 데이터였습니다.

vanilla에 +ensemble을 더하면 GPT-3.5 test는 61.9로 비슷하고, llama2-13b-chat은 dev 39.0으로 오르지만 test는 34.6으로 조금 내려갑니다.

#### CoT: 사람 추론 체인 없이 맞먹는 성능

CoT 블록은 H2를 직접 시험합니다. GSM8K에는 전문가가 쓴 풀이 과정이 있습니다. 이 사람 추론 체인을 데모에 넣은 fewshot + +human_CoT 설정은 GPT-3.5에서 78.6 dev / 72.4 test로, 추론 없는 fewshot의 63.0 dev보다 크게 오릅니다. 손으로 쓴 체인 오브 소트 예시의 효과를 보여 주는 기준선입니다.

bootstrap은 사람 추론 체인을 **전혀 쓰지 않고**(학습 데이터는 질문과 최종 답만 있는 trainset) 80.3 dev / 72.9 test를 냅니다. dev와 test 모두에서 사람 체인을 쓴 설정과 맞먹거나 조금 앞섭니다. 저자들은 이 결과가 DSPy로 수작업 프롬프트의 필요를 줄일 수 있다는 가설을 뒷받침한다고 봅니다. 데모에 들어간 추론 근거는 LM이 스스로 생성했고, 최종 답을 맞힌 것만 남았습니다. llama2-13b-chat에서는 차이가 더 뚜렷합니다. bootstrap이 43.3 dev로, +human_CoT의 34.3 dev보다 9.0%p 높습니다. 논문은 이 차이의 원인을 분석하지 않습니다.

여기에 +ensemble을 더하면 GPT-3.5가 88.3 dev / 81.6 test로 표 전체에서 가장 높습니다. 반면 llama2-13b-chat은 43.3에서 43.7 dev로 거의 변하지 않습니다.

#### reflection: 몇 줄 더 긴 프로그램의 우위

reflection은 다른 프로그램보다 몇 줄 길 뿐인데, 저자들이 명확한 승자로 꼽습니다. 컴파일 없이도 none에서 GPT-3.5 65.0, llama2-13b-chat 36.7 dev로, CoT의 none(50.0 / 26.7)보다 높게 출발합니다. bootstrap을 거치면 GPT-3.5 83.0 dev / 76.0 test, llama2-13b-chat 44.3 dev / 40.2 test로, 같은 설정의 CoT를 두 LM 모두에서 dev 기준으로 앞섭니다. +ensemble은 llama2-13b-chat을 49.0 dev / 46.9 test로 끌어올려 이 모델의 최고 기록을 냅니다. GPT-3.5에서는 +ensemble이 86.7 dev로, 앙상블한 CoT의 88.3보다는 낮습니다. 저자들이 CoT도 앙상블과 함께라면 꽤 효과적이라고 덧붙이는 이유입니다.

표 전체를 보면 bootstrap 컴파일은 모든 프로그램, 두 LM 모두에서 큰 향상을 가져옵니다. 표의 모든 프로그램은 DSPy 모듈과 텔레프롬프터 2–4개를 조합한 것입니다. 저자들의 요약은 이렇습니다. 문자열 프롬프트를 조작하는 대신 알맞은 범용 모듈을 조합하고 컴파일하면, LM에 따라 4–20%이던 정확도가 49–88%로 오릅니다.

### 다른 보고치와의 비공식 비교

저자들은 문헌의 GSM8K 수치와도 비교하는데, 모델·프롬프트·평가 조건이 다르므로 스스로 **비공식(informal)** 비교라고 밝힙니다.

| 출처 | 모델 | 방법 | GSM8K 정확도 |
| --- | --- | --- | --- |
| Auto-CoT (Zhang et al.) | text-davinci-002 | — | 48% |
| Auto-CoT (Zhang et al.) | codex | 수동 CoT / 자동 CoT | 59.4% / 62.8% |
| Wang et al. | PaLM 540B | CoT / +자기 일관성 | 57% / 74% |
| Llama2 저자들 | llama2-13b / 34b / 70b | — | 28.7% / 42.2% / 56.8% |
| Zhao et al. | gpt-3.5-turbo (2023년 4월) | CoT | 80.8% |
| GPT-4 저자들 | GPT-3.5 / GPT-4 | — | 57.1% / 92% |

저자들은 text-davinci-002의 48%가 자신들의 llama2-13b-chat 결과와 가깝다고 말합니다. 가장 흥미로운 비교는 Llama2 계열입니다. Llama2 저자들이 보고한 13b 결과는 28.7%, 34b는 42.2%입니다. 논문은 13b 변형으로 만든 DSPy 프로그램이 사람 추론 체인을 쓰지 않고도 34b 기반 결과와 견줄 만하다고 말합니다. Llama2 저자들의 보고치와 평가 조건이 같지 않을 수 있다는 점은 감안해야 합니다. GPT-4의 92%는 가장 높지만, GPT-4 저자들 스스로 GPT-4가 사전 학습 중에 GSM8K 학습셋 일부를 보았다고 밝혔습니다.

### 컴파일된 프롬프트의 실제 모습

컴파일된 프롬프트의 구조는 단순합니다. 맨 위에 시그니처에서 만든 범용 헤더 "Given the fields `question`, produce the fields `answer`."가 있고, 이어서 "Follow the following format." 아래에 `Question: ${question}`, `Answer: ${answer}` 같은 형식 명세가 옵니다. 그 뒤에 데모가 붙습니다. CoT 프로그램이라면 형식 명세에 "Reasoning: Let's think step by step in order to ..." 줄이 끼어들고, 데모마다 추론 근거와 답이 함께 들어갑니다. 예를 들어 빵 반죽을 120분씩 두 번 발효하는 문제의 데모에서는 추론 근거가 240 + 10 + 30 = 280을 계산한 뒤 `Answer: 280`을 냅니다. 수학이나 GSM8K를 위해 손으로 쓴 문장은 하나도 없습니다. 데모의 라벨(추론 근거와 답)과 어떤 데모를 넣을지의 선택은 모두 DSPy가 llama2-13b-chat을 써서 자동으로 만들었습니다.

## HotPotQA 사례 연구: 멀티홉 검색 질의응답, 에이전트 루프, 소형 모델 파인튜닝

### 질문 하나로는 근거를 다 찾을 수 없는 과제

[HotPotQA](https://arxiv.org/pdf/1809.09600)는 위키피디아 기반 멀티홉 질의응답(multi-hop question answering) 데이터셋입니다. 질문 하나에 답하려면 여러 문서의 근거를 엮어야 합니다. "어떤 다큐멘터리 시리즈를 쓴 작가는 몇 년에 태어났나"라는 질문을 예로 들면, 먼저 시리즈 문서에서 작가 이름을 찾고, 그 이름으로 작가 문서를 다시 찾아야 풀립니다. 두 번째 문서를 찾는 데 필요한 단어인 작가 이름은 원래 질문에 없습니다. 원 데이터셋의 설정은 둘입니다. distractor 설정은 후보 문단을 함께 주고, fullwiki(개방 도메인) 설정은 위키피디아 전체에서 근거를 직접 검색하게 합니다. DSPy 논문은 fullwiki 설정을 씁니다.

검색 대상은 HotPotQA가 공식 배포한 2017년 위키피디아 "abstracts" 덤프로 만든 색인이고, 검색기는 ColBERTv2입니다. HotPotQA 테스트셋은 공개되지 않습니다. 그래서 저자들은 공식 검증 집합을 테스트용으로 남겨 두고, 그 가운데 1000개를 샘플링했습니다. 공식 학습 분할은 70%/30%로 다시 나눠 학습용과 개발용으로 삼았습니다. 공식 검증·테스트셋이 "hard" 예시로 이루어져 있으므로, 학습 쪽에서도 원 데이터셋이 "hard"로 표시한 예시만 남겼습니다. 여기서 학습용 200개와 개발(dev) 보고용 300개를 샘플링했습니다. GSM8K 때와 달리 교차 검증에는 dev가 아니라 학습 집합 자체를 씁니다.

평가 지표는 둘입니다. **Ans**는 답의 EM으로, 정규화한 예측 답이 정답과 똑같은 예시의 비율입니다. **Psg**는 패시지 쌍 검색 정확도로, 정답에 필요한 금(gold) 패시지 두 개가 **모두** 검색 결과에 들어간 예시의 비율입니다. Psg는 검색 단계만 따로 보므로, 답이 틀렸을 때 원인이 검색인지 읽기인지 가려 볼 수 있습니다.

### 비교하는 네 프로그램

**vanilla**는 GSM8K에서 쓴 `question -> answer` Predict를 그대로 가져옵니다. 저자들은 이 시그니처가 충분히 범용적이어서, 적절히 컴파일하면 이 과제와 다른 많은 과제에서도 동작한다고 말합니다. 검색 단계가 없으므로 LM은 자기가 기억하는 지식만으로 답합니다.

**CoT_RAG**는 앞서 본 RAG 프로그램입니다. 원래 질문으로 패시지를 검색하고 ChainOfThought로 답을 냅니다. 저자들은 이 프로그램이 HotPotQA에서 그다지 뛰어나지 않다는 점을 두 멀티홉 프로그램을 시험하는 동기로 듭니다.

**react**는 내장 모듈 ReAct로 만든 다단계 도구 사용 에이전트이고, 선언은 한 줄입니다.

```
react = dspy.ReAct("question -> answer", tools=[dspy.Retrieve(k=1)], max_iters=5)
```

LM은 생각과 행동을 번갈아 내는 에이전트 루프(agent loop)를 돕니다. 행동으로 검색 도구를 부르면 패시지 1개(`k=1`)가 관찰로 돌아옵니다. 루프는 최대 5회까지 반복되고, 그 뒤 LM이 답을 냅니다. 어떤 질의로 몇 번 검색할지는 LM이 루프 안에서 정합니다.

**multihop**은 저자들이 직접 쓴 `BasicMultiHop` 프로그램입니다.

```
class BasicMultiHop(dspy.Module):
    def __init__(self, passages_per_hop):
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = dspy.ChainOfThought("context, question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = []

        for hop in range(2):
            query = self.generate_query(context=context, question=question).search_query
            context += self.retrieve(query).passages

        return self.generate_answer(context=context, question=question)

multihop = BasicMultiHop(passages_per_hop=3)
```

식으로 쓰면 아래와 같습니다. 여기서 $$q$$는 질문, $$G$$는 검색 질의 생성 모듈, $$R(\cdot, k)$$는 패시지 $$k$$개를 돌려주는 검색기, $$A$$는 답 생성 모듈입니다. 문맥은 빈 집합 $$c_0 = \emptyset$$에서 시작합니다.

$$
q_t = G(q, c_{t-1}), \qquad c_t = c_{t-1} \cup R(q_t, 3), \qquad t = 1, 2, \qquad \hat{y} = A(q, c_2)
$$

첫 홉에서는 문맥이 비어 있어서 질문만 보고 질의를 만듭니다. 두 번째 홉에서는 첫 검색으로 얻은 패시지 3개를 보고 새 질의를 만듭니다. 그래서 "작가 이름"처럼 질문에 없던 정보를 질의에 넣을 수 있습니다. 마지막으로 누적된 패시지 6개를 바탕으로 답을 냅니다. 질의 생성과 답 생성이 모두 ChainOfThought이므로 두 단계 모두 추론 근거를 먼저 씁니다.

저자들은 이 프로그램이 Baleen과 IRRR의 정보 흐름을 모사하고 IRCoT와도 닮았다고 설명합니다. [Baleen](https://arxiv.org/pdf/2101.00436)은 홉마다 질의를 생성해 검색하고, 그 결과를 바탕으로 다음 질의를 만드는 반복 검색 시스템입니다. Baleen에는 홉 사이의 패시지를 압축 요약하는 장치와 검색기 자체를 학습하는 장치도 있습니다. `BasicMultiHop`이 따르는 것은 "질의 생성 → 검색 → 이전 문맥을 보고 다음 질의 생성"이라는 흐름까지입니다. 패시지는 요약하지 않고 그대로 이어 붙이며, 검색기도 학습하지 않습니다. [IRCoT](https://arxiv.org/pdf/2212.10509)는 체인 오브 소트 문장을 하나 생성할 때마다 지금까지의 추론을 질의로 삼아 검색합니다. 검색한 패시지는 다음 추론 문장의 입력으로 들어가고, 답이 나올 때까지 이 과정을 반복합니다. `BasicMultiHop`은 홉 수가 2로 고정되어 있고 홉마다 한 번만 검색하므로 IRCoT보다 단순합니다.

### 텔레프롬프터 합성 두 가지

컴파일 설정은 GSM8K에서 쓴 것을 그대로 쓰고, 여기에 텔레프롬프터 합성 두 가지를 더합니다.

첫째는 react의 **bootstrap×2**입니다. react를 한 번 부트스트래핑해 얻은 프로그램을 교사로 삼아 BootstrapFewShotWithRandomSearch를 다시 돌립니다. 비교 기준선 **+human_r**은 ReAct 원 논문(Yao et al.)의 전문가가 쓴 추론 데모를 이 검색 설정에 맞게 조금 고친 것입니다.

둘째는 **multihop_t5**입니다. 부트스트래핑된 multihop 프로그램을 교사로 삼고, BootstrapFinetune으로 T5-Large를 파인튜닝합니다. 평가 지표로는 답 EM을 씁니다.

```
multihop_t5 = dspy.BootstrapFinetune(metric=answer_exact_match).compile(program, teacher=bootstrap, trainset=trainset, target=’t5-large’)
```

### 결과표

각 행은 별개의 파이프라인이며 단위는 %입니다. "–"는 보고되지 않은 칸이고, 별표(∗)가 붙은 값은 비용 때문에 테스트셋의 50%로만 평가한 결과입니다. 두 LM의 결과를 표 두 개로 나눴습니다.

| 프로그램 | 컴파일 | GPT-3.5 Dev Ans | GPT-3.5 Dev Psg | GPT-3.5 Test Ans | GPT-3.5 Test Psg |
| --- | --- | --- | --- | --- | --- |
| vanilla | fewshot | 34.3 | n/a | 31.5 | n/a |
| CoT_RAG | fewshot | 36.4 | 36.0 | 29.8 | 34.4 |
| CoT_RAG | bootstrap | 42.3 | 36.0 | – | – |
| react | none | 20.3 | – | – | – |
| react | +human_r | 33.0 | – | – | – |
| react | bootstrap | 31.0 | – | – | – |
| react | bootstrap×2 | 39.0 | – | – | – |
| multihop | fewshot | 36.9 | 38.3 | 31.2 | 40.8 |
| multihop | bootstrap | 48.7 | 47.0 | 39.6 | 43.8 |
| multihop | ensemble | 54.7 | – | 45.6∗ | – |

| 프로그램 | 컴파일 | llama2-13b-chat Dev Ans | llama2-13b-chat Dev Psg | llama2-13b-chat Test Ans | llama2-13b-chat Test Psg |
| --- | --- | --- | --- | --- | --- |
| vanilla | fewshot | 27.5 | n/a | 21.8 | n/a |
| CoT_RAG | fewshot | 34.5 | 36.0 | 28.0 | 34.4 |
| CoT_RAG | bootstrap | 38.3 | 36.0 | 32.9 | 34.4 |
| react | none | 20.0 | – | – | – |
| react | +human_r | 28.3 | – | – | – |
| react | bootstrap | 24.7 | – | – | – |
| react | bootstrap×2 | 40.0 | – | – | – |
| multihop | fewshot | 34.7 | 32.0 | 31.3 | 30.8 |
| multihop | bootstrap | 42.0 | 48.3 | 36.4 | 43.5 |
| multihop | ensemble | 50.0 | – | 41.0 | – |

#### CoT_RAG: 답은 오르지만 검색은 그대로

GPT-3.5 dev에서 CoT_RAG는 스스로 부트스트래핑해 Ans를 vanilla fewshot의 34.3에서 42.3으로 올립니다. 하지만 Psg는 fewshot에서도 bootstrap에서도 36.0으로 같고, llama2-13b-chat에서도 36.0입니다. CoT_RAG는 원래 질문만으로 ColBERTv2에 검색을 맡기므로, 컴파일해도 검색 결과는 달라지지 않습니다. 논문도 이 점이 패시지 재현율을 제한한다고 지적합니다. 검색 질의를 직접 생성하며 여러 홉을 도는 react와 multihop이 바로 이 한계를 겨냥합니다.

#### react: 중첩 부트스트래핑이 전문가 데모를 넘음

GPT-3.5 dev에서 데모 없는 none은 20.3에 그쳐 vanilla fewshot(34.3)보다도 한참 낮습니다. 전문가 데모를 넣은 +human_r은 33.0, 자동 부트스트래핑을 한 번 거친 bootstrap은 31.0이고, bootstrap×2는 39.0입니다. 즉 bootstrap×2가 +human_r보다 6.0%p 높습니다(39.0 − 33.0). llama2-13b-chat에서도 순서는 같고 격차는 더 큽니다. none 20.0, +human_r 28.3, bootstrap 24.7, bootstrap×2 40.0이며, bootstrap×2가 전문가 데모보다 11.7%p 높습니다(40.0 − 28.3).

1차 부트스트래핑은 전문가 데모보다 낮았습니다. 그래서 이 결과는 쉽게 읽히지 않습니다. 한 번의 자동 컴파일로는 사람이 공들인 데모를 따라잡지 못했는데, 그 결과를 교사로 삼아 다시 컴파일하자 순위가 뒤집혔습니다. 논문은 원인을 분석하지 않습니다. react는 dev 결과만 보고되었고 Psg도 보고되지 않았습니다.

텔레프롬프터를 이어 붙이는 일이 사람의 개입을 대신합니다. 한 번의 부트스트래핑으로 부족하면 그 결과를 교사로 다시 컴파일하면 됩니다. 사람이 데모를 한 줄도 쓰지 않았는데도, 이렇게 얻은 react는 전문가가 쓴 ReAct 데모를 두 LM 모두에서 앞섰습니다.

#### multihop: 가장 단순한 멀티홉 프로그램이 최고

전체적으로는 단순한 multihop이 가장 좋은 프로그램입니다. GPT-3.5에서 fewshot은 dev 36.9 Ans / 38.3 Psg로, CoT_RAG fewshot(36.4 / 36.0)과 큰 차이가 없습니다.

bootstrap을 쓰면 dev 48.7 / 47.0, test 39.6 / 43.8로 크게 오릅니다. CoT_RAG와 달리 이번에는 Psg도 오릅니다. dev Psg가 fewshot의 38.3에서 47.0으로 8.7%p 오르고, CoT_RAG의 고정값 36.0보다 11.0%p 높습니다. 부트스트래핑이 두 홉의 질의 생성기에도 데모를 채우면서 검색 자체가 나아진 것으로 보이지만, 논문이 이 원인을 따로 분석하지는 않습니다.

ensemble은 GPT-3.5에서 dev 54.7, test 45.6으로 표에서 가장 높습니다. 다만 이 test 45.6은 비용 때문에 테스트셋 1000개 가운데 절반으로만 평가한 값이라, 다른 test 칸과 표본이 다릅니다.

llama2-13b-chat에서도 흐름은 같습니다. fewshot의 dev 34.7 / 32.0이 bootstrap에서 dev 42.0 / 48.3, test 36.4 / 43.5로 오르고, ensemble은 dev 50.0, test 41.0입니다. 같은 설정끼리 비교하면 bootstrap test Ans는 36.4 대 39.6, ensemble dev Ans는 50.0 대 54.7로 GPT-3.5가 앞섭니다. 반면 bootstrap Psg는 dev 48.3 대 47.0, test 43.5 대 43.8로 두 LM이 비슷합니다. 저자들은 가장 중요한 결과로, 프로그램을 컴파일하는 것만으로 llama2-13b-chat을 GPT-3.5와 겨룰 만한 수준으로 만들 수 있다는 점을 꼽습니다.

자동으로 만들어진 질의 생성기 프롬프트를 보면 이 동작이 잘 드러납니다. llama2-13b-chat용 multihop을 bootstrap으로 컴파일하면, 두 번째 홉 질의 생성기의 프롬프트 맨 위에 시그니처에서 나온 헤더 "Given the fields `context`, `question`, produce the fields `search_query`."가 옵니다. 그 아래에 Context, Question, Reasoning, Search Query 형식 명세가 있고, 이어서 데모가 붙습니다. 데모 하나는 첫 홉에서 검색한 패시지 3개를 문맥으로 담고 있습니다. 그중에는 빅토리아 시대 미술과 문화를 다룬 다큐멘터리 시리즈에 관한 패시지가 있습니다. 질문은 "The Victorians - Their Story In Pictures라는 다큐멘터리 시리즈를 쓴 작가는 몇 년에 태어났나"입니다. 데모의 추론은 "이 시리즈는 Jeremy Paxman이 쓰고 진행했으므로 Jeremy Paxman의 출생 연도를 찾아야 한다"이고, 출력 질의는 `Jeremy Paxman birth year`입니다. 원래 질문에 없던 인물 이름이 두 번째 질의의 핵심어가 되었습니다. 문맥, 질문, 추론, 검색 질의라는 데모의 모든 칸은 사람이 아니라 llama2-13b-chat이 만들었습니다.

### 770M T5-Large로 옮긴 multihop

DSPy의 파인튜닝 능력은 multihop_t5로 평가했습니다. 7억 7천만(770M) 파라미터 T5-Large가 dev에서 39.3% Ans, 46.0% Psg를 냈습니다. 사용한 데이터는 라벨이 있는 입력 200개와 라벨 없는 질문 800개뿐입니다.

설정에서 주의할 점이 둘 있습니다. 첫째, 실제 교사는 앞의 코드에서 보인 단일 bootstrap 프로그램이 아닙니다. llama2-13b-chat으로 돌린 multihop 프로그램 두 개의 앙상블(합집합)입니다. 둘째, 이 결과는 dev에서만 보고되었습니다.

교사와 같은 llama2-13b-chat multihop bootstrap(dev 42.0 / 48.3)과 비교하면, T5-Large는 Ans에서 2.7%p(42.0 − 39.3), Psg에서 2.3%p(48.3 − 46.0) 낮습니다. 비교 대상은 130억 파라미터 모델이라는 점도 함께 봐야 합니다. 저자들은 T5-Large가 매우 작고 로컬에서 실행할 수 있다는 점을 들어, 이 컴파일된 프로그램의 추론 비용이 GPT-3.5 같은 독점 LM보다 몇 자릿수 낮을 것이라고 주장합니다. 다만 비용을 직접 측정한 결과는 논문에 없습니다. 저자들은 이 결과를 H3의 근거로 듭니다. 모듈성 덕분에 대형 LM 파이프라인의 행동을 작고 값싼 모델로 옮기는 탐색이 가능해졌다는 것입니다.

### 다른 HotPotQA 보고치와의 비공식 비교

저자들은 최근 논문들의 HotPotQA 수치와도 견주지만, 이 분야는 평가 방법과 테스트 표본이 연구마다 크게 다르다고 밝힙니다.

| 출처 | 모델 | 방법 | HotPotQA 결과 |
| --- | --- | --- | --- |
| Si et al. | — | CoT 프롬프팅 | 25.2% EM |
| Sun et al. | PaLM-62B | 근거 패시지를 암송한 뒤 답하는 recite-and-answer | 26.5% EM |
| Wang et al. | PaLM-540B | 자기 일관성 | 33.8% EM / 44.6% F1 |
| Yao et al. | PaLM-540B / text-davinci-002 | ReAct(위키피디아 API 검색 도구) | 27.4% EM / 30.8 |
| Yao et al. | PaLM-540B | ReAct + 자기 일관성을 쓴 추가 CoT 단계 | 35.1% EM |
| Trivedi et al. | code-davinci-002 | 파이프라인(500문항 표본) | 49% |

저자들은 Yao et al.이 PaLM 결과를 35.1%로 끌어올린 방법이 여러 답을 모은다는 점에서 DSPy의 앙상블과 닮았을 수 있다고 봅니다. DSPy react의 bootstrap×2는 dev 39.0(GPT-3.5) / 40.0(llama2-13b-chat)입니다. 하지만 이 값은 DSPy 자체 dev 표본에서 나왔고 외부 보고치와 모델·검색기·표본이 모두 다르므로, 두 수치를 우열로 비교할 수는 없습니다. 외부 수치 가운데 가장 높은 것은 Trivedi et al.의 49%로, 다른 모델을 써서 500문항 표본에서 얻은 값입니다. DSPy multihop의 GPT-3.5 ensemble test 45.6도 테스트셋 절반에서 얻은 값이라, 두 수치는 표본부터 다릅니다.

## 한계와 시사점

실무에서 가장 큰 강점은 프롬프트를 고치는 단위가 바뀐다는 데 있습니다. 모듈을 하나 끼워 넣거나 LM을 바꿀 때 새 프롬프트를 쓰거나 중간 단계 라벨을 만들 필요가 없습니다. 같은 코드를 다시 컴파일하면 됩니다. 평가 지표 하나가 데모 선택 기준이 되므로, 근거 일치 같은 원하는 성질을 지표로 표현하면 그대로 프로그램 행동에 반영됩니다. 다만 같은 이유로 지표가 허용하는 지름길도 학습됩니다. vanilla가 answer 칸에서 풀이를 쓰게 된 사례가 그렇습니다. 최종 수치만 보는 지표에서는 무해했지만, 출력 형식이 엄격한 하류 시스템이라면 형식 위반이 됩니다. 지표 설계가 곧 사양 설계입니다.

평가 쪽에는 본문이 짚지 않은 편향이 있습니다. GSM8K의 bootstrap은 devset을 검증 집합으로 써서 후보를 골랐으므로, 같은 dev 점수는 선택 효과만큼 낙관적입니다. 실제로 CoT bootstrap은 dev 80.3에서 test 72.9로 떨어집니다. 따라서 test 칸이 비어 있는 행은 그만큼 신중하게 읽어야 합니다. 문헌 수치 비교가 비공식이고 일부 수치가 테스트셋 절반이나 dev로만 보고되었다는 점은 저자들도 인정합니다. 실험은 두 과제와 두 주력 LM에 한정됩니다. 계산 효율, 개발 생산성, 코드 직관성 같은 프레임워크 평가 축은 일부러 제외했습니다. 정보 추출이나 저자원 합성 데이터 생성 같은 오픈소스 사용 사례는 통제된 실험 보고를 후속 과제로 남겼습니다. "조합 가능한 텍스트 변환 그래프와 옵티마이저가 진짜 표현력"이라는 주장도 두 사례 연구가 가리키는 방향일 뿐, 입증된 결론은 아닙니다.

방법 자체도 아직 첫 버전입니다. 최적화 대상은 사실상 데모뿐이고, 지시문과 필드 설명 최적화는 원리상 지원만 합니다. 타입 출력 필드와 제약 처리는 실험적 기능입니다. 3단계 고차 최적화는 앙상블만 보였습니다. 테스트 시점 부트스트래핑과 자동 백트래킹은 향후 과제입니다. 탐색기도 강화학습, LM 피드백, 베이지안 최적화를 쓸 수 있다고만 했고 실험은 무작위 탐색에 머뭅니다. 컴파일 비용은 프로그램 실행 수천 번 수준이라, 호출당 비용이 큰 LM이나 앙상블 교사에서는 무시하기 어렵습니다. 정리하면, 최종 출력을 자동 채점할 수 있고 단계가 여러 개인 파이프라인이라면 DSPy가 맞습니다. 특히 작은 오픈 모델로 옮기려는 경우에 잘 맞습니다. 반면 지표로 표현하기 어려운 품질이 중요하거나 호출 하나로 끝나는 과제라면 이점이 줄어듭니다.

- - -

### References
* [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/pdf/2310.03714v1)
