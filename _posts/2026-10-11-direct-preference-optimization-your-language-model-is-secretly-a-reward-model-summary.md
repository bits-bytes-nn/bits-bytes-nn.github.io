---
layout: post
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
seo_title: "Direct Preference Optimization"
subtitle: "Direct Preference Optimization: 언어 모델은 사실 보상 모델이다"
date: 2023-05-29 17:57:46
paper_author: "Stanford University"
categories: ["Paper Summaries", "Finetuning"]
tags: ["Reward-Model-Reparameterization", "Preference-Learning", "RLHF-Alternative", "Bradley-Terry-Model", "Alignment"]
cover: /assets/images/finetuning.jpg
use_math: true
---
## TL;DR

RLHF의 보상 모델 학습과 강화학습 단계를 하나의 이진 분류 손실로 대체하는 DPO를 제안합니다. 핵심은 KL 제약이 걸린 보상 최대화 문제의 최적 정책을 닫힌 형태로 풀고, 이를 이용해 보상을 정책으로 다시 쓰는 변수 변환입니다. 최대 6B 규모 모델로 감성 제어, Reddit 요약, 단일 턴 대화 과제를 평가한 결과, PPO 기반 RLHF와 같거나 더 좋은 성능을 보였고 구현은 훨씬 단순했습니다.

## 🔍 이 연구는 왜 필요합니까?

사전 학습된 언어 모델은 사람이 만든 방대한 텍스트를 모방하기 때문에, 그 안에 섞여 있는 바람직하지 않은 행동까지 함께 배웁니다. 원하는 행동을 골라내는 표준 방법은 RLHF입니다. RLHF는 먼저 지도 미세 조정(SFT)을 하고, 사람의 쌍대 선호 데이터로 보상 모델을 학습한 뒤, PPO로 정책을 최적화합니다. 이때 정책이 참조 모델에서 너무 멀어지지 않도록 KL 페널티를 둡니다.

이 파이프라인은 여러 개의 언어 모델을 따로 학습해야 하고, 학습 루프 안에서 정책으로부터 샘플링을 반복해야 하므로 계산 비용이 큽니다. 강화학습 특유의 불안정성 때문에 하이퍼파라미터 튜닝 부담도 큽니다. 같은 목적 함수를 강화학습 없이 최적화할 방법이 있느냐가 이 논문의 출발점입니다.

## 💡 어떤 해결책을 제시합니까?

출발점은 RLHF와 같은 목적 함수입니다. 보상 $$r$$을 최대화하면서 참조 정책 $$\pi_{\text{ref}}$$와의 KL을 $$\beta$$로 제약합니다. 정책 클래스에 제한이 없다고 가정하면 이 문제의 최적해는 닫힌 형태로 구할 수 있습니다.

$$
\pi_r(y\mid x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

여기서 $$Z(x)$$는 가능한 모든 응답에 대해 합을 구하는 분배 함수이므로 직접 계산할 수 없습니다. 논문의 착안은 이 식을 거꾸로 푸는 데 있습니다. 양변에 로그를 취하면 보상을 정책으로 표현할 수 있습니다.

$$
r(x,y)=\beta\log\frac{\pi_r(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)
$$

Bradley-Terry 선호 모델은 두 응답의 보상 **차이**만 사용합니다. 따라서 위 식을 대입하면 프롬프트에만 의존하는 $$\beta\log Z(x)$$ 항이 소거됩니다. 그 결과 사람의 선호 확률이 정책과 참조 정책만으로 표현되고, 이 확률에 대한 최대우도 추정이 곧 DPO 손실이 됩니다.

$$
\mathcal{L}_{\text{DPO}}=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]
$$

$$y_w$$는 선호된 응답, $$y_l$$은 선호되지 않은 응답, $$\sigma$$는 로지스틱 함수입니다. 형태는 보상 모델 학습에 쓰이던 손실과 같습니다. 다만 보상 자리에 암묵적 보상 $$\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$$가 들어갑니다. 즉 정책 네트워크 하나가 언어 모델 역할과 보상 모델 역할을 동시에 맡습니다.

> **핵심:** 보상 모델을 이 파라미터화로 제한해도 표현력은 줄지 않습니다. 프롬프트에만 의존하는 항 $$f(x)$$만큼 차이 나는 보상들은 같은 선호 분포와 같은 최적 정책을 만듭니다. 논문은 이런 보상의 동치류마다 위 형태로 표현되는 보상이 정확히 하나 존재함을 증명합니다. DPO는 그 동치류 안에서 분배 함수가 1이 되는 대표 보상을 고르는 셈입니다.

아래 그림은 두 파이프라인을 비교합니다. RLHF는 보상 모델을 학습한 뒤 샘플링과 강화학습 루프를 돌지만, DPO는 선호 데이터에서 최대우도 단계 하나로 최종 모델을 얻습니다.

![RLHF와 DPO 학습 파이프라인 비교 다이어그램](/assets/images/direct-preference-optimization-your-language-model-is-secretly-a-reward-model-summary-1.png)

## ⚙️ 제안한 방법은 어떻게 구현되었습니까?

DPO 손실의 그래디언트를 보면 작동 방식을 직관적으로 이해할 수 있습니다.

$$
\nabla_\theta\mathcal{L}_{\text{DPO}}=-\beta\,\mathbb{E}\Big[\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)\big(\nabla_\theta\log\pi(y_w\mid x)-\nabla_\theta\log\pi(y_l\mid x)\big)\Big]
$$

이 업데이트는 $$y_w$$의 우도를 높이고 $$y_l$$의 우도를 낮춥니다. 그 강도는 암묵적 보상이 두 응답의 순서를 얼마나 틀리게 매기고 있는지에 비례합니다. 예를 들어 이미 $$\hat r_\theta(x,y_w)$$가 $$\hat r_\theta(x,y_l)$$보다 훨씬 크면 가중치는 0에 가까워져 그 쌍은 거의 학습되지 않습니다. 반대로 순서가 뒤집혀 있으면 가중치가 1에 가까워집니다.

이 가중치를 없앤 단순한 버전, 곧 선호 응답의 확률을 올리고 비선호 응답의 확률을 무조건 내리는 Unlikelihood 방식은 모델을 망가뜨립니다. 실제로 요약 과제에서 "when when when …"만 반복하는 출력이 나왔습니다.

학습 절차는 두 단계입니다.

1. 각 프롬프트에 대해 참조 모델로 응답 쌍을 샘플링하고 사람의 선호를 라벨링하여 오프라인 데이터셋을 만듭니다. 공개 데이터셋을 재사용할 수도 있습니다.
2. 주어진 $$\pi_{\text{ref}}$$와 $$\beta$$로 DPO 손실을 최소화합니다.

$$\pi_{\text{ref}}$$로는 데이터를 생성한 SFT 모델을 쓰는 것이 원칙입니다. SFT 모델이 없으면 선호 응답 $$(x,y_w)$$에 대해 최대우도로 학습한 모델을 대신 사용합니다. 이렇게 하면 실제 데이터 생성 분포와의 분포 이동을 줄일 수 있습니다.

논문이 제공하는 PyTorch 구현은 정책과 참조 모델의 시퀀스 로그확률만 받아 `-F.logsigmoid(beta * (pi_logratios - ref_logratios))`를 계산하는 십여 줄짜리 코드입니다. 기본 설정은 $$\beta=0.1$$, 배치 크기 64, RMSprop 옵티마이저, 학습률 1e-6이며, 학습률은 150 스텝 동안 선형 워밍업합니다. TL;DR 요약 과제에서만 $$\beta=0.5$$를 사용했습니다.

이론적 분석은 PPO가 불안정한 이유도 설명합니다. PPO가 최적화하는 목적 함수에는 분배 함수의 로그, 즉 참조 정책의 소프트 가치 함수에 해당하는 정규화 항이 숨어 있습니다. 이 항이 없으면 정책 그래디언트의 분산이 커집니다. 기존 방법은 학습된 가치 함수나 사람이 쓴 응답 기반 베이스라인으로 이 항을 근사해야 했지만, DPO의 파라미터화에서는 베이스라인 자체가 필요 없습니다.

## 📊 핵심 실험 결과는 무엇입니까?

실험은 세 가지 과제로 구성됩니다. IMDb 감성 제어(GPT-2-large, 감성 분류기를 정답 보상으로 사용), Reddit TL;DR 요약(GPT-J SFT 모델), Anthropic HH 단일 턴 대화(Pythia-2.8B)입니다. 요약과 대화는 GPT-4를 평가자로 하는 승률로 측정했습니다.

### 보상-KL 효율

아래 그림은 감성 제어 과제에서 참조 정책과의 KL 대비 정답 보상의 프런티어를 보여줍니다. 총 22개 런을 하이퍼파라미터를 바꿔 가며 돌렸습니다. KL이 약 2 이상인 구간에서 DPO 곡선이 모든 기법보다 위에 있습니다. 같은 목적 함수를 최적화하는 PPO는 물론, 정답 보상에 직접 접근하는 PPO-GT보다도 효율적입니다.

![감성 제어 과제의 보상 대 KL 프런티어 산점도](/assets/images/direct-preference-optimization-your-language-model-is-secretly-a-reward-model-summary-2.png)

### 요약과 분포 이동

다음 그림은 TL;DR 요약에서 샘플링 온도별로 사람이 쓴 참조 요약 대비 승률을 나타냅니다. DPO도 온도가 높아지면 떨어지지만 PPO보다 훨씬 덜 떨어지고, PPO는 높은 온도에서 기본 GPT-J 수준까지 떨어집니다.

![TL;DR 요약의 샘플링 온도별 GPT-4 승률 곡선](/assets/images/direct-preference-optimization-your-language-model-is-secretly-a-reward-model-summary-3.png)

| 실험 | 비교 기준 | DPO | PPO |
| --- | --- | --- | --- |
| TL;DR, GPT-4 승률, 온도 0 | 테스트셋 참조 요약 | 약 61% | 57% |
| CNN/DailyMail(분포 외), 온도 0 | 정답 요약 | 0.36 | 0.26 |
| CNN/DailyMail(분포 외), 온도 0.25 | 정답 요약 | 0.31 | 0.23 |
| TL;DR 사람 평가 | PPO(온도 0) | 58% | — |

핵심 발견은 세 가지입니다.

첫째, 요약 과제에서 DPO는 PPO의 최적 온도 성능을 넘어섰습니다. $$\beta$$를 거의 튜닝하지 않았다는 점을 감안하면 의미 있는 결과입니다. 사람 평가에서도 DPO(온도 0.25)가 PPO(온도 0)보다 58% 선호되었습니다.

둘째, Reddit에서 학습한 정책을 뉴스 기사(CNN/DailyMail)에 적용해도 DPO가 PPO를 앞섰습니다. 일반화 측면에서 크게 불리하지 않다는 초기 근거입니다.

셋째, Anthropic HH 대화 과제에서 데이터셋의 선호 응답보다 나은 성능을 낸 계산 효율적인 기법은 DPO가 유일했습니다. 테스트 시점에 128개를 샘플링해야 하는 Best of 128과는 비슷하거나 더 나은 수준입니다. 공개된 PPO 모델(Pythia-6B)은 기본 Pythia-2.8B보다 나은 설정을 찾지 못해, 논문은 Best of 128을 PPO 수준의 대리 지표로 삼았습니다.

GPT-4 평가의 신뢰도는 사람 평가로 검증했습니다. 간결성을 함께 묻는 프롬프트 기준으로 GPT-4와 사람의 일치율은 67~85%였고, 이는 사람 간 일치율 65~87%와 비슷한 수준입니다.

## 🔮 이 연구의 의의와 한계는 무엇입니까?

DPO는 RLHF와 같은 KL 제약 목적 함수를 일반성을 잃지 않고 단순한 분류 손실로 바꿨다는 점에서 의미가 큽니다. 별도의 보상 모델, 학습 중 샘플링, 가치 함수가 모두 필요 없어 선호 학습의 진입 장벽이 크게 낮아집니다. 이론적 결과는 Bradley-Terry뿐 아니라 여러 응답의 순위를 다루는 Plackett-Luce 모델로도 확장됩니다.

한계도 분명합니다.

- **모델 규모:** 실험은 최대 6B 모델에 그쳤고, 수십 배 큰 최신 모델에서의 동작은 검증되지 않았습니다.
- **평가 방식:** 평가가 GPT-4 승률에 크게 의존하는데, 결과가 프롬프트에 따라 달라집니다. 단순 프롬프트를 쓰면 GPT-4는 사람보다 길고 반복적인 요약을 선호했습니다. 대화 예시에서는 DPO가 장황하면서 사실과 다른 내용을 생성하기도 했습니다.
- **분포 외 일반화:** PPO는 라벨 없는 프롬프트를 추가로 활용할 수 있지만 오프라인 DPO는 그렇지 않습니다. 명시적 보상 모델 대비 분포 외 일반화가 어떤지는 아직 열린 문제입니다.
- **보상 과최적화:** 대화 과제에서 학습 후반에 승률이 약간 떨어지는 현상이 보상 과최적화인지도 논문은 미해결 질문으로 남겼습니다.
- **기법 비교:** 대화 과제의 PPO 비교는 제대로 동작하는 PPO 모델을 확보하지 못해 Best of 128로 대체되었습니다. 따라서 직접 비교라고 보기는 어렵습니다.

- - -

### References
* [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290v3)
* [GPT-J TL;DR SFT 모델](https://huggingface.co/CarperAI/openai_summarize_tldr_sft)
* [Anthropic HH PPO 모델 (Pythia-6B)](https://huggingface.co/reciprocate/ppo_hh_pythia-6B)
* [TRLX HH 예제](https://github.com/CarperAI/trlx/tree/main/examples/hh)
