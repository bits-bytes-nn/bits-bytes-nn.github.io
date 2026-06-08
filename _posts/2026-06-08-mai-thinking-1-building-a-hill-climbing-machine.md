---
layout: post
title: "MAI-Thinking-1: Building a Hill-Climbing Machine"
date: 2026-06-08 13:14:51
author: "The Microsoft AI Team"
categories: ["Paper Summaries", "Language-Models"]
tags: ["Mixture-of-Experts-Sparse-Architecture", "Group-Relative-Policy-Optimization-with-Adaptive-Entropy-Control", "Self-Distillation-for-Long-Horizon-Reinforcement-Learning", "Multi-Stage-Specialist-Model-Consolidation", "Dropless-Mixture-of-Experts-Routing", "Interleaved-Local-Global-Attention-with-Sliding-Window", "FP8-Mixed-Precision-Training-with-Delayed-Scaling", "Asynchronous-Distributed-Reinforcement-Learning-Infrastructure", "Verifiable-Rewards-with-Multi-Turn-Agentic-Environments", "Data-Deduplication-and-Decontamination-Pipeline"]
cover: /assets/images/language-models.jpg
use_math: true
---

## 🔍 What motivated this research?

이 연구는 단일 모델이 아니라 **모델을 지속적으로 개선할 수 있는 시스템**(저자들이 "hill-climbing machine"이라 부름)을 구축하는 것을 목표로 한다. Microsoft AI 팀은 모델 개발을 시스템 수준의 최적화 문제로 보고, 세 가지 원칙을 제시한다: (1) 능력은 **증류(distillation)가 아닌 학습으로 획득**되어야 한다(증류된 지능은 steerability와 robustness가 부족), (2) 단순하고 확장 가능한 레시피가 지속 가능하다, (3) 모든 결정은 데이터 기반 ablation으로 검증되어야 한다. 그 첫 산출물이 35B active / 1T total MoE 모델 **MAI-Thinking-1**이며, 제3자 모델 증류 없이 30T 토큰의 정제된 데이터로 처음부터 학습되었다.

## 💡 What novel solution does this research propose?

**1. 확장성 중심 사전학습 프레임워크.** 모든 아키텍처/데이터 결정을 **scaling ladder**(동일 TPP로 여러 크기 학습)와 **efficiency gain (EG)** 지표로 검증한다. EG는 후보 모델의 손실에 도달하기 위해 baseline이 필요로 하는 비용 배수로 정의된다:

$$EG = \frac{f^{-1}(L')}{C'}, \quad L=f(C)=AC^{-\alpha}+E$$

FLOPs 기준 EG와 wall-clock 기준 EGTime을 분리해, 구현 최적화 편향 없이 아키텍처 본질을 비교한다.

**2. 처음부터 시작하는 RL climb.** 추론 trace에 노출된 적 없는 체크포인트에서 출발해 수천 스텝의 **로그-선형 성능 향상**을 지속한다. GRPO에 두 가지 핵심 수정을 가하고, self-distillation으로 붕괴 후 재개를 가능하게 한다.

**3. 전문가 모델 통합.** STEM/경쟁 코딩, agentic 코딩/도구 사용, helpfulness/safety의 세 전문가 모델을 각각 RL로 학습한 뒤, SFT로 단일 모델에 증류하고 최종 경량 RL로 마무리한다(아래 흐름).

```
Mid-trained → [STEM / Agentic / Helpfulness&Safety 전문가 RL]
            → Trace Distillation SFT → Final RL → MAI-Thinking-1
```

## ⚙️ How was the proposed method implemented?

**아키텍처 (MAI-Base-1).** Decoder-only 트랜스포머에서 **고희소성 MoE 레이어와 dense FFN을 교대 배치**하고, **로컬:글로벌 어텐션을 5:1**로 섞는다. 글로벌 어텐션은 위치 인코딩 없음(NoPE), 로컬은 sliding window 512 RoPE. MoE는 512개 전문가 중 8개를 활성화하는 **LatentMoE**(공유 down-projection으로 압축 후 dispatch)를 채택하고, expert capacity 제약을 없앤 **dropless** 구현으로 미세한 인과 누출과 메모리 불균형을 회피한다. 글로벌 배치 load-balancing loss가 손실 종류보다 더 중요했다. 교대 배치는 EGTime 기준 every-layer MoE보다 우수했다.

| 항목 | 값 |
|---|---|
| 활성/전체 파라미터 | 34.7B / 962B |
| 레이어 / hidden | 78 / 6656 |
| Top-k / 전문가 | 8 / 512 |
| 사전학습 토큰 | 30T (+mid 3.55T) |
| 최대 context | 256K |

학습 안정화 기법: **어텐션 출력을 0으로 초기화**(초기 MoE 라우팅 불균형 방지), 높은 dropout 0.15, FP8(E4M3 forward) + FP32 민감 구간 혼합, stochastic rounding. 학습은 자체 프레임워크 **YOLO**(custom ZeRO 1-3, Ulysses context parallelism, dropless MoE 커널)에서 **bitwise 결정성**을 보장하며 수행. 데이터는 NLL 기반 40여 개 벤치마크로 mixture를 최적화하되, **rank invariance 가설이 깨지는 사례**(소규모에서 우수하던 mix가 대규모에서 역전)를 발견해 mixture의 scaling 특성을 강조했다.

**RL 레시피.** 토큰 레벨 GRPO 목적함수에 두 수정을 가한다:
- **Adaptive entropy control**: 상한 clip 폭 \(k\)를 목표 엔트로피 \(H^\star\)에 맞춰 integral controller로 동적 조정. \(k \leftarrow \text{clip}(k+\delta\cdot\text{sign}(H^\star-\hat H(\pi\_\theta)),0,k\_{max})\). 명시적 엔트로피 보너스보다 우수.
- **Outer ratio clip**: 모든 branch에 hard clip \([r\_{min},r\_{max}]\) 적용해 gradient-norm 폭발 억제.

보상은 \(R = R\_{task} + w\_{lang}R\_{lang} - w\_{len}R\_{len}\)로 분해(언어 일관성 + 난이도 가중 길이 패널티). **top-p mask replay**(nucleus 밖 토큰 logit을 \(-\infty\)로)와 **MoE routing replay**로 학습/추론 간 정책 발산을 억제하고, 8k→128k로 rollout 길이를 점진 확장한다. **Self-distillation**은 O(1M) trace면 충분하며, 단일 최종 체크포인트보다 여러 후기 체크포인트의 trace 다양성이 중요했다.

**Agentic/도구 환경.** 실제 GitHub PR 1.02억 건에서 시작해 LLM 에이전트로 Docker 환경을 자동 빌드, F2P/P2P 테스트로 검증, 최종 26.5만 개 환경 확보. reward hacking(인터넷 검색, git 히스토리, 테스트 변조) 대응을 위해 네트워크 격리, "time-travel" 저장소 정리, 테스트 파일 리셋을 적용. RL 인프라 **Rocket**은 controller–problem worker–rollout worker와 SGLang 추론 풀로 구성되며, 추론:학습 GPU 비율이 최대 5:1에 달한다. 학습/추론 numerics gap을 줄이려 양쪽 모두 bf16 사용.

## 📊 What are the key experimental results?

주요 출력 토큰 256k 설정에서 MAI-Thinking-1은 **AIME 2025 97.0%, AIME 2026 94.5%, LiveCodeBench v6 87.7%, SWE-Bench Pro 52.8%**를 기록한다. 동일 활성 파라미터 base 모델 대비 4개 held-out 과제 모두에서 더 낮은 bits-per-byte를 달성했다(DeepSeek-V4-Pro는 1.4배 활성 파라미터로 우위). 인간 side-by-side 평가에서 **Sonnet 4.6 대비 선호(0.07)**, Opus 4.6에는 소폭 열세였으며 특히 간결성·스타일에서 강점을 보였다. 사전학습은 8K GB200에서 **goodput 90.0%**, MFU 20% 이상을 유지했다. SWE 학습이 bash/string-replace 도구만 사용했음에도 Terminal-Bench로 일반화된 점, STEM 과제 포함이 agentic climb를 안정화한 점이 주목된다.

## 🔮 What is the significance and future direction of this research?

이 연구의 핵심 기여는 특정 모델이 아니라 **데이터 파이프라인·학습 인프라·RL 환경·평가·안전 테스트를 하나의 경험적 최적화 루프로 통합**한 방법론 그 자체다. 증류 없이 처음부터 학습해도 frontier급 추론·코딩 성능에 도달할 수 있음을 입증했고, EG·scaling ladder·self-distillation은 다른 대규모 학습에도 이식 가능하다. 한계로는 일부 벤치마크(MRCR, Terminal-Bench)에서의 약점과 저자원 언어 안전성 long-tail이 남아 있다. 저자들은 향후 더 많은 modality, 더 큰 규모, 정제된 능력으로 hill-climbing을 확장할 계획이다.
- - -

### References
* [MAI-Thinking-1: Building a Hill-Climbing Machine](https://microsoft.ai/pdf/mai-thinking-1.pdf)