---
layout: post
title: "Billion-scale similarity search with GPUs"
date: 2017-02-28 10:42:31
author: "Facebook AI Research"
categories: ["Paper Reviews", "Training-&amp;-Inference-Optimization"]
tags: ["GPU-k-Selection-Algorithm", "WarpSelect", "Billion-Scale-Similarity-Search", "Product-Quantization-on-GPU", "IVFADC-GPU-Implementation", "Fused-Kernel-Design-for-k-NN", "In-Register-Sorting-Networks", "Odd-Size-Merging-Networks", "Multi-GPU-Index-Sharding-and-Replication", "Register-File-Based-Data-Structures"]
cover: /assets/images/default.jpg
use_math: true
---
### TL;DR
#### 이 연구를 시작하게 된 배경과 동기는 무엇입니까?

현대의 머신러닝과 딥러닝 기술이 발전하면서 이미지, 비디오, 텍스트 등 복잡한 실세계 데이터를 고차원 벡터로 표현하는 것이 일반화되었습니다. Word2vec, CNN 기반 이미지 표현, 인스턴스 검색용 이미지 디스크립터 등은 모두 50차원에서 1,000차원 이상의 고차원 실수 벡터 형태로 생성되며, 이러한 벡터 임베딩은 대부분 GPU 시스템에서만 효과적으로 생성할 수 있습니다. 그러나 벡터가 생성된 이후 이를 조작하고 유사도 검색을 수행하는 과정은 여전히 연산 집약적이며, 특히 십억 규모의 벡터 데이터에서 최근접 이웃을 찾는 작업은 기존 CPU 기반 방법으로는 실용적이지 않습니다. 차원의 저주로 인해 전수 탐색이나 정확한 인덱싱이 모두 비실용적이므로, 근사 탐색과 그래프 구축에 관한 연구가 활발하지만, 기존의 NN-Descent 같은 최신 방법들은 데이터셋 자체 외에 큰 메모리 오버헤드를 수반하여 십억 규모 데이터베이스로 확장하기 어렵습니다.

GPU는 테라플롭스 수준의 산술 처리량과 수백 기가바이트/초의 메모리 대역폭을 제공하지만, 이러한 성능에 근접하는 알고리즘을 구현하는 것은 복잡하고 직관에 반하는 작업입니다. 기존 GPU 기반 유사도 검색 구현은 대부분 이진 코드, 소규모 데이터셋, 또는 전수 탐색에 한정되며, 양자화 코드를 사용한 십억 규모 데이터셋에 적합한 GPU 구현은 매우 드문 상황입니다. 이 논문은 GPU의 빠른 레지스터 메모리에서 동작하는 효율적인 $k$-선택 알고리즘과 Product Quantization 기반 인덱싱의 최적화된 GPU 구현을 통해, 십억 규모의 벡터 데이터에 대한 유사도 검색을 현실적인 시간 내에 수행할 수 있는 방법을 제시하고자 합니다.

#### 이 연구에서 제시하는 새로운 해결 방법은 무엇입니까?

이 연구의 핵심 기여는 세 가지 주요 혁신으로 구성됩니다. 첫째, **WarpSelect**라는 GPU $k$-선택 알고리즘으로, 모든 상태을 레지스터 파일에 유지하면서 단일 패스로 동작하며 워프 간 동기화를 완전히 회피합니다. 이 알고리즘은 레인 스트라이드 레지스터 배열 위에 홀수 크기 병합 네트워크를 구축하여, 임의 길이의 배열에 대해 효율적인 정렬과 병합을 수행합니다. WarpSelect는 이론적 최대 성능의 55%에 도달하며 기존 GPU 기반 최근접 이웃 탐색 대비 8.5배 빠른 속도를 달성합니다. 둘째, 정확 탐색에서 **GEMM 기반 거리 분해와 융합 $k$-선택**을 통해 거리 행렬에 대한 2회 패스만으로 처리하며, 이는 기존 3회 이상의 패스가 필요한 구현보다 훨씬 효율적입니다. 셋째, IVFADC 인덱싱에서 **3-항 분해를 통한 룩업 테이블 최적화**로 쿼리와 무관한 항을 사전 계산하여 실제 리스트 스캔 시 계산량을 대폭 감소시킵니다.

이러한 알고리즘적 혁신은 GPU 아키텍처의 특성을 깊이 있게 이해한 결과입니다. GPU의 레지스터 파일은 공유 메모리보다 훨씬 큰 저장 공간(Pascal P100에서 14 MB)과 극도로 높은 대역폭(250배 이상)을 제공하므로, 구조화된 데이터를 레지스터에 유지하는 것이 성능 향상의 핵심입니다. 워프 셔플 명령어와 버터플라이 순열을 활용한 워프 수준 병렬성은 데이터 의존적 제어 흐름을 최소화하여 워프 발산을 회피합니다. 또한 루프라인 성능 모델에 따르면, 메모리 대역폭이 제한 요인인 상황에서 입력을 한 번 스캔하는 비용보다 빠르게 수행될 수 없으므로, WarpSelect의 단일 패스 설계는 이론적 하한에 근접하는 성능을 달성할 수 있습니다.

#### 제안된 방법은 어떻게 구현되었습니까?

구현은 정확 탐색, IVFADC 근사 탐색, 그리고 다중 GPU 병렬화의 세 계층으로 구성됩니다. 정확 탐색에서는 L2 거리 계산을 $\|x - y\|_2^2 = \|x\|^2 + \|y\|^2 - 2\langle x, y \rangle$로 분해하여, 행렬 곱셈 $XY^\top$을 cuBLAS의 최적화된 GEMM 루틴으로 처리하고, 거리 행렬의 각 항에 $\|y_i\|^2$을 더하면서 즉시 $k$-선택에 제출하는 융합 커널을 사용합니다. 이를 통해 거리 행렬에 대한 2회 패스(GEMM 쓰기 1회, $k$-선택 읽기 1회)만으로 처리하며, 타일링을 통해 GPU 메모리 제약을 극복합니다.

IVFADC 근사 탐색에서는 Product Quantization의 효율성을 극대화하기 위해 거리 계산을 세 항으로 분해합니다. 첫 번째 항은 양자화기로부터 사전 계산 가능하며 크기 $|\mathcal{C}_1| \times 256 \times b$인 테이블에 저장되고, 두 번째 항은 조대 양자화기의 부산물이며, 세 번째 항은 역색인 리스트와 무관하게 계산됩니다. 이 분해를 통해 단일 쿼리에 대해 $\tau \times d \times 256$번의 곱셈-덧셈이 필요한 직접 계산 대신 $256 \times d$번의 곱셈-덧셈과 $\tau \times b \times 256$번의 덧셈으로 충분하게 됩니다. 역색인 리스트 스캔 커널에서는 룩업 테이블이 공유 메모리에 저장되어 수조 회의 랜덤 액세스를 효율적으로 처리하며, 2-패스 $k$-선택을 통해 병렬성과 메모리 소비 사이의 균형을 달성합니다.

다중 GPU 병렬화는 복제와 샤딩 두 가지 전략을 제공합니다. 인덱스가 단일 GPU 메모리에 적재 가능한 경우 복제 방식으로 거의 선형적인 속도 향상을 달성하며, 인덱스가 너무 큰 경우 샤딩으로 데이터를 분할하되 각 샤드가 전체 쿼리를 처리하고 부분 결과를 합칩니다. 실제 구현에서는 4개의 Maxwell Titan X GPU에서 YFCC100M의 9,500만 이미지에 대한 고정확도 $k$-NN 그래프를 35분 만에 구축하고, 10억 벡터에 대한 그래프를 12시간 이내에 완성할 수 있습니다. 이 구현은 FAISS라는 이름으로 오픈소스로 공개되어 실무에서 즉시 활용 가능합니다.

#### 이 연구의 결과가 가지는 의미는 무엇입니까?

실험 결과는 제안된 방법의 우월성을 명확히 입증합니다. $k$-선택 성능에서 WarpSelect는 배열 길이 $\ell = 128,000$일 때 fgknn 대비 $k = 100$에서 1.62배, $k = 1000$에서 2.01배 빠르며, 피크 성능 대비 55%에 도달합니다. SIFT1M 데이터셋에서의 정확 탐색은 피크 가능 성능의 85%를 달성하며, 근사 탐색에서는 SIFT1B에서 단일 GPU로 R@10 = 0.376을 쿼리당 17.7 마이크로초에 달성하여 기존 구현의 8.5배 빠른 성능을 보입니다. DEEP1B에서는 4개 GPU로 R@1 = 0.4517을 쿼리당 0.0133 밀리초에 달성하며, 이는 원래 논문의 CPU 1 스레드 결과인 쿼리당 20 밀리초와 비교하여 게임 체인저 수준의 성능 향상입니다.

$k$-NN 그래프 구축 실험은 이 연구의 실용적 가치를 가장 명확히 보여줍니다. YFCC100M의 9,500만 이미지에서 0.8 이상의 정확도를 35분 만에 달성하며, DEEP1B의 10억 벡터에 대해 낮은 품질의 그래프를 약 6시간에, 높은 품질의 그래프를 약 12시간에 구축합니다. 이는 기존에 알려진 가장 큰 규모의 $k$-NN 그래프 구축(3,650만 개 벡터, 128대 CPU 서버, 108.7시간)과 비교하여 26배 이상 큰 데이터셋을 단 4개 GPU에서 처리하는 것으로, 대규모 유사도 검색의 패러다임을 근본적으로 변화시킵니다. 이미지 경로 탐색 응용에서는 구축된 $k$-NN 그래프를 활용하여 두 이미지 사이의 가장 매끄러운 시각적 전환 경로를 계산할 수 있으며, 이는 단순한 검색을 넘어 데이터셋의 구조적 탐색과 시각적 내비게이션이라는 실용적 응용을 가능하게 합니다.

이 연구의 이론적 의의는 GPU 아키텍처의 특성을 활용한 알고리즘 설계의 중요성을 입증한다는 점에 있습니다. 루프라인 모델에 따른 이론적 하한에 근접하는 성능 달성은 단순히 높은 숫자가 아니라, GPU의 연산 능력을 알고리즘적으로 최대한 활용할 수 있음을 보여줍니다. 특히 정확한 방법이 근사 방법보다도 빠를 수 있다는 결과는 전통적인 대규모 유사도 검색의 정확도-속도 트레이드오프 개념을 재정의합니다. 머신러닝 알고리즘의 인기로 인해 GPU 하드웨어가 이미 과학 워크스테이션에 보편화되어 있으므로, 이러한 GPU 기반 유사도 검색은 데이터베이스 응용에 대한 GPU의 유용성을 크게 확장할 수 있습니다. 공개된 FAISS 라이브러리는 이 연구의 정교하게 엔지니어링된 구현체로서 재현성과 실용적 파급력을 동시에 보장하며, 향후 대규모 데이터 처리 시스템의 표준 도구로 자리잡을 가능성이 높습니다.
- - -
# GPU를 활용한 십억 규모 유사도 검색

## 초록

이 논문은 Facebook AI Research에서 발표한 연구로, GPU를 활용하여 십억(billion) 규모의 벡터 데이터에 대한 유사도 검색(similarity search)을 효율적으로 수행하는 방법을 제안합니다. 핵심 기여는 GPU의 빠른 레지스터 메모리에서 동작하는 $k$-selection 알고리즘으로, 이론적 최대 성능의 55%에 도달하며 기존 GPU 기반 최근접 이웃 탐색 대비 $8.5 \times$ 빠른 속도를 달성합니다. 이 구현을 통해 Yfcc100M 데이터셋의 9,500만 이미지에 대한 고정확도 $k$-NN 그래프를 35분 만에 구축하고, 10억 벡터에 대한 그래프를 4개의 Maxwell Titan X GPU에서 12시간 이내에 완성할 수 있음을 보여줍니다. 해당 구현체는 FAISS(https://github.com/facebookresearch/faiss)라는 이름으로 오픈소스로 공개되었습니다.

## 서론

### 고차원 벡터 표현과 유사도 검색의 필요성

이미지, 비디오, 텍스트 등 복잡한 실세계 데이터를 해석하기 위해 다양한 머신러닝 및 딥러닝 알고리즘이 사용되고 있습니다. 대표적으로 텍스트 표현인 word2vec, 합성곱 신경망(CNN) 기반의 이미지 표현, 인스턴스 검색용 이미지 디스크립터 등이 있으며, 이러한 표현들은 통상 50차원에서 1,000차원 이상의 고차원 실수 벡터 형태로 생성됩니다.

이러한 벡터 임베딩은 대부분 GPU 시스템에서만 효과적으로 생성할 수 있는데, 이는 기반 프로세스가 높은 연산 복잡도와 데이터 대역폭을 요구하거나 통신 오버헤드 없이 효과적으로 분할하기 어렵기 때문입니다. 벡터가 생성된 이후에도 이를 조작하는 과정 자체가 연산 집약적입니다. 그러나 GPU 자원을 어떻게 활용할지는 자명하지 않으며, 이질적(heterogeneous) 아키텍처를 활용하는 방법은 데이터베이스 커뮤니티의 핵심 연구 주제입니다.

유사도 검색은 구조화된 관계가 아닌 수치적 유사성으로 데이터를 탐색하는 방식으로, 가장 유사한 이미지를 찾거나 전체 벡터 컬렉션에서 선형 분류기의 응답이 가장 높은 벡터를 찾는 등의 응용에 적합합니다. 대규모 컬렉션에서 수행하는 가장 비용이 큰 연산 중 하나가 바로 $k$-NN 그래프의 구축입니다. 이는 데이터베이스의 각 벡터가 노드이고, 각 노드가 자신의 $k$개 최근접 이웃과 연결되는 방향 그래프입니다. NN-Descent와 같은 기존 최신 방법은 데이터셋 자체 외에 큰 메모리 오버헤드를 수반하여, 이 논문에서 다루는 십억 규모 데이터베이스로 확장하기 어렵습니다.

### 벡터 압축과 Product Quantization

차원의 저주(curse of dimensionality)로 인해 십억 규모 데이터베이스에서는 전수 탐색이나 정확한 인덱싱 모두 비실용적이므로, 근사 탐색과 그래프 구축에 관한 연구가 활발합니다. RAM에 적재되지 않는 대규모 데이터셋을 처리하기 위해 벡터의 내부 압축 표현을 사용하는 접근법이 채택되며, 이는 메모리 제한이 있는 GPU에 특히 유리합니다. 최소한의 정확도 손실만 감수하면 수 자릿수(orders of magnitude)의 압축률을 달성할 수 있습니다.

벡터 압축 방법은 크게 이진 코드(binary codes)와 양자화(quantization) 방법으로 나뉘며, 두 방법 모두 벡터를 복원하지 않고도 이웃 검색이 가능하다는 장점이 있습니다. 이 논문은 이진 코드보다 효과적인 것으로 입증된 Product Quantization(PQ) 코드 기반 방법에 초점을 맞춥니다. 원래의 PQ 기반 인덱싱 방법(IVFADC) 이후 Inverted Multi-Index, Optimized Product Quantization(OPQ), Polysemous codes 등 여러 개선이 제안되었으나, 이들은 복잡한 알고리즘 구조로 인해 GPU에서 효율적으로 구현하기 어렵습니다. 기존 SIMD 최적화 IVFADC 구현 역시 부최적 파라미터에서만 동작하는 한계가 있습니다.

### 기존 GPU 기반 유사도 검색의 한계와 본 논문의 기여

GPU 기반 유사도 검색 구현은 다수 존재하지만, 대부분 이진 코드, 소규모 데이터셋, 또는 전수 탐색에 한정됩니다. 양자화 코드를 사용한 십억 규모 데이터셋에 적합한 GPU 구현은 Wieschollek et al.의 연구가 유일한 것으로 파악되며, 이것이 본 논문이 비교하는 GPU 기반 기존 최신 기술(state of the art)입니다.

본 논문의 핵심 기여는 세 가지입니다. 첫째, 빠른 레지스터 메모리에서 동작하며 다른 커널과 융합(fuse)할 수 있는 유연한 GPU $k$-selection 알고리즘을 복잡도 분석과 함께 제시합니다. 둘째, 정확 및 근사 $k$-최근접 이웃 탐색을 위한 근최적(near-optimal) 알고리즘 레이아웃을 GPU에 맞게 설계합니다. 셋째, 단일 및 다중 GPU 구성에서 중규모부터 대규모까지의 최근접 이웃 탐색 작업에서 기존 기술을 큰 차이로 능가함을 광범위한 실험으로 입증합니다.
## 문제 정의

이 절에서는 벡터 컬렉션에서의 유사도 검색 문제를 수학적으로 엄밀하게 정의합니다. 앞서 서론에서 개괄적으로 소개된 유사도 검색과 양자화 기반 접근법이 여기에서 구체적인 수식과 알고리즘 구조로 형식화됩니다.

### 최근접 이웃 탐색의 형식적 정의

$d$차원 실수 공간의 쿼리 벡터 $x \in \mathbb{R}^d$와 데이터베이스 벡터 컬렉션 $[y_i]_{i=0:\ell}$ $(y_i \in \mathbb{R}^d)$가 주어졌을 때, $k$-최근접 이웃 탐색은 다음과 같이 정의됩니다.

$$L = k\text{-}\textrm{argmin}_{i=0:\ell} \|x - y_i\|_2$$

여기서 $0:\ell$은 집합 $\{0, \ldots, \ell - 1\}$을 나타내는 0-기반 인덱싱 표기법입니다. 즉, 전체 데이터베이스에서 L2 거리(유클리드 거리) 기준으로 쿼리 $x$에 가장 가까운 $k$개의 벡터 인덱스를 찾는 것이 목표입니다. L2 거리가 가장 널리 사용되는 이유는 word2vec이나 CNN 기반 임베딩 등 다양한 학습 알고리즘이 설계 단계에서부터 L2 거리를 최적화하도록 만들어지며, 선형대수적으로도 유리한 성질을 가지기 때문입니다.

### $k$-선택 연산

$k$-최근접 이웃 탐색의 핵심 연산은 $k$-선택(k-selection)입니다. 이는 배열 $[a_i]_{i=0:\ell}$에서 가장 작은 $k$개의 값 $[a_{s_i}]_{i=0:k}$을 $a_{s_i} \leq a_{s_{i+1}}$ 순서로 정렬하여 찾고, 해당 원소들의 원래 인덱스 $[s_i]_{i=0:k}$ $(0 \leq s_i < \ell)$도 함께 반환하는 연산입니다. 값 $a_i$는 32비트 부동소수점이며, 인덱스 $s_i$는 32비트 또는 64비트 정수입니다. 코사인 유사도와 같이 최댓값을 찾아야 하는 경우에는 비교 연산자를 변경할 수 있으며, 동일한 키 값($a_{s_i} = a_{s_j}$)을 가진 원소들 간의 순서는 특별히 지정되지 않습니다.

다음 파이썬 코드는 $k$-선택 연산의 개념을 구체적으로 보여줍니다.

```python
import numpy as np

def k_selection(array, k):
    """
    배열에서 가장 작은 k개의 값과 해당 인덱스를 반환하는 k-선택 연산.
    실제 GPU 구현에서는 힙이나 소팅 네트워크를 사용하지만,
    여기서는 개념 이해를 위해 단순 구현을 제시합니다.
    """
    # 인덱스와 함께 정렬 (argsort는 오름차순 인덱스 반환)
    sorted_indices = np.argsort(array)  # O(ℓ log ℓ) - 단순 구현
    
    # 가장 작은 k개의 인덱스와 값 추출
    top_k_indices = sorted_indices[:k]           # s_i: 원래 배열에서의 위치
    top_k_values = array[top_k_indices]          # a_{s_i}: 해당 거리 값
    
    return top_k_values, top_k_indices

# 예시: 5개 벡터에서 k=2 최근접 이웃 찾기
distances = np.array([3.2, 1.5, 4.7, 0.8, 2.1], dtype=np.float32)
values, indices = k_selection(distances, k=2)
# values:  [0.8, 1.5]  → 가장 작은 2개의 거리
# indices: [3, 1]      → 해당 벡터의 원래 인덱스
```

### 배칭(Batching)

실제 응용에서는 단일 쿼리가 아닌 $n_{\text{q}}$개의 쿼리 벡터 $[x_j]_{j=0:n_{\text{q}}}$ $(x_j \in \mathbb{R}^d)$를 동시에 처리하는 배치 검색이 수행됩니다. 이는 다중 CPU 스레드나 GPU에서 병렬 처리를 가능하게 하여 효율성을 크게 향상시킵니다. 배치 $k$-선택은 $n_{\text{q}}$개의 개별 배열에서 각각 $k$개의 원소와 인덱스를 선택하는 것으로, 총 $n_{\text{q}} \times k$개의 결과를 생성하며, 각 배열의 길이 $\ell_i \geq k$는 서로 다를 수 있습니다.

### 정확 탐색(Exact Search)

정확 탐색은 전체 쌍별(pairwise) 거리 행렬 $D = [\|x_j - y_i\|_2^2]_{j=0:n_{\text{q}}, i=0:\ell} \in \mathbb{R}^{n_{\text{q}} \times \ell}$을 계산합니다. 이때 핵심적인 수학적 분해가 활용됩니다.

$$\|x_j - y_i\|_2^2 = \|x_j\|^2 + \|y_i\|^2 - 2\langle x_j, y_i \rangle$$

이 분해가 중요한 이유는 계산 효율성에 있습니다. $\|x_j\|^2$과 $\|y_i\|^2$은 각각 행렬 $X$와 $Y$에 대해 한 번의 패스로 사전 계산할 수 있으며, 계산의 병목(bottleneck)은 내적 $\langle x_j, y_i \rangle$, 즉 행렬 곱셈 $XY^\top$이 됩니다. 행렬 곱셈은 cuBLAS와 같은 고도로 최적화된 라이브러리를 통해 GPU에서 매우 효율적으로 수행할 수 있으므로, 이 분해를 통해 GPU의 연산 능력을 최대한 활용할 수 있습니다. 최종적으로 $D$의 각 행에 대해 $k$-선택을 수행하면 $n_q$개 각 쿼리의 $k$-최근접 이웃을 얻게 됩니다.

```python
import numpy as np

def exact_search_decomposed(X, Y, k):
    """
    정확 탐색의 거리 분해 구현.
    X: (n_q, d) 쿼리 행렬, Y: (ell, d) 데이터베이스 행렬
    핵심: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    """
    # 사전 계산: O(n_q * d) + O(ell * d)
    x_norms = np.sum(X ** 2, axis=1, keepdims=True)  # (n_q, 1)
    y_norms = np.sum(Y ** 2, axis=1, keepdims=True)  # (ell, 1)
    
    # 병목 연산: 행렬 곱셈 XY^T → O(n_q * ell * d)
    # GPU에서는 cuBLAS GEMM으로 수행
    inner_products = X @ Y.T  # (n_q, ell)
    
    # 거리 행렬 조립
    D = x_norms + y_norms.T - 2 * inner_products  # (n_q, ell)
    
    # 각 쿼리에 대해 k-선택 수행
    top_k_indices = np.argsort(D, axis=1)[:, :k]
    top_k_distances = np.take_along_axis(D, top_k_indices, axis=1)
    
    return top_k_distances, top_k_indices
```

### 압축 도메인 탐색(Compressed-Domain Search)

이제부터 논문의 핵심인 근사 최근접 이웃 탐색에 집중합니다. 특히 Product quantization for nearest neighbor search 논문에서 제안된 IVFADC 인덱싱 구조를 다룹니다. IVFADC는 두 단계의 양자화(quantization)를 사용하여 데이터베이스 벡터를 인코딩합니다. 데이터베이스 벡터 $y$는 다음과 같이 근사됩니다.

$$y \approx q(y) = q_1(y) + q_2(y - q_1(y))$$

여기서 $q_1: \mathbb{R}^d \rightarrow \mathcal{C}_1 \subset \mathbb{R}^d$는 **조대 양자화기(coarse quantizer)**이고, $q_2: \mathbb{R}^d \rightarrow \mathcal{C}_2 \subset \mathbb{R}^d$는 **정밀 양자화기(fine quantizer)**입니다. 양자화기란 입력 벡터를 유한 집합의 원소로 매핑하는 함수입니다. 직관적으로 비유하자면, $q_1$은 도시 단위로 위치를 특정하는 것이고, $q_2$는 해당 도시 내에서의 세부 주소를 지정하는 것에 해당합니다. $q_1(y)$가 대략적인 위치를 잡고, $q_2(y - q_1(y))$가 그 잔차(residual), 즉 $q_1$이 놓친 오차를 보정하는 구조입니다.

유한 집합의 원소이므로 $q(y)$는 $q_1(y)$의 인덱스와 $q_2(y - q_1(y))$의 인덱스만으로 인코딩할 수 있어 메모리 효율적입니다.
### 비대칭 거리 계산(ADC)과 IVFADC 탐색

앞서 정의한 2단계 양자화를 기반으로, 비대칭 거리 계산(Asymmetric Distance Computation, ADC) 탐색 방법은 다음과 같은 근사 결과를 반환합니다.

$$L_{\text{ADC}} = k\text{-}\textrm{argmin}_{i=0:\ell} \|x - q(y_i)\|_2$$

여기서 "비대칭"이라는 이름은 쿼리 벡터 $x$는 원본 그대로 사용하면서 데이터베이스 벡터 $y_i$만 양자화된 버전 $q(y_i)$로 대체한다는 데서 유래합니다. 만약 양쪽 모두 양자화하면 대칭(Symmetric) 방식이 되지만, 비대칭 방식이 정확도에서 더 유리하기 때문에 이 방식이 채택됩니다.

그러나 전체 데이터베이스의 모든 벡터에 대해 거리를 계산하는 것은 여전히 비효율적입니다. IVFADC는 이를 해결하기 위해 조대 양자화기 $q_1$을 활용한 사전 선택(pre-selection) 단계를 도입합니다. 먼저, 쿼리 $x$에 가장 가까운 $\tau$개의 조대 수준 센트로이드를 찾습니다.

$$L_{\text{IVF}} = \tau\text{-}\textrm{argmin}_{c \in \mathcal{C}_1} \|x - c\|_2$$

여기서 멀티 프로브(multi-probe) 파라미터 $\tau$는 검색할 조대 수준 센트로이드의 수를 결정합니다. 이 양자화기는 재생산 값(reproduction values) 집합 $\mathcal{C}_1$ 내에서 정확한 거리를 사용하는 최근접 이웃 탐색을 수행합니다.

최종적으로 IVFADC 탐색은 사전 선택된 센트로이드에 속하는 벡터들에 대해서만 거리를 계산합니다.

$$L_{\text{IVFADC}} = \underset{i=0:\ell \text{ s.t. } q_1(y_i) \in L_{\text{IVF}}}{k\text{-}\textrm{argmin}} \|x - q(y_i)\|_2$$

직관적으로 설명하면, IVFADC는 대형 도서관에서 책을 찾는 과정과 유사합니다. 먼저 $q_1$을 통해 관련 서가(bookshelf) $\tau$개를 선택하고($L_{\text{IVF}}$), 선택된 서가 내에서만 개별 책들의 거리를 계산합니다. 전체 도서관을 뒤질 필요 없이 관련 서가만 탐색하므로 검색 속도가 크게 향상됩니다.

### 역색인 파일(Inverted File) 구조

이 탐색을 효율적으로 지원하는 자료 구조가 역색인 파일(inverted file)입니다. 데이터베이스 벡터 $y_i$를 조대 양자화 결과 $q_1(y_i)$에 따라 $|\mathcal{C}_1|$개의 역색인 리스트 $\mathcal{I}_1, \ldots, \mathcal{I}_{|\mathcal{C}_1|}$로 그룹화합니다. 같은 리스트에 속한 벡터들은 동일한 $q_1(y_i)$ 값을 가지므로, 탐색 시 선택된 $\tau$개의 리스트만 순차적으로 스캔하면 됩니다. 따라서 가장 메모리 집약적인 연산은 $L_{\text{IVFADC}}$를 계산하는 것이며, 이는 $\tau$개의 역색인 리스트를 선형 스캔하는 것으로 귀결됩니다.

```python
import numpy as np

def ivfadc_search_conceptual(query, coarse_centroids, inverted_lists, 
                              pq_codebooks, tau, k):
    """
    IVFADC 탐색의 개념적 구현.
    1단계: 가장 가까운 tau개의 조대 센트로이드 선택 (IVF 단계)
    2단계: 선택된 리스트 내 벡터들에 대해 PQ 거리 계산 (ADC 단계)
    """
    d = query.shape[0]
    
    # 1단계: 쿼리에 가장 가까운 tau개의 조대 센트로이드 찾기
    coarse_distances = np.linalg.norm(coarse_centroids - query, axis=1)
    L_IVF = np.argsort(coarse_distances)[:tau]  # tau개의 센트로이드 인덱스
    
    # 2단계: 선택된 역색인 리스트만 스캔
    all_distances = []
    all_ids = []
    for list_idx in L_IVF:
        for (vector_id, pq_code) in inverted_lists[list_idx]:
            # PQ 코드로부터 근사 거리 계산 (ADC)
            residual_centroid = coarse_centroids[list_idx]
            approx_dist = compute_pq_distance(
                query - residual_centroid, pq_code, pq_codebooks
            )
            all_distances.append(approx_dist)
            all_ids.append(vector_id)
    
    # k-선택: 가장 작은 k개의 거리를 가진 벡터 반환
    all_distances = np.array(all_distances)
    top_k_idx = np.argsort(all_distances)[:k]
    return np.array(all_ids)[top_k_idx], all_distances[top_k_idx]
```

이 코드에서 볼 수 있듯이, 전체 $\ell$개 벡터가 아닌 $\tau$개의 역색인 리스트에 포함된 벡터들만 스캔하므로 계산량이 크게 줄어듭니다.

### 양자화기의 특성

두 양자화기 $q_1$과 $q_2$는 서로 다른 설계 요구사항을 가집니다. $q_1$은 역색인 리스트의 수가 지나치게 많아지지 않도록 비교적 적은 수의 재생산 값을 가져야 합니다. 일반적으로 $|C_1| \approx \sqrt{\ell}$로 설정하며, $k$-means 클러스터링으로 학습합니다. 예를 들어 10억($10^9$)개의 벡터를 다룰 때 $|C_1| \approx \sqrt{10^9} \approx 31{,}623$개 정도의 센트로이드가 사용됩니다.

반면 $q_2$에는 더 많은 메모리를 투자하여 더 정밀한 표현을 구성할 수 있습니다. 역색인 리스트에는 벡터의 ID(4바이트 또는 8바이트 정수)도 저장되므로, 이보다 짧은 코드를 사용하는 것은 비효율적입니다. 즉, $\log_2 |\mathcal{C}_2| > 4 \times 8 = 32$비트 이상의 코드가 필요합니다.

### 곱 양자화기(Product Quantizer)

$q_2$에는 곱 양자화기(Product Quantizer, PQ)가 사용됩니다. PQ의 핵심 아이디어는 고차원 벡터를 여러 저차원 부분벡터(sub-vector)로 분할하여 독립적으로 양자화하는 것입니다. 구체적으로, $d$차원 벡터 $y$를 $b$개의 부분벡터 $y = [y^0 \ldots y^{b-1}]$로 분할합니다. 여기서 $b$는 차원 $d$의 짝수 약수입니다. 각 부분벡터는 자체 양자화기 $q^j$로 양자화되어 튜플 $(q^0(y^0), \ldots, q^{b-1}(y^{b-1}))$을 생성합니다.

각 부분 양자화기는 일반적으로 256개의 재생산 값을 가지며, 이는 정확히 1바이트에 저장됩니다. 따라서 PQ의 전체 양자화 값은 다음과 같이 구성됩니다.

$$q_2(y) = q^0(y^0) + 256 \times q^1(y^1) + \ldots + 256^{b-1} \times q^{b-1}(y^{b-1})$$

저장 관점에서 이는 각 부분 양자화기가 생성한 바이트들의 단순 연결(concatenation)입니다. 결과적으로 PQ는 $b$바이트 코드를 생성하며, $|\mathcal{C}_2| = 256^b$개의 재생산 값을 가집니다. 예를 들어 $b = 8$이면 벡터당 8바이트만으로 $256^8 \approx 1.8 \times 10^{19}$개의 서로 다른 양자화 값을 표현할 수 있습니다. 각 부분 양자화기의 $k$-means 사전(dictionary)은 크기가 작아 양자화 자체의 계산 비용은 무시할 수 있을 정도입니다.

이러한 PQ의 설계가 탁월한 이유는 처리 비용을 증가시키지 않으면서도 재생산 값의 수를 기하급수적으로 늘릴 수 있다는 점에 있습니다. 단일 양자화기로 $256^b$개의 센트로이드를 직접 학습하는 것은 불가능하지만, PQ는 $b$개의 독립적인 256-센트로이드 양자화기를 학습하면 되므로 계산적으로 실현 가능합니다.

```python
import numpy as np

def product_quantize(vector, codebooks, b):
    """
    곱 양자화: d차원 벡터를 b바이트 코드로 압축.
    codebooks: b개의 (256, d/b) 크기 코드북 리스트
    """
    d = vector.shape[0]
    sub_dim = d // b  # 각 부분벡터의 차원
    pq_code = np.zeros(b, dtype=np.uint8)
    
    for j in range(b):
        # j번째 부분벡터 추출
        sub_vector = vector[j * sub_dim : (j + 1) * sub_dim]
        # 가장 가까운 센트로이드 찾기 (256개 중)
        distances = np.linalg.norm(codebooks[j] - sub_vector, axis=1)
        pq_code[j] = np.argmin(distances)  # 0~255의 인덱스 → 1바이트
    
    return pq_code  # b바이트 코드

# 예시: 128차원 벡터를 8바이트로 압축 (압축률 128*4/8 = 64배)
d, b = 128, 8
codebooks = [np.random.randn(256, d // b) for _ in range(b)]
vector = np.random.randn(d)
code = product_quantize(vector, codebooks, b)
# code: [42, 187, 3, 215, 91, 128, 67, 244] → 단 8바이트
```

이 코드에서 128차원(512바이트) 벡터가 단 8바이트의 PQ 코드로 압축되는 과정을 확인할 수 있습니다. 각 부분벡터가 독립적으로 256개 센트로이드 중 하나에 할당되어 1바이트 인덱스로 저장되므로, 이후 GPU에서의 거리 계산 시 룩업 테이블 기반의 매우 효율적인 연산이 가능해집니다.
## GPU 아키텍처 개요와 $k$-선택

이 절에서는 Nvidia의 범용 GPU 아키텍처와 프로그래밍 모델의 핵심 특성을 분석하고, 유사도 검색에서 GPU 친화적이지 않은 부분인 $k$-선택 문제를 집중적으로 다룹니다. 앞서 정의한 $k$-선택 연산이 GPU에서 왜 도전적인 문제인지 이해하기 위해서는 먼저 GPU의 실행 모델과 메모리 계층 구조에 대한 깊은 이해가 필요합니다.

### GPU 아키텍처

#### GPU 레인과 워프

Nvidia GPU는 32개의 CUDA 스레드로 구성된 벡터 단위인 **워프(warp)**를 기본 실행 단위로 사용하는 범용 컴퓨터입니다. 워프 내의 개별 스레드는 **레인(lane)**이라 불리며, 0부터 31까지의 레인 ID를 가집니다. "스레드"라는 용어가 사용되지만, 현대 벡터화된 멀티코어 CPU와의 가장 적절한 대응 관계는 각 워프가 하나의 독립적인 CPU 하드웨어 스레드에 해당한다는 것입니다. 이는 워프 내의 모든 레인이 하나의 명령어 카운터(instruction counter)를 공유하기 때문입니다.

이 구조가 의미하는 바를 직관적으로 설명하면, 워프는 마치 32명이 한 줄로 서서 동일한 지시를 동시에 수행하는 것과 같습니다. 32명 모두 "더하기"를 수행하되, 각자 다른 데이터에 대해 연산을 적용합니다. 그런데 만약 일부 레인이 다른 실행 경로를 취하게 되면 **워프 발산(warp divergence)**이 발생하여 성능이 저하됩니다. 예를 들어 `if-else` 조건문에서 절반의 레인이 `if` 경로를, 나머지 절반이 `else` 경로를 택하면, GPU는 두 경로를 순차적으로 실행해야 하므로 실질적인 병렬성이 반으로 줄어듭니다.

각 레인은 공유 레지스터 파일에서 최대 255개의 32비트 레지스터를 사용할 수 있습니다. CPU 관점으로 대응시키면, 폭(width) 32의 벡터 레지스터가 최대 255개 존재하고, 워프 레인이 SIMD 벡터 레인에 해당하는 구조입니다.

```python
class WarpExecution:
    """GPU 워프의 실행 모델을 개념적으로 시뮬레이션하는 클래스"""
    
    WARP_SIZE = 32  # 워프 내 레인 수 (고정)
    MAX_REGISTERS_PER_LANE = 255  # 레인당 최대 32비트 레지스터
    
    def __init__(self):
        # 각 레인은 독립적 데이터를 가지지만 동일한 명령을 수행
        self.lanes = [{"lane_id": i, "registers": [0] * 255} 
                      for i in range(self.WARP_SIZE)]
        self.instruction_counter = 0  # 공유 명령어 카운터 (핵심!)
    
    def execute_simd(self, operation, data_per_lane):
        """모든 레인이 동일한 연산을 서로 다른 데이터에 수행 (이상적 실행)"""
        results = []
        for lane_id in range(self.WARP_SIZE):
            results.append(operation(data_per_lane[lane_id]))
        self.instruction_counter += 1  # 단 1회 증가 → 32개 연산 동시 수행
        return results
    
    def execute_with_divergence(self, condition, data_per_lane, op_true, op_false):
        """워프 발산 발생 시: 두 경로를 순차 실행해야 함"""
        results = [None] * self.WARP_SIZE
        # 1단계: condition=True인 레인만 실행 (나머지는 비활성)
        for lane_id in range(self.WARP_SIZE):
            if condition(data_per_lane[lane_id]):
                results[lane_id] = op_true(data_per_lane[lane_id])
        self.instruction_counter += 1
        
        # 2단계: condition=False인 레인만 실행 (나머지는 비활성)
        for lane_id in range(self.WARP_SIZE):
            if not condition(data_per_lane[lane_id]):
                results[lane_id] = op_false(data_per_lane[lane_id])
        self.instruction_counter += 1  # 2번 증가 → 성능 절반으로 저하
        return results
```

이 코드에서 볼 수 있듯이, `execute_simd`는 명령어 카운터가 1회만 증가하면서 32개 연산을 동시에 처리하지만, `execute_with_divergence`에서는 분기 때문에 2회 증가하여 실질적 처리량이 절반으로 감소합니다. 이것이 바로 $k$-선택 같은 데이터 의존적(data-dependent) 알고리즘이 GPU에서 도전적인 근본 이유입니다.

#### 워프의 집합: 블록과 스트리밍 멀티프로세서

사용자가 구성 가능한 1개에서 32개의 워프 집합이 하나의 **블록(block)** 또는 **협력 스레드 배열(CTA, Cooperative Thread Array)**을 구성합니다. 각 블록은 최대 48 KiB 크기의 고속 **공유 메모리(shared memory)**를 가집니다. 블록 내 개별 CUDA 스레드는 블록 기준 상대 ID인 **스레드 ID(thread id)**를 통해 작업을 분할하고 할당받을 수 있습니다.

각 블록은 GPU의 단일 코어인 **스트리밍 멀티프로세서(SM, Streaming Multiprocessor)**에서 실행됩니다. 각 SM은 ALU, 메모리 로드/스토어 유닛, 특수 명령어 유닛 등의 **기능 유닛(functional units)**을 포함합니다. GPU가 실행 지연(latency)을 숨기는 핵심 전략은 모든 SM에 걸쳐 다수의 워프에서 많은 연산을 동시에 비행 중(in-flight) 상태로 유지하는 것입니다. 개별 워프 레인의 명령어 처리량은 낮고 지연은 높지만, 모든 SM의 총합 산술 처리량은 일반 CPU 대비 $5 \times$에서 $10 \times$ 높습니다.

#### 그리드와 커널

블록들은 **커널(kernel)** 내에서 **블록 그리드(grid of blocks)**로 구성됩니다. 각 블록에는 그리드 기준 상대 ID가 부여됩니다. 커널은 호스트 CPU가 GPU에 스케줄링하는 작업 단위(인수가 포함된 명령어 스트림)입니다. 블록이 완료되면 새로운 블록이 스케줄링되고, 서로 다른 커널의 블록들도 동시에 실행될 수 있습니다. 커널 간 순서는 **스트림(streams)**과 **이벤트(events)** 같은 순서 지정 프리미티브로 제어됩니다.

#### 리소스와 점유율

동시에 실행되는 블록의 수는 각 블록이 사용하는 공유 메모리와 레지스터 리소스에 의존합니다. CUDA 스레드당 레지스터 사용량은 컴파일 시점에 결정되고, 공유 메모리 사용량은 런타임에 선택할 수 있습니다. 이 사용량이 GPU의 **점유율(occupancy)**에 영향을 미칩니다. 만약 한 블록이 48 KiB 공유 메모리 전체를 사용하거나, 스레드당 32개가 아닌 128개의 레지스터를 요구하면, 동일 SM에서 동시에 실행 가능한 다른 블록이 1~2개로 제한되어 낮은 점유율이 발생합니다. 높은 점유율에서는 더 많은 블록이 모든 SM에 분포하여 더 많은 작업이 동시에 비행 중 상태가 됩니다.

#### 메모리 유형과 대역폭 계층

서로 다른 블록과 커널은 **글로벌 메모리(global memory)**를 통해 통신하며, 이는 일반적으로 4~32 GB 크기에 CPU 메인 메모리 대비 $5 \times$에서 $10 \times$ 높은 대역폭을 제공합니다. 공유 메모리는 속도 면에서 CPU L1 캐시에 비유됩니다. GPU 레지스터 파일 메모리는 가장 높은 대역폭을 가진 메모리입니다.

GPU에서 비행 중인 대량의 명령어를 유지하기 위해 방대한 레지스터 파일이 필요한데, 최신 Pascal P100의 경우 14 MB에 달하며, 이는 CPU의 수십 KB에 비해 압도적입니다. 세 가지 메모리 유형 간의 총 단면 대역폭(aggregate cross-sectional bandwidth) 비율은 레지스터 : 공유 : 글로벌 = $250 : 6.25 : 1$이 전형적이며, 레지스터 파일의 경우 $10 \sim 100$s TB/s에 달합니다.

```python
class GPUMemoryHierarchy:
    """GPU 메모리 계층의 특성을 정량적으로 모델링"""
    
    def __init__(self, gpu_model="Pascal_P100"):
        if gpu_model == "Pascal_P100":
            self.register_file_size_mb = 14        # 14 MB (CPU 대비 ~500배)
            self.shared_memory_per_block_kib = 48   # 48 KiB
            self.global_memory_gb = 16              # 16 GB HBM2
            self.global_bandwidth_gbps = 732        # GB/s
            
            # 대역폭 비율: 레지스터 : 공유 : 글로벌 = 250 : 6.25 : 1
            self.bandwidth_ratio = {"register": 250, "shared": 6.25, "global": 1}
    
    def estimate_bandwidth(self, memory_type):
        """메모리 유형별 추정 대역폭 (TB/s)"""
        base = self.global_bandwidth_gbps / 1000  # TB/s로 변환
        return base * self.bandwidth_ratio[memory_type]
    
    def occupancy_analysis(self, registers_per_thread, shared_mem_per_block_kib,
                           threads_per_block=1024):
        """점유율에 미치는 리소스 영향 분석"""
        # SM당 최대 레지스터: 65536 (32비트)
        max_registers_per_sm = 65536
        max_shared_per_sm_kib = 48  # SM당 공유 메모리
        
        # 레지스터 기준 최대 블록 수
        regs_needed = registers_per_thread * threads_per_block
        blocks_by_regs = max_registers_per_sm // regs_needed
        
        # 공유 메모리 기준 최대 블록 수
        blocks_by_shared = int(max_shared_per_sm_kib / shared_mem_per_block_kib)
        
        concurrent_blocks = min(blocks_by_regs, blocks_by_shared)
        return {
            "concurrent_blocks_per_sm": concurrent_blocks,
            "occupancy_note": "높음" if concurrent_blocks >= 4 else 
                             "보통" if concurrent_blocks >= 2 else "낮음"
        }

# 예시: 점유율 트레이드오프 분석
gpu = GPUMemoryHierarchy()
# 경우 1: 적은 리소스 사용 → 높은 점유율
case1 = gpu.occupancy_analysis(registers_per_thread=32, shared_mem_per_block_kib=8)
# 경우 2: 많은 리소스 사용 → 낮은 점유율 (하지만 더 빠른 메모리에 더 많은 데이터 보관)
case2 = gpu.occupancy_analysis(registers_per_thread=128, shared_mem_per_block_kib=48)
```

이 코드에서 보듯이, 스레드당 레지스터를 32개에서 128개로 늘리면 SM당 동시 실행 블록 수가 크게 감소하지만, 더 많은 데이터를 가장 빠른 레지스터 메모리에 유지할 수 있습니다. 이러한 점유율과 성능 간의 트레이드오프가 이 논문의 $k$-선택 알고리즘 설계에서 핵심적인 고려사항이 됩니다.

### GPU 레지스터 파일 활용

#### 구조화된 레지스터 데이터

공유 메모리와 레지스터 메모리의 활용에는 효율성 트레이드오프가 존재합니다. 이들의 사용은 점유율을 낮추지만, 더 빠른 메모리에 더 큰 작업 집합을 유지함으로써 전반적인 성능을 향상시킬 수 있습니다. 점유율을 희생하거나 공유 메모리 대신 레지스터 상주 데이터를 적극 활용하는 것이 종종 이득이 되는데, 이는 논문 [43]에서도 입증된 바 있습니다.

GPU 레지스터 파일이 매우 크기 때문에(Pascal P100에서 14 MB), 단순한 임시 피연산자뿐만 아니라 **구조화된 데이터(structured data)**를 저장하는 것이 유용합니다. 단일 레인이 자신의 스칼라 레지스터를 사용하여 로컬 작업을 해결할 수도 있지만, 이는 병렬성과 저장 용량이 제한됩니다. 대신, 워프 내 레인들이 **워프 셔플(warp shuffle) 명령어**를 통해 레지스터 데이터를 교환할 수 있으며, 이를 통해 워프 전체 수준의 병렬성과 저장 능력을 확보할 수 있습니다.

#### 레인 스트라이드 레지스터 배열

이러한 워프 수준 병렬성을 달성하기 위한 핵심 패턴이 **레인 스트라이드 레지스터 배열(lane-stride register array)**입니다. 원소 $[a_i]_{i=0:\ell}$이 주어졌을 때, 연속적인 값들이 인접한 레인의 레지스터에 분배되는 방식입니다. 배열은 레인당 $\ell / 32$개의 레지스터에 저장되며, $\ell$은 32의 배수여야 합니다.

구체적으로, 레인 $j$는 원소 $\{a_j, a_{32+j}, \ldots, a_{\ell - 32 + j}\}$를 저장합니다. 반면 레지스터 $r$은 원소 $\{a_{32r}, a_{32r+1}, \ldots, a_{32r+31}\}$을 보유합니다. 이 구조에서 원소 $a_i$를 조작하려면, 해당 원소가 저장된 레지스터 인덱스(즉, $\lfloor i / 32 \rfloor$)와 $\ell$은 어셈블리 시점에 알려져야 하고, 레인 인덱스(즉, $i \bmod 32$)는 런타임에 결정될 수 있습니다.

```python
import numpy as np

def lane_stride_register_array(data, warp_size=32):
    """
    레인 스트라이드 레지스터 배열의 데이터 배치를 시뮬레이션.
    data: 길이 ℓ인 배열 (ℓ은 32의 배수)
    
    핵심 개념:
    - 레인 j가 저장하는 원소: a_j, a_{32+j}, a_{64+j}, ...
    - 레지스터 r이 보유하는 원소: a_{32r}, a_{32r+1}, ..., a_{32r+31}
    """
    ell = len(data)
    assert ell % warp_size == 0, "ℓ must be a multiple of 32"
    registers_per_lane = ell // warp_size
    
    # 각 레인이 저장하는 원소들 시각화
    lane_storage = {}
    for j in range(warp_size):
        # 레인 j → {a_j, a_{32+j}, a_{64+j}, ...}
        lane_storage[f"lane_{j}"] = [data[j + 32 * r] for r in range(registers_per_lane)]
    
    # 각 레지스터가 보유하는 원소들 시각화
    register_storage = {}
    for r in range(registers_per_lane):
        # 레지스터 r → {a_{32r}, a_{32r+1}, ..., a_{32r+31}}
        register_storage[f"reg_{r}"] = [data[32 * r + lane] for lane in range(warp_size)]
    
    return lane_storage, register_storage

def access_element(i, warp_size=32):
    """
    원소 a_i에 접근하기 위한 레지스터/레인 인덱스 계산.
    - 레지스터 인덱스: floor(i/32) → 컴파일 타임에 결정 필요
    - 레인 인덱스: i mod 32 → 런타임에 결정 가능
    """
    register_idx = i // warp_size   # 어셈블리 시점에 알아야 함
    lane_idx = i % warp_size        # 런타임에 결정 가능
    return register_idx, lane_idx

# 예시: 128개 원소를 레인 스트라이드 배열로 배치
data = np.arange(128)  # [0, 1, 2, ..., 127]
lanes, regs = lane_stride_register_array(data)

# 레인 0: [0, 32, 64, 96]  → 4개 레지스터에 분산
# 레인 1: [1, 33, 65, 97]
# 레지스터 0: [0, 1, 2, ..., 31]  → 32개 레인에 걸쳐 분포
# 레지스터 1: [32, 33, 34, ..., 63]

# 원소 a_47 접근: 레지스터 1 (=47//32), 레인 15 (=47%32)
reg, lane = access_element(47)  # reg=1, lane=15
```

이 배열 구조가 강력한 이유는 다양한 접근 패턴(시프트, 임의 대 임의 등)이 제공되기 때문이며, 논문에서는 특히 **버터플라이 순열(butterfly permutation)** [29]을 광범위하게 활용합니다. 버터플라이 순열은 FFT(고속 푸리에 변환)에서 유래한 통신 패턴으로, 레인 간 데이터 교환을 로그 단계 수만에 완료할 수 있게 해줍니다.

### CPU 대 GPU에서의 $k$-선택

앞서 정의한 $k$-선택 연산을 GPU에서 수행하는 것은 여러 근본적 도전을 수반합니다. 임의로 큰 $\ell$과 $k$에 대한 기존 $k$-선택 알고리즘들은 GPU로 변환될 수 있지만, 각각 고유한 문제점을 가집니다.

**기수 선택(radix selection)**과 **버킷 선택(bucket selection)** [1], **확률적 선택(probabilistic selection)** [33], **퀵셀렉트(quickselect)** [14], 그리고 **절단 정렬(truncated sorts)** [40] 등이 존재하지만, 이들의 성능은 글로벌 메모리 상의 입력을 여러 번 순회하는 것에 지배됩니다. 유사도 검색에서는 입력 거리가 즉석에서(on-the-fly) 계산되거나 작은 블록 단위로만 저장될 수 있으며, 전체 배열이 어떤 메모리에도 담기지 않거나 처리 시작 시점에 크기를 알 수 없어 다중 패스를 요구하는 알고리즘이 비실용적입니다.

퀵셀렉트는 $\mathcal{O}(\ell)$ 크기의 저장 공간에서 파티셔닝을 수행해야 하는데, 이는 데이터 의존적 메모리 이동을 초래합니다. 이로 인해 과도한 메모리 트랜잭션이 발생하거나, 쓰기 오프셋을 결정하기 위한 병렬 접두사 합(parallel prefix sum)과 동기화 오버헤드가 필요합니다. 기수 선택은 파티셔닝이 없지만 여전히 다중 패스가 필요합니다.

#### 힙 기반 접근법의 한계

유사도 검색 응용에서는 보통 $k < 1000$ 정도의 적은 수의 결과만 필요합니다. 이 영역에서 CPU의 전형적 선택은 **최대 힙(max-heap)**이지만, 힙은 트리 업데이트의 직렬 특성으로 인해 데이터 병렬성을 거의 노출하지 못하여 SIMD 실행 유닛을 포화시킬 수 없습니다. **ad-heap** [31]은 이기종 시스템에서 사용 가능한 병렬성을 더 잘 활용하지만, 여전히 직렬과 병렬 작업을 적절한 실행 유닛 사이에 분할하려는 시도입니다.

힙 업데이트의 직렬적 특성에도 불구하고, 작은 $k$에서 CPU는 모든 상태를 L1 캐시에 쉽게 유지할 수 있으며, L1 캐시 지연과 대역폭이 제한 요인으로 남습니다. CPU에서는 PQ 코드 조작 같은 다른 유사도 검색 구성 요소가 성능에 더 큰 영향을 미치는 경향이 있습니다 [2].

GPU에서도 힙을 유사하게 구현할 수 있지만 [7], 직관적인 GPU 힙 구현은 삽입되는 각 원소의 경로가 힙 내 다른 값들에 의존하므로 높은 워프 발산과 불규칙적이고 데이터 의존적인 메모리 이동 문제를 겪습니다. GPU **병렬 우선순위 큐(parallel priority queues)** [24]는 다수의 동시 업데이트를 허용하여 직렬 힙 업데이트를 개선하지만, 각 삽입마다 잠재적으로 소규모 정렬이 필요하고 데이터 의존적 메모리 이동이 발생합니다. 또한 서로 다른 스트림에서의 커널 실행을 통한 다중 동기화 장벽, 연속적인 커널 실행 지연, CPU 호스트와의 조율 등의 추가 지연이 요구됩니다.

작은 $k$에 대한 더 독창적인 GPU 알고리즘으로 **fgknn 라이브러리** [41]의 선택 알고리즘이 있습니다. 이는 복잡한 알고리즘으로, 지나치게 많은 동기화 지점, 큰 커널 실행 오버헤드, 더 느린 메모리의 사용, 과도한 계층 구조 활용, 파티셔닝과 버퍼링 등의 문제를 겪을 수 있습니다. 그러나 논문에서는 이 알고리즘의 **병합 큐(merge queue) 구조**에서 사용되는 **병렬 병합(parallel merges)** 개념에서 영감을 받았다고 밝히고 있으며, 이것이 다음 절에서 소개될 WarpSelect 알고리즘의 설계 근간이 됩니다.

```python
def demonstrate_gpu_kselection_challenges():
    """
    GPU에서 k-선택이 도전적인 이유를 코드로 시연.
    핵심 문제: 데이터 의존적 제어 흐름 → 워프 발산
    """
    import numpy as np
    
    # 힙 삽입의 데이터 의존적 경로 시뮬레이션
    # 32개 레인이 동시에 힙에 삽입하려는 상황
    warp_size = 32
    heap_size = 100  # k=100인 최대 힙
    
    # 각 레인의 삽입 값
    values_to_insert = np.random.rand(warp_size)
    heap_root_value = 0.5  # 힙의 루트 (현재 최대값)
    
    # 문제: 각 레인이 다른 경로를 따라감
    divergence_count = 0
    for lane in range(warp_size):
        if values_to_insert[lane] < heap_root_value:
            # 이 레인은 삽입 필요 → 트리 내려가기 (경로가 값에 의존)
            path = "insert_and_sift_down"
            divergence_count += 1
        else:
            # 이 레인은 삽입 불필요 → 아무것도 안 함
            path = "skip"
    
    # 결과: 32개 레인 중 일부만 삽입을 수행하고, 삽입하는 레인들도
    # 각각 다른 깊이까지 트리를 순회 → 심각한 워프 발산!
    divergence_ratio = divergence_count / warp_size
    
    # 다중 패스 문제 시연
    ell = 1_000_000  # 100만 개 원소
    global_memory_bandwidth_gbps = 732  # P100 기준
    element_size_bytes = 4  # float32
    
    # 기수 선택: 32비트 float → 최대 32패스 필요
    radix_passes = 32  # 최악의 경우
    total_data_movement_gb = (ell * element_size_bytes * radix_passes) / 1e9
    time_radix_ms = (total_data_movement_gb / global_memory_bandwidth_gbps) * 1000
    
    # 반면, 레지스터에서 단일 패스로 처리하면:
    register_bandwidth_tbps = 100  # ~100 TB/s
    single_pass_data_gb = (ell * element_size_bytes) / 1e9
    time_register_ms = (single_pass_data_gb / (register_bandwidth_tbps * 1000)) * 1000
    
    return {
        "워프_발산_비율": f"{divergence_ratio:.0%}",
        "기수_선택_추정_시간_ms": f"{time_radix_ms:.4f}",
        "레지스터_단일패스_추정_시간_ms": f"{time_register_ms:.6f}",
        "속도_향상": f"{time_radix_ms / time_register_ms:.0f}x"
    }
```

이 분석이 보여주는 핵심 메시지는, 글로벌 메모리에서 다중 패스를 수행하는 기존 $k$-선택 알고리즘과 레지스터 메모리에서 단일 패스로 처리하는 알고리즘 사이에는 대역폭 비율 $250:1$에서 비롯되는 근본적인 성능 차이가 존재한다는 것입니다. 이것이 바로 다음 절에서 제시될 WarpSelect 알고리즘이 모든 상태를 레지스터에 유지하면서 단일 패스로 $k$-선택을 수행하는 설계 방향을 채택하게 된 핵심 동기입니다.
## GPU에서의 빠른 $k$-선택

앞서 GPU 아키텍처와 기존 $k$-선택 알고리즘의 한계를 분석한 바 있습니다. 이 절에서는 이러한 분석을 토대로, 레지스터 파일에서 모든 상태를 유지하며 단일 패스로 동작하는 WarpSelect 알고리즘의 구체적 설계와 구현을 심층적으로 다룹니다.

[루프라인 성능 모델(Roofline performance model)](https://en.wikipedia.org/wiki/Roofline_model)에 따르면, 모든 CPU 또는 GPU 알고리즘에서 메모리 대역폭이나 산술 처리량 중 하나가 성능의 제한 요인이 됩니다. 글로벌 메모리에서 입력을 받는 $k$-선택의 경우, 최대 메모리 대역폭으로 입력을 한 번 스캔하는 데 소요되는 시간보다 빠르게 수행될 수 없습니다. 따라서 이 이론적 한계에 최대한 근접하는 것이 설계 목표가 됩니다. 이를 위해 입력 데이터에 대한 단일 패스 처리를 지향하며, 중간 상태를 가장 빠른 메모리인 레지스터 파일에 유지합니다. 다만 레지스터 메모리의 가장 큰 제약은 레지스터 파일에 대한 인덱싱이 반드시 어셈블리 시점에 결정되어야 한다는 점이며, 이것이 알고리즘 설계에 강한 제약 조건으로 작용합니다.

### 레지스터 내 정렬

WarpSelect의 핵심 빌딩 블록은 레지스터 내에서 동작하는 정렬 프리미티브입니다. [Batcher의 바이토닉 정렬 네트워크(bitonic sorting network)](https://en.wikipedia.org/wiki/Bitonic_sorter)는 SIMD 아키텍처에서 벡터 병렬성을 활용할 수 있기 때문에 널리 사용되며, GPU에서도 쉽게 구현할 수 있습니다. 이 논문에서는 앞서 소개한 레인 스트라이드 레지스터 배열 위에 정렬 네트워크를 구축합니다.

정렬 네트워크의 기본 개념을 직관적으로 설명하면, 이는 마치 미리 정해진 위치 쌍들을 비교하여 작은 값을 왼쪽으로, 큰 값을 오른쪽으로 교환하는 "비교-교환(compare-swap)" 연산들의 고정된 시퀀스와 같습니다. 핵심적인 장점은 비교 패턴이 데이터 값과 무관하게 사전에 결정되어 있다는 것입니다. 이는 GPU에서 워프 발산을 최소화하는 데 매우 유리합니다.

바이토닉 정렬 네트워크는 크기 $2^k$인 배열에 대한 병렬 병합(merge)의 집합으로 구성됩니다. 각 병합 단계에서 $s$개의 길이 $t$인 배열(여기서 $s$와 $t$는 모두 2의 거듭제곱)을 $s/2$개의 길이 $2t$인 배열로 변환하며, 이때 $\log_2(t)$개의 병렬 단계가 필요합니다. 바이토닉 정렬은 이 병합을 재귀적으로 적용합니다. 길이 $\ell$인 배열을 정렬하려면, $\ell$개의 길이 1인 배열에서 시작하여 $\ell/2$개의 길이 2인 배열로, 다시 $\ell/4$개의 길이 4인 배열로, 최종적으로 1개의 길이 $\ell$인 정렬된 배열로 변환합니다. 이 과정에서 필요한 전체 병렬 병합 단계 수는 다음과 같습니다.

$$\frac{1}{2}(\log_2(\ell)^2 + \log_2(\ell))$$

예를 들어 $\ell = 8$이면 $\frac{1}{2}(9 + 3) = 6$개의 병렬 단계가 필요합니다. 이 복잡도는 비교 기반 정렬의 병렬 깊이로서 이론적으로 잘 알려진 결과이며, [SIMD CPU 아키텍처에서의 효율적 정렬 구현](https://en.wikipedia.org/wiki/Sorting_network)에 관한 연구에서도 이 구조가 SIMD 벡터 레인에 자연스럽게 매핑됨이 입증되었습니다.

#### 홀수 크기 병합 네트워크

실제 응용에서는 입력 데이터가 2의 거듭제곱 크기를 갖지 않는 경우가 대부분이고, 일부 데이터가 이미 정렬되어 있을 수도 있습니다. 이러한 상황을 효율적으로 처리하기 위해 **홀수 크기 병합 네트워크(odd-size merging network)**가 도입됩니다. Algorithm 1은 이미 정렬된 임의 길이의 좌측 배열 $[L_i]_{i=0:\ell_L}$과 우측 배열 $[R_i]_{i=0:\ell_R}$을 병합하는 알고리즘입니다.

이 알고리즘의 핵심 아이디어를 직관적으로 이해하기 위해, 배열을 다음 2의 거듭제곱 크기로 패딩하되 결코 교환되지 않는 더미(dummy) 원소를 사용하는 것으로 생각할 수 있습니다. 바이토닉 병합이 바이토닉 시퀀스(먼저 증가했다 감소하는 시퀀스)를 대상으로 하는 반면, 여기서는 단조(monotonic) 시퀀스, 즉 이미 오름차순 또는 내림차순으로 정렬된 시퀀스에서 출발합니다. 바이토닉 병합을 단조 병합으로 전환하는 핵심 트릭은 첫 번째 비교기(comparator) 단계를 뒤집는(reverse) 것입니다. 좌측 배열은 시작 부분에 더미 원소가 패딩되고, 우측 배열은 끝 부분에 패딩되며, 더미 원소와의 모든 비교는 생략(elide)됩니다.

`merge-odd` 함수는 먼저 뒤집힌 첫 단계를 수행합니다. $i$를 $0$부터 $\min(\ell_L, \ell_R)$까지 반복하면서 $L_{\ell_L - i - 1}$과 $R_i$를 비교-교환합니다. 입력이 이미 정렬되어 있으므로 이 단계가 바이토닉 병합의 역전된 첫 단계에 해당합니다. 이후 좌측과 우측에 대해 `merge-odd-continue`를 병렬로 재귀 호출합니다.

`merge-odd-continue` 함수에서는 배열 길이 $\ell$이 1보다 큰 경우, $\ell$보다 작은 가장 큰 2의 거듭제곱 $h = 2^{\lceil \log_2 \ell \rceil - 1}$을 계산합니다. 그런 다음 $i = 0$부터 $\ell - h$까지 $x_i$와 $x_{i+h}$를 비교-교환합니다. 이 비교-교환은 워프 셔플의 버터플라이(butterfly) 패턴으로 구현됩니다. 이후 좌측(left)인지 우측(right)인지에 따라 서로 다른 재귀 분할을 수행하는데, 이는 좌측 배열의 더미 원소가 시작 부분에, 우측 배열의 더미 원소가 끝 부분에 있기 때문에 비대칭적 처리가 필요하기 때문입니다.

길이 $\ell_L$과 $\ell_R$인 두 정렬된 배열을 병합하여 길이 $\ell_L + \ell_R$의 정렬된 배열을 만드는 데 필요한 병렬 단계 수는 다음과 같습니다.

$$\lceil \log_2(\max(\ell_L, \ell_R)) \rceil + 1$$

다음 그림은 크기 5와 3인 배열에 대한 홀수 크기 병합 네트워크를 시각적으로 보여줍니다. $\lceil \log_2(\max(5, 3)) \rceil + 1 = \lceil \log_2(5) \rceil + 1 = 3 + 1 = 4$개의 병렬 단계가 필요합니다.

![홀수 크기 네트워크 병합](https://ar5iv.labs.arxiv.org//html/1702.08734/assets/x1.png)

이 그림에서 검은 점(bullet)은 병렬 비교-교환 연산을 나타내며, 점선은 생략된 원소 또는 비교를 의미합니다. 5개 원소와 3개 원소가 4개의 병렬 단계를 거쳐 정렬된 8개 원소의 배열로 병합되는 과정을 확인할 수 있습니다. 더미 원소와의 비교가 생략됨으로써 불필요한 연산을 절약하는 것이 핵심입니다.

비교-교환 연산의 물리적 구현은 레인 스트라이드 레지스터 배열에서 워프 셔플을 사용하여 이루어집니다. 스트라이드가 32의 배수인 교환은 동일 레인 내에서 직접 수행됩니다(해당 레인이 두 원소를 모두 로컬로 보유하므로). 스트라이드가 16 이하이거나 32의 배수가 아닌 교환은 워프 셔플 명령어를 통해 수행됩니다. 실제로 사용되는 배열 길이는 레인 스트라이드 배열에 저장되므로 32의 배수입니다.

```python
import numpy as np

def compare_swap(arr, i, j, ascending=True):
    """비교-교환 프리미티브: 두 위치의 값을 비교하여 필요시 교환"""
    if ascending:
        if arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]
    else:
        if arr[i] < arr[j]:
            arr[i], arr[j] = arr[j], arr[i]

def merge_odd(L, R):
    """
    Algorithm 1: 홀수 크기 병합 네트워크
    이미 정렬된 좌측 배열 L과 우측 배열 R을 병합.
    GPU에서는 각 compare-swap이 병렬로 수행됨.
    """
    ell_L, ell_R = len(L), len(R)
    result = list(L) + list(R)  # 연결
    
    # 1단계: 뒤집힌 첫 비교기 단계 (단조 → 바이토닉 변환)
    for i in range(min(ell_L, ell_R)):
        # L의 끝에서부터 R의 시작과 비교
        idx_L = ell_L - i - 1
        idx_R = ell_L + i
        compare_swap(result, idx_L, idx_R)
    
    # 2단계: 좌측과 우측에 대해 재귀적 병합 계속
    _merge_odd_continue(result, 0, ell_L, 'left')
    _merge_odd_continue(result, ell_L, ell_L + ell_R, 'right')
    
    return result

def _merge_odd_continue(arr, start, end, side):
    """merge-odd-continue: 재귀적 비교-교환 수행"""
    ell = end - start
    if ell <= 1:
        return
    
    # h = 2^(ceil(log2(ell)) - 1): ell보다 작은 가장 큰 2의 거듭제곱
    import math
    h = 2 ** (math.ceil(math.log2(ell)) - 1)
    
    # 스트라이드 h로 비교-교환
    for i in range(ell - h):
        compare_swap(arr, start + i, start + i + h)
    
    # 좌측/우측에 따라 다른 재귀 분할
    if side == 'left':
        _merge_odd_continue(arr, start, start + (ell - h), 'left')
        _merge_odd_continue(arr, start + (ell - h), end, 'right')
    else:
        _merge_odd_continue(arr, start, start + h, 'left')
        _merge_odd_continue(arr, start + h, end, 'right')

# 예시: Figure 1과 동일한 크기 5와 3의 병합
L = [1, 3, 5, 7, 9]   # 이미 정렬됨
R = [2, 4, 6]          # 이미 정렬됨
merged = merge_odd(L, R)
print(f"L={L}, R={R}")
print(f"병합 결과: {merged}")  # [1, 2, 3, 4, 5, 6, 7, 9]
```

#### 홀수 크기 정렬 네트워크

Algorithm 2는 홀수 크기 병합을 확장하여 전체 정렬을 수행합니다. `sort-odd` 함수는 배열을 절반으로 분할하여 각각 재귀적으로 정렬한 후, `merge-odd`로 병합하는 분할 정복(divide-and-conquer) 방식입니다. 입력 데이터에 사전 구조가 없다고 가정할 때, 길이 $\ell$의 데이터를 정렬하는 데 필요한 병렬 단계 수는 다음과 같습니다.

$$\frac{1}{2}(\lceil \log_2(\ell) \rceil^2 + \lceil \log_2(\ell) \rceil)$$

이 공식은 표준 바이토닉 정렬의 올림(ceiling) 버전으로, 2의 거듭제곱이 아닌 크기도 처리할 수 있습니다.

```python
def sort_odd(arr):
    """
    Algorithm 2: 홀수 크기 정렬 네트워크
    임의 길이의 배열을 재귀적으로 정렬.
    """
    ell = len(arr)
    if ell <= 1:
        return arr
    
    mid = ell // 2
    # 재귀적으로 양쪽 절반 정렬 (GPU에서는 병렬 수행)
    left_sorted = sort_odd(arr[:mid])
    right_sorted = sort_odd(arr[mid:])
    
    # 정렬된 두 배열을 홀수 크기 병합으로 합침
    return merge_odd(left_sorted, right_sorted)

# 예시: 비정규 크기 배열 정렬
data = [8, 3, 1, 7, 5, 2, 9]  # 길이 7 (2의 거듭제곱이 아님)
sorted_data = sort_odd(data)
print(f"입력: {data}")
print(f"정렬 결과: {sorted_data}")  # [1, 2, 3, 5, 7, 8, 9]

# 병렬 단계 수 계산
import math
ell = 7
steps = 0.5 * (math.ceil(math.log2(ell))**2 + math.ceil(math.log2(ell)))
print(f"ℓ={ell}일 때 병렬 단계 수: {steps}")  # 0.5*(9+3) = 6
```

### WarpSelect

이제 핵심 알고리즘인 WarpSelect를 상세히 설명합니다. WarpSelect는 모든 상태를 레지스터에 유지하면서 단일 패스로 $k$-선택을 수행하며, 워프 간 동기화(cross-warp synchronization)를 완전히 회피합니다. 앞서 소개한 `merge-odd`와 `sort-odd`를 프리미티브로 활용하며, 레지스터 파일이 공유 메모리보다 훨씬 큰 저장 공간을 제공하므로 $k \leq 1024$까지 지원합니다. 각 워프는 $n$개 배열 $[a_i]$ 중 하나에 대한 $k$-선택에 전담됩니다.

다음 그림은 WarpSelect의 전체 구조를 개관합니다.

![WarpSelect 개요](https://ar5iv.labs.arxiv.org//html/1702.08734/assets/x2.png)

왼쪽에서 입력 값이 스트리밍으로 들어오고, 오른쪽의 워프 큐(warp queue)가 최종 출력 결과를 보유합니다. Algorithm 3의 의사 코드와 함께 이 구조의 핵심 데이터 구조와 동작 원리를 살펴보겠습니다.

#### 데이터 구조

WarpSelect는 두 수준의 계층적 데이터 구조를 유지합니다. 첫째, 각 레인 $j$는 $t$개의 원소로 구성된 작은 큐를 레지스터에 유지하며, 이를 **스레드 큐(thread queue)** $[T_i^j]_{i=0:t}$라 합니다. 이 큐는 가장 큰 값에서 가장 작은 값의 순서로 정렬됩니다. 즉, $T_i^j \geq T_{i+1}^j$입니다. 스레드 큐는 새로 들어오는 값에 대한 **1차 필터** 역할을 합니다. 새로운 값 $a_{32i+j}$가 큐의 최댓값인 $T_0^j$보다 크면, 이 값은 최종 $k$개 최솟값에 포함될 수 없으므로 즉시 거부됩니다.

둘째, 워프 전체가 공유하는 레인 스트라이드 레지스터 배열로 구성된 **워프 큐(warp queue)** $[W_i]_{i=0:k}$가 있으며, 이는 지금까지 관찰된 $k$개의 가장 작은 원소를 유지합니다. 워프 큐는 가장 작은 값에서 가장 큰 값의 순서로 정렬됩니다. 즉, $W_i \leq W_{i+1}$입니다. 요청된 $k$가 32의 배수가 아니면 올림하여 처리합니다. 스레드 큐와 워프 큐 모두 최대 센티널 값(예: $+\infty$)으로 초기화됩니다.

이 두 수준 구조의 설계 직관을 비유하자면, 워프 큐는 "최종 결승전 출전자 명단"이고 스레드 큐는 각 레인의 "예비 후보자 명단"입니다. 새 선수(값)가 도착하면 먼저 자기 레인의 예비 후보 중 가장 약한 선수와 비교되고, 그보다 강하면(값이 작으면) 예비 명단에 올라갑니다. 예비 후보 중 결승 출전자보다 강한 선수가 발견되면, 전체 명단을 재정렬하여 가장 강한 $k$명을 결승 명단에 올립니다.

#### 불변 조건과 업데이트 과정

WarpSelect는 세 가지 불변 조건(invariant)을 유지합니다. 첫째, 모든 레인의 $T_0^j$는 최솟값-$k$에 포함되지 않습니다. 둘째, 모든 레인의 $T_0^j$는 워프 큐의 모든 키 $W_i$보다 큽니다. 셋째, 지금까지 관찰된 모든 $a_i$ 중 최솟값-$k$에 해당하는 것들은 어떤 레인의 스레드 큐 또는 워프 큐에 포함되어 있습니다.

업데이트 과정은 다음과 같이 진행됩니다. 레인 $j$가 새로운 $a_{32i+j}$를 수신하면, 먼저 $a_{32i+j} > T_0^j$인지 확인합니다. 만약 그렇다면 이 값은 정의상 $k$개 최솟값에 포함될 수 없으므로 거부됩니다. 그렇지 않으면 스레드 큐의 적절한 정렬 위치에 삽입되어 기존의 $T_0^j$를 밀어냅니다. 모든 레인이 이 작업을 완료한 후, 두 번째 불변 조건이 위반되었을 가능성이 있습니다.

이 시점에서 **워프 투표(warp ballot)** 명령어를 사용하여 어떤 레인이라도 $T_0^j < W_{k-1}$인지, 즉 두 번째 불변 조건을 위반했는지 확인합니다. 위반이 없으면 다음 원소 처리를 계속합니다. 워프 투표는 32개 레인의 불리언 값을 단일 32비트 마스크로 수집하는 매우 효율적인 워프 수준 통신 프리미티브입니다.

#### 불변 조건의 복원

어떤 레인이라도 불변 조건을 위반하면, 워프는 스레드 큐와 워프 큐를 함께 병합하고 정렬해야 합니다. 이 과정의 핵심 단계를 Algorithm 3의 의사 코드를 따라 설명하겠습니다.

먼저, 스레드 큐 레지스터를 (비정렬된) 레인 스트라이드 배열 $[\alpha_i]_{i=0:32t}$로 재해석합니다. 32개 레인 각각의 $t$개 정렬된 값이 동일 레인 내 서로 다른 레지스터에 보관되어 있으므로, 이를 직접 `merge-odd`로 병합하려면 구조체 배열(struct-of-arrays)에서 배열의 구조체(array-of-structs)로의 전치가 필요합니다. [행렬 내부 전치 분해 기법](https://en.wikipedia.org/wiki/In-place_matrix_transposition)을 통해 이러한 전치가 가능하지만, 비슷한 수의 워프 셔플이 소요되므로 논문에서는 스레드 큐 레지스터를 비정렬 레인 스트라이드 배열로 단순히 재해석한 뒤 `sort-odd`로 처음부터 정렬하는 방식을 채택합니다. 이후 정렬된 스레드 큐 배열과 워프 큐를 `merge-odd`로 병합합니다. 새로운 워프 큐는 병합된 배열의 최솟값-$k$개 원소가 되고, 새로운 스레드 큐는 최솟값-$(k+1)$부터 최솟값-$(k+32t+1)$까지의 원소가 됩니다. 마지막으로 스레드 큐를 역순으로 뒤집어 큰 값에서 작은 값 순서의 불변 조건을 복원합니다.

홀수 크기 병합의 지원이 특히 중요한 이유는, Batcher의 원래 공식에서는 $32t = k$이고 이것이 2의 거듭제곱이어야 하기 때문입니다. 이 경우 $k = 1024$이면 $t$가 반드시 32여야 하는데, 실험적으로 최적의 $t$는 이보다 훨씬 작습니다. 홀수 크기 병합을 사용함으로써 $k$와 $32t$가 서로 다른 임의의 값을 가질 수 있게 되어 파라미터 선택의 유연성이 크게 향상됩니다.

```python
import numpy as np

class WarpSelectSimulator:
    """
    WarpSelect 알고리즘의 동작을 시뮬레이션하는 구현.
    실제 GPU에서는 32개 레인이 병렬로 동작하지만,
    여기서는 개념 이해를 위해 순차적으로 시뮬레이션합니다.
    """
    
    WARP_SIZE = 32
    
    def __init__(self, k, t):
        self.k = k
        self.t = t  # 레인당 스레드 큐 크기
        
        # 워프 큐: k개 최소값 (오름차순), 센티널로 초기화
        self.W = np.full(k, np.inf, dtype=np.float32)
        
        # 스레드 큐: 레인별 t개 원소 (내림차순), 센티널로 초기화
        # T[j][i]: 레인 j의 i번째 원소, T[j][0]이 최대
        self.T = np.full((self.WARP_SIZE, t), np.inf, dtype=np.float32)
    
    def process_element(self, lane_j, value):
        """레인 j에 새 값 삽입 시도"""
        if value >= self.T[lane_j, 0]:
            return False  # 1차 필터에서 거부
        
        # 스레드 큐에 삽입 (내림차순 유지하며 삽입 정렬)
        # T[0]이 최대이므로, value < T[0]이면 삽입
        self.T[lane_j, 0] = value  # 최대값을 대체
        # 적절한 위치로 이동 (내림차순 유지)
        for i in range(self.t - 1):
            if self.T[lane_j, i] < self.T[lane_j, i + 1]:
                self.T[lane_j, i], self.T[lane_j, i + 1] = \
                    self.T[lane_j, i + 1], self.T[lane_j, i]
            else:
                break
        return True
    
    def check_invariant_violation(self):
        """워프 투표: 어떤 레인이라도 T[j][0] < W[k-1]인지 확인"""
        for j in range(self.WARP_SIZE):
            if self.T[j, 0] < self.W[self.k - 1]:
                return True  # 불변 조건 위반 감지
        return False
    
    def restore_invariants(self):
        """스레드 큐를 정렬하고 워프 큐와 병합하여 불변 조건 복원"""
        # 모든 스레드 큐 값을 하나의 배열로 수집
        all_thread_values = self.T.flatten()  # 32*t 개 원소
        all_thread_values.sort()  # sort-odd 시뮬레이션
        
        # 워프 큐와 병합 (merge-odd 시뮬레이션)
        merged = np.sort(np.concatenate([self.W, all_thread_values]))
        
        # 새 워프 큐: 최소 k개
        self.W = merged[:self.k]
        
        # 새 스레드 큐: k+1번째부터 k+32t번째
        remainder = merged[self.k:self.k + self.WARP_SIZE * self.t]
        # 내림차순으로 각 레인에 분배
        remainder_sorted_desc = remainder[::-1]
        for j in range(self.WARP_SIZE):
            for i in range(self.t):
                idx = j * self.t + i
                if idx < len(remainder_sorted_desc):
                    self.T[j, i] = remainder_sorted_desc[idx]
                else:
                    self.T[j, i] = np.inf
    
    def select(self, data):
        """전체 k-선택 수행"""
        n = len(data)
        # 32개씩 그룹으로 처리
        for group_start in range(0, n, self.WARP_SIZE):
            group_end = min(group_start + self.WARP_SIZE, n)
            
            # 각 레인이 자기 원소를 처리 (GPU에서는 병렬)
            for j in range(group_end - group_start):
                self.process_element(j, data[group_start + j])
            
            # 워프 투표로 불변 조건 위반 확인
            if self.check_invariant_violation():
                self.restore_invariants()
        
        # 최종 병합
        self.restore_invariants()
        return np.sort(self.W)[:self.k]

# 사용 예시
np.random.seed(42)
data = np.random.rand(10000).astype(np.float32)
k, t = 10, 2

ws = WarpSelectSimulator(k=k, t=t)
result = ws.select(data)
ground_truth = np.sort(data)[:k]

print(f"WarpSelect 결과:  {result[:5]}...")
print(f"정답 (brute-force): {ground_truth[:5]}...")
print(f"일치 여부: {np.allclose(result, ground_truth)}")
```

#### 나머지 처리와 출력

$\ell$이 32의 배수가 아닌 경우 나머지 원소가 존재하며, 이들은 해당 레인의 스레드 큐에 삽입된 후 출력 단계로 진행됩니다. 최종 출력은 스레드 큐와 워프 큐에 대한 마지막 정렬 및 병합을 수행한 후, 워프 큐에 모든 최솟값-$k$개 값이 담기게 됩니다.

### 복잡도 분석과 파라미터 선택

WarpSelect의 복잡도를 세 가지 상수 시간 연산으로 분해하여 분석할 수 있습니다. 들어오는 32개 원소 그룹마다 1개, 2개, 또는 3개의 연산이 수행되며, 모두 워프 단위 병렬 시간으로 처리됩니다.

첫 번째 연산은 32개 원소를 읽고 모든 스레드 큐 헤드 $T_0^j$와 비교하는 것으로, 비용은 $C_1$이며 $N_1$번 발생합니다. 두 번째 연산은 $a_{32n+j} < T_0^j$인 레인 $j$가 존재할 때 해당 스레드 큐에 삽입 정렬을 수행하는 것으로, 비용은 $C_2 = \mathcal{O}(t)$이며 $N_2$번 발생합니다. 세 번째 연산은 $T_0^j < W_{k-1}$인 레인 $j$가 존재할 때 전체 큐를 정렬하고 병합하는 것으로, 비용은 $C_3 = \mathcal{O}(t \log(32t)^2 + k \log(\max(k, 32t)))$이며 $N_3$번 발생합니다.

따라서 총 비용은 $N_1 C_1 + N_2 C_2 + N_3 C_3$입니다. $N_1 = \ell / 32$이며, 독립적으로 추출된 랜덤 데이터에 대해 $N_2 = \mathcal{O}(k \log(\ell))$이고 $N_3 = \mathcal{O}(k \log(\ell) / t)$입니다. 이 결과는 부록에서 확률 분석과 재귀 관계를 통해 엄밀하게 유도됩니다.

핵심적인 트레이드오프는 $N_2 C_2$와 $N_3 C_3$ 사이의 균형입니다. $t$를 키우면 스레드 큐가 더 많은 원소를 필터링하여 비용이 큰 전체 병합 횟수 $N_3$가 줄어들지만, 각 삽입 정렬 비용 $C_2 = \mathcal{O}(t)$와 전체 정렬 비용 $C_3$가 증가합니다. 반대로 $t$를 줄이면 삽입 비용은 감소하지만 병합이 더 자주 발생합니다.

실험적으로 결정된 $t$의 최적값은 $k$ 범위에 따라 다음과 같습니다. $k \leq 32$일 때 $t = 2$, $k \leq 128$일 때 $t = 3$, $k \leq 256$일 때 $t = 4$, $k \leq 1024$일 때 $t = 8$이며, 모두 $\ell$과 무관합니다. 주목할 점은 $t$의 값이 매우 작다는 것입니다. 예를 들어 $k = 1024$에서도 $t = 8$이면 각 레인은 8개의 레지스터만 스레드 큐에 사용합니다. Batcher의 원래 공식에서는 $32t = k$여야 하므로 $t = 32$가 필요했지만, 홀수 크기 병합을 통해 $t = 8$로 대폭 줄일 수 있으며, 이는 레지스터 사용량 절감과 점유율 향상에 직결됩니다.

```python
def estimate_warpselect_cost(ell, k, t):
    """
    WarpSelect의 예상 비용 분석.
    랜덤 데이터에 대한 이론적 복잡도 기반 추정.
    """
    import math
    
    N1 = ell / 32                                    # 항상 발생
    N2 = k * math.log2(ell) if ell > 1 else k       # O(k log ℓ) 삽입 정렬
    N3 = k * math.log2(ell) / t if ell > 1 else k/t # O(k log ℓ / t) 전체 병합
    
    C1 = 1                                           # 비교만 수행
    C2 = t                                           # O(t) 삽입 정렬
    # C3: sort-odd + merge-odd
    C3_sort = t * (math.log2(32 * t))**2 if 32*t > 1 else 0
    C3_merge = k * math.log2(max(k, 32 * t)) if max(k, 32*t) > 1 else 0
    C3 = C3_sort + C3_merge
    
    total = N1 * C1 + N2 * C2 + N3 * C3
    
    return {
        "N1": int(N1), "C1": C1, "N1*C1": N1 * C1,
        "N2_approx": int(N2), "C2": C2, "N2*C2": N2 * C2,
        "N3_approx": int(N3), "C3_approx": int(C3), "N3*C3": N3 * C3,
        "total_approx": total,
        "dominant_term": "읽기(N1C1)" if N1*C1 > N2*C2 + N3*C3 
                         else "삽입(N2C2)" if N2*C2 > N3*C3 else "병합(N3C3)"
    }

# 파라미터 선택 테이블 재현
param_table = {32: 2, 128: 3, 256: 4, 1024: 8}
ell = 100000

print(f"{'k':>6} {'t':>4} {'N1':>8} {'N2~':>8} {'N3~':>8} {'지배적 비용':>12}")
print("-" * 55)
for k_val, t_val in param_table.items():
    result = estimate_warpselect_cost(ell, k_val, t_val)
    print(f"{k_val:>6} {t_val:>4} {result['N1']:>8} "
          f"{result['N2_approx']:>8} {result['N3_approx']:>8} "
          f"{result['dominant_term']:>12}")
```

이 분석에서 가장 중요한 통찰은, 충분히 큰 $\ell$에서 $N_1 C_1$ 항이 지배적이 된다는 것입니다. 이는 WarpSelect의 비용이 입력을 한 번 스캔하는 비용에 수렴함을 의미하며, 앞서 루프라인 모델에서 제시한 이론적 하한에 근접하는 결과입니다. 삽입과 병합 연산은 $\ell$이 커질수록 상대적으로 미미해지므로, WarpSelect는 대규모 데이터에서 메모리 대역폭 제한(memory-bound) 성능에 가까워집니다.
# 연산 레이아웃

이 절에서는 앞서 소개한 Product Quantization 기반 인덱싱 방법인 IVFADC가 GPU에서 어떻게 효율적으로 구현되는지를 상세히 설명합니다. 거리 계산의 구체적인 분해 방식과 $k$-선택과의 결합이 핵심이며, 이러한 설계가 더 최근의 GPU 호환 근사 최근접 이웃 전략을 능가할 수 있는 이유를 이해하는 데 필수적입니다.

## 정확 탐색의 GPU 구현

앞서 정의한 전수 탐색(exhaustive search) 방법의 GPU 구현을 먼저 살펴봅니다. 이 방법은 소규모 데이터셋에서의 정확 최근접 이웃 탐색에 자체적으로 유용할 뿐만 아니라, IVFADC의 조대 양자화기 $q_1$에서도 핵심 구성 요소로 활용됩니다.

앞서 설명한 바와 같이 거리 계산은 행렬 곱셈으로 귀결됩니다. 구현에서는 cuBLAS 라이브러리의 최적화된 GEMM(General Matrix Multiply) 루틴을 사용하여 L2 거리의 $-2\langle x_j, y_i \rangle$ 항을 계산하며, 이를 통해 부분 거리 행렬 $D'$을 얻습니다. 거리 계산을 완료하기 위해 **융합(fused) $k$-선택 커널**이 사용되는데, 이 커널은 거리 행렬의 각 항에 $\|y_i\|^2$ 항을 더한 뒤 즉시 그 값을 레지스터 내 $k$-선택에 제출합니다. 한편 $\|x_j\|^2$ 항은 $k$-선택 이전에 고려할 필요가 없습니다. 이는 동일한 쿼리 $x_j$에 대해 모든 데이터베이스 벡터에 동일한 상수값이 더해지므로 상대적 순위에 영향을 미치지 않기 때문입니다.

이러한 커널 융합 전략의 핵심 이점은 $D'$에 대해 단 **2회의 패스**(GEMM 쓰기 1회, $k$-선택 읽기 1회)만 필요하다는 것입니다. 다른 구현에서는 3회 이상의 패스가 필요할 수 있습니다. 행 단위(row-wise) $k$-선택을 잘 튜닝된 GEMM 커널에 직접 융합하는 것은 불가능하거나, 가능하더라도 전체 효율이 낮아질 수 있기 때문에 이 2-패스 접근법이 채택되었습니다.

```python
import numpy as np

def gpu_exact_search_fused_layout(X, Y, k, tile_size=1024):
    """
    GPU 정확 탐색의 연산 레이아웃 시뮬레이션.
    핵심: GEMM → 융합 k-선택으로 2-패스만 수행.
    
    X: (n_q, d) 쿼리, Y: (ell, d) 데이터베이스
    """
    n_q, d = X.shape
    ell = Y.shape[0]
    
    # ||y_i||^2 사전 계산 (1회, 재사용)
    y_norms_sq = np.sum(Y ** 2, axis=1)  # (ell,)
    
    all_results = []
    
    # 타일링: GPU 메모리에 D'이 맞지 않으므로 t_q개씩 처리
    for tile_start in range(0, n_q, tile_size):
        t_q = min(tile_size, n_q - tile_start)
        X_tile = X[tile_start:tile_start + t_q]  # (t_q, d)
        
        # 패스 1: cuBLAS GEMM으로 D' = -2 * X_tile @ Y^T 계산
        # 메모리: O(t_q * ell) — GPU 글로벌 메모리에 기록
        D_prime = -2.0 * (X_tile @ Y.T)  # (t_q, ell)
        
        # 패스 2: 융합 커널 — D'을 읽으면서 ||y_i||^2을 더하고
        #          즉시 k-선택에 제출 (별도 저장 불필요!)
        for q in range(t_q):
            distances = D_prime[q] + y_norms_sq  # ||y_i||^2만 추가
            # ||x_j||^2는 상수이므로 k-선택 전에 불필요
            # (순위에 영향 없음)
            top_k_idx = np.argpartition(distances, k)[:k]
            top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]
            all_results.append(top_k_idx)
    
    return all_results

# 메모리 복잡도 분석
def memory_analysis(n_q, ell, tile_size, num_streams=2):
    """
    타일링과 멀티 스트리밍의 메모리 사용량 분석.
    2개 스트림 병렬 실행 시: O(2 * ell * t_q) 부동소수점
    """
    t_q = min(tile_size, n_q)
    num_tiles = (n_q + t_q - 1) // t_q  # ceil(n_q / t_q)
    
    # 유효 메모리 요구량: 2개 타일 동시 실행
    effective_memory = num_streams * ell * t_q * 4  # float32 = 4 bytes
    
    return {
        "타일_수": num_tiles,
        "타일당_쿼리": t_q,
        "유효_메모리_GB": effective_memory / 1e9,
        "전체_D_메모리_GB": n_q * ell * 4 / 1e9,
        "메모리_절감_비율": f"{n_q * ell / (num_streams * ell * t_q):.1f}x"
    }

# 예시: 10000 쿼리, 100만 데이터베이스
result = memory_analysis(n_q=10000, ell=1_000_000, tile_size=512)
print(f"전체 D 저장 시: {result['전체_D_메모리_GB']:.1f} GB")
print(f"타일링 후: {result['유효_메모리_GB']:.1f} GB")
print(f"메모리 절감: {result['메모리_절감_비율']}")
```

현실적 문제 크기에서 $D'$은 GPU 메모리에 적재되지 않으므로, 쿼리 배치에 대한 **타일링(tiling)**이 필수적입니다. $t_q \leq n_q$개의 쿼리를 단일 타일에서 처리하며, $\lceil n_q / t_q \rceil$개의 타일 각각은 독립적 문제입니다. GPU를 더 효과적으로 점유하기 위해 2개의 타일을 서로 다른 스트림(stream)에서 병렬로 실행하므로, $D$의 유효 메모리 요구량은 $\mathcal{O}(2\ell t_q)$가 됩니다. 계산은 $\ell$ 방향으로도 유사하게 타일링할 수 있습니다. CPU에서 대용량 입력이 전송되는 경우, 고정 메모리(pinned memory)를 사용한 버퍼링으로 CPU→GPU 복사와 GPU 연산을 중첩(overlap)시킬 수 있습니다.

## IVFADC 인덱싱

### PQ 룩업 테이블

IVFADC의 핵심에는 벡터에서 Product Quantization 재생산 값들까지의 거리를 계산하는 과정이 있습니다. 앞서 정의한 2단계 양자화 근사를 전개하면 다음과 같은 거리 공식을 얻습니다.

$$\|x - q(y)\|_2^2 = \|x - q_1(y) - q_2(y - q_1(y))\|_2^2$$

이 수식은 데이터베이스 벡터 $y$에 대한 근사 거리를 두 양자화기 $q_1$과 $q_2$의 출력으로 표현합니다. $q_1(y)$이 조대 수준 근사를 제공하고, $q_2(y - q_1(y))$가 잔차(residual)를 보정합니다. 이제 $q_1$ 이후의 잔차 벡터를 $b$개의 부분벡터(sub-vector)로 분해합니다.

$$y - q_1(y) = [\widetilde{y^1} \cdots \widetilde{y^b}] \quad \text{and} \quad x - q_1(y) = [\widetilde{x^1} \cdots \widetilde{x^b}]$$

이렇게 분해하면 거리를 각 부분벡터별 독립적인 거리의 합으로 다시 쓸 수 있습니다.

$$\|x - q(y)\|_2^2 = \|\widetilde{x^1} - q^1(\widetilde{y^1})\|_2^2 + \ldots + \|\widetilde{x^b} - q^b(\widetilde{y^b})\|_2^2$$

각 부분 양자화기 $q^1, \ldots, q^b$는 256개의 재생산 값을 가지므로, $x$와 $q_1(y)$가 알려진 상태에서 모든 가능한 부분 거리를 사전 계산하여 크기 256인 룩업 테이블 $T_1, \ldots, T_b$에 저장할 수 있습니다. 이후 특정 벡터 $y$에 대한 거리 계산은 $b$번의 테이블 조회(lookup)와 덧셈만으로 완료됩니다. 이것이 Product Quantizer의 효율성의 근간입니다.

$n$개 벡터에 대한 거리 계산 비용을 비교하면 그 차이가 극명합니다. 명시적 계산(explicit computation)은 $n \times d$번의 곱셈-덧셈이 필요하지만, 룩업 테이블 방식에서는 테이블 구성에 $256 \times d$번의 곱셈-덧셈, 그리고 실제 거리 계산에 $n \times b$번의 조회-덧셈만 소요됩니다. $b \ll d$이고 $n \gg 256$인 전형적 상황에서 이 절감은 막대합니다.

```python
import numpy as np

def pq_lookup_table_distance(query, q1_centroid, pq_codebooks, pq_codes, b):
    """
    PQ 룩업 테이블을 이용한 거리 계산의 구체적 구현.
    
    query: (d,) 쿼리 벡터
    q1_centroid: (d,) 조대 양자화기의 센트로이드
    pq_codebooks: b개의 (256, d/b) 코드북
    pq_codes: (n, b) 데이터베이스 벡터들의 PQ 코드 (uint8)
    """
    d = query.shape[0]
    sub_dim = d // b
    n = pq_codes.shape[0]
    
    # 잔차 벡터: x~ = x - q1(y)
    residual_query = query - q1_centroid  # (d,)
    
    # 1단계: 룩업 테이블 구성 — 256 × d 곱셈-덧셈
    # 각 부분벡터에 대해 256개 센트로이드와의 거리 사전 계산
    tables = []  # T_1, ..., T_b
    for j in range(b):
        x_sub = residual_query[j * sub_dim:(j + 1) * sub_dim]  # x_tilde^j
        # 256개 재생산 값과의 L2 거리 사전 계산
        T_j = np.sum((pq_codebooks[j] - x_sub) ** 2, axis=1)  # (256,)
        tables.append(T_j)
    # 총 비용: 256 × d 곱셈-덧셈
    
    # 2단계: n개 벡터의 거리 계산 — n × b 조회-덧셈만!
    distances = np.zeros(n, dtype=np.float32)
    for i in range(n):
        for j in range(b):
            code = pq_codes[i, j]  # uint8: 0~255
            distances[i] += tables[j][code]  # 단순 조회 + 덧셈
    # 총 비용: n × b 조회-덧셈
    
    return distances

# 비용 비교 시연
def cost_comparison(n, d, b):
    """명시적 계산 vs 룩업 테이블 비용 비교"""
    explicit_cost = n * d               # 곱셈-덧셈
    table_build = 256 * d               # 테이블 구성
    table_lookup = n * b                # 조회-덧셈
    lookup_total = table_build + table_lookup
    
    return {
        "명시적_계산": f"{explicit_cost:,} 연산",
        "룩업_테이블_구성": f"{table_build:,} 연산",
        "룩업_조회": f"{table_lookup:,} 연산",
        "룩업_총합": f"{lookup_total:,} 연산",
        "속도_향상": f"{explicit_cost / lookup_total:.1f}x"
    }

# 전형적 설정: 100만 벡터, 128차원, 8바이트 PQ 코드
result = cost_comparison(n=1_000_000, d=128, b=8)
for k, v in result.items():
    print(f"  {k}: {v}")
# 명시적: 128,000,000 vs 룩업: 8,032,768 → ~16x 속도 향상!
```

GPU 구현에서 $b$는 4의 배수이며 최대 64까지 지원됩니다. PQ 코드는 역색인 리스트 내에서 벡터당 $b$바이트의 순차적 그룹으로 저장됩니다.

### IVFADC 룩업 테이블의 최적화

역색인 리스트 $\mathcal{I}_L$의 원소를 스캔할 때, 정의에 의해 $q_1(y)$가 상수이므로 위의 룩업 테이블 방법을 직접 적용할 수 있습니다. 그러나 테이블 $T_1 \ldots T_b$의 계산을 더욱 최적화할 수 있습니다. 앞서 정의한 근사 거리 $\|x - q(y)\|_2^2$를 다음 세 항으로 분해합니다.

$$\underbrace{\|q_2(\ldots)\|_2^2 + 2\langle q_1(y), q_2(\ldots)\rangle}_{\text{term 1}} + \underbrace{\|x - q_1(y)\|_2^2}_{\text{term 2}} - 2\underbrace{\langle x, q_2(\ldots)\rangle}_{\text{term 3}}$$

이 분해의 목표는 내부 루프(inner loop)의 계산량을 최소화하는 것입니다. 각 항의 사전 계산 가능성을 분석하면 다음과 같습니다. **Term 1**은 쿼리와 무관하게 양자화기만으로부터 사전 계산할 수 있으며, 크기 $|\mathcal{C}_1| \times 256 \times b$인 테이블 $\mathcal{T}$에 저장됩니다. **Term 2**는 $q_1$의 재생산 값까지의 거리로, 1단계 양자화기 $q_1$의 부산물(by-product)입니다. **Term 3**은 역색인 리스트와 무관하게 계산 가능하며, $d \times 256$번의 곱셈-덧셈이 소요됩니다.

단일 쿼리에 대해 $\tau \times b$개의 테이블을 처음부터 계산하면 $\tau \times d \times 256$번의 곱셈-덧셈이 필요합니다. 반면 이 분해를 사용하면 $256 \times d$번의 곱셈-덧셈과 $\tau \times b \times 256$번의 덧셈으로 충분합니다. $\tau$가 크고 $b \ll d$인 상황에서 이 절감은 상당합니다. 다만 GPU에서 $\mathcal{T}$의 메모리 사용량이 과도할 수 있으므로, 메모리가 충분할 때만 이 분해를 활성화합니다.

```python
import numpy as np

def ivfadc_lookup_table_optimization(query, coarse_centroids, pq_codebooks,
                                      tau_lists, b, d, use_decomposition=True):
    """
    IVFADC 룩업 테이블 최적화: 3-항 분해 vs 직접 계산 비교.
    
    Parameters:
        query: (d,) 쿼리 벡터
        coarse_centroids: 선택된 tau개의 조대 센트로이드
        pq_codebooks: b개의 (256, d/b) 코드북 리스트
        tau_lists: 탐색할 역색인 리스트 인덱스들
    """
    tau = len(tau_lists)
    sub_dim = d // b
    
    if use_decomposition:
        # === 3-항 분해 방식 ===
        
        # Term 3: 쿼리와 q2 재생산 값의 내적 (역색인 리스트 무관)
        # 비용: 256 × d 곱셈-덧셈 (1회만!)
        term3_tables = []
        for j in range(b):
            x_sub = query[j * sub_dim:(j + 1) * sub_dim]
            # -2 * <x, q2(...)> 의 부분벡터 기여분
            inner_products = pq_codebooks[j] @ x_sub  # (256,)
            term3_tables.append(-2.0 * inner_products)
        
        # 각 역색인 리스트에 대한 최종 테이블 조립
        # 비용: tau × b × 256 덧셈만 추가
        all_tables = {}
        for list_idx in tau_lists:
            centroid = coarse_centroids[list_idx]
            
            # Term 2: ||x - q1(y)||^2 (q1의 부산물, 스칼라)
            term2 = np.sum((query - centroid) ** 2)
            
            # Term 1 + Term 3 → 최종 테이블 T_1, ..., T_b
            tables = []
            for j in range(b):
                c_sub = centroid[j * sub_dim:(j + 1) * sub_dim]
                # Term 1: ||q2||^2 + 2<q1, q2> (사전 계산 테이블 T에서 조회)
                term1 = (np.sum(pq_codebooks[j] ** 2, axis=1) + 
                        2.0 * pq_codebooks[j] @ c_sub)
                # 최종 테이블: term1 + term3 (term2는 상수로 별도 추가)
                tables.append(term1 + term3_tables[j])
            all_tables[list_idx] = (tables, term2)
        
        return all_tables
    
    else:
        # === 직접 계산 방식 ===
        # 비용: tau × d × 256 곱셈-덧셈
        all_tables = {}
        for list_idx in tau_lists:
            centroid = coarse_centroids[list_idx]
            residual = query - centroid
            tables = []
            for j in range(b):
                x_sub = residual[j * sub_dim:(j + 1) * sub_dim]
                T_j = np.sum((pq_codebooks[j] - x_sub) ** 2, axis=1)
                tables.append(T_j)
            all_tables[list_idx] = (tables, 0.0)
        return all_tables

# 비용 비교
def decomposition_cost_comparison(tau, d, b):
    """분해 방식 vs 직접 방식의 비용 비교"""
    direct = tau * d * 256
    decomp_mulads = 256 * d
    decomp_adds = tau * b * 256
    
    print(f"tau={tau}, d={d}, b={b}")
    print(f"  직접 계산: {direct:,} 곱셈-덧셈")
    print(f"  분해 방식: {decomp_mulads:,} 곱셈-덧셈 + {decomp_adds:,} 덧셈")
    print(f"  = 총 {decomp_mulads + decomp_adds:,} 연산")
    print(f"  절감율: {direct / (decomp_mulads + decomp_adds):.1f}x")

decomposition_cost_comparison(tau=64, d=128, b=8)
# tau=64일 때: 직접 2,097,152 vs 분해 163,840 → ~12.8x 절감
```

## GPU 구현 세부사항

### 리스트 스캔과 공유 메모리

역색인 리스트는 PQ 코드와 관련 ID를 위한 두 개의 분리된 배열로 저장됩니다. ID는 $k$-선택이 $k$-최근접 소속 여부를 결정한 경우에만 해석(resolve)되며, 이 조회는 대용량 배열에서 소수의 희소 메모리 읽기만 발생시키므로 ID를 CPU 메모리에 저장해도 성능 저하가 미미합니다.

리스트 스캔 커널은 각 쿼리에 대해 가장 가까운 $\tau$개의 역색인 리스트를 스캔하고 룩업 테이블 $T_i$를 사용하여 벡터 쌍 거리를 계산합니다. $T_i$는 **공유 메모리(shared memory)**에 저장됩니다. 전체 쿼리 세트에 대해 최대 $n_q \times \tau \times \max_i |\mathcal{I}_i| \times b$번의 조회가 필요하며, 실전에서는 조(trillion) 단위의 접근이 발생하고 이들은 모두 랜덤 액세스입니다. 공유 메모리의 48 KiB 제한으로 인해 $b$는 32비트 부동소수점 기준 최대 48, 16비트 부동소수점 기준 최대 96까지 지원됩니다.
앞서 설명한 리스트 스캔 커널에서 룩업 테이블 $T_i$가 공유 메모리에 저장되며, 수식 분해를 사용하지 않는 경우 $T_i$는 스캔 이전에 별도의 커널에서 계산된다는 점을 추가로 주목할 필요가 있습니다.

### 다중 패스 커널

$n_q \times \tau$개의 쿼리-역색인 리스트 쌍은 각각 독립적으로 처리될 수 있습니다. 한쪽 극단에서는 각 쌍에 하나의 블록을 전담시키는 방식이 있으며, 이 경우 최대 $n_q \times \tau \times \max_i |\mathcal{I}_i|$개의 부분 결과가 글로벌 메모리에 기록된 후, 이를 대상으로 $k$-선택을 수행하여 $n_q \times k$개의 최종 결과를 얻습니다. 이 방식은 높은 병렬성을 제공하지만, 글로벌 메모리 용량을 초과할 수 있습니다. 앞서 정확 탐색에서 도입한 것과 동일하게 타일 크기 $t_q \leq n_q$를 선택하여 메모리 소비를 제한하며, 멀티 스트리밍을 적용하면 메모리 복잡도는 $\mathcal{O}(2 t_q \tau \max_i |\mathcal{I}_i|)$로 한정됩니다.

단일 워프가 각 $t_q$개 쿼리 세트의 리스트에 대한 $k$-선택을 전담하면 병렬성이 낮아질 수 있습니다. 이를 해결하기 위해 **2-패스 $k$-선택(two-pass $k$-selection)**이 도입됩니다. 첫 번째 패스에서 $t_q \times \tau \times \max_i |\mathcal{I}_i|$개의 부분 결과를 특정 분할 인자(subdivision factor) $f$를 사용하여 $t_q \times f \times k$개로 축소합니다. 두 번째 패스에서 이를 다시 $k$-선택하여 최종 $t_q \times k$개의 결과를 얻습니다. 이 계층적 축소 전략은 병렬성과 메모리 소비 사이의 균형을 효과적으로 달성합니다.

```python
import numpy as np

def two_pass_k_selection(partial_results, t_q, tau, max_list_size, k, f):
    """
    2-패스 k-선택의 개념적 시뮬레이션.
    
    1단계: t_q × τ × max|I_i| → t_q × f × k (f개 그룹으로 분할)
    2단계: t_q × f × k → t_q × k (최종 결과)
    """
    # 1단계: 각 쿼리의 부분 결과를 f개 그룹으로 분할하여 k-선택
    # (GPU에서는 각 그룹이 별도 블록에서 병렬 처리)
    first_pass_results = np.zeros((t_q, f * k))
    for q in range(t_q):
        query_results = partial_results[q]  # 해당 쿼리의 모든 부분 결과
        chunk_size = len(query_results) // f
        for g in range(f):
            chunk = query_results[g * chunk_size:(g + 1) * chunk_size]
            # 각 그룹에서 k개 최소값 선택
            top_k = np.partition(chunk, k)[:k]
            first_pass_results[q, g * k:(g + 1) * k] = np.sort(top_k)
    
    # 2단계: f × k개에서 최종 k개 선택
    final_results = np.zeros((t_q, k))
    for q in range(t_q):
        candidates = first_pass_results[q]  # f × k개 후보
        final_results[q] = np.sort(np.partition(candidates, k)[:k])
    
    return final_results

# 메모리 사용량 비교
def memory_comparison(t_q, tau, max_list_size, k, f):
    """다중 패스 커널의 메모리 사용량 분석"""
    # 단일 패스: 모든 부분 결과 저장
    single_pass_mem = t_q * tau * max_list_size * 4  # float32
    # 2-패스: 1단계 축소 후 저장
    two_pass_mem = t_q * f * k * 4
    
    print(f"단일 패스 메모리: {single_pass_mem / 1e6:.1f} MB")
    print(f"2-패스 메모리: {two_pass_mem / 1e6:.1f} MB")
    print(f"메모리 절감: {single_pass_mem / two_pass_mem:.0f}x")

# 전형적 설정
memory_comparison(t_q=512, tau=64, max_list_size=10000, k=100, f=32)
# 단일 패스: ~1.3 GB vs 2-패스: ~6.6 MB → ~200x 절감
```

### 융합 커널의 시도와 한계

정확 탐색에서 성공적이었던 커널 융합 전략을 IVFADC에도 적용하는 실험이 수행되었습니다. 이 융합 커널은 단일 블록이 단일 쿼리에 대한 $\tau$개의 모든 리스트를 스캔하면서, 거리 계산과 $k$-선택을 동시에 수행하는 구조입니다. 앞서 소개한 WarpSelect는 공유 메모리 자원을 경쟁하지 않으므로 이러한 융합이 원칙적으로 가능합니다. 이 방식은 거의 모든 중간 결과를 즉시 제거할 수 있어 글로벌 메모리 기록(write-back)을 크게 줄입니다.

그러나 정확 탐색에서의 $k$-선택 오버헤드와 달리, IVFADC에서는 런타임의 상당 부분이 공유 메모리의 $T_i$로부터의 수집(gather)과 글로벌 메모리의 역색인 리스트 $\mathcal{I}_i$에 대한 선형 스캔에 소비됩니다. 따라서 기록 비용(write-back)이 지배적 기여자가 아니며, 융합 커널의 성능 개선은 최대 **15%**에 그칩니다. 일부 문제 크기에서는 낮은 병렬성으로 인해 오히려 성능이 저하될 수 있으므로, 구현 단순성을 고려하여 이 레이아웃은 최종적으로 채택되지 않았습니다.

### IVFPQ 배치 탐색 루틴

Algorithm 4는 전체 IVFADC(IVFPQ) 배치 탐색 과정을 요약합니다. 이 알고리즘은 쿼리 벡터 집합 $[x_1, \ldots, x_{n_q}]$과 역색인 리스트 $\mathcal{I}_1, \ldots, \mathcal{I}_{|\mathcal{C}_1|}$을 입력으로 받아 다음 과정을 수행합니다. 먼저 모든 쿼리에 대해 앞서 설명한 정확 탐색의 배치 양자화를 수행하여 $L_{\text{IVF}}^i = \tau\text{-}\textrm{argmin}_{c \in \mathcal{C}_1} \|x - c\|_2$를 계산합니다. 이후 각 쿼리 $i$에 대해 거리 테이블의 Term 3을 계산하고, $L_{\text{IVF}}^i$에 속하는 $\tau$개의 리스트 각각에 대해 룩업 테이블 $T_1, \ldots, T_b$를 구성합니다. 구성된 테이블을 사용하여 각 역색인 리스트 $\mathcal{I}_L$ 내의 벡터들에 대해 앞서 정의한 부분벡터 거리 합산 공식으로 $d = \|x_i - q(y_j)\|_2^2$를 추정하고, 결과 튜플 $(d, L, j)$를 수집합니다. 최종적으로 수집된 모든 거리에 대해 $k$-선택을 수행하여 각 쿼리의 $k$-최근접 이웃 $R_i$를 반환합니다.

```python
import numpy as np

def ivfpq_batch_search(queries, coarse_centroids, inverted_lists, 
                        pq_codebooks, tau, k, b):
    """
    Algorithm 4: IVFPQ 배치 탐색 루틴의 완전한 구현.
    GPU에서는 각 단계가 개별 커널로 실행됨.
    
    queries: (n_q, d) 쿼리 행렬
    coarse_centroids: (|C1|, d) 조대 센트로이드
    inverted_lists: {list_idx: [(vector_id, pq_code), ...]}
    pq_codebooks: b개의 (256, d/b) 코드북
    """
    n_q, d = queries.shape
    sub_dim = d // b
    results = []
    
    # ========================================
    # 단계 1: 배치 양자화 (정확 탐색 커널 사용)
    # GPU: cuBLAS GEMM + 융합 k-선택
    # ========================================
    L_IVF = np.zeros((n_q, tau), dtype=np.int32)
    for i in range(n_q):
        dists = np.linalg.norm(coarse_centroids - queries[i], axis=1)
        L_IVF[i] = np.argsort(dists)[:tau]
    
    # ========================================
    # 단계 2: 각 쿼리별 리스트 스캔
    # ========================================
    for i in range(n_q):
        candidates = []  # (distance, list_idx, vector_idx) 튜플
        
        # Term 3 사전 계산: 256 × d 곱셈-덧셈 (1회)
        term3_tables = []
        for j in range(b):
            x_sub = queries[i, j * sub_dim:(j + 1) * sub_dim]
            term3_tables.append(-2.0 * pq_codebooks[j] @ x_sub)
        
        # tau개 역색인 리스트 순회
        for L in L_IVF[i]:
            centroid = coarse_centroids[L]
            
            # 룩업 테이블 T_1,...,T_b 구성
            # GPU: 공유 메모리에 저장 (48 KiB 제한)
            tables = []
            for j in range(b):
                c_sub = centroid[j * sub_dim:(j + 1) * sub_dim]
                term1 = (np.sum(pq_codebooks[j] ** 2, axis=1) +
                        2.0 * pq_codebooks[j] @ c_sub)
                tables.append(term1 + term3_tables[j])
            
            # 리스트 내 벡터 스캔: n × b 조회-덧셈
            for (vec_id, pq_code) in inverted_lists.get(L, []):
                dist = sum(tables[j][pq_code[j]] for j in range(b))
                candidates.append((dist, L, vec_id))
        
        # k-선택: 가장 작은 k개 거리의 벡터 반환
        # GPU: WarpSelect 또는 2-패스 k-선택
        candidates.sort(key=lambda x: x[0])
        R_i = [(c[2], c[0]) for c in candidates[:k]]
        results.append(R_i)
    
    return results
```

이 코드에서 GPU 구현의 각 단계가 명확히 구분됩니다. 단계 1의 배치 양자화는 cuBLAS GEMM과 융합 $k$-선택으로 처리되고, 단계 2의 리스트 스캔에서는 룩업 테이블이 공유 메모리에 적재되어 수조 회의 랜덤 액세스를 효율적으로 처리합니다.

## 다중 GPU 병렬화

현대 서버는 여러 GPU를 탑재할 수 있으며, 이를 연산 능력과 메모리 용량 모두를 위해 활용할 수 있습니다. 본 논문은 **복제(replication)**와 **샤딩(sharding)**이라는 두 가지 분배 전략을 제시합니다.

### 복제

인덱스 인스턴스가 단일 GPU 메모리에 적재 가능한 경우, $\mathcal{R}$개의 서로 다른 GPU에 동일한 인덱스를 복제할 수 있습니다. $n_q$개의 쿼리를 처리할 때 각 복제본은 $n_q / \mathcal{R}$개의 쿼리만 담당하며, 결과를 단일 GPU 또는 CPU 메모리에서 합칩니다. 복제 방식은 거의 선형적인 속도 향상을 달성하지만, $n_q$가 작은 경우에는 분배 오버헤드로 인해 효율이 약간 저하될 수 있습니다.

### 샤딩

인덱스가 단일 GPU 메모리에 적재되지 않는 경우, $\mathcal{S}$개의 GPU에 인덱스를 분할(shard)할 수 있습니다. 벡터 추가 시 각 샤드는 $\ell / \mathcal{S}$개의 벡터를 수신하고, 쿼리 시에는 각 샤드가 전체 $n_q$개의 쿼리를 처리합니다. 부분 결과를 합치기 위해 추가적인 $k$-선택 라운드가 필요합니다. 동일한 인덱스 크기 $\ell$에서 샤딩은 속도 향상을 제공하지만(각 샤드가 $\ell / \mathcal{S}$에 대해 $n_q$를 처리하므로), 고정 오버헤드와 후속 $k$-선택 비용으로 인해 순수 복제보다는 통상 낮은 성능을 보입니다.

```python
def multi_gpu_strategy_analysis(n_q, ell, num_gpus, gpu_memory_gb, 
                                 index_size_gb, query_time_single):
    """
    복제 vs 샤딩 vs 결합 전략의 성능 분석.
    """
    strategies = {}
    
    # 전략 1: 순수 복제 (인덱스가 단일 GPU에 적재 가능할 때)
    if index_size_gb <= gpu_memory_gb:
        R = num_gpus
        queries_per_replica = n_q / R
        # 거의 선형 속도 향상, 소규모 n_q에서 약간의 손실
        speedup = R * min(1.0, queries_per_replica / 100)  # 임계값 모델
        strategies["복제"] = {
            "설정": f"R={R} 복제본",
            "GPU당_쿼리": int(queries_per_replica),
            "GPU당_데이터": f"{ell:,} 벡터 (전체)",
            "추정_속도향상": f"{speedup:.1f}x",
            "추가_k선택": "불필요"
        }
    
    # 전략 2: 순수 샤딩 (인덱스가 클 때)
    S = num_gpus
    vecs_per_shard = ell // S
    # 샤딩: 각 GPU가 전체 쿼리를 처리하되 데이터 축소
    # 추가 k-선택 오버헤드 존재
    shard_overhead = 0.85  # 고정 오버헤드 + 후속 k-선택
    speedup_shard = S * shard_overhead
    strategies["샤딩"] = {
        "설정": f"S={S} 샤드",
        "GPU당_쿼리": f"{n_q:,} (전체)",
        "GPU당_데이터": f"{vecs_per_shard:,} 벡터",
        "추정_속도향상": f"{speedup_shard:.1f}x",
        "추가_k선택": "필요 (부분 결과 병합)"
    }
    
    # 전략 3: 결합 (S 샤드 × R 복제본)
    if num_gpus >= 4:
        S, R = 2, num_gpus // 2
        strategies["결합"] = {
            "설정": f"S={S} 샤드 × R={R} 복제본 = {S*R} GPU",
            "GPU당_쿼리": f"{n_q // R:,}",
            "GPU당_데이터": f"{ell // S:,} 벡터",
            "추정_속도향상": f"{S * R * 0.8:.1f}x",
            "추가_k선택": "부분적 필요"
        }
    
    return strategies

# 십억 규모 시나리오 분석
results = multi_gpu_strategy_analysis(
    n_q=10000, ell=1_000_000_000, num_gpus=4,
    gpu_memory_gb=12, index_size_gb=16,
    query_time_single=1.0
)
for name, info in results.items():
    print(f"\n=== {name} 전략 ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
```

복제와 샤딩은 결합하여 사용할 수도 있습니다. $\mathcal{S}$개의 샤드 각각에 $\mathcal{R}$개의 복제본을 두면 총 $\mathcal{S} \times \mathcal{R}$개의 GPU가 필요합니다. 두 전략 모두 비교적 단순한(fairly trivial) 구현이며, 동일한 원칙을 다중 머신 환경으로 확장하여 분산 인덱스를 구성할 수 있습니다. 이러한 다중 GPU 전략은 단일 GPU의 메모리와 연산 한계를 넘어 십억 규모의 데이터셋을 효과적으로 처리하는 데 핵심적인 역할을 합니다.
## 실험 및 응용

이 절에서는 앞서 설명한 GPU $k$-선택 알고리즘과 최근접 이웃 탐색 방법의 실험적 성능을 기존 라이브러리와 비교하고, 다양한 실세계 응용 시나리오에서의 효과를 입증합니다. 특별히 명시하지 않는 한, 모든 실험은 2개의 2.8GHz Intel Xeon E5-2680v2 CPU와 4개의 Maxwell Titan X GPU, CUDA 8.0 환경에서 수행되었습니다.

### $k$-선택 성능 비교

WarpSelect의 성능은 두 가지 기존 GPU 소규모 $k$-선택 구현과 비교됩니다. 첫 번째는 Tang et al.의 fgknn 라이브러리에서 추출한 행 기반 Merge Queue with Buffered Search and Hierarchical Partition이며, 두 번째는 Sismanis et al.의 Truncated Bitonic Sort(TBiS)입니다. 두 방법 모두 각각의 정확 탐색 라이브러리에서 추출되었습니다.

실험에서는 $n_q = 10{,}000$개의 행을 가진 행 우선(row-major) 행렬에서 $k = 100$과 $k = 1000$에 대한 $k$-선택을 수행합니다. 배열 길이 $\ell$은 1,000부터 128,000까지 변화시키며, 입출력 데이터는 모두 GPU 메모리에 상주합니다. 입력 크기는 $\ell = 1{,}000$일 때 40 MB에서 $\ell = 128{,}000$일 때 5.12 GB까지 범위를 가집니다. TBiS는 큰 보조 저장 공간을 필요로 하여 $\ell \leq 48{,}000$까지만 테스트가 가능했습니다.

![k-선택 방법별 런타임 비교](https://ar5iv.labs.arxiv.org//html/1702.08734/assets/x3.png)

위 그림은 배열 길이 $\ell$에 따른 각 $k$-선택 방법의 런타임을 보여줍니다. 실선은 $k = 100$, 점선은 $k = 1000$에 해당하며, Titan X의 메모리 대역폭 한계(memory bandwidth limit)도 함께 표시되어 이론적 최대 성능의 기준선을 제공합니다.

가장 큰 배열 길이인 $\ell = 128{,}000$에서 WarpSelect는 fgknn 대비 $k = 100$일 때 $1.62\times$, $k = 1000$일 때 $2.01\times$ 빠른 성능을 달성합니다. 특히 큰 $k$에서 WarpSelect의 상대적 우위가 더 커지며, TBiS조차 $k = 1000$에서 큰 $\ell$에 대해 fgknn을 추월하기 시작합니다. 피크 성능 대비로 보면, WarpSelect는 $k = 100$에서 이론적 최대 성능의 55%에 도달하지만 $k = 1000$에서는 16%로 떨어집니다. 이는 앞서 분석한 바와 같이 더 큰 스레드 큐와 병합/정렬 네트워크의 오버헤드 때문입니다.

fgknn과의 핵심 차이점은 다음과 같습니다. WarpSelect는 모든 상태를 레지스터에 유지하여 공유 메모리를 사용하지 않으며, 워프 간 동기화나 버퍼링 없이 동작하고, "계층적 파티션" 단계가 없으며, 다른 커널과 융합이 가능하고, 홀수 크기 네트워크를 활용한 효율적 병합·정렬을 수행합니다.

### $k$-means 클러스터링

$k = 1$인 정확 탐색은 $k$-means 클러스터링의 할당 단계에서 $n_q$개 학습 벡터를 $|\mathcal{C}_1|$개 센트로이드에 배정하는 데 사용됩니다. $k = 1$인 경우에는 WarpSelect 대신 병렬 리덕션(parallel reduction)이 사용되지만, $k$-means는 앞서 설명한 조대 양자화기 $q_1$을 학습하는 클러스터링의 좋은 벤치마크가 됩니다.

MNIST8m 데이터셋(810만 개의 784차원 그레이스케일 숫자 이미지)에서 20회 반복 실험한 결과는 다음과 같습니다.

| 센트로이드 수 | 방법 | GPU 수 | 시간 |
|:---:|:---:|:---:|:---:|
| 256 | BIDMach | 1 | 320 s |
| 256 | 제안 방법 | 1 | 140 s |
| 256 | 제안 방법 | 4 | 84 s |
| 4096 | BIDMach | 1 | 735 s |
| 4096 | 제안 방법 | 1 | 316 s |
| 4096 | 제안 방법 | 4 | 100 s |

위 표에서 확인할 수 있듯이, 동일하게 cuBLAS를 기반으로 하는 BIDMach 대비 $2\times$ 이상 빠른 성능을 보입니다. BIDMach는 여러 분산 $k$-means 구현(수십 대의 머신을 필요로 하는)보다 효율적인 것으로 입증된 구현임에도 불구하고 이러한 성능 차이가 나타납니다. 이는 앞서 설계한 $k$-선택과 L2 거리 계산의 융합(fusion)이 기여한 결과입니다. 다중 GPU 실행 시에는 4개 GPU에서 4096 센트로이드 기준 $3.16\times$의 거의 선형적인 속도 향상이 관찰됩니다.

대규모 실험에서는 근사 CPU 방법과도 비교됩니다. Babenko et al.의 방법은 $10^8$개의 128차원 벡터를 85k 센트로이드로 클러스터링하는 데 46분의 실행 시간과 최소 56분의 전처리 시간이 필요하여 총 102분 이상이 소요됩니다. 반면 제안된 방법은 4개 GPU에서 전처리 없이 정확(exact) $k$-means를 52분 만에 완료합니다.

### 정확 최근접 이웃 탐색

SIFT1M 데이터셋($\ell = 10^6$, $d = 128$, $n_q = 10^4$)에서의 정확 탐색 실험이 수행됩니다. 부분 거리 행렬 $D'$ 계산에는 $n_q \times \ell \times d = 1.28$ Tflop이 필요하며, 현세대 GPU에서 1초 미만에 완료됩니다.

![SIFT1M 정확 탐색 k-NN 시간](https://ar5iv.labs.arxiv.org//html/1702.08734/assets/x4.png)

위 그림은 다양한 $k$ 값에 대해 거리 계산 비용과 $k$-선택 비용을 분리하여 보여줍니다. 앞서 설명한 GEMM 타일링과 $k$-선택의 피크 가능 성능도 함께 표시됩니다. 핵심 관찰 결과로, `thrust::sort_by_key`를 사용한 나이브 알고리즘은 비교 대상 방법들보다 $10\times$ 이상 느립니다. 또한 제안된 방법을 제외한 모든 구현에서 L2 거리와 $k$-선택 비용이 지배적이며, 제안된 방법은 피크 가능 성능의 85%를 달성합니다. 특히 앞서 설계한 융합 L2/$k$-선택 커널이 핵심적인데, 융합 없이 $D'$에 대한 추가 패스가 필요한 동일 알고리즘은 최소 25% 더 느립니다.

효율적인 $k$-선택은 근사 방법에서 더욱 중요해집니다. 근사 방법이 거리 계산 비용을 줄이면 $k$-선택의 상대적 비중이 증가하기 때문입니다.

### 십억 규모 근사 탐색

GPU 기반 대규모 근사 최근접 이웃 탐색에 관한 연구는 매우 드물며, 여기서는 표준 데이터셋과 평가 프로토콜을 사용한 비교 결과를 제시합니다.

**SIFT1M**에서 Wieschollek et al.의 구현과 비교하면, 동일한 시간 예산(쿼리당 0.02 ms) 내에서 R@1 = 0.80, R@100 = 0.95를 달성하는 반면, 기존 구현은 R@1 = 0.51, R@100 = 0.86에 그칩니다.

**SIFT1B**(10억 개의 SIFT 이미지 특징)에서는 동일한 메모리 사용량($m = 8$바이트/벡터) 조건에서, 단일 GPU로 R@10 = 0.376을 쿼리당 17.7 $\mu$s에 달성합니다. 이는 기존 구현의 R@10 = 0.35, 쿼리당 150 $\mu$s 대비 더 정확하면서 $8.5\times$ 빠른 결과입니다.

**DEEP1B**(10억 개의 CNN 이미지 표현)에서는 $m = 20$, $d = 80$(OPQ 사용), $|\mathcal{C}_1| = 2^{18}$의 설정으로 단일 GPU 메모리를 초과하므로 4개 GPU를 앞서 설명한 $\mathcal{S} = 2$, $\mathcal{R} = 2$ 구성으로 사용합니다. R@1 = 0.4517을 쿼리당 0.0133 ms에 달성하며, 이는 원래 논문의 CPU 1 스레드 결과인 R@1 = 0.45, 쿼리당 20 ms와 비교하여 하드웨어 플랫폼은 다르지만 GPU 탐색이 단일 머신에서 달성 가능한 속도의 판도를 바꾸는(game-changer) 수준임을 보여줍니다.
### $k$-NN 그래프 구축

유사도 검색 방법의 대표적 응용 사례로, 데이터셋 전체에 대해 브루트 포스 방식으로 $k$-최근접 이웃 그래프를 구축하는 실험이 수행됩니다. 이는 모든 벡터를 쿼리로 사용하여 전체 인덱스에 대해 탐색하는 종단 간(end-to-end) 테스트입니다.

실험은 두 가지 대규모 데이터셋에서 진행됩니다. 첫 번째는 YFCC100M 데이터셋에서 추출한 9,500만 장의 이미지로, ResNet의 끝에서 두 번째 레이어에서 CNN 디스크립터를 계산하고 PCA로 $d = 128$차원으로 축소한 벡터를 사용합니다. 두 번째는 앞서 근사 탐색에서 사용한 DEEP1B 데이터셋입니다. 평가는 세 가지 측면의 트레이드오프를 측정합니다. **속도**는 IVFADC 인덱스를 처음부터 구축하고 $k = 10$인 전체 $k$-NN 그래프를 완성하는 데 걸리는 시간이며, **품질**은 10,000개 샘플에 대해 정확한 최근접 이웃을 계산한 뒤, 발견된 10개 이웃 중 실제 10-최근접 이웃에 포함되는 비율로 측정됩니다.

YFCC100M에서는 $2^{16}$개의 조대 센트로이드와 $m = 16, 32, 64$바이트 PQ 인코딩을, DEEP1B에서는 OPQ를 통해 $d = 120$으로 전처리하고 $|\mathcal{C}_1| = 2^{18}$, $m = 20, 40$을 사용합니다. 멀티 프로브 파라미터 $\tau$를 1에서 256까지 변화시켜 효율성과 품질 간의 트레이드오프를 관찰합니다.

![YFCC100M k-NN 그래프 구축 속도/정확도 트레이드오프](https://ar5iv.labs.arxiv.org//html/1702.08734/assets/x5.png)

![DEEP1B k-NN 그래프 구축 속도/정확도 트레이드오프](https://ar5iv.labs.arxiv.org//html/1702.08734/assets/x6.png)

위 그림들은 YFCC100M과 DEEP1B에서의 브루트 포스 10-NN 그래프 구축의 속도/정확도 트레이드오프를 보여줍니다. YFCC100M에서는 $\mathcal{S} = 1$, $\mathcal{R} = 4$ 구성(4개 Titan X GPU)으로 0.8 이상의 정확도를 35분 만에 달성합니다. DEEP1B에서는 낮은 품질의 그래프를 약 6시간에, 높은 품질의 그래프를 약 12시간에 구축할 수 있습니다. 8개의 Maxwell M40 GPU(Titan X와 대략 동등한 성능)로 복제본 수를 2배로 늘리면 $m = 20$에서 약 $1.6\times$, $m = 40$에서 약 $1.7\times$의 준선형(sub-linear) 성능 향상이 관찰됩니다.

이 결과의 의미를 기존 연구와 비교하면 매우 인상적입니다. 기존에 알려진 가장 큰 규모의 $k$-NN 그래프 구축은 3,650만 개의 384차원 벡터에 대해 128대의 CPU 서버 클러스터에서 NN-Descent 알고리즘으로 108.7시간이 소요된 사례입니다. 본 논문의 방법은 이보다 26배 이상 큰 데이터셋(10억 벡터)을 단 4개 GPU에서 처리합니다. 또한 가장 큰 GPU 기반 $k$-NN 그래프 구축 사례는 2,000만 개의 15,000차원 벡터에 대해 32개의 Tesla C2050 GPU 클러스터에서 10일이 소요되었으며, 이 방법을 DEEP1B에 적용하면 GEMM 비용 기준으로 약 200일의 비현실적인 계산 시간이 필요합니다. NN-Descent는 그래프 저장 자체에도 DEEP1B 기준 80 GB가 필요하고, 모든 벡터에 대한 랜덤 액세스(384 GB)를 요구하여 메모리 오버헤드가 매우 큽니다.

### $k$-NN 그래프의 활용

이미지 데이터셋에 대해 $k$-NN 그래프가 구축되면, 임의의 두 이미지 사이에서 그래프 내 경로를 탐색할 수 있습니다(단일 연결 성분이 존재하는 경우). 출발 이미지 $S$와 도착 이미지 $D$가 주어졌을 때, 경로 $P = \{p_1, \ldots, p_n\}$에서 $p_1 = S$, $p_n = D$이며 다음을 최적화합니다.

$$\min_P \max_{i=1..n} d_{p_i p_{i+1}}$$

이 수식의 직관적 의미는 경로 상의 연속된 이미지 쌍 간 거리의 최댓값을 최소화하는 것으로, 가능한 한 매끄러운(smooth) 전환을 선호한다는 뜻입니다. 일반적인 최단 경로가 총 거리를 최소화하는 것과 달리, 이 공식은 경로의 가장 급격한 전환을 최소화하여 시각적으로 자연스러운 이미지 시퀀스를 생성합니다.

![YFCC100M k-NN 그래프 경로 탐색 예시](https://ar5iv.labs.arxiv.org//html/1702.08734/assets/figs/flowers.jpg)

위 그림은 YFCC100M의 9,500만 이미지에서 구축된 $k$-NN 그래프($k = 15$)에서 두 꽃 이미지 사이의 경로를 보여줍니다. 첫 번째와 마지막 이미지가 주어지면, 알고리즘이 둘 사이의 가장 매끄러운 경로를 계산합니다. 데이터셋에 다양한 꽃 이미지가 풍부하게 존재하기 때문에 전환이 매우 자연스럽게 이루어지며, 이 경로는 20초의 전파(propagation) 과정을 통해 얻어졌습니다. 이는 대규모 $k$-NN 그래프가 단순한 검색을 넘어 데이터셋의 구조적 탐색과 시각적 내비게이션이라는 실용적 응용에 활용될 수 있음을 보여주는 인상적인 사례입니다.
## 결론

이 논문은 GPU의 산술 처리량과 메모리 대역폭이 각각 테라플롭스(teraflops)와 수백 기가바이트/초(hundreds of GB/s) 수준에 이르지만, 이러한 성능 수준에 근접하는 알고리즘을 구현하는 것은 복잡하고 직관에 반하는(counter-intuitive) 작업이라는 점을 출발점으로 삼고 있습니다. 이 연구에서는 GPU에서 근최적(near-optimal) 성능을 달성하는 유사도 검색 방법의 알고리즘적 구조를 제시하였습니다.

앞서 상세히 다룬 WarpSelect 알고리즘, 정확 탐색의 GEMM 기반 거리 분해와 융합 $k$-선택, 그리고 IVFADC 인덱싱의 룩업 테이블 최적화와 다중 GPU 병렬화 전략이 결합되어, 기존에 복잡한 근사 알고리즘이 필요했던 응용들을 단순한 브루트 포스 방식으로도 해결할 수 있게 되었다는 점이 이 논문의 핵심 메시지입니다. 구체적으로, 앞서 실험 결과에서 확인한 바와 같이, 정확(exact) $k$-means 클러스터링이나 $k$-NN 그래프 구축을 단순 전수 탐색 방식으로 수행하더라도, CPU(또는 CPU 클러스터)가 근사적으로 수행하는 것보다 더 빠르게 완료할 수 있음이 입증되었습니다.

이 결론이 시사하는 바는 매우 의미 깊습니다. 전통적으로 대규모 유사도 검색은 정확도를 희생하여 속도를 얻는 근사 알고리즘의 영역이었으나, GPU의 연산 능력을 알고리즘적으로 최대한 활용하면 정확한 방법이 근사 방법보다도 빠를 수 있다는 패러다임 전환을 보여줍니다. 특히 머신러닝 알고리즘의 인기 덕분에 과학 워크스테이션에 GPU 하드웨어가 이미 보편화되어 있으므로, 이러한 GPU 기반 유사도 검색이 데이터베이스 응용에 대한 GPU의 유용성을 더욱 확장할 수 있다는 점을 강조하고 있습니다.

논문과 함께 공개된 FAISS 라이브러리는 이 연구에서 제시된 알고리즘들의 정교하게 엔지니어링된 구현체로, GPU를 활용한 효율적 유사도 검색을 실무에서 즉시 활용할 수 있도록 합니다. 이는 연구 결과의 재현성과 실용적 파급력을 동시에 보장하는 중요한 기여입니다.

종합하면, 이 논문은 GPU 아키텍처의 특성(레지스터 파일의 방대한 대역폭, 워프 수준 병렬성, 메모리 계층 구조)을 깊이 이해하고 이에 맞춘 알고리즘 설계를 통해, 십억 규모의 벡터 데이터에 대한 유사도 검색 문제를 근본적으로 재정의한 연구로 평가할 수 있습니다.
## 부록: WarpSelect의 복잡도 분석

이 부록에서는 앞서 복잡도 분석 절에서 요약적으로 제시된 $N_2$와 $N_3$의 이론적 유도 과정을 확률론적으로 엄밀하게 전개합니다. WarpSelect에 무작위 순열 입력이 주어졌을 때, 삽입 정렬과 전체 정렬이 각각 평균적으로 몇 번 발생하는지를 수학적으로 증명하는 것이 목표입니다.

### 기본 설정과 연속 최솟값-$k$의 확률

$k$-선택의 입력을 서로 다른 원소들의 무작위 순열인 시퀀스 $\{a_1, a_2, \ldots, a_\ell\}$(1-기반 인덱싱)로 정의합니다. 원소들은 크기 $w$인 그룹(워프) $c$개로 순차적으로 읽히며, GPU에서는 $w = 32$입니다. $\ell$이 $w$의 배수라고 가정하면 $c = \ell / w$가 성립합니다. 여기서 $t$는 앞서 소개한 스레드 큐의 길이입니다.

위치 $n$까지 관찰된 원소들 중 최솟값-$k$에 해당하는 집합을 **연속 최솟값-$k$(successive min-$k$)**라 정의합니다. 핵심 확률 함수 $\alpha(n, k)$는 원소 $a_n$이 위치 $n$에서의 연속 최솟값-$k$에 포함될 확률을 나타냅니다.

$$\alpha(n, k) := \begin{cases} 1 & \text{if } n \leq k \\ k/n & \text{if } n > k \end{cases}$$

이 수식의 직관은 매우 자연스럽습니다. 처음 $k$개의 원소는 정의상 모두 연속 최솟값-$k$에 포함되므로 확률이 1입니다. $n > k$인 경우, 모든 순열이 동일한 확률을 가지므로 $a_n$이 지금까지 본 $n$개 원소 중 가장 작은 $k$개에 포함될 확률은 정확히 $k/n$입니다. 이를 비유하자면, $n$명의 학생이 무작위 순서로 시험지를 제출할 때, $n$번째 학생의 점수가 상위 $k$등 안에 들 확률이 $k/n$인 것과 같습니다.

[Sorting networks and their applications](https://en.wikipedia.org/wiki/Sorting_network)에서 확립된 정렬 네트워크 이론에서 도출되는 로그 복잡도 인자가 이 확률 분석에서 자연스럽게 나타나며, 이것이 WarpSelect의 효율성을 뒷받침하는 수학적 근거가 됩니다.

### 삽입 정렬 횟수 $N_2$의 유도

개별 레인에서 삽입 정렬이 발생하려면, 들어오는 값이 연속 최솟값-$(k+t)$에 포함되어야 합니다. 그런데 해당 레인이 "본" 값의 수는 전체 시퀀스가 아니라 $wc_0 + (c - c_0)$개입니다. 여기서 $c_0$는 이전에 워프 투표에서 승리한 횟수입니다. 이 확률은 다음과 같이 근사됩니다.

$$\alpha(wc_0 + (c - c_0), k + t) \approx \frac{k + t}{wc} \quad \text{for } c > k$$

이 근사는 스레드 큐가 자기 레인에 할당된 값만이 아닌 전체 $wc$개의 값을 모두 관찰한 것으로 간주하는 것입니다. 실제로 워프 큐를 통해 다른 레인의 정보도 간접적으로 공유되므로 이 근사가 합리적입니다.

$w$개 레인 중 **어느 하나라도** 삽입 정렬을 발생시킬 확률은 여사건의 확률로 계산됩니다. 모든 레인이 삽입을 발생시키지 않을 확률은 $(1 - \frac{k+t}{wc})^w$이므로, 적어도 하나가 발생할 확률은 다음과 같습니다.

$$1 - \left(1 - \frac{k+t}{wc}\right)^w \approx \frac{k+t}{c}$$

여기서 근사는 1차 테일러 전개 $(1-x)^w \approx 1 - wx$를 적용한 것입니다. $\frac{k+t}{wc}$가 충분히 작을 때(즉, $c$가 충분히 클 때) 이 근사의 정확도가 높아집니다.

이제 모든 그룹 $c$에 대해 이 확률을 합산하면 삽입 정렬의 기대 횟수를 얻습니다.

$$N_2 \approx \sum_{c=1}^{\ell/w} \frac{k+t}{c} = (k+t) \sum_{c=1}^{\ell/w} \frac{1}{c} = (k+t) H_{\ell/w} \approx (k+t)\log(c) = \mathcal{O}(k\log(\ell/w))$$

여기서 $H_n = 1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{n}$은 조화급수(harmonic series)이며, $H_n \approx \ln(n) + \gamma$로 수렴합니다($\gamma$는 오일러-마스케로니 상수). 이 결과는 앞서 복잡도 분석에서 제시된 $N_2 = \mathcal{O}(k\log(\ell/w))$를 정당화합니다.

```python
import numpy as np

def verify_insertion_sort_count(ell, k, t, w, num_simulations=10000):
    """
    몬테카를로 시뮬레이션으로 삽입 정렬 횟수 N2의 이론값 검증.
    무작위 순열에 대해 실제 삽입 발생 횟수를 측정합니다.
    """
    c = ell // w
    insertion_counts = []
    
    for _ in range(num_simulations):
        # 무작위 순열 생성
        perm = np.random.permutation(ell)
        
        # 연속 최솟값-(k+t) 추적
        min_k_plus_t = np.full(k + t, np.inf)
        insertions = 0
        
        for group_idx in range(c):
            group = perm[group_idx * w : (group_idx + 1) * w]
            any_insertion = False
            
            for val in group:
                # 현재 k+t번째 최솟값보다 작으면 삽입 발생
                if val < min_k_plus_t[0]:  # min_k_plus_t[0]은 최대값 (내림차순)
                    any_insertion = True
                    min_k_plus_t[0] = val
                    min_k_plus_t.sort()  # 재정렬 (오름차순으로 유지)
                    min_k_plus_t = min_k_plus_t[::-1]  # 내림차순
            
            if any_insertion:
                insertions += 1
        
        insertion_counts.append(insertions)
    
    empirical_mean = np.mean(insertion_counts)
    # 이론적 예측: (k+t) * H_{ℓ/w}
    H_c = sum(1.0 / i for i in range(1, c + 1))
    theoretical = (k + t) * H_c
    
    # 단순화된 근사: (k+t) * ln(c) 
    approx = (k + t) * np.log(c)
    
    return {
        "실험_평균_N2": f"{empirical_mean:.1f}",
        "이론값_(k+t)*H_c": f"{theoretical:.1f}",
        "근사값_(k+t)*ln(c)": f"{approx:.1f}",
        "실험/이론_비율": f"{empirical_mean / theoretical:.3f}"
    }

# 검증 예시
result = verify_insertion_sort_count(ell=1024, k=10, t=2, w=32, num_simulations=5000)
for key, val in result.items():
    print(f"  {key}: {val}")
```

### 전체 정렬 횟수 $N_3$의 유도: 단일 레인 분석

전체 정렬의 기대 횟수 $N_3 = \pi(\ell, k, t, w)$를 구하기 위해 먼저 $w = 1$(단일 레인, $c = \ell$)인 경우를 분석합니다. 이를 위해 보조 함수 $\gamma(\ell, m, k)$를 정의합니다. 이 함수는 길이 $\ell$인 시퀀스에서 순차 스캐너($w = 1$)가 관찰할 때 정확히 $m$개의 원소가 연속 최솟값-$k$에 포함될 확률입니다.

$\gamma(\ell, m, k)$는 다음의 재귀 관계(recurrence relation)로 정의됩니다.

$$\gamma(\ell, m, k) := \begin{cases} 1 & \ell = 0 \text{ and } m = 0 \\ 0 & \ell = 0 \text{ and } m > 0 \\ 0 & \ell > 0 \text{ and } m = 0 \\ \gamma(\ell-1, m-1, k) \cdot \alpha(\ell, k) + \gamma(\ell-1, m, k) \cdot (1 - \alpha(\ell, k)) & \text{otherwise} \end{cases}$$

마지막 경우의 직관적 해석이 핵심입니다. 현재 원소 $a_\ell$이 연속 최솟값-$k$에 **포함되는** 경우(확률 $\alpha(\ell, k)$)와 **포함되지 않는** 경우(확률 $1 - \alpha(\ell, k)$)를 분리합니다. 포함되면 이전 $\ell - 1$개에서 $m - 1$개가 연속 최솟값-$k$였어야 하고, 포함되지 않으면 이전에 이미 $m$개가 있었어야 합니다. 이는 동적 프로그래밍(dynamic programming)으로 효율적으로 계산 가능합니다.

이제 $\delta(\ell, b, k, t)$ 함수를 정의합니다. 이 함수는 길이 $\ell$인 모든 시퀀스 중에서 정확히 $b$번의 전체 정렬(워프 투표 승리)을 유발하는 시퀀스의 비율입니다.

$$\delta(\ell, b, k, t) := \sum_{m=bt}^{\min((bt + \max(0, t-1)), \ell)} \gamma(\ell, m, k)$$

이 수식의 의미를 이해하기 위해, 스레드 큐가 $t$번 가득 차야 한 번의 전체 정렬이 발생한다는 점을 상기합니다. $b$번의 전체 정렬이 발생하려면 연속 최솟값-$k$ 원소의 수 $m$이 최소 $bt$개(정확히 $b$번 가득 참)에서 최대 $bt + \max(0, t-1)$개(마지막 큐가 아직 가득 차지 않은 상태)여야 합니다.

최대 $\lfloor \ell / t \rfloor$번의 워프 투표 승리가 가능하므로, $\pi(\ell, k, t, 1)$은 이 분포의 기대값으로 표현됩니다.

$$\pi(\ell, k, t, 1) = \sum_{b=1}^{\lfloor \ell / t \rfloor} b \cdot \delta(\ell, b, k, t)$$

이 수식은 각 가능한 전체 정렬 횟수 $b$에 그 확률 $\delta(\ell, b, k, t)$를 곱하여 합산하는 기대값의 정의 그 자체입니다.
### 특수 경우에 대한 해석적 결과

앞서 동적 프로그래밍으로 계산 가능한 기대값 공식을 유도하였으므로, 이제 특수한 파라미터 조합에서 닫힌 형태(closed-form)의 결과를 도출합니다. 가장 단순한 경우인 $t = 1$, $k = 1$에서 $\pi(\ell, 1, 1, 1)$은 조화수(harmonic number) $H_\ell$과 정확히 일치합니다.

$$H_\ell = 1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{\ell}$$

이 결과는 $\ell \rightarrow \infty$일 때 $\ln(\ell) + \gamma$로 수렴합니다. 여기서 $\gamma \approx 0.5772$는 오일러-마스케로니 상수(Euler-Mascheroni constant)입니다. 이 결과의 직관은 명쾌합니다. $t = 1$이면 스레드 큐가 단 하나의 원소만 보유하므로, 연속 최솟값-1에 새로운 원소가 들어올 때마다 즉시 전체 정렬이 발생합니다. $n$번째 원소가 현재까지의 최솟값을 갱신할 확률이 앞서 정의한 $\alpha(n, 1) = 1/n$이므로, 기대 갱신 횟수는 $\sum_{n=1}^{\ell} 1/n = H_\ell$이 됩니다.

$t = 1$이지만 $k > 1$이고 $\ell > k$인 경우에는 다음과 같은 닫힌 형태의 해가 얻어집니다.

$$\pi(\ell, k, 1, 1) = k + k(H_\ell - H_k)$$

이 수식의 첫 번째 항 $k$는 처음 $k$개 원소가 정의상 모두 연속 최솟값-$k$에 포함되어 $k$번의 전체 정렬을 유발하는 데서 비롯됩니다. 두 번째 항 $k(H_\ell - H_k)$는 $k+1$번째부터 $\ell$번째까지의 원소들이 연속 최솟값-$k$를 갱신할 기대 횟수입니다. 구체적으로, 각 위치 $n > k$에서 갱신 확률이 $k/n$이므로 기대 갱신 횟수의 합은 다음과 같습니다.

$$\frac{k}{k+1} + \frac{k}{k+2} + \cdots + \frac{k}{\ell} = k\left(\frac{1}{k+1} + \frac{1}{k+2} + \cdots + \frac{1}{\ell}\right) = k(H_\ell - H_k)$$

이 결과는 $\mathcal{O}(k\log(\ell))$의 점근적 복잡도를 가집니다. [Fast k-selection algorithms for graphics processing units](https://en.wikipedia.org/wiki/Selection_algorithm)에서 확립된 GPU 기반 선택 알고리즘의 이론적 틀에서도, 이 조화급수 기반의 로그 복잡도가 무작위 입력에 대한 자연스러운 결과임이 확인됩니다.

이제 핵심적인 일반 경우인 $t > 1$, $k > 1$, $\ell > k$를 분석합니다. 가능한 각 시퀀스 $\{a_1, \ldots, a_\ell\}$에 대해 연속 최솟값-$k$의 판정 횟수 $D$가 존재하며, $k \leq D \leq \ell$입니다. 스레드 큐의 길이가 $t$이므로, 큐가 $t$번 가득 차야 한 번의 전체 정렬이 발생합니다. 따라서 워프 투표 승리 횟수는 정의상 $\lfloor D / t \rfloor$입니다. 이로부터 $t$가 일종의 나눗셈 인자로 작용하여 다음의 결과를 얻습니다.

$$\pi(\ell, k, t, 1) = \mathcal{O}(k\log(\ell) / t)$$

이것이 바로 스레드 큐 길이 $t$의 핵심 역할입니다. $t$를 증가시키면 전체 정렬 횟수가 $t$에 반비례하여 감소하므로, 비용이 큰 전체 정렬-병합 연산의 빈도를 효과적으로 제어할 수 있습니다.

```python
import numpy as np
from math import log, ceil

def analytical_full_sort_count(ell, k, t):
    """
    단일 레인(w=1)에서의 전체 정렬 횟수 π(ℓ,k,t,1)의 해석적 결과.
    
    특수 경우:
    - t=1, k=1: H_ℓ (조화수)
    - t=1, k>1: k + k(H_ℓ - H_k)
    - t>1, k>1: O(k log(ℓ) / t)
    """
    # 조화수 계산
    H = lambda n: sum(1.0 / i for i in range(1, n + 1))
    
    if t == 1 and k == 1:
        exact = H(ell)
        asymptotic = log(ell) + 0.5772  # ln(ℓ) + γ
        return {"정확값": exact, "점근값": asymptotic}
    
    elif t == 1 and k > 1:
        exact = k + k * (H(ell) - H(k))
        return {"정확값": exact, "복잡도": f"O({k}·log({ell}))"}
    
    else:  # t > 1, k > 1
        # 정확한 값은 동적 프로그래밍 필요
        # 점근적 상한만 제공
        estimate = k * log(ell) / t
        return {"점근_추정값": estimate, "복잡도": f"O({k}·log({ell})/{t})"}

# 스레드 큐 길이 t의 효과 시연
ell = 100000
k = 100
print(f"ℓ={ell}, k={k}에서 t에 따른 전체 정렬 횟수 추정:")
print(f"{'t':>4} {'π 추정값':>12} {'t=1 대비 감소율':>16}")
for t_val in [1, 2, 4, 8, 16]:
    result = analytical_full_sort_count(ell, k, t_val)
    base = k * log(ell)
    val = base / t_val
    reduction = val / base
    print(f"{t_val:>4} {val:>12.1f} {reduction:>14.1%}")
# t=1 → 100%, t=2 → 50%, t=4 → 25%, t=8 → 12.5%
```

### 다중 레인 분석 ($w > 1$)

실제 GPU에서는 $w = 32$개 레인이 동시에 동작하므로, 단일 레인 분석을 다중 레인으로 확장해야 합니다. 이 확장은 레인 간의 **결합 확률(joint probabilities)** 때문에 복잡해집니다. 구체적으로, 한 그룹에서 $w$개 워커 중 둘 이상이 동시에 전체 정렬을 유발하더라도 실제로는 단 한 번의 정렬만 수행됩니다.

이 문제를 다루기 위해 $\pi'(\ell, k, t, w)$를 정의합니다. 이는 $w$개 워커 간의 **상호 간섭(mutual interference)**이 없다고 가정하되, 각 정렬 이후의 공유 최솟값-$k$ 집합은 결합 시퀀스로부터 갱신된다는 조건 하에서의 기대 워프 투표 승리 횟수입니다. $k \geq w$를 가정하면, $t = 1$인 경우 다음의 상한이 성립합니다.

$$\pi'(\ell, k, 1, w) \leq w\left(\left\lceil\frac{k}{w}\right\rceil + \sum_{i=1}^{\lceil \ell/w \rceil - \lceil k/w \rceil} \frac{k}{w(\lceil k/w \rceil + i)}\right)$$

$$\leq w \cdot \pi(\lceil \ell/w \rceil, k, 1, 1) = \mathcal{O}(wk\log(\ell/w))$$

이 부등식의 핵심 논거는 $w$개 워커 각각이 연속 최솟값-$k$ 원소를 관찰할 확률의 상한이, 각 단계에서 **첫 번째 워커**의 확률로 제한된다는 것입니다. 직관적으로, 각 레인은 전체 시퀀스의 $1/w$에 해당하는 원소만 처리하므로, 단일 레인이 길이 $\lceil \ell/w \rceil$인 시퀀스를 처리하는 것과 유사합니다. 그러나 $w$개 레인이 독립적으로 투표를 승리할 수 있으므로 $w$배의 계수가 곱해집니다.

앞서 단일 레인에서 도출한 $t$에 의한 스케일링을 적용하면, 최종적으로 다음의 상한을 얻습니다.

$$\pi'(\ell, k, t, w) = \mathcal{O}(wk\log(\ell/w) / t)$$

마지막으로, 상호 간섭은 투표 승리 횟수를 **감소**시킬 수만 있습니다(여러 레인이 동시에 승리해도 한 번의 정렬만 발생하므로). 따라서 실제 $\pi(\ell, k, t, w)$에 대해서도 동일한 상한이 성립합니다.

$$\pi(\ell, k, t, w) \leq \pi'(\ell, k, t, w) = \mathcal{O}(wk\log(\ell/w) / t)$$

이 결과는 앞서 WarpSelect의 복잡도 분석에서 $N_3 = \mathcal{O}(k\log(\ell)/t)$로 요약된 바 있으며, 여기서 $w$가 상수($w = 32$)이므로 점근적 표기에서 흡수됩니다. [Efficient selection algorithm for fast k-nn search on GPUs](https://en.wikipedia.org/wiki/Selection_algorithm)에서 제시된 GPU 선택 알고리즘의 이론적 틀과 일치하며, 워프 수준 병렬성을 활용하면서도 예측 가능한 복잡도 보장을 제공한다는 것이 이 분석의 핵심 의의입니다.

```python
import numpy as np
from math import log, ceil

def multi_lane_sort_bound(ell, k, t, w=32):
    """
    다중 레인에서의 전체 정렬 횟수 상한 π'(ℓ,k,t,w) 계산.
    
    π'(ℓ,k,t,w) = O(w·k·log(ℓ/w)/t)
    상호 간섭은 횟수를 줄이기만 하므로 π ≤ π'.
    """
    # 단일 레인 기준: π(⌈ℓ/w⌉, k, 1, 1)
    ell_per_lane = ceil(ell / w)
    H = lambda n: sum(1.0 / i for i in range(1, n + 1)) if n > 0 else 0
    
    # π(⌈ℓ/w⌉, k, 1, 1) = k + k(H_{⌈ℓ/w⌉} - H_k)
    single_lane_t1 = k + k * (H(ell_per_lane) - H(k)) if ell_per_lane > k else k
    
    # π'(ℓ, k, 1, w) ≤ w · π(⌈ℓ/w⌉, k, 1, 1)
    upper_bound_t1 = w * single_lane_t1
    
    # t에 의한 스케일링
    upper_bound = upper_bound_t1 / t
    
    # 점근적 근사
    asymptotic = w * k * log(ell / w) / t if ell > w else w * k / t
    
    return {
        "단일레인_π": f"{single_lane_t1:.1f}",
        "상한_π'(t=1)": f"{upper_bound_t1:.1f}",
        "상한_π'(t={t})": f"{upper_bound:.1f}",
        "점근_O(wk·log(ℓ/w)/t)": f"{asymptotic:.1f}",
        "w=32_흡수_후": f"O(k·log(ℓ/w)/t) = {k * log(ell / w) / t:.1f}"
    }

# GPU 실제 파라미터로 분석
for ell_val in [10000, 100000, 1000000]:
    result = multi_lane_sort_bound(ell=ell_val, k=100, t=3, w=32)
    print(f"\nℓ={ell_val:>10,}, k=100, t=3, w=32:")
    for key, val in result.items():
        print(f"  {key}: {val}")
```

종합하면, 이 부록의 분석은 WarpSelect의 세 가지 비용 구성 요소 중 $N_2$와 $N_3$에 대한 확률론적 근거를 완전히 제공합니다. 삽입 정렬 횟수 $N_2 = \mathcal{O}(k\log(\ell/w))$는 조화급수의 근사에서 직접 도출되고, 전체 정렬 횟수 $N_3 = \mathcal{O}(k\log(\ell/w)/t)$는 재귀 관계와 다중 레인 상한 분석을 통해 유도됩니다. 스레드 큐 길이 $t$가 전체 정렬 빈도를 역수 관계로 제어한다는 사실이 앞서 실험적으로 결정된 최적 $t$ 값의 수학적 정당성을 제공하며, 충분히 큰 $\ell$에서 $N_1 C_1$ 항이 지배적이 되어 WarpSelect가 메모리 대역폭 한계에 근접하는 이론적 근거를 완성합니다.
- - -
### References
* [Billion-scale similarity search with GPUs](https://arxiv.org/pdf/1702.08734v1)