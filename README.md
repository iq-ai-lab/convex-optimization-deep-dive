<div align="center">

# 📐 Convex Optimization Deep Dive

### **경사하강법을 쓰는 것** 과, 볼록 함수라면

$$\text{국소 최적} \;=\; \text{전역 최적}$$

### 이라는 **전역성 보장의 근거** 를 아는 것은 **다르다.**

<br/>

> *SVM 의 Lagrange dual 을 **푸는 것** 과,*
>
> $$\min_x f(x) \;=\; \max_\lambda \; g(\lambda) \quad (\text{Slater 조건 하})$$
>
> *의 **강쌍대성 (strong duality)** 이 성립하는 이유를 증명할 수 있는 것은 다르다.*
>
> *ADMM 을 **쓰는 것** 과, **Proximal Operator***
>
> $$\mathrm{prox}_{\eta g}(v) = \arg\min_x \left\{ g(x) + \frac{1}{2\eta}\|x - v\|^2 \right\}$$
>
> *가 왜 경사하강의 **일반화** 인지 유도할 수 있는 것은 다르다.*

<br/>

**다루는 정리 (시간순)**

Cauchy 1847 *Steepest descent* · Lagrange 1797 *Lagrange 승수* · Karush 1939 / Kuhn–Tucker 1951 *KKT 조건* · Slater 1950 *Slater 조건 + 강쌍대성* · Fenchel 1949 *Fenchel duality* · Moreau 1965 *Moreau envelope + Proximal Operator* · Boyd–Parikh 2011 *ADMM 현대 해설* · Vapnik 1995 *SVM dual*

<br/>

**핵심 질문**

> **왜 전역 최적이 보장되는가** — 볼록 집합의 공리적 정의에서 시작해 Lagrange 쌍대 이론 완전 유도 · Proximal 방법 체계화 · 딥러닝의 비볼록 현실까지, 최적화의 수학적 기반을 끝까지 파헤칩니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![CVXPY](https://img.shields.io/badge/CVXPY-1.4-E34F26?style=flat-square)](https://www.cvxpy.org/)
[![Docs](https://img.shields.io/badge/Docs-39개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

볼록 최적화에 관한 자료는 넘쳐납니다. 하지만 대부분은 **"어떤 알고리즘을 쓰는가"** 에서 멈춥니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "볼록 함수는 극솟값이 하나입니다" | epigraph가 볼록 집합이라는 기하 정의에서 출발해 Jensen 부등식 / 1차 조건 / 2차 조건이 모두 동치임을 완전 증명 |
| "SVM은 QP를 풀면 됩니다" | Hard-margin SVM의 primal → dual 전 과정 전개, Support Vector의 상보적 여유(complementary slackness)가 무엇을 의미하는지 KKT로 해석 |
| "LASSO는 sparsity를 유도합니다" | L1 ball과 등고선의 axis-aligned 접점이 왜 sparse 해를 선호하는지 기하학적으로 증명, ISTA/FISTA로 수렴률 비교 실험 |
| "강쌍대성이 성립하면 KKT를 쓸 수 있습니다" | Slater 조건이 강쌍대성의 충분조건인 이유를 분리 초평면 정리로 증명, KKT가 볼록 문제에서 필요충분조건임을 완전 증명 |
| "ADMM은 분산 최적화에 씁니다" | Augmented Lagrangian + 분리 업데이트 원리 유도, Proximal Operator가 경사하강의 implicit 일반화임을 Moreau envelope으로 연결 |
| "딥러닝은 non-convex라 이론이 안 통합니다" | over-parameterization, Neural Tangent Kernel, Loss Landscape 기하, Mode Connectivity로 "그럼에도 왜 작동하는가"의 수학적 근거 제시 |
| 이론 나열 | NumPy + CVXPY로 직접 구현 + 수렴 곡선 로그-로그 플롯 + 알고리즘 결과 일치 검증 |

---

## 📌 선행 레포 & 후속 레포

```
[Linear Algebra Deep Dive]  ──►  [Calculus & Optimization Deep Dive]  ──►  이 레포
  양의 정부호 행렬, 고유값 분해         경사하강 수렴 분석, 라그랑주 승수          볼록성·쌍대성·Interior Point
  스펙트럼 정리 이해 필수               기초 KKT 조건                            SVM, LASSO, ADMM

                                        이 레포  ──►  [Statistical Learning Theory]
                                                 ──►  [Information Geometry]
                                                 ──►  [LLM Alignment / RLHF]
```

> ⚠️ **선행 학습**: 양의 정부호 행렬과 고유값 분해를 모른다면 [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive)를,  
> 경사하강법 수렴 분석과 라그랑주 승수 기초를 모른다면 [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive)를 먼저 학습하세요.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-볼록_집합의_정의와_예제-4A90D9?style=for-the-badge)](./ch1-convex-sets/01-convex-set-definition.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-볼록_함수의_3개_동치_정의-4A90D9?style=for-the-badge)](./ch2-convex-functions/01-convex-function-definitions.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-볼록_최적화_표준형-4A90D9?style=for-the-badge)](./ch3-convex-problems/01-standard-form.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-Lagrangian과_쌍대_함수-4A90D9?style=for-the-badge)](./ch4-duality/01-lagrangian-dual-function.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-경사하강법_수렴_정리_완전판-4A90D9?style=for-the-badge)](./ch5-algorithms/01-gd-convergence-full.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Proximal_Operator_정의-4A90D9?style=for-the-badge)](./ch6-proximal/01-proximal-operator.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-Logistic_Regression은_볼록이다-4A90D9?style=for-the-badge)](./ch7-ml-applications/01-logistic-regression-convex.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: 볼록 집합(Convex Sets) — 최적화의 공간적 기반

> **핵심 질문:** 임의의 두 점을 잇는 선분이 집합에 포함된다는 조건이 왜 그토록 강력한가? 분리 초평면 정리가 쌍대 이론의 기하학적 출발점인 이유는 무엇인가? LP의 최적해가 왜 반드시 꼭짓점에 있는가?

<details>
<summary><b>볼록 집합의 정의부터 Krein-Milman 정리까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 볼록 집합의 정의와 예제](./ch1-convex-sets/01-convex-set-definition.md) | $\lambda x + (1-\lambda)y \in C$ ($\forall \lambda \in [0,1]$)의 공리적 정의, 초평면·반공간·다면체·노름 볼·양의 정부호 콘 $\mathbb{S}^n_+$ 각각의 볼록성 증명, NumPy로 2D 볼록·비볼록 집합 시각화, "왜 볼록이어야 하는가"의 최적화론적 이유 |
| [02. 볼록 집합의 연산](./ch1-convex-sets/02-convex-set-operations.md) | 교집합·아핀변환·Minkowski 합·합성곱이 볼록성을 보존함을 증명, 볼록 포(convex hull)의 정의와 최소성 증명, 합집합이 볼록성을 일반적으로 보존하지 않는 반례, 볼록 집합의 내부(interior)와 상대 내부(relative interior)의 차이 |
| [03. 분리 초평면 정리(Separating Hyperplane Theorem)](./ch1-convex-sets/03-separating-hyperplane.md) | 두 서로소 볼록 집합을 분리하는 초평면의 존재 증명(Hahn-Banach의 유한차원 버전), 지지 초평면 정리(Supporting Hyperplane Theorem), 쌍대 이론과의 연결 — 이 정리가 강쌍대성 증명의 핵심 도구임을 예고, matplotlib으로 분리 초평면 시각화 |
| [04. 볼록 콘(Convex Cone)과 쌍대 콘](./ch1-convex-sets/04-convex-cone-dual.md) | 볼록 콘의 정의($x \in C, \theta \geq 0 \Rightarrow \theta x \in C$), 정부호 콘 $\mathbb{S}^n_+$·이차 콘(SOC)·비음수 직교체의 예, 쌍대 콘 $C^* = \{y \mid y^\top x \geq 0, \forall x \in C\}$의 기하학적 의미, 자기 쌍대(self-dual) 콘의 특성 |
| [05. Extreme Point와 Krein-Milman 정리](./ch1-convex-sets/05-extreme-point-krein-milman.md) | 극점(extreme point)의 정의 — 다른 두 점의 볼록 결합으로 표현 불가능한 점, Krein-Milman 정리: 콤팩트 볼록 집합은 극점들의 볼록 포, LP 최적해가 꼭짓점(다면체의 극점)에 있는 이유를 완전 증명, Simplex 방법의 이론적 정당성 |

</details>

<br/>

### 🔹 Chapter 2: 볼록 함수(Convex Functions) — 전역 최적의 구조

> **핵심 질문:** epigraph가 볼록 집합이라는 기하 정의와 Jensen 부등식은 왜 동치인가? 강볼록(strongly convex)이 조건수(condition number)를 결정하는 이유는 무엇인가? 켤레 함수(conjugate function)의 기하학적 의미는 무엇인가?

<details>
<summary><b>볼록 함수의 동치 정의부터 주요 함수 카탈로그까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 볼록 함수의 3개 동치 정의](./ch2-convex-functions/01-convex-function-definitions.md) | Jensen 부등식 $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$, epigraph가 볼록 집합, 1차·2차 조건 — 세 정의의 동치 완전 증명, 각 정의가 다른 맥락에서 어떻게 활용되는지, NumPy로 볼록·비볼록 함수 epigraph 시각화 |
| [02. 일계·이계 조건](./ch2-convex-functions/02-first-second-order-conditions.md) | $f(y) \geq f(x) + \nabla f(x)^\top (y-x)$의 완전 증명 (1차 조건: 접선이 함수 아래에 있음), $\nabla^2 f \succeq 0 \Leftrightarrow f$ 볼록의 증명, log-sum-exp·quadratic·$-\log x$의 볼록성 이계 조건 검증, SymPy로 헤시안 기호 계산 |
| [03. 강볼록(Strong Convexity)과 매끄러움(Smoothness)](./ch2-convex-functions/03-strong-convexity-smoothness.md) | $\mu$-strongly convex: $f(y) \geq f(x) + \nabla f(x)^\top(y-x) + \frac{\mu}{2}\|y-x\|^2$ 정의와 유일 최솟값 존재 증명, $L$-smooth: $\|\nabla f(x) - \nabla f(y)\| \leq L\|x-y\|$ 정의, 조건수 $\kappa = L/\mu$가 GD 수렴률 $(1-\mu/L)^k$를 결정하는 이유 |
| [04. 볼록 함수의 연산](./ch2-convex-functions/04-convex-function-operations.md) | 비음 계수 가중합, 포인트와이즈 상한, 아핀 합성, Inf-convolution(하부 합성곱)이 볼록성을 보존함을 증명, "볼록성 보존 규칙" DCP(Disciplined Convex Programming)의 이론적 기반, 비볼록 연산의 반례 |
| [05. Conjugate Function과 Legendre 변환](./ch2-convex-functions/05-conjugate-legendre.md) | $f^*(y) = \sup_x (y^\top x - f(x))$의 정의와 기하 — 지지 초평면의 절편으로 해석, $f^{**} = f$ (닫힌 볼록 함수의 이중 켤레 정리) 완전 증명, Fenchel 부등식, 노름과 쌍대 노름의 관계, Moreau envelope과의 연결 |
| [06. 주요 볼록 함수 카탈로그](./ch2-convex-functions/06-convex-function-catalog.md) | $\ell_p$ 노름, 음의 엔트로피 $\sum x_i \log x_i$, log-sum-exp $\log\sum e^{x_i}$, 스펙트럴 노름, 핵 노름(nuclear norm) 각각의 볼록성 증명과 켤레 유도, 각 함수가 ML에서 어디에 쓰이는지 대응 |

</details>

<br/>

### 🔹 Chapter 3: 볼록 최적화 문제의 형태 — 구조와 모델링

> **핵심 질문:** LP, QP, SOCP, SDP는 어떤 표현력의 계층을 이루는가? 비볼록 문제를 볼록으로 변환하는 일반적인 트릭은 무엇인가? CVXPY의 DCP 규칙은 어떻게 볼록성을 자동으로 확인하는가?

<details>
<summary><b>볼록 최적화 표준형부터 CVXPY 모델링까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 볼록 최적화 문제의 표준형](./ch3-convex-problems/01-standard-form.md) | $\min f_0(x)$ s.t. $f_i(x) \leq 0$, $Ax = b$ — $f_0, f_i$ 볼록, 등식 제약이 아핀이어야 하는 이유 증명, "국소 최적 = 전역 최적"의 핵심 정리와 완전 증명, 최적성 조건 $\nabla f_0(x^*) = 0$ (비제약), KKT (제약) |
| [02. LP, QP, QCQP, SOCP, SDP의 계층](./ch3-convex-problems/02-problem-hierarchy.md) | 각 문제 클래스의 정의와 표준형, LP ⊂ QP ⊂ QCQP ⊂ SOCP ⊂ SDP의 포함 관계 증명, 각 클래스의 해결 복잡도 비교, SVM(QP), Matrix Completion(SDP), Lasso(QP 변환 가능)의 분류, CVXPY로 각 형태 예제 |
| [03. Geometric Programming](./ch3-convex-problems/03-geometric-programming.md) | GP의 표준형 — posynomial 최소화, log 변환 $x_i = e^{y_i}$으로 볼록 문제로 변환하는 기법, GP의 쌍대 문제 유도, 회로 설계·물리 시스템 설계에서의 응용, CVXPY의 GP 모드 사용법 |
| [04. 모델링 기법](./ch3-convex-problems/04-modeling-techniques.md) | Epigraph trick ($\min f \Leftrightarrow \min t$ s.t. $f \leq t$), 관점 함수(perspective function)로 볼록성 보존, 슬랙 변수 도입, 비볼록 제약의 볼록 완화(relaxation), 절댓값·최댓값·노름을 선형/이차 제약으로 변환하는 방법 목록 |
| [05. CVXPY로 문제 표현](./ch3-convex-problems/05-cvxpy-dcp.md) | DCP(Disciplined Convex Programming) 규칙 — atom의 볼록성·단조성·부호 조건이 조합될 때 볼록성이 보존되는 규칙, CVXPY의 자동 볼록성 검증 원리, Lasso·SVM·Logistic Regression·Portfolio Optimization을 CVXPY로 구현 및 시각화 |

</details>

<br/>

### 🔹 Chapter 4: 쌍대 이론(Duality) — 전체의 중심축

> **핵심 질문:** 쌍대 함수가 항상 오목인 이유는 무엇인가? Slater 조건이 강쌍대성의 충분조건인 이유를 분리 초평면 정리로 어떻게 증명하는가? KKT 조건이 볼록 문제에서 필요충분조건인 이유는 무엇인가?

<details>
<summary><b>Lagrangian부터 SVM 쌍대 완전 유도까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Lagrangian과 쌍대 함수](./ch4-duality/01-lagrangian-dual-function.md) | $L(x, \lambda, \nu) = f_0 + \sum \lambda_i f_i + \sum \nu_j h_j$ 정의, 쌍대 함수 $g(\lambda, \nu) = \inf_x L(x, \lambda, \nu)$가 항상 오목인 이유 (무한 개의 아핀 함수의 하한), $g$가 $p^*$의 하한임을 증명, NumPy로 간단한 QP의 Lagrangian과 쌍대 함수 시각화 |
| [02. 약쌍대성(Weak Duality)](./ch4-duality/02-weak-duality.md) | $d^* \leq p^*$ 증명 — 가능 $x$와 $(\lambda, \nu)$에 대해 $g(\lambda, \nu) \leq L(x, \lambda, \nu) \leq f_0(x)$의 연쇄 부등식, 비볼록 문제에서도 항상 성립하는 이유, duality gap $p^* - d^*$의 의미, 비볼록 문제에서의 완화 하한으로의 활용 |
| [03. Slater 조건과 강쌍대성(Strong Duality)](./ch4-duality/03-slater-strong-duality.md) | $d^* = p^*$의 충분조건: Slater 조건 — 내부 가능점(strictly feasible point)의 존재, 증명 핵심: $(p^*, 0)$ 근방의 집합과 epigraph를 분리 초평면 정리로 분리하여 쌍대 변수를 구성, affine 제약에서의 Slater 조건 완화, duality gap이 0임을 수치 실험으로 확인 |
| [04. KKT 조건 — 필요충분조건으로서](./ch4-duality/04-kkt-conditions.md) | KKT 4개 조건: stationarity / primal feasibility / dual feasibility / complementary slackness의 기하학적 의미, 볼록 문제에서 KKT = 최적성의 필요충분조건 완전 증명, 비볼록 문제에서 KKT는 필요조건에 그침, 각 조건이 깨질 때 최적성이 어떻게 무너지는지 반례 |
| [05. 쌍대 해석 — 그림자 가격](./ch4-duality/05-dual-interpretation.md) | 쌍대 변수 $\lambda_i^*$가 "제약 $f_i(x) \leq 0$을 한 단위 완화했을 때 최적값의 변화율"임을 감도 분석으로 증명, $\frac{\partial p^*}{\partial b_i} = -\lambda_i^*$ (그림자 가격), complementary slackness의 경제적 해석 — 비활성 제약의 쌍대 변수는 0, 포트폴리오 최적화에서의 해석 |
| [06. SVM의 쌍대 유도 — 완전판](./ch4-duality/06-svm-dual-derivation.md) | Hard-margin SVM primal: $\min \frac{1}{2}\|w\|^2$ s.t. $y_i(w^\top x_i + b) \geq 1$, Lagrangian 전개, $\nabla_w L = 0, \nabla_b L = 0$ 정류 조건으로 $w = \sum \alpha_i y_i x_i$ 유도, 쌍대 문제로 변환, Support Vector의 complementary slackness 의미, 커널 트릭 적용 지점 명시 |

</details>

<br/>

### 🔹 Chapter 5: 알고리즘 — 경사 기반과 2차 방법

> **핵심 질문:** GD의 $O(1/k)$ 수렴률은 어떻게 증명되는가? Nesterov 가속이 $O(1/k^2)$를 달성하는 수학적 이유는 무엇인가? 1차 방법의 최적 수렴률이 $O(1/k^2)$임을 lower bound로 어떻게 보이는가?

<details>
<summary><b>GD 수렴 정리부터 Interior Point Method까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 경사하강법 수렴 정리 완전판](./ch5-algorithms/01-gd-convergence-full.md) | $L$-smooth 볼록: $f(x_k) - f^* \leq \frac{L\|x_0 - x^*\|^2}{2k}$의 완전 증명(Nesterov Thm), $\mu$-strongly convex: $\|x_k - x^*\|^2 \leq (1 - \mu/L)^k\|x_0-x^*\|^2$ 선형 수렴 증명, 가정이 깨질 때 수렴이 실패하는 반례, NumPy로 수렴 곡선 로그 플롯 |
| [02. Nesterov 가속 경사법(AGM)](./ch5-algorithms/02-nesterov-accelerated.md) | $O(1/k^2)$ 수렴률 달성 — Estimating Sequence 기법으로 가속의 수학적 원리 유도, 모멘텀 계수 $t_{k+1} = (1 + \sqrt{1+4t_k^2})/2$의 유도, strongly convex 경우 $O((1-\sqrt{\mu/L})^k)$ 수렴, Momentum vs Nesterov 수렴 곡선 비교 실험 |
| [03. 하한 경계(Lower Bound)](./ch5-algorithms/03-lower-bound.md) | Nemirovski-Yudin의 first-order oracle lower bound — 임의의 1차 방법은 $O(1/k^2)$보다 빠를 수 없음을 증명, strongly convex의 최적 수렴률 $O((1-\sqrt{\mu/L})^k)$, Nesterov 가속이 이 하한을 달성하는 유일한 방법임, "최적 알고리즘"의 엄밀한 정의 |
| [04. 뉴턴 방법의 국소·전역 수렴](./ch5-algorithms/04-newton-method.md) | $x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1}\nabla f(x_k)$의 2차 수렴 증명 (수렴 반경 내), Damped Newton: $x_{k+1} = x_k - \frac{1}{1+\lambda(x)}[\nabla^2 f]^{-1}\nabla f$ 의 전역 수렴 보장, 헤시안 역행렬 비용 $O(n^3)$의 한계, L-BFGS로의 자연스러운 전환 |
| [05. Interior Point Method](./ch5-algorithms/05-interior-point.md) | 로그 장벽 함수 $\phi(x) = -\sum \log(-f_i(x))$로 부등식 제약을 목적함수에 내재화, 중앙 경로(central path)의 정의와 $t \to \infty$에서 원래 최적해로 수렴, self-concordance가 Newton step의 전역 수렴 보장에 사용되는 원리, LP의 다항 시간 복잡도 보장 |
| [06. Stochastic 방법과 분산 감소](./ch5-algorithms/06-stochastic-variance-reduction.md) | SGD 수렴 분석: $\mathbb{E}[f(x_k)] - f^* \leq O(1/\sqrt{k})$, 분산이 수렴 한계를 만드는 이유, SVRG의 주기적 full gradient로 분산 감소하여 strongly convex에서 $O((1-\mu/L)^k)$ 회복, SAG·SARAH와의 비교, 딥러닝에서의 Mini-batch 분산 |

</details>

<br/>

### 🔹 Chapter 6: Proximal 방법과 분해 알고리즘

> **핵심 질문:** Proximal Operator가 왜 경사하강의 implicit 일반화인가? FISTA가 $O(1/k^2)$를 달성하는 원리는 ISTA와 어떻게 다른가? ADMM의 수렴은 어떻게 보장되는가?

<details>
<summary><b>Proximal Operator부터 Primal-Dual Splitting까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Proximal Operator의 정의](./ch6-proximal/01-proximal-operator.md) | $\text{prox}_f(v) = \arg\min_x \left(f(x) + \frac{1}{2}\|x-v\|^2\right)$ — $f$가 볼록이면 유일하게 존재함을 증명, 경사하강 $x_{k+1} = x_k - \eta\nabla f(x_k)$이 $\text{prox}_{\eta f}(x_k)$와 어떻게 연결되는지 Moreau envelope으로 유도, Proximal 해석의 직관 시각화 |
| [02. 주요 Proximal 연산](./ch6-proximal/02-proximal-examples.md) | Soft-thresholding: $\text{prox}_{\lambda\|\cdot\|_1}(v)_i = \text{sign}(v_i)\max(|v_i|-\lambda, 0)$ 유도, Euclidean projection onto convex set, Group Lasso, Nuclear norm의 proximal — 각각 닫힌 형태 유도와 NumPy 구현, 어떤 함수의 prox가 닫힌 형태로 존재하는지 판단 기준 |
| [03. Proximal Gradient Method(ISTA)와 FISTA](./ch6-proximal/03-ista-fista.md) | $\min f(x) + g(x)$ ($f$: smooth convex, $g$: non-smooth convex)의 구조 분리, ISTA: $x_{k+1} = \text{prox}_{\eta g}(x_k - \eta\nabla f(x_k))$의 $O(1/k)$ 수렴 증명, FISTA: Nesterov 가속 추가로 $O(1/k^2)$ 달성 증명, 로그-로그 플롯으로 수렴률 실험적 확인 |
| [04. Lasso의 완전 풀이](./ch6-proximal/04-lasso-complete.md) | $\min \frac{1}{2}\|Ax-b\|^2 + \lambda\|x\|_1$을 ISTA/FISTA로 푸는 전 과정, CVXPY 결과와 수치 일치 검증, sparsity 발생의 기하학적 이유 — L1 ball과 등고선의 axis-aligned 접점 시각화, $\lambda$ 값에 따른 sparsity 패턴 변화 실험, 수렴 곡선 비교 |
| [05. ADMM(Alternating Direction Method of Multipliers)](./ch6-proximal/05-admm.md) | Augmented Lagrangian $L_\rho = f(x) + g(z) + y^\top(Ax+Bz-c) + \frac{\rho}{2}\|Ax+Bz-c\|^2$의 분리 업데이트, $x$-step / $z$-step / dual update의 교대 최적화, 수렴 조건과 $O(1/k)$ 수렴률 증명, 분산 ML(Federated Learning)에서의 응용 |
| [06. Douglas-Rachford, Primal-Dual Splitting](./ch6-proximal/06-operator-splitting.md) | Douglas-Rachford splitting: $\min f(x) + g(x)$의 operator splitting 해석, Primal-Dual Splitting: $\min f(x) + g(Ax)$ 구조의 일반 알고리즘(Chambolle-Pock), 각 방법의 수렴 조건 비교, "어느 상황에서 어느 방법을 선택하는가"의 실용적 가이드 |

</details>

<br/>

### 🔹 Chapter 7: AI/ML에서의 볼록 최적화 — 이론과 실전의 연결

> **핵심 질문:** Logistic Regression이 왜 전역 최적을 보장하는가? 딥러닝이 비볼록인데도 왜 작동하는가? Online Convex Optimization의 Regret 경계가 왜 $O(\sqrt{T})$인가?

<details>
<summary><b>Logistic Regression부터 Online Convex Optimization까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Logistic Regression은 볼록이다](./ch7-ml-applications/01-logistic-regression-convex.md) | 이진 로그 가능도의 헤시안 $H = X^\top \text{diag}(p(1-p)) X$가 양의 준정부호임을 증명 — 따라서 볼록, MLE가 전역 최적을 보장하는 이유, $\ell_2$ 정규화 추가 시 강볼록으로 유일해 보장, NumPy로 경사하강과 CVXPY 결과 비교 |
| [02. Support Vector Machine의 완전 유도](./ch7-ml-applications/02-svm-complete.md) | Hard-margin SVM(Ch4-06 복습) → Soft-margin: slack 변수 $\xi_i \geq 0$ 추가로 비가분 케이스 처리, 쌍대 유도에서 $\alpha_i \in [0, C]$로 클리핑, 커널 트릭: 쌍대 목적함수에서 $x_i^\top x_j$ → $K(x_i, x_j)$로 치환, scikit-learn 결과와 직접 구현 비교 |
| [03. Regularization의 기하 — L1 vs L2](./ch7-ml-applications/03-regularization-geometry.md) | L1: 정규화 영역이 마름모 — 등고선과의 접점이 axis-aligned 모서리에 놓여 sparsity 유도, L2: 구 형태 — 접점이 어디든 가능하여 sparsity 유도하지 않음, Elastic Net의 두 효과 결합, Lasso path 시뮬레이션, 고차원에서의 sparsity 효과 실험 |
| [04. 딥러닝은 왜 비볼록인데 동작하는가](./ch7-ml-applications/04-deep-learning-non-convex.md) | 비볼록인 이유: 레이어 합성의 비선형성으로 헤시안이 부정부호, Over-parameterization 이론: 파라미터 수 > 데이터 수이면 전역 최솟값이 "충분히 많다", Neural Tangent Kernel 직관, Loss Landscape 기하 — Sharp vs Flat Minima, Mode Connectivity 현상 시각화 |
| [05. Online Convex Optimization과 Regret 경계](./ch7-ml-applications/05-online-convex-optimization.md) | OCO 프레임워크: $T$회 라운드에서 손실 $f_t(x_t)$를 누적, Regret $= \sum f_t(x_t) - \min_x \sum f_t(x)$의 정의, Online Gradient Descent의 $O(\sqrt{T})$ regret 증명, AdaGrad: 좌표별 적응 학습률로 희소 gradient에 $O(\sqrt{T}\log T)$ 개선, Bandit OCO로의 확장 |

</details>

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
cvxpy==1.4.0        # 볼록 최적화 DSL
matplotlib==3.8.0
scikit-learn==1.3.0   # SVM, LogReg 비교
cvxopt==1.3.0       # Interior Point 저수준 구현
sympy==1.12         # 헤시안 기호 계산 (Ch2-02)
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 cvxpy==1.4.0 \
            matplotlib==3.8.0 scikit-learn==1.3.0 cvxopt==1.3.0 \
            sympy==1.12 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 예시 — ISTA vs FISTA 수렴률 비교 (O(1/k) vs O(1/k²))
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def soft_thresh(v, lam):
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0)

def ista(A, b, lam, max_iter=500):
    x = np.zeros(A.shape[1])
    L = np.linalg.norm(A, 2)**2  # Lipschitz 상수
    losses = []
    for _ in range(max_iter):
        grad = A.T @ (A @ x - b)
        x = soft_thresh(x - grad / L, lam / L)
        losses.append(0.5 * np.linalg.norm(A @ x - b)**2 + lam * np.linalg.norm(x, 1))
    return x, losses

def fista(A, b, lam, max_iter=500):
    x = np.zeros(A.shape[1])
    y, t = x.copy(), 1.0
    L = np.linalg.norm(A, 2)**2
    losses = []
    for _ in range(max_iter):
        grad = A.T @ (A @ y - b)
        x_new = soft_thresh(y - grad / L, lam / L)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
        losses.append(0.5 * np.linalg.norm(A @ x - b)**2 + lam * np.linalg.norm(x, 1))
    return x, losses

# CVXPY로 정확한 최적값 계산 후 loss gap을 로그-로그 플롯
np.random.seed(42)
A = np.random.randn(50, 100)
b = np.random.randn(50)
lam = 0.1

x_var = cp.Variable(100)
prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(A @ x_var - b) + lam * cp.norm1(x_var)))
prob.solve()
f_star = prob.value

_, losses_ista = ista(A, b, lam)
_, losses_fista = fista(A, b, lam)

plt.figure(figsize=(8, 5))
iters = np.arange(1, 501)
plt.loglog(iters, np.array(losses_ista) - f_star, label='ISTA: $O(1/k)$', linewidth=2)
plt.loglog(iters, np.array(losses_fista) - f_star, label='FISTA: $O(1/k^2)$', linewidth=2)
plt.loglog(iters, 1.0 / iters, 'k--', alpha=0.5, label='Reference $O(1/k)$')
plt.loglog(iters, 1.0 / iters**2, 'r--', alpha=0.5, label='Reference $O(1/k^2)$')
plt.xlabel('Iteration $k$')
plt.ylabel('$f(x_k) - f^*$')
plt.title('ISTA vs FISTA: 수렴률 비교 (로그-로그 플롯)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ista-fista-convergence.png', dpi=150, bbox_inches='tight')
plt.show()
# FISTA의 기울기 ≈ -2, ISTA의 기울기 ≈ -1 → O(1/k²) vs O(1/k) 실험적 확인
```

---

## 📖 각 문서 구성 방식

모든 문서는 동일한 구조로 작성됩니다.

| 섹션 | 설명 |
|------|------|
| 🎯 **핵심 질문** | 이 문서를 읽고 나면 답할 수 있는 질문 |
| 🔍 **왜 이 이론이 AI에서 중요한가** | SVM, LASSO, 딥러닝 최적화 등 실제 구현과의 연결 |
| 📐 **수학적 선행 조건** | LA·Calc 레포 참조 링크 포함 |
| 📖 **직관적 이해** | 2D·3D 그림으로 볼록성의 기하 직관 |
| ✏️ **엄밀한 정의** | 공리 수준의 정형적 정의 |
| 🔬 **정리와 증명** | 보조정리부터 차근차근, "자명하다" 생략 없음 |
| 💻 **NumPy/CVXPY 구현으로 검증** | 알고리즘 직접 구현 + CVXPY 결과와 일치 검증 |
| 🔗 **AI/ML 연결** | SVM·LASSO·딥러닝·RLHF 등 구체 사례 |
| ⚖️ **가정과 한계** | 볼록성이 깨질 때, Slater 조건이 불만족될 때 |
| 📌 **핵심 정리** | 한 화면 요약 |
| 🤔 **생각해볼 문제** | 개념 심화 질문 + 해설 |

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "SVM의 쌍대 유도를 처음부터 끝까지 이해하고 싶다" — 쌍대 이론 집중 (4일)</b></summary>

<br/>

```
Day 1  Ch1-03  분리 초평면 정리 → 쌍대 이론의 기하학적 기반
Day 2  Ch4-01  Lagrangian과 쌍대 함수 → 오목성과 하한 보장
       Ch4-02  약쌍대성 → d* ≤ p* 증명
Day 3  Ch4-03  Slater 조건과 강쌍대성 → d* = p* 완전 증명
       Ch4-04  KKT 조건 → 볼록 문제에서 필요충분조건
Day 4  Ch4-06  SVM 쌍대 유도 완전판 → complementary slackness 해석
       Ch7-02  SVM 완전 유도(커널 포함) → 직접 구현 vs scikit-learn 비교
```

</details>

<details>
<summary><b>🟡 "LASSO와 Proximal 방법을 이론부터 구현까지 마스터하고 싶다" — Proximal 집중 (1주)</b></summary>

<br/>

```
Day 1  Ch2-05  Conjugate Function → Moreau envelope 연결
       Ch2-06  볼록 함수 카탈로그 → L1 노름의 성질
Day 2  Ch6-01  Proximal Operator 정의 → 경사하강의 일반화
       Ch6-02  주요 Proximal 연산 → Soft-thresholding 유도
Day 3  Ch6-03  ISTA/FISTA → O(1/k) vs O(1/k²) 수렴 증명
Day 4  Ch6-04  Lasso 완전 풀이 → CVXPY와 수치 비교
       Ch7-03  Regularization 기하 → sparsity의 기하학
Day 5  Ch6-05  ADMM → Augmented Lagrangian 분리 업데이트
Day 6  Ch6-06  Douglas-Rachford / Primal-Dual Splitting
Day 7  Ch5-01  GD 수렴 정리 완전판 → Lipschitz 상수와 학습률 연결
```

</details>

<details>
<summary><b>🔴 "볼록 집합 공리에서 딥러닝 비볼록 이론까지 완전 정복한다" — 전체 정복 (7주)</b></summary>

<br/>

```
1주차  Chapter 1 전체 — 볼록 집합
        → 분리 초평면 정리 직접 증명, 2D 시각화
        → LP 꼭짓점 최적성의 수학적 이유 파악

2주차  Chapter 2 전체 — 볼록 함수
        → Jensen·1차·2차 조건 동치 증명 작성
        → 주요 함수 카탈로그 켤레 직접 유도

3주차  Chapter 3 전체 — 문제 형태
        → LP·QP·SOCP·SDP를 CVXPY로 구현
        → Epigraph trick으로 비볼록 → 볼록 변환 실험

4주차  Chapter 4 전체 — 쌍대 이론
        → KKT 조건 4개를 반례와 함께 완전 이해
        → SVM primal → dual 전 과정 손으로 유도

5주차  Chapter 5 전체 — 알고리즘
        → GD / Nesterov / Newton / Interior Point 수렴 실험
        → lower bound로 최적 알고리즘 이해

6주차  Chapter 6 전체 — Proximal 방법
        → ISTA / FISTA / ADMM NumPy 직접 구현
        → CVXPY 결과와 수치 일치 검증

7주차  Chapter 7 전체 — AI/ML 응용
        → Logistic Regression 볼록성 증명
        → Online GD regret bound 직접 유도
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | 행렬, 고유값, SVD, 양의 정부호 행렬 | Ch2-03(강볼록과 헤시안 PD), Ch4-01(쌍대 함수 오목성), Ch5-04(뉴턴 헤시안 역행렬) |
| [calculus-optimization-deep-dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) | GD 수렴 분석, 라그랑주 승수 기초, 야코비안 | Ch5-01(GD 수렴 심화), Ch4-04(KKT 완전 증명), Ch6-03(Proximal Gradient) |
| [probability-statistics-deep-dive](https://github.com/iq-ai-lab/probability-statistics-deep-dive) | 확률론, 기댓값, MLE | Ch5-06(SGD·SVRG 확률적 수렴 분석), Ch7-01(Logistic Regression MLE) |
| [statistical-learning-theory-deep-dive](https://github.com/iq-ai-lab/statistical-learning-theory-deep-dive) | VC dimension, Rademacher, PAC learning | Ch7-05(OCO Regret → Generalization 연결) |

> 💡 이 레포는 **볼록성의 수학적 기반**에 집중합니다. AI 경험이 없어도 Chapter 1~4는 순수 수학 레포로 학습 가능합니다.  
> Chapter 5~7은 경사하강법과 SVM·LASSO 사용 경험이 있을 때 이론적 연결이 더욱 깊어집니다.

---

## 📖 Reference

- **Convex Optimization** (Boyd & Vandenberghe) — 레포의 기본 뼈대, [무료 PDF 공개](https://web.stanford.edu/~boyd/cvxbook/)
- **Lectures on Convex Optimization** (Nesterov, 2nd ed.) — 수렴률 분석의 바이블, Estimating Sequence 원전
- **Proximal Algorithms** (Parikh & Boyd, 2014) — Proximal 방법 통합 참고서, [무료 공개](https://web.stanford.edu/~boyd/papers/prox_algs.html)
- **First-Order Methods in Optimization** (Amir Beck, 2017) — ISTA·FISTA의 교과서적 정리
- **Online Convex Optimization** (Elad Hazan, 2022) — OCO 프레임워크와 Regret 이론
- **Stanford EE364a/b 강의노트** (Boyd) — 부교재, [강의 자료 공개](https://stanford.edu/class/ee364a/)
- **Introductory Lectures on Stochastic Optimization** (Nesterov) — SGD·SVRG 수렴 분석
- **Understanding Machine Learning** (Shalev-Shwartz & Ben-David) — SVM·Regularization의 통계 학습론 관점

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"경사하강법을 쓰는 것과, 볼록 함수라면 국소 최적 = 전역 최적이라는 전역성 보장의 근거를 아는 것은 다르다"*

</div>
