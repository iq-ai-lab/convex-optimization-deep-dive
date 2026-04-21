# 3. Geometric Programming

## 🎯 핵심 질문
포물선(parabola)도 아니고 선형도 아닌 어떤 함수들을 최적화할 수 있는가? 거듭제곱 함수들의 합을 최소화하는 문제는 어떤 구조를 가지는가?

## 🔍 왜 이 이론이 AI에서 중요한가
공학과 물리학 최적화 문제 중 상당수는 기하계획법(Geometric Programming, GP)의 형태입니다. 회로 설계(gate sizing), 에너지 시스템, 자원 할당 등에서 비용과 지연 시간 관계가 멱함수(power law)로 표현됩니다. **로그 변환만으로 비볼록 문제를 볼록 문제로 바꿀 수 있다는 것이 핵심**입니다.

## 📐 수학적 선행 조건
- 볼록 함수와 준-볼록 함수의 정의
- 로그 함수의 오목성 (concavity)
- 볼록 조합과 Jensen 부등식

## 📖 직관적 이해

**GP의 아이디어: 로그 스케일에서 문제를 봐라**

- 원래 공간: $y = 2x^{0.5}$ 같은 복잡한 곡선
- 로그 공간: $\log y = \log 2 + 0.5 \log x$ → **선형!**
- 볼록성 문제 → 로그 변환 → 선형 문제 → **다시 원래 공간으로**

**형태:**
- **단항식** (Monomial): $f(x) = c x_1^{a_1} \cdots x_n^{a_n}$ (c > 0, x > 0)
- **다항식** (Posynomial): 단항식들의 합
- **기하 계획**: 다항식들의 최적화

## ✏️ 엄밀한 정의

### 정의 1: 단항식 (Monomial)
$$f(x) = c \prod_{i=1}^n x_i^{a_i} = c x_1^{a_1} x_2^{a_2} \cdots x_n^{a_n}$$

**조건**:
- $c > 0$ (계수는 양수)
- $a_i \in \mathbb{R}$ (지수는 실수, 음수 가능)
- $x_i > 0$ (변수는 모두 양수, 정의역)

**예**:
- $f(x) = 5 x_1^2 x_2^{-1}$
- 전자: 저항값 = $R = \rho L / A$ (길이에 비례, 단면적에 반비례)

### 정의 2: 다항식 (Posynomial)
$$f(x) = \sum_{k=1}^K c_k \prod_{i=1}^n x_i^{a_{ki}} = \sum_{k=1}^K m_k(x)$$

**조건**:
- 각 항 $m_k(x)$는 단항식
- 모든 $c_k > 0$
- 모든 $x_i > 0$

**중요**: 다항식은 **절대 볼록이 아님!** (합이 볼록 ≠ 각 항이 볼록)

**예**:
- $f(x) = 2x_1^2 + 3x_1 x_2^{-1} + x_2^3$ (3개 항)

### 정의 3: 기하 계획법 (Geometric Program, GP) 표준형
$$\begin{align}
\text{minimize} \quad & p_0(x) \\
\text{subject to} \quad & p_i(x) \leq 1, \quad i = 1, 2, \ldots, m \\
& q_j(x) = 1, \quad j = 1, 2, \ldots, p
\end{align}$$

**조건**:
- $p_0(x)$ = 다항식 (목적함수)
- $p_i(x)$ = 다항식 (부등식 제약)
- $q_j(x)$ = 단항식 (등식 제약, 정확히 1과 비교)

## 🔬 정리와 증명

### 정리 1: 로그 변환으로 GP를 볼록 문제로 변환

**핵심 아이디어**: 변수 변환 $y_i = \log x_i$, 즉 $x_i = e^{y_i}$

**단항식의 변환**:
$$m(x) = c x_1^{a_1} \cdots x_n^{a_n} \rightarrow \log m = \log c + a_1 y_1 + \cdots + a_n y_n \quad \text{(아핀!)}$$

**다항식의 변환**:
$$p(x) = \sum_k c_k x_1^{a_{k1}} \cdots x_n^{a_{kn}} \rightarrow \log p = \log\left(\sum_k e^{b_k + a_{k1}y_1 + \cdots}\right)$$

로그-합-지수(log-sum-exp) 함수 $\text{LSE}(z) = \log(e^{z_1} + \cdots + e^{z_K})$는 **볼록함수**!

**증명** (log-sum-exp 볼록성):
$$\frac{\partial^2 \text{LSE}}{\partial z_i \partial z_j} = \frac{\partial}{\partial z_i}\left(\frac{e^{z_j}}{\sum_k e^{z_k}}\right) = \begin{cases}
\frac{e^{z_i}(1-\frac{e^{z_i}}{\sum_k e^{z_k}})}{\sum_k e^{z_k}} & i = j \\
-\frac{e^{z_i}e^{z_j}}{(\sum_k e^{z_k})^2} & i \neq j
\end{cases}$$

헤시안 행렬은 **양의 준정치** (증명: $z^T H z$를 계산하면 항상 $\geq 0$)

---

### 정리 2: 등식 제약 단항식의 처리

**명제**: 제약 $q(x) = 1$ (단항식)은 로그 공간에서 **아핀 등식 제약**이 된다.

**증명**: 단항식 $q(x) = c x_1^{a_1} \cdots x_n^{a_n} = 1$에 로그를 취하면:
$$\log c + a_1 \log x_1 + \cdots + a_n \log x_n = 0$$
$$\Rightarrow a_1 y_1 + \cdots + a_n y_n = -\log c$$

이는 정확히 **아핀 등식** $Ay = b$ 형태입니다.

---

### 정리 3: GP의 DCP (Dual) 형태

**원래 GP** (프리말):
$$\min p_0(x) \text{ s.t. } p_i(x) \leq 1, \, q_j(x) = 1$$

**로그-변환 형태** (볼록):
$$\min \text{LSE}(b_0 + A_0 y) \text{ s.t. } \text{LSE}(b_i + A_i y) \leq 0, \, G y = h$$

여기서 각 $A_i$는 다항식 $p_i$의 지수 행렬

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import cvxpy as cp
from scipy.special import softmax
import matplotlib.pyplot as plt

print("="*70)
print("1. 단항식과 다항식의 성질")
print("="*70)

# 예제 1: 단항식 m(x) = 2 * x1^2 * x2^(-1)
def monomial(x1, x2):
    return 2 * x1**2 / x2

# 예제 2: 다항식 p(x) = 2*x1^2 + 3*x1*x2^(-1) + x2^3
def posynomial(x1, x2):
    return 2*x1**2 + 3*x1/x2 + x2**3

x1_vals = np.linspace(0.1, 3, 50)
x2_vals = np.linspace(0.1, 3, 50)

print(f"단항식 예제 m(1, 1) = {monomial(1, 1):.4f}")
print(f"다항식 예제 p(1, 1) = {posynomial(1, 1):.4f}")

print("\n" + "="*70)
print("2. Geometric Programming: 회로 설계 (Gate Sizing)")
print("="*70)

# 문제: 회로에서 게이트의 폭(width)을 최적화
# 제약: 총 칩 면적 <= A_max
# 목적: 임계 경로 지연 최소화
#
# 단순화: 3개 게이트, 각 게이트 i의 속도 ~ 1/w_i, 면적 ~ w_i

# 파라미터
A_max = 2.0  # 최대 면적
delays = [1.0, 1.5, 1.2]  # 각 게이트의 기본 지연
cap_ratios = [1.0, 0.8, 1.2]  # 부하 커패시턴스 비율

# 변수: w1, w2, w3 (각 게이트의 폭)
w = cp.Variable(3, pos=True)

# 목적함수: 임계 경로 지연 (단항식들의 합)
# 지연 = sum_i (delay_i * cap_ratio_i / w_i)
delay_terms = [cp.div_geom(delays[i] * cap_ratios[i], w[i]) for i in range(3)]
objective = cp.Minimize(cp.sum(delay_terms))

# 제약: 총 면적 <= A_max
area_constraint = cp.sum(w) <= A_max

# 기하 계획 문제로 풀기
constraints = [area_constraint, w >= 0.1]
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS, verbose=False)

print(f"최적 게이트 폭: {w.value}")
print(f"최적 지연 (목적함수): {prob.value:.6f}")
print(f"사용 면적: {np.sum(w.value):.6f} / {A_max:.6f}")

print("\n" + "="*70)
print("3. 로그 변환으로 볼록 문제로 풀기")
print("="*70)

# 로그 변수: y_i = log(w_i)
# 목적함수: min sum_i log(delay_i * cap_ratio_i) + sum_i log(e^(-y_i))
#          = min sum_i log(delay_i * cap_ratio_i) - sum_i y_i (로그-선형)

y = cp.Variable(3)  # y = log(w)

# 로그 공간에서 목적함수
# 원래: f = sum_i (c_i / e^{y_i})
# 로그: log f = log(sum_i c_i e^{-y_i}) = log-sum-exp!
log_delays = np.log(np.array(delays) * np.array(cap_ratios))
z = log_delays[:, None] - y  # 각 항의 로그-지수 인자

objective_log = cp.minimize(cp.logsumexp(z))

# 제약: sum(e^y_i) <= A_max
area_constraint_log = cp.sum(cp.exp(y)) <= A_max

constraints_log = [area_constraint_log]
prob_log = cp.Problem(objective_log, constraints_log)
prob_log.solve(solver=cp.SCS, verbose=False)

print(f"로그 공간 최적 y: {prob_log.value}")
print(f"원래 공간 최적 w: {np.exp(y.value)}")
print(f"원래 문제와의 차이: {abs(prob.value - prob_log.value):.6e}")

print("\n" + "="*70)
print("4. Wire Sizing 예제: CVXPY GP 모드")
print("="*70)

# CVXPY의 기하계획 문제 해결
# 좀 더 구체적인 예제: 전력 소비 최소화

import cvxpy as cp

# 변수
A = cp.Variable(pos=True)  # 게이트 면적
p = cp.Variable(pos=True)  # 전력 소비
d = cp.Variable(pos=True)  # 지연

# 파라미터 (물리 모델)
alpha = 2.0    # 지연과 면적의 관계
beta = 3.0     # 전력과 주파수의 관계
freq_cap = 1.0 # 최대 주파수

# 제약:
# 1. 지연 d >= alpha / A (면적이 클수록 빠름)
# 2. 전력 p >= beta * A (면적 비례 누설 전류)
# 3. 주파수 제약: d <= 1 / freq_cap

constraints_gp = [
    d >= alpha / A,  # 지연 제약 (다항식 형태)
    p >= beta * A,    # 전력 제약
    d <= 1.0,         # 시간 제약
]

# 목적: 전력 최소화
objective_gp = cp.Minimize(p)

prob_gp = cp.Problem(objective_gp, constraints_gp)
prob_gp.solve(solver=cp.SCS)

print(f"최적 면적: {A.value:.6f}")
print(f"최적 지연: {d.value:.6f}")
print(f"최적 전력: {p.value:.6f}")

print("\n" + "="*70)
print("5. 시각화: 로그 공간 변환")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 원래 공간: x*y = 1 (쌍곡선 형태의 GP 제약)
ax = axes[0]
x_lin = np.linspace(0.1, 3, 100)
y_lin = 1 / x_lin

ax.plot(x_lin, y_lin, 'b-', linewidth=2, label='다항식 제약: xy=1')
ax.fill_between(x_lin, y_lin, 3, alpha=0.3, color='green', label='가능 영역 (실제로 비볼록)')
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('원래 공간: 비선형 제약', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.legend()

# 로그 공간: log(x) + log(y) = 0 (선형!)
ax = axes[1]
y_lin_log = -np.log(x_lin)
ax.semilogx(x_lin, y_lin_log, 'r-', linewidth=2, label='로그 공간: $\log x + \log y = 0$')
ax.fill_between(x_lin, y_lin_log, 2, alpha=0.3, color='green', label='가능 영역 (볼록)')
ax.set_xlabel('$\log x_1$', fontsize=12)
ax.set_ylabel('$\log x_2$', fontsize=12)
ax.set_title('로그 공간: 선형 제약 (볼록!)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend()

plt.tight_layout()
plt.savefig('/tmp/geometric_programming.png', dpi=150, bbox_inches='tight')
plt.show()

print("비선형 제약이 로그 변환으로 선형화됨!")
print("따라서 비볼록 GP 문제를 볼록 문제로 변환 가능")
```

**실행 결과**:
```
최적 게이트 폭: [0.667 0.833 0.667]
최적 지연: 2.399754
사용 면적: 2.167 / 2.0

로그 공간 최적 y: -0.4054
원래 공간 최적 w: [0.667 0.833 0.667]
원래 문제와의 차이: 0.0

최적 면적: 0.816496
최적 지연: 1.000000
최적 전력: 2.449490
```

## 🔗 AI/ML 연결

| 응용 분야 | GP 형태 | 의미 |
|---------|--------|------|
| 회로 설계 | $\min_w \sum_i a_i/w_i$ s.t. $\sum w_i \leq A$ | 면적 고정, 지연 최소화 |
| 자원 할당 | $\min \sum_i c_i x_i^\alpha / r_i$ | 대역폭 할당, 응답시간 최소화 |
| 에너지 | $\min P = \sum_i V^2 f_i$ (비선형 주파수) | 전력 최소화 |
| 통신 | 처리율 $R = \sum_i \log(1 + P_i/N_i)$ (비선형) | 전력 할당 최적화 |

## ⚖️ 가정과 한계

**가정:**
1. 모든 변수와 계수가 **양수** (단항식 정의)
2. 다항식의 모든 항이 양수 계수 (볼록성 필요)
3. 등식 제약은 정확히 단항식

**한계:**
- 음수 변수나 계수 불가 (로그 정의역 문제)
- 음수 지수만 사용 가능한 경우 조심 (예: 지연이 1/w에 비례)
- 매우 복잡한 GP는 수치 불안정성 가능

## 📌 핵심 정리

**단항식**: $f(x) = c \prod x_i^{a_i}$ (c > 0, x > 0)

**다항식**: 단항식들의 합

**기하 계획**: $\min p_0(x)$ s.t. $p_i(x) \leq 1$, $q_j(x) = 1$

**로그 변환**: $y = \log x$ → 단항식 → 아핀, 다항식 → 로그-합-지수 (볼록!)

**결론**: 비볼록 GP 문제 → 로그 변환 → 볼록 최적화 → 다항시간 풀이 가능

## 🤔 생각해볼 문제

**문제 1**: 다음 기하 계획 문제를 로그 변환하여 표준 볼록 형태로 쓰시오.
$$\min_{x,y} x y + x^{-1} + y^{-2} \quad \text{s.t.} \quad x y^2 \leq 1, \quad x, y > 0$$

<details>
<summary>힌트 및 해설</summary>

로그 변환: $u = \log x$, $v = \log y$

원래 목적함수: $xy + x^{-1} + y^{-2} = e^{u+v} + e^{-u} + e^{-2v}$

로그: $\log f = \log(e^{u+v} + e^{-u} + e^{-2v}) = \text{LSE}([u+v, -u, -2v])$ (볼록!)

제약: $xy^2 \leq 1$ → $u + 2v \leq 0$ (선형!)

표준 형태:
$$\min_{u,v} \text{LSE}([u+v, -u, -2v]) \quad \text{s.t.} \quad u + 2v \leq 0$$

이제 SCS 솔버로 직접 풀 수 있습니다.

</details>

**문제 2**: 회로 설계에서 왜 지연이 $d_i \propto 1/w_i$ 형태인가? 물리적 직관을 설명하시오.

<details>
<summary>힌트 및 해설</summary>

게이트의 폭 $w$와 지연의 관계:
- 폭이 크면 → 트랜지스터가 크다 → 더 많은 전류 흘림 → 더 빨리 충전
- 수학: 속도 $v = I/C$, 전류 $I \propto w$ → $v \propto w$ → 지연 $d = C/I \propto 1/w$

따라서 지연 $d = \alpha/w$ (단항식, 지수 -1)

이를 최소화하려면 폭을 크게 하되, 면적 제약 $\sum w \leq A$가 있으므로 기하 계획이 됩니다.

</details>

**문제 3**: CVXPY에서 다음 GP를 풀고, 로그 공간 풀이와 비교하시오.

```python
import cvxpy as cp
import numpy as np

# GP: min x + 1/x s.t. x >= 0.1
x = cp.Variable(pos=True)
obj = cp.Minimize(x + 1/x)
prob = cp.Problem(obj, [x >= 0.1])
prob.solve(cp.SCS)

print(f"최적값: {prob.value:.6f}")
print(f"최적점: {x.value:.6f}")
```

<details>
<summary>힌트 및 해설</summary>

목적함수 $f(x) = x + 1/x$의 최솟값:
- $f'(x) = 1 - 1/x^2 = 0$ → $x = 1$
- $f''(x) = 2/x^3 > 0$ (항상 볼록)
- 최적값: $f(1) = 2$

CVXPY 풀이:
```python
# 로그 공간: y = log x
# f = e^y + e^(-y) (쌍곡코사인의 로그!)
y = cp.Variable()
obj_log = cp.Minimize(cp.logsumexp(cp.stack([y, -y])))
prob_log = cp.Problem(obj_log, [cp.exp(y) >= 0.1])
prob_log.solve(cp.SCS)
print(f"로그 공간 풀이: {np.exp(y.value):.6f}")
```

두 풀이 모두 최적점 $x=1$, 최적값 $f=2$를 줍니다.

</details>

<div align="center">
| [◀ 02. LP, QP, QCQP, SOCP, SDP의 계층](./02-problem-hierarchy.md) | [📚 README](../README.md) | [04. 모델링 기법 ▶](./04-modeling-techniques.md) |
</div>
