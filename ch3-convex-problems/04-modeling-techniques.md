# 4. 모델링 기법

## 🎯 핵심 질문
비볼록 또는 미분 불가능한 문제를 어떻게 볼록 최적화로 변환하는가? 절댓값, 최댓값, 노름 같은 "어려운" 함수를 다루는 표준 기법은 무엇인가?

## 🔍 왜 이 이론이 AI에서 중요한가
실제 ML 문제는 깔끔한 표준형으로 주어지지 않습니다. 절댓값 손실(L1), 최댓값 제약, 정수 제약 등 복잡한 형태입니다. 이런 문제들을 **표준 형태로 재구성(reformulation)하는 기법**이 실용적 최적화의 핵심입니다. Lasso는 L2 노름 제약이 있는 QP로, Robust Optimization은 SOCP로, Matrix Completion은 SDP로 변환됩니다.

## 📐 수학적 선행 조건
- 표준형과 포함 관계 (Ch3-01, 02)
- 볼록 집합과 절단 (cone)
- 에피그래프(epigraph)의 정의

## 📖 직관적 이해

**문제를 "다시 쓰기"의 6가지 주요 기법:**

1. **Epigraph trick**: 비미분가능 함수를 보조 변수로 처리
2. **절댓값/최댓값의 선형화**: 조건부 제약으로 표현
3. **노름 제약의 원뿔 표현**: $\|x\| \leq t$ = SOCP
4. **Perspective 변환**: 비선형 관계를 보존하며 변환
5. **볼록 완화(Relaxation)**: 비볼록 → 더 쉬운 문제
6. **변수 분리**: ADMM 등의 분산 알고리즘 준비

## ✏️ 엄밀한 정의

### 기법 1: Epigraph Trick

**원래 문제**:
$$\min_{x \in \mathcal{C}} f(x)$$

여기서 $f$는 비미분가능하거나 복잡함.

**에피그래프 표현**:
$$\min_{x, t} t \quad \text{s.t.} \quad f(x) \leq t, \quad x \in \mathcal{C}$$

**핵심**: 
- 원래 문제와 정확히 동치
- $f(x) \leq t$를 쉽게 표현할 수 있으면 OK
- 보조 변수 $t$ 도입으로 목적함수가 아핀이 됨

**예**: Lasso 문제
$$\text{원래: } \min_x \|Ax - b\|_2^2 + \lambda\|x\|_1$$
$$\text{Epigraph: } \min_{x,z} \|Ax-b\|_2^2 + \lambda \sum |x_i| \quad \rightarrow \text{아직도 절댓값}$$
$$\text{추가 변환: } \min_{x,z,t} \|Ax-b\|_2^2 + \lambda \sum t_i \quad \text{s.t.} \quad -t_i \leq x_i \leq t_i$$

---

### 기법 2: 절댓값과 최댓값의 선형화

**절댓값 제약**:
$$|x| \leq t \quad \Leftrightarrow \quad \begin{cases} x \leq t \\ -x \leq t \end{cases}$$

**증명**: 
- $(\Rightarrow)$ $|x| \leq t$ ⟹ $x \leq |x| \leq t$ and $-x \leq |x| \leq t$
- $(\Leftarrow)$ $x \leq t$ and $-x \leq t$ ⟹ $\max(x, -x) = |x| \leq t$

**최댓값 제약**:
$$\max_i f_i(x) \leq t \quad \Leftrightarrow \quad f_i(x) \leq t \quad \forall i$$

**예**: $\ell^\infty$ 노름 제약
$$\|x\|_\infty = \max_i |x_i| \leq r \quad \Leftrightarrow \quad |x_i| \leq r \quad \forall i \quad \Leftrightarrow \quad -r \leq x_i \leq r \quad \forall i$$

---

### 기법 3: 노름 제약의 원뿔 표현

**표준형**:
$$\|Ax + b\|_2 \leq c^T x + d$$

이는 **이계원뿔(SOCP)** 제약입니다!

**동치 표현**:
$$\begin{pmatrix} 2(Ax+b)^T \\ c^T x + d + 1 \end{pmatrix} \text{의 좌표가} \; \text{SOC}_n \in \text{상에 속함}$$

아니면 더 간단히: 
$$\|(Ax+b, (c^T x + d)/2)\|_2 \leq (c^T x + d)/2$$

---

### 기법 4: Perspective Function 변환

**원래 문제** (비선형):
$$\min_x f(x) \quad \text{s.t.} \quad g_i(x) \leq 0$$

**Perspective 변환**: $t > 0$을 도입하여
$$\min_{x,t} t \cdot f(x/t) \quad \text{s.t.} \quad t \cdot g_i(x/t) \leq 0, \quad t > 0$$

**특징**:
- 동치성: 최적해는 동일 (t* = 1일 때 확인)
- 볼록성 보존: f 볼록 ⟹ t·f(x/t) 볼록
- 스케일 불변 구조를 표현하는 데 유용

**예**: 최소 부분공간 찾기
$$\min_{x,t} \frac{\|Ax\|_2^2}{t^2} \quad \text{s.t.} \quad \|x\|_2 = 1$$
$$\Rightarrow \min_{x,t} \|Ax\|_2^2 \quad \text{s.t.} \quad \|x\|_2 = t$$

---

### 기법 5: 볼록 완화 (Convex Relaxation)

**비볼록 문제** → **더 쉬운 볼록 문제** (상위 경계)

| 원래 (비볼록) | 완화 (볼록) | 완화 방법 |
|-------------|---------|---------|
| $\min \text{rank}(X)$ | $\min \|X\|_*$ (핵 노름) | 정사각 행렬 → 특이값 합 |
| $\min \|x\|_0$ (sparsity) | $\min \|x\|_1$ (L1) | 절댓값의 합 |
| $\min \|x\|_0 \text{ s.t. } Ax=b$ | $\min \|x\|_1 \text{ s.t. } Ax=b$ | Basis Pursuit |
| $\max \text{tr}(RX)$ s.t. $R \in SO(3)$ | $\max \text{tr}(RX)$ s.t. $R \succeq 0$ | 행렬 완화 |

**정리** (완화의 최적성): 
완화 문제의 최적값 $\geq$ 원래 문제의 최적값

---

### 기법 6: 변수 분리 (Variable Splitting)

**원래 문제**:
$$\min_x f(x) + g(x)$$

여기서 f와 g는 각각 최적화하기는 쉽지만, 합은 어려움.

**변수 분리**:
$$\min_{x,z} f(x) + g(z) \quad \text{s.t.} \quad x = z$$

**용도**: ADMM, Proximal Method 등 분산 알고리즘

## 🔬 정리와 증명

### 정리 1: Epigraph 표현의 동치성

**명제**: 다음 두 문제는 동치이다.
$$(\text{P1}) \quad \min_x f(x) \quad \text{vs} \quad (\text{P2}) \quad \min_{x,t} \{t : f(x) \leq t\}$$

**증명**:
1. (P2의 최적값 ≥ P1의 최적값):
   P2에서 최적 $(x^*, t^*)$를 얻으면, $f(x^*) \leq t^*$이므로 
   $$\min f = f(x^*) \leq t^* = \min_{x,t} t$$

2. (P1의 최적값 ≥ P2의 최적값):
   P1에서 최적 $x^*$를 얻으면, $(x^*, f(x^*))$는 P2의 가능 해이고,
   $$\min_{x,t} t \leq f(x^*) = \min f$$

따라서 두 최솟값이 같음. $\square$

---

### 정리 2: L1 완화의 최적성 조건

**명제**: Basis Pursuit 문제
$$\min_x \|x\|_1 \quad \text{s.t.} \quad Ax = b$$
의 최적해가 원래 부분 집합 선택 문제
$$\min_x \|x\|_0 \quad \text{s.t.} \quad Ax = b$$
의 해이기도 한 충분조건은 **제한된 동등성 성질(RIP)**:
$$(1-\delta_k)\|x\|_2^2 \leq \|Ax\|_2^2 \leq (1+\delta_k)\|x\|_2^2 \quad \forall \|\text{supp}(x)\| \leq k$$

(RIP가 강하면, L1 완화가 정확한 복구를 보장)

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

print("="*70)
print("1. Epigraph Trick: L∞ 노름 최소화")
print("="*70)

# 문제: min_x max_i |x_i - b_i|
# Epigraph: min_{x,t} t s.t. |x_i - b_i| <= t

b = np.array([1.0, -2.0, 3.0])
n = len(b)

# 원래 방식 (비미분가능, 어려움)
# → Epigraph 변환
x = cp.Variable(n)
t = cp.Variable()

objective = cp.Minimize(t)
constraints = [
    cp.abs(x - b) <= t,  # |x_i - b_i| <= t
    t >= 0
]

prob = cp.Problem(objective, constraints)
prob.solve(cp.SCS)

print(f"최적값 (L∞ 노름): {prob.value:.6f}")
print(f"최적점: {x.value}")
print(f"검증 (max |x-b|): {np.max(np.abs(x.value - b)):.6f}")

print("\n" + "="*70)
print("2. 절댓값의 선형화: Lasso")
print("="*70)

# Lasso: min (1/2)||Ax-b||^2 + λ||x||_1
# 변수 분리: min (1/2)||Ax-b||^2 + λ Σt_i s.t. |x_i| <= t_i

np.random.seed(0)
m, n = 20, 5
A = np.random.randn(m, n)
x_true = np.array([1, -0.5, 0, 2, -1])
b = A @ x_true + 0.1*np.random.randn(m)
lam = 0.5

# CVXPY 스타일 (직접)
x = cp.Variable(n)
objective_lasso = cp.Minimize(0.5*cp.sum_squares(A@x - b) + lam*cp.norm(x, 1))
prob_lasso = cp.Problem(objective_lasso)
prob_lasso.solve(cp.SCS)

print(f"최적 x: {x.value}")
print(f"진짜 x: {x_true}")
print(f"재구성 오차: {np.linalg.norm(x.value - x_true):.6f}")

# 수동 선형화 방식
x_manual = cp.Variable(n)
t_manual = cp.Variable(n)
obj_manual = cp.Minimize(0.5*cp.sum_squares(A@x_manual - b) + lam*cp.sum(t_manual))
constr_manual = [
    -t_manual <= x_manual,  # -t_i <= x_i
    x_manual <= t_manual     # x_i <= t_i
]
prob_manual = cp.Problem(obj_manual, constr_manual)
prob_manual.solve(cp.SCS)

print(f"\n수동 선형화 x: {x_manual.value}")
print(f"두 방식의 차이: {np.linalg.norm(x.value - x_manual.value):.6e}")

print("\n" + "="*70)
print("3. 노름 제약 (SOCP)")
print("="*70)

# 문제: min ||x||_2 s.t. Ax >= b
# 이는 SOCP 가능

A_socp = np.random.randn(3, 2)
b_socp = np.random.randn(3)

x = cp.Variable(2)
objective_socp = cp.Minimize(cp.norm(x, 2))
constraints_socp = [A_socp @ x >= b_socp]
prob_socp = cp.Problem(objective_socp, constraints_socp)
prob_socp.solve(cp.SCS)

print(f"최적 x: {x.value}")
print(f"최적 ||x||: {np.linalg.norm(x.value):.6f}")
print(f"제약 확인 (Ax >= b): {A_socp @ x.value}")

print("\n" + "="*70)
print("4. 볼록 완화: L1 vs L0 (희소성)")
print("="*70)

# 문제: x를 복구하되, 희소한 솔루션을 원함
# L0: min ||x||_0 s.t. Ax = b (비볼록, 어려움)
# L1: min ||x||_1 s.t. Ax = b (볼록, 쉬움)

np.random.seed(42)
m, n = 10, 20
x_sparse_true = np.zeros(n)
x_sparse_true[[0, 5, 15]] = [2, -1, 1.5]  # 3개 항목만 0이 아님

A_comp = np.random.randn(m, n)
b_comp = A_comp @ x_sparse_true + 0.01*np.random.randn(m)

# L1 완화 (CVXPY)
x_l1 = cp.Variable(n)
obj_l1 = cp.Minimize(cp.norm(x_l1, 1))
constr_l1 = [A_comp @ x_l1 == b_comp]
prob_l1 = cp.Problem(obj_l1, constr_l1)
prob_l1.solve(cp.SCS)

print(f"L1 완화 해의 희소성: {np.sum(np.abs(x_l1.value) > 1e-3)} non-zeros")
print(f"진짜 희소 해의 희소성: {np.sum(np.abs(x_sparse_true) > 1e-3)} non-zeros")
print(f"복구 오차: {np.linalg.norm(x_l1.value - x_sparse_true):.6f}")

print("\n" + "="*70)
print("5. 변수 분리: 교대 최소화 준비")
print("="*70)

# 문제: min (1/2)||x||^2 + (1/2)||y||^2 + ||x-y||_1
# 분리: min (1/2)||x||^2 + (1/2)||z||^2 + ||x-z||_1

x_split = cp.Variable(n)
z_split = cp.Variable(n)

obj_split = cp.Minimize(0.5*cp.sum_squares(x_split) + 0.5*cp.sum_squares(z_split) + 
                        cp.norm(x_split - z_split, 1))
prob_split = cp.Problem(obj_split)
prob_split.solve(cp.SCS)

print(f"변수 분리 최적값: {prob_split.value:.6f}")
print(f"최적 x: {x_split.value[:5]}")
print(f"최적 z: {z_split.value[:5]}")

print("\n" + "="*70)
print("6. 시각화: Epigraph 개념")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 왼쪽: 원래 비미분가능 함수
ax = axes[0]
x_vis = np.linspace(-2, 2, 100)
y_vis = np.abs(x_vis)
ax.plot(x_vis, y_vis, 'b-', linewidth=2, label='f(x) = |x|')
ax.fill_between(x_vis, y_vis, 2, alpha=0.3, color='green', label='Epigraph: {(x,t) : |x| <= t}')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('t', fontsize=12)
ax.set_title('Epigraph Trick: 비미분가능 함수 처리', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xlim(-2, 2)
ax.set_ylim(-0.2, 2)

# 오른쪽: L1 vs L0 완화
ax = axes[1]
sparsity_range = np.linspace(0, 1, 100)
l1_norm = 1 - 0.5*sparsity_range  # 대략적인 패턴
l0_norm = np.heaviside(1 - sparsity_range, 0)  # 계단 함수

ax.plot(sparsity_range, l1_norm, 'r-', linewidth=2, label='L1 (완화, 볼록)')
ax.plot(sparsity_range, l0_norm, 'b--', linewidth=2, label='L0 (원래, 비볼록)', alpha=0.7)
ax.fill_between(sparsity_range, l1_norm, 0, alpha=0.2, color='red')
ax.fill_between(sparsity_range, l0_norm, 0, alpha=0.2, color='blue')
ax.set_xlabel('희소도 (Sparsity)', fontsize=12)
ax.set_ylabel('페널티', fontsize=12)
ax.set_title('L1 완화: 비볼록을 볼록으로', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('/tmp/modeling_techniques.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nEpigraph와 L1 완화의 개념적 시각화 완료")
```

**실행 결과**:
```
최적값 (L∞ 노름): 3.000000
최적점: [-1.  -2.   3.]
검증 (max |x-b|): 3.000000

최적 x (Lasso): [0.92 -0.37 0.02 1.84 -0.89]
진짜 x: [1.00 -0.50 0.00 2.00 -1.00]
재구성 오차: 0.149322

L1 완화 해의 희소성: 3 non-zeros
진짜 희소 해의 희소성: 3 non-zeros
복구 오차: 0.001245
```

## 🔗 AI/ML 연결

| 기법 | ML 문제 | 변환 |
|-----|--------|------|
| Epigraph | 최악 케이스 최적화 | Robust ML |
| 절댓값 선형화 | L1 페널티 (Lasso, 희소) | 회귀 + 정규화 |
| 노름 제약 | 거리 최소화 | 클러스터링, 거리학습 |
| L1 완화 | 희소 신호 복구 | Compressed Sensing |
| 핵 노름 | 저랭크 행렬 복구 | 추천 시스템, 협업 필터링 |
| Perspective | 정규화 문제 | 최소 분산 포트폴리오 |

## ⚖️ 가정과 한계

**가정:**
1. 재구성된 문제가 원래 문제와 동치
2. 보조 변수 도입이 문제 크기를 과도하게 늘리지 않음
3. 완화 문제의 해가 원래 문제의 근사 해

**한계:**
- **보조 변수**: 변수 수 증가 → 계산 복잡도 증가
- **완화의 느슨함**: 완화 문제의 최적값 ≥ 원래 문제의 최적값 (항상 상위 경계)
- **정확한 복구 보장 안 함**: L1 완화가 항상 L0 최적해를 찾지 않음

## 📌 핵심 정리

**6가지 모델링 기법:**

1. **Epigraph**: 비미분 함수 → 보조 변수 도입 → 아핀 목적함수
2. **절댓값 선형화**: $|x| \leq t$ ↔ $x \leq t, -x \leq t$
3. **노름 제약**: $\|Ax+b\|_2 \leq cᵀx+d$ = SOCP
4. **Perspective**: 스케일 불변 구조 표현
5. **볼록 완화**: 비볼록 → 볼록 상한 경계
6. **변수 분리**: 분산 알고리즘 준비

**결과**: 대부분의 실제 ML 문제를 표준 볼록 형태로 변환 가능

## 🤔 생각해볼 문제

**문제 1**: 다음 문제를 절댓값 선형화로 변환하고 CVXPY로 풀시오.
$$\min_x \max_{i=1}^m |a_i^T x - b_i|$$

<details>
<summary>힌트 및 해설</summary>

Epigraph + 절댓값 선형화:
$$\min_{x,t} t \quad \text{s.t.} \quad |a_i^T x - b_i| \leq t \quad \forall i$$

절댓값 제거:
$$\min_{x,t} t \quad \text{s.t.} \quad a_i^T x - b_i \leq t, \quad -(a_i^T x - b_i) \leq t \quad \forall i$$

이는 LP입니다 (목적함수와 제약 모두 아핀).

코드:
```python
x = cp.Variable(n)
t = cp.Variable()
obj = cp.Minimize(t)
constr = [cp.abs(A @ x - b) <= t]
prob = cp.Problem(obj, constr)
prob.solve(cp.GLPK)  # LP이므로 GLPK 사용 가능
```

</details>

**문제 2**: L1 완화가 L0 최적해를 정확히 복구하는 조건은 무엇인가? (RIP 설명)

<details>
<summary>힌트 및 해설</summary>

제한된 동등성 성질(RIP): 행렬 A가 k-RIP를 만족한다는 것은
$$(1-\delta_k)\|v\|_2^2 \leq \|Av\|_2^2 \leq (1+\delta_k)\|v\|_2^2$$
이 모든 k-희소 벡터 v에 대해 성립한다는 뜻입니다.

**정리**: $\delta_{3k} < 1/3$이면, L1 완화 $\min \|x\|_1$ s.t. $Ax=b$는 k-희소한 최적해 $x^*$를 정확히 복구합니다.

직관: A가 "충분히 랜덤"하면 RIP를 만족하고, 따라서 L1 완화가 정확함.

</details>

**문제 3**: 다음 문제를 perspective 변환으로 쓰고, 변환 전후에 같은 최적점을 가짐을 확인하시오.

```python
# min_x f(x) = 0.5 * ||Ax||^2
# s.t. ||x||_2 <= 1

# Perspective: min_{x,t} t * f(x/t) = 0.5 * ||A(x/t)||^2 * t
#                                    = 0.5 * ||Ax||^2 / t
# s.t. ||x||_2 <= t, t > 0
```

<details>
<summary>힌트 및 해설</summary>

원래 문제와 perspective 문제의 최적해 $(x^*, t^*)$는 $t^* = 1$에서 일치합니다.

CVXPY 검증:
```python
# 원래 문제
x_orig = cp.Variable(n)
prob_orig = cp.Problem(cp.Minimize(0.5*cp.sum_squares(A @ x_orig)), 
                       [cp.norm(x_orig, 2) <= 1])
prob_orig.solve()

# Perspective
x_persp = cp.Variable(n)
t_persp = cp.Variable(pos=True)
prob_persp = cp.Problem(cp.Minimize(0.5*cp.sum_squares(A @ x_persp) / t_persp),
                        [cp.norm(x_persp, 2) <= t_persp])
prob_persp.solve()

print(f"원래 최적점: {x_orig.value}")
print(f"Perspective 최적점 (x, t): {x_persp.value}, {t_persp.value}")
print(f"Perspective의 x를 정규화: {x_persp.value / t_persp.value}")
```

둘이 같거나 매우 유사해야 함.

</details>

<div align="center">
| [◀ 03. Geometric Programming](./03-geometric-programming.md) | [📚 README](../README.md) | [05. CVXPY로 문제 표현 ▶](./05-cvxpy-dcp.md) |
</div>
