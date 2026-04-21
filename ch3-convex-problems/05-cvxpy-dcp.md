# 5. CVXPY로 문제 표현 (DCP 규칙)

## 🎯 핵심 질문
내가 쓴 CVXPY 코드가 정말 볼록인가? "우리는 컴파일러가 아니다" - 자동으로 볼록성을 검증하는 방법은? DCP (Disciplined Convex Programming) 규칙이란?

## 🔍 왜 이 이론이 AI에서 중요한가
CVXPY는 수학적 선언만으로 복잡한 최적화 문제를 푸는 도구입니다. 하지만 "이 함수 조합이 볼록인가?"를 판단하는 것은 수학적으로 미묘합니다. DCP 규칙은 **프로그래밍 규칙(discipline)**을 따르면 자동으로 볼록성을 보장하는 체계입니다. 이를 이해하면 CVXPY 에러를 디버깅하고, 문제를 올바르게 표현할 수 있습니다.

## 📐 수학적 선행 조건
- 볼록/오목/아핀 함수의 정의
- 함수 조합의 볼록성 (composition rules)
- CVXPY 기본 문법

## 📖 직관적 이해

**DCP의 핵심 아이디어: "자동 형식 검증"**

```
f(x) = h(g1(x), g2(x), ..., gk(x))
```

h와 각 gi의 곡률(curvature)을 알면, 전체 f의 곡률을 규칙에 따라 결정할 수 있습니다.

**곡률 종류:**
- **Convex (⊕)**: 밥그릇 모양 (아래로 볼록)
- **Concave (⊖)**: 산 모양 (위로 볼록)
- **Affine (◉)**: 직선 (동시에 볼록+오목)
- **Unknown (?)**: 판단 불가

**조합 규칙 예:**
- 볼록 + 볼록 = 볼록 ✓
- 볼록 + 오목 = ❌ (판단 불가)
- 볼록 × (양수) = 볼록 ✓
- 오목 × (음수) = 볼록 ✓
- f(g(x)): f 볼록이고 g 증가함수 → f∘g 볼록

## ✏️ 엄밀한 정의

### 정의 1: Atom과 그 곡률

**Atom**: CVXPY가 미리 정의한 기본 함수들 (변환 불가능)

| 함수 | 곡률 | 정의역 | 예 |
|------|------|--------|-----|
| $x$ | Affine | ℝ | 선형 함수 |
| $\|x\|$ | Convex | ℝ | L1 노름 |
| $\sqrt{x}$ | Concave | ℝ+ | 제곱근 |
| $x^2$ | Convex | ℝ | 제곱 |
| $1/x$ | Convex | ℝ+ | 역함수 |
| $\log(x)$ | Concave | ℝ+ | 로그 |
| $\exp(x)$ | Convex | ℝ | 지수 |
| $\max(x_1, ..., x_n)$ | Convex | ℝⁿ | 최댓값 |
| $\sum x_i$ | Affine | ℝⁿ | 합 |
| $\|x\|_2$ | Convex | ℝⁿ | L2 노름 |
| $\|x\|_1$ | Convex | ℝⁿ | L1 노름 |
| $\|x\|_∞$ | Convex | ℝⁿ | L∞ 노름 |

### 정의 2: 함수 조합의 곡률 전파

**규칙 1**: 볼록 함수의 양수 스칼라 곱
$$c \geq 0, \; f \text{ convex} \quad \Rightarrow \quad cf \text{ convex}$$

**규칙 2**: 오목 함수의 음수 스칼라 곱
$$c \leq 0, \; f \text{ concave} \quad \Rightarrow \quad cf \text{ convex}$$

**규칙 3**: 동일 곡률의 합
$$f_1, f_2 \text{ convex} \quad \Rightarrow \quad f_1 + f_2 \text{ convex}$$

**규칙 4**: 내부 함수의 곡률 변화

**Chain Rule**: $f(g(x))$의 곡률
- f 볼록, g 증가, g 볼록 → f∘g 볼록
- f 볼록, g 감소, g 오목 → f∘g 볼록
- f 오목, g 증가, g 오목 → f∘g 오목
- f 오목, g 감소, g 볼록 → f∘g 오목

**규칙 5**: 상한함수(perspective)
$$f(x) \text{ convex} \quad \Rightarrow \quad g(x, t) = t \cdot f(x/t) \text{ convex} \; (t > 0)$$

### 정의 3: DCP-compliant 표현의 4계층

```
원래 문제 (수학적)
    ↓
Atom 분해 (기본 함수로 표현)
    ↓
곡률 전파 (DCP 규칙 적용)
    ↓
가능 여부 판단 (DCP-compliant? Yes/No)
    ↓
문제 해결 또는 재구성
```

## 🔬 정리와 증명

### 정리 1: DCP 조합 규칙의 완전성

**명제**: 다음 함수 조합 규칙을 따르면, 생성되는 함수의 볼록성이 자동으로 결정된다.

1. 모든 atom의 곡률을 알 수 있다
2. 곡률 전파 규칙을 재귀적으로 적용 가능
3. 결과는 항상 {Convex, Concave, Affine, Unknown} 중 하나

**의의**: 프로그래머가 복잡한 함수의 볼록성을 수동으로 증명할 필요 없음

---

### 정리 2: 내부 함수 규칙 증명 (체인룰)

**명제**: $f$ 볼록, $g$ 증가함수이고 볼록이면, $f(g(x))$ 볼록

**증명**:
$$f(g(\theta x + (1-\theta)y))$$
$$\leq f(\theta g(x) + (1-\theta)g(y)) \quad (\text{f 볼록})$$
$$\leq \theta f(g(x)) + (1-\theta) f(g(y)) \quad (\text{f 증가, g 볼록})$$

따라서 $f \circ g$ 볼록. $\square$

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import cvxpy as cp
from cvxpy.expressions.expression import Expression
import warnings

print("="*70)
print("1. DCP 규칙의 기본: Atom 곡률")
print("="*70)

# 변수
x = cp.Variable(5, name='x')

# 기본 Atom들의 곡률 확인
atoms = {
    'x (linear)': x,
    'x**2 (convex)': x**2,
    '|x| (convex)': cp.abs(x),
    'sqrt(x) (concave)': cp.sqrt(x),
    'exp(x) (convex)': cp.exp(x),
    'log(x) (concave)': cp.log(x),
    'norm(x, 2) (convex)': cp.norm(x, 2),
    'sum(x) (affine)': cp.sum(x),
}

for name, expr in atoms.items():
    print(f"{name:30} → DCP info: {expr.curvature}")

print("\n" + "="*70)
print("2. DCP 조합 규칙")
print("="*70)

# 규칙 1: 볼록 + 볼록 = 볼록
expr1 = cp.norm(x, 2) + x**2
print(f"norm(x,2) + x**2 = {expr1.curvature} ✓")

# 규칙 2: 볼록 - 오목 = ? (원인: 음수로 곱한 오목)
try:
    expr2 = cp.norm(x, 2) - cp.sqrt(x)
    print(f"norm(x,2) - sqrt(x) = {expr2.curvature}")
except:
    print(f"norm(x,2) - sqrt(x) = Error (정의역 문제)")

# 규칙 3: 양수 × 볼록 = 볼록
expr3 = 2 * cp.norm(x, 2)
print(f"2 * norm(x,2) = {expr3.curvature} ✓")

# 규칙 4: 음수 × 오목 = 볼록
expr4 = -1 * cp.sqrt(x)  # -sqrt는 convex (오목의 음수배)
print(f"-1 * sqrt(x) = {expr4.curvature} ✓")

# 규칙 5: 체인룰 - f 볼록 + g 증가, 볼록 = f(g(x)) 볼록
expr5 = cp.sqrt(cp.norm(x, 2))
print(f"sqrt(norm(x,2)) = {expr5.curvature} ✓ (sqrt 오목 + norm 증가, 볼록)")

print("\n" + "="*70)
print("3. 실전: Lasso (DCP-compliant)")
print("="*70)

# Lasso: min (1/2)||Ax - b||^2 + λ||x||_1

np.random.seed(0)
m, n = 30, 10
A = np.random.randn(m, n)
x_true = np.random.randn(n)
x_true[5:] = 0  # 희소
b = A @ x_true + 0.1*np.random.randn(m)
lam = 0.1

x = cp.Variable(n)

# 목적함수: (1/2) norm(Ax-b,2)^2 + λ norm(x,1)
#          = convex + convex = convex ✓
objective = cp.Minimize(0.5*cp.sum_squares(A@x - b) + lam*cp.norm(x, 1))

prob = cp.Problem(objective)

# DCP 검증
print(f"문제 DCP 상태: {prob.is_dcp()}")
print(f"문제는 convex인가: {prob.is_convex()}")

prob.solve(cp.SCS, verbose=False)

print(f"최적값: {prob.value:.6f}")
print(f"최적 x (처음 5개): {x.value[:5]}")
print(f"스파시티 (0 아닌 개수): {np.sum(np.abs(x.value) > 1e-3)}")

print("\n" + "="*70)
print("4. 실전: SVM (Soft-Margin)")
print("="*70)

# SVM: min (1/2)||w||^2 + C sum(ξ_i)
#      s.t. y_i(w^T x_i + b) >= 1 - ξ_i, ξ_i >= 0

# 데이터 생성
n_samples = 20
n_features = 5
X = np.random.randn(n_samples, n_features)
y = np.random.choice([-1, 1], n_samples)
C = 1.0

w = cp.Variable(n_features)
b = cp.Variable()
xi = cp.Variable(n_samples, nonneg=True)

# 목적함수: (1/2)||w||^2 + C sum(ξ) = convex ✓
objective_svm = cp.Minimize(0.5*cp.sum_squares(w) + C*cp.sum(xi))

# 제약: y_i(w^T x_i + b) >= 1 - ξ_i
# ⟺ w^T x_i * y_i + b*y_i >= 1 - ξ_i (아핀 >= 아핀) ✓
constraints_svm = []
for i in range(n_samples):
    constraints_svm.append(y[i]*(w @ X[i] + b) >= 1 - xi[i])

prob_svm = cp.Problem(objective_svm, constraints_svm)

print(f"SVM 문제 DCP: {prob_svm.is_dcp()}")
print(f"SVM 문제는 convex: {prob_svm.is_convex()}")

prob_svm.solve(cp.SCS, verbose=False)

print(f"최적 목적값: {prob_svm.value:.6f}")
print(f"최적 ||w||: {np.linalg.norm(w.value):.6f}")

print("\n" + "="*70)
print("5. 실전: Logistic Regression")
print("="*70)

# Logistic: min sum log(1 + exp(-y_i w^T x_i))

x_log = cp.Variable(n_features)
X_log = np.random.randn(n_samples, n_features)
y_log = np.random.choice([-1, 1], n_samples)

# 목적함수: sum log(1 + exp(-y_i * w^T x))
#          = sum log_sum_exp([0, -y_i w^T x])
#          = convex (log-sum-exp 볼록) ✓
objective_lr = cp.Minimize(
    cp.sum([cp.logistic(-y_log[i]*(x_log @ X_log[i])) for i in range(n_samples)])
)

prob_lr = cp.Problem(objective_lr)

print(f"로지스틱 문제 DCP: {prob_lr.is_dcp()}")

prob_lr.solve(cp.SCS, verbose=False)

print(f"최적 목적값: {prob_lr.value:.6f}")

print("\n" + "="*70)
print("6. 실전: Portfolio Optimization")
print("="*70)

# 포트폴리오: min x^T Σ x
#             s.t. μ^T x >= r, 1^T x = 1, x >= 0
#
# 목적: min (convex quadratic)
# 제약: 아핀 + 비음수 = convex ✓

n_assets = 5
Sigma = np.random.randn(n_assets, n_assets)
Sigma = Sigma @ Sigma.T + np.eye(n_assets)  # 양정치
mu = np.random.rand(n_assets) + 0.5
target_return = 0.7

x_port = cp.Variable(n_assets, nonneg=True)

objective_port = cp.Minimize(cp.quad_form(x_port, Sigma))
constraints_port = [
    mu @ x_port >= target_return,
    cp.sum(x_port) == 1
]

prob_port = cp.Problem(objective_port, constraints_port)

print(f"포트폴리오 문제 DCP: {prob_port.is_dcp()}")
print(f"포트폴리오 문제는 convex: {prob_port.is_convex()}")

prob_port.solve(cp.SCS, verbose=False)

print(f"최적 분산: {prob_port.value:.6f}")
print(f"포트폴리오 비중 (첫 3개): {x_port.value[:3]}")

print("\n" + "="*70)
print("7. DCP 위반 예제와 에러 해석")
print("="*70)

# 에러 1: 오목 함수를 minimize (비볼록)
try:
    x_bad = cp.Variable(3, pos=True)
    obj_bad = cp.Minimize(cp.sqrt(cp.sum(x_bad)))  # sqrt(sum) 오목
    prob_bad = cp.Problem(obj_bad)
    print(f"sqrt(sum(x)) 최소화: DCP = {prob_bad.is_dcp()} ❌")
except Exception as e:
    print(f"sqrt(sum(x)) 최소화: Error - {type(e).__name__}")

# 에러 2: 곱셈 (일반적으로 DCP 위반)
try:
    x_prod = cp.Variable(2)
    obj_prod = cp.Minimize(x_prod[0] * x_prod[1])  # 쌍선형, 비볼록
    prob_prod = cp.Problem(obj_prod)
    print(f"x*y 최소화: DCP = {prob_prod.is_dcp()} ❌")
except Exception as e:
    print(f"x*y 최소화: Error - {type(e).__name__}")

# 에러 3: 역함수 체인 (DCP 위반)
try:
    x_inv = cp.Variable(2, pos=True)
    obj_inv = cp.Minimize(cp.inv_pos(cp.sum(x_inv)))  # 1/sum(x), 비볼록 조합
    prob_inv = cp.Problem(obj_inv)
    print(f"1/sum(x) 최소화: DCP = {prob_inv.is_dcp()} ✓ (사실 이건 오케이)")
except Exception as e:
    print(f"1/sum(x) 최소화: Error")

print("\n" + "="*70)
print("DCP 규칙 요약")
print("="*70)
print("""
✓ 가능한 조합:
  - convex + convex = convex
  - concave + concave = concave
  - c*convex (c≥0) = convex
  - c*concave (c≤0) = convex
  - f(g(x)): f convex + g 증가·convex = convex

✗ 불가능한 조합:
  - convex + concave = Unknown (판단 불가)
  - convex - convex = Unknown (두 번째가 음수배)
  - minimize(concave) = 비볼록 (최댓값)
  - x*y (곱셈) = 일반 비볼록
""")

print("\n" + "="*70)
print("8. 시각화: DCP 플로우")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 왼쪽 위: Lasso 경로
ax = axes[0, 0]
lam_range = np.linspace(0.01, 1, 50)
x_lasso_path = []
for lam in lam_range:
    x_temp = cp.Variable(n)
    obj_temp = cp.Minimize(0.5*cp.sum_squares(A @ x_temp - b) + lam*cp.norm(x_temp, 1))
    prob_temp = cp.Problem(obj_temp)
    prob_temp.solve(cp.SCS, verbose=False)
    x_lasso_path.append(np.linalg.norm(x_temp.value))

ax.plot(lam_range, x_lasso_path, 'b-', linewidth=2)
ax.set_xlabel('λ (정규화)', fontsize=11)
ax.set_ylabel('||x|| (L2)', fontsize=11)
ax.set_title('Lasso 경로: λ증가 → 희소성증가', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 오른쪽 위: SVM 마진
ax = axes[0, 1]
np.random.seed(42)
X_plot = np.random.randn(50, 2)
y_plot = np.sign(X_plot[:, 0] + X_plot[:, 1] - 0.5)

w_plot = cp.Variable(2)
b_plot = cp.Variable()
xi_plot = cp.Variable(50, nonneg=True)

obj_svm_plot = cp.Minimize(cp.sum_squares(w_plot) + 10*cp.sum(xi_plot))
constr_svm_plot = [y_plot[i]*(w_plot @ X_plot[i] + b_plot) >= 1 - xi_plot[i] 
                   for i in range(50)]
prob_svm_plot = cp.Problem(obj_svm_plot, constr_svm_plot)
prob_svm_plot.solve(cp.SCS, verbose=False)

# 분류 경계 그리기
xx = np.linspace(-4, 4, 100)
if abs(w_plot.value[0]) > 1e-6:
    yy = -(w_plot.value[0]*xx + b_plot.value)/w_plot.value[1]
    ax.plot(xx, yy, 'k-', linewidth=2, label='결정 경계')
    ax.plot(xx, yy + 1/np.linalg.norm(w_plot.value), 'k--', alpha=0.5, label='마진')
    ax.plot(xx, yy - 1/np.linalg.norm(w_plot.value), 'k--', alpha=0.5)

colors = ['red' if yi > 0 else 'blue' for yi in y_plot]
ax.scatter(X_plot[:, 0], X_plot[:, 1], c=colors, alpha=0.6, s=50)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel('x₁', fontsize=11)
ax.set_ylabel('x₂', fontsize=11)
ax.set_title('SVM: 최댓값 마진 분류', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 왼쪽 아래: 포트폴리오 효율 경계
ax = axes[1, 0]
target_returns = np.linspace(0.5, 1.2, 20)
min_risks = []
for ret in target_returns:
    x_eff = cp.Variable(n_assets, nonneg=True)
    obj_eff = cp.Minimize(cp.quad_form(x_eff, Sigma))
    constr_eff = [mu @ x_eff >= ret, cp.sum(x_eff) == 1]
    prob_eff = cp.Problem(obj_eff, constr_eff)
    prob_eff.solve(cp.SCS, verbose=False)
    min_risks.append(np.sqrt(prob_eff.value))

ax.plot(min_risks, target_returns, 'g-', linewidth=2, marker='o', markersize=4)
ax.set_xlabel('위험도 (표준편차)', fontsize=11)
ax.set_ylabel('수익률', fontsize=11)
ax.set_title('효율 경계: 위험-수익 트레이드오프', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 오른쪽 아래: DCP 규칙 플로우
ax = axes[1, 1]
ax.text(0.5, 0.95, 'DCP 규칙 플로우', fontsize=13, fontweight='bold', 
        ha='center', transform=ax.transAxes)

rules_text = """
1️⃣  Atom 곡률 확인
   (convex, concave, affine)

2️⃣  조합 규칙 적용
   ➜ convex + convex = convex ✓
   ➜ concave + concave = concave ✓
   ➜ convex + concave = ? ❌

3️⃣  체인룰 확인
   f∘g: f convex + g단조증가·convex

4️⃣  DCP-compliant 결정
   prob.is_dcp() → True/False

5️⃣  수치 해결
   CVXPY는 자동으로 솔버 선택
"""

ax.text(0.05, 0.7, rules_text, fontsize=10, family='monospace',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.axis('off')

plt.tight_layout()
plt.savefig('/tmp/cvxpy_dcp.png', dpi=150, bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
```

## 🔗 AI/ML 연결

| ML 알고리즘 | CVXPY 표현 | DCP 상태 | 목적함수 곡률 |
|-----------|----------|--------|----------|
| 선형 회귀 | `norm(Ax-b)^2` | ✓ | Convex |
| Ridge | `norm(Ax-b)^2 + λ norm(x)^2` | ✓ | Convex |
| Lasso | `norm(Ax-b)^2 + λ norm(x,1)` | ✓ | Convex |
| SVM | `norm(w)^2 + C∑ξ` | ✓ | Convex |
| 로지스틱 | `∑log(1+exp(...))` | ✓ | Convex |
| 포트폴리오 | `x^T Σ x` | ✓ | Convex |
| 클러스터링 | K-means 비목적 | ❌ | Non-convex |
| Neural Network | MSE+ReLU | ❌ | Non-convex |

## ⚖️ 가정과 한계

**가정:**
1. 모든 함수가 DCP 규칙 형태로 표현 가능
2. 변수 정의역이 명확함 (양수, 행렬 등)
3. 곡률이 원래 수학적으로 맞음

**한계:**
- **DCP 제약**: 모든 볼록 함수를 직접 표현 불가
- **복합 함수**: 복잡한 함수 조합은 DCP 위반 가능
- **수치 안정성**: Atom이 수치적으로 부정확할 수 있음

## 📌 핵심 정리

**DCP (Disciplined Convex Programming)**: 볼록성을 프로그래밍으로 검증

**4가지 곡률**: Convex (⊕), Concave (⊖), Affine (◉), Unknown (?)

**5가지 조합 규칙:**
1. 볼록 + 볼록 = 볼록
2. 양수 × 볼록 = 볼록
3. 음수 × 오목 = 볼록
4. 체인룰: f 볼록 + g 증가, 볼록 = f∘g 볼록
5. Perspective: f 볼록 → t·f(x/t) 볼록

**결과**: DCP를 따르면 `prob.is_dcp() == True` → 자동으로 글로벌 최솟값 계산 가능

## 🤔 생각해볼 문제

**문제 1**: 다음 CVXPY 식의 곡률을 예측하고, `is_dcp()`로 검증하시오.

```python
x = cp.Variable(5)
expr = cp.norm(x, 2) + cp.sum(x**2) - 2*cp.sqrt(cp.sum(x))
prob = cp.Problem(cp.Minimize(expr))
print(f"DCP: {prob.is_dcp()}")  # True인가?
```

<details>
<summary>힌트 및 해설</summary>

각 항의 곡률:
- `norm(x,2)` = Convex (atom)
- `sum(x**2)` = Convex (convex 합)
- `-2*sqrt(sum(x))` = ? (음수 × 오목 = Convex)

전체: Convex + Convex + Convex = **Convex ✓**

따라서 `prob.is_dcp() == True`이고, 최소화 문제로 유효합니다.

</details>

**문제 2**: 왜 `x*y` (곱셈)는 DCP로 표현할 수 없는가? DCP-compliant 방식으로 재구성하시오.

<details>
<summary>힌트 및 해설</summary>

`x*y` 자체는 쌍선형(bilinear)으로, 일반적으로 비볼록입니다.

DCP-compliant 방식:
```python
# x*y를 최소화하되, ||x||, ||y||에 상한이 있으면
# SOCP나 SDP로 완화 가능

# 예: min x*y s.t. ||x||<=1, ||y||<=1
# 대신: min t s.t. t >= xy (또는 SDP 완화)

# CVXPY에서는 직접 표현 불가 → 완화 필요
```

예를 들어 행렬 변수 X = xy^T로 바꾸고 `rank(X) <= 1` 완화:
```python
X = cp.Variable((n, m))
obj = cp.Minimize(cp.trace(C @ X))
constr = [X >> 0]  # SDP 완화
```

</details>

**문제 3**: CVXPY로 다음 문제를 풀고, DCP 상태를 확인하고, `is_dcp()`의 결과를 설명하시오.

```python
import cvxpy as cp
import numpy as np

x = cp.Variable(3)
A = np.random.randn(2, 3)
b = np.random.randn(2)

# 문제 1: min ||Ax - b||_2^2 (명백히 convex)
prob1 = cp.Problem(cp.Minimize(cp.sum_squares(A@x - b)))

# 문제 2: min ||Ax - b||_2 s.t. ||x||_2 <= 1
prob2 = cp.Problem(cp.Minimize(cp.norm(A@x - b, 2)), [cp.norm(x, 2) <= 1])

# 문제 3: min log(||Ax-b||_2^2) (로그-오목?)
prob3 = cp.Problem(cp.Minimize(cp.log(cp.sum_squares(A@x - b))))

print(f"prob1.is_dcp(): {prob1.is_dcp()}")
print(f"prob2.is_dcp(): {prob2.is_dcp()}")
print(f"prob3.is_dcp(): {prob3.is_dcp()}")
```

<details>
<summary>힌트 및 해설</summary>

- **prob1**: `sum_squares` = Convex atom → minimize(convex) ✓ → **True**
- **prob2**: `norm` = Convex, 제약도 convex → **True**
- **prob3**: `log(convex)` = ❌ Concave composition → minimize(concave) = 최댓값 → **False**

문제 3은 DCP 위반입니다. 만약 최소화하고 싶으면:
```python
# 불가: minimize(log(convex))

# 대신: minimize(convex하게 표현된 log)
# 예: minimize ||Ax-b||_2를 사용 (log는 단조이므로)
```

</details>

<div align="center">
| [◀ 04. 모델링 기법](./04-modeling-techniques.md) | [📚 README](../README.md) | [Ch4-01. Lagrangian과 쌍대 함수 ▶](../ch4-duality/01-lagrangian-dual-function.md) |
</div>
