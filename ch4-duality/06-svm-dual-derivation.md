# 6. SVM의 쌍대 유도 — 완전판

## 🎯 핵심 질문
- SVM의 원 문제와 쌍대 문제는 어떻게 유도되는가?
- 왜 쌍대 형식이 계산상 유리한가?
- Support vector의 개념은 KKT 조건에서 어떻게 나타나는가?

## 🔍 왜 이 이론이 AI에서 중요한가
SVM은 현대 머신러닝의 고전적 알고리즘이며, 쌍대 형식은 커널 메서드를 적용하는 열쇠입니다. 고차원 특성 공간에서 계산 가능한 이유가 쌍대 형식입니다. 또한 이 유도 과정은 다른 제약 최적화 문제의 표본이 됩니다.

## 📐 수학적 선행 조건
- Lagrangian과 쌍대 함수 (Chapter 4-1)
- 강쌍대성과 Slater 조건 (Chapter 4-3)
- KKT 조건 (Chapter 4-4)
- 커널 함수의 개념

## 📖 직관적 이해
SVM은 두 클래스 사이의 "최대 마진"을 찾는 문제입니다. 원 형식(primal)에서는 가중치 w를 직접 최적화하지만, 쌍대 형식에서는 각 표본의 가중치 α를 최적화합니다. 쌍대 형식의 이점은 특성의 내적만 필요하므로 커널 트릭 적용 가능하다는 것입니다.

## ✏️ 엄밀한 정의

**Hard-margin SVM 원 문제:**

주어진: 표본 $(x_i, y_i)$, $i = 1,...,n$, 여기서 $x_i \in \mathbb{R}^d$, $y_i \in \{-1, +1\}$

```
minimize   (1/2)‖w‖²
subject to yᵢ(w^T xᵢ + b) ≥ 1,  i = 1,...,n
```

**Soft-margin SVM 원 문제 (C > 0):**

```
minimize   (1/2)‖w‖² + C Σᵢ ξᵢ
subject to yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ,  i = 1,...,n
          ξᵢ ≥ 0,  i = 1,...,n
```

## 🔬 정리와 증명

**정리 1: Hard-margin SVM의 쌍대 유도 (완전 증명)**

*Primal 문제:*
$$\min_{w,b} \frac{1}{2}\|w\|^2$$
$$\text{s.t. } y_i(w^T x_i + b) \geq 1, \quad i = 1,...,n$$

*Step 1: Lagrangian 구성*

$$L(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^n \alpha_i [y_i(w^T x_i + b) - 1]$$

여기서 $\alpha_i \geq 0$는 쌍대 변수입니다.

*Step 2: 정류 조건 계산*

$$\frac{\partial L}{\partial w} = w - \sum_{i=1}^n \alpha_i y_i x_i = 0$$
$$\Rightarrow w^* = \sum_{i=1}^n \alpha_i y_i x_i \quad \cdots (*)$$

$$\frac{\partial L}{\partial b} = -\sum_{i=1}^n \alpha_i y_i = 0$$
$$\Rightarrow \sum_{i=1}^n \alpha_i y_i = 0 \quad \cdots (**)$$

*Step 3: w 대입하여 쌍대 함수 계산*

$(*)$를 Lagrangian에 대입:

$$g(\alpha) = \frac{1}{2}\left\|\sum_{i=1}^n \alpha_i y_i x_i\right\|^2 - \sum_{i=1}^n \alpha_i y_i \left[\left(\sum_{j=1}^n \alpha_j y_j x_j\right)^T x_i + b\right] + \sum_{i=1}^n \alpha_i$$

$$= \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j - \sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_j^T x_i - b\sum_{i=1}^n \alpha_i y_i + \sum_{i=1}^n \alpha_i$$

처음 두 항이 같으므로:

$$g(\alpha) = -\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j + \sum_{i=1}^n \alpha_i - b\sum_{i=1}^n \alpha_i y_i$$

$(**)$에서 $\sum_i \alpha_i y_i = 0$이므로:

$$g(\alpha) = \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j$$

*Step 4: 쌍대 문제 정식화*

$$\max_\alpha \left[\sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j}^n \alpha_i \alpha_j y_i y_j x_i^T x_j\right]$$
$$\text{s.t. } \sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \geq 0, \quad i=1,...,n$$

또는 최소화 형식으로:

$$\min_\alpha \left[\frac{1}{2}\sum_{i,j}^n \alpha_i \alpha_j y_i y_j x_i^T x_j - \sum_{i=1}^n \alpha_i\right]$$
$$\text{s.t. } \sum_{i=1}^n \alpha_i y_i = 0, \quad 0 \leq \alpha_i$$

**정리 2: KKT 조건과 Support Vector**

KKT 조건:

1. **정류**: $w^* = \sum_i \alpha_i^* y_i x_i$ (이미 적용)

2. **원 가능성**: $y_i(w^{*T} x_i + b^*) \geq 1$ for all $i$

3. **쌍대 가능성**: $\alpha_i^* \geq 0$ for all $i$

4. **상보 느슨함**: 
   $$\alpha_i^* [y_i(w^{*T} x_i + b^*) - 1] = 0$$

상보 느슨함에서:
- **Support Vector**: $\alpha_i^* > 0 \Rightarrow y_i(w^{*T} x_i + b^*) = 1$ (마진 위에 정확히)
- **Non-support Vector**: $\alpha_i^* = 0 \Rightarrow y_i(w^{*T} x_i + b^*) > 1$ (마진 안쪽)

**정리 3: Kernel trick**

내적 $x_i^T x_j$를 커널 함수 $K(x_i, x_j)$로 대체:

$$\min_\alpha \left[\frac{1}{2}\sum_{i,j}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j) - \sum_{i=1}^n \alpha_i\right]$$

분류기:
$$f(x) = \text{sign}\left(\sum_{i=1}^n \alpha_i^* y_i K(x_i, x) + b^*\right)$$

**계산 이점**:
- Primal: $d$가 크면 w ∈ ℝ^d 계산 어려움
- Dual: n개의 α만 최적화, d 크기 무관

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

print("=" * 70)
print("SVM 쌍대 유도: 완전 구현 및 검증")
print("=" * 70)

# 데이터 생성
np.random.seed(42)
X, y = make_blobs(n_samples=100, centers=2, n_features=2, 
                   cluster_std=1.5, random_state=42)
y = 2*y - 1  # {0,1} → {-1,+1}

print(f"\n[데이터]")
print(f"표본 수: {len(X)}")
print(f"특성 수: {X.shape[1]}")
print(f"클래스: {np.unique(y)}")

# Hard-margin SVM: CVXPY로 쌍대 문제 풀이
print("\n" + "=" * 70)
print("Hard-margin SVM: 쌍대 형식")
print("=" * 70)

n = len(X)
K = X @ X.T  # 커널 행렬 (선형)

alpha = cp.Variable(n)
objective = 0.5 * cp.quad_form(alpha, np.diag(y) @ K @ np.diag(y)) - cp.sum(alpha)

constraints = [
    y @ alpha == 0,
    alpha >= 0,
]

problem = cp.Problem(cp.Minimize(objective), constraints)
problem.solve(solver=cp.ECOS)

alpha_opt = alpha.value
print(f"\n[쌍대 문제 해]")
print(f"최적 α: {alpha_opt[:10]}... (처음 10개)")
print(f"목적함수: {problem.value:.6f}")

# Support vector 식별
sv_indices = np.where(alpha_opt > 1e-4)[0]
print(f"\n[Support Vectors]")
print(f"SV 수: {len(sv_indices)} / {n} ({100*len(sv_indices)/n:.1f}%)")
print(f"SV 인덱스 (처음 10개): {sv_indices[:10]}")

# w 복원
w_opt = np.sum(alpha_opt[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)
print(f"\nw* = {w_opt}")
print(f"‖w*‖ = {np.linalg.norm(w_opt):.6f}")

# b 계산 (support vector의 마진 조건)
sv_y = y[sv_indices]
sv_X = X[sv_indices]
sv_alpha = alpha_opt[sv_indices]

b_vals = []
for i in sv_indices:
    if alpha_opt[i] > 1e-4:
        decision = np.sum(alpha_opt * y * (X @ X[i]))
        b_i = y[i] - decision
        b_vals.append(b_i)

b_opt = np.mean(b_vals) if b_vals else 0
print(f"b* = {b_opt:.6f}")

# 마진 확인
margin = 1 / np.linalg.norm(w_opt)
print(f"마진 = 1/‖w*‖ = {margin:.6f}")

# KKT 조건 검증
print("\n" + "=" * 70)
print("[KKT 조건 검증]")
print("=" * 70)

# 1. 정류 조건
w_from_alpha = np.sum(alpha_opt[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)
print(f"\n1. Stationarity: w = Σαᵢyᵢxᵢ")
print(f"   w (계산) = {w_from_alpha}")
print(f"   w (최적) = {w_opt}")
print(f"   오차: {np.linalg.norm(w_from_alpha - w_opt):.2e}")

# 2. 원 가능성
margins = y * (X @ w_opt + b_opt)
feasibility = np.min(margins)
print(f"\n2. Primal Feasibility: yᵢ(w^T xᵢ + b) ≥ 1")
print(f"   최소값: {feasibility:.6f} (≥ 1? {feasibility >= 0.99})")

# 3. 쌍대 가능성
dual_feas = np.min(alpha_opt)
print(f"\n3. Dual Feasibility: αᵢ ≥ 0")
print(f"   최소값: {dual_feas:.2e} (≥ 0? {dual_feas >= -1e-6})")

# 4. 상보 느슨함
comp_slack = []
for i in range(n):
    margin_i = y[i] * (X[i] @ w_opt + b_opt)
    if alpha_opt[i] > 1e-4:
        print(f"\n   i={i}: α={alpha_opt[i]:.4f}, y(w^T x + b)={margin_i:.6f}")
        comp_slack.append(alpha_opt[i] * (margin_i - 1))
        if abs(margin_i - 1) < 1e-4:
            print(f"      → 마진 위 (활성 제약) ✓")
    else:
        comp_slack.append(0)

comp_slack_max = np.max(np.abs(comp_slack))
print(f"\n   상보 느슨함 위반 최대값: {comp_slack_max:.2e}")

# Scikit-learn SVM과 비교
print("\n" + "=" * 70)
print("[Scikit-learn SVM과 비교]")
print("=" * 70)

svm_sklearn = SVC(kernel='linear', C=1e10)  # 큰 C ≈ hard-margin
svm_sklearn.fit(X, y)

print(f"Scikit-learn SVM:")
print(f"  w = {svm_sklearn.coef_[0]}")
print(f"  b = {svm_sklearn.intercept_[0]:.6f}")
print(f"  support vectors: {len(svm_sklearn.support_)} / {n}")

# 우리의 SVM과 비교
print(f"\n우리의 구현:")
print(f"  w = {w_opt}")
print(f"  b = {b_opt:.6f}")
print(f"  support vectors: {len(sv_indices)} / {n}")

# 예측 비교
y_pred_ours = np.sign(X @ w_opt + b_opt)
y_pred_sklearn = svm_sklearn.predict(X)
accuracy_ours = np.mean(y_pred_ours == y)
accuracy_sklearn = np.mean(y_pred_sklearn == y)

print(f"\n정확도:")
print(f"  우리: {accuracy_ours:.4f}")
print(f"  Scikit-learn: {accuracy_sklearn:.4f}")

# Soft-margin SVM (C > 0)
print("\n" + "=" * 70)
print("Soft-margin SVM: C = 1.0")
print("=" * 70)

C = 1.0
alpha_soft = cp.Variable(n)
objective_soft = 0.5 * cp.quad_form(alpha_soft, np.diag(y) @ K @ np.diag(y)) - cp.sum(alpha_soft)

constraints_soft = [
    y @ alpha_soft == 0,
    0 <= alpha_soft,
    alpha_soft <= C,
]

problem_soft = cp.Problem(cp.Minimize(objective_soft), constraints_soft)
problem_soft.solve(solver=cp.ECOS)

alpha_soft_opt = alpha_soft.value
sv_soft = np.where(alpha_soft_opt > 1e-4)[0]

print(f"Soft-margin α: {alpha_soft_opt[:10]}... (처음 10개)")
print(f"SV 수: {len(sv_soft)} / {n}")
print(f"경계 SV (α = C): {np.sum(alpha_soft_opt > C - 1e-4)}")

# 가시화
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 왼쪽 위: 데이터와 결정 경계
ax = axes[0, 0]
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = np.sign(np.c_[xx.ravel(), yy.ravel()] @ w_opt + b_opt).reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=[-1.5, -0.5, 0.5, 1.5], colors=['blue', 'red'], alpha=0.2)
ax.contour(xx, yy, np.c_[xx.ravel(), yy.ravel()] @ w_opt.reshape(-1,1) + b_opt, 
          levels=[0, 1, -1], colors=['k', 'g', 'g'], linestyles=['solid', 'dashed', 'dashed'])

# 표본 표시
colors = ['blue' if yi == 1 else 'red' for yi in y]
ax.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.6, edgecolors='k', linewidths=0.5)

# Support vectors 강조
ax.scatter(X[sv_indices, 0], X[sv_indices, 1], s=200, linewidths=2, 
          facecolors='none', edgecolors='black', label='Support vectors')

ax.set_xlabel('특성 1')
ax.set_ylabel('특성 2')
ax.set_title('Hard-margin SVM: 결정 경계 및 마진')
ax.legend()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# 오른쪽 위: 쌍대 변수 분포
ax = axes[0, 1]
ax.bar(range(n), alpha_opt, color='steelblue', alpha=0.7)
ax.scatter(sv_indices, alpha_opt[sv_indices], color='red', s=100, 
          marker='^', label='Support vectors', zorder=3)
ax.set_xlabel('표본 인덱스')
ax.set_ylabel('α')
ax.set_title('쌍대 변수 분포')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 왼쪽 아래: 마진 값 분포
ax = axes[1, 0]
margins_all = y * (X @ w_opt + b_opt)
ax.hist(margins_all, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=1, color='r', linestyle='--', linewidth=2, label='마진 (=1)')
ax.axvline(x=0, color='orange', linestyle='--', linewidth=2, label='결정 경계')
ax.set_xlabel('yᵢ(w^T xᵢ + b)')
ax.set_ylabel('표본 수')
ax.set_title('마진 분포')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 오른쪽 아래: Hard vs Soft margin
ax = axes[1, 1]
width = 0.35
x_pos = np.arange(3)
hard_margin_sv = len(sv_indices)
soft_margin_sv = len(sv_soft)
interior_sv = np.sum((alpha_soft_opt > 1e-4) & (alpha_soft_opt < C - 1e-4))
boundary_sv = np.sum(alpha_soft_opt > C - 1e-4)

categories = ['전체 SV', '경계내 SV', '경계상 SV']
hard_vals = [hard_margin_sv, np.sum(alpha_opt > 1e-4), 0]
soft_vals = [soft_margin_sv, interior_sv, boundary_sv]

x_pos_hard = np.arange(len(categories))
bars1 = ax.bar(x_pos_hard - width/2, hard_vals, width, label='Hard-margin', alpha=0.8)
bars2 = ax.bar(x_pos_hard + width/2, soft_vals, width, label=f'Soft-margin (C={C})', alpha=0.8)

ax.set_ylabel('표본 수')
ax.set_title('Hard-margin vs Soft-margin SVM')
ax.set_xticks(x_pos_hard)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/tmp/svm_dual_derivation.png', dpi=150, bbox_inches='tight')
print("\n✓ SVM 쌍대 유도 시각화 저장됨")

plt.show()
```

## 🔗 AI/ML 연결
- **커널 SVM**: 비선형 분류를 위해 고차원 공간에 암묵적으로 매핑
- **다중클래스 SVM**: One-vs-Rest, One-vs-One 전략에 쌍대 형식 적용
- **이상 탐지**: One-class SVM의 쌍대 형식
- **온라인 학습**: SGD SVM에서 각 표본의 α 업데이트

## ⚖️ 가정과 한계
- Hard-margin SVM은 선형 분리 가능성 가정 (비현실적)
- Soft-margin의 C는 정규화 계수이지만, 선택이 어려움
- 비선형 커널 선택도 문제 의존적
- 계산량: O(n²) (커널 행렬), O(n³) (최적화)

## 📌 핵심 정리
1. **원 문제**: w ∈ ℝ^d 직접 최적화, 차원 크기에 민감
2. **쌍대 문제**: α ∈ ℝ^n 최적화, 특성 차원 무관
3. **KKT 조건**: Support vector = 마진 위의 표본
4. **커널 트릭**: 내적을 커널로 대체, 고차원 암묵적 매핑
5. **강쌍대성**: Slater 조건 만족하므로 원 = 쌍대 최적값

## 🤔 생각해볼 문제

**문제 1:** Hard-margin SVM의 쌍대 문제에서 제약 $\sum_i \alpha_i y_i = 0$이 어디서 나왔는지 유도하세요.

<details>
<summary>힌트 및 해설</summary>

Lagrangian:
$$L(w,b,\alpha) = \frac{1}{2}\|w\|^2 - \sum_i \alpha_i[y_i(w^T x_i + b) - 1]$$

정류 조건:
$$\frac{\partial L}{\partial b} = -\sum_i \alpha_i y_i = 0$$

따라서 $\sum_i \alpha_i y_i = 0$

이는 b에 대해 최적화할 때 자동으로 도출되는 제약조건입니다. 즉, 쌍대 변수들이 클래스 균형을 맞춰야 한다는 의미입니다.

</details>

**문제 2:** Support vector의 정의를 KKT 상보 느슨함으로부터 유도하세요.

<details>
<summary>힌트 및 해설</summary>

KKT 상보 느슨함:
$$\alpha_i^* [y_i(w^{*T} x_i + b^*) - 1] = 0$$

따라서:
- $\alpha_i^* > 0$ 이면 $y_i(w^{*T} x_i + b^*) = 1$ (마진 위)
- $\alpha_i^* = 0$ 이면 $y_i(w^{*T} x_i + b^*) \geq 1$ (마진 안쪽 또는 위)

Support vector는 쌍대 변수가 0이 아닌 표본, 즉 마진 위 또는 위반하는 표본입니다.

기하학적 의미: Support vector들이 분류기를 "지탱"하므로 그 이름이 붙었습니다.

</details>

**문제 3:** 커널 트릭 적용 후 쌍대 문제가 특성 차원 d에 무관해지는 이유를 설명하세요.

<details>
<summary>힌트 및 해설</summary>

원래 쌍대 문제:
$$\min_\alpha \left[\frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j - \sum_i \alpha_i\right]$$

여기서 계산이 필요한 것은:
- x_i^T x_j: d차원 내적 (O(d) 계산)

커널 트릭 적용:
$$\min_\alpha \left[\frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j) - \sum_i \alpha_i\right]$$

이제 계산이 필요한 것은:
- K(x_i, x_j): 커널 값 (미리 계산, O(d) → O(1))
- 쌍대 변수 α: n개

따라서 최적화는 O(n²) 또는 O(n³)이며, d와 무관합니다.

예: RBF 커널 $K(x,y) = \exp(-\gamma\|x-y\|^2)$는 암묵적으로 무한 차원 특성 공간을 사용하지만, 계산량은 O(d) 유지

</details>

<div align="center">

| [◀ 05. 쌍대 해석 — 그림자 가격](./05-dual-interpretation.md) | [📚 README](../README.md) | [Ch5-01. 경사하강법 수렴 정리 ▶](../ch5-algorithms/01-gd-convergence-full.md) |

</div>
