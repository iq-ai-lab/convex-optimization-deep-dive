# 02. 볼록 집합의 연산

## 🎯 핵심 질문

- 교집합·아핀변환·Minkowski 합은 왜 볼록성을 보존하는가?
- 볼록 포(convex hull)란 무엇이고, 왜 "가장 작은 볼록 집합"인가?
- 볼록 집합의 내부(interior)와 상대 내부(relative interior)는 무엇이 다른가?
- CVXPY의 DCP 규칙이 볼록 집합의 연산 규칙을 어떻게 반영하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

볼록 집합의 연산 규칙은 복잡한 최적화 문제를 볼록으로 유지한 채 변환할 수 있게 해준다.

- **제약 집합의 교집합**: 여러 제약이 동시에 있는 문제 — $\|w\| \leq r$이고 $w^\top \mathbf{1} = 0$이고 $Aw \leq b$ — 의 가능 영역은 볼록 집합들의 교집합이므로 볼록이다.
- **데이터 전처리의 아핀 변환**: 정규화(normalization) $x \mapsto (x - \mu)/\sigma$는 아핀 변환이다. 볼록 집합 위의 문제는 변환 후에도 볼록 집합 위의 문제로 유지된다.
- **앙상블의 볼록 포**: 여러 모델 예측의 볼록 결합 $\sum \theta_i f_i(x)$ ($\theta_i \geq 0$, $\sum \theta_i = 1$)은 모델의 볼록 포 안에 있다. 이것이 앙상블이 단일 모델보다 일반화 성능이 좋을 수 있는 기하학적 이유다.
- **Minkowski 합과 Margin**: SVM에서 margin = "두 클래스의 볼록 포 사이의 거리"로 해석할 수 있다.

---

## 📐 수학적 선행 조건

- [01. 볼록 집합의 정의와 예제](./01-convex-set-definition.md): 볼록 집합의 기본 정의
- **선형 사상**: $f(x) = Ax + b$의 정의 — [Linear Algebra Deep Dive Ch2](https://github.com/iq-ai-lab/linear-algebra-deep-dive)
- **집합의 상(image)과 역상(preimage)**: $f(C) = \{f(x) \mid x \in C\}$, $f^{-1}(D) = \{x \mid f(x) \in D\}$

---

## 📖 직관적 이해

### 볼록성을 보존하는 연산 vs 보존하지 않는 연산

```
볼록성 보존 ✓               볼록성 파괴 ✗
─────────────────────────   ─────────────────────────
교집합 C₁ ∩ C₂              합집합 C₁ ∪ C₂
아핀 변환 {Ax+b | x ∈ C}    일반 비선형 변환
역상 f⁻¹(D)                 일반 상(image)은 보장 안 됨
Minkowski 합 C₁ + C₂        차집합 C₁ \ C₂
볼록 포 conv(S)              임의의 조합
```

> **핵심 직관**: 볼록성은 "선형 구조와 호환되는" 연산에 의해 보존된다. 두 볼록 집합을 "섞거나(교집합)", "이동시키거나(아핀 변환)", "더하면(Minkowski 합)" 볼록이 유지된다. 하지만 "일부를 제거하면(차집합)" 구멍이 생겨 볼록이 깨진다.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — 교집합 (Intersection)

$$C_1 \cap C_2 = \{x \mid x \in C_1 \text{ and } x \in C_2\}$$

임의의 개수(무한 개 포함)의 교집합으로 일반화된다.

### 정의 2.2 — 아핀 변환의 상과 역상

$f: \mathbb{R}^n \to \mathbb{R}^m$, $f(x) = Ax + b$에 대해:
- **상(image)**: $f(C) = \{Ax + b \mid x \in C\}$
- **역상(preimage)**: $f^{-1}(D) = \{x \mid Ax + b \in D\}$

### 정의 2.3 — Minkowski 합 (Minkowski Sum)

$$C_1 + C_2 = \{x + y \mid x \in C_1,\ y \in C_2\}$$

### 정의 2.4 — 볼록 포 (Convex Hull)

집합 $S \subseteq \mathbb{R}^n$의 **볼록 포**:

$$\text{conv}(S) = \left\{ \sum_{i=1}^k \theta_i x_i \;\middle|\; k \geq 1,\ x_i \in S,\ \theta_i \geq 0,\ \sum_{i=1}^k \theta_i = 1 \right\}$$

즉, $S$에 속하는 점들의 모든 가능한 볼록 결합의 집합이다.

### 정의 2.5 — 내부와 상대 내부

집합 $C$의 **내부(interior)**:

$$\text{int}(C) = \{x \in C \mid \exists \varepsilon > 0: B(x, \varepsilon) \subseteq C\}$$

집합 $C$의 **상대 내부(relative interior)**:

$$\text{relint}(C) = \{x \in C \mid \exists \varepsilon > 0: B(x, \varepsilon) \cap \text{aff}(C) \subseteq C\}$$

여기서 $\text{aff}(C)$는 $C$를 포함하는 가장 작은 아핀 부분공간(affine hull)이다.

---

## 🔬 정리와 증명

### 정리 2.1 — 볼록 집합의 교집합

**명제**: $C_1, C_2$가 볼록이면 $C_1 \cap C_2$도 볼록이다. 임의의 집합족 $\{C_\alpha\}_{\alpha \in \mathcal{A}}$에 대해서도 $\bigcap_{\alpha} C_\alpha$는 볼록이다.

**증명**: $x, y \in C_1 \cap C_2$이면 $x, y \in C_1$이고 $x, y \in C_2$.  
$C_1$의 볼록성에 의해 $\lambda x + (1-\lambda)y \in C_1$.  
$C_2$의 볼록성에 의해 $\lambda x + (1-\lambda)y \in C_2$.  
따라서 $\lambda x + (1-\lambda)y \in C_1 \cap C_2$. $\square$

> **중요한 응용**: 다면체 $P = \{x \mid Ax \preceq b\}$는 $n$개의 반공간 $\{x \mid a_i^\top x \leq b_i\}$의 교집합이다. 각 반공간이 볼록이므로 $P$는 볼록이다.

---

### 정리 2.2 — 아핀 변환의 상

**명제**: $C \subseteq \mathbb{R}^n$이 볼록이면, 아핀 함수 $f(x) = Ax + b$에 대해 $f(C) = \{Ax + b \mid x \in C\}$도 볼록이다.

**증명**: $u, v \in f(C)$이면 $u = Ax_1 + b$, $v = Ax_2 + b$ ($x_1, x_2 \in C$)인 점이 존재.  
$C$의 볼록성에 의해 $\lambda x_1 + (1-\lambda)x_2 \in C$.  

$$\lambda u + (1-\lambda)v = \lambda(Ax_1+b) + (1-\lambda)(Ax_2+b)$$
$$= A(\lambda x_1 + (1-\lambda)x_2) + b = f(\lambda x_1 + (1-\lambda)x_2) \in f(C)$$

따라서 $f(C)$는 볼록. $\square$

**따름정리**: 볼록 집합의 투영(projection)은 볼록이다.  
$C \subseteq \mathbb{R}^m \times \mathbb{R}^n$이 볼록이면, $\{x_1 \mid (x_1, x_2) \in C \text{ for some } x_2\}$는 볼록이다.  
(증명: 투영은 아핀 변환 $f(x_1, x_2) = x_1$의 상)

---

### 정리 2.3 — 아핀 변환의 역상

**명제**: $D \subseteq \mathbb{R}^m$이 볼록이고 $f(x) = Ax + b$이면, $f^{-1}(D) = \{x \mid Ax + b \in D\}$도 볼록이다.

**증명**: $x, y \in f^{-1}(D)$이면 $Ax + b \in D$, $Ay + b \in D$.  
$D$의 볼록성에 의해 $\lambda(Ax+b) + (1-\lambda)(Ay+b) = A(\lambda x + (1-\lambda)y) + b \in D$.  
따라서 $\lambda x + (1-\lambda)y \in f^{-1}(D)$. $\square$

**응용**:
- 타원 $\{x \mid (x-x_c)^\top A^{-1}(x-x_c) \leq 1\}$은 볼 $\{u \mid \|u\| \leq 1\}$의 역아핀 변환 — 볼록.
- 행렬 부등식 $\{X \mid AXB + C \succeq 0\}$: 아핀 함수의 역상 — 볼록.

---

### 정리 2.4 — Minkowski 합

**명제**: $C_1, C_2$가 볼록이면 $C_1 + C_2$도 볼록이다.

**증명**: $u = x_1 + y_1$, $v = x_2 + y_2$ ($x_i \in C_1$, $y_i \in C_2$)에 대해:

$$\lambda u + (1-\lambda)v = \lambda(x_1+y_1) + (1-\lambda)(x_2+y_2)$$
$$= \underbrace{(\lambda x_1 + (1-\lambda)x_2)}_{\in C_1} + \underbrace{(\lambda y_1 + (1-\lambda)y_2)}_{\in C_2} \in C_1 + C_2 \quad \square$$

---

### 정리 2.5 — 볼록 포의 최소성

**명제**: $\text{conv}(S)$는 $S$를 포함하는 가장 작은 볼록 집합이다. 즉, $S$를 포함하는 임의의 볼록 집합 $C$에 대해 $\text{conv}(S) \subseteq C$.

**증명**: 
1. $\text{conv}(S)$는 볼록이다: 두 볼록 결합의 볼록 결합도 볼록 결합임 (볼록 결합의 계수를 합치면 다시 합이 1인 양수 계수를 얻음).
2. $S \subseteq \text{conv}(S)$: 단일 점 $x \in S$는 $\theta = 1$인 볼록 결합.
3. 최소성: $C$가 볼록이고 $S \subseteq C$이면, $x_1, \ldots, x_k \in S \subseteq C$이고 $C$가 볼록이므로 $\sum \theta_i x_i \in C$. 따라서 $\text{conv}(S) \subseteq C$. $\square$

---

### 정리 2.6 — 상대 내부의 존재

**명제**: 공집합이 아닌 볼록 집합 $C$의 상대 내부 $\text{relint}(C)$는 공집합이 아니다.

이 정리는 특히 저차원 볼록 집합(예: 선분, 평면 위의 다각형)에서 중요하다. 선분의 내부(open interval)는 공집합이 아니지만, $\mathbb{R}^2$에서의 내부는 공집합이다.

**직관**: 선분 $\{(t, 0) \mid t \in [0,1]\} \subseteq \mathbb{R}^2$의 내부(2D 기준)는 $\emptyset$이지만, 상대 내부(1D 직선 기준)는 열린 구간 $(0, 1) \times \{0\}$이다.

---

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ─────────────────────────────────────────────
# 1. 두 볼록 집합의 교집합 시각화
#    C₁: 원 (반경 1.5, 중심 (-0.5, 0))
#    C₂: 사각형 [−1, 1] × [−1, 1]
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

theta = np.linspace(0, 2 * np.pi, 300)

# 각 집합 샘플링으로 교집합 시각화
x_grid = np.linspace(-3, 3, 400)
y_grid = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_grid, y_grid)
pts = np.stack([X.ravel(), Y.ravel()], axis=1)

in_C1 = (pts[:, 0] + 0.5)**2 + pts[:, 1]**2 <= 1.5**2
in_C2 = (np.abs(pts[:, 0]) <= 1) & (np.abs(pts[:, 1]) <= 1)
in_both = in_C1 & in_C2

ax = axes[0]
ax.contourf(X, Y, in_C1.reshape(X.shape), levels=[0.5, 1.5], colors=['steelblue'], alpha=0.3)
ax.contourf(X, Y, in_C2.reshape(X.shape), levels=[0.5, 1.5], colors=['orange'], alpha=0.3)
ax.contourf(X, Y, in_both.reshape(X.shape), levels=[0.5, 1.5], colors=['green'], alpha=0.5)
ax.set_title('$C_1 \\cap C_2$: 교집합 (초록)\n볼록 ✓', fontsize=11)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.set_xlim(-3, 3); ax.set_ylim(-2, 2)

# ─────────────────────────────────────────────
# 2. Minkowski 합 시각화
#    C₁: 정삼각형, C₂: 원
# ─────────────────────────────────────────────

tri = np.array([[0, 0.6], [0.5, -0.3], [-0.5, -0.3]])
circle_theta = np.linspace(0, 2 * np.pi, 30)
circle_pts = 0.4 * np.stack([np.cos(circle_theta), np.sin(circle_theta)], axis=1)

# Minkowski 합 = 삼각형 각 꼭짓점 + 원의 이동
mink_pts = []
for t in tri:
    for c in circle_pts:
        mink_pts.append(t + c)
mink_pts = np.array(mink_pts)

# 볼록 포로 시각화
hull = ConvexHull(mink_pts)
ax = axes[1]
tri_closed = np.vstack([tri, tri[0]])
ax.fill(tri_closed[:, 0], tri_closed[:, 1], alpha=0.4, color='steelblue', label='$C_1$ (삼각형)')
c_x, c_y = 1.5, 0
ax.fill(c_x + circle_pts[:, 0], c_y + circle_pts[:, 1], alpha=0.4, color='orange', label='$C_2$ (원)')
hull_pts = mink_pts[hull.vertices]
hull_closed = np.vstack([hull_pts, hull_pts[0]])
ax.fill(hull_closed[:, 0], hull_closed[:, 1], alpha=0.35, color='green')
ax.plot(hull_closed[:, 0], hull_closed[:, 1], 'g-', lw=2, label='$C_1 + C_2$ (Minkowski 합)')
ax.set_title('Minkowski 합\n볼록 ✓', fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

# ─────────────────────────────────────────────
# 3. 볼록 포 시각화
#    비볼록 점 집합 S와 conv(S)
# ─────────────────────────────────────────────

np.random.seed(42)
# 별 모양 경계 위의 점들
star_angles = np.linspace(0, 2 * np.pi, 12)[:-1]
r_star = np.array([1 if i % 2 == 0 else 0.45 for i in range(11)])
S = np.stack([r_star * np.cos(star_angles), r_star * np.sin(star_angles)], axis=1)

hull_S = ConvexHull(S)
hull_S_pts = S[hull_S.vertices]
hull_S_closed = np.vstack([hull_S_pts, hull_S_pts[0]])

S_closed = np.vstack([S, S[0]])
ax = axes[2]
ax.fill(S_closed[:, 0], S_closed[:, 1], alpha=0.4, color='red', label='$S$ (별 모양, 비볼록)')
ax.plot(S_closed[:, 0], S_closed[:, 1], 'r-', lw=1.5)
ax.fill(hull_S_closed[:, 0], hull_S_closed[:, 1], alpha=0.25, color='green')
ax.plot(hull_S_closed[:, 0], hull_S_closed[:, 1], 'g-', lw=2, label='$\\mathrm{conv}(S)$ (볼록 포)')
ax.scatter(S[:, 0], S[:, 1], color='darkred', s=40, zorder=5)
ax.set_title('볼록 포 conv(S)\n가장 작은 볼록 집합', fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

plt.suptitle('볼록 집합의 주요 연산', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('02-convex-set-operations.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 4. 아핀 변환이 볼록성을 보존하는지 수치 검증
# ─────────────────────────────────────────────

np.random.seed(0)
A_mat = np.array([[2, 1], [0.5, 1.5]])  # 임의의 행렬
b_vec = np.array([1, -0.5])

# 원래 볼록 집합 C: 단위 볼
def in_unit_ball(x):
    return np.linalg.norm(x) <= 1

# 변환된 집합 f(C)
def in_image(y):
    """y = Ax + b인 x가 단위 볼 안에 있는지"""
    try:
        x = np.linalg.solve(A_mat, y - b_vec)
        return in_unit_ball(x)
    except:
        return False

# 볼록성 확인: f(C)에서 두 점의 볼록 결합도 f(C) 안에 있는지
def check_image_convexity(n_trials=3000):
    violations = 0
    for _ in range(n_trials):
        # f(C) 안의 두 점 샘플링
        x1 = np.random.randn(2)
        x1 /= max(np.linalg.norm(x1), 1)  # 단위 볼 안
        x2 = np.random.randn(2)
        x2 /= max(np.linalg.norm(x2), 1)
        y1 = A_mat @ x1 + b_vec
        y2 = A_mat @ x2 + b_vec

        lam = np.random.uniform(0, 1)
        y_mid = lam * y1 + (1 - lam) * y2

        if not in_image(y_mid):
            violations += 1
    return violations / n_trials

violations = check_image_convexity()
print(f"아핀 변환 상의 볼록성 위반율: {violations:.4f} (기대: 0)")
```

**출력**:
```
아핀 변환 상의 볼록성 위반율: 0.0000 (기대: 0)
```

---

## 🔗 AI/ML 연결

### 다층 선형 변환과 볼록성

딥러닝의 선형 레이어 $z = Wx + b$는 아핀 변환이다. 볼록 집합 $C$ (예: 입력 제약)를 통과시켜도 $WC + b$는 볼록이다. 그러나 비선형 활성화 함수 $\sigma(z)$ (ReLU, sigmoid 등)는 일반적인 아핀 변환이 아니므로, 활성화 통과 후 볼록성 보존이 보장되지 않는다.

### 볼록 포와 앙상블 학습

$k$개의 모델 예측 $\hat{y}_1, \ldots, \hat{y}_k$의 **가중 평균** $\hat{y} = \sum_i \theta_i \hat{y}_i$ ($\theta_i \geq 0$, $\sum \theta_i = 1$)은 볼록 포 $\text{conv}(\{\hat{y}_1, \ldots, \hat{y}_k\})$ 안에 있다.

Jensen 부등식에 의해 손실 함수 $\ell$이 볼록이면:
$$\ell\!\left(\sum_i \theta_i \hat{y}_i\right) \leq \sum_i \theta_i \ell(\hat{y}_i)$$

즉, **앙상블의 손실 ≤ 개별 모델 손실의 가중 평균**. 이것이 앙상블이 이론적으로 정당화되는 볼록 최적화 관점의 설명이다.

### 상대 내부와 볼록 최적화

볼록 최적화의 여러 정리(강쌍대성, Slater 조건)는 가능 영역의 상대 내부에 점이 존재할 것을 요구한다. 선형 프로그램에서 제약이 등식으로만 이루어진 경우 (가능 영역이 선분이나 평면 조각), 내부(interior)는 공집합이지만 상대 내부는 존재할 수 있다.

---

## ⚖️ 가정과 한계

| 연산 | 볼록성 보존 | 주의사항 |
|------|-----------|---------|
| 교집합 (임의 개수) | ✓ 항상 | 공집합이 될 수 있음 |
| 합집합 | ✗ 일반적으로 | 두 집합이 포함 관계일 때만 ✓ |
| 아핀 변환의 상 | ✓ 항상 | 비선형 변환은 보장 없음 |
| 아핀 변환의 역상 | ✓ 항상 | — |
| Minkowski 합 | ✓ 항상 | 계산 복잡도가 높을 수 있음 |
| 볼록 포 | ✓ (정의상) | 무한 집합의 볼록 포는 닫혀 있지 않을 수 있음 |
| 차집합 $C_1 \setminus C_2$ | ✗ 일반적 | — |

**무한 볼록 포의 닫힘 문제**: 볼록 포는 볼록이지만 닫혀 있지 않을 수 있다. 닫힌 볼록 포(closed convex hull)는 볼록 포의 위상적 폐포(closure)다.

---

## 📌 핵심 정리

$$\text{볼록성 보존 규칙}$$

| 연산 | 수식 | 보존 이유 |
|------|------|---------|
| 교집합 | $\bigcap_\alpha C_\alpha$ | 각 집합의 조건을 동시에 만족 |
| 아핀 상 | $\{Ax+b \mid x \in C\}$ | 선형성으로 볼록 결합 보존 |
| 아핀 역상 | $\{x \mid Ax+b \in D\}$ | 선형성으로 역방향도 보존 |
| Minkowski 합 | $C_1 + C_2$ | 성분별 볼록 결합 |
| 볼록 포 | $\text{conv}(S)$ | 정의상 가장 작은 볼록 집합 |

**볼록 포의 최소성**: $S \subseteq C$ (C 볼록) $\Rightarrow$ $\text{conv}(S) \subseteq C$.

---

## 🤔 생각해볼 문제

**문제 1** (기초): 볼록 집합 $C_1, C_2$의 합집합 $C_1 \cup C_2$가 볼록이기 위한 필요충분조건을 찾아라.

<details>
<summary>힌트 및 해설</summary>

**필요충분조건**: $C_1 \subseteq C_2$ 또는 $C_2 \subseteq C_1$ (포함 관계).

**증명 (충분)**: $C_1 \subseteq C_2$이면 $C_1 \cup C_2 = C_2$이므로 볼록.

**증명 (필요)**: $C_1 \not\subseteq C_2$이고 $C_2 \not\subseteq C_1$이면 $x \in C_1 \setminus C_2$, $y \in C_2 \setminus C_1$인 점이 존재. 만약 $C_1 \cup C_2$가 볼록이면 $\frac{1}{2}(x+y) \in C_1 \cup C_2$. 이 점이 $C_1$에 속한다고 하면 $y = 2 \cdot \frac{x+y}{2} - x \in C_1$ (볼록성), 즉 $y \in C_1$ — 모순. $C_2$에 속한다고 해도 같은 방식으로 $x \in C_2$ — 모순.

</details>

**문제 2** (심화): 볼록 집합 $C \subseteq \mathbb{R}^n$에 대해 다음을 증명하라.  
"$x \in \text{int}(C)$이고 $y \in \text{cl}(C)$ (폐포)이면, $\lambda x + (1-\lambda)y \in \text{int}(C)$ for all $\lambda \in (0, 1]$."

<details>
<summary>힌트 및 해설</summary>

$x \in \text{int}(C)$이므로 $B(x, \varepsilon) \subseteq C$인 $\varepsilon > 0$이 존재.  
$y \in \text{cl}(C)$이므로 $y$에 수렴하는 수열 $y_k \in C$가 존재.  
$z_k = \lambda x + (1-\lambda)y_k \in C$ (볼록성). $z_k \to \lambda x + (1-\lambda)y = z$.  
임의의 $u \in B(z, \lambda\varepsilon)$에 대해 $u = z + \delta$ ($\|\delta\| < \lambda\varepsilon$)이면 $\frac{u - (1-\lambda)y}{\lambda} = x + \delta/\lambda$이고 $\|\delta/\lambda\| < \varepsilon$이므로 $x + \delta/\lambda \in C$. 따라서 $u = \lambda(x+\delta/\lambda) + (1-\lambda)y \in C$.

</details>

**문제 3** (AI 연결): $k$개의 분류기 $f_1, \ldots, f_k: \mathbb{R}^d \to [0,1]$이 있을 때, 이들의 볼록 포 $\text{conv}(\{f_1, \ldots, f_k\})$가 함수 공간에서 볼록 집합임을 설명하라. 이것이 Boosting 알고리즘의 이론적 기반과 어떻게 연결되는가?

<details>
<summary>힌트 및 해설</summary>

함수 공간에서 볼록 포 $\{\sum \theta_i f_i \mid \theta_i \geq 0, \sum \theta_i = 1\}$은 정의상 볼록. AdaBoost는 각 라운드에서 손실을 최소화하는 $f_i$를 선택하고 가중치 $\theta_i$를 갱신하는데, 이는 함수의 볼록 포 안에서 손실을 최소화하는 과정이다. 손실 함수가 볼록이면 (예: 지수 손실, 로그 손실), 이 최소화 문제도 볼록 최적화가 된다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. 볼록 집합의 정의와 예제](./01-convex-set-definition.md) | [📚 README](../README.md) | [03. 분리 초평면 정리 ▶](./03-separating-hyperplane.md) |

</div>
