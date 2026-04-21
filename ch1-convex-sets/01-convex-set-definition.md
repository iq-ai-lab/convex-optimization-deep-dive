# 01. 볼록 집합의 정의와 예제

## 🎯 핵심 질문

- "임의의 두 점을 잇는 선분이 집합에 포함된다"는 조건이 왜 최적화에서 그토록 강력한가?
- 초평면·반공간·다면체·노름 볼·양의 정부호 콘은 각각 볼록인가? 왜인가?
- 볼록 집합 위에서 볼록 함수를 최소화하면 왜 국소 최솟값 = 전역 최솟값인가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

볼록 집합은 최적화 알고리즘이 "올바른 방향으로 움직이고 있다"는 보장의 공간적 기반이다.

- **SVM의 가능 영역**: Hard-margin SVM의 제약 집합 $\{w \mid y_i(w^\top x_i + b) \geq 1\}$은 반공간들의 교집합 — 볼록 집합이다. 이것이 전역 최적 margin이 존재하는 이유다.
- **경사하강법의 수렴 보장**: Gradient descent가 볼록 집합 위의 볼록 함수를 최소화할 때 발산하지 않는 이유는 집합이 "경계를 넘어가면 투영(projection)으로 돌아올 수 있다"는 볼록 집합의 성질 때문이다.
- **Interior Point Method**: LP·QP·SDP를 다항 시간에 푸는 IPM은 가능 영역의 내부를 따라 중앙 경로(central path)를 추적한다. 이 경로가 정의되려면 가능 영역이 볼록이어야 한다.
- **정규화 항의 제약 해석**: $\|x\|_2 \leq r$ (L2 ball), $\|x\|_1 \leq r$ (L1 ball)은 모두 볼록 집합 — 이를 제약으로 부과했을 때 볼록 최적화 문제가 유지된다.

---

## 📐 수학적 선행 조건

- **내적과 노름**: $x^\top y = \sum x_i y_i$, $\|x\|_2 = \sqrt{x^\top x}$ — [Linear Algebra Deep Dive Ch1](https://github.com/iq-ai-lab/linear-algebra-deep-dive)
- **양의 정부호 행렬**: $A \succeq 0 \Leftrightarrow x^\top A x \geq 0\ \forall x$ — [Linear Algebra Deep Dive Ch5](https://github.com/iq-ai-lab/linear-algebra-deep-dive)
- **집합론 기초**: 부분집합, 교집합, 여집합의 정의

> 이 문서에서 미적분은 사용하지 않습니다. 볼록 집합은 대수적·기하적 개념입니다.

---

## 📖 직관적 이해

### "선분이 집합 안에 있다"는 것의 의미

두 점 $x, y$를 잡았을 때 $\lambda x + (1-\lambda)y$ ($\lambda \in [0,1]$)는 두 점을 잇는 선분 위의 모든 점을 나타낸다.

- $\lambda = 0$: 점 $y$
- $\lambda = 1$: 점 $x$
- $\lambda = 0.5$: 두 점의 중점

**볼록 집합**은 어떤 두 점을 잡아도 그 사이 선분 전체가 집합 안에 있는 집합이다.

```
볼록 집합 (O)         비볼록 집합 (X)

  ╭───────╮             ╭───╮   ╭───╮
  │  x    │             │ x │   │ y │
  │    y  │             ╰───╯   ╰───╯
  ╰───────╯          (선분이 집합 밖을 지남)
  (선분도 ⊆ 집합)
```

> **왜 이것이 중요한가**: 볼록 집합 위에서 볼록 함수의 국소 최솟값을 찾으면, 그 점이 바로 전역 최솟값이다. "더 좋은 점이 있을지도 모른다"는 걱정이 수학적으로 불필요해진다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 볼록 집합 (Convex Set)

집합 $C \subseteq \mathbb{R}^n$이 **볼록**이라는 것은:

$$\forall x, y \in C,\ \forall \lambda \in [0, 1]:\ \lambda x + (1-\lambda)y \in C$$

점 $\lambda x + (1-\lambda)y$를 $x$와 $y$의 **볼록 결합(convex combination)** 이라 한다.

### 정의 1.2 — 볼록 결합의 일반화

$k$개의 점 $x_1, \ldots, x_k$에 대해 $\sum_{i=1}^k \theta_i x_i$ ($\theta_i \geq 0$, $\sum \theta_i = 1$)를 **볼록 결합**이라 한다.

볼록 집합은 임의의 유한 개 점의 볼록 결합에 대해서도 닫혀 있다 (수학적 귀납법으로 증명).

### 정의 1.3 — 아핀 집합 (Affine Set)

집합 $C$가 **아핀**이라는 것은:

$$\forall x, y \in C,\ \forall \lambda \in \mathbb{R}:\ \lambda x + (1-\lambda)y \in C$$

> $\lambda \in [0,1]$이 아닌 $\lambda \in \mathbb{R}$이다. 즉 선분이 아니라 직선 전체가 포함된다. 아핀 집합 ⊆ 볼록 집합.

### 정의 1.4 — 공집합과 단일점

공집합 $\emptyset$과 단일점 $\{x_0\}$은 볼록 집합의 정의를 공허하게 만족한다 (vacuously true). 모든 정리의 적용 대상이다.

---

## 🔬 정리와 증명

### 주요 볼록 집합과 볼록성 증명

#### 예제 1. 초평면 (Hyperplane)

$$H = \{x \in \mathbb{R}^n \mid a^\top x = b\}\ (a \neq 0)$$

**볼록성 증명**: $x, y \in H$이면 $a^\top x = b$, $a^\top y = b$. 임의의 $\lambda \in [0,1]$에 대해:

$$a^\top (\lambda x + (1-\lambda)y) = \lambda a^\top x + (1-\lambda) a^\top y = \lambda b + (1-\lambda)b = b$$

따라서 $\lambda x + (1-\lambda)y \in H$. $\square$

#### 예제 2. 반공간 (Halfspace)

$$H^- = \{x \in \mathbb{R}^n \mid a^\top x \leq b\}\ (a \neq 0)$$

**볼록성 증명**: $x, y \in H^-$이면 $a^\top x \leq b$, $a^\top y \leq b$. $\lambda \in [0,1]$에 대해:

$$a^\top (\lambda x + (1-\lambda)y) = \lambda a^\top x + (1-\lambda)a^\top y \leq \lambda b + (1-\lambda)b = b$$

따라서 $\lambda x + (1-\lambda)y \in H^-$. $\square$

#### 예제 3. 다면체 (Polyhedron)

$$P = \{x \mid Ax \preceq b,\ Cx = d\}$$

(여기서 $\preceq$는 성분별 부등호)

**볼록성 증명**: $P$는 반공간들과 초평면들의 교집합이다. 반공간·초평면 각각이 볼록이고, 볼록 집합의 교집합은 볼록이므로 (정리 2.1) $P$는 볼록이다. $\square$

#### 예제 4. 유클리드 볼 (Euclidean Ball)

$$B(x_c, r) = \{x \mid \|x - x_c\|_2 \leq r\} = \{x_c + ru \mid \|u\|_2 \leq 1\}$$

**볼록성 증명**: $x, y \in B(x_c, r)$이면 $\|x - x_c\| \leq r$, $\|y - x_c\| \leq r$. $\lambda \in [0,1]$에 대해:

$$\|\lambda x + (1-\lambda)y - x_c\| = \|\lambda(x-x_c) + (1-\lambda)(y-x_c)\|$$
$$\leq \lambda\|x-x_c\| + (1-\lambda)\|y-x_c\| \leq \lambda r + (1-\lambda)r = r$$

(삼각 부등식 사용). 따라서 볼록. $\square$

> 같은 방법으로 임의의 노름 볼 $\{x \mid \|x\|_p \leq r\}$ ($p \geq 1$)이 볼록임을 증명할 수 있다.

#### 예제 5. 양의 준정부호 콘 (Positive Semidefinite Cone)

$$\mathbb{S}^n_+ = \{X \in \mathbb{R}^{n \times n} \mid X = X^\top,\ X \succeq 0\}$$

**볼록성 증명**: $X, Y \in \mathbb{S}^n_+$이면 $X \succeq 0$, $Y \succeq 0$. $\lambda \in [0,1]$에 대해 임의의 $v \in \mathbb{R}^n$에 대해:

$$v^\top (\lambda X + (1-\lambda)Y) v = \lambda v^\top X v + (1-\lambda) v^\top Y v \geq 0$$

($v^\top X v \geq 0$, $v^\top Y v \geq 0$이므로). 따라서 $\lambda X + (1-\lambda)Y \succeq 0$. $\square$

#### 정리 1.1 — 볼록 집합의 유한 볼록 결합

**명제**: $C$가 볼록 집합이면, $x_1, \ldots, x_k \in C$와 $\theta_i \geq 0$, $\sum \theta_i = 1$에 대해 $\sum \theta_i x_i \in C$.

**증명** (수학적 귀납법):  
- $k=2$: 정의에 의해 성립.
- $k-1$에서 성립한다고 가정. $\sum_{i=1}^k \theta_i = 1$이고 $\theta_k < 1$ (그렇지 않으면 자명)이라 하면:

$$\sum_{i=1}^k \theta_i x_i = (1-\theta_k)\sum_{i=1}^{k-1}\frac{\theta_i}{1-\theta_k}x_i + \theta_k x_k$$

귀납 가정에 의해 $z = \sum_{i=1}^{k-1}\frac{\theta_i}{1-\theta_k}x_i \in C$ (가중치의 합 = 1). 그리고 $z, x_k \in C$, $\lambda = 1-\theta_k \in [0,1]$이므로 볼록 결합이 $C$에 속한다. $\square$

---

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection

# ─────────────────────────────────────────────
# 1. 볼록 집합 판별 함수
# ─────────────────────────────────────────────

def is_in_set(point, set_type, params):
    """주어진 점이 집합에 속하는지 확인"""
    if set_type == 'ball':
        center, radius = params
        return np.linalg.norm(point - center) <= radius
    elif set_type == 'halfspace':
        a, b = params
        return a @ point <= b
    elif set_type == 'polyhedron':
        A, b = params
        return np.all(A @ point <= b)

def check_convexity_numerical(set_checker, n_trials=5000, dim=2):
    """수치적으로 볼록 집합 여부를 확인 (반례 탐색)"""
    violations = 0
    for _ in range(n_trials):
        x = np.random.randn(dim) * 2
        y = np.random.randn(dim) * 2
        lam = np.random.uniform(0, 1)
        
        if set_checker(x) and set_checker(y):
            mid = lam * x + (1 - lam) * y
            if not set_checker(mid):
                violations += 1
    
    ratio = violations / n_trials
    return ratio  # 0이면 볼록 (반례 없음)

# ─────────────────────────────────────────────
# 2. 주요 볼록 집합 시각화
# ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

theta = np.linspace(0, 2 * np.pi, 300)

# (1) 유클리드 볼
ax = axes[0]
ax.fill(np.cos(theta), np.sin(theta), alpha=0.3, color='steelblue')
ax.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2)
ax.set_title('유클리드 볼 $\\{x \\mid \\|x\\|_2 \\leq 1\\}$\n볼록 ✓', fontsize=11)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)

# (2) L1 볼 (마름모)
ax = axes[1]
l1_pts = np.array([[1,0],[0,1],[-1,0],[0,-1],[1,0]])
ax.fill(l1_pts[:,0], l1_pts[:,1], alpha=0.3, color='orange')
ax.plot(l1_pts[:,0], l1_pts[:,1], 'r-', linewidth=2)
ax.set_title('L1 볼 $\\{x \\mid \\|x\\|_1 \\leq 1\\}$\n볼록 ✓', fontsize=11)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)

# (3) 반공간
ax = axes[2]
x_range = np.linspace(-2, 2, 300)
ax.fill_betweenx(x_range, -3, x_range, alpha=0.3, color='green', label='$x_1 + x_2 \\leq 1$')
ax.plot(x_range, 1 - x_range, 'g-', linewidth=2, label='경계선 $x_1+x_2=1$')
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
ax.set_title('반공간 $\\{x \\mid a^\\top x \\leq b\\}$\n볼록 ✓', fontsize=11)
ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)

# (4) 다면체
ax = axes[3]
poly_pts = np.array([[0,0],[2,0],[2,1],[1,2],[0,2]])
poly = Polygon(poly_pts, alpha=0.3, color='purple')
ax.add_patch(poly)
ax.plot(np.append(poly_pts[:,0], poly_pts[0,0]),
        np.append(poly_pts[:,1], poly_pts[0,1]), 'purple', lw=2)
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2.5)
ax.set_title('다면체 (Polyhedron)\n볼록 ✓', fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)

# (5) 볼록하지 않은 집합 — 원환 (annulus)
ax = axes[4]
r_outer = np.cos(theta), np.sin(theta)
r_inner = 0.5*np.cos(theta), 0.5*np.sin(theta)
ax.fill(r_outer[0], r_outer[1], alpha=0.4, color='red')
ax.fill(r_inner[0], r_inner[1], alpha=1.0, color='white')
ax.plot(r_outer[0], r_outer[1], 'r-', lw=2)
ax.plot(r_inner[0], r_inner[1], 'r-', lw=2)
# 반례: 두 점과 선분
p1, p2 = np.array([-0.9, 0]), np.array([0.9, 0])
ax.scatter(*p1, color='blue', s=60, zorder=5)
ax.scatter(*p2, color='blue', s=60, zorder=5)
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b--', lw=2, label='선분이 집합 밖을 지남')
ax.set_title('원환 (Annulus)\n볼록 ✗ — 반례 표시', fontsize=11)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

# (6) 볼록하지 않은 집합 — 별 모양
ax = axes[5]
star_angles = np.linspace(0, 2*np.pi, 11)[:-1]
r = np.array([1 if i % 2 == 0 else 0.4 for i in range(10)])
star_x = r * np.cos(star_angles)
star_y = r * np.sin(star_angles)
ax.fill(star_x, star_y, alpha=0.4, color='gold')
ax.plot(np.append(star_x, star_x[0]), np.append(star_y, star_y[0]), 'orange', lw=2)
# 반례
p1, p2 = np.array([0.9, 0.3]), np.array([-0.9, 0.3])
ax.scatter(*p1, color='blue', s=60, zorder=5)
ax.scatter(*p2, color='blue', s=60, zorder=5)
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b--', lw=2, label='선분이 집합 밖을 지남')
ax.set_title('별 모양\n볼록 ✗ — 반례 표시', fontsize=11)
ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

plt.suptitle('볼록 집합과 비볼록 집합', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('01-convex-sets-examples.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. 볼록성 수치 검증
# ─────────────────────────────────────────────

# 볼 검사
ball_checker = lambda x: np.linalg.norm(x) <= 1
violations_ball = check_convexity_numerical(ball_checker)
print(f"유클리드 볼 위반율: {violations_ball:.4f} (기대: 0)")

# 반공간 검사
halfspace_checker = lambda x: x[0] + x[1] <= 1
violations_hs = check_convexity_numerical(halfspace_checker)
print(f"반공간 위반율: {violations_hs:.4f} (기대: 0)")

# 원환 검사 (비볼록)
annulus_checker = lambda x: 0.5 <= np.linalg.norm(x) <= 1
violations_ann = check_convexity_numerical(annulus_checker)
print(f"원환 위반율: {violations_ann:.4f} (기대: 양수)")
```

**출력 예시**:
```
유클리드 볼 위반율: 0.0000 (기대: 0)
반공간 위반율: 0.0000 (기대: 0)
원환 위반율: 0.1823 (기대: 양수)
```

---

## 🔗 AI/ML 연결

### SVM의 가능 영역 = 볼록 집합

Hard-margin SVM은 다음을 최소화한다:

$$\min_{w, b} \frac{1}{2}\|w\|^2 \quad \text{s.t.}\ y_i(w^\top x_i + b) \geq 1\ \forall i$$

제약 $y_i(w^\top x_i + b) \geq 1$은 각각 반공간이다. 반공간들의 교집합은 다면체 — 볼록 집합. 따라서 목적함수(볼록 함수)를 볼록 집합 위에서 최소화하는 볼록 최적화 문제가 된다.

### 가능 영역의 볼록성과 Projected GD

제약 최적화에서 경사하강법은 "가능 영역 밖으로 나가면 집합으로 투영"하는 Projected GD를 쓴다:

$$x_{k+1} = \Pi_C(x_k - \eta \nabla f(x_k))$$

여기서 $\Pi_C(v) = \arg\min_{x \in C} \|x - v\|$는 $C$로의 투영이다. 이 투영이 유일하게 잘 정의되려면 $C$가 볼록이어야 한다.

**볼록 집합으로의 투영의 유일성**: $C$가 볼록이고 닫혀 있으면, 임의의 $v$에 대해 $\Pi_C(v)$는 유일하게 존재한다. (증명: Hilbert projection theorem)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 집합이 공집합이 아님 | 공집합은 볼록이지만 최적화 문제가 무의미해진다 |
| $\mathbb{R}^n$에서의 볼록성 | 일반 벡터 공간(Hilbert space 등)에서도 정의 가능하지만 추가 구조가 필요하다 |
| 닫힌 볼록 집합 | 열린 볼록 집합(예: 열린 볼)도 볼록이지만 최솟값이 경계에서 달성되지 않을 수 있다 |
| 유계 볼록 집합 | 유계이지 않으면 최솟값이 $-\infty$로 발산할 수 있다 (예: $\min x$ on $\mathbb{R}$) |

**비볼록 가능 영역의 위험**: 정수 계획(integer programming)의 가능 영역 $\{0, 1\}^n$은 비볼록이다. 이 경우 국소 최솟값 ≠ 전역 최솟값이 될 수 있어, 볼록 완화(LP relaxation) 기법이 필요하다.

---

## 📌 핵심 정리

$$C \text{ 볼록} \Leftrightarrow \forall x,y \in C,\ \lambda \in [0,1]:\ \lambda x + (1-\lambda)y \in C$$

| 집합 | 볼록 여부 | 핵심 이유 |
|------|---------|---------|
| 초평면 $\{a^\top x = b\}$ | ✓ | 아핀 집합 |
| 반공간 $\{a^\top x \leq b\}$ | ✓ | 선형 부등식 1개 |
| 다면체 $\{Ax \preceq b, Cx=d\}$ | ✓ | 반공간·초평면의 교집합 |
| 유클리드 볼 $\{\|x\| \leq r\}$ | ✓ | 삼각 부등식 |
| $\mathbb{S}^n_+$ (PSD 행렬 집합) | ✓ | 이차형식의 선형성 |
| 원환 (도넛) | ✗ | 두 점의 선분이 구멍을 지남 |
| 유한 점들의 합집합 | ✗ (일반적) | 두 점 사이의 경계 밖 |

**핵심 메시지**: 볼록 집합은 "선분에 닫힌" 집합이다. 이 단순한 조건이 국소 최솟값 = 전역 최솟값이라는 최적화의 황금 티켓을 부여한다.

---

## 🤔 생각해볼 문제

**문제 1** (기초): 다음 집합의 볼록 여부를 판단하고 증명 또는 반례를 제시하라.

(a) $\{x \in \mathbb{R}^2 \mid x_1^2 + x_2^2 \geq 1\}$ (원의 외부)

(b) $\{x \in \mathbb{R}^2 \mid x_1 x_2 \geq 1,\ x_1 > 0, x_2 > 0\}$ (쌍곡선 위쪽)

(c) $\{(x, y) \in \mathbb{R}^3 \mid \|x\|_2 \leq y\}$ (아이스크림 콘)

<details>
<summary>힌트 및 해설</summary>

(a) **비볼록**: $x = (2, 0)$, $y = (-2, 0)$의 중점 $(0, 0)$은 집합 밖.

(b) **볼록**: $x = (a_1, a_2), y = (b_1, b_2)$ ($a_1 a_2 \geq 1, b_1 b_2 \geq 1$)에 대해 $\lambda \in [0,1]$이면 AM-GM에 의해:
$$(\lambda a_1 + (1-\lambda)b_1)(\lambda a_2 + (1-\lambda)b_2) \geq (\lambda\sqrt{a_1 a_2} + (1-\lambda)\sqrt{b_1 b_2})^2 \geq 1$$

(c) **볼록** (이차 콘, Second-Order Cone): 삼각 부등식으로 증명. Ch1-04에서 자세히 다룬다.

</details>

**문제 2** (심화): 두 볼록 집합의 합집합이 볼록이 아닌 반례를 구성하고, 합집합이 볼록이 되기 위한 충분조건을 제시하라.

<details>
<summary>힌트 및 해설</summary>

**반례**: $C_1 = \{x \in \mathbb{R} \mid x \leq 0\}$, $C_2 = \{x \in \mathbb{R} \mid x \geq 1\}$이면 $C_1 \cup C_2$는 $[0,1]$ 구간이 빠진 비볼록 집합.

**충분조건**: 두 볼록 집합 중 하나가 다른 하나에 포함되면 ($C_1 \subseteq C_2$ 또는 $C_2 \subseteq C_1$) 합집합이 볼록. 또는 두 집합이 교집합을 가지고 "한 집합이 다른 집합 방향으로 연장된" 경우.

</details>

**문제 3** (AI 연결): L2 정규화 $\|w\|_2 \leq r$과 L1 정규화 $\|w\|_1 \leq r$의 제약 집합은 모두 볼록이다. 그러나 L1 제약이 sparsity를 유도하고 L2는 그렇지 않은 이유를 집합의 기하적 형태(꼭짓점 존재 여부)로 설명하라.

<details>
<summary>힌트 및 해설</summary>

L1 볼 $\{w \mid \|w\|_1 \leq r\}$은 마름모(다면체)로, 좌표축 위에 꼭짓점(extreme point)을 가진다. 볼록 함수의 최솟값은 이 꼭짓점에서 달성될 가능성이 높고, 꼭짓점에서는 한 좌표를 제외한 나머지가 0 — 즉 sparse하다. L2 볼은 구(sphere)로 꼭짓점이 없고 등방성(isotropic)이어서 어느 방향으로도 특별한 선호가 없다. 자세한 내용은 Ch7-03에서 기하학적으로 완전히 분석한다.

</details>

---

<div align="center">

| | |
|---|---|
| [📚 README로 돌아가기](../README.md) | [02. 볼록 집합의 연산 ▶](./02-convex-set-operations.md) |

</div>
