# 03. 분리 초평면 정리(Separating Hyperplane Theorem)

## 🎯 핵심 질문

- 서로소인 두 볼록 집합을 분리하는 초평면이 항상 존재하는가?
- "지지 초평면(supporting hyperplane)"이란 무엇이며, 볼록 함수의 1차 조건과 어떻게 연결되는가?
- 이 정리가 왜 Lagrange 쌍대 이론과 Slater 조건 증명의 핵심 도구인가?
- SVM의 최적 분리 초평면이 수학적으로 왜 유일하게 존재하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

분리 초평면 정리는 볼록 최적화의 전체 이론 구조에서 가장 기초적인 기하 원리다.

- **강쌍대성(Strong Duality)의 핵심 도구**: Ch4-03에서 증명할 Slater 조건 → 강쌍대성은 이 정리의 직접 응용이다. "primal 문제의 infimum과 dual 문제의 supremum 사이에 gap이 없다"는 것을 두 볼록 집합을 분리하는 초평면의 존재로 증명한다.
- **SVM의 이론적 기반**: 두 클래스가 선형 분리 가능하다는 것은 정확히 "양성 클래스와 음성 클래스의 볼록 포가 서로소"와 동치이며, 분리 초평면 정리에 의해 분리 초평면(= decision boundary)이 존재한다.
- **볼록 함수의 1차 조건**: 볼록 함수 $f$의 epigraph에 경계점에서 지지 초평면이 존재한다 → 이것이 $f(y) \geq f(x) + \nabla f(x)^\top(y-x)$라는 1차 조건의 기하학적 의미다.
- **Regularization의 기하**: L1 정규화에서 sparsity가 발생하는 이유 — L1 ball의 꼭짓점이 손실 함수의 등고선과 접하는 지지 초평면이 axis-aligned 방향으로 놓이기 때문.

---

## 📐 수학적 선행 조건

- [01. 볼록 집합의 정의와 예제](./01-convex-set-definition.md): 볼록 집합의 기본 정의
- [02. 볼록 집합의 연산](./02-convex-set-operations.md): 볼록 집합의 폐포와 내부
- **내적과 초평면**: $\{x \mid a^\top x = b\}$가 $\mathbb{R}^n$을 두 반공간으로 나눔 — [Linear Algebra Deep Dive Ch1](https://github.com/iq-ai-lab/linear-algebra-deep-dive)
- **노름과 거리**: $\|x - y\|_2$ — 최단 거리 점의 존재 (볼록 집합으로의 투영)

---

## 📖 직관적 이해

### 분리 초평면의 아이디어

두 사람 A와 B가 각자의 영역(볼록 집합)을 가지고 있을 때, 두 영역 사이에 울타리(초평면)를 세울 수 있는가?

```
          분리 초평면 a^T x = b
               │
    C₁         │         C₂
   (볼록)       │        (볼록)
   ●            │            ●
  ●●●           │           ●●
   ●            │            ●
               │
 a^T x < b    │    a^T x > b
```

두 집합이 서로소이고 볼록이면 — 항상 이런 울타리를 세울 수 있다.

> **왜 볼록이어야 하는가**: 비볼록 집합에서는 한 집합이 다른 집합을 "감싸는" 형태가 가능해 선형 초평면으로 분리할 수 없다. 볼록성이 "U자형이나 도넛형"을 금지하기 때문에 분리가 보장된다.

### 지지 초평면의 아이디어

볼록 집합의 경계에 있는 한 점 $x_0$에서, 집합 전체가 "울타리의 한쪽"에 있도록 울타리를 세울 수 있다.

```
            지지 초평면 a^T x = a^T x₀
               │
        볼록   │
       집합 C  x₀  ← 경계점
          ●●  │
         ●●●  │
          ●●  │
               │
    a^T x ≤ a^T x₀
```

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 초평면 (Hyperplane)

$a \in \mathbb{R}^n$ ($a \neq 0$), $b \in \mathbb{R}$에 대해:

$$H = \{x \in \mathbb{R}^n \mid a^\top x = b\}$$

초평면은 $\mathbb{R}^n$을 두 반공간 $\{a^\top x \leq b\}$, $\{a^\top x \geq b\}$으로 나눈다.

### 정의 3.2 — 분리 초평면 (Separating Hyperplane)

집합 $C_1, C_2 \subseteq \mathbb{R}^n$에 대해 초평면 $\{a^\top x = b\}$가 이 두 집합을 **분리(separates)**한다는 것은:

$$\forall x \in C_1:\ a^\top x \leq b \quad \text{and} \quad \forall x \in C_2:\ a^\top x \geq b$$

**강한 분리(strict separation)**: 위의 $\leq, \geq$를 $<, >$로 강화.

### 정의 3.3 — 지지 초평면 (Supporting Hyperplane)

볼록 집합 $C$와 경계점 $x_0 \in \partial C$에 대해, 초평면 $\{a^\top x = a^\top x_0\}$ ($a \neq 0$)가 **지지 초평면**이라는 것은:

$$\forall x \in C:\ a^\top x \leq a^\top x_0$$

(또는 $a^\top x \geq a^\top x_0$)

---

## 🔬 정리와 증명

### 정리 3.1 — 볼록 집합과 외부 점의 분리

**명제**: $C \subseteq \mathbb{R}^n$이 공집합이 아닌 닫힌 볼록 집합이고 $y \notin C$이면, $y$와 $C$를 강하게 분리하는 초평면이 존재한다.

**보조 정리**: 닫힌 볼록 집합 $C$와 $y \notin C$에 대해, $C$에서 $y$에 가장 가까운 점 $x^* = \Pi_C(y) = \arg\min_{x \in C} \|x - y\|$가 유일하게 존재하고:

$$\forall z \in C:\ (y - x^*)^\top (z - x^*) \leq 0 \tag{*}$$

**조건 (*)의 기하학적 의미**: 벡터 $y - x^*$는 $C$의 경계에서 "바깥쪽"을 가리킨다. $z - x^*$는 $x^*$에서 $C$ 안의 임의의 점으로의 벡터. 두 벡터의 내적이 ≤ 0 — 즉 90도 이상 벌어진다.

**분리 초평면 구성**: $a = y - x^*$로 놓으면:

$$b = \frac{a^\top y + a^\top x^*}{2} = a^\top x^* + \frac{\|a\|^2}{2}$$

**증명**:

*1단계 — $y$와 $C$ 분리*:  
$a^\top y = (y-x^*)^\top y = (y-x^*)^\top(y - x^* + x^*) = \|y-x^*\|^2 + a^\top x^* > a^\top x^*$  
(∵ $y \neq x^*$이므로 $\|y-x^*\|^2 > 0$)

*2단계 — $C$의 임의의 점 $z$에 대해*:  
$(*)$에 의해 $(y-x^*)^\top(z-x^*) \leq 0$, 즉 $a^\top z \leq a^\top x^* < a^\top y$.

따라서 $a^\top x^* < b < a^\top y$로 잡으면 강한 분리가 된다. $\square$

---

### 정리 3.2 — 분리 초평면 정리 (Separating Hyperplane Theorem)

**명제**: $C_1, C_2 \subseteq \mathbb{R}^n$이 공집합이 아닌 볼록 집합이고 $C_1 \cap C_2 = \emptyset$이면, 두 집합을 분리하는 초평면이 존재한다.

**증명 (볼록 집합 간 거리 이용)**:  
Minkowski 차 $C = C_1 - C_2 = \{x - y \mid x \in C_1, y \in C_2\}$를 정의하면:
- $C$는 볼록 (볼복 집합의 Minkowski 합은 볼록, 음수 스케일링도 볼록)
- $0 \notin C$ ($C_1 \cap C_2 = \emptyset$이므로 $x - y = 0$인 점 없음)

정리 3.1에 의해 $0$과 $C$를 분리하는 초평면 $a^\top z = b$ ($b > 0$)이 존재:

$$\forall z \in C:\ a^\top z \geq b > 0$$

임의의 $x \in C_1$, $y \in C_2$에 대해 $x - y \in C$이므로:

$$a^\top(x - y) \geq b > 0 \implies a^\top x \geq a^\top y + b$$

$b_0 = \inf_{x \in C_1} a^\top x$로 놓으면 $b_0 \geq \sup_{y \in C_2} a^\top y + b > \sup_{y \in C_2} a^\top y$.

따라서 임의의 $b' \in (\sup_{y \in C_2} a^\top y, b_0)$에 대해 $\{a^\top x = b'\}$이 분리 초평면이다. $\square$

> **주의**: 내부(interior)를 가지지 않는 경우 (예: 두 집합이 점으로 만나는 경우) 강한 분리(strict separation)는 보장되지 않는다.

---

### 정리 3.3 — 지지 초평면 정리 (Supporting Hyperplane Theorem)

**명제**: $C$가 공집합이 아닌 볼록 집합이고 $x_0 \in \partial C$ (경계)이면, $x_0$에서 $C$의 지지 초평면이 존재한다.

**증명**: $x_0 \notin \text{int}(C)$이므로 $x_0 \to$ 점들의 수열 $y_k \notin C$, $y_k \to x_0$이 존재.  
정리 3.1에 의해 각 $y_k$와 $C$를 분리하는 단위 벡터 $a_k$ ($\|a_k\|=1$)가 존재:

$$\forall x \in C:\ a_k^\top x \leq a_k^\top y_k$$

유계 수열 $\{a_k\}$는 수렴 부분 수열 $a_{k_j} \to a$ ($\|a\|=1$)를 가진다. 극한을 취하면:

$$\forall x \in C:\ a^\top x \leq a^\top x_0$$

이것이 $x_0$에서의 지지 초평면이다. $\square$

---

### 따름정리 — 볼록 함수의 1차 조건과의 연결

미분가능한 볼록 함수 $f$에서 점 $(x_0, f(x_0))$는 epigraph $\text{epi}(f) = \{(x, t) \mid f(x) \leq t\}$의 경계에 있다.

지지 초평면 정리에 의해 어떤 $g \in \mathbb{R}^n$이 존재하여:

$$\forall (x, t) \in \text{epi}(f):\ g^\top(x - x_0) \leq t - f(x_0)$$

특히 $t = f(x)$로 놓으면:

$$f(x) \geq f(x_0) + g^\top(x - x_0) \quad \forall x$$

이것이 볼록 함수의 1차 조건이며, $f$가 미분가능하면 $g = \nabla f(x_0)$이다. 미분불가능한 경우 $g$를 **subgradient**라 한다.

---

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ─────────────────────────────────────────────
# 1. 두 서로소 볼록 집합의 분리 초평면 시각화
# ─────────────────────────────────────────────

def find_separating_hyperplane(C1_pts, C2_pts):
    """
    두 볼록 집합을 분리하는 초평면을 찾는다.
    접근: Minkowski 차의 원점과의 최단 거리 벡터
    여기서는 두 집합의 각 점 쌍 중 최단 거리 쌍을 찾아 분리 벡터를 구한다.
    """
    min_dist = np.inf
    best_x, best_y = None, None

    # 간단한 버전: 모든 볼록 포 꼭짓점 쌍을 탐색
    for x in C1_pts:
        for y in C2_pts:
            d = np.linalg.norm(x - y)
            if d < min_dist:
                min_dist = d
                best_x, best_y = x, y

    # 분리 벡터: 두 가장 가까운 점을 잇는 벡터
    a = best_x - best_y  # C2 → C1 방향
    a = a / np.linalg.norm(a)  # 단위 벡터
    b = a @ ((best_x + best_y) / 2)  # 중간점에서의 절편

    return a, b, best_x, best_y

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 예시 1: 두 타원형 볼록 집합
theta = np.linspace(0, 2 * np.pi, 100)

# C₁: 중심 (-2, 0), 반축 (1, 0.6)
C1_x = -2 + 1.0 * np.cos(theta)
C1_y = 0.6 * np.sin(theta)
C1_pts = np.stack([C1_x, C1_y], axis=1)

# C₂: 중심 (1.5, 0.5), 반축 (0.8, 1.0)
C2_x = 1.5 + 0.8 * np.cos(theta)
C2_y = 0.5 + 1.0 * np.sin(theta)
C2_pts = np.stack([C2_x, C2_y], axis=1)

# 볼록 포 꼭짓점만 추출
hull1 = ConvexHull(C1_pts)
hull2 = ConvexHull(C2_pts)
C1_hull = C1_pts[hull1.vertices]
C2_hull = C2_pts[hull2.vertices]

a, b, best_x, best_y = find_separating_hyperplane(C1_hull, C2_hull)

ax = axes[0]
ax.fill(C1_x, C1_y, alpha=0.35, color='steelblue', label='$C_1$')
ax.fill(C2_x, C2_y, alpha=0.35, color='orange', label='$C_2$')

# 분리 초평면 그리기: a^T x = b → 법선 방향 a, 직교 방향 a_perp
a_perp = np.array([-a[1], a[0]])
t_range = np.linspace(-3, 3, 100)
hyp_x = b * a[0] + t_range * a_perp[0]
hyp_y = b * a[1] + t_range * a_perp[1]
ax.plot(hyp_x, hyp_y, 'r-', lw=2, label='분리 초평면')

# 최근접 점 및 화살표
ax.scatter(*best_x, color='blue', s=80, zorder=6)
ax.scatter(*best_y, color='blue', s=80, zorder=6)
ax.annotate('', xy=best_x, xytext=best_y,
            arrowprops=dict(arrowstyle='<->', color='blue', lw=2))

ax.set_xlim(-4, 3.5); ax.set_ylim(-2, 2.5)
ax.set_title('분리 초평면 정리\n(두 서로소 볼록 집합)', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 예시 2: 지지 초평면
theta2 = np.linspace(0, 2 * np.pi, 200)
# 비대칭 볼록 집합
C_x = 1.2 * np.cos(theta2) - 0.3 * np.cos(2*theta2)
C_y = np.sin(theta2)
# 경계점: 가장 오른쪽 점
idx = np.argmax(C_x)
x0 = np.array([C_x[idx], C_y[idx]])

# 지지 초평면: 이 점에서 x축 방향 법선 (볼록 집합의 최외곽)
a_sup = np.array([1, 0])  # 오른쪽 방향
b_sup = a_sup @ x0

t2 = np.linspace(-2, 2, 100)
sup_x = b_sup * np.ones(100)
sup_y = t2

ax = axes[1]
ax.fill(C_x, C_y, alpha=0.35, color='steelblue', label='볼록 집합 $C$')
ax.plot(C_x, C_y, 'b-', lw=1.5)
ax.plot(sup_x, sup_y, 'r-', lw=2.5, label='지지 초평면 at $x_0$')
ax.scatter(*x0, color='red', s=100, zorder=6, label=f'경계점 $x_0 = ({x0[0]:.2f}, {x0[1]:.2f})$')
ax.annotate('$x_0$', xy=x0, xytext=(x0[0]+0.2, x0[1]+0.2), fontsize=12, color='red')
ax.set_xlim(-2, 2.5); ax.set_ylim(-1.8, 1.8)
ax.set_title('지지 초평면 정리\n경계점에서 전체 집합이 한쪽에', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.suptitle('분리 초평면과 지지 초평면', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('03-separating-supporting-hyperplane.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 2. SVM의 최적 분리 초평면 시각화
# ─────────────────────────────────────────────

np.random.seed(42)
# 선형 분리 가능한 두 클래스
X_pos = np.random.randn(20, 2) + np.array([2, 1])
X_neg = np.random.randn(20, 2) + np.array([-1, -1])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_pos[:, 0], X_pos[:, 1], c='steelblue', s=60, label='Class +1', zorder=5)
ax.scatter(X_neg[:, 0], X_neg[:, 1], c='orange', s=60, label='Class -1', zorder=5)

# 간단한 퍼셉트론으로 분리 초평면 찾기 (sklearn SVM)
from sklearn.svm import SVC
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(20), -np.ones(20)])
svm = SVC(kernel='linear', C=1e6)
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]

x_range = np.linspace(-4, 6, 300)
decision = -(w[0] * x_range + b) / w[1]
margin_pos = -(w[0] * x_range + b + 1) / w[1]
margin_neg = -(w[0] * x_range + b - 1) / w[1]

ax.plot(x_range, decision, 'r-', lw=2.5, label='분리 초평면 (SVM)')
ax.plot(x_range, margin_pos, 'r--', lw=1.5, alpha=0.6, label='Margin (+1)')
ax.plot(x_range, margin_neg, 'r--', lw=1.5, alpha=0.6, label='Margin (-1)')
ax.fill_between(x_range, margin_neg, margin_pos, alpha=0.1, color='red', label='Margin 영역')

# Support vectors
ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
           s=150, facecolors='none', edgecolors='red', lw=2.5,
           zorder=6, label='Support Vectors')

ax.set_xlim(-4, 6); ax.set_ylim(-4, 5)
ax.set_title('SVM의 최적 분리 초평면\n분리 초평면 정리의 응용', fontsize=12)
ax.legend(fontsize=9, loc='upper left'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('03-svm-separating-hyperplane.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"SVM 법선 벡터 w = {w}")
print(f"SVM 절편 b = {b:.4f}")
print(f"Margin 폭 = {2 / np.linalg.norm(w):.4f}")
```

**출력 예시**:
```
SVM 법선 벡터 w = [0.712  0.348]
SVM 절편 b = -1.2043
Margin 폭 = 2.5103
```

---

## 🔗 AI/ML 연결

### 강쌍대성 증명의 핵심 구조

Ch4-03에서 자세히 다루지만, Slater 조건이 강쌍대성을 보장하는 증명의 핵심 아이디어는 다음과 같다.

두 볼록 집합을 정의한다:
- $A = \{(u, v, t) \mid \exists x: f_i(x) \leq u_i, h_j(x) = v_j, f_0(x) \leq t\}$ (확장된 가능 집합)
- $B = \{(u, v, t) \mid u \leq 0, v = 0, t < p^*\}$ (primal optimal보다 나은 영역)

$A \cap B = \emptyset$이므로 (Slater 조건 하에서) 분리 초평면 정리에 의해 분리 초평면이 존재하고, 이 초평면의 계수가 바로 쌍대 변수 $(\lambda, \nu)$가 된다.

### 볼록 함수의 Subgradient

볼록 함수 $f$가 $x_0$에서 미분불가능하더라도 (예: $f(x) = |x|$, $x_0 = 0$), 지지 초평면 정리에 의해 subgradient $g$가 존재:

$$f(x) \geq f(x_0) + g^\top(x - x_0) \quad \forall x$$

Subgradient는 볼록 비미분가능 최적화(Proximal methods, LASSO 등)의 이론적 기반이다.

### SVM과 선형 분리 가능성

두 클래스의 볼록 포 $C_+ = \text{conv}(\{x_i \mid y_i = 1\})$, $C_- = \text{conv}(\{x_i \mid y_i = -1\})$이 서로소이면:

분리 초평면 정리 → 분리 초평면 존재 → Hard-margin SVM 해가 존재한다.

최적 margin은 $\frac{2}{\|w\|}$이며, 이를 최대화하는 $w$가 유일하게 존재한다 (목적함수 $\|w\|^2/2$가 강볼록이므로).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 두 집합이 서로소 ($C_1 \cap C_2 = \emptyset$) | 교집합이 있으면 분리 불가 |
| 볼록성 | 비볼록 집합은 볼록 포로 근사해야 함 |
| 닫힌 집합 (강한 분리) | 열린 볼록 집합은 분리 초평면이 경계에 닿을 수 있음 |
| 유한 차원 ($\mathbb{R}^n$) | 무한 차원 함수 공간에서는 Hahn-Banach 정리 필요 |

**강한 분리(strict separation) 실패 사례**: $C_1 = \{(x, y) \mid y \geq e^x\}$, $C_2 = \{(x, y) \mid y \leq 0\}$는 서로소 볼록 집합이지만, 임의의 분리 초평면이 $x \to -\infty$에서 두 집합에 동시에 접근 — 강한 분리 불가. 이 경우 분리 초평면은 $y = 0$이며 두 집합의 경계가 이 초평면에 점근한다.

**실용적 함의**: SVM의 Soft-margin은 두 클래스의 볼록 포가 겹칠 때 (강한 분리 불가) 사용한다. Slack 변수 $\xi_i$로 선형 분리 불가능한 경우를 처리한다.

---

## 📌 핵심 정리

$$C_1 \cap C_2 = \emptyset,\ C_1, C_2 \text{ 볼록} \implies \exists a \neq 0, b:\ a^\top x \leq b \leq a^\top y\ \forall x \in C_1, y \in C_2$$

| 정리 | 조건 | 결론 |
|------|------|------|
| 분리 초평면 정리 | $C_1, C_2$ 볼록, 서로소 | 분리 초평면 존재 |
| 지지 초평면 정리 | $C$ 볼록, $x_0 \in \partial C$ | $x_0$에서 지지 초평면 존재 |
| 볼록 함수 1차 조건 | $f$ 볼록, 미분가능 | $f(y) \geq f(x) + \nabla f(x)^\top(y-x)$ |

**이 정리의 의미**: 볼록 집합은 항상 "초평면으로 잘릴 수 있다". 이것이 볼록 최적화의 전역 최적 보장, 쌍대 이론, SVM의 이론적 기반 모두를 가능하게 한다.

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\mathbb{R}^2$에서 두 볼록 집합 $C_1 = \{(x_1, x_2) \mid x_1^2 + x_2^2 \leq 1\}$과 $C_2 = \{(x_1, x_2) \mid x_1 \geq 2\}$의 분리 초평면을 명시적으로 구하라.

<details>
<summary>힌트 및 해설</summary>

두 집합의 최근접 점: $C_1$에서 $(1, 0)$, $C_2$에서 $(2, 0)$.  
분리 벡터: $a = (2,0) - (1,0) = (1,0)$.  
분리 초평면: $x_1 = b$, $1 \leq b \leq 2$. (예: $x_1 = 1.5$)  
검증: $\forall (x_1,x_2) \in C_1$: $x_1 \leq 1 \leq b$. $\forall (x_1,x_2) \in C_2$: $x_1 \geq 2 \geq b$.

</details>

**문제 2** (심화): 강한 분리(strict separation)가 성립하기 위한 충분조건을 서술하라. $C_1 = \{(x, 0) \mid x \geq 0\}$과 $C_2 = \{(x, e^{-x}) \mid x \geq 0\}$은 서로소이지만 강한 분리가 불가능함을 보여라.

<details>
<summary>힌트 및 해설</summary>

**충분조건**: $C_1$이 콤팩트(닫힌 + 유계)이거나 두 집합이 양의 거리 $d(C_1, C_2) = \inf\{\|x-y\| \mid x \in C_1, y \in C_2\} > 0$를 가지면 강한 분리 가능.

**반례**: $C_1 = \{(x,0) \mid x \geq 0\}$, $C_2 = \{(x, e^{-x}) \mid x \geq 0\}$. $x \to \infty$에서 $e^{-x} \to 0$이므로 두 집합의 거리 $\inf_{x \geq 0} e^{-x} = 0$ (최솟값 미달성). 따라서 강한 분리 초평면 불가.

</details>

**문제 3** (AI 연결): SVM에서 두 클래스의 볼록 포가 서로 교차할 때(선형 분리 불가능) Soft-margin SVM이 어떻게 이 문제를 해결하는지 분리 초평면 정리의 관점에서 설명하라.

<details>
<summary>힌트 및 해설</summary>

볼록 포가 겹치면 분리 초평면이 존재하지 않는다. Soft-margin SVM은 slack 변수 $\xi_i \geq 0$을 도입하여 제약을 $y_i(w^\top x_i + b) \geq 1 - \xi_i$로 완화한다. 이는 원래 데이터 공간이 아닌 "허용된 오분류 비용을 포함한 확장 공간"에서 분리 초평면을 찾는 것과 동치다. $C$ 파라미터는 margin 최대화와 오분류 허용 간의 trade-off를 제어한다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. 볼록 집합의 연산](./02-convex-set-operations.md) | [📚 README](../README.md) | [04. 볼록 콘과 쌍대 콘 ▶](./04-convex-cone-dual.md) |

</div>
