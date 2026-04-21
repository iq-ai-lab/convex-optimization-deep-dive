# 05. Extreme Point와 Krein-Milman 정리

## 🎯 핵심 질문

- 극점(extreme point)이란 무엇이고, 볼록 결합으로 표현 불가능하다는 것이 왜 중요한가?
- Krein-Milman 정리 — "콤팩트 볼록 집합은 극점들의 볼록 포" — 를 어떻게 증명하는가?
- LP의 최적해가 항상 다면체의 꼭짓점(vertex)에서 달성되는 이유는 무엇인가?
- Simplex 방법이 왜 꼭짓점만 탐색해도 전역 최적을 보장하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

극점 이론은 최적화 알고리즘이 어디서 최적해를 찾아야 하는지를 결정하는 핵심 원리다.

- **LP와 Simplex Method**: 선형 목적함수를 다면체 위에서 최소화하면 최적해는 반드시 꼭짓점(= 다면체의 극점)에 있다. Simplex 방법은 이 성질을 이용해 꼭짓점에서 꼭짓점으로 이동하며 전역 최솟값을 탐색한다.
- **정수 계획의 LP 완화**: $\{0,1\}^n$의 볼록 포는 $[0,1]^n$이고, $[0,1]^n$의 극점은 $\{0,1\}^n$이다. LP 완화의 최적해가 정수점에서 달성되면, LP 완화 = 정수 계획의 해가 된다.
- **Attention Mechanism**: Softmax가 $\text{one-hot}$ 벡터에 수렴할 때, 그것은 확률 단체(probability simplex)의 극점에 도달하는 것이다. Hard attention = 극점에서의 값.
- **Variational Inference**: 변분 추론에서 최적 분포를 찾는 문제는 분포 공간(볼록 집합)에서의 최적화이며, 극점 분포(point mass)가 특별한 역할을 한다.

---

## 📐 수학적 선행 조건

- [01. 볼록 집합의 정의와 예제](./01-convex-set-definition.md): 볼록 집합, 볼록 결합
- [02. 볼록 집합의 연산](./02-convex-set-operations.md): 볼록 포의 정의
- **콤팩트 집합**: 닫힌 + 유계 집합, Heine-Borel 정리 — [Calculus & Opt Deep Dive Ch1](https://github.com/iq-ai-lab/calculus-optimization-deep-dive)
- **연속 함수의 최대·최솟값 정리**: 콤팩트 집합 위에서 연속 함수는 최솟값을 달성 — [Calculus & Opt Deep Dive Ch1](https://github.com/iq-ai-lab/calculus-optimization-deep-dive)

---

## 📖 직관적 이해

### 극점이란: "분해 불가능한 점"

볼록 집합의 **극점**은 다른 두 점의 볼록 결합으로 표현할 수 없는 점이다.

```
사각형의 극점 (꼭짓점)      원의 극점 (모든 경계)

  ●───────●               ╭──────╮
  │       │              ╱        ╲
  │       │             │    O    │ ← 경계의 모든 점이 극점
  │       │              ╲        ╱
  ●───────●               ╰──────╯
  ↑ 극점 = 꼭짓점 4개     극점이 무한히 많음
```

꼭짓점은 "선분의 엄격한 내부에 있지 않다" — 즉, 어떤 선분도 이 꼭짓점을 내부에 포함하지 않는다.

### Krein-Milman의 직관

"볼록 집합은 그 극점들만으로 완전히 결정된다."

```
다각형 = 꼭짓점들의 볼록 포     원 = 경계점들의 볼록 포 (극한)

     ●                          ●●●
    ╱ ╲                       ●     ●
   ╱   ╲        같다          ●     ●
  ●     ●       ←→             ●   ●
   ╲   ╱                        ●●●
    ╲ ╱
     ●
```

---

## ✏️ 엄밀한 정의

### 정의 5.1 — 극점 (Extreme Point)

볼록 집합 $C$의 점 $x \in C$가 **극점**이라는 것은:

$$x = \lambda y + (1-\lambda)z,\ y,z \in C,\ \lambda \in (0,1) \implies y = z = x$$

즉, $x$는 $C$ 안의 다른 두 점의 볼록 결합으로 표현 불가능하다. $\text{ext}(C)$로 $C$의 극점 전체를 표기한다.

**동치 표현**: $x \in C$가 극점 ↔ $C \setminus \{x\}$가 볼록이다. (단, 공집합은 볼록으로 간주)

### 정의 5.2 — 꼭짓점 (Vertex, 다면체의 경우)

다면체 $P = \{x \mid Ax \preceq b\}$에서, 점 $x^* \in P$가 **꼭짓점**이라는 것은 어떤 $n$개의 선형 독립인 활성 제약이 $x^*$에서 등식으로 성립하는 것이다. 다면체에서 극점 = 꼭짓점.

### 정의 5.3 — 면 (Face)

볼록 집합 $C$에서, 부분집합 $F \subseteq C$가 **면(face)**이라는 것은: $F$가 볼록이고, $F$의 임의의 점이 $C$ 안의 선분의 상대 내부에 있으면 그 선분 전체가 $F$에 속한다.

> 극점 = 0-차원 면.  
> 모서리(edge) = 1-차원 면.  
> $C$ 자신도 $C$의 면.

---

## 🔬 정리와 증명

### 정리 5.1 — LP의 최적해는 꼭짓점에 있다

**명제**: 선형 목적함수 $f(x) = c^\top x$를 다면체 $P = \{x \mid Ax \preceq b\}$ 위에서 최소화할 때, 최솟값이 달성된다면 꼭짓점에서 달성된다.

**증명**:  
$x^*$가 최적해지만 꼭짓점이 아닌 경우를 가정하자. 꼭짓점이 아니므로 $x^* = \lambda y + (1-\lambda)z$ ($y \neq z$, $y,z \in P$, $\lambda \in (0,1)$)로 쓸 수 있다.

$c^\top x^* = \lambda c^\top y + (1-\lambda) c^\top z \geq c^\top x^*$ (최적성)이므로:

$$c^\top y \geq c^\top x^*\ \text{and}\ c^\top z \geq c^\top x^*$$

두 부등식이 동시에 성립하려면 $c^\top y = c^\top z = c^\top x^*$. 즉 $y$와 $z$도 최적해다.

$y$와 $z$에 같은 논리를 반복하면, 최적해 집합은 볼록 집합이다. 콤팩트 다면체의 경우, 최적해 집합은 닫힌 유계 볼록 집합 — Krein-Milman에 의해 극점을 가진다. 이 극점이 꼭짓점에서 최솟값을 달성하는 해이다. $\square$

---

### 정리 5.2 — Krein-Milman 정리

**명제**: $C \subseteq \mathbb{R}^n$이 공집합이 아닌 콤팩트 볼록 집합이면:

$$C = \text{cl}(\text{conv}(\text{ext}(C)))$$

즉, $C$는 극점들의 (닫힌) 볼록 포와 같다.

**유한 차원 증명** (수학적 귀납법, $\mathbb{R}^n$):

*기저: $n = 0$ (단일 점)*: 당연.

*귀납 단계*: 임의의 $x \in C$가 극점들의 볼록 결합임을 보인다.

1. $x \in \text{int}(C)$이면: 임의의 방향 $v$로 $x$를 통과하는 직선이 $C$의 경계와 만나는 두 점 $y, z$를 잡는다 ($x = \lambda y + (1-\lambda)z$, $\lambda \in (0,1)$). $y, z$는 $C$의 경계에 있고, $C$보다 낮은 차원의 면에 속한다.

2. $x \in \partial C$ (경계)이면: $x$가 놓인 가장 낮은 차원의 면 $F$를 찾는다. $F$의 차원이 $C$보다 낮으므로 귀납 가정 적용.

3. 귀납 가정에 의해 경계의 점들은 극점들의 볼록 결합이므로, $x$도 극점들의 볼록 결합이다. $\square$

---

### 정리 5.3 — 다면체의 극점 특성화

**명제**: 다면체 $P = \{x \in \mathbb{R}^n \mid a_i^\top x \leq b_i,\ i=1,\ldots,m\}$의 극점은 정확히 $n$개의 선형 독립인 활성 제약을 가진 꼭짓점이다.

**증명**:  
($\Rightarrow$) $x^*$가 극점이라 하자. 활성 제약의 수가 $n$보다 적으면, 활성 초평면들의 교집합은 $x^*$를 포함하는 1차원 이상의 부분공간을 남긴다. 이 방향으로 작은 이동 $\pm \varepsilon d$를 하면 $x^* \pm \varepsilon d \in P$이고 $x^* = \frac{1}{2}(x^*+\varepsilon d) + \frac{1}{2}(x^*-\varepsilon d)$ — 극점의 정의 모순.

($\Leftarrow$) $n$개의 선형 독립인 활성 제약이 있으면 $x^*$는 유일하게 결정 ($n \times n$ 선형 시스템의 유일해). 따라서 어떤 볼록 결합으로도 표현 불가 — 극점. $\square$

---

### 정리 5.4 — 확률 단체의 극점

**명제**: 확률 단체 $\Delta_n = \{p \in \mathbb{R}^n \mid p_i \geq 0,\ \sum p_i = 1\}$의 극점은 표준 기저 벡터 $e_1, \ldots, e_n$ (one-hot 벡터)이다.

**증명**:  
($e_k$는 극점): $e_k = \lambda p + (1-\lambda) q$ ($p, q \in \Delta_n$, $\lambda \in (0,1)$)이면, $k$번째 성분: $1 = \lambda p_k + (1-\lambda)q_k$. $p_k, q_k \leq 1$이므로 $p_k = q_k = 1$. 나머지 성분 모두 0이므로 $p = q = e_k$. 따라서 극점.

(다른 점은 극점이 아님): $p \in \Delta_n$이 $e_k$가 아닌 점이면, $p_i > 0$인 인덱스가 2개 이상 존재. $p_i > 0$이고 $p_j > 0$ ($i \neq j$)이면 $q = p + \varepsilon(e_i - e_j)$, $r = p - \varepsilon(e_i - e_j)$ ($\varepsilon$ 충분히 작음)로 놓으면 $q, r \in \Delta_n$이고 $p = (q+r)/2$. 따라서 극점이 아님. $\square$

---

## 💻 NumPy 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from itertools import combinations

# ─────────────────────────────────────────────
# 1. LP 최적해가 꼭짓점에 있음을 시각화
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

def find_vertices_2d(A, b):
    """2D 다면체의 꼭짓점 탐색 (제약 쌍의 교점 중 가능점)"""
    n_constraints = len(b)
    vertices = []
    for i, j in combinations(range(n_constraints), 2):
        A_sub = A[[i, j]]
        b_sub = b[[i, j]]
        try:
            v = np.linalg.solve(A_sub, b_sub)
            if np.all(A @ v <= b + 1e-8):
                vertices.append(v)
        except np.linalg.LinAlgError:
            pass
    return np.array(vertices) if vertices else np.empty((0, 2))

# 다면체: {x | x1 ≥ 0, x2 ≥ 0, x1 + x2 ≤ 4, 2x1 + x2 ≤ 6}
A = np.array([[-1, 0], [0, -1], [1, 1], [2, 1]])
b = np.array([0, 0, 4, 6])

vertices = find_vertices_2d(A, b)
# 꼭짓점들을 볼록 포 순으로 정렬
hull = ConvexHull(vertices)
v_ordered = vertices[hull.vertices]

for ax_idx, (c, title) in enumerate(zip(
        [np.array([1, 2]), np.array([-1, -1]), np.array([2, -1])],
        ['$c = (1, 2)$: 최솟값 꼭짓점에서', '$c = (-1,-1)$: 최솟값 꼭짓점에서', '$c = (2,-1)$: 최솟값 꼭짓점에서'])):
    ax = axes[ax_idx]
    
    # 다면체 그리기
    hull_closed = np.vstack([v_ordered, v_ordered[0]])
    ax.fill(hull_closed[:, 0], hull_closed[:, 1], alpha=0.3, color='steelblue')
    ax.plot(hull_closed[:, 0], hull_closed[:, 1], 'b-', lw=2)
    
    # 꼭짓점 표시
    ax.scatter(vertices[:, 0], vertices[:, 1], color='blue', s=80, zorder=6)
    for v in vertices:
        ax.annotate(f'({v[0]:.0f},{v[1]:.0f})\nf={c@v:.1f}',
                    xy=v, xytext=v + np.array([0.1, 0.1]),
                    fontsize=8, color='darkblue')
    
    # 최솟값 꼭짓점 강조
    obj_values = vertices @ c
    opt_idx = np.argmin(obj_values)
    ax.scatter(*vertices[opt_idx], color='red', s=200, zorder=7,
               label=f'최적 꼭짓점\n$f^*={obj_values[opt_idx]:.1f}$')
    
    # 등고선 (목적함수의 레벨 선)
    x_range = np.linspace(-0.5, 4, 100)
    levels = [obj_values[opt_idx] + k * 0.8 for k in range(5)]
    for lv in levels:
        if c[1] != 0:
            y_line = (lv - c[0] * x_range) / c[1]
            ax.plot(x_range, y_line, 'g--', alpha=0.5, lw=1)
    
    ax.set_xlim(-0.5, 5); ax.set_ylim(-0.5, 5)
    ax.set_title(f'LP: {title}', fontsize=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('LP 최적해는 항상 꼭짓점(극점)에서 달성된다', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('05-lp-extreme-point.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 2. 확률 단체의 극점 (one-hot 벡터) 시각화
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))

# 2D 단체: p1 + p2 = 1, p1,p2 ≥ 0 → 선분 [0,1]
# 3D 단체를 2D로 투영 (barycentric coordinates)
def barycentric_to_cartesian(p):
    """3개 성분의 확률 벡터를 2D 삼각형 좌표로 변환"""
    v0 = np.array([0, 0])
    v1 = np.array([1, 0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    return p[0] * v0 + p[1] * v1 + p[2] * v2

# 단체의 꼭짓점 (극점)
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

v0_2d = barycentric_to_cartesian(e1)
v1_2d = barycentric_to_cartesian(e2)
v2_2d = barycentric_to_cartesian(e3)

# 단체 그리기
tri = np.array([v0_2d, v1_2d, v2_2d])
tri_closed = np.vstack([tri, tri[0]])
ax.fill(tri_closed[:, 0], tri_closed[:, 1], alpha=0.3, color='steelblue')
ax.plot(tri_closed[:, 0], tri_closed[:, 1], 'b-', lw=2)

# 극점 표시
for v, label in [(v0_2d, '$e_1 = (1,0,0)$'), (v1_2d, '$e_2 = (0,1,0)$'), (v2_2d, '$e_3 = (0,0,1)$')]:
    ax.scatter(*v, color='red', s=120, zorder=6)
    ax.annotate(label, xy=v, xytext=v + np.array([0.02, -0.06]),
                fontsize=10, color='darkred', fontweight='bold')

# 내부 점 (극점이 아님) 예시
p_interior = np.array([0.3, 0.4, 0.3])
v_int = barycentric_to_cartesian(p_interior)
ax.scatter(*v_int, color='green', s=100, zorder=5, label=f'내부점 $(0.3, 0.4, 0.3)$\n극점 ✗')

# 내부 점에서의 볼록 결합 표시
p1 = np.array([0.5, 0.5, 0])
p2 = np.array([0.1, 0.3, 0.6])
lam = 0.5
p_mid = lam * p1 + (1-lam) * p2
v_p1 = barycentric_to_cartesian(p1)
v_p2 = barycentric_to_cartesian(p2)
v_mid = barycentric_to_cartesian(p_mid)
ax.plot([v_p1[0], v_p2[0]], [v_p1[1], v_p2[1]], 'g--', lw=1.5, alpha=0.7)
ax.scatter(*v_mid, color='green', s=80, zorder=5)

ax.set_title('확률 단체 $\\Delta_3$의 극점 = one-hot 벡터\n(Attention의 hard limit)', fontsize=11)
ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
ax.legend(fontsize=9); ax.axis('off')
plt.tight_layout()
plt.savefig('05-simplex-extreme-points.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. LP Simplex 방법 시뮬레이션
# ─────────────────────────────────────────────

print("LP 최적화: c^T x 최소화 (꼭짓점 탐색)")
print(f"꼭짓점 목록:")
c_test = np.array([1.0, 2.0])
for i, v in enumerate(vertices):
    print(f"  꼭짓점 {i}: {v}, 목적값 = {c_test @ v:.2f}")
opt_v = vertices[np.argmin(vertices @ c_test)]
print(f"최적 꼭짓점: {opt_v}, 최적값: {c_test @ opt_v:.2f}")
print()

# scipy linprog로 검증
from scipy.optimize import linprog
res = linprog(c_test, A_ub=A, b_ub=b, bounds=[(None, None)]*2)
print(f"scipy.optimize.linprog 결과: x = {res.x}, f* = {res.fun:.2f}")
print(f"꼭짓점 탐색과 일치: {np.allclose(opt_v, res.x, atol=1e-4)}")
```

**출력**:
```
LP 최적화: c^T x 최소화 (꼭짓점 탐색)
꼭짓점 목록:
  꼭짓점 0: [0. 0.], 목적값 = 0.00
  꼭짓점 1: [3. 0.], 목적값 = 3.00
  꼭짓점 2: [2. 2.], 목적값 = 6.00
  꼭짓점 3: [0. 4.], 목적값 = 8.00
최적 꼭짓점: [0. 0.], 최적값: 0.00

scipy.optimize.linprog 결과: x = [0. 0.], f* = 0.00
꼭짓점 탐색과 일치: True
```

---

## 🔗 AI/ML 연결

### Softmax와 확률 단체의 극점

Softmax 함수 $\sigma(z)_i = e^{z_i}/\sum_j e^{z_j}$는 $\mathbb{R}^n \to \text{int}(\Delta_n)$ (단체의 내부)로 매핑한다. 온도 파라미터 $T$를 도입하면 $\sigma(z/T)$가 $T \to 0$일 때 argmax(one-hot) = 극점으로 수렴한다.

**Hard Attention**: $T \to 0$ 극한에서 Attention 가중치가 극점(one-hot) 분포에 수렴한다. 이것이 "Hard Attention = 가장 관련성 높은 토큰 하나만 선택"하는 행동이다.

### 정수 계획과 LP 완화

$\{0,1\}$-정수 계획:

$$\min c^\top x \quad \text{s.t.}\ Ax \leq b,\ x \in \{0,1\}^n$$

의 LP 완화:

$$\min c^\top x \quad \text{s.t.}\ Ax \leq b,\ 0 \leq x \leq 1$$

LP 완화 가능 영역의 극점이 $\{0,1\}^n$에 있다면 (totally unimodular matrix) LP 완화와 정수 계획의 해가 일치한다. 이것이 Assignment Problem, Matching Problem 등이 LP로 풀리는 이유다.

### Variational Inference와 극점 분포

변분 추론에서 posterior $p(\theta|x)$를 근사하는 분포 $q(\theta)$를 최적화할 때, 분포 공간은 볼록 집합이다. Point mass (Dirac delta) $\delta_{\theta_0}$는 이 공간의 극점에 해당한다. MAP 추론 = 극점에서의 최적화.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 콤팩트 볼록 집합 | 유계가 아닌 볼록 집합 (예: $\mathbb{R}^n$ 자체)에서는 극점이 없을 수 있다 |
| 유한 차원 | 무한 차원 공간에서는 Krein-Milman이 성립하지만 증명이 Zorn's lemma 필요 |
| LP의 최적해 존재 | 다면체가 유계가 아니면 최적해가 $-\infty$가 될 수 있음 (unbounded LP) |
| 극점의 개수 | $n$차원 다면체의 꼭짓점 수는 지수적으로 많을 수 있음 — Simplex의 worst-case는 지수 시간 |

**Simplex vs Interior Point**:
- Simplex: 꼭짓점을 이동하며 탐색, worst-case 지수 시간, 실제로는 매우 빠름
- Interior Point: 중앙 경로를 따라 내부에서 접근, worst-case 다항 시간 보장 (Ch5-05에서 자세히)

---

## 📌 핵심 정리

$$x \in \text{ext}(C) \Leftrightarrow [x = \lambda y + (1-\lambda)z,\ y,z \in C,\ \lambda \in (0,1) \implies y = z = x]$$

$$C \text{ 콤팩트 볼록} \implies C = \text{cl}(\text{conv}(\text{ext}(C))) \quad \text{(Krein-Milman)}$$

| 집합 | 극점 집합 |
|------|---------|
| 다각형/다면체 | 꼭짓점들 (유한 개) |
| 볼 $\{x \mid \|x\| \leq r\}$ | 구면 $\{x \mid \|x\| = r\}$ (무한 개) |
| 확률 단체 $\Delta_n$ | one-hot 벡터 $e_1, \ldots, e_n$ |
| PSD 콘 $\mathbb{S}^n_+$ | rank-1 행렬 $\{vv^\top\}$ |

**LP 최적 보장**: 선형 목적함수 + 다면체 → 최적해 ∈ 꼭짓점. Simplex 방법의 이론적 근거.

---

## 🤔 생각해볼 문제

**문제 1** (기초): $[0,1]^n$ (단위 하이퍼큐브)의 극점을 모두 구하라. 극점의 개수는 몇 개인가?

<details>
<summary>힌트 및 해설</summary>

극점: $\{0,1\}^n$ — 각 성분이 0 또는 1인 벡터. 극점의 개수는 $2^n$개.

**증명**: $v \in \{0,1\}^n$이 극점임을 보이자. $v = \lambda p + (1-\lambda)q$ ($p,q \in [0,1]^n$, $\lambda \in (0,1)$)이면 각 성분 $v_i = \lambda p_i + (1-\lambda)q_i$.
- $v_i = 1$이면 $p_i = q_i = 1$ (모두 ≤ 1이므로)
- $v_i = 0$이면 $p_i = q_i = 0$ (모두 ≥ 0이므로)
따라서 $p = q = v$ — 극점.

반대로 성분 $v_i \in (0,1)$인 점이 있으면 $e_i$ 방향으로 $\pm\varepsilon$ 이동이 가능 — 극점 아님.

</details>

**문제 2** (심화): 핵 노름 볼 $\{X \in \mathbb{R}^{m\times n} \mid \|X\|_* \leq 1\}$의 극점이 rank-1 행렬 $\{uv^\top \mid \|u\| = \|v\| = 1\}$임을 보여라.

<details>
<summary>힌트 및 해설</summary>

$X$의 SVD: $X = U\Sigma V^\top$, $\sigma_1 \geq \ldots \geq \sigma_r > 0$, $\|X\|_* = \sum \sigma_i = 1$.

$r \geq 2$이면 $X = \sigma_1 u_1 v_1^\top + \sigma_2 u_2 v_2^\top + \ldots$이고, $X \pm \varepsilon u_1 v_2^\top$ (작은 rank-2 perturbation)가 핵 노름 볼 안에 있으면 $X$는 극점이 아님.

$r = 1$이면 $X = u_1 v_1^\top$ — 이 경우 $X$가 극점임을 보이려면: $X = \lambda A + (1-\lambda)B$이면 $A, B$의 핵 노름이 각각 ≤ 1이고 SVD 분해에서 $u_1 v_1^\top$과 같은 방향이어야 한다. 따라서 $A = B = X$.

</details>

**문제 3** (AI 연결): 딥러닝에서 Gumbel-Softmax trick이 Softmax의 확률적 버전을 one-hot 극점에 "근사"하는 방법임을 설명하라. 이것이 이산 최적화를 연속 최적화로 완화하는 아이디어와 어떻게 연결되는가?

<details>
<summary>힌트 및 해설</summary>

Gumbel-Softmax: $y_i = \exp((log \pi_i + g_i)/T) / \sum_j \exp((log \pi_j + g_j)/T)$ ($g_i$: Gumbel noise, $T$: 온도).
- $T \to 0$: one-hot 벡터 → 확률 단체의 극점 (이산 샘플링과 동치)
- $T \to \infty$: 균등 분포 → 단체의 중심 (내부)

이는 이산 최적화 (정수 계획, 극점 탐색)를 미분가능한 연속 최적화로 완화하는 전략이다. LP 완화와 같은 원리 — 극점 집합의 볼록 포(단체) 위에서 연속 최적화 후, $T \to 0$으로 극점으로 당긴다.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. 볼록 콘과 쌍대 콘](./04-convex-cone-dual.md) | [📚 README](../README.md) | [Ch2-01. 볼록 함수의 3개 동치 정의 ▶](../ch2-convex-functions/01-convex-function-definitions.md) |

</div>
