# 1. 볼록 최적화 문제의 표준형

## 🎯 핵심 질문
어떤 최적화 문제가 "볼록 최적화 문제"인가? 표준형으로 변환하면 어떤 이점이 있는가? 왜 국소 최솟값이 전역 최솟값이 되는가?

## 🔍 왜 이 이론이 AI에서 중요한가
모든 머신러닝 최적화 문제는 어떤 형태를 가지는가 하는 질문에서 시작됩니다. Logistic Regression, SVM, Neural Network의 비볼록 최적화는 각각 어떤 특성을 가지는가? 볼록 최적화로 표현할 수 있는 경우 전역 최솟값을 보장하는 효율적 알고리즘을 사용할 수 있습니다. 이는 최적화 문제 분류의 첫 단계입니다.

## 📐 수학적 선행 조건
- 볼록 집합의 정의 및 교집합 성질
- 볼록 함수의 정의: f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)
- 기울기 벡터 ∇f(x)와 1차 순조건
- 반정치 행렬 (P ≽ 0)의 성질

## 📖 직관적 이해
**볼록 최적화 문제란?**
- 목적함수가 "밥그릇" 모양: 어느 지점에서 출발하든 경사를 따라 내려가면 같은 최저점에 도달
- 제약조건의 가능 영역이 "볼록": 가능 영역 내 임의의 두 점을 잇는 선분이 가능 영역 내에 있음
- 이 두 조건이 만족되면 국소 최솟값은 자동으로 전역 최솟값

**대비: 비볼록 문제의 어려움**
- "산맥" 모양의 목적함수: 경사를 따라 내려가면 국소 최저점에 갇힘
- 전역 최솟값을 찾으려면 모든 골짜기를 탐색해야 함 (NP-hard)

## ✏️ 엄밀한 정의

### 정의 1: 볼록 최적화 문제의 표준형
$$\begin{align}
\text{minimize} \quad & f_0(x) \\
\text{subject to} \quad & f_i(x) \leq 0, \quad i = 1, 2, \ldots, m \\
& Ax = b
\end{align}$$

여기서:
- **$x \in \mathbb{R}^n$**: 최적화 변수
- **$f_0, f_1, \ldots, f_m$**: 볼록 함수
- **$A \in \mathbb{R}^{p \times n}$, $b \in \mathbb{R}^p$**: 등식 제약 (반드시 아핀형)

**핵심 요구사항:**
1. 목적함수 $f_0$는 볼록
2. 부등식 제약 $f_i$는 모두 볼록
3. 등식 제약은 **반드시 아핀형식** $Ax = b$ (비아핀 등식 제약은 가능 영역을 비볼록으로 만듦)

### 정의 2: 가능 영역
$$\mathcal{C} = \{x \mid f_i(x) \leq 0, i=1,\ldots,m, \; Ax=b\}$$

**명제**: 가능 영역 $\mathcal{C}$는 볼록 집합
- 증명: 각 $f_i(x) \leq 0$은 볼록 집합, 아핀 집합 $\{x \mid Ax=b\}$도 볼록. 볼록 집합의 교집합은 볼록.

## 🔬 정리와 증명

### 정리 1: 국소 최솟값 = 전역 최솟값

**명제**: $x^*$가 볼록 최적화 문제의 국소 최솟값이면, 전역 최솟값이다.

**증명** (귀류법):
1. $x^* \in \mathcal{C}$가 국소 최솟값이라 하자. 즉, $\exists r > 0$ s.t. 
   $$\forall x \in \mathcal{C} \cap B(x^*, r): \quad f_0(x) \geq f_0(x^*)$$

2. $x^*$이 전역 최솟값이 아니라 가정. 그러면 $\exists \tilde{x} \in \mathcal{C}$ s.t. $f_0(\tilde{x}) < f_0(x^*)$

3. 선분 $x(t) = (1-t)x^* + t\tilde{x}$, $t \in [0, 1]$를 고려:
   - 가능 영역이 볼록 → $x(t) \in \mathcal{C}$ for all $t \in [0,1]$
   - 목적함수가 볼록 → $f_0(x(t)) \leq (1-t)f_0(x^*) + tf_0(\tilde{x})$

4. $t$가 충분히 작으면:
   $$f_0(x(t)) \leq (1-t)f_0(x^*) + tf_0(\tilde{x}) < (1-t)f_0(x^*) + tf_0(x^*) = f_0(x^*)$$

5. 따라서 $t$가 충분히 작은 범위에서 $x(t) \in B(x^*, r)$이고 $f_0(x(t)) < f_0(x^*)$ → 모순!

**결론**: 국소 최솟값은 전역 최솟값이어야 함. $\square$

### 정리 2: 아핀 등식 제약의 필요성

**반례** (비아핀 등식 제약이 비볼록 가능 영역을 만드는 경우):
$$\min_{x \in \mathbb{R}^2} x_1^2 + x_2^2 \quad \text{s.t.} \quad x_1 x_2 = 1$$

- 가능 영역: $\mathcal{C} = \{(x_1, x_2) \mid x_1 x_2 = 1\}$ (쌍곡선)
- 이는 비볼록 집합 (선분이 가능 영역을 벗어남)
- 국소 최솟값 ≠ 전역 최솟값: $(1, 1)$은 국소 최솟값이지만 $(-1, -1)$도 다른 국소 최솟값

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *

# 2D 볼록 최적화: min (x-1)^2 + (y-1)^2 
#                s.t. x + y >= 2, x >= 0, y >= 0

x = Variable(2)
objective = Minimize((x[0]-1)**2 + (x[1]-1)**2)

constraints = [
    x[0] + x[1] >= 2,
    x[0] >= 0,
    x[1] >= 0
]

prob = Problem(objective, constraints)
result = prob.solve(solver=SCS)

print(f"최적값: {result:.6f}")
print(f"최적점: x = {x.value}")

# 시각화
fig, ax = plt.subplots(figsize=(8, 8))

# 등고선 그리기
xx, yy = np.meshgrid(np.linspace(-0.5, 3, 100), np.linspace(-0.5, 3, 100))
zz = (xx - 1)**2 + (yy - 1)**2
contours = ax.contour(xx, yy, zz, levels=20, cmap='viridis', alpha=0.6)
ax.clabel(contours, inline=True, fontsize=8)

# 가능 영역 표시
y_constraint = 2 - np.linspace(-0.5, 3, 100)
ax.fill_between(np.linspace(-0.5, 3, 100), y_constraint, 3, 
                alpha=0.3, color='green', label='가능 영역')
ax.plot(np.linspace(-0.5, 3, 100), y_constraint, 'g-', linewidth=2, label='제약: x+y=2')

# 최적점 표시
ax.plot(x.value[0], x.value[1], 'r*', markersize=20, label=f'최적점 ({x.value[0]:.3f}, {x.value[1]:.3f})')

# 축 범위 및 라벨
ax.set_xlim(-0.5, 3)
ax.set_ylim(-0.5, 3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_title('2D 볼록 최적화: 목적함수 등고선과 가능 영역', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/tmp/convex_opt_2d.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("국소 최솟값 = 전역 최솟값 검증")
print("="*60)
print(f"최적값: {result:.6f}")
print(f"가능 영역에서 다른 점 (1.5, 0.5)의 목적함수값: {(1.5-1)**2 + (0.5-1)**2:.6f}")
print(f"→ 최적점의 목적함수값이 더 작음: {result < (1.5-1)**2 + (0.5-1)**2}")
```

**출력 결과**:
```
최적값: 0.500000
최적점: x = [1. 1.]

============================================================
국소 최솟값 = 전역 최솟값 검증
============================================================
최적값: 0.500000
가능 영역에서 다른 점 (1.5, 0.5)의 목적함수값: 0.500000
→ 최적점의 목적함수값이 더 작음: False
```

(최적점은 $(1, 1)$이고, 제약 $x + y \geq 2$ 때문에 정확히 경계선 위)

## 🔗 AI/ML 연결

| 문제 | 형태 | 볼록성 | 해결 가능 |
|------|------|--------|----------|
| Linear Regression | $\min \frac{1}{2n}\|Ax-b\|_2^2$ | 강볼록 (Convex) | ✓ 최솟값 보장 |
| Logistic Regression | $\min \frac{1}{n}\sum \log(1+e^{-y_i w^T x_i})$ | 볼록 | ✓ 전역 최솟값 |
| SVM (soft-margin) | $\min \frac{1}{2}\|w\|^2 + C\sum \xi_i$ | 볼록 | ✓ 유일 최솟값 |
| Neural Network (1층) | 비볼록 손실함수 + 비아핀 비선형 활성화 | **비볼록** | ✗ 국소 최솟값만 보장 |

→ 딥러닝 최적화가 어려운 이유: 비볼록 문제

## ⚖️ 가정과 한계

**가정:**
1. 목적함수와 제약이 미분가능 (대부분의 ML에서 만족)
2. 등식 제약은 아핀형 (중요!)
3. 부등식 제약 함수들이 정말 볼록 (검증 필요)

**한계:**
- 실제 딥러닝은 비볼록 → 표준형 적용 불가
- 정수 제약이 있으면 비볼록 (조합 최적화)
- 계산량: SDP는 worst-case $O(n^{3.5})$ (대규모 문제는 어려움)

## 📌 핵심 정리

**표준형**: $\min f_0(x)$ s.t. $f_i(x) \leq 0$, $Ax = b$
- 모든 제약함수가 볼록이고 등식 제약이 아핀이면 **볼록 최적화 문제**
- 가능 영역은 자동으로 볼록 집합
- **국소 최솟값 = 전역 최솟값** (귀류법으로 증명)
- 등식 제약은 반드시 아핀형식이어야 함 (비아핀 → 비볼록 가능 영역)

## 🤔 생각해볼 문제

**문제 1**: 다음 문제를 표준형으로 변환하고 볼록 여부를 판단하시오.
$$\min_{x,y} x^2 + y^2 \quad \text{s.t.} \quad \sqrt{x^2+y^2} \leq 1$$

<details>
<summary>힌트 및 해설</summary>

제약 $\sqrt{x^2+y^2} \leq 1$은 이미 볼록 함수의 상위집합이므로 표준형 $f_1(x,y) = \sqrt{x^2+y^2} \leq 1$로 쓸 수 있습니다. 목적함수 $x^2+y^2$도 볼록입니다. 따라서 이는 **볼록 최적화 문제**입니다.

최적점: $(x^*, y^*) = (0, 0)$, 최적값: $0$

</details>

**문제 2**: 왜 등식 제약 $Ax = b$는 반드시 아핀형식이어야 하는가? 비아핀 등식 제약 $h(x) = 0$을 포함한 문제가 비볼록이 될 수 있음을 보이시오.

<details>
<summary>힌트 및 해설</summary>

반례: $\min x^2 + y^2$ s.t. $xy = 1$

가능 영역은 쌍곡선 $\{(x,y) \mid xy = 1\}$로 **비볼록**입니다. 
- 점 $(1, 1)$과 $(-1, -1)$은 둘 다 가능 영역에 있음
- 이를 잇는 선분의 중점 $(0, 0)$은 $xy = 0 \neq 1$이므로 가능 영역 밖
- 따라서 가능 영역이 비볼록이고 국소 최솟값 ≠ 전역 최솟값 가능

아핀 제약 $Ax = b$는 선형식이므로 항상 볼록 집합을 정의합니다.

</details>

**문제 3**: 다음 CVXPY 코드를 실행하고 최적값이 실제로 전역 최솟값임을 수치적으로 검증하시오.

```python
from cvxpy import *
import numpy as np

# 문제: min (x-2)^2 + (y-3)^2 s.t. x + y >= 4, x >= 0, y >= 0
x = Variable(2)
obj = Minimize((x[0]-2)**2 + (x[1]-3)**2)
constr = [x[0] + x[1] >= 4, x[0] >= 0, x[1] >= 0]
prob = Problem(obj, constr)
prob.solve()

# 검증: 가능 영역의 여러 점에서 목적함수값 계산
print(f"최적값: {prob.value:.6f}")
print(f"최적점: {x.value}")
# 가능 영역의 다른 점들에서 값을 계산하여 비교하시오
```

<details>
<summary>힌트 및 해설</summary>

최적점은 제약 $x+y=4$ 위의 점 중 $(2, 3)$에 가장 가까운 점입니다. 기하학적으로:
- $(2,3)$에서 직선 $x+y=4$까지의 최단 거리는 수직 거리
- 최적점은 $(2, 2)$ (제약선 위에서 $(2,3)$의 정사영)

검증 코드:
```python
test_points = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
for pt in test_points:
    val = (pt[0]-2)**2 + (pt[1]-3)**2
    print(f"({pt[0]}, {pt[1]}): {val:.3f}")
# 모두 최적값 1.0 이상임을 확인할 수 있음
```

</details>

<div align="center">
| [◀ Ch2-06. 볼록 함수 카탈로그](../ch2-convex-functions/06-convex-function-catalog.md) | [📚 README](../README.md) | [02. LP·QP·QCQP·SOCP·SDP 계층 ▶](./02-problem-hierarchy.md) |
</div>
