# 05. Interior Point Method

## 🎯 핵심 질문
- 부등식 제약이 있는 최적화를 어떻게 풀 것인가?
- 장벽 함수는 왜 실행 가능 영역 내에 머물게 하는가?
- Interior point method가 다항 시간에 LP를 풀 수 있는 이유는?

## 🔍 왜 이 이론이 AI에서 중요한가
Interior point method (IPM)는 제약 최적화 문제를 unconstrained로 변환합니다. 선형계획 문제를 다항 시간에 풀 수 있으며, 이론적으로도 실무적으로도 중요합니다. 기계학습의 SVM, 정규화된 손실함수도 제약 문제로 변환 가능합니다.

## 📐 수학적 선행 조건
- **부등식 제약 최적화**: $\min f_0(x)$ s.t. $f_i(x) \le 0$, $Ax = b$
- **볼록성**: 모든 $f_i$가 볼록
- **로그**: $\log$ 함수의 성질 (오목, 미분가능)
- **중앙 경로**: 파라미터화된 최적해 궤적

## 📖 직관적 이해

**제약의 어려움**: 부등식 제약은 비가용 영역을 만듭니다. 최솟값이 경계에 있을 수 있으므로 경계를 추적해야 합니다.

**장벽 함수의 아이디어**: 제약 영역 안에서는 자유롭지만, 경계에 가까워지면 "벽"이 나타나는 함수를 만듭니다.
$$\phi(x) = -\sum_{i=1}^m \log(-f_i(x))$$

$f_i(x) = 0$ (경계)에서 $\phi(x) \to +\infty$입니다.

**중앙 경로**: 파라미터 $t > 0$를 제어하며 최적해가 내부에서 경계 쪽으로 이동합니다:
$$x^*(t) = \arg\min \{ t f_0(x) + \phi(x) \}$$

## ✏️ 엄밀한 정의

**정의 (제약 최적화 문제)**:
$$\begin{align}
\min \quad & f_0(x) \\
\text{s.t.} \quad & f_i(x) \le 0, \quad i=1,\ldots,m \\
& Ax = b
\end{align}$$

여기서 $f_i$는 모두 볼록, $f_0$는 목적함수.

**정의 (Log barrier function)**:
$$\phi(x) = -\sum_{i=1}^m \log(-f_i(x))$$

정의 영역: $\{x : f_i(x) < 0, i=1,\ldots,m\}$ (strictly feasible set).

**정의 (Barrier subproblem)**:
$$x^*(t) = \arg\min \{ t f_0(x) + \phi(x) : Ax = b \}$$

**정의 (Central path)**:
$$\mathcal{C} = \{x^*(t) : t > 0\}$$

## 🔬 정리와 증명

**정리 1 (중앙 경로 수렴)**

$f_0$가 볼록, $f_i$가 strongly convex이고 정상성 조건을 만족할 때:
$$\|x^*(t) - x^*\| \le \frac{m}{t}$$

여기서 $x^*$는 원래 문제의 최적해.

*증명*:

**Step 1**: Barrier subproblem의 최적성 조건 (KKT):
$$t \nabla f_0(x^*(t)) + \nabla \phi(x^*(t)) + A^T \lambda^*(t) = 0$$

분석적으로:
$$t \nabla f_0(x^*(t)) - \sum_{i=1}^m \frac{1}{-f_i(x^*(t))} \nabla f_i(x^*(t)) + A^T \lambda^*(t) = 0$$

$\mu_i^*(t) = -\frac{1}{t f_i(x^*(t))}$로 놓으면:
$$\nabla f_0(x^*(t)) + \sum_{i=1}^m \mu_i^*(t) \nabla f_i(x^*(t)) + A^T \lambda^*(t) = 0$$

**Step 2**: 원래 문제의 KKT 조건:
$$\nabla f_0(x^*) + \sum_{i=1}^m \mu_i^* \nabla f_i(x^*) + A^T \lambda^* = 0$$

보강 조건: $\mu_i^* f_i(x^*) = 0$ (complementarity).

**Step 3**: 오차 분석:
$$\|x^*(t) - x^*\|^2 \le \frac{c_1}{t^2} \|\mu^*(t) - \mu^*\|^2$$

Barrier property에서:
$$\mu_i^*(t) = -\frac{1}{t f_i(x^*(t))} \approx \frac{1}{t \cdot (\text{slack})} \sim O(1/t)$$

**Step 4**: 엄밀한 bound:
$$\|x^*(t) - x^*\| \le \frac{M}{t}$$

상수 $M$은 $m$과 함수 특성에 의존합니다. 대체로 $M = O(m)$. □

**정리 2 (IPM 수렴)**

Interior point method:
- 초기값: $t_0 > 0$, tolerence $\epsilon > 0$
- 반복: 
  - Newton으로 $x^*(t_k)$ 계산 ($\epsilon$ 정확도)
  - $t_{k+1} = \mu t_k$ ($\mu > 1$ factor)
  - $t_k > m/\epsilon$이 될 때까지 반복

그러면 $O(\sqrt{m} \log(m/\epsilon))$번의 Newton step으로 원래 문제를 $\epsilon$-최적으로 풉니다.

*증명 스케치*:

**Step 1**: Newton의 2차 수렴을 사용하면, 좋은 시작점에서 $O(\log \log(1/\epsilon))$번의 Newton으로 수렴.

**Step 2**: $t_0$에서 $t = m/\epsilon$까지 도달하려면:
$$t_k = \mu^k t_0 \ge m/\epsilon \Rightarrow k \ge \frac{1}{\log \mu}\log(m/(\epsilon t_0))$$

대체로 $\mu = 10$이면 $k = O(\log(m/\epsilon))$.

**Step 3**: 각 $t_k$에서 Newton이 수렴하는 데 $O(\sqrt{m})$번 필요 (self-concordance).

**Step 4**: 전체: $O(\sqrt{m}) \times O(\log(m/\epsilon)) = O(\sqrt{m}\log(m/\epsilon))$ Newton steps. □

**정리 3 (Self-concordance)**

$f$가 self-concordant라는 것은:
$$|D^3 f(x)[h,h,h]| \le 2(\nabla^2 f(x)[h,h])^{3/2}$$

Self-concordant 함수에 대해 barrier function $\phi(x) = -\sum \log(-f_i(x))$는 self-concordant이고, Newton method는:
$$\|x_{k+1} - x^*\| \le \frac{1}{2}(\frac{\lambda(x_k)}{2})^2 \|x_k - x^*\|$$

여기서 $\lambda(x) = \sqrt{\nabla \phi(x)^T (\nabla^2 \phi(x))^{-1} \nabla \phi(x)}$ (Newton decrement).

$\lambda(x) < 0.5$이면 quadratic convergence. □

## 💻 NumPy/CVXPY 구현으로 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 간단한 LP 문제:
# min  c^T x
# s.t. A x <= b (부등식 제약)
#      x >= 0 (비음 제약)

# 예제 1: 간단한 LP
c = np.array([1, 2])  # min x1 + 2*x2
A_ineq = np.array([
    [1, 1],     # x1 + x2 <= 4
    [2, -1],    # 2*x1 - x2 <= 2
    [-1, 0],    # -x1 <= 0 (x1 >= 0)
    [0, -1]     # -x2 <= 0 (x2 >= 0)
])
b_ineq = np.array([4, 2, 0, 0])

print("=" * 60)
print("Linear Programming Problem")
print("=" * 60)
print(f"Minimize: {c[0]}*x1 + {c[1]}*x2")
print(f"Subject to:")
print(f"  x1 + x2 <= 4")
print(f"  2*x1 - x2 <= 2")
print(f"  x1, x2 >= 0")

# Barrier method 구현
def barrier_method(c, A, b, t0=1.0, mu=10.0, max_iter=50, verbose=True):
    """
    Interior point method with logarithmic barrier
    
    Parameters:
    - c: objective coefficient
    - A: inequality constraint matrix (A*x <= b)
    - b: RHS of inequality constraints
    - t0: initial barrier parameter
    - mu: update factor (t_new = mu * t_old)
    - max_iter: maximum iterations
    """
    
    m = A.shape[0]  # number of constraints
    n = A.shape[1]  # number of variables
    
    # 초기화: strictly feasible point
    x = np.array([1.0, 1.0])  # x0
    
    # Feasibility 확인
    if np.any(A @ x >= b):
        x = (b - 0.1) / np.max(A, axis=0)  # 간단한 feasible point
    
    t = t0
    barrier_values = []
    objective_values = []
    x_trajectory = [x.copy()]
    
    for iteration in range(max_iter):
        # Barrier subproblem: min {t*c^T*x - sum(log(-f_i(x)))}
        # f_i(x) = A_i^T*x - b_i
        
        def barrier_objective(x):
            # t*f0(x) + phi(x)
            slacks = b - A @ x
            
            # Check feasibility
            if np.any(slacks <= 0):
                return 1e20  # infeasible
            
            f0 = c @ x
            phi = -np.sum(np.log(slacks))
            return t * f0 + phi
        
        def barrier_gradient(x):
            slacks = b - A @ x
            f0_grad = c
            phi_grad = A.T @ (1 / slacks)  # gradient of -sum(log(-f_i))
            return t * f0_grad + phi_grad
        
        # Newton 방법으로 최적화
        result = minimize(
            barrier_objective,
            x,
            method='BFGS',  # or 'Newton-CG'
            jac=barrier_gradient,
            options={'maxiter': 100, 'gtol': 1e-8}
        )
        
        x = result.x
        x_trajectory.append(x.copy())
        
        # 기록
        slacks = b - A @ x
        barrier_val = barrier_objective(x)
        obj_val = c @ x
        
        barrier_values.append(barrier_val)
        objective_values.append(obj_val)
        
        if verbose:
            print(f"Iteration {iteration:2d}: t={t:.1e}, "
                  f"obj={obj_val:.6f}, barrier={barrier_val:.6f}, "
                  f"x={x}")
        
        # 종료 조건: m/t < epsilon
        duality_gap = m / t
        if duality_gap < 1e-6:
            print(f"\nConverged at iteration {iteration}")
            break
        
        # t 업데이트
        t = mu * t
    
    return x, np.array(x_trajectory), np.array(barrier_values), np.array(objective_values)

# IPM 실행
x_opt, x_traj, barriers, objectives = barrier_method(c, A_ineq, b_ineq, verbose=True)

print(f"\nOptimal solution: x = {x_opt}")
print(f"Optimal objective: f(x*) = {c @ x_opt:.6f}")

# 비교: scipy의 linear_sum 또는 LP 솔버
from scipy.optimize import linprog
result_scipy = linprog(c, A_ub=A_ineq, b_ub=b_ineq, method='highs')
print(f"\nSciPy result: x = {result_scipy.x}")
print(f"SciPy objective: f(x*) = {result_scipy.fun:.6f}")

# 시각화 1: 중앙 경로
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) 중앙 경로 2D 시각화
ax = axes[0, 0]

# Feasible region 그리기
x_range = np.linspace(-0.5, 5, 200)
y_range = np.linspace(-0.5, 5, 200)
X, Y = np.meshgrid(x_range, y_range)

# 각 제약 확인
constraint1 = X + Y <= 4
constraint2 = 2*X - Y <= 2
constraint3 = X >= 0
constraint4 = Y >= 0

feasible = constraint1 & constraint2 & constraint3 & constraint4

ax.contourf(X, Y, feasible.astype(int), levels=[0.5, 1.5], colors=['lightgreen'], alpha=0.5)

# 제약선 그리기
x_line = np.linspace(-0.5, 5, 100)
ax.plot(x_line, 4 - x_line, 'b-', label='x1+x2=4', linewidth=2)
ax.plot(x_line, 2*x_line - 2, 'r-', label='2*x1-x2=2', linewidth=2)
ax.axvline(0, color='k', linestyle='-', linewidth=1)
ax.axhline(0, color='k', linestyle='-', linewidth=1)

# 중앙 경로 그리기
if len(x_traj) > 0:
    ax.plot(x_traj[:, 0], x_traj[:, 1], 'mo-', label='Central path', linewidth=2, markersize=5)
    ax.plot(x_opt[0], x_opt[1], 'r*', markersize=20, label='Optimal solution')

ax.set_xlim(-0.5, 3)
ax.set_ylim(-0.5, 3)
ax.set_xlabel('x1', fontsize=11)
ax.set_ylabel('x2', fontsize=11)
ax.set_title('Central Path Visualization', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (2) 목적함수값 수렴
ax = axes[0, 1]
ax.semilogy(np.abs(objectives - objectives[-1]) + 1e-15, 'b-o', linewidth=2, markersize=4)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('|f(x_k) - f*| (log)', fontsize=11)
ax.set_title('Objective Value Convergence', fontsize=12)
ax.grid(True, alpha=0.3)

# (3) Barrier 함수값
ax = axes[1, 0]
ax.semilogy(barriers, 'r-o', linewidth=2, markersize=4)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('Barrier value (log)', fontsize=11)
ax.set_title('Barrier Function Decay', fontsize=12)
ax.grid(True, alpha=0.3)

# (4) Duality gap (근사)
ax = axes[1, 1]
m = A_ineq.shape[0]
iteration_range = np.arange(len(barriers))
duality_gaps = m / (np.logspace(0, len(barriers)-1, len(barriers), base=10) * 1)
ax.semilogy(duality_gaps, 'g-o', linewidth=2, markersize=4)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('Duality gap m/t (log)', fontsize=11)
ax.set_title('Duality Gap Reduction', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ipm_visualization.png', dpi=150)
plt.show()

# 예제 2: 더 큰 문제
print("\n" + "=" * 60)
print("Larger LP Problem")
print("=" * 60)

# Randomly generated LP
np.random.seed(42)
n_vars = 10
n_constraints = 15

c_large = np.random.randn(n_vars)
A_large = np.random.randn(n_constraints, n_vars)
b_large = np.abs(np.random.randn(n_constraints)) + 1  # RHS > 0

# Add non-negativity constraints
A_large = np.vstack([A_large, -np.eye(n_vars)])
b_large = np.concatenate([b_large, np.zeros(n_vars)])

x_opt_large, _, barriers_large, objectives_large = barrier_method(
    c_large, A_large, b_large, verbose=False
)

print(f"Solution dimension: {len(x_opt_large)}")
print(f"Optimal objective: {c_large @ x_opt_large:.6f}")
print(f"IPM iterations: {len(barriers_large)}")

# 시각화 2: 큰 문제의 수렴
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(np.abs(objectives_large - objectives_large[-1]) + 1e-15, 'b-o', linewidth=2, markersize=5)
ax.set_xlabel('Iteration k', fontsize=11)
ax.set_ylabel('Optimality gap (log)', fontsize=11)
ax.set_title('IPM Convergence: Large-scale LP', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ipm_large_scale.png', dpi=150)
plt.show()
```

## 🔗 AI/ML 연결

**SVM 최적화**: SVM은 제약 최적화 문제로 표현되며, IPM이 효율적인 솔버입니다:
$$\min \frac{1}{2}\|w\|^2 + C\sum \xi_i \quad \text{s.t.} \quad y_i(w^T x_i + b) \ge 1 - \xi_i, \, \xi_i \ge 0$$

**Convex ML**: 정규화된 손실함수 $\min f(x) + r(x)$는 보조 변수로 제약 문제로 변환 가능합니다.

**LP의 다항 시간 해**: IPM은 ellipsoid method보다 실무적으로 빠르며, 현대 LP 솔버의 표준입니다.

## ⚖️ 가정과 한계

**가정 1**: strictly feasible 초기점이 필요합니다 (phase-I problem로 해결).

**가정 2**: 모든 함수가 볼록이어야 합니다. 비볼록 문제는 불가능합니다.

**한계**: 자기-일관성(self-concordance) 요구는 구조화된 문제에만 적용됩니다.

## 📌 핵심 정리

1. **Log barrier**: 제약을 부등식에서 가산항으로 변환.
2. **중앙 경로**: 파라미터 $t$로 제어되는 최적해 궤적.
3. **경로 추종**: $t$를 증가시키며 Newton으로 최적해 업데이트.
4. **다항 수렴**: $O(\sqrt{m}\log(m/\epsilon))$ Newton steps (LP).
5. **Self-concordance**: Newton 수렴 보장 기법.

## 🤔 생각해볼 문제

**문제 1**: Log barrier $\phi(x) = -\sum \log(-f_i(x))$에서 $f_i(x) \to 0^-$일 때 왜 $\phi(x) \to +\infty$인가?

<details>
<summary>힌트 및 해설</summary>

$f_i(x) = -\epsilon$ (작은 양수 $\epsilon$)이면:
$$\log(-f_i(x)) = \log(\epsilon) \to -\infty$$

따라서:
$$\phi(x) = -\sum \log(-f_i(x)) \to +\infty$$

경계에 접근하면 "벽"이 솟아오르므로, 알고리즘은 내부에 머물러 있으려 합니다.

</details>

**문제 2**: 중앙 경로가 실제로 원래 최적해로 수렴함을 보이시오 (정리 1의 핵심 아이디어).

<details>
<summary>힌트 및 해설</summary>

Barrier subproblem의 최적성 조건:
$$t \nabla f_0(x^*(t)) + \sum \frac{1}{-f_i(x^*(t))} \nabla f_i(x^*(t)) = 0$$

이를 다시 쓰면:
$$\nabla f_0(x^*(t)) + \sum \mu_i(t) \nabla f_i(x^*(t)) = 0$$

$t \to \infty$일 때, $\mu_i(t) = -1/(tf_i(x^*(t)))$는:
- $f_i(x^*(t)) < 0$이면 $\mu_i(t) > 0$
- 보강 조건: $\mu_i f_i = 0$이 되려면 $\mu_i = 0$ 또는 $f_i = 0$

따라서 $x^*(t) \to x^*$이고 최적성 조건을 만족합니다.

</details>

**문제 3**: IPM의 복잡도가 $O(\sqrt{m})$에 의존하는 이유는? Newton step의 수렴 반경과 관련이 있는가?

<details>
<summary>힌트 및 해설</summary>

Self-concordance 함수의 Newton 수렴은 Newton decrement $\lambda(x)$에 의존합니다.

각 $t_k$에서:
$$\lambda(x^*(t_k)) \le \sqrt{\text{(관련 dimension)}} = O(\sqrt{m})$$

따라서 각 barrier parameter에서 Newton이 수렴하려면 $O(\sqrt{m})$ 스텝이 필요합니다.

</details>

<div align="center">
| [◀ 04. Newton 방법](./04-newton-method.md) | [📚 README](../README.md) | [06. Stochastic 방법과 분산 감소 ▶](./06-stochastic-variance-reduction.md) |
</div>
