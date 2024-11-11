# Calculation Verification for Factor Analysis

The purpose of this simulation is to verify the calculation of the following paper:
[\[link\]](https://doi.org/10.1021/ac00255a011)

## Calculation

Let $D$ be the mixture spectra matrix, with:

```python
D = np.array([
    [0.26, 0.22, 0.14],
    [0.20, 0.40, 0.80],
    [1.60, 1.20, 0.40],
    [0.12, 0.14, 0.18]
])
```

So, the covariance or correlation matrix $C$ is calculated as follows:

```python
C = np.matmul(D.T, D)
```

The eigenvalues and eigenvectors of $C$ are calculated as follows:

```python
eigenvalues, eigenvectors = np.linalg.eig(C)
```

By looking at this published paper from the same author [\[link\]](https://doi.org/10.1366/0003702824639394):

> In the absence of noise, the number of pure components is simply equal to the number of
nonzero eigenvalues. Ordinarily, however, one is confronted with the situation in which several of the eigenvalues are much larger than those remaining in the solution, and the number of pure components can be taken as the number of "large" eigenvalues. 

We can set the threshold of pure components as the eigenvalues that are larger than $ 10^{-8}$

```python
pure_eigenvalues = eigenvalues[eigenvalues > 1e-8]
pure_eigenvalues = np.append(pure_eigenvalues, 0)

pure_eigenvectors = eigenvectors[:, eigenvalues > 1e-8]
pure_eigenvectors = np.append(pure_eigenvectors, np.array([[0, 0, 0]]).T, axis=1)
```

where `pure_eigenvectors` is the same as real eigenvector matrix $E'$.

The abstract eigenspectra can be calculated as follows:
$$
A = D \cdot E'
$$

```python
A = np.matmul(D, pure_eigenvectors)
```
where:
$$
A = \begin{bmatrix}
    A_1 & A_2
\end{bmatrix}
$$

The ratio of the first and second components of $A$ and the cross product of the first and second components of $A$ are calculated as follows:

```python
ratio_of_A1_and_A2 = A[:, 0] / A[:, 1]
cross_A1_and_A2 = A[:, 0] * A[:, 1]
```

$$
\text{ratio(A1, A2)} = \frac{A_1}{A_2} \\
= [11.15907773,  0.93848173, -8.95237007,  1.85375695]
$$

and

$$
\text{cross(A1, A2)} = A_1 \times A_2 \\
= [ 0.01205474,  0.41915486, -0.45895485,  0.02774526]
$$

The line passing through the real eigenvector matrix $E_1$ and $E_2$ is calculated as follows:

```python
# Line passing through eigenvector 1 and 2
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(E_acc[:, 0].reshape(-1,1), E_acc[:, 1])
```

where:
$$
E_2 = \text{reg.coef\_} \cdot E_1 + \text{reg.intercept\_} \\
 = -2.956 E_1 + 1.802
$$

By looking at the value of ratio and cross product, we can determine the concentration line that bounds the concentration of the mixture. In this example, both the ratio and cross product indicate that the ratios corresponding to the second and third elements should be used to bound the concentration line.

The transformation matrix $T$ is obtained by solving:

- $E_2 = -2.956 E_1 + 1.801$
- $E_2 = 8.952 E_1$
- $E_2 = -0.938 E_1$

$$
E_2 = \begin{bmatrix}
    -2.956 \\
    8.952 \\
    -0.938 \\
\end{bmatrix} E_1 +
\begin{bmatrix}
    1.801 \\
    0 \\
    0 \\
\end{bmatrix}
$$

$$
\begin{bmatrix}
    2.956 & 1 \\
    -8.952 & 1 \\
    0.938 & 1 \\
\end{bmatrix}

\begin{bmatrix}
    E_1 \\
    E_2 \\
\end{bmatrix} =

\begin{bmatrix}
    1.801 \\
    0 \\
    0 \\
\end{bmatrix}
$$

to solve it, we can use the following code:

```python
A_T_1 = np.array([
    [2.956, 1],
    [-8.952, 1],
])
b_T_1 = np.array([1.801, 0])

A_T_2 = np.array([
    [2.956, 1],
    [0.938, 1],
])
b_T_2 = np.array([1.801, 0])

T_1 = np.linalg.solve(A_T_1, b_T_1)
T_2 = np.linalg.solve(A_T_2, b_T_2)

T = np.concatenate((T_1.reshape(2, 1), T_2.reshape(2, 1)), axis=1)
```
The pure component spectra matrix $P$ is calculated as follows:

```python
P = np.matmul(A, T)
```

and the concentration of the pure components is calculated as follows:

```python
K = np.matmul(np.linalg.inv(T), E_acc.T)
```