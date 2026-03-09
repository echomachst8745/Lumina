# Lumina

Lumina is a small C++23 numerical library with fixed-size vectors and matrices,
ODE solvers, and common direct and iterative linear algebra routines.

## Requirements

- C++23 compiler: MSVC 2022, GCC 13+, or Clang 16+
- CMake 3.16+
- Internet access when configuring tests, because Google Test is fetched automatically

## Building

From the project root:

```bash
cmake -S . -B build
cmake --build build --config Release
```

Use `--config Release` so the same commands work with MSVC, GCC, and Clang
generators.

## Tests

Tests are enabled by default:

```bash
ctest --test-dir build -C Release --output-on-failure
```

To configure without tests:

```bash
cmake -S . -B build -DLUMINA_BUILD_TESTS=OFF
cmake --build build --config Release
```

## Documentation

Doxygen must be installed and available on your `PATH`.

```bash
cmake -S . -B build -DLUMINA_BUILD_DOCS=ON
cmake --build build --config Release
```

HTML output is written to `build/docs/html/index.html`.

## CMake Options

Pass options at configure time with `-D`:

| Option               | Default | Description                                           |
|----------------------|---------|-------------------------------------------------------|
| `LUMINA_BUILD_TESTS` | `ON`    | Build and register the Google Test suite              |
| `LUMINA_BUILD_DOCS`  | `OFF`   | Generate Doxygen HTML docs, if Doxygen is on `PATH`   |

## Installing

Install into a local prefix:

```bash
cmake -S . -B build
cmake --build build --config Release
cmake --install build --config Release --prefix ./install
```

This creates:

```text
install/
  include/
    lumina/
      core/           vector.hpp, matrix.hpp
      linearalgebra/  gaussian_elimination.hpp, lu_decomposition.hpp, iterative_solvers.hpp
      ode/            euler.hpp, runge_kutta.hpp
      detail/         template implementations required at compile time
  lib/
    lumina.lib        or lumina.a
```

To include generated docs in the package, configure with `LUMINA_BUILD_DOCS=ON`.
The installed docs are written to `doc/html/index.html` inside the prefix.

To install somewhere else, change the prefix:

```bash
cmake --install build --config Release --prefix /usr/local
```

## API Overview

### Vector and Matrix

```cpp
#include <lumina/core/vector.hpp>
#include <lumina/core/matrix.hpp>

using namespace lumina;

Vector<double, 3> v{1.0, 2.0, 3.0};
double n = v.Norm();
double d = v.Dot(other);

Matrix<double, 3, 3> A = Matrix<double, 3, 3>::Identity();
Matrix<double, 3, 3> B = A * A;
A.SwapRows(0, 1);
```

### ODE Solvers

```cpp
#include <lumina/ode/euler.hpp>
#include <lumina/ode/runge_kutta.hpp>

using namespace lumina::ode;

using S = Vector<double, 2>;
auto f = [](double t, S y) -> S { return {y[1], -y[0]}; };

ODEResult<double, 2> result = RK4(f, 0.0, S{1.0, 0.0}, 0.01, 100);
```

Available ODE solvers:

```text
Euler(f, t0, y0, h, steps)
EulerImplicit(f, t0, y0, h, steps, max_iter, tol)
RK4(f, t0, y0, h, steps)
RK45(f, t0, y0, h0, t_end, a_delta, r_delta)
```

### Direct Linear Solvers

```cpp
#include <lumina/linearalgebra/gaussian_elimination.hpp>
#include <lumina/linearalgebra/lu_decomposition.hpp>

using namespace lumina::linearalgebra;

Matrix<double, 3, 3> A = ...;
Vector<double, 3> b = ...;

Vector<double, 3> x = GaussianSolve(A, b);
double det = Determinant(A);
auto Ainv = Inverse(A);

auto lu = LUDecompose(A);
auto x1 = LUSolve(lu, b1);
auto x2 = LUSolve(lu, b2);
double lu_det = LUDeterminant(lu);
```

### Iterative Linear Solvers

```cpp
#include <lumina/linearalgebra/iterative_solvers.hpp>

using namespace lumina::linearalgebra;

auto r1 = Jacobi(A, b, /*x0=*/{}, /*max_iter=*/1000, /*tol=*/1e-10);
auto r2 = GaussSeidel(A, b);
auto r3 = ConjugateGradient(A, b);

// r.converged, r.iterations, r.residual, r.x
```
