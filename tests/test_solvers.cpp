#include <gtest/gtest.h>
#include <cmath>

#include <lumina/linearalgebra/gaussian_elimination.hpp>
#include <lumina/linearalgebra/iterative_solvers.hpp>
#include <lumina/linearalgebra/lu_decomposition.hpp>

using namespace lumina::core;
using namespace lumina::linearalgebra;

// Tridiagonal diagonally-dominant SPD matrix of compile-time size N.
// Diagonal = N+1, off-diagonals = -1.
template<std::size_t N>
static Matrix<double, N, N> make_spd() {
    Matrix<double, N, N> A;
    for (std::size_t i = 0; i < N; ++i) {
        A(i, i) = static_cast<double>(N + 1);
        if (i > 0) {
            A(i, i - 1) = -1.0;
            A(i - 1, i) = -1.0;
        }
    }
    return A;
}

// Compute ||Ax - b||_inf
template<std::size_t N>
static double residual_inf(const Matrix<double, N, N>& A,
    const Vector<double, N>& x,
    const Vector<double, N>& b) {
    auto Ax = A * x;
    double res = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        res = std::max(res, std::abs(Ax[i] - b[i]));
    return res;
}

// ---- Gaussian Elimination ----

TEST(GaussianEliminationTest, Simple2x2) {
    // 2x + y = 11,  5x + 7y = 13  =>  x = 64/9, y = -29/9
    Matrix<double, 2, 2> A{ {2.0, 1.0}, {5.0, 7.0} };
    Vector<double, 2> b{ 11.0, 13.0 };
    auto x = GaussianSolve(A, b);
    EXPECT_NEAR(x[0], 64.0 / 9.0, 1e-10);
    EXPECT_NEAR(x[1], -29.0 / 9.0, 1e-10);
}

TEST(GaussianEliminationTest, Identity) {
    auto I = Matrix<double, 3, 3>::Identity();
    Vector<double, 3> b{ 1.0, 2.0, 3.0 };
    auto x = GaussianSolve(I, b);
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 2.0, 1e-10);
    EXPECT_NEAR(x[2], 3.0, 1e-10);
}

TEST(GaussianEliminationTest, Residual3x3) {
    Matrix<double, 3, 3> A{ {4.0, 1.0, -1.0},
                            {2.0, 7.0,  1.0},
                            {1.0, 1.0,  3.0} };
    Vector<double, 3> b{ 3.0, 19.0, 10.0 };
    auto x = GaussianSolve(A, b);
    EXPECT_LT(residual_inf(A, x, b), 1e-10);
}

TEST(GaussianEliminationTest, PivotingRequired) {
    Matrix<double, 2, 2> A{ {0.0, 1.0}, {2.0, 3.0} };
    Vector<double, 2> b{ 1.0, 8.0 };
    auto x = GaussianSolve(A, b);
    EXPECT_LT(residual_inf(A, x, b), 1e-10);
}

TEST(GaussianEliminationTest, Determinant2x2) {
    Matrix<double, 2, 2> A{ {1.0, 2.0}, {3.0, 4.0} };
    EXPECT_NEAR(Determinant(A), -2.0, 1e-10);
}

TEST(GaussianEliminationTest, DeterminantIdentity) {
    auto I = Matrix<double, 4, 4>::Identity();
    EXPECT_NEAR(Determinant(I), 1.0, 1e-10);
}

TEST(GaussianEliminationTest, Inverse2x2) {
    Matrix<double, 2, 2> A{ {1.0, 2.0}, {3.0, 4.0} };
    auto Ainv = Inverse(A);
    auto prod = A * Ainv;
    EXPECT_NEAR(prod(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(prod(0, 1), 0.0, 1e-10);
    EXPECT_NEAR(prod(1, 0), 0.0, 1e-10);
    EXPECT_NEAR(prod(1, 1), 1.0, 1e-10);
}

TEST(GaussianEliminationTest, Inverse3x3) {
    Matrix<double, 3, 3> A{ {2.0, 1.0, 0.0},
                            {1.0, 3.0, 1.0},
                            {0.0, 1.0, 2.0} };
    auto Ainv = Inverse(A);
    auto I = A * Ainv;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(I(i, j), (i == j) ? 1.0 : 0.0, 1e-10);
}

// ---- Iterative Solvers ----

TEST(JacobiTest, SmallSystem) {
    auto A = make_spd<5>();
    Vector<double, 5> b(1.0);
    auto result = Jacobi(A, b, {}, 5000, 1e-12);
    EXPECT_TRUE(result.converged);
    EXPECT_LT(residual_inf(A, result.x, b), 1e-8);
}

TEST(JacobiTest, IterationCount) {
    auto A = make_spd<3>();
    Vector<double, 3> b(1.0);
    auto result = Jacobi(A, b, {}, 5000, 1e-12);
    EXPECT_GT(result.iterations, 0);
    EXPECT_LE(result.iterations, 5000);
}

TEST(GaussSeidelTest, SmallSystem) {
    auto A = make_spd<5>();
    Vector<double, 5> b(1.0);
    auto result = GaussSeidel(A, b, {}, 1000, 1e-12);
    EXPECT_TRUE(result.converged);
    EXPECT_LT(residual_inf(A, result.x, b), 1e-8);
}

TEST(GaussSeidelTest, FasterThanJacobi) {
    auto A = make_spd<6>();
    Vector<double, 6> b(1.0);
    auto jacobi_r = Jacobi(A, b, {}, 5000, 1e-10);
    auto gs_r     = GaussSeidel(A, b, {}, 5000, 1e-10);
    EXPECT_TRUE(jacobi_r.converged);
    EXPECT_TRUE(gs_r.converged);
    EXPECT_LE(gs_r.iterations, jacobi_r.iterations);
}

TEST(CGTest, SmallSystem) {
    auto A = make_spd<10>();
    Vector<double, 10> b(1.0);
    auto result = ConjugateGradient(A, b, {}, 100, 1e-12);
    EXPECT_TRUE(result.converged);
    EXPECT_LT(residual_inf(A, result.x, b), 1e-8);
}

TEST(CGTest, ConvergesInAtMostNSteps) {
    auto A = make_spd<8>();
    Vector<double, 8> b(1.0);
    auto result = ConjugateGradient(A, b, {}, 16, 1e-10);
    EXPECT_TRUE(result.converged);
}

TEST(CGTest, ZeroRHS) {
    auto A = make_spd<4>();
    Vector<double, 4> b{};
    auto result = ConjugateGradient(A, b, {}, 100, 1e-12);
    EXPECT_TRUE(result.converged);
}

// ---- LU Decomposition ----

TEST(LUTest, Solve3x3) {
    Matrix<double, 3, 3> A{{2.0, 1.0, 1.0},
                            {4.0, 3.0, 3.0},
                            {8.0, 7.0, 9.0}};
    Vector<double, 3> b{1.0, 1.0, 1.0};
    auto lu = LUDecompose(A);
    auto x  = LUSolve(lu, b);
    EXPECT_LT(residual_inf(A, x, b), 1e-10);
}

TEST(LUTest, SolveIdentity) {
    auto I = Matrix<double, 3, 3>::Identity();
    Vector<double, 3> b{4.0, 5.0, 6.0};
    auto x = LUSolve(I, b);
    EXPECT_NEAR(x[0], 4.0, 1e-10);
    EXPECT_NEAR(x[1], 5.0, 1e-10);
    EXPECT_NEAR(x[2], 6.0, 1e-10);
}

TEST(LUTest, SolveConvenience) {
    Matrix<double, 2, 2> A{{3.0, 1.0}, {1.0, 2.0}};
    Vector<double, 2> b{5.0, 5.0};
    auto x = LUSolve(A, b);
    EXPECT_LT(residual_inf(A, x, b), 1e-10);
}

TEST(LUTest, Determinant2x2) {
    Matrix<double, 2, 2> A{{1.0, 2.0}, {3.0, 4.0}};
    auto lu = LUDecompose(A);
    EXPECT_NEAR(LUDeterminant(lu), -2.0, 1e-10);
}

TEST(LUTest, DeterminantIdentity) {
    auto lu = LUDecompose(Matrix<double, 4, 4>::Identity());
    EXPECT_NEAR(LUDeterminant(lu), 1.0, 1e-10);
}

TEST(LUTest, MultipleRHS) {
    Matrix<double, 2, 2> A{{4.0, 1.0}, {2.0, 3.0}};
    auto lu = LUDecompose(A);
    Vector<double, 2> b1{5.0, 5.0};
    Vector<double, 2> b2{1.0, 0.0};
    auto x1 = LUSolve(lu, b1);
    auto x2 = LUSolve(lu, b2);
    EXPECT_LT(residual_inf(A, x1, b1), 1e-10);
    EXPECT_LT(residual_inf(A, x2, b2), 1e-10);
}