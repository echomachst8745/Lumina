#include <gtest/gtest.h>
#include <cmath>

#include <lumina/linearalgebra/eigen_solvers.hpp>

using namespace lumina::core;
using namespace lumina::linearalgebra;

// diag(1, 2, 5): dominant eigenvalue = 5.
TEST(PowerIterationTest, KnownEigenvalue) {
    Matrix<double, 3, 3> A;
    A(0, 0) = 1.0; A(1, 1) = 2.0; A(2, 2) = 5.0;
    auto result = PowerIteration(A);
    EXPECT_NEAR(result.eigenvalue, 5.0, 1e-8);
}

// Identity: every vector is an eigenvector, eigenvalue = 1.
TEST(PowerIterationTest, IdentityMatrix) {
    auto result = PowerIteration(Matrix<double, 3, 3>::Identity());
    EXPECT_NEAR(result.eigenvalue, 1.0, 1e-8);
    EXPECT_TRUE(result.converged);
}

// Well-conditioned diagonal matrix: converged flag must be true.
TEST(PowerIterationTest, ConvergesFlag) {
    Matrix<double, 3, 3> A;
    A(0, 0) = 3.0; A(1, 1) = 1.0; A(2, 2) = 0.5;
    auto result = PowerIteration(A, {}, 1000, 1e-10);
    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.iterations, 0u);
    EXPECT_LT(result.iterations, 1000u);
}

// Returned eigenvector must always have unit Euclidean norm.
TEST(PowerIterationTest, EigenvectorIsUnit) {
    Matrix<double, 3, 3> A;
    A(0, 0) = 4.0; A(1, 1) = 2.0; A(2, 2) = 1.0;
    auto result = PowerIteration(A);
    EXPECT_NEAR(result.eigenvector.Norm(), 1.0, 1e-12);
}

// Av = lambda*v must hold within the accuracy of the converged eigenvector.
// Tolerance 1e-4 is appropriate: eigenvalue convergence at 1e-12 leaves
// off-diagonal eigenvector residuals ~(r2/r1)^k that satisfy ||Av - lv||~1e-5.
TEST(PowerIterationTest, VerifyAvEqualsLambdaV) {
    Matrix<double, 3, 3> A;
    A(0, 0) = 6.0; A(1, 1) = 2.0; A(2, 2) = 1.0;
    auto result = PowerIteration(A, {}, 1000, 1e-12);
    auto Av       = A * result.eigenvector;
    auto lambda_v = result.eigenvector * result.eigenvalue;
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(Av[i], lambda_v[i], 1e-4);
}

// max_iter=1 forces early exit: must not crash, converged=false, eigenvector still unit.
TEST(PowerIterationTest, NonConvergedDoesNotCrash) {
    Matrix<double, 3, 3> A;
    A(0, 0) = 5.0; A(1, 1) = 2.0; A(2, 2) = 1.0;
    auto result = PowerIteration(A, {}, 1, 1e-12);
    EXPECT_FALSE(result.converged);
    EXPECT_NEAR(result.eigenvector.Norm(), 1.0, 1e-12);
}
