#include <gtest/gtest.h>
#include <cmath>

#include <lumina/linearalgebra/cholesky.hpp>
#include <lumina/linearalgebra/lu_decomposition.hpp>

using namespace lumina::core;
using namespace lumina::linearalgebra;

// Compute ||Ax - b||_inf
template<std::size_t N>
static double residual_inf(const Matrix<double, N, N>& A,
    const Vector<double, N>& x,
    const Vector<double, N>& b)
{
    auto Ax = A * x;
    double res = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        res = std::max(res, std::abs(Ax[i] - b[i]));
    return res;
}

// Tridiagonal SPD matrix: diagonal = N+1, off-diagonals = -1.
template<std::size_t N>
static Matrix<double, N, N> make_spd()
{
    Matrix<double, N, N> A;
    for (std::size_t i = 0; i < N; ++i)
    {
        A(i, i) = static_cast<double>(N + 1);
        if (i > 0) { A(i, i - 1) = -1.0; A(i - 1, i) = -1.0; }
    }
    return A;
}

// ---- CholeskyDecompose ----

TEST(CholeskyDecomposeTest, Simple3x3Reconstruction)
{
    // A = {{4, 2, 2}, {2, 5, 3}, {2, 3, 6}} is SPD.
    Matrix3d A{{4.0, 2.0, 2.0},
               {2.0, 5.0, 3.0},
               {2.0, 3.0, 6.0}};

    auto ch  = CholeskyDecompose(A);
    auto LLT = ch.L * ch.L.Transpose();

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(LLT(i, j), A(i, j), 1e-12);
}

TEST(CholeskyDecomposeTest, LIsLowerTriangular)
{
    Matrix3d A{{4.0, 2.0, 2.0},
               {2.0, 5.0, 3.0},
               {2.0, 3.0, 6.0}};

    auto ch = CholeskyDecompose(A);

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = i + 1; j < 3; ++j)
            EXPECT_DOUBLE_EQ(ch.L(i, j), 0.0);
}

TEST(CholeskyDecomposeTest, DiagonalEntries)
{
    // A = I is SPD; L should also be I.
    auto A  = Matrix3d::Identity();
    auto ch = CholeskyDecompose(A);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(ch.L(i, i), 1.0, 1e-12);
}

TEST(CholeskyDecomposeTest, TridiagonalSPD4x4)
{
    auto A   = make_spd<4>();
    auto ch  = CholeskyDecompose(A);
    auto LLT = ch.L * ch.L.Transpose();
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_NEAR(LLT(i, j), A(i, j), 1e-12);
}

TEST(CholeskyDecomposeTest, NotPositiveDefiniteThrows)
{
    // [[1, 2], [2, 1]] is not SPD.
    Matrix<double, 2, 2> A{{1.0, 2.0}, {2.0, 1.0}};
    EXPECT_THROW(CholeskyDecompose(A), std::runtime_error);
}

// ---- CholeskySolve (precomputed) ----

TEST(CholeskySolveTest, Simple3x3)
{
    Matrix3d A{{4.0, 2.0, 2.0},
               {2.0, 5.0, 3.0},
               {2.0, 3.0, 6.0}};
    Vec3d b{1.0, 2.0, 3.0};

    auto ch = CholeskyDecompose(A);
    auto x  = CholeskySolve(ch, b);

    EXPECT_LT(residual_inf(A, x, b), 1e-12);
}

TEST(CholeskySolveTest, IdentityMatrix)
{
    auto A = Matrix3d::Identity();
    Vec3d b{3.0, -1.0, 2.0};

    auto ch = CholeskyDecompose(A);
    auto x  = CholeskySolve(ch, b);

    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(x[i], b[i], 1e-12);
}

TEST(CholeskySolveTest, TridiagonalSPD8x8)
{
    auto A = make_spd<8>();
    Vector<double, 8> b;
    for (std::size_t i = 0; i < 8; ++i) b[i] = static_cast<double>(i + 1);

    auto ch = CholeskyDecompose(A);
    auto x  = CholeskySolve(ch, b);

    EXPECT_LT(residual_inf(A, x, b), 1e-10);
}

// ---- CholeskySolve (convenience overload) ----

TEST(CholeskySolveConvenienceTest, Simple3x3)
{
    Matrix3d A{{4.0, 2.0, 2.0},
               {2.0, 5.0, 3.0},
               {2.0, 3.0, 6.0}};
    Vec3d b{1.0, 2.0, 3.0};

    auto x = CholeskySolve(A, b);
    EXPECT_LT(residual_inf(A, x, b), 1e-12);
}

// ---- CholeskyDeterminant ----

TEST(CholeskyDeterminantTest, Identity3x3)
{
    auto ch = CholeskyDecompose(Matrix3d::Identity());
    EXPECT_NEAR(CholeskyDeterminant(ch), 1.0, 1e-12);
}

TEST(CholeskyDeterminantTest, Known2x2)
{
    // A = {{4, 2}, {2, 3}}, det = 4*3 - 2*2 = 8.
    Matrix<double, 2, 2> A{{4.0, 2.0}, {2.0, 3.0}};
    auto ch = CholeskyDecompose(A);
    EXPECT_NEAR(CholeskyDeterminant(ch), 8.0, 1e-12);
}

TEST(CholeskyDeterminantTest, MatchesLUDeterminant)
{
    Matrix3d A{{4.0, 2.0, 2.0},
               {2.0, 5.0, 3.0},
               {2.0, 3.0, 6.0}};

    auto chDet = CholeskyDeterminant(CholeskyDecompose(A));
    auto luDet = LUDeterminant(LUDecompose(A));

    EXPECT_NEAR(chDet, luDet, 1e-10);
}
