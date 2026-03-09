#include <gtest/gtest.h>
#include <lumina/core/matrix.hpp>
#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>

using namespace lumina::core;

using M22d = Matrix<double, 2, 2>;
using M23d = Matrix<double, 2, 3>;
using M32d = Matrix<double, 3, 2>;
using M33d = Matrix<double, 3, 3>;
using M24d = Matrix<double, 2, 4>;

TEST(MatrixTest, DefaultConstruction) {
    M33d m;
    EXPECT_EQ(M33d::Rows(), 3u);
    EXPECT_EQ(M33d::Cols(), 3u);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_EQ(m(i, j), 0.0);
}

TEST(MatrixTest, FillConstruction) {
    M23d m(5.0);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_EQ(m(i, j), 5.0);
}

TEST(MatrixTest, Identity) {
    auto I = M33d::Identity();
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_EQ(I(i, j), (i == j) ? 1.0 : 0.0);
}

TEST(MatrixTest, Zeros) {
    auto Z = M24d::Zeros();
    EXPECT_EQ(M24d::Rows(), 2u);
    EXPECT_EQ(M24d::Cols(), 4u);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_EQ(Z(i, j), 0.0);
}

TEST(MatrixTest, Ones) {
    auto O = M23d::Ones();
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_EQ(O(i, j), 1.0);
}

TEST(MatrixTest, InitializerList2D) {
    M22d m{{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_EQ(m(0, 0), 1.0);
    EXPECT_EQ(m(0, 1), 2.0);
    EXPECT_EQ(m(1, 0), 3.0);
    EXPECT_EQ(m(1, 1), 4.0);
}

TEST(MatrixTest, Addition) {
    M22d A{{1.0, 2.0}, {3.0, 4.0}};
    M22d B{{5.0, 6.0}, {7.0, 8.0}};
    M22d C = A + B;
    EXPECT_EQ(C(0, 0),  6.0);
    EXPECT_EQ(C(0, 1),  8.0);
    EXPECT_EQ(C(1, 0), 10.0);
    EXPECT_EQ(C(1, 1), 12.0);
}

TEST(MatrixTest, Subtraction) {
    M22d A{{5.0, 6.0}, {7.0, 8.0}};
    M22d B{{1.0, 2.0}, {3.0, 4.0}};
    M22d C = A - B;
    EXPECT_EQ(C(0, 0), 4.0);
    EXPECT_EQ(C(0, 1), 4.0);
    EXPECT_EQ(C(1, 0), 4.0);
    EXPECT_EQ(C(1, 1), 4.0);
}

TEST(MatrixTest, ScalarMultiply) {
    M22d A{{1.0, 2.0}, {3.0, 4.0}};
    M22d B = A * 2.0;
    EXPECT_EQ(B(0, 0), 2.0);
    EXPECT_EQ(B(0, 1), 4.0);
    EXPECT_EQ(B(1, 0), 6.0);
    EXPECT_EQ(B(1, 1), 8.0);
}

TEST(MatrixTest, CompoundAdd) {
    M22d A{{1.0, 2.0}, {3.0, 4.0}};
    M22d B{{1.0, 0.0}, {0.0, 1.0}};
    A += B;
    EXPECT_EQ(A(0, 0), 2.0);
    EXPECT_EQ(A(1, 1), 5.0);
}

TEST(MatrixTest, MatMul2x2) {
    M22d A{{1.0, 2.0}, {3.0, 4.0}};
    M22d B{{5.0, 6.0}, {7.0, 8.0}};
    M22d C = A * B;
    EXPECT_EQ(C(0, 0), 19.0);
    EXPECT_EQ(C(0, 1), 22.0);
    EXPECT_EQ(C(1, 0), 43.0);
    EXPECT_EQ(C(1, 1), 50.0);
}

TEST(MatrixTest, MatMulIdentity) {
    M22d A{{1.0, 2.0}, {3.0, 4.0}};
    M22d C = A * M22d::Identity();
    EXPECT_EQ(C(0, 0), A(0, 0));
    EXPECT_EQ(C(0, 1), A(0, 1));
    EXPECT_EQ(C(1, 0), A(1, 0));
    EXPECT_EQ(C(1, 1), A(1, 1));
}

TEST(MatrixTest, MatVec) {
    M22d A{{1.0, 2.0}, {3.0, 4.0}};
    Vec2d v{1.0, 1.0};
    Vec2d w = A * v;
    EXPECT_EQ(w[0], 3.0);
    EXPECT_EQ(w[1], 7.0);
}

TEST(MatrixTest, MatVecNonSquare) {
    M23d A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    Vec3d v{1.0, 0.0, 0.0};
    Vec2d w = A * v;
    EXPECT_EQ(w[0], 1.0);
    EXPECT_EQ(w[1], 4.0);
}

TEST(MatrixTest, Transpose) {
    M23d A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    M32d At = A.Transpose();
    EXPECT_EQ(M32d::Rows(), 3u);
    EXPECT_EQ(M32d::Cols(), 2u);
    EXPECT_EQ(At(0, 0), 1.0);
    EXPECT_EQ(At(1, 0), 2.0);
    EXPECT_EQ(At(2, 0), 3.0);
    EXPECT_EQ(At(0, 1), 4.0);
    EXPECT_EQ(At(2, 1), 6.0);
}

TEST(MatrixTest, TransposeSquare) {
    M22d A{{1.0, 2.0}, {3.0, 4.0}};
    auto At = A.Transpose();
    EXPECT_EQ(At(0, 1), 3.0);
    EXPECT_EQ(At(1, 0), 2.0);
}

TEST(MatrixTest, Trace) {
    M22d A{{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_DOUBLE_EQ(A.Trace(), 5.0);
}

TEST(MatrixTest, FrobeniusNorm) {
    // ||A||_F = sqrt(3^2 + 4^2) = 5
    M22d A{{3.0, 0.0}, {4.0, 0.0}};
    EXPECT_DOUBLE_EQ(A.FrobeniusNorm(), 5.0);
}

TEST(MatrixTest, IsSquare) {
    EXPECT_TRUE(M33d::IsSquare());
    EXPECT_FALSE(M23d::IsSquare());
}

TEST(MatrixTest, IsSymmetric) {
    M22d A{{1.0, 2.0}, {2.0, 3.0}};
    EXPECT_TRUE(A.IsSymmetric());
    M22d B{{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_FALSE(B.IsSymmetric());
}

TEST(MatrixTest, RowAccess) {
    M23d A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    auto r = A.Row(0);
    EXPECT_EQ(r[0], 1.0);
    EXPECT_EQ(r[1], 2.0);
    EXPECT_EQ(r[2], 3.0);
    auto r2 = A.Row(1);
    EXPECT_EQ(r2[0], 4.0);
}

TEST(MatrixTest, ColAccess) {
    M23d A{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    auto c = A.Col(1);
    EXPECT_EQ(c[0], 2.0);
    EXPECT_EQ(c[1], 5.0);
}

TEST(MatrixTest, AtOutOfBounds) {
    M22d A;
    EXPECT_THROW(A.at(2, 0), std::out_of_range);
    EXPECT_THROW(A.at(0, 2), std::out_of_range);
}

TEST(MatrixTest, Equality) {
    M22d A{{1.0, 2.0}, {3.0, 4.0}};
    M22d B{{1.0, 2.0}, {3.0, 4.0}};
    M22d C{{1.0, 2.0}, {3.0, 5.0}};
    EXPECT_EQ(A, B);
    EXPECT_NE(A, C);
}

TEST(MatrixTest, FlatInitializerList) {
    M23d m{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    EXPECT_EQ(m(0, 0), 1.0); EXPECT_EQ(m(0, 1), 2.0); EXPECT_EQ(m(0, 2), 3.0);
    EXPECT_EQ(m(1, 0), 4.0); EXPECT_EQ(m(1, 1), 5.0); EXPECT_EQ(m(1, 2), 6.0);
}

TEST(MatrixTest, SetRow) {
    M23d m{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    m.SetRow(0, std::array<double, 3>{10.0, 20.0, 30.0});
    EXPECT_EQ(m(0, 0), 10.0); EXPECT_EQ(m(0, 1), 20.0); EXPECT_EQ(m(0, 2), 30.0);
    EXPECT_EQ(m(1, 0), 4.0); // Row 1 unchanged.
}

TEST(MatrixTest, SetCol) {
    M22d m{{1.0, 2.0}, {3.0, 4.0}};
    m.SetCol(1, std::array<double, 2>{7.0, 8.0});
    EXPECT_EQ(m(0, 1), 7.0); EXPECT_EQ(m(1, 1), 8.0);
    EXPECT_EQ(m(0, 0), 1.0); EXPECT_EQ(m(1, 0), 3.0); // Col 0 unchanged.
}

TEST(MatrixTest, SwapRows) {
    M22d m{{1.0, 2.0}, {3.0, 4.0}};
    m.SwapRows(0, 1);
    EXPECT_EQ(m(0, 0), 3.0); EXPECT_EQ(m(0, 1), 4.0);
    EXPECT_EQ(m(1, 0), 1.0); EXPECT_EQ(m(1, 1), 2.0);
}

TEST(MatrixTest, SwapCols) {
    M22d m{{1.0, 2.0}, {3.0, 4.0}};
    m.SwapCols(0, 1);
    EXPECT_EQ(m(0, 0), 2.0); EXPECT_EQ(m(0, 1), 1.0);
    EXPECT_EQ(m(1, 0), 4.0); EXPECT_EQ(m(1, 1), 3.0);
}

TEST(MatrixTest, ScalarDivide) {
    M22d B = M22d{{4.0, 8.0}, {12.0, 16.0}} / 4.0;
    EXPECT_DOUBLE_EQ(B(0, 0), 1.0); EXPECT_DOUBLE_EQ(B(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(B(1, 0), 3.0); EXPECT_DOUBLE_EQ(B(1, 1), 4.0);
}

TEST(MatrixTest, CompoundSubtract) {
    M22d A{{5.0, 6.0}, {7.0, 8.0}};
    A -= M22d{{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_DOUBLE_EQ(A(0, 0), 4.0); EXPECT_DOUBLE_EQ(A(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(A(1, 0), 4.0); EXPECT_DOUBLE_EQ(A(1, 1), 4.0);
}

TEST(MatrixTest, CompoundScalarMul) {
    M22d A{{1.0, 2.0}, {3.0, 4.0}};
    A *= 3.0;
    EXPECT_DOUBLE_EQ(A(0, 0), 3.0); EXPECT_DOUBLE_EQ(A(0, 1), 6.0);
    EXPECT_DOUBLE_EQ(A(1, 0), 9.0); EXPECT_DOUBLE_EQ(A(1, 1), 12.0);
}

TEST(MatrixTest, CompoundScalarDiv) {
    M22d A{{2.0, 4.0}, {6.0, 8.0}};
    A /= 2.0;
    EXPECT_DOUBLE_EQ(A(0, 0), 1.0); EXPECT_DOUBLE_EQ(A(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(A(1, 0), 3.0); EXPECT_DOUBLE_EQ(A(1, 1), 4.0);
}

TEST(MatrixTest, NamedAdd) {
    M22d C = M22d{{1.0, 2.0}, {3.0, 4.0}}.Add(M22d{{5.0, 6.0}, {7.0, 8.0}});
    EXPECT_DOUBLE_EQ(C(0, 0), 6.0); EXPECT_DOUBLE_EQ(C(1, 1), 12.0);
}

TEST(MatrixTest, NamedSubtract) {
    M22d C = M22d{{5.0, 6.0}, {7.0, 8.0}}.Subtract(M22d{{1.0, 2.0}, {3.0, 4.0}});
    EXPECT_DOUBLE_EQ(C(0, 0), 4.0); EXPECT_DOUBLE_EQ(C(1, 1), 4.0);
}

TEST(MatrixTest, NamedScalarMultiply) {
    M22d B = M22d{{1.0, 2.0}, {3.0, 4.0}}.Multiply(2.0);
    EXPECT_DOUBLE_EQ(B(0, 0), 2.0); EXPECT_DOUBLE_EQ(B(1, 1), 8.0);
}

TEST(MatrixTest, NamedScalarDivide) {
    M22d B = M22d{{4.0, 8.0}, {12.0, 16.0}}.Divide(4.0);
    EXPECT_DOUBLE_EQ(B(0, 0), 1.0); EXPECT_DOUBLE_EQ(B(1, 1), 4.0);
}

TEST(MatrixTest, EqualsMethod) {
    M22d A{{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_TRUE(A.Equals(M22d{{1.0, 2.0}, {3.0, 4.0}}));
    EXPECT_FALSE(A.Equals(M22d{{1.0, 2.0}, {3.0, 5.0}}));
}

TEST(MatrixTest, IsSymmetricCustomDelta) {
    M22d A{{1.0, 2.0}, {2.000001, 3.0}};
    EXPECT_FALSE(A.IsSymmetric());
    EXPECT_TRUE(A.IsSymmetric(1e-5));
}

TEST(MatrixTest, ToStringFormat) {
    M22d A{{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_EQ(A.ToString(), "[[1, 2], [3, 4]]");
}

TEST(MatrixTest, StreamInsertionOperator) {
    M22d A{{1.0, 2.0}, {3.0, 4.0}};
    std::ostringstream oss;
    oss << A;
    EXPECT_EQ(oss.str(), "[[1, 2], [3, 4]]");
}
