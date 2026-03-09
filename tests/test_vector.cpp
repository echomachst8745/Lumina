#include <gtest/gtest.h>
#include <lumina/core/vector.hpp>
#include <cmath>
#include <sstream>
#include <stdexcept>

using namespace lumina::core;

TEST(VectorTest, DefaultConstructorIsZero) {
    Vec3d v;
    EXPECT_EQ(v[0], 0.0);
    EXPECT_EQ(v[1], 0.0);
    EXPECT_EQ(v[2], 0.0);
}

TEST(VectorTest, InitializerListConstruction) {
    Vec3d v{1.0, 2.0, 3.0};
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 2.0);
    EXPECT_EQ(v[2], 3.0);
}

TEST(VectorTest, FillConstruction) {
    Vector<double, 5> v(7.0);
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_EQ(v[i], 7.0);
}

TEST(VectorTest, Size) {
    Vec3d v;
    EXPECT_EQ(v.size(), 3u);
}

TEST(VectorTest, Addition) {
    Vec3d a{1.0, 2.0, 3.0};
    Vec3d b{4.0, 5.0, 6.0};
    Vec3d c = a + b;
    EXPECT_EQ(c[0], 5.0);
    EXPECT_EQ(c[1], 7.0);
    EXPECT_EQ(c[2], 9.0);
}

TEST(VectorTest, Subtraction) {
    Vec3d a{4.0, 5.0, 6.0};
    Vec3d b{1.0, 2.0, 3.0};
    Vec3d c = a - b;
    EXPECT_EQ(c[0], 3.0);
    EXPECT_EQ(c[1], 3.0);
    EXPECT_EQ(c[2], 3.0);
}

TEST(VectorTest, Negation) {
    Vec3d a{1.0, -2.0, 3.0};
    Vec3d b = -a;
    EXPECT_EQ(b[0], -1.0);
    EXPECT_EQ(b[1],  2.0);
    EXPECT_EQ(b[2], -3.0);
}

TEST(VectorTest, ScalarMultiplyRight) {
    Vec3d v{1.0, 2.0, 3.0};
    Vec3d w = v * 2.0;
    EXPECT_EQ(w[0], 2.0);
    EXPECT_EQ(w[1], 4.0);
    EXPECT_EQ(w[2], 6.0);
}


TEST(VectorTest, ScalarDivide) {
    Vec3d v{2.0, 4.0, 6.0};
    Vec3d w = v / 2.0;
    EXPECT_EQ(w[0], 1.0);
    EXPECT_EQ(w[1], 2.0);
    EXPECT_EQ(w[2], 3.0);
}

TEST(VectorTest, CompoundAdd) {
    Vec3d a{1.0, 2.0, 3.0};
    a += Vec3d{1.0, 1.0, 1.0};
    EXPECT_EQ(a[0], 2.0);
    EXPECT_EQ(a[1], 3.0);
    EXPECT_EQ(a[2], 4.0);
}

TEST(VectorTest, CompoundSubtract) {
    Vec3d a{3.0, 3.0, 3.0};
    a -= Vec3d{1.0, 1.0, 1.0};
    EXPECT_EQ(a[0], 2.0);
    EXPECT_EQ(a[1], 2.0);
    EXPECT_EQ(a[2], 2.0);
}

TEST(VectorTest, CompoundScalarMul) {
    Vec3d a{1.0, 2.0, 3.0};
    a *= 2.0;
    EXPECT_EQ(a[0], 2.0);
    EXPECT_EQ(a[1], 4.0);
    EXPECT_EQ(a[2], 6.0);
}

TEST(VectorTest, DotProduct) {
    Vec3d a{1.0, 2.0, 3.0};
    Vec3d b{4.0, 5.0, 6.0};
    EXPECT_DOUBLE_EQ(a.Dot(b), 32.0);
}

TEST(VectorTest, DotProductOrthogonal) {
    Vec3d a{1.0, 0.0, 0.0};
    Vec3d b{0.0, 1.0, 0.0};
    EXPECT_DOUBLE_EQ(a.Dot(b), 0.0);
}

TEST(VectorTest, Norm) {
    Vec3d v{3.0, 4.0, 0.0};
    EXPECT_DOUBLE_EQ(v.Norm(), 5.0);
}

TEST(VectorTest, NormSquared) {
    Vec3d v{1.0, 2.0, 2.0};
    EXPECT_DOUBLE_EQ(v.NormSquared(), 9.0);
}

TEST(VectorTest, NormL1) {
    Vec3d v{-1.0, 2.0, -3.0};
    EXPECT_DOUBLE_EQ(v.NormL1(), 6.0);
}

TEST(VectorTest, NormInf) {
    Vec3d v{-1.0, 5.0, -3.0};
    EXPECT_DOUBLE_EQ(v.NormLInf(), 5.0);
}

TEST(VectorTest, Normalized) {
    Vec3d v{3.0, 0.0, 0.0};
    Vec3d n = v.Normalised();
    EXPECT_DOUBLE_EQ(n[0], 1.0);
    EXPECT_DOUBLE_EQ(n[1], 0.0);
    EXPECT_DOUBLE_EQ(n[2], 0.0);
    EXPECT_NEAR(n.Norm(), 1.0, 1e-15);
}

TEST(VectorTest, NormalizedArbitrary) {
    Vec3d v{1.0, 2.0, 3.0};
    Vec3d n = v.Normalised();
    EXPECT_NEAR(n.Norm(), 1.0, 1e-15);
}

TEST(VectorTest, CrossProduct) {
    Vec3d a{1.0, 0.0, 0.0};
    Vec3d b{0.0, 1.0, 0.0};
    Vec3d c = a.Cross(b);
    EXPECT_DOUBLE_EQ(c[0], 0.0);
    EXPECT_DOUBLE_EQ(c[1], 0.0);
    EXPECT_DOUBLE_EQ(c[2], 1.0);
}

TEST(VectorTest, CrossProductAnticommutative) {
    Vec3d a{1.0, 2.0, 3.0};
    Vec3d b{4.0, 5.0, 6.0};
    Vec3d ab = a.Cross(b);
    Vec3d ba = b.Cross(a);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(ab[i], -ba[i], 1e-15);
}

TEST(VectorTest, Sum) {
    Vec3d v{1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(v.Sum(), 6.0);
}

TEST(VectorTest, Min) {
    Vec3d v{3.0, 1.0, 2.0};
    EXPECT_DOUBLE_EQ(v.Min(), 1.0);
}

TEST(VectorTest, Max) {
    Vec3d v{3.0, 1.0, 2.0};
    EXPECT_DOUBLE_EQ(v.Max(), 3.0);
}

TEST(VectorTest, Equality) {
    Vec3d a{1.0, 2.0, 3.0};
    Vec3d b{1.0, 2.0, 3.0};
    Vec3d c{1.0, 2.0, 4.0};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(VectorTest, AtInBounds) {
    Vec3d v{1.0, 2.0, 3.0};
    EXPECT_EQ(v.at(0), 1.0);
    EXPECT_EQ(v.at(2), 3.0);
}

TEST(VectorTest, AtOutOfBounds) {
    Vec3d v{1.0, 2.0, 3.0};
    EXPECT_THROW(v.at(3), std::out_of_range);
}

TEST(VectorTest, DataPointer) {
    Vec3d v{1.0, 2.0, 3.0};
    const double* p = v.data();
    EXPECT_EQ(p[0], 1.0);
    EXPECT_EQ(p[2], 3.0);
}

TEST(VectorTest, IteratorRange) {
    Vec3d v{1.0, 2.0, 3.0};
    double sum = 0.0;
    for (double x : v) sum += x;
    EXPECT_DOUBLE_EQ(sum, 6.0);
}

TEST(VectorTest, CompoundScalarDiv) {
    Vec3d a{6.0, 4.0, 2.0};
    a /= 2.0;
    EXPECT_DOUBLE_EQ(a[0], 3.0);
    EXPECT_DOUBLE_EQ(a[1], 2.0);
    EXPECT_DOUBLE_EQ(a[2], 1.0);
}

TEST(VectorTest, NamedAdd) {
    Vec3d a{1.0, 2.0, 3.0};
    Vec3d c = a.Add(Vec3d{4.0, 5.0, 6.0});
    EXPECT_DOUBLE_EQ(c[0], 5.0);
    EXPECT_DOUBLE_EQ(c[1], 7.0);
    EXPECT_DOUBLE_EQ(c[2], 9.0);
}

TEST(VectorTest, NamedSubtract) {
    Vec3d c = Vec3d{5.0, 7.0, 9.0}.Subtract(Vec3d{1.0, 2.0, 3.0});
    EXPECT_DOUBLE_EQ(c[0], 4.0);
    EXPECT_DOUBLE_EQ(c[1], 5.0);
    EXPECT_DOUBLE_EQ(c[2], 6.0);
}

TEST(VectorTest, NamedMultiply) {
    Vec3d w = Vec3d{1.0, 2.0, 3.0}.Multiply(3.0);
    EXPECT_DOUBLE_EQ(w[0], 3.0);
    EXPECT_DOUBLE_EQ(w[1], 6.0);
    EXPECT_DOUBLE_EQ(w[2], 9.0);
}

TEST(VectorTest, NamedDivide) {
    Vec3d w = Vec3d{3.0, 6.0, 9.0}.Divide(3.0);
    EXPECT_DOUBLE_EQ(w[0], 1.0);
    EXPECT_DOUBLE_EQ(w[1], 2.0);
    EXPECT_DOUBLE_EQ(w[2], 3.0);
}

TEST(VectorTest, EqualsMethod) {
    Vec3d a{1.0, 2.0, 3.0};
    EXPECT_TRUE(a.Equals(Vec3d{1.0, 2.0, 3.0}));
    EXPECT_FALSE(a.Equals(Vec3d{1.0, 2.0, 0.0}));
}

TEST(VectorTest, ConstIteratorsSum) {
    const Vec3d v{1.0, 2.0, 3.0};
    double sum = 0.0;
    for (auto it = v.cbegin(); it != v.cend(); ++it)
        sum += *it;
    EXPECT_DOUBLE_EQ(sum, 6.0);
}

TEST(VectorTest, ToStringFormat) {
    Vec3d v{1.0, 2.0, 3.0};
    EXPECT_EQ(v.ToString(), "[1, 2, 3]");
}

TEST(VectorTest, StreamInsertionOperator) {
    Vec3d v{4.0, 5.0, 6.0};
    std::ostringstream oss;
    oss << v;
    EXPECT_EQ(oss.str(), "[4, 5, 6]");
}
