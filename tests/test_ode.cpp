#include <gtest/gtest.h>
#include <lumina/ode/euler.hpp>
#include <lumina/ode/runge_kutta.hpp>
#include <cmath>
#include <vector>

using namespace lumina::ode;

static constexpr double PI = 3.14159265358979323846;

// ---- Euler (forward) ----

// dy/dt = -y, y(0) = 1 -> y(t) = exp(-t)
TEST(EulerTest, ExponentialDecay) {
    auto f = [](double /*t*/, const State& y) -> State {
        return { -y[0] };
        };
    ODEResult result = Euler(f, 0.0, 2.0, { 1.0 }, 0.001);
    EXPECT_FALSE(result.t.empty());
    EXPECT_EQ(result.t.size(), result.y.size());
    double y_final = result.y.back()[0];
    EXPECT_NEAR(y_final, std::exp(-2.0), 1e-3);
}

// dy/dt = y, y(0) = 1 -> y(t) = exp(t)
TEST(EulerTest, ExponentialGrowth) {
    auto f = [](double /*t*/, const State& y) -> State {
        return { y[0] };
        };
    ODEResult result = Euler(f, 0.0, 1.0, { 1.0 }, 0.0001);
    EXPECT_NEAR(result.y.back()[0], std::exp(1.0), 1e-3);
}

// Constant ODE: dy/dt = 0, y(0) = 5 -> y(t) = 5 always
TEST(EulerTest, ConstantSolution) {
    auto f = [](double /*t*/, const State& /*y*/) -> State {
        return { 0.0 };
        };
    ODEResult result = Euler(f, 0.0, 3.0, { 5.0 }, 0.5);
    for (const State& s : result.y)
        EXPECT_DOUBLE_EQ(s[0], 5.0);
}

TEST(EulerTest, OutputSizesMatch) {
    auto f = [](double /*t*/, const State& y) -> State { return { -y[0] }; };
    ODEResult result = Euler(f, 0.0, 1.0, { 1.0 }, 0.1);
    EXPECT_EQ(result.t.size(), result.y.size());
    EXPECT_GE(result.t.size(), 2u);
    EXPECT_DOUBLE_EQ(result.t.front(), 0.0);
    EXPECT_NEAR(result.t.back(), 1.0, 1e-14);
}

// System: [x', y'] = [y, -x] -> x=cos(t), y=sin(t)
TEST(EulerTest, HarmonicOscillator) {
    auto f = [](double /*t*/, const State& y) -> State {
        return { y[1], -y[0] };
        };
    ODEResult result = Euler(f, 0.0, PI / 2.0, { 1.0, 0.0 }, 0.0001);
    EXPECT_NEAR(result.y.back()[0], 0.0,  1e-3); // x = cos(pi/2) = 0
    EXPECT_NEAR(result.y.back()[1], -1.0, 1e-3); // v = -sin(pi/2) = -1
}

// ---- RK4 ----

// dy/dt = -y, y(0) = 1 -> y(t) = exp(-t)
TEST(RK4Test, ExponentialDecay) {
    auto f = [](double /*t*/, const State& y) -> State {
        return { -y[0] };
        };
    ODEResult result = RK4(f, 0.0, 2.0, { 1.0 }, 0.01);
    EXPECT_EQ(result.t.size(), result.y.size());
    double y_final = result.y.back()[0];
    EXPECT_NEAR(y_final, std::exp(-2.0), 1e-7);
}

// RK4 should be ~1000x more accurate than Euler for same step size
TEST(RK4Test, MoreAccurateThanEuler) {
    auto f = [](double /*t*/, const State& y) -> State {
        return { -y[0] };
        };
    double exact = std::exp(-1.0);
    ODEResult euler_r = Euler(f, 0.0, 1.0, { 1.0 }, 0.01);
    ODEResult rk4_r = RK4(f, 0.0, 1.0, { 1.0 }, 0.01);
    double euler_err = std::abs(euler_r.y.back()[0] - exact);
    double rk4_err = std::abs(rk4_r.y.back()[0] - exact);
    EXPECT_LT(rk4_err, euler_err);
}

// d2x/dt2 + x = 0 -> x(t) = cos(t), x'(t) = -sin(t)
// At t = pi: x(pi) = -1
TEST(RK4Test, HarmonicOscillator) {
    auto f = [](double /*t*/, const State& y) -> State {
        return { y[1], -y[0] };
        };
    ODEResult result = RK4(f, 0.0, PI, { 1.0, 0.0 }, 0.001);
    EXPECT_NEAR(result.y.back()[0], -1.0, 1e-6); // cos(pi) = -1
    EXPECT_NEAR(result.y.back()[1], 0.0, 1e-6); // sin(pi) = 0 (x' = -sin)
}

TEST(RK4Test, ConstantSolution) {
    auto f = [](double /*t*/, const State& /*y*/) -> State {
        return { 0.0 };
        };
    ODEResult result = RK4(f, 0.0, 5.0, { 3.0 }, 0.5);
    for (const State& s : result.y)
        EXPECT_DOUBLE_EQ(s[0], 3.0);
}

TEST(RK4Test, OutputSizesMatch) {
    auto f = [](double /*t*/, const State& y) -> State { return { -y[0] }; };
    ODEResult result = RK4(f, 0.0, 1.0, { 1.0 }, 0.1);
    EXPECT_EQ(result.t.size(), result.y.size());
    EXPECT_GE(result.t.size(), 2u);
}

// ---- RK45 (adaptive Dormand-Prince) ----

// dy/dt = -y, y(0) = 1 -> y(t) = exp(-t)
TEST(RK45Test, ExponentialDecay) {
    auto f = [](double /*t*/, const State& y) -> State {
        return { -y[0] };
        };
    RK45Result result = RK45(f, 0.0, 2.0, { 1.0 }, 1e-8, 1e-10);
    EXPECT_GT(result.steps_accepted, 0);
    EXPECT_EQ(result.t.size(), result.y.size());
    EXPECT_NEAR(result.y.back()[0], std::exp(-2.0), 1e-6);
}

TEST(RK45Test, AcceptsMoreThanRejects) {
    auto f = [](double /*t*/, const State& y) -> State {
        return { -y[0] };
        };
    RK45Result result = RK45(f, 0.0, 1.0, { 1.0 }, 1e-6, 1e-9);
    EXPECT_GT(result.steps_accepted, result.steps_rejected);
}

TEST(RK45Test, HarmonicOscillator) {
    auto f = [](double /*t*/, const State& y) -> State {
        return { y[1], -y[0] };
        };
    RK45Result result = RK45(f, 0.0, PI, { 1.0, 0.0 }, 1e-8, 1e-10);
    EXPECT_NEAR(result.y.back()[0], -1.0, 1e-5); // cos(pi) = -1
}

TEST(RK45Test, LooseTolUsesFewerSteps) {
    auto f = [](double /*t*/, const State& y) -> State { return { -y[0] }; };
    RK45Result tight = RK45(f, 0.0, 1.0, { 1.0 }, 1e-10, 1e-12);
    RK45Result loose = RK45(f, 0.0, 1.0, { 1.0 }, 1e-4, 1e-6);
    EXPECT_LT(loose.steps_accepted, tight.steps_accepted);
}

// ---- Implicit Euler ----

// dy/dt = -y, y(0) = 1 -> y(1) = exp(-1). Implicit Euler: y_{n+1} = y_n/(1+h).
TEST(EulerImplicitTest, ExponentialDecay) {
    auto f = [](double /*t*/, const State& y) -> State { return { -y[0] }; };
    ODEResult result = EulerImplicit(f, 0.0, 1.0, { 1.0 }, 0.1);
    EXPECT_FALSE(result.t.empty());
    EXPECT_EQ(result.t.size(), result.y.size());
    EXPECT_NEAR(result.y.back()[0], std::exp(-1.0), 1e-1);
}

// dy/dt = 0, y(0) = 5 -> y = 5 always (implicit iteration converges in 1 step).
TEST(EulerImplicitTest, ConstantSolution) {
    auto f = [](double /*t*/, const State& /*y*/) -> State { return { 0.0 }; };
    ODEResult result = EulerImplicit(f, 0.0, 3.0, { 5.0 }, 0.5);
    for (const State& s : result.y)
        EXPECT_DOUBLE_EQ(s[0], 5.0);
}

TEST(EulerImplicitTest, OutputSizesMatch) {
    auto f = [](double /*t*/, const State& y) -> State { return { -y[0] }; };
    ODEResult result = EulerImplicit(f, 0.0, 1.0, { 1.0 }, 0.1);
    EXPECT_EQ(result.t.size(), result.y.size());
    EXPECT_GE(result.t.size(), 2u);
    EXPECT_DOUBLE_EQ(result.t.front(), 0.0);
    EXPECT_NEAR(result.t.back(), 1.0, 1e-14);
}

// For a purely dissipative ODE dy/dt = -y, implicit Euler (fixed-point iteration
// with h*K = 0.5 < 1) should produce a strictly monotone decreasing sequence.
TEST(EulerImplicitTest, DecaysMonotonically) {
    auto f = [](double /*t*/, const State& y) -> State { return { -y[0] }; };
    ODEResult result = EulerImplicit(f, 0.0, 2.0, { 1.0 }, 0.5);
    for (std::size_t i = 1; i < result.y.size(); ++i)
        EXPECT_LT(result.y[i][0], result.y[i - 1][0]);
}
