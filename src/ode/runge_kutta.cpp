#include <cmath>
#include <algorithm>
#include <stdexcept>

#include <lumina/ode/runge_kutta.hpp>

namespace lumina::ode {

// Classic fourth-order Runge-Kutta method.
ODEResult RK4(ODEFunction f, double t0, double t_end, State y0, double h)
{
    if (h <= 0)      throw std::invalid_argument("Step size must be positive");
    if (t_end <= t0) throw std::invalid_argument("t_end must be greater than t0");

    ODEResult result;
    double t = t0;
    State  y = y0;
    const std::size_t n = y.size();

    // Store the initial condition as the first output point.
    result.t.push_back(t);
    result.y.push_back(y);

    // Builds RK stage inputs.
    auto add = [&](const State& a, double s, const State& b) -> State
    {
        State out(n);
        for (std::size_t i = 0; i < n; ++i)
            out[i] = a[i] + s * b[i];
        return out;
    };

    // Avoid an extra final step from floating-point drift.
    while (t < t_end - h * 1e-8)
    {
        // Clamp the last step so we land exactly on t_end.
        double step = std::min(h, t_end - t);

        // Evaluate the four RK4 slopes.
        auto k1 = f(t,              y);
        auto k2 = f(t + step/2.0,  add(y, step/2.0, k1));
        auto k3 = f(t + step/2.0,  add(y, step/2.0, k2));
        auto k4 = f(t + step,      add(y, step,      k3));

        // Weighted combination: midpoint slopes k2 and k3 count double.
        for (std::size_t i = 0; i < n; ++i)
            y[i] += (step/6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
        t += step;

        result.t.push_back(t);
        result.y.push_back(y);
    }

    return result;
}

// Adaptive Dormand-Prince RK45 method.
RK45Result RK45(ODEFunction f, double t0, double t_end, State y0,
                double r_delta, double a_delta, double h_init)
{

    // Dormand-Prince Butcher tableau.
    // aij: stage coefficients.
    // bi:  weights for the 5th-order solution (note: b2 = 0, so k2 is not used here).
    // ei:  weights for the error estimate = difference between 4th and 5th order solutions.
    static constexpr double
        a21=1./5.,
        a31=3./40.,     a32=9./40.,
        a41=44./45.,    a42=-56./15.,   a43=32./9.,
        a51=19372./6561.,a52=-25360./2187.,a53=64448./6561.,a54=-212./729.,
        a61=9017./3168.,a62=-355./33.,  a63=46732./5247.,a64=49./176.,a65=-5103./18656.,
        b1=35./384.,    b3=500./1113.,  b4=125./192.,   b5=-2187./6784.,b6=11./84.,
        e1=71./57600.,  e3=-71./16695., e4=71./1920.,   e5=-17253./339200.,
        e6=22./525.,    e7=-1./40.;

    RK45Result result;
    double t = t0;
    State  y = y0;
    double h = h_init;
    const std::size_t n = y.size();
    int accepted = 0, rejected = 0;

    // Store the initial condition as the first output point.
    result.t.push_back(t);
    result.y.push_back(y);

    // Builds weighted RK stage inputs.
    auto wadd = [&](const State& base,
                    double s1, const State& d1,
                    double s2, const State* d2,
                    double s3, const State* d3,
                    double s4, const State* d4,
                    double s5, const State* d5) -> State
    {
        State out(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            out[i] = base[i] + s1 * d1[i];
            if (d2) out[i] += s2 * (*d2)[i];
            if (d3) out[i] += s3 * (*d3)[i];
            if (d4) out[i] += s4 * (*d4)[i];
            if (d5) out[i] += s5 * (*d5)[i];
        }
        return out;
    };

    while (t < t_end)
    {
        // Clamp so we don't overshoot t_end, and guard against degenerate steps.
        if (t + h > t_end) h = t_end - t;
        if (h < 1e-14) break;

        // Evaluate the six Dormand-Prince stages.
        auto k1 = f(t,             y);
        auto k2 = f(t + h/5.,      wadd(y, a21*h, k1, 0.0, nullptr, 0.0, nullptr, 0.0, nullptr, 0.0, nullptr));
        auto k3 = f(t + 3.*h/10.,  wadd(y, a31*h, k1, a32*h, &k2,   0.0, nullptr, 0.0, nullptr, 0.0, nullptr));
        auto k4 = f(t + 4.*h/5.,   wadd(y, a41*h, k1, a42*h, &k2,   a43*h, &k3,  0.0, nullptr, 0.0, nullptr));
        auto k5 = f(t + 8.*h/9.,   wadd(y, a51*h, k1, a52*h, &k2,   a53*h, &k3,  a54*h, &k4,   0.0, nullptr));
        auto k6 = f(t + h,         wadd(y, a61*h, k1, a62*h, &k2,   a63*h, &k3,  a64*h, &k4,   a65*h, &k5));

        // 5th-order solution (b2 = 0 so k2 drops out).
        State y_new(n);
        for (std::size_t i = 0; i < n; ++i)
            y_new[i] = y[i] + h*(b1*k1[i] + b3*k3[i] + b4*k4[i] + b5*k5[i] + b6*k6[i]);

        // k7 supports the error estimate and FSAL reuse.
        auto k7 = f(t + h, y_new);

        // RMS-scaled local error estimate.
        double err = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            double ei = h*(e1*k1[i]+e3*k3[i]+e4*k4[i]+e5*k5[i]+e6*k6[i]+e7*k7[i]);
            double sc = a_delta + r_delta * std::max(std::abs(y[i]), std::abs(y_new[i]));
            err += (ei/sc)*(ei/sc);
        }
        err = std::sqrt(err / static_cast<double>(n));

        if (err <= 1.0)
        {
            // Accept the step and grow h conservatively.
            ++accepted; t += h; y = y_new;
            result.t.push_back(t); result.y.push_back(y);
            h *= std::min(5.0, 0.9 * std::pow(1.0/std::max(err, 1e-10), 0.2));
        }
        else
        {
            // Reject the step and shrink h.
            ++rejected;
            h *= std::max(0.1, 0.9 * std::pow(1.0/err, 0.25));
        }
    }
    result.steps_accepted = accepted;
    result.steps_rejected = rejected;
    
    return result;
}

} // namespace lumina::ode
