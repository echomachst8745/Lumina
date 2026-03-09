#include <cmath>
#include <stdexcept>
#include <algorithm>

#include <lumina/ode/euler.hpp>

namespace lumina::ode {

ODEResult Euler(ODEFunction f, double t0, double t_end, State y0, double h)
{
    if (h <= 0) throw std::invalid_argument("Step size must be greater than 0");
    if (t_end <= t0) throw std::invalid_argument("t_end must be greater than t0");

    ODEResult result;

    double t = t0;
    State y = y0;

    // Store the initial condition as the first output point.
    result.t.push_back(t);
    result.y.push_back(y);

    // Avoid an extra final step from floating-point drift.
    while (t < t_end - h * 1e-8)
    {
        // Clamp the last step so we land exactly on t_end.
        double step = std::min(h, t_end - t);

        // Evaluate the explicit RHS.
        auto dy = f(t, y);

        // Euler update: advance each state component by step * dy.
        for (std::size_t i = 0; i < y.size(); ++i)
            y[i] += step * dy[i];
        t += step;

        result.t.push_back(t);
        result.y.push_back(y);
    }

    return result;
}

ODEResult EulerImplicit(ODEFunction f, double t0, double t_end, State y0, double h,
                        int max_iter, double delta)
{
    if (h <= 0) throw std::invalid_argument("Step size must be greater than 0");
    if (t_end <= t0) throw std::invalid_argument("t_end must be greater than t0");

    ODEResult result;

    double t = t0;
    State y = y0;

    // Store the initial condition as the first output point.
    result.t.push_back(t);
    result.y.push_back(y);

    while (t < t_end - h * 1e-8)
    {
        // Clamp the last step so we land exactly on t_end.
        double step = std::min(h, t_end - t);
        double t_next = t + step;
        const std::size_t n = y.size();

        // Start fixed-point iteration from the current state.
        State y_new = y;

        for (int iter = 0; iter < max_iter; ++iter)
        {
            // Evaluate RHS at the next time point using the current iterate.
            auto dy = f(t_next, y_new);

            State y_iter(n);
            double err = 0.0;

            for (std::size_t i = 0; i < n; ++i)
            {
                // Apply the implicit Euler formula to get the next iterate.
                y_iter[i] = y[i] + step * dy[i];

                // Accumulate squared difference to measure convergence.
                double d = y_iter[i] - y_new[i];
                err += d * d;
            }
            y_new = y_iter;

            // Converged when the L2 norm of the update falls below delta.
            if (std::sqrt(err) < delta) { break; }
        }
        y = y_new;
        t = t_next;

        result.t.push_back(t);
        result.y.push_back(y);
    }

    return result;
}

} // namespace lumina::ode
