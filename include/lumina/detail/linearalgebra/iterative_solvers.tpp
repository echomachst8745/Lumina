namespace lumina::linearalgebra {

using lumina::core::Vector;
using lumina::core::Matrix;

// Jacobi iteration using the previous solution vector.
template<typename T, std::size_t N>
SolverResult<T, N> Jacobi(const Matrix<T, N, N>& A, const Vector<T, N>& b,
                          Vector<T, N> x0, int max_iter, T tol)
{
    Vector<T, N> x    = x0;
    Vector<T, N> x_new{};
    int          iter = 0;
    double       res  = 0.0;

    for (; iter < max_iter; ++iter)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            // Sum off-diagonal terms from the old x.
            T sigma = T{};
            for (std::size_t j = 0; j < N; ++j)
                if (j != i) sigma += A(i, j) * x[j];

            x_new[i] = (b[i] - sigma) / A(i, i);
        }

        // Measure the step size; stop when the update is below tolerance.
        res = static_cast<double>((x_new - x).Norm());
        x   = x_new;
        if (res < static_cast<double>(tol)) { ++iter; break; }
    }

    return { x, iter, res, res < static_cast<double>(tol) };
}

// Gauss-Seidel iteration with in-place updates.
template<typename T, std::size_t N>
SolverResult<T, N> GaussSeidel(const Matrix<T, N, N>& A, const Vector<T, N>& b,
                               Vector<T, N> x, int max_iter, T tol)
{
    int    iter = 0;
    double res  = 0.0;

    for (; iter < max_iter; ++iter)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            // Earlier entries already contain this iteration's values.
            T sigma = T{};
            for (std::size_t j = 0; j < N; ++j)
                if (j != i) sigma += A(i, j) * x[j];

            x[i] = (b[i] - sigma) / A(i, i);
        }

        // Compute the residual r = b - Ax and stop when ||r|| < tol.
        Vector<T, N> r = b - A * x;
        res = static_cast<double>(r.Norm());
        if (res < static_cast<double>(tol)) { ++iter; break; }
    }

    return { x, iter, res, res < static_cast<double>(tol) };
}

// Conjugate Gradient solver for SPD systems.
template<typename T, std::size_t N>
SolverResult<T, N> ConjugateGradient(const Matrix<T, N, N>& A, const Vector<T, N>& b,
                                     Vector<T, N> x, int max_iter, T tol)
{
    // Initial residual r = b - Ax, initial search direction p = r.
    Vector<T, N> r = b - A * x;
    Vector<T, N> p = r;

    T      r_dot    = r.Dot(r);               // Cached residual norm squared.
    double b_norm   = static_cast<double>(b.Norm());
    int    iter     = 0;
    double res      = 0.0;

    // Zero right-hand side already satisfies Ax = b.
    if (b_norm < static_cast<double>(std::numeric_limits<T>::epsilon()))
        return { x, 0, 0.0, true };

    for (; iter < max_iter; ++iter)
    {
        Vector<T, N> Ap    = A * p;            // Matrix-vector product for this step.
        T            pAp   = p.Dot(Ap);        // Denominator of the step length.

        // Degenerate direction; A may not be SPD.
        if (std::abs(pAp) < std::numeric_limits<T>::epsilon())
            break;

        T alpha = r_dot / pAp;                 // Step length along p.

        x = x + p * alpha;                     // Advance solution.
        r = r - Ap * alpha;                    // Update residual.

        T r_dot_new = r.Dot(r);

        res = static_cast<double>(r.Norm()) / b_norm;
        if (res < static_cast<double>(tol)) { ++iter; break; }

        T beta = r_dot_new / r_dot;            // Direction update scalar.
        p      = r + p * beta;                 // New A-conjugate search direction.
        r_dot  = r_dot_new;
    }

    return { x, iter, res, res < static_cast<double>(tol) };
}

} // namespace lumina::linearalgebra
