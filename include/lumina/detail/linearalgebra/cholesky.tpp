namespace lumina::linearalgebra {

using lumina::core::Vector;
using lumina::core::Matrix;

// Computes the Cholesky factor A = L * L^T.
template<typename T, std::size_t N>
CholeskyResult<T, N> CholeskyDecompose(Matrix<T, N, N> A)
{
    CholeskyResult<T, N> result{};

    for (std::size_t i = 0; i < N; ++i)
    {
        // Fill the strict lower row.
        for (std::size_t j = 0; j < i; ++j)
        {
            T sum = A(i, j);
            for (std::size_t k = 0; k < j; ++k)
                sum -= result.L(i, k) * result.L(j, k);
            result.L(i, j) = sum / result.L(j, j);
        }

        // Fill the diagonal entry.
        T diag_sum = A(i, i);
        for (std::size_t k = 0; k < i; ++k)
            diag_sum -= result.L(i, k) * result.L(i, k);

        if (diag_sum <= T{})
            throw std::runtime_error("Matrix is not positive-definite");

        result.L(i, i) = std::sqrt(diag_sum);
    }

    return result;
}

// Solves Ax = b from a precomputed Cholesky factor.
template<typename T, std::size_t N>
Vector<T, N> CholeskySolve(const CholeskyResult<T, N>& ch, Vector<T, N> b)
{
    // Forward substitution: solve Ly = b.
    Vector<T, N> y{};
    for (std::size_t i = 0; i < N; ++i)
    {
        T sum = b[i];
        for (std::size_t j = 0; j < i; ++j)
            sum -= ch.L(i, j) * y[j];
        y[i] = sum / ch.L(i, i);
    }

    // Back substitution: solve L^T x = y.
    Vector<T, N> x{};
    for (int i = static_cast<int>(N) - 1; i >= 0; --i)
    {
        T sum = y[i];
        for (std::size_t j = i + 1; j < N; ++j)
            sum -= ch.L(j, i) * x[j];   // L^T(i,j) = L(j,i)
        x[i] = sum / ch.L(i, i);
    }

    return x;
}

// Convenience overload: decomposes A on the fly then solves.
template<typename T, std::size_t N>
Vector<T, N> CholeskySolve(const Matrix<T, N, N>& A, Vector<T, N> b)
{
    return CholeskySolve(CholeskyDecompose(A), b);
}

// Computes det(A) from a precomputed Cholesky factor.
template<typename T, std::size_t N>
T CholeskyDeterminant(const CholeskyResult<T, N>& ch)
{
    T prod = T{1};
    for (std::size_t i = 0; i < N; ++i)
        prod *= ch.L(i, i);
    return prod * prod;
}

} // namespace lumina::linearalgebra
