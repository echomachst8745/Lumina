namespace lumina::linearalgebra {

using lumina::core::Vector;
using lumina::core::Matrix;

// Computes PA = LU with partial pivoting.
template<typename T, std::size_t N>
LUResult<T, N> LUDecompose(Matrix<T, N, N> A)
{
    LUResult<T, N> result;

    // Start with the identity permutation.
    std::iota(result.perm.begin(), result.perm.end(), 0);
    result.sign = 1;

    for (std::size_t col = 0; col < N; ++col)
    {
        // Find the partial pivot.
        std::size_t pivot = col;
        T max_val = std::abs(A(col, col));
        for (std::size_t row = col + 1; row < N; ++row)
        {
            T val = std::abs(A(row, col));
            if (val > max_val) { max_val = val; pivot = row; }
        }

        if (max_val < std::numeric_limits<T>::epsilon() * 1000)
            throw std::runtime_error("Matrix is singular");

        if (pivot != col)
        {
            // Record the pivot row swap.
            A.SwapRows(col, pivot);
            std::swap(result.perm[col], result.perm[pivot]);
            result.sign = -result.sign;
        }

        // Store multipliers below the diagonal.
        for (std::size_t row = col + 1; row < N; ++row)
        {
            T factor = A(row, col) / A(col, col);
            A(row, col) = factor;               // Store multiplier in place.
            for (std::size_t k = col + 1; k < N; ++k)
                A(row, k) -= factor * A(col, k);
        }
    }

    // Unpack the combined matrix into L and U.
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            if (i > j)
            {
                result.L(i, j) = A(i, j);
                result.U(i, j) = T{};
            }
            else if (i == j)
            {
                result.L(i, j) = T{1};
                result.U(i, j) = A(i, j);
            }
            else
            {
                result.L(i, j) = T{};
                result.U(i, j) = A(i, j);
            }
        }
    }

    return result;
}

// Solves Ax = b from a precomputed LU decomposition.
template<typename T, std::size_t N>
Vector<T, N> LUSolve(const LUResult<T, N>& lu, Vector<T, N> b)
{
    // Apply the recorded row permutation.
    Vector<T, N> pb{};
    for (std::size_t i = 0; i < N; ++i)
        pb[i] = b[lu.perm[i]];

    // Forward substitution.
    Vector<T, N> y{};
    for (std::size_t i = 0; i < N; ++i)
    {
        T sum = pb[i];
        for (std::size_t j = 0; j < i; ++j)
            sum -= lu.L(i, j) * y[j];
        y[i] = sum;
    }

    // Back substitution.
    Vector<T, N> x{};
    for (int i = static_cast<int>(N) - 1; i >= 0; --i)
    {
        T sum = y[i];
        for (std::size_t j = i + 1; j < N; ++j)
            sum -= lu.U(i, j) * x[j];
        x[i] = sum / lu.U(i, i);
    }

    return x;
}

// Convenience overload: decompose, then solve.
template<typename T, std::size_t N>
Vector<T, N> LUSolve(const Matrix<T, N, N>& A, Vector<T, N> b)
{
    return LUSolve(LUDecompose(A), b);
}

// Computes det(A) from a precomputed LU decomposition.
template<typename T, std::size_t N>
T LUDeterminant(const LUResult<T, N>& lu)
{
    T det = static_cast<T>(lu.sign);
    for (std::size_t i = 0; i < N; ++i)
        det *= lu.U(i, i);
    return det;
}

} // namespace lumina::linearalgebra
