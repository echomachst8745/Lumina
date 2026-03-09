namespace lumina::linearalgebra {

using lumina::core::Vector;
using lumina::core::Matrix;

// Solves Ax = b with partial-pivoting Gaussian elimination.
template<typename T, std::size_t N>
Vector<T, N> GaussianSolve(Matrix<T, N, N> A, Vector<T, N> b)
{
    // Forward elimination.
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

        // Reject singular columns.
        if (max_val < std::numeric_limits<T>::epsilon() * 1000)
            throw std::runtime_error("Matrix is singular or nearly singular");

        // Keep A and b in sync when pivoting.
        if (pivot != col) { A.SwapRows(col, pivot); std::swap(b[col], b[pivot]); }

        T diag = A(col, col);

        // Eliminate entries below the diagonal.
        for (std::size_t row = col + 1; row < N; ++row)
        {
            T factor = A(row, col) / diag;
            if (factor == T{}) continue;                 // Already zero, skip.
            for (std::size_t k = col; k < N; ++k)
                A(row, k) -= factor * A(col, k);
            b[row] -= factor * b[col];                   // Apply the same row step to b.
        }
    }

    // Back substitution.
    Vector<T, N> x{};
    for (int i = static_cast<int>(N) - 1; i >= 0; --i)
    {
        T sum = b[i];
        for (std::size_t j = i + 1; j < N; ++j)
            sum -= A(i, j) * x[j];
        x[i] = sum / A(i, i);
    }

    return x;
}

// Computes A^-1 with Gauss-Jordan elimination.
template<typename T, std::size_t N>
Matrix<T, N, N> Inverse(Matrix<T, N, N> A)
{
    // Row operations transform identity into the inverse.
    Matrix<T, N, N> inv = Matrix<T, N, N>::Identity();

    for (std::size_t col = 0; col < N; ++col)
    {
        // Partial pivot: find the row with the largest value in this column.
        std::size_t pivot = col;
        T max_val = std::abs(A(col, col));
        for (std::size_t row = col + 1; row < N; ++row)
        {
            T val = std::abs(A(row, col));
            if (val > max_val) { max_val = val; pivot = row; }
        }

        if (max_val < std::numeric_limits<T>::epsilon() * 1000)
            throw std::runtime_error("Matrix is singular");

        // Apply the same row swap to both A and inv to keep them in sync.
        if (pivot != col) { A.SwapRows(col, pivot); inv.SwapRows(col, pivot); }

        T diag = A(col, col);

        // Scale the pivot row to unit diagonal.
        for (std::size_t k = 0; k < N; ++k) { A(col,k)/=diag; inv(col,k)/=diag; }

        // Eliminate the rest of this column.
        for (std::size_t row = 0; row < N; ++row)
        {
            if (row == col) continue;               // Skip the pivot row itself.
            T factor = A(row, col);
            if (factor == T{}) continue;            // Entry is already zero.
            for (std::size_t k = 0; k < N; ++k)
            {
                A(row,k)   -= factor * A(col,k);
                inv(row,k) -= factor * inv(col,k);
            }
        }
    }

    return inv;
}

// Computes det(A) with partial-pivoting elimination.
template<typename T, std::size_t N>
T Determinant(Matrix<T, N, N> A)
{
    T det = T{1};

    for (std::size_t col = 0; col < N; ++col)
    {
        // Partial pivot.
        std::size_t pivot = col;
        T max_val = std::abs(A(col, col));
        for (std::size_t row = col + 1; row < N; ++row)
        {
            T val = std::abs(A(row, col));
            if (val > max_val) { max_val = val; pivot = row; }
        }

        // Singular matrix has zero determinant.
        if (max_val < std::numeric_limits<T>::epsilon() * 1000) return T{};

        // Each row swap negates the determinant.
        if (pivot != col) { A.SwapRows(col, pivot); det = -det; }

        // Accumulate the diagonal entry into the determinant product.
        det *= A(col, col);
        T diag = A(col, col);

        // Eliminate entries below the diagonal.
        for (std::size_t row = col + 1; row < N; ++row)
        {
            T factor = A(row, col) / diag;
            for (std::size_t k = col; k < N; ++k)
                A(row,k) -= factor * A(col,k);
        }
    }

    return det;
}

} // namespace lumina::linearalgebra
