#include <iostream>
#include <array>
#include <type_traits>
#include <random>
#include <utility>
#include <cmath>
#include <algorithm>
#include <complex>
#include <stdexcept>

namespace eigencipy
{
  const double EPS = 10e-14;
  std::default_random_engine rng;

  template <typename Num>
  class matrix
  {
  private:
    Num *data_;
    size_t height_, width_;

  public:
    /*
     * Memory management.
     */
    friend void swap(matrix &first, matrix &second) noexcept
    {
      using std::swap;
      swap(first.height_, second.height_);
      swap(first.width_, second.width_);
      swap(first.data_, second.data_);
    }

    friend std::ostream &operator<<(std::ostream &stream, const matrix &in)
    {
      for (size_t h = 0; h < in.height(); h++)
      {
        stream << "[";
        for (size_t w = 0; w < in.width(); w++)
        {
          if (w != 0)
            stream << " ";

          if (std::norm(in.element(h, w)) < EPS)
            stream << 0;
          else if (std::imag(in.element(h, w)) < EPS)
            stream << std::real(in.element(h, w));
          else
            stream << std::real(in.element(h, w)) << "+"
                   << std::imag(in.element(h, w)) << "i";
        }
        stream << "]";
        if (h != in.height() - 1)
          stream << "\n";
      }
      return stream;
    }

    matrix() : data_(nullptr), height_(0), width_(0) {}
    matrix(size_t height, size_t width) : height_(height), width_(width)
    {
      data_ = new Num[height * width];
    }
    matrix(matrix &&other) : matrix() { swap(*this, other); }

    matrix(const matrix &other) : matrix(other.height_, other.width_)
    {
      std::copy_n(other.data_, other.height_ * other.width_, data_);
    }

    matrix &operator=(matrix other)
    {
      swap(*this, other);
      return *this;
    }

    ~matrix() { delete[] data_; }

    /*
      Element access.
    */
    inline size_t height() const { return height_; }

    inline size_t width() const { return width_; }

    inline Num element(size_t row, size_t col) const
    {
      return data_[row * width_ + col];
    }

    inline Num &element(size_t row, size_t col)
    {
      return data_[row * width_ + col];
    }

    /*
      Basic operations.
    */
    matrix operator*(const matrix &rhs) const
    {
      matrix result(height(), rhs.width());
      for (size_t i = 0; i < height(); ++i)
        for (size_t j = 0; j < rhs.width(); ++j)
        {
          result.element(i, j) = 0;
          for (size_t t = 0; t < width(); ++t)
            result.element(i, j) += element(i, t) * rhs.element(t, j);
        }
      return result;
    }

    matrix &operator*=(Num rhs)
    {
      for (size_t i = 0; i < height(); ++i)
        for (size_t j = 0; j < width(); ++j)
          element(i, j) *= rhs;
      return *this;
    }

    matrix &operator+=(const matrix &rhs)
    {
      for (size_t i = 0; i < height(); ++i)
        for (size_t j = 0; j < width(); ++j)
          element(i, j) += rhs.element(i, j);
      return *this;
    }

    matrix &operator-=(const matrix &rhs)
    {
      for (size_t i = 0; i < height(); ++i)
        for (size_t j = 0; j < width(); ++j)
          element(i, j) -= rhs.element(i, j);
      return *this;
    }

    matrix operator+(const matrix &rhs) const
    {
      matrix result(*this);
      result += rhs;
      return result;
    }

    matrix operator-(const matrix &rhs) const
    {
      matrix result(*this);
      result -= rhs;
      return result;
    }

    /*
      Slicing elements.
    */
    matrix column(size_t n) const
    {
      matrix result(height(), 1);
      for (size_t i = 0; i < height(); ++i)
        result.element(i, 0) = element(i, n);
      return result;
    }

    matrix row(size_t n) const
    {
      matrix result(1, width());
      for (size_t i = 0; i < width(); ++i)
        result.element(0, i) = element(n, i);
      return result;
    }
  };

  /*
   * Frobenius norm
   * --------------
   * Returns the Frobenius norm of a given matrix.
   *
   * @params: matrix, a given matrix
   */
  template <typename Num>
  Num frobenius(const matrix<Num> &matrix)
  {
    Num result = 0;
    for (size_t i = 0; i < matrix.height(); ++i)
      for (size_t j = 0; j < matrix.width(); ++j)
        result += std::norm(matrix.element(i, j));
    return std::sqrt(result);
  }

  /*
   * Dot product
   * -----------
   * Returns the scalar product of two matrix.
   *
   * @params: matrix, a given matrix
   */
  template <typename Num>
  Num dot(const matrix<Num> &m1, const matrix<Num> &m2)
  {
    Num result = 0;
    for (size_t i = 0; i < m1.height(); ++i)
      for (size_t j = 0; j < m2.width(); ++j)
        result += std::conj(m2.element(i, j)) * m1.element(i, j);
    return result;
  }

  template <typename Num>
  Num rayleigh(const matrix<Num> &A, const matrix<Num> &x)
  {
    return dot(x, A * x) / dot(x, x);
  }

  template <typename Num>
  Num distance(const matrix<Num> &m1, const matrix<Num> &m2)
  {
    Num result = 0;
    for (size_t i = 0; i < m1.height(); ++i)
      for (size_t j = 0; j < m2.width(); ++j)
        result += std::norm(m1.element(i, j) - m2.element(i, j));
    return std::sqrt(result);
  }

  template <typename Num>
  void normalize(matrix<Num> &matrix)
  {
    Num norm = frobenius(matrix);
    matrix *= (1 / norm);
  }


  // The identity matrix.
  template <typename Num>
  matrix<Num> identity(size_t height, size_t width)
  {
    matrix<Num> result(height, width);
    for (size_t i = 0; i < height; ++i)
      for (size_t j = 0; j < width; ++j)
        result.element(i, j) = i == j ? 1 : 0;
    return result;
  }

  // The all-ones matrix.
  template <typename Num>
  matrix<Num> ones(size_t height, size_t width)
  {
    matrix<Num> result(height, width);
    for (size_t i = 0; i < height; ++i)
      for (size_t j = 0; j < width; ++j)
        result.element(i, j) = 1;
    return result;
  }

  // The diagonal matrix.
  template <typename Num, typename T>
  matrix<Num> diagonal(T &&diag)
  {
    size_t size = diag.end() - diag.begin();
    matrix<Num> start(size, size);
    std::cout << size << std::endl;
    for (size_t i = 0; i < size; ++i)
      for (size_t j = 0; j < size; ++j)
      {
        start.element(i, j) = (i == j) ? *(diag.begin() + i) : 0;
      }
    return start;
  }

  // The all-zeroes matrix.
  template <typename Num>
  matrix<Num> zeros(size_t height, size_t width)
  {
    matrix<Num> result(height, width);
    for (size_t i = 0; i < height; ++i)
      for (size_t j = 0; j < width; ++j)
        result.element(i, j) = 0;
    return result;
  }

  // Computation settings.
  template <typename Num>
  typename std::enable_if<std::is_floating_point<Num>::value, matrix<Num>>::type
  random(size_t height, size_t width)
  {
    matrix<Num> start(height, width);
    std::normal_distribution<Num> dist;
    for (size_t i = 0; i < height; ++i)
      for (size_t j = 0; j < width; ++j)
        start.element(i, j) = dist(rng);
    return start;
  }

  template <typename Num>
  typename std::enable_if<std::is_integral<Num>::value, matrix<Num>>::type
  random(size_t height, size_t width)
  {
    matrix<Num> start(height, width);
    std::normal_distribution<double> dist;
    for (size_t i = 0; i < height; ++i)
      for (size_t j = 0; j < width; ++j)
        start.element(i, j) = (Num)dist(rng);
    return start;
  }

  template <typename Num>
  typename std::enable_if<std::is_floating_point<typename Num::value_type>::value,
                          matrix<Num>>::type
  random(size_t height, size_t width)
  {
    matrix<Num> start(height, width);
    std::normal_distribution<typename Num::value_type> dist;
    for (size_t i = 0; i < height; ++i)
      for (size_t j = 0; j < width; ++j)
        start.element(i, j) = Num(dist(rng), dist(rng));
    return start;
  }

  /*
   * Transposition
   * --------------
   * Returns a matrix flipped about its major diagonal.
   *
   * @params: A, a given matrix
   */
  template <typename Num>
  matrix<Num> transpose(matrix<Num> A)
  {
    matrix<Num> result(A.width(), A.height());
    for (size_t i = 0; i < A.height(); ++i)
      for (size_t j = 0; j < A.width(); ++j)
        result.element(j, i) = A.element(i, j);
    return result;
  }

  /*
   * Hermitian transposition
   * -----------------------
   * Returns the complex-conjugated transpose of a given matrix.
   *
   * @params: A, a given matrix
   */
  template <typename Num>
  matrix<Num> conjugate_transpose(matrix<Num> A)
  {
    matrix<Num> result(A.width(), A.height());
    for (size_t i = 0; i < A.height(); ++i)
      for (size_t j = 0; j < A.width(); ++j)
        result.element(j, i) = std::conj(A.element(i, j));
    return result;
  }

  /*
   * QU decomposition
   * ----------------
   * Computes the QR decomposition of the matrix with modified Gram-Schmidt
   * orthonormalization.
   *
   * @params: A, a given matrix
   */
  template <typename Num>
  std::pair<matrix<Num>, matrix<Num>> qu_decomposition_gs(const matrix<Num> &A)
  {
    matrix<Num> Q(A.height(), A.width());
    matrix<Num> U(zeros<Num>(A.width(), A.width()));

    for (size_t k = 0; k < A.width(); ++k)
    {
      for (size_t j = 0; j < A.height(); ++j)
      {
        Q.element(j, k) = A.element(j, k);
      }
      for (size_t i = 0; i < k; ++i)
      {
        U.element(i, k) = 0;
        for (size_t j = 0; j < A.height(); ++j)
        {
          U.element(i, k) += std::conj(Q.element(j, i)) * Q.element(j, k);
        }
        for (size_t j = 0; j < A.height(); ++j)
        {
          Q.element(j, k) -= U.element(i, k) * Q.element(j, i);
        }
      }

      Num norm = 0;
      for (size_t j = 0; j < A.height(); ++j)
      {
        norm += std::norm(Q.element(j, k));
      }
      U.element(k, k) = sqrt(norm);

      for (size_t j = 0; j < A.height(); ++j)
      {
        Q.element(j, k) /= U.element(k, k);
      }
    }
    return {Q, U};
  }

  /*
   * Random orthogonal matrix
   * ------------------------
   * Returns a random orthogonal matrix using the Gram-Schmidt
   * variant of the QU decomposition.
   *
   * @params: size
   */
  template <typename Num>
  matrix<Num> random_orthogonal(size_t size)
  {
    matrix<Num> start = random<Num>(size, size);
    return qu_decomposition_gs(start).first;
  }

  void signature()
  { // I hereby declare that:
    // I am a self-made nillionaire,
    // Grothendieck was the real GOAT,
    // The Weeping Angels are the best creatures of TV,
    // Eckmann-Hilton begets commutativity eternally,
    // There is no more time.
    // Signed, that lifelong Chelsea FC fan,
    // Ekene    
  }

  /*
  * Back substitution
  * -----------------
  * Returns a solution to the linear problem Ax = b.
  * 
  * @params: A, a given upper-triangular matrix
  */
  template <typename Num>
  matrix<Num> back_substitution(const matrix<Num> &A, matrix<Num> b)
  {
    for (size_t k = 0; k < b.width(); ++k)
    {
      for (size_t i = A.height(); i >= 1; --i)
      {
        Num sum = 0;
        for (size_t j = i; j < A.width(); j++)
          sum += A.element(i - 1, j) * b.element(j, k);
        b.element(i - 1, k) =
            (b.element(i - 1, k) - sum) / A.element(i - 1, i - 1);
      }
    }
    return b;
  }

  /*
  * Forward substitution
  * --------------------
  * Returns a solution to the linear problem Ax = b.
  * 
  * @params: A, a given lower-triangular matrix
  */
  template <typename Num>
  matrix<Num> forward_substitution(const matrix<Num> &A, matrix<Num> b)
  {
    for (size_t k = 0; k < b.width(); ++k)
    {
      for (size_t i = 0; i < A.height(); ++i)
      {
        Num sum = 0;
        for (size_t j = 0; j < i; j++)
          sum += A.element(i, j) * b.element(j, k);
        b.element(i, k) = (b.element(i, k) - sum) / A.element(i, i);
      }
    }
    return b;
  }


  /*
  * LU decomposition
  * ----------------
  * Computes the A = LU decomposition without pivoting of a square matrix A where
  * L is a lower triangular matrix and U is an upper-triangular matrix.
  * The result is given in the form L + U - I.
  * 
  * @params: L, a given lower-triangular matrix
  */
  template <typename Num>
  matrix<Num> lu_decomposition(matrix<Num> L)
  {
    for (size_t i = 0; i < L.height(); ++i)
    {
      for (size_t j = 0; j < i; ++j)
      {
        auto alpha = L.element(i, j);
        for (size_t p = 0; p < j; ++p)
          alpha -= L.element(i, p) * L.element(p, j);
        L.element(i, j) = alpha / L.element(j, j);
      }
      for (size_t j = i; j < L.height(); ++j)
      {
        auto alpha = L.element(i, j);
        for (size_t p = 0; p < i; ++p)
          alpha -= L.element(i, p) * L.element(p, j);
        L.element(i, j) = alpha;
      }
    }
    return L;
  }

  /*
  * LU linear solver
  * ----------------
  * Solves the full-rank square linear system Ax = b via the LU decomposition.
  * 
  * @params: A, b, given matrices
  */
  template <typename Num>
  matrix<Num> linear_solve_lu(const matrix<Num> &A, matrix<Num> b)
  {
    matrix<Num> lu = lu_decomposition(A);
    for (size_t k = 0; k < b.width(); ++k)
    {
      for (size_t i = 0; i < lu.height(); ++i)
      {
        Num sum = 0;
        for (size_t j = 0; j < i; j++)
          sum += lu.element(i, j) * b.element(j, k);
        b.element(i, k) = (b.element(i, k) - sum);
      }
    }
    for (size_t k = 0; k < b.width(); ++k)
    {
      for (size_t i = lu.height(); i >= 1; --i)
      {
        Num sum = 0;
        for (size_t j = i; j < A.width(); j++)
          sum += lu.element(i - 1, j) * b.element(j, k);
        b.element(i - 1, k) =
            (b.element(i - 1, k) - sum) / lu.element(i - 1, i - 1);
      }
    }
    return b;
  }

  /*
  * LU Inverter
  * -----------
  * Returns the inverse of a given matrix via the LU decomposition.
  * 
  * @params: A, b, given matrices
  */
  template <typename Num>
  matrix<Num> inverse_lu(matrix<Num> A)
  {
    return linear_solve_lu(A, identity<Num>(A.height(), A.height()));
  }

  
  /*
  * Least squares solver
  * --------------------
  * Solves the overdetermined system Ax = b with modified Gram-Schmidt
  * orthonormalization.
  * 
  * @params: A, b, given matrices
  */
  template <typename Num>
  matrix<Num> least_squares_gs(const matrix<Num> &A, const matrix<Num> &b)
  {
    matrix<Num> full(A.height(), A.width() + 1);
    for (size_t i = 0; i < A.height(); ++i)
    {
      for (size_t j = 0; j < A.width(); ++j)
        full.element(i, j) = A.element(i, j);
      full.element(i, A.width()) = b.element(i, 0);
    }

    matrix<Num> result(A.width(), 1);
    auto qu = qu_decomposition_gs(full);
    for (size_t i = 0; i < result.height(); ++i)
      result.element(i, 0) = qu.second.element(i, A.width());

    for (size_t i = A.width(); i >= 1; --i)
    {
      Num sum = 0;
      for (size_t j = i; j < A.width(); j++)
        sum += qu.second.element(i - 1, j) * result.element(j, 0);
      result.element(i - 1, 0) =
          (result.element(i - 1, 0) - sum) / qu.second.element(i - 1, i - 1);
    }
    return result;
  }

  /*
  * Givens left
  * -----------
  * Multiplies the matrix from the left with the conjugate transpose of the
  * Givens rotation parameterized by c and s.
  * 
  * @params: A, a given matrix
  */
  template <typename Num>
  void givens_left(matrix<Num> &A, size_t i1, size_t i2, Num c, Num s)
  {
    for (size_t i = 0; i < A.width(); ++i)
    {
      Num a = A.element(i1, i);
      Num b = A.element(i2, i);
      A.element(i1, i) = c * a - s * b;
      A.element(i2, i) = std::conj(s) * a + std::conj(c) * b;
    }
  }

  /*
   * Givens right
   * ------------
   * Multiplies the matrix from the right with the conjugate transpose of the
   * Givens rotation parameterized by c and s.
   */
  template <typename Num>
  void givens_right(matrix<Num> &A, size_t i1, size_t i2, Num c, Num s)
  {
    for (size_t i = 0; i < A.height(); ++i)
    {
      Num a = A.element(i, i1);
      Num b = A.element(i, i2);
      A.element(i, i1) = std::conj(c) * a - std::conj(s) * b;
      A.element(i, i2) = s * a + c * b;
    }
  }

  /*
   * Givens rotation
   * ---------------
   * Applies the Givens rotation specified by elements A[i1, col] and A[i2, col]
   * to A from the left side and its inverse to Q from the right side.
   *
   * @params: matrices A, Q & hyperparameters
   */
  template <typename Num>
  void givens_rotate(matrix<Num> &Q, matrix<Num> &A, size_t i1, size_t i2,
                     size_t col)
  {
    Num i1_cols = A.element(i1, col);
    Num i2_cols = A.element(i2, col);
    Num dist = std::sqrt(std::norm(i1_cols) + std::norm(i2_cols));
    Num normalized_i1_cols = std::conj(i1_cols) / dist;
    Num normalized_i2_cols = -std::conj(i2_cols) / dist;

    givens_left(A, i1, i2, normalized_i1_cols, normalized_i2_cols);
    givens_right(Q, i1, i2, normalized_i1_cols, normalized_i2_cols);
  }

  /*
   * Matrix equivalent via Givens
   * ----------------------------
   * Transforms the matrix into an equivalent one by conjugating with a
   * Givens rotation specified by elements A[i1, cols] and A[i2, cols].
   */
  template <typename Num>
  void givens_equivalent(matrix<Num> &A, size_t i1, size_t i2, size_t cols)
  {
    givens_rotate(A, A, i1, i2, cols);
  }

  /* QU decomposition via Givens
   * ---------------------------
   * Returns the QU decomposition of a matrix via Givens rotations.
   *
   * @params: A, given matrix
   */
  template <typename Num>
  std::pair<matrix<Num>, matrix<Num>> qu_decomposition_givens(matrix<Num> A)
  {
    matrix<Num> Q = identity<Num>(A.height(), A.height());
    for (size_t j = 0; j < A.width(); ++j)
      for (size_t i = A.height() - 1; i >= j + 1; --i)
        givens_rotate(Q, A, i - 1, i, j);
    return {Q, A};
  }

  /* Upper Hessenberg
   * ----------------
   * Returns a similar but reduced upper-Hessenberg matrix with Givens
   * rotations, given a matrix.
   *
   * @params: A, given matrix
   */
  template <typename Num>
  matrix<Num> upper_hessenberg(matrix<Num> A)
  {
    for (size_t j = 0; j < A.width(); ++j)
      for (size_t i = A.height() - 1; i >= j + 2; --i)
        givens_equivalent(A, i - 1, i, j);
    return A;
  }

  /* Power iteration
   * ----------------
   * Returns the eigenvector corresponding to the dominant eigenvalue of a
   * square matrix A.
   *
   * @params: A, given diagonlizable matrix
   * @params: approx_vect, approximation to dominant eigenvector
   */
  template <typename Num>
  matrix<Num> power_iteration(const matrix<Num> &A, matrix<Num> approx_vect)
  {
    for (;;)
    {
      auto new_vect = A * approx_vect;
      normalize(new_vect);
      if (distance(approx_vect, new_vect) < EPS)
        return new_vect;
      swap(approx_vect, new_vect);
    }
  }
}

/*
 * Sample proof of concept.
 */

using namespace std;

int main()
{
  auto A = eigencipy::random<complex<double>>(10, 10);
  auto qu1 = eigencipy::qu_decomposition_gs(A);
  auto qu2 = eigencipy::qu_decomposition_givens(A);
  cout << A << endl
       << endl;

  cout << "Gram-Schmidt: \n";
  cout << qu1.first << std::endl
       << std::endl
       << qu1.second << std::endl
       << std::endl
       << qu1.first * qu1.second - A << std::endl
       << endl;
  cout << qu1.first * conjugate_transpose(qu1.first) << endl
       << endl;

  cout << "Givens: \n";
  cout << qu2.first << std::endl
       << std::endl
       << qu2.second << std::endl
       << std::endl
       << qu2.first * qu2.second - A << std::endl
       << endl;
  cout << qu2.first * conjugate_transpose(qu2.first) << endl
       << endl;
}