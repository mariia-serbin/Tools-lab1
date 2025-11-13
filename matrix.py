"""!
@file matrix.py
@brief Matrix operations module.
@details Contains functions for basic and advanced matrix operations, including addition, subtraction,
         scalar multiplication, matrix multiplication, transposition, row operations (swap, scale, add),
         determinant, inverse, and rank calculation. Provides proper exception handling for invalid inputs
         such as empty matrices, non-square matrices, singular matrices, and out-of-range row indices.
@author
Maria Serbin
@date
11.11.2025
"""
from typing import List

EPSILON = 1e-9

class Matrix: 
    """!
    @brief Program implementation of matrix operations.
    
    @details This class provides the program implementation of both basic and advanced matrix operations,
             including addition, subtraction, scalar and matrix multiplication, as well as fundamental row operations
             (swapping, scaling, and row addition). It also supports more advanced computations such as determining
             the rank, determinant, and inverse of a matrix.

             Each method in this class is defined as a static method, meaning you do not need to create
             an instance of the Matrix class — simply call the methods directly with matrices as input arguments.
    """
    @staticmethod
    ##@brief Performs addition of two matrices with the same dimensions.

    #@param a The first matrix to add, of size m×n. Elements can be integers or floating-point numbers,
             #including zero and negative values.
    # @param b The second matrix to add, of size m×n. Elements can be integers or floating-point numbers,
    #          including zero and negative values.
    # @return A new matrix of the same dimensions as the input matrices, where each element
    #         C[i][j] = A[i][j] + B[i][j] (the sum of the corresponding elements of the input matrices).
    # @throws ValueError If the matrices have different dimensions or if one or both matrices are empty.
    #         An empty matrix is defined as having one or more empty rows, or no rows at all.
    # @example
    # add([[1,2],[3,4]], [[5,6],[7,8]]) -> [[6,8],[10,12]]
    def add(a: List[List[float|int]], b: List[List[float|int]]) -> List[List[float|int]]:

        if not a or not b or any(len(row) == 0 for row in a) or any(len(row) == 0 for row in b):
            raise ValueError('Matrices are empty')
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            raise ValueError('Matrices should have the same dimensions')
        n, m = len(a), len(a[0])

        return [[a[i][j] + b[i][j] for j in range(m)] for i in range(n)]

    @staticmethod
    def subtract(a: List[List[float|int]], b: List[List[float|int]]) -> List[List[float|int]]:
        """!
    @brief Performs subtraction of two matrices of the same dimensions.

    @param a The first matrix (minuend), of size m×n. Elements can be integers or floating-point numbers,
             including zero and negative values.
    @param b The second matrix (subtrahend), of size m×n. Elements can be integers or floating-point numbers,
             including zero and negative values.
    @return A new matrix of the same dimensions where each element
            C[i][j] = A[i][j] - B[i][j] (the difference of the corresponding elements of the input matrices).
    @throws ValueError If the matrices have different dimensions or if one or both matrices are empty.
            An empty matrix is defined as having one or more empty rows, or no rows at all.
    @example
    subtract([[5,6],[7,8]], [[1,2],[3,4]]) -> [[4,4],[4,4]]
    """
        if not a or not b or any(len(row) == 0 for row in a) or any(len(row) == 0 for row in b):
            raise ValueError('Matrices are empty')
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            raise ValueError('Matrices should have the same dimensions')
        n, m = len(a), len(a[0])
        return [[a[i][j] - b[i][j] for j in range(m)] for i in range(n)]

    @staticmethod
    def multiply_scalar(a: List[List[float|int]], num: float|int) -> List[List[float|int]]:
        """!
    @brief Multiplies every element of a matrix by a scalar value.

    @param a The matrix to multiply, of size m×n. Each element can be an integer or floating-point number,
             including zero and negative values. The matrix must not be empty (cannot have zero rows or empty rows).
    @param num The scalar multiplier. Can be any integer or floating-point number, including zero or negative numbers.
             Each element of the matrix will be multiplied by this scalar.
    @return A new matrix of the same dimensions as the input matrix, where each element
            C[i][j] = A[i][j] * num.
    @throws ValueError If the matrix is empty (has zero rows or one or more empty rows).
            An empty matrix is defined as having one or more empty rows, or no rows at all.
    @example
    multiply_scalar([[1,2],[3,4]], 2) -> [[2,4],[6,8]]
    multiply_scalar([[1.5,2.5],[3.0,4.0]], 0.5) -> [[0.75,1.25],[1.5,2.0]]
    multiply_scalar([[1,2],[3,4]], -1) -> [[-1,-2],[-3,-4]]
    """
        if not a or any(len(row) == 0 for row in a):
            raise ValueError('Matrix is empty')
        return [[x * num for x in row] for row in a]

    @staticmethod
    def multiply(a: List[List[float|int]], b: List[List[float|int]]) -> List[List[float|int]]:
        """!
    @brief Performs matrix multiplication of two matrices (A * B).

    @param a The first matrix (A) of size n×m. Each element can be an integer or floating-point number,
             including zero and negative values. The matrix must not be empty (cannot have zero rows or empty rows).
    @param b The second matrix (B) of size m×p. Each element can be an integer or floating-point number,
             including zero and negative values. The matrix must not be empty (cannot have zero rows or empty rows).
             The number of rows in B must equal the number of columns in A (m) for multiplication to be valid.
    @return A new matrix of size n×p where each element
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(m)).
            Each element of the result is calculated as the sum of products of the corresponding row from A
            and column from B.
    @throws ValueError If the matrices have incompatible dimensions for multiplication,
            or if one or both matrices are empty.
            An empty matrix is defined as having zero rows or one or more empty rows.
    @example
    multiply([[1,2],[3,4]], [[5,6],[7,8]]) -> [[19,22],[43,50]]
    multiply([[1,0,2],[0,1,3]], [[1,2],[3,4],[5,6]]) -> [[11,16],[18,22]]
    """
        if not a or not b or any(len(row) == 0 for row in a) or any(len(row) == 0 for row in b):
            raise ValueError('Matrices cannot be empty for multiplying them')
        n_a, m_a = len(a), len(a[0])
        n_b, m_b = len(b), len(b[0])
        if m_a != n_b:
            raise ValueError('Matrices cannot be multiplied: incompatible sizes')
        return [[sum(a[i][k] * b[k][j] for k in range(m_a)) for j in range(m_b)] for i in range(n_a)]

    @staticmethod
    def transpose(a: List[List[float|int]]) -> List[List[float|int]]:
        """!
    @brief Returns the transposed matrix of the input matrix (rows become columns and columns become rows).

    @param a The matrix to transpose, of size m×n. Each element can be an integer or floating-point number,
             including zero and negative values. The matrix must not be empty.
    @return A new matrix of size n×m where each element
            C[i][j] = A[j][i], effectively swapping rows and columns.
    @throws ValueError If the matrix is empty. An empty matrix is defined as having zero rows or one or more empty rows.
    @example
    transpose([[1,2],[3,4]]) -> [[1,3],[2,4]]
    transpose([[1,2,3],[4,5,6]]) -> [[1,4],[2,5],[3,6]]
    """
        if not a or any(len(row) == 0 for row in a):
            raise ValueError('Matrix is empty or has empty rows')
        return [list(row) for row in zip(*a)]

    @staticmethod
    def swap_rows(a: List[List[float|int]], i: int, j: int) -> List[List[float|int]]:
        """!
    @brief Swaps two rows of a matrix in place.

    @param a The matrix to modify, of size m×n. Each element can be an integer or floating-point number,
             including zero and negative values. The matrix must not be empty.
    @param i The index of the first row to swap (0-based, i.e., the first row is at index 0).
    @param j The index of the second row to swap (0-based, i.e., the first row is at index 0).
    @throws ValueError If the matrix is empty. An empty matrix is defined as having zero
            rows or one or more empty rows.
    @throws IndexError If either row index is out of range (less than 0 or greater than the number of rows minus one).
    @example
    swap_rows([[1,2],[3,4]], 0, 1) -> [[3,4],[1,2]]
    swap_rows([[1,2,3],[4,5,6],[7,8,9]], 0, 2) -> [[7,8,9],[4,5,6],[1,2,3]]
    """
        if not a or any(len(row) == 0 for row in a):
            raise ValueError('Matrix is empty')
        if i >= len(a) or j >= len(a) or i < 0 or j < 0:
            raise IndexError('Indexes out of range')
        a[i], a[j] = a[j], a[i]

    @staticmethod
    def scale_rows(a: List[List[float|int]], i: int, factor: float|int) -> List[List[float|int]]:
        """!
    @brief Multiplies a single row of a matrix by a scalar factor in place.

    @param a The matrix to modify, of size m×n. Each element can be an integer or floating-point number,
             including zero and negative values. The matrix must not be empty.
    @param i The index of the row to scale (0-based, i.e., the first row is at index 0).
    @param factor The scalar multiplier. Can be any integer or floating-point number, except zero.
             Each element of the row will be multiplied by this factor.
    @throws ValueError If the matrix is empty, or if factor is zero.
            An empty matrix is defined as having zero rows or one or more empty rows.
    @throws IndexError If the row index is out of range (less than 0 or greater than the number of rows minus one).
    @example
    scale_rows([[1,2],[3,4]], 0, 2) -> [[2,4],[3,4]]
    scale_rows([[1,2,3],[4,5,6]], 1, -1) -> [[1,2,3],[-4,-5,-6]]
    """
        if not a or any(len(row) == 0 for row in a):
            raise ValueError('Matrix is empty')
        if i >= len(a) or i < 0:
            raise IndexError('Index out of range')
        if abs(factor) < EPSILON:
            raise ValueError('You cannot multiply rows by zero')
        a[i] = [x * factor for x in a[i]]

    @staticmethod
    def add_rows(a: List[List[float|int]], i: int, j: int, factor: float|int = 1) -> List[List[float|int]]:
        """!
    @brief Adds a multiple of one row to another row in a matrix (row_i += row_j * factor).

    @param a The matrix to modify, of size m×n. Each element can be an integer or floating-point number,
             including zero and negative values. The matrix must not be empty.
    @param i The index of the row to which another row will be added (0-based, i.e., the first row is at index 0).
    @param j The index of the row to multiply by factor and add to row i (0-based).
    @param factor The scalar multiplier applied to row j before adding to row i. Defaults to 1.
    @throws ValueError If the matrix is empty, or if factor is zero.
            An empty matrix is defined as having zero rows or one or more empty rows.
    @throws IndexError If either row index is out of range (less than 0 or greater than the number of rows minus one).
    @example
    add_rows([[1,2],[3,4]], 0, 1) -> [[4,6],[3,4]]  # factor defaults to 1
    add_rows([[1,2],[3,4]], 0, 1, 2) -> [[7,10],[3,4]]  # factor = 2
    """
        if not a or any(len(row) == 0 for row in a):
            raise ValueError('Matrix is empty')
        if i >= len(a) or j >= len(a) or i < 0 or j < 0:
            raise IndexError('Index out of range')
        if abs(factor) < EPSILON:
            raise ValueError('You cannot multiply rows by zero')
        a[i] = [a[i][k] + a[j][k] * factor for k in range(len(a[i]))]

    @staticmethod
    def det(a: List[List[float|int]]) -> float|int:
        """!
    @brief Computes the determinant of a square matrix.

    @param a The square matrix of size n×n. Each element can be an integer or floating-point number,
             including zero and negative values. The matrix must not be empty.
    @return The determinant of the matrix as a single number (integer or float).
    @throws ValueError If the matrix is empty or not square.
            An empty matrix is defined as having zero rows or one or more empty rows.
            A non-square matrix is defined as having a number of rows not equal to the number of columns.
    @example
    det([[1,2],[3,4]]) -> -2
    det([[2,0,1],[1,1,0],[3,2,1]]) -> 3
    """
        if not a or any(len(row) == 0 for row in a):
            raise ValueError('Matrix is empty')
        n = len(a)
        if any(len(row) != n for row in a):
            raise ValueError('Matrix has to be square')
        copy = [row[:] for row in a]
        det_val = 1
        swaps = 0
        for i in range(n):
            pivot_row = max(range(i, n), key=lambda r: abs(copy[r][i]))
            if abs(copy[pivot_row][i]) < EPSILON:
                return 0
            if pivot_row != i:
                copy[i], copy[pivot_row] = copy[pivot_row], copy[i]
                swaps += 1
            for j in range(i + 1, n):
                factor = copy[j][i] / copy[i][i]
                for k in range(i, n):
                    copy[j][k] -= factor * copy[i][k]
        for i in range(n):
            det_val *= copy[i][i]
        det_val *= (-1) ** swaps
        return det_val

    @staticmethod
    def inverse(a: List[List[float|int]]) -> List[List[float|int]]:
        """!
    @brief Computes the inverse of a square matrix using the Gauss-Jordan elimination method.

    @param a The square matrix of size n×n. Each element can be an integer or floating-point number,
             including zero and negative values. The matrix must not be empty.
    @return A new matrix of size n×n representing the inverse of the input matrix.
    @throws ValueError If the matrix is empty, not square, or singular (determinant is zero).
            An empty matrix is defined as having zero rows or one or more empty rows.
            A non-square matrix is defined as having a number of rows not equal to the number of columns.
            A singular matrix is one that cannot be inverted because its determinant is zero.
    @details The function performs inversion using the Gauss-Jordan elimination algorithm:
             1. The matrix is augmented with the identity matrix of the same size.
             2. For each pivot row, the row with the largest absolute value in the current column
                is swapped to the pivot position (partial pivoting).
             3. The pivot row is normalized so that the pivot element becomes 1.
             4. Other rows are updated to make all elements in the pivot column zero.
             5. After processing all rows, the right half of the augmented matrix becomes the inverse.
    @example
    inverse([[1,2],[3,4]]) -> [[-2.0, 1.0],[1.5, -0.5]]
    inverse([[2,0,1],[1,1,0],[3,2,1]]) -> [[-2.0, 1.0, 1.0],[1.0, 0.0, 0.0],[1.0, -1.0, 0.0]]
    """
        if not a or any(len(row) == 0 for row in a):
            raise ValueError('Matrix is empty')
        n = len(a)
        if any(len(row) != n for row in a):
            raise ValueError('Matrix must be square')
        copy = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(a)]
        for i in range(n):
            pivot_row = max(range(i, n), key=lambda r: abs(copy[r][i]))
            if abs(copy[pivot_row][i]) < EPSILON:
                raise ValueError('Matrix is singular and cannot be inverted')
            if pivot_row != i:
                copy[i], copy[pivot_row] = copy[pivot_row], copy[i]
            factor = copy[i][i]
            copy[i] = [x / factor for x in copy[i]]
            for j in range(n):
                if j != i:
                    factor2 = copy[j][i]
                    copy[j] = [copy[j][k] - factor2 * copy[i][k] for k in range(2*n)]
        return [row[n:] for row in copy]

    @staticmethod
    def rank(a: List[List[float|int]]) -> int:
        """!
    @brief Computes the rank of a matrix using Gaussian elimination.

    @param a The matrix of size n×m. Each element can be an integer or floating-point number,
             including zero and negative values. The matrix must not be empty.
    @return The rank of the matrix as an integer, representing the number of linearly independent rows.
    @throws ValueError If the matrix is empty. An empty matrix is defined as having zero rows or one or more empty rows.
    @details The function determines the rank by performing Gaussian elimination:
             1. Iterate over each column up to the minimum of the number of rows and columns.
             2. Find a pivot row with a non-zero element in the current column.
             3. Swap the pivot row with the current row if necessary.
             4. Eliminate the current column entries in all rows below the pivot.
             5. Count the number of non-zero pivot rows, which equals the rank of the matrix.
    @example
    rank([[1,2],[3,4]]) -> 2
    rank([[1,2,3],[2,4,6],[3,6,9]]) -> 1
    rank([[1,0,0],[0,1,0],[0,0,1]]) -> 3
    """
        if not a or any(len(row) == 0 for row in a):
            raise ValueError('Matrix cannot be empty')
        n, m = len(a), len(a[0])
        copy = [row[:] for row in a]
        rank_val = 0
        for i in range(min(n, m)):
            pivot_row = -1
            for j in range(i, n):
                if abs(copy[j][i]) > EPSILON:
                    pivot_row = j
                    break
            if pivot_row == -1:
                continue
            if pivot_row != i:
                copy[i], copy[pivot_row] = copy[pivot_row], copy[i]
            for j in range(i + 1, n):
                if abs(copy[i][i]) < EPSILON:
                    continue
                factor = copy[j][i] / copy[i][i]
                copy[j] = [copy[j][k] - factor * copy[i][k] for k in range(m)]
            rank_val += 1
        return rank_val
