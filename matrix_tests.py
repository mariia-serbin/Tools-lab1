"""
@file test_matrix_operations.py
@brief Unit tests for matrix operations module.
@details Contains unit tests for all functions in the matrix module,
         including addition, subtraction, multiplication, scalar multiplication,
         transposition, row operations, determinant, inverse, and rank calculation.
         Ensures correct results and proper exception handling for invalid inputs such as
         empty matrices, non-square matrices,
         singular matrices, and out-of-range indices..
@note These tests use Python's unittest framework.
@author
Maria Serbin
@date
11.11.2025
"""
import matrix
import unittest

class MatrixTests(unittest.TestCase):
    """
    @brief Class for testing all matrix functions from matrix.py.

    This class contains tests for addition, subtraction, scalar multiplication, matrix multiplication,
    transposition, row operations, determinant, inverting, and rank calculation.

    Different types of matrices are initialized in setUp() to be used in tests.
    """

    def setUp(self):
        """
        @brief Initializes matrices and constants for testing.

        Initializes:
            - matrix_same_dim_a, matrix_same_dim_b: matrices for addition and subtraction tests
            - matrix_INV: invertible matrix used for determinant, inverse, rank, and row operation tests
            - matrix_SINGULAR: singular matrix used for testing edge cases and negative scenarios
              (inverse cannot be computed, determinant may be zero, etc.)
            - I: identity matrix
            - Z: zero matrix
            - to_mul1, to_mul2: matrices for multiplication tests
            - factor, integer: scalars for scalar multiplication tests
        """
        self.matrix_same_dim_a =[[1, 9, 6],
                                [2, 8, 10]]
        self.matrix_same_dim_b = [[5, 3, 1],
                                  [0, 2, 4]]

        self.matrix_INV = [[1, 4, 3],
                           [-2, 3, 9],
                           [5, 6, 3]]
        self.matrix_SINGULAR =[[1, 2, 3],
                               [3, 6, 9],
                               [4, 5, 2]]

        self.I = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]
        self.Z = [[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]]

        self.to_mul1 = [[1, 4, 3],
                        [5, 8, 1]]
        self.to_mul2 =[[3, 1, 0, 5],
                       [2, 2, 1, 2],
                       [1, 3, 2, -2]]

        self.factor = 1/2
        self.integer = 3

    def assertMatrixAlmostEqual(self, res, expected, places=7):
        """
        @brief Helper method to assert that two matrices with floating-point numbers are almost equal.

        @param res The resulting matrix obtained from a function under test.
        @param expected The expected matrix to compare against.
        @param places The number of decimal places to use for comparison (default is 7).

        @throws AssertionError If the matrices have different dimensions or if any element differs
               by more than the specified number of decimal places.

        @details
        Compares each element of two matrices element-wise using self.assertAlmostEqual.
        First checks that the dimensions (number of rows and columns) match, then compares elements.
        """
        self.assertEqual(len(res), len(expected))
        for i in range(len(res)):
            self.assertEqual(len(res[i]), len(expected[i]))
            for j in range(len(res[i])):
                self.assertAlmostEqual(res[i][j], expected[i][j], places=places)

#-------------testing add function----------------

    def test_add(self):
        """
        @brief Tests addition of two matrices with identical dimensions.

        @details Checks that the add() function correctly sums corresponding elements
                 of two matrices of the same size and returns the expected result.

        @see matrix.add
        """
        res = matrix.add(self.matrix_same_dim_a, self.matrix_same_dim_b)
        expected = [[6, 12, 7],
                    [2, 10, 14]]
        self.assertEqual(res, expected)

    def test_add_empty(self):
        """
        @brief Tests handling of empty matrices in the add() function.

        @details Verifies that ValueError is raised when one or both input matrices are empty.

        @throws ValueError If matrices are empty.
        @see matrix.add
        """
        a = [[]]
        b = []
        with self.assertRaises(ValueError) as cm:
            matrix.add(a, b)
        self.assertEqual(str(cm.exception), 'Matrices are empty')

        a = self.matrix_same_dim_a
        b = []
        with self.assertRaises(ValueError) as cm:
            matrix.add(a, b)
        self.assertEqual(str(cm.exception), 'Matrices are empty')

        a = [[]]
        b = [[]]
        with self.assertRaises(ValueError) as cm:
            matrix.add(a, b)
        self.assertEqual(str(cm.exception), 'Matrices are empty')

    def test_add_not_same(self):
        """
        @brief Tests add() with matrices of different dimensions.

        @details Ensures that the function raises ValueError when matrix dimensions differ.

        @throws ValueError If matrices have different dimensions.
        @see matrix.add
        """
        with self.assertRaises(ValueError) as cm:
            matrix.add(self.matrix_same_dim_a, self.matrix_INV)
        self.assertEqual(str(cm.exception), 'Matrices should have the same dimensions')
#---------------testing substraction for matrices---------------
    def test_sub(self):
        """
        @brief Tests subtraction of two matrices with identical dimensions.

        @details Checks that the subtract() function correctly computes the element-wise
                 difference of two matrices of the same size.

        @see matrix.subtract
        """
        a = self.matrix_same_dim_a
        b = self.matrix_same_dim_b
        expected = [[-4, 6, 5],
                    [2, 6, 6]]
        res = matrix.subtract(a, b)
        self.assertEqual(res, expected)

    def test_sub_empty(self):
        """
        @brief Tests handling of empty matrices in the subtract() function.

        @details Verifies that ValueError is raised when one or both input matrices are empty.

        @throws ValueError If matrices are empty.
        @see matrix.subtract
        """
        a = [[]]
        b = []
        with self.assertRaises(ValueError) as cm:
            matrix.subtract(a, b)
        self.assertEqual(str(cm.exception), 'Matrices are empty')

        a = self.matrix_same_dim_a
        b = []
        with self.assertRaises(ValueError) as cm:
            matrix.subtract(a, b)
        self.assertEqual(str(cm.exception), 'Matrices are empty')

        a = [[]]
        b = [[]]
        with self.assertRaises(ValueError) as cm:
            matrix.subtract(a, b)
        self.assertEqual(str(cm.exception), 'Matrices are empty')

    def test_sub_not_same(self):
        """
        @brief Tests subtract() with matrices of different dimensions.

        @details Ensures that the function raises ValueError when matrix dimensions differ.

        @throws ValueError If matrices have different dimensions.
        @see matrix.subtract
        """
        with self.assertRaises(ValueError) as cm:
            matrix.subtract(self.matrix_same_dim_a, self.matrix_INV)
        self.assertEqual(str(cm.exception), 'Matrices should have the same dimensions')

#--------------testing multiplication by scalar-----------
    def test_mul_scalar(self):
        """
        @brief Tests multiplication of a matrix by a scalar.

        @details Verifies that multiply_scalar() correctly multiplies all elements of a matrix
                 by a given scalar, including integer and floating-point factors.

        @see matrix.multiply_scalar
        """
        a = self.matrix_SINGULAR
        num = self.integer
        expected = [[3, 6, 9],
                    [9, 18, 27],
                    [12, 15, 6]]
        res = matrix.multiply_scalar(a, num)
        self.assertMatrixAlmostEqual(res, expected)

        num = self.factor
        expected = [[0.5, 1, 1.5],
                    [1.5, 3, 4.5],
                    [2, 2.5, 1]]
        res = matrix.multiply_scalar(a, num)
        self.assertMatrixAlmostEqual(res, expected)

    def test_mul_scalar_zero(self):
        """
        @brief Tests multiplication of a matrix by zero.

        @details Ensures that multiplying any matrix by zero returns a zero matrix.

        @see matrix.multiply_scalar
        """
        num = 0
        a = self.matrix_SINGULAR
        expected = self.Z
        res = matrix.multiply_scalar(a, num)
        self.assertMatrixAlmostEqual(res, expected)

        a = self.Z
        num = self.integer
        expected = self.Z
        res = matrix.multiply_scalar(a, num)
        self.assertMatrixAlmostEqual(res, expected)

    def test_mul_scalar_empty(self):
        """
        @brief Tests handling of empty matrices in multiply_scalar().

        @details Verifies that ValueError is raised when the input matrix is empty
                 or contains empty rows.

        @throws ValueError If the matrix is empty.
        @see matrix.multiply_scalar
        """
        a = []
        num = self.integer
        with self.assertRaises(ValueError) as cm:
            matrix.multiply_scalar(a, num)
        self.assertEqual(str(cm.exception), 'Matrix is empty')

        a = [[1], []]
        num = self.factor
        with self.assertRaises(ValueError) as cm:
            matrix.multiply_scalar(a, num)
        self.assertEqual(str(cm.exception), 'Matrix is empty')
#--------testing multiplication of 2 matrices---------------
    def test_mul(self):
        """
        @brief Tests multiplication of two matrices with compatible dimensions.

        @details Verifies that multiply() correctly computes the matrix product
                 when the number of columns in the first matrix equals the number
                 of rows in the second matrix. Also checks multiplication by the identity matrix.

        @see matrix.multiply
        """
        a = self.to_mul1
        b = self.to_mul2
        expected = [[14, 18, 10, 7],
                    [32, 24, 10, 39]]
        res = matrix.multiply(a, b)
        self.assertEqual(res, expected)

        a = self.matrix_INV
        b = self.I
        expected = self.matrix_INV
        res = matrix.multiply(a, b)
        self.assertEqual(res, expected)

    def test_mul_empty(self):
        """
        @brief Tests handling of empty matrices in multiply().

        @details Ensures that ValueError is raised when one or both input matrices
                 are empty or contain empty rows.

        @throws ValueError If matrices are empty or contain empty rows.
        @see matrix.multiply
        """
        a = [[]]
        b = []
        with self.assertRaises(ValueError) as cm:
            matrix.multiply(a, b)
        self.assertEqual(str(cm.exception), 'Matrices cannot be empty for multiplying them')

        a = self.matrix_INV
        b = []
        with self.assertRaises(ValueError) as cm:
            matrix.multiply(a, b)
        self.assertEqual(str(cm.exception), 'Matrices cannot be empty for multiplying them')

        a = [[12], [0], []]
        b = [[1],
             [3],
             [0]]
        with self.assertRaises(ValueError) as cm:
            matrix.multiply(a, b)
        self.assertEqual(str(cm.exception), 'Matrices cannot be empty for multiplying them')

    def test_mul_not_appropriate(self):
        """
        @brief Tests multiply() with incompatible matrices.

        @details Verifies that ValueError is raised when the number of columns
                 in the first matrix does not equal the number of rows in the second matrix.

        @throws ValueError If matrices cannot be multiplied due to incompatible sizes.
        @see matrix.multiply
        """
        a = self.to_mul1
        b = self.matrix_same_dim_b
        with self.assertRaises(ValueError) as cm:
            matrix.multiply(a, b)
        self.assertEqual(str(cm.exception), 'Matrices cannot be multiplied: incompatible sizes')

#-------testing transposing of matrix----------

    def test_transpose(self):
        """
        @brief Tests transposition of a matrix.

        @details Verifies that transpose() correctly flips rows and columns
                 of a non-empty matrix.

        @see matrix.transpose
        """
        a = self.matrix_same_dim_a
        expected = [[1, 2],
                    [9, 8],
                    [6, 10]]
        res = matrix.transpose(a)
        self.assertEqual(res, expected)

    def test_transpose_empty(self):
        """
        @brief Tests handling of empty matrices in transpose().

        @details Ensures that ValueError is raised when the input matrix is empty
                 or contains empty rows.

        @throws ValueError If the matrix is empty or has empty rows.
        @see matrix.transpose
        """
        a = [[]]
        with self.assertRaises(ValueError) as cm:
            matrix.transpose(a)
        self.assertEqual(str(cm.exception), 'Matrix is empty or has empty rows')

        a = [[12], []]
        with self.assertRaises(ValueError) as cm:
            matrix.transpose(a)
        self.assertEqual(str(cm.exception), 'Matrix is empty or has empty rows')

#-----------testing operations on rows of matrix-----------

    def test_swap(self):
        """
        @brief Tests swapping of two rows in a matrix.

        @details Verifies that swap_rows() correctly swaps the specified rows in a non-empty matrix.

        @see matrix.swap_rows
        """
        a = [row[:] for row in self.matrix_INV]
        i, j = 1, 2
        expected = [[1, 4, 3],
                    [5, 6, 3],
                    [-2, 3, 9]]
        res = matrix.swap_rows(a, i, j)
        self.assertIsNone(res)
        self.assertEqual(a, expected)

    def test_swap_empty(self):
        """
        @brief Tests swapping rows in an empty matrix.

        @details Ensures that ValueError is raised when the input matrix is empty.

        @throws ValueError If the matrix is empty.
        @see matrix.swap_rows
        """
        a = [[]]
        with self.assertRaises(ValueError) as cm:
            matrix.swap_rows(a, 0, 1)
        self.assertEqual(str(cm.exception), 'Matrix is empty')

    def test_swap_not_exists(self):
        """
        @brief Tests swapping rows with invalid indices.

        @details Verifies that IndexError is raised when the specified row indices
                 are out of range.

        @throws IndexError If row indices are out of range.
        @see matrix.swap_rows
        """
        i, j = 1, 2
        a = self.matrix_same_dim_a
        with self.assertRaises(IndexError) as cm:
            matrix.swap_rows(a, i, j)
        self.assertEqual(str(cm.exception), 'Indexes out of range')

    def test_scale(self):
        """
        @brief Tests scaling a row of a matrix by a scalar factor.

        @details Verifies that scale_rows() correctly multiplies all elements of
                 the specified row by the given factor, including integer and float factors.

        @see matrix.scale_rows
        """
        a = [row[:] for row in self.I]
        factor = self.integer
        i = 1
        expected = [[1, 0, 0],
                    [0, 3, 0],
                    [0, 0, 1]]
        res = matrix.scale_rows(a, i, factor)
        self.assertIsNone(res)
        self.assertMatrixAlmostEqual(a, expected)

        a  = [row[:] for row in self.matrix_INV]
        factor = self.integer
        expected = [[1, 4, 3],
                    [-6, 9, 27],
                    [5, 6, 3]]
        res = matrix.scale_rows(a, i, factor)
        self.assertIsNone(res)
        self.assertMatrixAlmostEqual(a, expected)

        a = [row[:] for row in self.I]
        factor = self.factor
        expected = [[1, 0, 0],
                    [0, 0.5, 0],
                    [0, 0, 1]]
        res = matrix.scale_rows(a, i, factor)
        self.assertIsNone(res)
        self.assertMatrixAlmostEqual(a, expected)

    def test_scale_empty(self):
        """
        @brief Tests scaling a row in an empty matrix.

        @details Ensures that ValueError is raised when the matrix is empty or contains empty rows.

        @throws ValueError If the matrix is empty.
        @see matrix.scale_rows
        """
        a = [[]]
        with self.assertRaises(ValueError) as cm:
            matrix.scale_rows(a, 0, 1)
        self.assertEqual(str(cm.exception), 'Matrix is empty')

        a = [[12], [0], []]
        with self.assertRaises(ValueError) as cm:
            matrix.scale_rows(a, 0, 1)
        self.assertEqual(str(cm.exception), 'Matrix is empty')

    def test_scale_zero(self):
        """
        @brief Tests scaling a row by zero.

        @details Verifies that ValueError is raised when attempting to multiply a row by zero.

        @throws ValueError If the scaling factor is zero.
        @see matrix.scale_rows
        """
        a = self.I
        factor = 0
        i = 0
        with self.assertRaises(ValueError) as cm:
            matrix.scale_rows(a, i, factor)
        self.assertEqual(str(cm.exception), 'You cannot multiply rows by zero')

    def test_scale_not_exists(self):
        """
        @brief Tests scaling a non-existent row.

        @details Verifies that IndexError is raised when the specified row index
                 is out of range.

        @throws IndexError If row index is out of range.
        @see matrix.scale_rows
        """
        a = self.matrix_INV
        i = 5
        factor = self.integer
        with self.assertRaises(IndexError) as cm:
            matrix.scale_rows(a, i, factor)
        self.assertEqual(str(cm.exception), 'Index out of range')

    def test_add_rows(self):
        """
        @brief Tests adding one row multiplied by a factor to another row.

        @details Verifies that add_rows() correctly adds the scaled source row to the target row,
                 and returns None. Checks both with and without specifying a factor.

        @see matrix.add_rows
        """
        a = [row[:] for row in self.matrix_INV]
        i, j = 1, 2
        factor = self.integer
        expected = [[1, 4, 3],
                    [13, 21, 18],
                    [5, 6, 3]]
        res = matrix.add_rows(a, i, j, factor)
        self.assertIsNone(res)
        self.assertMatrixAlmostEqual(a, expected)

        a = [row[:] for row in self.matrix_INV]

        expected = [[1, 4, 3],
                    [3, 9, 12],
                    [5, 6, 3]]
        res = matrix.add_rows(a, i, j)
        self.assertIsNone(res)
        self.assertMatrixAlmostEqual(a, expected)

    def test_add_rows_empty(self):
        """
        @brief Tests adding rows in an empty matrix.

        @details Ensures that ValueError is raised when the matrix is empty.

        @throws ValueError If the matrix is empty.
        @see matrix.add_rows
        """
        a = [[]]
        i, j = 1, 2
        with self.assertRaises(ValueError) as cm:
            matrix.add_rows(a, i, j)
        self.assertEqual(str(cm.exception), 'Matrix is empty')

    def test_add_rows_zero_factor(self):
        """
        @brief Tests adding rows with a zero multiplication factor.

        @details Verifies that ValueError is raised when the factor is zero.

        @throws ValueError If factor is zero.
        @see matrix.add_rows
        """
        a = [row[:] for row in self.matrix_INV]

        with self.assertRaises(ValueError) as cm:
            matrix.add_rows(a, 0, 1, factor=0)
        self.assertEqual(str(cm.exception), 'You cannot multiply rows by zero')

    def test_add_rows_not_exists(self):
        """
        @brief Tests adding rows with invalid indices.

        @details Verifies that IndexError is raised when one of the specified row indices
                 does not exist in the matrix.

        @throws IndexError If row index is out of range.
        @see matrix.add_rows
        """
        a = self.matrix_INV
        i, j = 2, 5
        factor = self.factor
        with self.assertRaises(IndexError) as cm:
            matrix.add_rows(a, i, j, factor)
        self.assertEqual(str(cm.exception), 'Index out of range')

#-------------testing computing of determinant of matrix-----

    def test_det(self):
        """
        @brief Tests calculation of the determinant of a square matrix.

        @details Verifies that det() correctly computes the determinant
                 for square matrices, including the identity matrix.

        @see matrix.det
        """
        a = self.matrix_INV
        expected = 78
        res = matrix.det(a)
        self.assertAlmostEqual(res, expected)

        a = self.I
        expected = 1
        res = matrix.det(a)
        self.assertAlmostEqual(res, expected)

    def test_det_zero(self):
        """
        @brief Tests calculation of the determinant for a singular matrix.

        @details Ensures that det() correctly returns zero for singular matrices.

        @see matrix.det
        """
        a = self.matrix_SINGULAR
        expected = 0
        res = matrix.det(a)
        self.assertAlmostEqual(res, expected)

    def test_det_not_exists(self):
        """
        @brief Tests determinant calculation for a non-square matrix.

        @details Verifies that ValueError is raised when attempting to compute
                 the determinant of a non-square matrix.

        @throws ValueError If the matrix is not square.
        @see matrix.det
        """
        a = self.matrix_same_dim_a

        with self.assertRaises(ValueError) as cm:
            matrix.det(a)
        self.assertEqual(str(cm.exception), 'Matrix has to be square')

#---------tests inverting matrix----------------

    def test_inverse(self):
        """
        @brief Tests calculation of the inverse of a square matrix.

        @details Verifies that inverse() correctly computes the inverse of an invertible matrix.

        @see matrix.inverse
        """
        a = self.matrix_INV
        expected = [[-15/26, 1/13, 9/26],
                    [17/26, -2/13, -5/26],
                    [-9/26, 7/39, 11/78]]
        res = matrix.inverse(a)
        self.assertMatrixAlmostEqual(res, expected)

    def test_inverse_singular(self):
        """
        @brief Tests inversion of a singular matrix.

        @details Ensures that ValueError is raised when attempting to invert a singular matrix.

        @throws ValueError If the matrix is singular and cannot be inverted.
        @see matrix.inverse
        """
        a = self.matrix_SINGULAR
        with self.assertRaises(ValueError) as cm:
            matrix.inverse(a)
        self.assertEqual(str(cm.exception), 'Matrix is singular and cannot be inverted')

    def test_inverse_not_square(self):
        """
        @brief Tests inversion of a non-square matrix.

        @details Verifies that ValueError is raised when attempting to invert a non-square matrix.

        @throws ValueError If the matrix is not square.
        @see matrix.inverse
        """
        a = self.matrix_same_dim_a
        with self.assertRaises(ValueError) as cm:
            matrix.inverse(a)
        self.assertEqual(str(cm.exception), 'Matrix must be square')

#---------testing computing of rank----------

    def test_rank(self):
        """
        @brief Tests calculation of the rank of a matrix.

        @details Verifies that rank() correctly computes the rank for both
                 full-rank and singular matrices.

        @see matrix.rank
        """
        a = self.matrix_INV
        expected = 3
        res = matrix.rank(a)
        self.assertEqual(res, expected)

        a = self.matrix_SINGULAR
        expected = 2
        res = matrix.rank(a)
        self.assertEqual(res, expected)

    def test_rank_empty(self):
        """
        @brief Tests rank calculation for an empty matrix.

        @details Ensures that ValueError is raised when the input matrix is empty.

        @throws ValueError If the matrix is empty.
        @see matrix.rank
        """
        a = [[]]
        with self.assertRaises(ValueError) as cm:
            matrix.rank(a)
        self.assertEqual(str(cm.exception), 'Matrix cannot be empty')

    def tearDown(self):
        """
        @brief Cleans up resources after each test.

        @details Deletes all matrix and scalar attributes used in the tests to
                 ensure no state is carried over to the next test case.
        """
        del self.matrix_same_dim_a
        del self.matrix_same_dim_b
        del self.matrix_INV
        del self.I
        del self.Z
        del self.integer
        del self.factor
        del self.to_mul1
        del self.to_mul2

if __name__ == '__main__':
    unittest.main()
