from matrix import *
import unittest

class MatrixTests(unittest.TestCase):
    def setUp(self):
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
        self.assertEqual(len(res), len(expected))
        for i in range(len(res)):
            self.assertEqual(len(res[i]), len(expected[i]))
            for j in range(len(res[i])):
                self.assertAlmostEqual(res[i][j], expected[i][j], places=places)

    def test_add(self):
        res = add(self.matrix_same_dim_a, self.matrix_same_dim_b)
        expected = [[6, 12, 7],
                    [2, 10, 14]]
        self.assertEqual(res, expected)

    def test_add_empty(self):
        a = [[]]
        b = []
        self.assertRaises(ValueError, add, a, b)

        a = self.matrix_same_dim_a
        b = []
        self.assertRaises(ValueError, add, a, b)

        a = [[]]
        b = [[]]
        self.assertRaises(ValueError, add, a, b)

    def test_add_not_same(self):
        self.assertRaises(ValueError, add, self.matrix_same_dim_a, self.matrix_INV)

    def test_sub(self):
        a = self.matrix_same_dim_a
        b = self.matrix_same_dim_b
        expected = [[-4, 6, 5],
                    [2, 6, 6]]
        res = subtract(a, b)
        self.assertEqual(res, expected)

    def test_sub_empty(self):
        a = [[]]
        b = []
        self.assertRaises(ValueError, subtract, a, b)

        a = self.matrix_same_dim_a
        b = []
        self.assertRaises(ValueError, subtract, a, b)

        a = [[]]
        b = [[]]
        self.assertRaises(ValueError, subtract, a, b)

    def test_sub_not_same(self):
        self.assertRaises(ValueError, subtract, self.matrix_same_dim_a, self.matrix_INV)

    def test_mul_scalar(self):
        a = self.matrix_SINGULAR
        num = self.integer
        expected = [[3, 6, 9],
                    [9, 18, 27],
                    [12, 15, 6]]
        res = multiply_scalar(a, num)
        self.assertMatrixAlmostEqual(res, expected)

        num = self.factor
        expected = [[0.5, 1, 1.5],
                    [1.5, 3, 4.5],
                    [2, 2.5, 1]]
        res = multiply_scalar(a, num)
        self.assertMatrixAlmostEqual(res, expected)

    def test_mul_scalar_zero(self):
        num = 0
        a = self.matrix_SINGULAR
        expected = self.Z
        res = multiply_scalar(a, num)
        self.assertMatrixAlmostEqual(res, expected)

        a = self.Z
        num = self.integer
        expected = self.Z
        res = multiply_scalar(a, num)
        self.assertMatrixAlmostEqual(res, expected)

    def test_mul_scalar_empty(self):
        a = []
        num = self.integer
        self.assertRaises(ValueError, multiply_scalar, a, num)

        a = [[1], []]
        num = self.factor
        self.assertRaises(ValueError, multiply_scalar, a, num)

    def test_mul(self):
        a = self.to_mul1
        b = self.to_mul2
        expected = [[14, 18, 10, 7],
                    [32, 24, 10, 39]]
        res = multiply(a, b)
        self.assertEqual(res, expected)

        a = self.matrix_INV
        b = self.I
        expected = self.matrix_INV
        res = multiply(a, b)
        self.assertEqual(res, expected)

    def test_mul_empty(self):
        a = [[]]
        b = []
        self.assertRaises(ValueError, multiply, a, b)

        a = self.matrix_INV
        b = []
        self.assertRaises(ValueError, multiply, a, b)

        a = [[12], [0], []]
        b = [[1],
             [3],
             [0]]
        self.assertRaises(ValueError, multiply, a, b)

    def test_mul_not_appropriate(self):
        a = self.to_mul1
        b = self.matrix_same_dim_b
        self.assertRaises(ValueError, multiply, a, b)

    def test_transpose(self):
        a = self.matrix_same_dim_a
        expected = [[1, 2],
                    [9, 8],
                    [6, 10]]
        res = transpose(a)
        self.assertEqual(res, expected)

    def test_transpose_empty(self):
        a = [[]]
        self.assertRaises(ValueError, transpose, a)

        a = [[12], []]
        self.assertRaises(ValueError, transpose, a)

    def test_swap(self):
        a = self.matrix_INV
        i, j = 1, 2
        expected = [[1, 4, 3],
                    [5, 6, 3],
                    [-2, 3, 9]]
        res = swap_rows(a, i, j)
        self.assertEqual(res, expected)

    def test_swap_empty(self):
        a = [[]]
        self.assertRaises(ValueError, swap_rows, a, 0, 1)

    def test_swap_not_exists(self):
        i, j = 1, 2
        a = self.matrix_same_dim_a
        self.assertRaises(IndexError, swap_rows, a, i, j)

    def test_scale(self):
        a = self.I
        factor = self.integer
        i = 1
        expected = [[1, 0, 0],
                    [0, 3, 0],
                    [0, 0, 1]]
        res = scale_rows(a, i, factor)
        self.assertMatrixAlmostEqual(res, expected)

        a = self.matrix_INV
        factor = self.integer
        expected = [[1, 4, 3],
                    [-6, 9, 27],
                    [5, 6, 3]]
        res = scale_rows(a, i, factor)
        self.assertMatrixAlmostEqual(res, expected)

        a = self.I
        factor = self.factor
        expected = [[1, 0, 0],
                    [0, 0.5, 0],
                    [0, 0, 1]]
        res = scale_rows(a, i, factor)
        self.assertMatrixAlmostEqual(res, expected)

    def test_scale_empty(self):
        a = [[]]
        self.assertRaises(ValueError, scale_rows, a, 0, 1)

        a = [[12], [0], []]
        self.assertRaises(ValueError, scale_rows, a, 1, 1)

    def test_scale_zero(self):
        a = self.I
        factor = 0
        i = 0
        self.assertRaises(ValueError, scale_rows, a, i, factor)

    def test_scale_not_exists(self):
        a = self.matrix_INV
        i = 5
        factor = self.integer
        self.assertRaises(IndexError, scale_rows, a, i, factor)

    def test_add_rows(self):
        a = self.matrix_INV
        i, j = 1, 2
        factor = self.integer
        expected = [[1, 4, 3],
                    [13, 21, 18],
                    [5, 6, 3]]
        res = add_rows(a, i, j, factor)
        self.assertMatrixAlmostEqual(res, expected)

        expected = [[1, 4, 3],
                    [3, 9, 12],
                    [5, 6, 3]]
        res = add_rows(a, i, j)
        self.assertMatrixAlmostEqual(res, expected)

    def test_add_rows_empty(self):
        a = [[]]
        i, j = 1, 2
        self.assertRaises(ValueError, add_rows, a, i, j)

    def test_add_rows_not_exists(self):
        a = self.matrix_INV
        i, j = 2, 5
        factor = self.factor
        self.assertRaises(IndexError, add_rows, a, i, j, factor)

    def test_det(self):
        a = self.matrix_INV
        expected = 78
        res = det(a)
        self.assertAlmostEqual(res, expected)

        a = self.I
        expected = 1
        res = det(a)
        self.assertAlmostEqual(res, expected)

    def test_det_zero(self):
        a = self.matrix_SINGULAR
        expected = 0
        res = det(a)
        self.assertAlmostEqual(res, expected)

    def test_det_not_exists(self):
        a = self.matrix_same_dim_a
        self.assertRaises(ValueError, det, a)

    def test_inverse(self):
        a = self.matrix_INV
        expected = [[-15/26, 1/13, 9/26],
                    [17/26, -2/13, -5/26],
                    [-9/26, 7/39, 11/78]]
        res = inverse(a)
        self.assertMatrixAlmostEqual(res, expected)

    def test_inverse_singular(self):
        a = self.matrix_SINGULAR
        self.assertRaises(ValueError, inverse, a)

    def test_inverse_not_square(self):
        a = self.matrix_same_dim_a
        self.assertRaises(ValueError, inverse, a)

    def test_rank(self):
        a = self.matrix_INV
        expected = 3
        res = rank(a)
        self.assertEqual(res, expected)

        a = self.matrix_SINGULAR
        expected = 2
        res = rank(a)
        self.assertEqual(res, expected)

    def test_rank_empty(self):
        a = [[]]
        self.assertRaises(ValueError, rank, a)

    def tearDown(self):
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
