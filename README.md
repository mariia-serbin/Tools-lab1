# Matrix Operations

This laboratory work is about testing basic matrix operations such as addition, subtraction, multiplication, scalar multiplication, transposition, row operations, determinant calculation, matrix inversion, and rank calculation.

# Features

* Matrix addition and subtraction.

* Scalar multiplication and matrix multiplication.

* Transposition of matrices.

* Row operations: swap, scale, and add.

* Determinant calculation.

* Inversion of square matrices.

* Rank calculation.

* Unit tests for both normal and edge cases, including empty matrices, singular matrices, and mismatched dimensions.

# Technologies used

* Python 3.13. Both matrix and test files are implemented in Python programming language  
* unittest: built-in module in Python programming language which stands for unit testing code

# Notes

**Matrix Dimensions**: Most operations require compatible dimensions. Addition and subtraction require matrices of the same size. Multiplication requires the number of columns in the first matrix to equal the number of rows in the second. Operations on incompatible matrices will raise a `ValueError`.

**Square Matrices**: Only square matrices (same number of rows and columns) can be inverted or have a determinant calculated. Attempting these operations on non-square matrices will raise a `ValueError`.

**Singular Matrices**: Matrices with zero determinant are singular and cannot be inverted. Attempting to invert a singular matrix will raise a `ValueError`.

**Empty Matrices**: Operations on empty matrices or matrices containing empty rows are not allowed and will raise a `ValueError`.

**Scalar Multiplication**: Works with both integers and floating-point numbers. Multiplying by zero will produce a zero matrix of the same size.

**Row Operations**:

* `swap_rows`: swaps two rows of a matrix. Raises `IndexError` if the row index does not exist.

* `scale_rows`: multiplies a row by a scalar. Raises `ValueError` if the scalar is zero.

* `add_rows`: adds a multiple of one row to another. Raises `IndexError` if a row index does not exist.

  **Testing**: The project includes comprehensive unit tests for normal and edge cases, ensuring correctness of all operations.

