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

# Documentation

Detailed technical documentation for this library, including function signatures, parameter descriptions, return values, and explicit code examples, is automatically generated and deployed.

## Viewing the Documentation

The full documentation is available online via **GitHub Pages**:

**[Link to GitHub Pages Documentation](https://mariia-serbin.github.io/Tools-lab1)**

## Automation

Documentation is generated using **Doxygen** and is updated automatically upon every push to the designated branch via a **GitHub Actions** workflow, ensuring that the documentation is always synchronized with the latest codebase.

* **Tool:** Doxygen
* **Process:** Continuous Integration / Continuous Deployment (CI/CD)

## Local Documentation Generation

To generate and view the documentation locally (without pushing to GitHub Actions), follow these steps:

1.  **Install Doxygen:** Ensure Doxygen (and Graphviz, if you want dependency graphs) is installed on your system.
    ```bash
    # Example for Debian/Ubuntu
    sudo apt install doxygen graphviz
    ```

2.  **Run Generation:** Navigate to the root directory of the repository (where `Doxyfile` is located).
    ```bash
    doxygen Doxyfile
    ```

3.  **View Results:** Open the generated HTML file in your web browser.
    ```bash
    # The output path is usually defined in Doxyfile (e.g., HTML_OUTPUT)
    start ./docs/html/index.html 
    ```