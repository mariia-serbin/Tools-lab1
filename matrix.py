from typing import *

def add(a: List[List[float|int]], b: List[List[float|int]]) -> List[List[float|int]]:
    if not a or not b or any(len(row) == 0 for row in a) or any(len(row) == 0 for row in b):
        raise ValueError('Matrices are empty')
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError('Matrices should have the same dimensions')
    n, m = len(a), len(a[0])
    return [[a[i][j] + b[i][j] for j in range(m)] for i in range(n)]

def subtract(a: List[List[float|int]], b: List[List[float|int]]) -> List[List[float|int]]:
    if not a or not b or any(len(row) == 0 for row in a) or any(len(row) == 0 for row in b):
        raise ValueError('Matrices are empty')
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError('Matrices should have the same dimensions')
    n, m = len(a), len(a[0])
    return [[a[i][j] - b[i][j] for j in range(m)] for i in range(n)]

def multiply_scalar(a: List[List[float|int]], num: float|int) -> List[List[float|int]]:
    if not a or any(len(row) == 0 for row in a):
        raise ValueError('Matrix is empty')
    return [[round(x * num, 10) for x in row] for row in a]

def multiply(a: List[List[float|int]], b: List[List[float|int]]) -> List[List[float|int]]:
    if not a or not b or any(len(row) == 0 for row in a) or any(len(row) == 0 for row in b):
        raise ValueError('Matrices cannot be empty for multiplying them')
    n_a, m_a = len(a), len(a[0])
    n_b, m_b = len(b), len(b[0])
    if m_a != n_b:
        raise ValueError('Matrices cannot be multiplied: incompatible sizes')
    return [[sum(a[i][k] * b[k][j] for k in range(m_a)) for j in range(m_b)] for i in range(n_a)]

def transpose(a: List[List[float|int]]) -> List[List[float|int]]:
    if not a or any(len(row) == 0 for row in a):
        raise ValueError('Matrix is empty or has empty rows')
    return [list(row) for row in zip(*a)]

def swap_rows(a: List[List[float|int]], i: int, j: int) -> List[List[float|int]]:
    """
        Swaps two rows (i and j) of a matrix.
        Note: Row indexing is 0-based (i.e., the first row is at index 0).
    """
    if not a or any(len(row) == 0 for row in a):
        raise ValueError('Matrix is empty')
    if i >= len(a) or j >= len(a) or i < 0 or j < 0:
        raise IndexError('Indexes out of range')
    copy = [row[:] for row in a]
    copy[i], copy[j] = copy[j], copy[i]
    return copy

def scale_rows(a: List[List[float|int]], i: int, factor: float|int) -> List[List[float|int]]:
    """
        Multiplies a row by a scalar factor.
        Note: Row indexing is 0-based (i.e., the i-th row is at index i).
    """
    if not a or any(len(row) == 0 for row in a):
        raise ValueError('Matrix is empty')
    if i >= len(a) or i < 0:
        raise IndexError('Index out of range')
    if factor == 0:
        raise ValueError('You cannot multiply rows by zero')
    copy = [row[:] for row in a]
    copy[i] = [round(x * factor, 10) for x in copy[i]]
    return copy

def add_rows(a: List[List[float|int]], i: int, j: int, factor: float|int = 1) -> List[List[float|int]]:
    """
        Adds row 'j' (multiplied by 'factor') to row 'i'.
        Note: Row indexing is 0-based (i.e., the first row is at index 0).
    """
    if not a or any(len(row) == 0 for row in a):
        raise ValueError('Matrix is empty')
    if i >= len(a) or j >= len(a) or i < 0 or j < 0:
        raise IndexError('Index out of range')
    if factor == 0:
        raise ValueError('You cannot multiply rows by zero')
    copy = [row[:] for row in a]
    copy[i] = [round(copy[i][k] + copy[j][k] * factor, 10) for k in range(len(copy[i]))]
    return copy

def det(a: List[List[float|int]]) -> float|int:
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
        if copy[pivot_row][i] == 0:
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
    return round(det_val, 10)

def inverse(a: List[List[float|int]]) -> List[List[float|int]]:
    if not a or any(len(row) == 0 for row in a):
        raise ValueError('Matrix is empty')
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError('Matrix must be square')
    copy = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(a)]
    for i in range(n):
        pivot_row = max(range(i, n), key=lambda r: abs(copy[r][i]))
        if copy[pivot_row][i] == 0:
            raise ValueError('Matrix is singular and cannot be inverted')
        if pivot_row != i:
            copy[i], copy[pivot_row] = copy[pivot_row], copy[i]
        factor = copy[i][i]
        copy[i] = [x / factor for x in copy[i]]
        for j in range(n):
            if j != i:
                factor2 = copy[j][i]
                copy[j] = [round(copy[j][k] - factor2 * copy[i][k], 10) for k in range(2*n)]
    return [row[n:] for row in copy]

def rank(a: List[List[float|int]]) -> int:
    if not a or any(len(row) == 0 for row in a):
        raise ValueError('Matrix cannot be empty')
    n, m = len(a), len(a[0])
    copy = [row[:] for row in a]
    rank_val = 0
    for i in range(min(n, m)):
        pivot_row = -1
        for j in range(i, n):
            if abs(copy[j][i]) > 1e-10:
                pivot_row = j
                break
        if pivot_row == -1:
            continue
        if pivot_row != i:
            copy[i], copy[pivot_row] = copy[pivot_row], copy[i]
        for j in range(i + 1, n):
            factor = copy[j][i] / copy[i][i]
            copy[j] = [copy[j][k] - factor * copy[i][k] for k in range(m)]
        rank_val += 1
    return rank_val
