from matrix import Matrix

a = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
i, j = 0, 2

try:
    Matrix.swap_rows(a, i, j)
    for row in a:
        print(row)
except (ValueError, IndexError) as e:
    print(f"Error: {e}")