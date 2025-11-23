from matrix import Matrix

a = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
i, j, factor = 0, 1, 3

try:
    Matrix.add_rows(a, i, j, factor)
    for row in a:
        print(row)
except (ValueError, IndexError) as e:
    print(f"Error: {e}")