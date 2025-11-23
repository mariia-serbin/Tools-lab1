from matrix import Matrix

a = [[1, 2, 3],
     [4, 5, 17],
     [18, 6, 20]]
factor = 0.5
i = 1

try:
    Matrix.scale_rows(a, i, factor)
    for row in a:
        print(row)
except ValueError as e:
    print(f"Error: {e}")