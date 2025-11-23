from matrix import Matrix

a = [[1, 2],
     [3, 4]]
scalar = 5
try:
    res = Matrix.multiply_scalar(a, scalar)
    for row in res:
        print(row)
except ValueError as e:
    print(f"Error: {e}")