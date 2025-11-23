from matrix import Matrix

a = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

try:
    det = Matrix.det(a)
    print(det)
except ValueError as e:
    print(f"Error: {e}")