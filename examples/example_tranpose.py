from matrix import Matrix

a = [[0, 1],
     [2, 3],
     [4, 5],
     [6, 7]]

try:
    res = Matrix.transpose(a)
    for row in res:
        print(row)
except ValueError as e:
    print(f"Error: {e}")