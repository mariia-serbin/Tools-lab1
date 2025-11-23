from matrix import Matrix

a = [[1, 2],
     [3, 4]]
b = [[5, 6],
     [7, 8]]

try:
    res = Matrix.add(a, b)
    for row in res:
        print(row)
except ValueError as e:
    print(f"Error: {e}")