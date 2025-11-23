from matrix import Matrix

a = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
try:
    inverted = Matrix.inverse(a)
    for row in inverted:
        print(row)
except ValueError as e:
    print(f"Error: {e}")