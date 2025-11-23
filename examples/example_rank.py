from matrix import  Matrix

a = [[1, 2, 3],
     [4, 8, 12],
     [7, 8, 9]]

try:
    rank = Matrix.rank(a)
    print(rank)
except ValueError as e:
    print(f"Error: {e}")