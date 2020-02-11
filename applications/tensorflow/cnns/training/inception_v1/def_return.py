import numpy as np

def matmul():
    x = np.random.randint(1, 5, (2, 3))
    y = np.random.randint(1, 5, (3, 2))
    z = np.matmul(x, y)
    return x, y, z

a, b ,c = matmul()
print(a)
print(b)
print(c)