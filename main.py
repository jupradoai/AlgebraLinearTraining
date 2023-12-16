import numpy as np

v1 = np.array([2, 1, 1])
v2 = np.array([1, 2, 2])
v3 = np.array([1, 1, 2])

# Definir o vetor V que você quer verificar
V = np.array([8, 9, 10])

# Criar uma matriz com os vetores v1, v2 e v3 como colunas
A = np.column_stack((v1, v2, v3))

# Resolver o sistema linear Ax = V
# O sistema é sobredeterminado, então usamos a função de mínimos quadrados
x, residuals, rank, s = np.linalg.lstsq(A, V, rcond=None)

# Verificar se x é uma solução
if np.allclose(np.dot(A, x), V):
    print("O vetor V pertence ao espaço gerado por v1, v2 e v3.")

else:
    print("O vetor V NÃO pertence ao espaço gerado por v1, v2 e v3.")