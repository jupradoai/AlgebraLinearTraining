import numpy as np

# Definir os vetores v1, v2 e v3
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

# Verificar a natureza do sistema
if rank == A.shape[1]:
    if np.allclose(np.dot(A, x), V):
        print("O sistema é possível determinado (SPD), e o vetor V pertence ao espaço gerado por v1, v2 e v3.")
    else:
        print("O sistema é possível determinado (SPD), mas o vetor V NÃO pertence ao espaço gerado por v1, v2 e v3.")
elif rank < A.shape[1]:
    print("O sistema é possível indeterminado (SPI).")
else:
    print("O sistema é impossível (SI).")
