from numpy import diag, dot
from numpy.linalg import svd, norm, matrix_rank
import matplotlib.pyplot as plt
import scipy.misc as misc

a = np.array([[1, 1, 1, 0, 0],
              [3, 3, 3, 0, 0],
              [4, 4, 4, 0, 0],
              [5, 5, 5, 0, 0],
              [0, 2, 0, 4, 4],
              [0, 0, 0, 5, 5],
              [0, 1, 0, 2, 2]])

# -------- INIT STUFF -------------
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

img = misc.ascent()
print("--- FULL ---")
U, S, VT = svd(a, full_matrices=False)
print("U:\n {}".format(U))
print("s:\n {}".format(S))
print("VT:\n {}".format(VT))


# --------------------------------


def zero_k_sing_values(s, k):
    # TODO delete k
    if s is None or k is None or k < 0:
        raise ValueError("bad input")

    for sing_val in range(k, len(s)):
        s[sing_val] = 0

    return s


def reconstruct_rank_k(u, s, vt, k):
    # TODO delete k
    if u is None or s is None or vt is None or k < 0:
        raise ValueError("bad input")

    sing_mat_after_zero_k = zero_k_sing_values(s, k)
    return dot(
        dot(u, diag(sing_mat_after_zero_k)),
        vt)


def calc_compression_ratio(square_mat, k):
    r = matrix_rank(square_mat)  # r=rank
    n = len(square_mat)
    return 1 - \
           ((2 * k * n) + k) / ((2 * n * r) + n)


def forbenius_dist(mat1, mat2):
    return norm(mat1 - mat2)

def main():
    x = 5
    y = 6

    # [forb1, forb2, forb2,......forbK]
    #

# print(diag(S))
# print(diag(zero_k_sing_values(S,1)))
# print(a)
# print(reconstruct_rank_k(U,S,VT,2))
if  __name__ =='__main__':main()