from numpy import diag, dot
from numpy.linalg import svd, norm
from matplotlib.pyplot import *
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
U, s, VT = svd(a, full_matrices=False)
print("U:\n {}".format(U))
print("s:\n {}".format(s))
print("VT:\n {}".format(VT))
# --------------------------------


def zero_k_sing_values(sing_mat, k):
    if sing_mat is None or k is None or k < 0:
        raise ValueError("bad input")

    for sing_val in range(k, len(sing_mat)):
        sing_mat[sing_val] = 0

    return sing_mat

def reconstruct_rank_k(U, s, VT, k):
    return np.dot()
