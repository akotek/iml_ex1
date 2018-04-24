from numpy import diag, dot
from numpy.linalg import svd, norm, matrix_rank
import scipy.misc as misc
import copy
import utils


def zero_k_sing_values(s, k):
    for sing_val in range(k, len(s)):
        s[sing_val] = 0

    return s


def reconstruct_rank_k_mat(u, s, vt, k):
    zerod_s = zero_k_sing_values(s, k)
    us_mat = dot(u, diag(zerod_s))
    return dot(us_mat, vt)


def calc_compression_ratio(square_mat, k):
    r = matrix_rank(square_mat)  # r=rank
    n = len(square_mat)
    return ((2 * k * n) + k) / ((2 * n * r) + n)


def forbenius_dist(mat1, mat2):
    return norm(mat1 - mat2)


def get_rec_forb_comp(img, u, s, vt, k):
    if img is None or u is None or s is None or vt is None:
        raise ValueError("bad input")

    rec_matrix = reconstruct_rank_k_mat(u, s, vt, k)
    y1_forb_dist = (forbenius_dist(img, rec_matrix))
    y2_comp_ratio = (calc_compression_ratio(img, k))

    return rec_matrix, y1_forb_dist, y2_comp_ratio


def main():
    # ---- Init data ----------
    img = misc.ascent()
    U, S, VT = svd(img, full_matrices=False)
    # -------------------------

    # -------- 1st Plot --------
    k_x_axis = []
    y1_forb_dist = []
    y2_comp_ratio = []
    for k in range(len(img)):
        k_x_axis.append(k)
        rec_img, y1, y2 = get_rec_forb_comp(img, U, copy.deepcopy(S), VT, k)
        y1_forb_dist.append(y1)
        y2_comp_ratio.append(y2)

    utils.plot_graph(k_x_axis, "k", y1_forb_dist, "Forb Dist")
    utils.plot_graph(k_x_axis, "k", y2_comp_ratio, "Compression Ratio")
    # -------------------------

    # -------- 2nd Plot --------
    k_image_list = [5, 20, 45, 250, 511]
    image_list = []
    y1_forb_dist = []
    y2_comp_ratio = []
    for k in k_image_list:
        rec_img, y1, y2 = get_rec_forb_comp(img, U, copy.deepcopy(S), VT, k)
        image_list.append(rec_img)
        y1_forb_dist.append(y1)
        y2_comp_ratio.append(y2)

    utils.plot_images_from_list(image_list, k_image_list, y1_forb_dist,
                                y2_comp_ratio)
    # -------------------------


if __name__ == '__main__': main()
