import cv2 as cv
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
from scipy.ndimage import laplace
import skimage
from skimage.measure import label, regionprops
import numpy as np


def opencv_inpaint(image, image_mask):
    telea = cv.inpaint(image, image_mask, 3, cv.INPAINT_TELEA)
    ns = cv.inpaint(image, image_mask, 3, cv.INPAINT_NS)
    return telea, ns


def numbergrid(mask):
    n = np.sum(mask)
    G1 = np.zeros_like(mask, dtype=np.uint32)
    G1[mask] = np.arange(1, n + 1)
    return G1


def delsq_laplacian(G):
    [m, n] = G.shape
    G1 = G.flatten()
    p = np.where(G1)[0]
    N = len(p)
    i = G1[p] - 1
    j = G1[p] - 1
    s = 4 * np.ones(N)
    for offset in [-1, m, 1, -m]:
        Q = G1[p + offset]
        q = np.where(Q)[0]
        i = np.concatenate([i, G1[p[q]] - 1])
        j = np.concatenate([j, Q[q] - 1])
        s = np.concatenate([s, -np.ones(q.shape)])

    sp = sparse.csr_matrix((s, (i, j)), (N, N))
    return sp


def delsq_bilaplacian(G):
    [n, m] = G.shape
    G1 = G.flatten()
    p = np.where(G1)[0]
    N = len(p)
    i = G1[p] - 1
    j = G1[p] - 1
    s = 20 * np.ones(N)
    coeffs = np.array([1, 2, -8, 2, 1, -8, -8, 1, 2, -8, 2, 1])
    offsets = np.array([-2 * m, -m - 1, -m, -m + 1, -2, -1, 1, 2, m - 1, m, m + 1, 2 * m])
    for coeff, offset in zip(coeffs, offsets):
        Q = G1[p + offset]
        q = np.where(Q)[0]
        i = np.concatenate([i, G1[p[q]] - 1])
        j = np.concatenate([j, Q[q] - 1])
        s = np.concatenate([s, coeff * np.ones(q.shape)])

    sp = sparse.csr_matrix((s, (i, j)), (N, N))
    return sp


def generate_stencials():
    stencils = []
    for i in range(5):
        for j in range(5):
            A = np.zeros((5, 5))
            A[i, j] = 1
            S = laplace(laplace(A))
            x_range = np.array([i - 2, i + 3]).clip(0, 5)
            y_range = np.array([j - 2, j + 3]).clip(0, 5)
            S = S[x_range[0]:x_range[1], y_range[0]:y_range[1]]
            stencils.append(S)

    return stencils


def _inpaint_biharmonic_single_channel(mask, out, limits):
    G = numbergrid(mask)
    L = delsq_bilaplacian(G)
    out[mask] = 0
    B = -laplace(laplace(out))
    b = B[mask]
    result = spsolve(L, b)
    result = np.clip(result, *limits)
    result = result.ravel()
    out[mask] = result
    return out


def dilate_rect(rect, d, nd_shape):
    rect[0:2] = (rect[0:2] - d).clip(min=0)
    rect[2:4] = (rect[2:4] + d).clip(max=nd_shape)
    return rect


def k_inpaint_biharmonic(image, mask, multichannel=False):
    if image.ndim < 1:
        raise ValueError('Input array has to be at least 1D')

    img_baseshape = image.shape[:-1] if multichannel else image.shape
    if img_baseshape != mask.shape:
        raise ValueError('Input arrays have to be the same shape')

    if np.ma.isMaskedArray(image):
        raise TypeError('Masked arrays are not supported')

    image = skimage.img_as_float(image)
    mask = mask.astype(bool)

    kernel = ndi.generate_binary_structure(mask.ndim, 1)
    mask_dilated = ndi.binary_dilation(mask, structure=kernel)
    mask_labeled, num_labels = label(mask_dilated, return_num=True)
    mask_labeled *= mask
    if not multichannel:
        image = image[..., np.newaxis]

    out = np.copy(image)

    props = regionprops(mask_labeled)
    comp_out_imgs = []
    comp_masks = []
    for i in range(num_labels):
        rect = np.array(props[i].bbox)
        rect = dilate_rect(rect, 2, image.shape[:2])
        out_sub_img = out[rect[0]:rect[2], rect[1]:rect[3], :]
        comp_mask = mask[rect[0]:rect[2], rect[1]:rect[3]]
        comp_out_imgs.append(out_sub_img)
        comp_masks.append(comp_mask)

    for idx_channel in range(image.shape[-1]):
        known_points = image[..., idx_channel][~mask]
        limits = (np.min(known_points), np.max(known_points))
        for i in range(num_labels):
            _inpaint_biharmonic_single_channel(comp_masks[i], comp_out_imgs[i][..., idx_channel], limits)

    if not multichannel:
        out = out[..., 0]

    return out


img = cv.imread('source.png')
mask = cv.imread('mask.png', cv.IMREAD_GRAYSCALE)

dst_telea, dst_ns = opencv_inpaint(img, mask)
dst_biharmonic = k_inpaint_biharmonic(img, mask, multichannel=True)

cv.imshow('dst telea', dst_telea)
cv.imshow('dst ns', dst_ns)
cv.imshow('dst biharmonic', dst_biharmonic)

cv.waitKey(0)
cv.destroyAllWindows()
