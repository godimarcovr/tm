import numpy as np
import scipy.io
import collections
import cv2

import scipy.ndimage
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy import signal
from scipy.stats import kurtosis
from skimage.feature import greycomatrix, greycoprops
from skimage.exposure import rescale_intensity
import time
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import mahalanobis


NUM_CELL = 2
MINGLCM=[0.0626, 0.1800, 0.1740, 0.1192, 0.3642, 0.3935, 0.4446, 0.3456, 0.3086, 0.2503, 0.0029, 0.0510, 0.0592, 0.0519, 0.0518, 0.0525, 0.0441, 0.0446, 0.6510, 0.6019, 0.5981, 0.6064, 0.5528, 0.5621]
MAXGLCM=[2.3322, 2.7357, 2.8263, 3.1194, 4.7967, 3.7929, 0.9934, 0.9784, 0.9824, 0.9807, 0.9333, 0.9374, 0.5754, 0.5579, 0.5668, 0.5338, 0.5247, 0.5324, 0.9715, 0.9423, 0.9416, 0.9458, 0.9101, 0.8985]
MINTAMURA=[3.8163, 0.0028, 0.2382]
MAXTAMURA=[4.5139, 0.1155, 0.5926]
MINRFILT=428919
MAXRFILT=2327776
MASK_RESIZE_LIMIT=200


def get_features(path, mask=None, patches=True):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if mask is None:
        snake, mask = get_mask(img)
    else:
        snake = get_snake_from_mask(mask)

    # Extract patches from image
    if patches:
        p_img, p_mask = _get_patches(img, mask, snake, NUM_CELL)
    else:
        p_img = [img]
        p_mask = [mask]

    feat_rfilt = [0] * len(p_img)
    feat_tamura = [0] * len(p_img)
    feat_glcm = [0] * len(p_img)
    feat_color = [0] * len(p_img)
    for i in range(len(p_img)):
        # Get rfilt vector
        feat_rfilt[i] = _get_rfilt(p_img[i])

        # Get tamura features
        feat_tamura[i] = _get_tamura(p_img[i])

        # Get glcm features
        feat_glcm[i] = _get_glcm(p_img[i])

        # Get color features
        feat_color[i] = get_color_hist(p_img[i], p_mask[i])

    # Quantization
    sym_rfilt = []
    sym_glcm = []
    sym_tamura = []
    sym_color = []
    for i in range(len(p_img)):
        h_rfilt = np.histogram(feat_rfilt[i], 16, [MINRFILT, MAXRFILT])
        sym_rfilt.append(np.argmax(h_rfilt[0]))

        for k in range(len(feat_glcm[i])):
            h_glcm = np.histogram(feat_glcm[i][k], 16, [MINGLCM[k], MAXGLCM[k]])
            sym_glcm.append(np.argmax(h_glcm[0]))

        for k in range(len(feat_tamura[i])):
            h_tamura = np.histogram(feat_tamura[i][k], 16, [MINTAMURA[k], MAXTAMURA[k]])
            sym_tamura.append(np.argmax(h_tamura[0]))

        sym_color = sym_color + [round(c * 16) for c in feat_color[i]]

    return np.concatenate([ np.tile(sym_rfilt, 24), np.tile(sym_tamura, 8), sym_glcm, sym_color ])
    #return np.concatenate([np.tile(sym_rfilt, 24), np.tile(sym_tamura, 8), sym_glcm])


def get_color_hist(img, mask=None):
    # 0 black
    # 1 blue
    # 2 brown
    # 3 gray
    # 4 green
    # 5 orange
    # 6 pink
    # 7 purple
    # 8 red
    # 9 white
    # 10 yellow

    if mask is None:
        _, mask = get_mask(img)

    if np.sum(mask)<1:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    RR = (img[:, :, 0]).astype('float')
    GG = (img[:, :, 1]).astype('float')
    BB = (img[:, :, 2]).astype('float')

    index_im = np.floor(RR.flatten() / 8) + 32 * np.floor(GG.flatten() / 8) + 32 * 32 * np.floor(BB.flatten() / 8)

    w2c = scipy.io.loadmat('w2c.mat')
    w2cM = np.argmax(w2c['w2c'], axis=1)
    # out = np.reshape(w2cM[index_im.flatten().astype('int32')], (height, width))

    out = w2cM[index_im.flatten().astype('int32')]
    CNI = out[mask.flatten().astype('bool')]
    H = collections.Counter(CNI)
    normH = []
    for i in range(11):
        normH.append(H[i] / len(CNI))

    return normH


def get_snake_from_mask(init_mask):

    contour_info = []
    _, contours, _ = cv2.findContours(init_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

    if len(contour_info) == 0:
        return None

    max_contour = contour_info[0]

    return np.squeeze(max_contour[0]).astype(float)


def get_mask(img):
    # img = RGB image as ndarray
    # == Processing =======================================================================
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    ratio = h / w
    imgr = img
    scale = 1
    if h > MASK_RESIZE_LIMIT:
        scale = MASK_RESIZE_LIMIT / h
        # if ratio > 1:
        gray = cv2.resize(gray, (int(w * scale), MASK_RESIZE_LIMIT))
        imgr = cv2.resize(img, (int(w * scale), MASK_RESIZE_LIMIT))
        # else:
        #     gray = cv2.resize(gray, (int(w * scale), MASK_RESIZE_LIMIT))
        #     imgr = cv2.resize(img, (int(w * scale), MASK_RESIZE_LIMIT))

    h, w = gray.shape
    if w > MASK_RESIZE_LIMIT:
        scale = MASK_RESIZE_LIMIT / w
        gray = cv2.resize(gray, (MASK_RESIZE_LIMIT, int(h * scale)))
        imgr = cv2.resize(img, (MASK_RESIZE_LIMIT, int(h * scale)))

    kernel = np.ones((5, 5), np.float32) / 25
    imgr = cv2.filter2D(imgr, -1, kernel)

    init_mask = np.zeros(gray.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # perc=0.04
    # rect = (int(1+gray.shape[1]*perc), int(1+gray.shape[0]*perc), int(gray.shape[1]*(1-perc/2)), int(gray.shape[0]*(1-perc/2)))
    rect = (1, 1, gray.shape[1], gray.shape[0])
    t = time.time()
    cv2.grabCut(imgr, init_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    elapsed = time.time() - t
    # print(elapsed)

    init_mask = np.where((init_mask == 2) | (init_mask == 0), 0, 1).astype('uint8')

    # init_mask = cv2.resize(init_mask, (img.shape[1], img.shape[0]))
    # gray = cv2.resize(gray, (img.shape[1], img.shape[0]))
    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    t = time.time()
    _, contours, _ = cv2.findContours(init_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    elapsed = time.time() - t
    # print(elapsed)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

    if len(contour_info) == 0:
        return None, np.ones((img.shape[0], img.shape[1]))

    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(init_mask.shape)
    init = np.squeeze(max_contour[0]).astype(float)
    t = time.time()
    snake = active_contour(gaussian(gray, 3),
                           init, alpha=0.14, beta=9, gamma=0.05)
                           #init, alpha=0.05, beta=1, gamma=0.1)
                           #init, alpha=0.15, beta=10, gamma=0.01)
                           # init, alpha=0.10, beta=5, gamma=0.05)
    elapsed = time.time() - t
    # print(elapsed)

    cv2.fillPoly(mask, [snake.astype('int32')], 1)

    # riporto alla dimensione dell'immagine originale
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask[mask > 0.5] = 1
    mask[mask < 0.5] = 0
    snake = snake / scale
    return snake, mask


def _get_patches(img, mask, snake, num_cell):

    if not snake.all():
        width = int(img.shape[1] / num_cell)
        height = int(img.shape[0] / num_cell)
        x0 = 0
        y0 = 0
    else:
        x0, y0, width, height = cv2.boundingRect(snake.astype('int32'))
        height = int(height / num_cell)
        width = int(width / num_cell)

    patches_img = []
    patches_mask = []
    for i in range(num_cell):
        y1 = y0 + i * height
        for j in range(num_cell):
            x1 = x0 + j * width
            patches_img.append(np.copy(img[y1:y1+height, x1:x1+width]))
            patches_mask.append(np.copy(mask[y1:y1 + height, x1:x1 + width]))

    return patches_img, patches_mask


def _get_rfilt(patch):

    # Computes RFILT features from a patch or image with the 3 channels RGB
    RR = (patch[:, :, 0]).astype('float')
    GG = (patch[:, :, 1]).astype('float')
    BB = (patch[:, :, 2]).astype('float')

    RR = scipy.ndimage.grey_dilation(RR, size=(3,3)) - scipy.ndimage.grey_erosion(RR, size=(3,3))
    GG = scipy.ndimage.grey_dilation(GG, size=(3,3)) - scipy.ndimage.grey_erosion(GG, size=(3,3))
    BB = scipy.ndimage.grey_dilation(BB, size=(3,3)) - scipy.ndimage.grey_erosion(BB, size=(3,3))

    return np.sum(RR) + np.sum(GG) + np.sum(BB)


def _get_glcm(patch):

    # Computes GLCM features from a patch or image with the 3 channels RGB
    angles = [0, np.pi/6, 2*np.pi/6, 3*np.pi/6, 4*np.pi/6, 5*np.pi/6]
    offset = [1]  # [[0, 1], [-1, 1], [-1, -1], [0, 3], [-3, 3], [-3, -3]]
    levels = 8

    if len(patch.shape) > 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

    rescaled_img = rescale_intensity(patch, out_range=(0, levels - 1))
    glcm = greycomatrix(rescaled_img, offset, angles, levels=8, symmetric=False, normed=True)
    features = np.concatenate([greycoprops(glcm, 'contrast')[0, :],
                               greycoprops(glcm, 'correlation')[0, :],
                               greycoprops(glcm, 'energy')[0, :],
                               greycoprops(glcm, 'homogeneity')[0, :], ])

    return features


def _get_tamura(patch):

    # Computes Tamura features from a patch or image with the 3 channels RGB
    img_gray= cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    img_d = _im2double(img_gray)

    # First feature: Tamura Directionality
    gx, gy = np.gradient(img_d)
    r = np.sqrt(gx ** 2 + gy ** 2)
    t = np.arctan2(gy, gx)

    good = np.where(r.flatten() > 0.15 * np.amax(r))

    if len(good) < 1:
        Fdir = 0
    else:
        r = r.flatten()[good]
        t = t.flatten()[good]
        Fdir = 1 / (_entropy(t) + 1)

    # Second feature: Tamura Coarseness
    kk = range(7)
    Hdelta = []
    Vdelta = []
    for ii in range(1, kk[-1], 1):
        A = _moveav(img_d, 2 ** kk[ii])
        shift = 2 ^ kk[ii]
        implus = np.zeros(A.shape)
        implus[:, 0:-shift - 1] = A[:, shift:-1]
        iminus = np.zeros(A.shape)
        iminus[:, shift:-1] = A[:, 0:-shift - 1]
        Hdelta.append(np.abs(implus - iminus))
        implus = np.zeros(A.shape)
        implus[0:-shift - 1, :] = A[shift:-1, :]
        iminus = np.zeros(A.shape)
        iminus[shift:-1, :] = A[0:-shift - 1, :]
        Vdelta.append(np.abs(implus - iminus))

    Hdelta = np.swapaxes(np.swapaxes(np.asarray(Hdelta), 0, 2), 0, 1)
    Vdelta = np.swapaxes(np.swapaxes(np.asarray(Vdelta), 0, 2), 0, 1)
    HdeltaMax = np.amax(Hdelta, axis=0)
    hs = np.sum(HdeltaMax.flatten())
    VdeltaMax = np.amax(Vdelta, axis=0)
    vs = np.sum(VdeltaMax.flatten())
    hij = np.reshape(Hdelta, [Hdelta.shape[0] * Hdelta.shape[1], Hdelta.shape[2]])
    vij = np.reshape(Vdelta, [Vdelta.shape[0] * Vdelta.shape[1], Vdelta.shape[2]])
    newh = np.zeros([hij.shape[0], 1])

    for ii in range(0, hij.shape[0], 1):
        tmp1 = hij[ii, :]
        tmp2 = vij[ii, :]
        mtmp1 = np.amax(tmp1)
        mtmp2 = np.amax(tmp2)
        mm = np.amax([mtmp1, mtmp2])
        im1 = np.where(tmp1 == mtmp1)
        im2 = np.where(tmp2 == mtmp2)
        if mm == mtmp1:
            imm = im1[0]
        else:
            imm = im2[0]
        for idx in imm:
            newh[ii] = kk[idx]

    Fcoarseness = np.mean(newh)

    # Third feature: Tamura Contrast
    ss = np.std(img_d)
    if np.abs(ss) < 0.0000000001:
        Fc = 0
        return Fc
    else:
        k = kurtosis(img_d.flatten(), fisher=False)

    alf = k / ss ** 4
    Fcont = ss / (alf ** .25)

    return [Fcoarseness, Fcont, Fdir]


def _im2double(img):
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    out = (img.astype('float') - min_val) / (max_val - min_val)

    return out


def _entropy(signal_input):

    signal_input = _im2uint8(signal_input)

    # lensig = signal.size
    # symset = list(set(signal))
    # numsym = len(symset)
    # propab = [np.size(signal[signal == i]) / (1.0 * lensig) for i in symset]
    # ent = np.sum([p * np.log2(1.0 / p) for p in propab])

    hist, bin_edges = np.histogram(signal_input, bins=np.arange(256), density=True)
    hist = hist[hist > 0]
    ent = -np.sum(hist * np.log2(hist))

    return ent


def _im2uint8(array):

    old_min = min(array)
    old_range = max(array) - old_min
    new_min = 0
    new_range = 255 + 0.9999999999 - new_min
    output = np.asarray([int((n - old_min) / old_range * new_range + new_min) for n in array])

    return output


def _moveav(img, nk):

    kern = np.ones([nk, nk]) / (nk ** 2)
    sm = signal.convolve2d(img, kern, boundary='symm', mode='same')

    return sm


def _plsa_learn(counts, T, epsilon, labels):
    # labels devono partire da 0

    # labels = np.array(labels)

    eps = np.finfo(float).eps
    W, D = counts.shape

    wt = np.zeros([W, T])
    for t in range(T):
        idx = np.where(labels == t)
        wt[:, t] = np.mean(np.mean(counts[:, idx], axis=1), axis=1)

    wt = _normalize(wt)

    # Learn
    td = _normalize(np.ones([T, D]))

    tot = np.sum(counts)
    done = False
    iter = 0

    # Free Energy init
    E = np.sum(counts * np.log(np.dot(wt, td) + eps), axis=0)
    F = [np.sum(E) / tot]

    while not done:

        iter += 1
        td = _normalize(td * np.dot(np.transpose(wt), counts / (np.dot(wt, td) + eps)))

        wt = _normalize(wt * np.dot(counts / (np.dot(wt, td) + eps), np.transpose(td)))

        # Recompute new Free energy
        E = np.sum(counts * np.log(np.dot(wt, td) + eps), axis=0)
        F_new = np.sum(E) / tot

        F.append(F_new)

        # Convergence check
        rel_ch = (F[iter] - F[iter - 1]) / abs(F[iter - 1])
        if rel_ch < epsilon and iter > 5:
            done = True

        if iter > 1000:
            done = True

    return wt, td, F


def _normalize(M):

    z = np.sum(M, axis=0)
    s = z + (z == 0).astype(float)

    return M/s


def get_style(path, wt, no_color):

    features = get_features(path)
    if len(features.shape)<2:
        features = np.asarray([features])
    FFF = np.matrix.transpose(np.asarray(features))
    if no_color:
        FFF = FFF[0:-44, :]
    # Perform inference
    td, F = _plsa_inference(FFF, 2, 0.0000001, wt)

    return td


def _plsa_inference(counts, T, epsilon, wt):
    eps = np.finfo(float).eps

    W, D = counts.shape

    # init p(z|d)
    td = _normalize(np.ones([T, D]))

    tot = np.sum(counts)
    done = False
    iter = 0

    # Free Energy init
    E = np.sum(counts * np.log(np.dot(wt, td) + eps), axis=0)
    F = [np.sum(E) / tot]

    while not done:

        iter += 1
        td = _normalize(td * np.dot(np.transpose(wt), counts / (np.dot(wt, td) + eps)))

        # Recompute new Free energy
        E = np.sum(counts * np.log(np.dot(wt, td) + eps), axis=0)
        F_new = np.sum(E) / tot

        F.append(F_new)

        # Convergence check
        rel_ch = (F[iter] - F[iter - 1]) / abs(F[iter - 1])
        if rel_ch < epsilon and iter > 5:
            done = True

        if iter > 1000:
            done = True

    return td, F


def query(q_stile, tds):

    thr = 1.25
    tds = np.squeeze(tds)
    query_mod = np.array([q_stile, 1 - q_stile])

    L = len(tds)

    dist = [np.linalg.norm(query_mod - tds[i, :]) for i in range(L)]

    # Find closest indices
    idx = np.argsort(dist)
    val = np.sort(dist)

    idxok = idx[val < thr]

    return idxok


def knnsearch(features, test_instance, k, distance_type='euclidean'):
    distances = []
    for x in range(len(features)):
        if distance_type=='euclidean':
            dist = euclidean(np.float32(test_instance), np.float32(features[x]))
        elif distance_type=='cosine':
            dist = scipy.spatial.distance.cosine(np.float32(test_instance), np.float32(features[x]))
        elif distance_type=='hamming':
            dist = scipy.spatial.distance.hamming(test_instance,features[x])

        distances.append(dist)
    idx_distances = [i[0] for i in sorted(enumerate(distances), key=lambda x:x[1])]
    neighbors = []
    ng_distances = []
    ng_idx = []
    for x in range(k):
        ng_idx.append(idx_distances[x])
        neighbors.append(features[ng_idx[x]])
        ng_distances.append(distances[ng_idx[x]])
    return ng_idx, neighbors, ng_distances