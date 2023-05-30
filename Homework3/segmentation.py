import numpy as np
import random
import math
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float
from skimage import color


### Clustering Methods for 1-D points
def kmeans(features, k, num_iters=500):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """
    print(features.shape)
    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        for i in range(N):
            distances = np.linalg.norm(features[i] - centers, axis=1)
            assignments[i] = np.argmin(distances)

        # Step 3: Update cluster centers
        new_centers = np.empty_like(centers)
        for j in range(k):
            cluster_points = features[assignments == j]
            new_centers[j] = np.mean(cluster_points, axis=0)

        # Step 4: Check if cluster assignments did not change
        if np.array_equal(centers, new_centers):
            break

        centers = new_centers
        ### END YOUR CODE

    return assignments


### Clustering Methods for colorful image
import numpy as np


def kmeans_color(features, k, num_iters=500):
    H, W = features.shape[0], features.shape[1]
    features_new = features.reshape(-1, 3)
    N = features_new.shape[0]  # 像素个数
    assignments = np.zeros(N, dtype=np.uint32)

    # 选择初始聚类中心
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features_new[idxs]

    for n in range(num_iters):
        dis2c = []

        # 计算每个样本到各个聚类中心的距离的平方
        for i in range(k):
            ds = np.sum(np.square(features_new - centers[i]),
                        axis=1,
                        keepdims=True)
            dis2c.append(ds)
        dis2c = np.concatenate(dis2c, axis=1)

        # 将样本分配到最近的聚类中心
        assignments = np.argmin(dis2c, axis=1)

        new_centers = []

        # 计算每个聚类的新中心
        for i in range(k):
            new_centers.append(np.mean(features_new[assignments == i], axis=0))

        # 检查聚类中心是否变化，如果未变化则停止迭代
        if np.allclose(centers, new_centers):
            break
        else:
            centers = new_centers

    assignments = assignments.reshape(H, W)

    return assignments


#找每个点最后会收敛到的地方（peak）
def findpeak(data, idx, r):
    t = 0.01
    shift = np.array([1])
    data_point = data[:, idx]
    dataT = data.T
    data_pointT = data_point.T
    data_pointT = data_pointT.reshape(1, 3)

    # 迭代直到shift小于阈值t
    while (shift > t).any():
        # 计算当前点和所有点之间的距离
        ds = np.sqrt(
            np.sum(np.square(dataT - data_point), axis=1, keepdims=True))

        # 在半径r内筛选出点，并计算均值向量（也可尝试高斯加权）
        point_in = dataT[np.where(ds < r)[0]]
        point_new = np.average(point_in, axis=0)

        # 更新当前点，并计算新的shift值
        shift = np.linalg.norm(data_point - point_new)

    data_point = point_new.reshape((1, 3))
    return data_point.T


# Mean shift algorithm
# 可以改写代码，鼓励自己的想法，但请保证输入输出与notebook一致
def meanshift(data, r):
    labels = np.zeros(len(data.T))
    peaks = []  # 聚集的类中心
    label_no = 1  # 当前label
    labels[0] = label_no

    # 对第一个索引调用findpeak函数
    peak = findpeak(data, 0, r)
    peakT = np.concatenate(peak, axis=0).T
    peaks.append(peakT)

    # 遍历每个数据点
    for idx in range(0, len(data.T)):
        # 寻找当前点的峰值点（peak）
        # 并实时检查当前peak是否会收敛到一个新的聚类中心（与已有peaks进行比较）
        # 如果是，更新label_no、peaks和labels，继续下一个数据点的处理
        # 如果不是，当前点属于已有类，继续下一个数据点的处理
        peakT = np.concatenate(findpeak(data, idx, r), axis=0).T
        flag = True
        for i in range(len(peaks)):
            if np.linalg.norm(peaks[i] - peakT) < r:
                flag = False
        if flag:
            peaks.append(peakT)
        for i in range(len(peaks)):
            if np.linalg.norm(peaks[i] - peakT) < r:
                label_no = i + 1
        labels[idx] = label_no

    return labels, np.array(peaks).T


# image segmentation
def segmIm(img, r):
    # Image gets reshaped to a 2D array
    img_reshaped = np.reshape(img, (img.shape[0] * img.shape[1], 3))

    # We will work now with CIELAB images
    imglab = color.rgb2lab(img_reshaped)
    # segmented_image is declared
    segmented_image = np.zeros((img_reshaped.shape[0], img_reshaped.shape[1]))

    labels, peaks = meanshift(imglab.T, r)
    # Labels are reshaped to only one column for easier handling
    labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

    # We iterate through every possible peak and its corresponding label
    for label in range(0, peaks.shape[1]):
        # Obtain indices for the current label in labels array
        inds = np.where(labels_reshaped == label + 1)[0]

        # The segmented image gets indexed peaks for the corresponding label
        corresponding_peak = peaks[:, label]
        segmented_image[inds, :] = corresponding_peak
    # The segmented image gets reshaped and turn back into RGB for display
    segmented_image = np.reshape(segmented_image,
                                 (img.shape[0], img.shape[1], 3))

    res_img = color.lab2rgb(segmented_image)
    res_img = color.rgb2gray(res_img)
    return res_img


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    mask_end = mask_gt - mask
    count = len(mask_end[np.where(mask_end == 0)])
    accuracy = count / (mask_gt.shape[0] * mask_gt.shape[1])
    ### END YOUR CODE

    return accuracy
