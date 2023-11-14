from collections.abc import MutableMapping

import tensorflow as tf
import numpy as np
import keras
from os import listdir
from os.path import isfile, join
import re
import pandas as pd
import argparse
import os

from tqdm import tqdm

import math
import sys

RETURN_VALUE = 0
RETURN_PATH = 1
RETURN_ALL = -1


# Core DTW
def _traceback(DTW, slope_constraint):
    i, j = np.array(DTW.shape) - 1
    p, q = [i - 1], [j - 1]

    if slope_constraint == "asymmetric":
        while (i > 1):
            tb = np.argmin((DTW[i - 1, j], DTW[i - 1, j - 1], DTW[i - 1, j - 2]))

            if (tb == 0):
                i = i - 1
            elif (tb == 1):
                i = i - 1
                j = j - 1
            elif (tb == 2):
                i = i - 1
                j = j - 2

            p.insert(0, i - 1)
            q.insert(0, j - 1)
    elif slope_constraint == "symmetric":
        while (i > 1 or j > 1):
            tb = np.argmin((DTW[i - 1, j - 1], DTW[i - 1, j], DTW[i, j - 1]))

            if (tb == 0):
                i = i - 1
                j = j - 1
            elif (tb == 1):
                i = i - 1
            elif (tb == 2):
                j = j - 1

            p.insert(0, i - 1)
            q.insert(0, j - 1)
    else:
        sys.exit("Unknown slope constraint %s" % slope_constraint)

    return (np.array(p), np.array(q))


def dtw(prototype, sample, return_flag=RETURN_VALUE, slope_constraint="asymmetric", window=None):
    """ Computes the DTW of two sequences.
    :param prototype: np array [0..b]
    :param sample: np array [0..t]
    :param extended: bool
    """
    p = prototype.shape[0]
    assert p != 0, "Prototype empty!"
    s = sample.shape[0]
    assert s != 0, "Sample empty!"

    if window is None:
        window = s

    cost = np.full((p, s), np.inf)
    for i in range(p):
        start = max(0, i - window)
        end = min(s, i + window) + 1
        cost[i, start:end] = np.linalg.norm(sample[start:end] - prototype[i], axis=1)

    DTW = _cummulative_matrix(cost, slope_constraint, window)

    if return_flag == RETURN_ALL:
        return DTW[-1, -1], cost, DTW[1:, 1:], _traceback(DTW, slope_constraint)
    elif return_flag == RETURN_PATH:
        return _traceback(DTW, slope_constraint)
    else:
        return DTW[-1, -1]


def _cummulative_matrix(cost, slope_constraint, window):
    p = cost.shape[0]
    s = cost.shape[1]

    # Note: DTW is one larger than cost and the original patterns
    DTW = np.full((p + 1, s + 1), np.inf)

    DTW[0, 0] = 0.0

    if slope_constraint == "asymmetric":
        for i in range(1, p + 1):
            if i <= window + 1:
                DTW[i, 1] = cost[i - 1, 0] + min(DTW[i - 1, 0], DTW[i - 1, 1])
            for j in range(max(2, i - window), min(s, i + window) + 1):
                DTW[i, j] = cost[i - 1, j - 1] + min(DTW[i - 1, j - 2], DTW[i - 1, j - 1], DTW[i - 1, j])
    elif slope_constraint == "symmetric":
        for i in range(1, p + 1):
            for j in range(max(1, i - window), min(s, i + window) + 1):
                DTW[i, j] = cost[i - 1, j - 1] + min(DTW[i - 1, j - 1], DTW[i, j - 1], DTW[i - 1, j])
    else:
        sys.exit("Unknown slope constraint %s" % slope_constraint)

    return DTW


def shape_dtw(prototype, sample, return_flag=RETURN_VALUE, slope_constraint="asymmetric", window=None,
              descr_ratio=0.05):
    """ Computes the shapeDTW of two sequences.
    :param prototype: np array [0..b]
    :param sample: np array [0..t]
    :param extended: bool
    """
    # shapeDTW
    # https://www.sciencedirect.com/science/article/pii/S0031320317303710

    p = prototype.shape[0]
    assert p != 0, "Prototype empty!"
    s = sample.shape[0]
    assert s != 0, "Sample empty!"

    if window is None:
        window = s

    p_feature_len = np.clip(np.round(p * descr_ratio), 5, 100).astype(int)
    s_feature_len = np.clip(np.round(s * descr_ratio), 5, 100).astype(int)

    # padding
    p_pad_front = (np.ceil(p_feature_len / 2.)).astype(int)
    p_pad_back = (np.floor(p_feature_len / 2.)).astype(int)
    s_pad_front = (np.ceil(s_feature_len / 2.)).astype(int)
    s_pad_back = (np.floor(s_feature_len / 2.)).astype(int)

    prototype_pad = np.pad(prototype, ((p_pad_front, p_pad_back), (0, 0)), mode="edge")
    sample_pad = np.pad(sample, ((s_pad_front, s_pad_back), (0, 0)), mode="edge")
    p_p = prototype_pad.shape[0]
    s_p = sample_pad.shape[0]

    cost = np.full((p, s), np.inf)
    for i in range(p):
        for j in range(max(0, i - window), min(s, i + window)):
            cost[i, j] = np.linalg.norm(sample_pad[j:j + s_feature_len] - prototype_pad[i:i + p_feature_len])

    DTW = _cummulative_matrix(cost, slope_constraint=slope_constraint, window=window)

    if return_flag == RETURN_ALL:
        return DTW[-1, -1], cost, DTW[1:, 1:], _traceback(DTW, slope_constraint)
    elif return_flag == RETURN_PATH:
        return _traceback(DTW, slope_constraint)
    else:
        return DTW[-1, -1]


# Draw helpers
def draw_graph2d(cost, DTW, path, prototype, sample):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    # plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

    # cost
    plt.subplot(2, 3, 1)
    plt.imshow(cost.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0], path[1], 'y')
    plt.xlim((-0.5, cost.shape[0] - 0.5))
    plt.ylim((-0.5, cost.shape[0] - 0.5))

    # dtw
    plt.subplot(2, 3, 2)
    plt.imshow(DTW.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0] + 1, path[1] + 1, 'y')
    plt.xlim((-0.5, DTW.shape[0] - 0.5))
    plt.ylim((-0.5, DTW.shape[0] - 0.5))

    # prototype
    plt.subplot(2, 3, 4)
    plt.plot(prototype[:, 0], prototype[:, 1], 'b-o')

    # connection
    plt.subplot(2, 3, 5)
    for i in range(0, path[0].shape[0]):
        plt.plot([prototype[path[0][i], 0], sample[path[1][i], 0]], [prototype[path[0][i], 1], sample[path[1][i], 1]],
                 'y-')
    plt.plot(sample[:, 0], sample[:, 1], 'g-o')
    plt.plot(prototype[:, 0], prototype[:, 1], 'b-o')

    # sample
    plt.subplot(2, 3, 6)
    plt.plot(sample[:, 0], sample[:, 1], 'g-o')

    plt.tight_layout()
    plt.show()


def draw_graph1d(cost, DTW, path, prototype, sample):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    # plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    p_steps = np.arange(prototype.shape[0])
    s_steps = np.arange(sample.shape[0])

    # cost
    plt.subplot(2, 3, 1)
    plt.imshow(cost.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0], path[1], 'y')
    plt.xlim((-0.5, cost.shape[0] - 0.5))
    plt.ylim((-0.5, cost.shape[0] - 0.5))

    # dtw
    plt.subplot(2, 3, 2)
    plt.imshow(DTW.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0] + 1, path[1] + 1, 'y')
    plt.xlim((-0.5, DTW.shape[0] - 0.5))
    plt.ylim((-0.5, DTW.shape[0] - 0.5))

    # prototype
    plt.subplot(2, 3, 4)
    plt.plot(p_steps, prototype[:, 0], 'b-o')

    # connection
    plt.subplot(2, 3, 5)
    for i in range(0, path[0].shape[0]):
        plt.plot([path[0][i], path[1][i]], [prototype[path[0][i], 0], sample[path[1][i], 0]], 'y-')
    plt.plot(p_steps, sample[:, 0], 'g-o')
    plt.plot(s_steps, prototype[:, 0], 'b-o')

    # sample
    plt.subplot(2, 3, 6)
    plt.plot(s_steps, sample[:, 0], 'g-o')

    plt.tight_layout()
    plt.show()

def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    return np.multiply(x, factor[:, np.newaxis, :])


def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    return flip[:, np.newaxis, :] * x[:, :, rotate_axis]


def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array(
            [CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret


def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    return ret


def window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len),
                                       pat[starts[i]:ends[i], dim]).T
    return ret


def window_warp(x, window_ratio=0.1, scales=[0.5, 2.]):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i], dim]
            window_seg = np.interp(np.linspace(0, warp_size - 1, num=int(warp_size * warp_scales[i])), window_steps,
                                   pat[window_starts[i]:window_ends[i], dim])
            end_seg = pat[window_ends[i]:, dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[i, :, dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1] - 1., num=warped.size),
                                       warped).T
    return ret


def spawner(x, labels, sigma=0.05, verbose=0):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6983028/
    # use verbose=-1 to turn off warnings
    # use verbose=1 to print out figures

    random_points = np.random.randint(low=1, high=x.shape[1] - 1, size=x.shape[0])
    window = np.ceil(x.shape[1] / 10.).astype(int)
    orig_steps = np.arange(x.shape[1])
    l = labels

    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:
            random_sample = x[np.random.choice(choices)]
            # SPAWNER splits the path into two randomly
            path1 = dtw(pat[:random_points[i]], random_sample[:random_points[i]], RETURN_PATH,
                            slope_constraint="symmetric", window=window)
            path2 = dtw(pat[random_points[i]:], random_sample[random_points[i]:], RETURN_PATH,
                            slope_constraint="symmetric", window=window)
            combined = np.concatenate((np.vstack(path1), np.vstack(path2 + random_points[i])), axis=1)
            if verbose:
                print(random_points[i])
                dtw_value, cost, DTW_map, path = dtw(pat, random_sample, return_flag=RETURN_ALL,
                                                         slope_constraint=slope_constraint, window=window)
                draw_graph1d(cost, DTW_map, path, pat, random_sample)
                draw_graph1d(cost, DTW_map, combined, pat, random_sample)
            mean = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=mean.shape[0]),
                                           mean[:, dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average" % l[i])
            ret[i, :] = pat
    return jitter(ret, sigma=sigma)


def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, verbose=0):
    # https://ieeexplore.ieee.org/document/8215569
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"

    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = labels

    ret = np.zeros_like(x)
    for i in tqdm(range(ret.shape[0])):
        # get the same class as i
        choices = np.where(l == l[i])[0]
        if choices.size > 0:
            # pick random intra-class pattern
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]

            # calculate dtw between all
            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.
                    else:
                        dtw_matrix[p, s] = dtw(prototype, sample, RETURN_VALUE,
                                                   slope_constraint=slope_constraint, window=window)

            # get medoid
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]

            # start weighted DBA
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern
                    weighted_sums += np.ones_like(weighted_sums)
                else:
                    path = dtw(medoid_pattern, random_prototypes[nid], RETURN_PATH,
                                   slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    weight = np.exp(np.log(0.5) * dtw_value / dtw_matrix[medoid_id, nearest_order[1]])
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight

            ret[i, :] = average_pattern / weighted_sums[:, np.newaxis]
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping pattern average" % l[i])
            ret[i, :] = x[i]
    return ret


# Proposed

def random_guided_warp(x, labels, slope_constraint="symmetric", use_window=True, dtw_type="normal", verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for  "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or  "normal" or "shape"

    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = labels

    ret = np.zeros_like(x)
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)
        # remove ones of different classes
        choices = np.where(l[choices] == l[i])[0]
        if choices.size > 0:
            # pick random intra-class pattern
            random_prototype = x[np.random.choice(choices)]

            if dtw_type == "shape":
                path = shape_dtw(random_prototype, pat, RETURN_PATH, slope_constraint=slope_constraint,
                                     window=window)
            else:
                path = dtw(random_prototype, pat, RETURN_PATH, slope_constraint=slope_constraint, window=window)

            # Time warp
            warped = pat[path[1]]
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=warped.shape[0]),
                                           warped[:, dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d, skipping timewarping" % l[i])
            ret[i, :] = pat
    return ret


def random_guided_warp_shape(x, labels, slope_constraint="symmetric", use_window=True):
    return random_guided_warp(x, labels, slope_constraint, use_window, dtw_type="shape")


def discriminative_guided_warp(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True,
                               dtw_type="normal", use_variable_slice=True, verbose=0):
    # use verbose = -1 to turn off warnings
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # dtw_type is for shapeDTW or DTW. "normal" or "shape"

    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    orig_steps = np.arange(x.shape[1])
    l = labels

    positive_batch = np.ceil(batch_size / 2).astype(int)
    negative_batch = np.floor(batch_size / 2).astype(int)

    ret = np.zeros_like(x)
    warp_amount = np.zeros(x.shape[0])
    for i, pat in enumerate(tqdm(x)):
        # guarentees that same one isnt selected
        choices = np.delete(np.arange(x.shape[0]), i)

        # remove ones of different classes
        positive = np.where(l[choices] == l[i])[0]
        negative = np.where(l[choices] != l[i])[0]

        if positive.size > 0 and negative.size > 0:
            pos_k = min(positive.size, positive_batch)
            neg_k = min(negative.size, negative_batch)
            positive_prototypes = x[np.random.choice(positive, pos_k, replace=False)]
            negative_prototypes = x[np.random.choice(negative, neg_k, replace=False)]

            # vector embedding and nearest prototype in one
            pos_aves = np.zeros((pos_k))
            neg_aves = np.zeros((pos_k))
            if dtw_type == "shape":
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1. / (pos_k - 1.)) * shape_dtw(pos_prot, pos_samp, RETURN_VALUE,
                                                                               slope_constraint=slope_constraint,
                                                                               window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1. / neg_k) * shape_dtw(pos_prot, neg_samp, RETURN_VALUE,
                                                                    slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = shape_dtw(positive_prototypes[selected_id], pat, RETURN_PATH,
                                     slope_constraint=slope_constraint, window=window)
            else:
                for p, pos_prot in enumerate(positive_prototypes):
                    for ps, pos_samp in enumerate(positive_prototypes):
                        if p != ps:
                            pos_aves[p] += (1. / (pos_k - 1.)) * dtw(pos_prot, pos_samp, RETURN_VALUE,
                                                                         slope_constraint=slope_constraint,
                                                                         window=window)
                    for ns, neg_samp in enumerate(negative_prototypes):
                        neg_aves[p] += (1. / neg_k) * dtw(pos_prot, neg_samp, RETURN_VALUE,
                                                              slope_constraint=slope_constraint, window=window)
                selected_id = np.argmax(neg_aves - pos_aves)
                path = dtw(positive_prototypes[selected_id], pat, RETURN_PATH,
                               slope_constraint=slope_constraint, window=window)

            # Time warp
            warped = pat[path[1]]
            warp_path_interp = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=warped.shape[0]), path[1])
            warp_amount[i] = np.sum(np.abs(orig_steps - warp_path_interp))
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(orig_steps, np.linspace(0, x.shape[1] - 1., num=warped.shape[0]),
                                           warped[:, dim]).T
        else:
            if verbose > -1:
                print("There is only one pattern of class %d" % l[i])
            ret[i, :] = pat
            warp_amount[i] = 0.
    if use_variable_slice:
        max_warp = np.max(warp_amount)
        if max_warp == 0:
            # unchanged
            ret = window_slice(ret, reduce_ratio=0.9)
        else:
            for i, pat in enumerate(ret):
                # Variable Sllicing
                ret[i] = window_slice(pat[np.newaxis, :, :], reduce_ratio=0.9 + 0.1 * warp_amount[i] / max_warp)[0]
    return ret


def discriminative_guided_warp_shape(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True):
    return discriminative_guided_warp(x, labels, batch_size, slope_constraint, use_window, dtw_type="shape")


def load_data_from_file(data_file, label_file=None, delimiter=" "):
    if label_file:
        data = np.genfromtxt(data_file, delimiter=delimiter)
        labels = np.genfromtxt(label_file, delimiter=delimiter)
        if len(labels) > 1:
            labels = labels[:,1]
    else:
        data = np.genfromtxt(data_file, delimiter=delimiter)
        labels = data[:,0]
        data = data[:,1:]
    return data, labels
    
def read_data_sets(train_file, train_label=None, test_file=None, test_label=None, test_split=0.1, delimiter=" "):
    train_data, train_labels = load_data_from_file(train_file, train_label, delimiter)
    if test_file:
        test_data, test_labels = load_data_from_file(test_file, test_label, delimiter)
    else:
        test_size = int(test_split * float(train_labels.shape[0]))
        test_data = train_data[:test_size]
        test_labels = train_labels[:test_size]
        train_data = train_data[test_size:]
        train_labels = train_labels[test_size:]
    return train_data, train_labels, test_data, test_labels


def get_datasets(args):
    # Load data
    if args.preset_files:
        if args.ucr:
            train_data_file = os.path.join(args.data_dir, args.dataset, "%s_TRAIN"%args.dataset)
            test_data_file = os.path.join(args.data_dir, args.dataset, "%s_TEST"%args.dataset)
            x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter=",")
        elif args.ucr2018:
            train_data_file = os.path.join(args.data_dir, args.dataset, "%s_TRAIN.tsv"%args.dataset)
            test_data_file = os.path.join(args.data_dir, args.dataset, "%s_TEST.tsv"%args.dataset)
            x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter="\t")
        else:
            x_train_file = os.path.join(args.data_dir, "train-%s-data.txt"%(args.dataset))
            y_train_file = os.path.join(args.data_dir, "train-%s-labels.txt"%(args.dataset))
            x_test_file = os.path.join(args.data_dir, "test-%s-data.txt"%(args.dataset))
            y_test_file = os.path.join(args.data_dir, "test-%s-labels.txt"%(args.dataset))
            x_train, y_train, x_test, y_test = read_data_sets(x_train_file, y_train_file, x_test_file, y_test_file, test_split=args.test_split, delimiter=args.delimiter)
    else:
        x_train, y_train, x_test, y_test = read_data_sets(args.train_data_file, args.train_labels_file, args.test_data_file, args.test_labels_file, test_split=args.test_split, delimiter=args.delimiter)
    
    # Normalize
    if args.normalize_input:
        x_train_max = np.nanmax(x_train)
        x_train_min = np.nanmin(x_train)
        x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
        # Test is secret
        x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.
        
    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)
    return x_train, y_train, x_test, y_test

def run_augmentation(x, y, args):
    print("Augmenting %s"%args.dataset)
    np.random.seed(args.seed)
    x_aug = x
    y_aug = y
    if args.augmentation_ratio > 0:
        augmentation_tags = "%d"%args.augmentation_ratio
        for n in range(args.augmentation_ratio):
            x_temp, augmentation_tags = augment(x, y, args)
            x_aug = np.append(x_aug, x_temp, axis=0)
            y_aug = np.append(y_aug, y, axis=0)
            print("Round %d: %s done"%(n, augmentation_tags))
        if args.extra_tag:
            augmentation_tags += "_"+args.extra_tag
    else:
        augmentation_tags = args.extra_tag
    return x_aug, y_aug, augmentation_tags

def augment(x, y, args):
    augmentation_tags = ""
    if args.jitter:
        x = jitter(x)
        augmentation_tags += "_jitter"
    if args.scaling:
        x = scaling(x)
        augmentation_tags += "_scaling"
    if args.rotation:
        x = rotation(x)
        augmentation_tags += "_rotation"
    if args.permutation:
        x = permutation(x)
        augmentation_tags += "_permutation"
    if args.randompermutation:
        x = permutation(x, seg_mode="random")
        augmentation_tags += "_randomperm"
    if args.magwarp:
        x = magnitude_warp(x)
        augmentation_tags += "_magwarp"
    if args.timewarp:
        x = time_warp(x)
        augmentation_tags += "_timewarp"
    if args.windowslice:
        x = window_slice(x)
        augmentation_tags += "_windowslice"
    if args.windowwarp:
        x = window_warp(x)
        augmentation_tags += "_windowwarp"
    if args.spawner:
        x = spawner(x, y)
        augmentation_tags += "_spawner"
    if args.dtwwarp:
        x = random_guided_warp(x, y)
        augmentation_tags += "_rgw"
    if args.shapedtwwarp:
        x = random_guided_warp_shape(x, y)
        augmentation_tags += "_rgws"
    if args.wdba:
        x = wdba(x, y)
        augmentation_tags += "_wdba"
    if args.discdtw:
        x = discriminative_guided_warp(x, y)
        augmentation_tags += "_dgw"
    if args.discsdtw:
        x = discriminative_guided_warp_shape(x, y)
        augmentation_tags += "_dgws"
    return x, augmentation_tags

dirname = "../pretrained_models_ucr/UCR_TS_Archive_2015/"
onlyfiles = [f for f in listdir(dirname)]

x_train_list = [np.load("data/TL/x_train" + str(i) + "_pfps.npy") for i in range(1,11)]
x_test_list = [np.load("data/TL/x_test" + str(i) + "_pfps.npy") for i in range(1,11)]
y_train_list = [np.load("data/TL/y_train" + str(i) + "_pfps.npy") for i in range(1,11)]
y_test_list = [np.load("data/TL/y_test" + str(i) + "_pfps.npy") for i in range(1,11)]

input_shape = x_train_list[0].shape[1:]

# aug ratios
augmentation_ratios = [0,2,4,8,12]

### transfer learn function
def transfer_learn(input_shape, prtrmod):
    
    resh = False
    if prtrmod.layers[0].input_shape[0][2] == 1:
        input_shape = (input_shape[0]*input_shape[1], 1)
        resh = True
    inp_layer = keras.layers.Input(shape=input_shape)
    prtrlay = prtrmod.layers[1:-1]
    for i in range(len(prtrlay)):
        prtrlay[i].trainable = False
    mod = keras.Sequential([inp_layer] + 
                           prtrlay + 
                           [keras.layers.Dense(units=1, activation="sigmoid")])
    
    mod.compile(
        optimizer = keras.optimizers.Adam(),
        loss = "categorical_crossentropy",
        metrics = "accuracy"
    )
    
    return(mod, resh)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs augmentation model.')
  
    # Augmentation
    # parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=True, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=True, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=True, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=True, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=True, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    
    args = parser.parse_args()
    
    for data_ind in reversed(range(len(x_train_list))):
	
        print("Data set " + str(data_ind) + "/10 \n")
        
        x_train_org = x_train_list[data_ind]
        x_test = x_test_list[data_ind]
        y_train_org = y_train_list[data_ind]-1
        y_test = y_test_list[data_ind]-1
            
        for augrat in augmentation_ratios:
            
            args.augmentation_ratio = augrat
            args.dataset = ""
            
            x_train, y_train, _ = run_augmentation(x_train_org, y_train_org, args)
    
            for flname in onlyfiles:
    
                # print("Check if results exist...\n")    
    
                if flname + "_aug_x" + str(augrat) + "_fold_" + str(data_ind) + "_pfps.csv" in listdir("output/TL_AUG"):
                    continue
    
                # if sum(mt)==10:
                  #  continue
        
                print("Transfer-learning on " + flname + "\n")
    
                fns = listdir(dirname + flname)
                fn = [fn for fn in fns  if re.match(r"best_model\.hdf5$", fn) is not None][0]
                mod = keras.models.load_model(dirname + flname + "/" + fn)
                mod, resh = transfer_learn(input_shape, mod)
                
                if resh:
                    x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2],1))
                    x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2],1))

                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=5,
                        verbose=0
                    )
                ]
        
                print("Learning network head...\n")
        
                history = mod.fit(
                    x_train,
                    y_train,
                    batch_size=64,
                    epochs=50,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose = False
                )
        
                print("Predicting...\n")

                y_hat = mod.predict(x_test, verbose = False)
        
                res = pd.DataFrame(np.c_[y_hat, y_test], 
                                   index=['Row'+str(i) for i in range(1, len(y_test_list[data_ind])+1)])
                res.columns = ['prob', 'truth']
        
                res.to_csv("output/TL_AUG/" + flname + "_aug_x" + str(augrat) + "_fold_" + str(data_ind) + "_pfps.csv")
