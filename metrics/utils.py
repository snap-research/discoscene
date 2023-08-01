# python3.7
"""Utility functions used for computing metrics."""

import numpy as np
import scipy.linalg

import torch

__all__ = [
    'compute_fid', 'compute_fid_from_feature', 'kid_kernel',
    'compute_kid_from_feature', 'compute_is', 'compute_pairwise_distance',
    'compute_gan_precision_recall'
]


def random_sample(array, size=1, replace=True):
    """Randomly pick `size` samples from `array`.

    Args:
        array: `numpy.ndarray` or `torch.Tensor`, the array to be sampled from.
        size: `int`, number of samples.
        replace: `bool`, whether to sample with replacement.

    Returns:
        `numpy.ndarray` or `torch.Tensor` with shape [num_samples, ndim].
    """
    return array[np.random.choice(len(array), size=size, replace=replace)]


def compute_fid(fake_mean, fake_cov, real_mean, real_cov):
    """Computes FID based on the statistics of fake and real data.

    FID metric is introduced in paper https://arxiv.org/pdf/1706.08500.pdf,
    which measures the distance between real data distribution and the
    synthesized data distribution.

    Given the mean and covariance (fake_mean, fake_cov) of fake data and
    (real_mean, real_cov) of real data, the FID metric can be computed by

    d^2 = ||fake_mean - real_mean||_2^2 +
          Trace(fake_cov + real_cov - 2(fake_cov @ real_cov)^0.5)

    Args:
        fake_mean: The mean of features extracted from fake data.
        fake_cov: The covariance of features extracted from fake data.
        real_mean: The mean of features extracted from real data.
        real_cov: The covariance of features extracted from real data.

    Returns:
        A real number, suggesting the FID value.
    """
    fid = np.square(fake_mean - real_mean).sum()
    temp = scipy.linalg.sqrtm(np.dot(fake_cov, real_cov))
    fid += np.real(np.trace(fake_cov + real_cov - 2 * temp))
    return float(fid)


def compute_fid_from_feature(fake_features, real_features):
    """Computes FID based on the features extracted from fake and real data.

    FID metric is introduced in paper https://arxiv.org/pdf/1706.08500.pdf,
    which measures the distance between real data distribution and the
    synthesized data distribution.

    Args:
        fake_features: The features extracted from fake data.
        real_features: The features extracted from real data.

    Returns:
        A real number, suggesting the FID value.
    """
    fake_mean = np.mean(fake_features, axis=0)
    fake_cov = np.cov(fake_features, rowvar=False)
    real_mean = np.mean(real_features, axis=0)
    real_cov = np.cov(real_features, rowvar=False)

    return compute_fid(fake_mean, fake_cov, real_mean, real_cov)


def kid_kernel(x, y):
    """KID kernel introduced in https://arxiv.org/pdf/1801.01401.pdf.

    k(x, y) = (1/ndim * x @ y + 1)^3

    Args:
        x: `numpy.ndarray` or `torch.Tensor` with shape [num_samples, ndim].
        y: `numpy.ndarray` or `torch.Tensor` with shape [num_samples, ndim].

    Returns:
        `numpy.ndarray` or `torch.Tensor` with shape [num_samples, num_samples].
    """
    ndim = x.shape[1]  # number of dimensionality
    return (x @ y.T / ndim + 1) ** 3


def compute_kid_from_feature(fake_features,
                             real_features,
                             num_subsets=100,
                             max_subset_size=1000):
    """Computes Kernel Inception Distance (KID) based on the extracted features.

    KID metric is introduced in https://arxiv.org/pdf/1801.01401.pdf, with
    official code https://github.com/mbinkowski/MMD-GAN.

    Args:
        fake_features: `numpy.ndarray` or `torch.Tensor`, the features extracted
            from fake data.
        real_features: `numpy.ndarray` or `torch.Tensor`, the features extracted
            from real data.
        num_subsets: Number of subsets. (default: 100)
        max_subset_size: The maximum size of a subset. (default: 1000)

    Returns:
        A real number, suggesting the KID value.
    """
    num_samples = min(fake_features.shape[0], real_features.shape[0],
                      max_subset_size)

    total = 0
    for _subset_idx in range(num_subsets):
        x = random_sample(fake_features, num_samples, replace=False)
        y = random_sample(real_features, num_samples, replace=False)
        sum_kxx_kyy = kid_kernel(x, x) + kid_kernel(y, y)  # k(x,x) + k(y,y)
        kxy = kid_kernel(x, y)  # k(x,y)
        temp = sum_kxx_kyy.sum() - np.diag(sum_kxx_kyy).sum()
        temp = temp / (num_samples - 1)
        total += temp - 2 * kxy.sum() / num_samples
    kid = total / num_subsets / num_samples
    return float(kid)


def compute_is(probs, num_splits):
    """Computes Inception Score (IS) based on inception prediction.

    IS metric is introduced in

    https://proceedings.neurips.cc/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf

    with official code

    https://github.com/openai/improved-gan/tree/master/inception_score

    Args:
        probs: Probabilities predicted from generated samples from inception
            model.
        num_splits: Number of splits (sub-sampling), within each of which the
            KL divergence is computed.

    Returns:
        A two-element tuple, suggesting the mean and standard deviation of the
            Inception Score.
    """
    scores = []
    interval = probs.shape[0] // num_splits
    for i in range(num_splits):
        split = probs[i * interval:(i + 1) * interval]
        split_mean = np.mean(split, axis=0, keepdims=True)
        kl = split * (np.log(split) - np.log(split_mean))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


def compute_pairwise_distance(row_features,
                              col_features,
                              dist_type='l2',
                              use_cuda=True):
    """Computes pair-wise distance between features.

    Args:
        row_features: A tensor, with shape [R, dim].
        col_features: A tensor, with shape [C, dim].
        dist_type: Type of distance, which is case insensitive. Only `l2` and
            `cos` are supported for now. (default: `l2`)
        use_cuda: Whether to use CUDA to speed up the computation. This will
            save a lot of time if the number of features is incredibly large.
            But please make sure the GPU memory does not complain.
            (default: True)

    Returns:
        A tensor, with shape [R, C], where each entry represents a distance
            between one sample from `row_features` and another from
            `col_features`.
    """
    dist_type = dist_type.lower()
    assert dist_type in ['l2', 'cos'], f'Invalid distance type `{dist_type}`!'

    if use_cuda:
        row_features = torch.as_tensor(row_features).cuda()
        col_features = torch.as_tensor(col_features).cuda()
        row_square_sum = row_features.square().sum(1, keepdim=True)
        col_square_sum = col_features.square().sum(1, keepdim=True)
        cross_dot = row_features.matmul(col_features.T)
    else:
        row_square_sum = np.square(row_features).sum(1, keepdims=True)
        col_square_sum = np.square(col_features).sum(1, keepdims=True)
        cross_dot = row_features.dot(col_features.T)

    if dist_type == 'l2':
        if use_cuda:
            distance = row_square_sum + col_square_sum.T - 2 * cross_dot
            return distance.clamp(0).detach().cpu().numpy()
        return np.maximum(row_square_sum + col_square_sum.T - 2 * cross_dot, 0)
    if dist_type == 'cos':
        if use_cuda:
            norm = row_square_sum.sqrt() * row_square_sum.sqrt().T
            return (1 - cross_dot / norm).clamp(0, 1).detach().cpu().numpy()
        norm = np.sqrt(row_square_sum) * np.sqrt(col_square_sum).T
        return np.clip(1 - cross_dot / norm, 0, 1)
    raise NotImplementedError(f'Not implemented distance type `{dist_type}`!')


def compute_gan_precision_recall(fake_features,
                                 real_features,
                                 chunk_size=10000,
                                 top_k=3):
    """Computes precision and recall for GAN evaluation.

    GAN precision and recall are introduced in

    https://arxiv.org/pdf/1904.06991.pdf, with official code

    https://github.com/kynkaat/improved-precision-and-recall-metric.

    Concretely, when computing `precision`, `real_features` are treated as a
    manifold, while `fake_features` are treated as probes. For each sample in
    the manifold, we first compute its distance to all other samples in the
    manifold and then find the `k-th` (as `top_k`) smallest distance as the
    threshold. After that, we compute its distance to all probe samples and see
    if any distance is smaller than the threshold (i.e., positive). Intuitively,
    `precision` measures the image quality (high precision means high quality)
    with "given a real sample, can we synthesize a fake sample that is very
    similar to it?".

    Similarly, when computing `recall`, `fake_features` are treated as a
    manifold, while `real_features` are treated as probes. In this way, `recall`
    measures the image variation/diversity (high recall means high diversity)
    with "given a fake sample, can we find a real image whose distance to the
    fake sample is smaller than that from other fake samples?". In other words,
    if all synthesized samples are very similar to each other, it will be hard
    to find such a real image whose distance to the fake sample is very small.

    Args:
        fake_features: The features extracted from fake data.
        real_features: The features extracted from real data.
        chunk_size: Chunk size for distance computation, which will save memory.
            (default: 10000)
        top_k: This field determines the maximum distance that will be treated
            as positive. (default: 3)

    Returns:
        A two-element tuple, suggesting the precision and recall respectively.
    """
    real_num = real_features.shape[0]
    fake_num = fake_features.shape[0]
    assert real_num > top_k and fake_num > top_k

    # Compute precision.
    thresholds = []
    for row_idx in range(0, real_num, chunk_size):
        distances = []
        for col_idx in range(0, real_num, chunk_size):
            distances.append(compute_pairwise_distance(
                real_features[row_idx:row_idx + chunk_size],
                real_features[col_idx:col_idx + chunk_size]))
        distances = np.concatenate(distances, axis=1)
        thresholds.append(np.partition(distances, top_k, axis=1)[:, top_k])
    thresholds = np.concatenate(thresholds, axis=0).reshape(1, -1)
    assert thresholds.shape == (1, real_num)

    predictions = []
    for row_idx in range(0, fake_num, chunk_size):
        distances = []
        for col_idx in range(0, real_num, chunk_size):
            distances.append(compute_pairwise_distance(
                fake_features[row_idx:row_idx + chunk_size],
                real_features[col_idx:col_idx + chunk_size]))
        distances = np.concatenate(distances, axis=1)
        predictions.append(np.any(distances <= thresholds, axis=1))
    predictions = np.concatenate(predictions, axis=0)
    assert predictions.shape == (fake_num,)
    precision = predictions.astype(np.float32).mean()

    # Compute recall.
    thresholds = []
    for row_idx in range(0, fake_num, chunk_size):
        distances = []
        for col_idx in range(0, fake_num, chunk_size):
            distances.append(compute_pairwise_distance(
                fake_features[row_idx:row_idx + chunk_size],
                fake_features[col_idx:col_idx + chunk_size]))
        distances = np.concatenate(distances, axis=1)
        thresholds.append(np.partition(distances, top_k, axis=1)[:, top_k])
    thresholds = np.concatenate(thresholds, axis=0).reshape(1, -1)
    assert thresholds.shape == (1, fake_num)

    predictions = []
    for row_idx in range(0, real_num, chunk_size):
        distances = []
        for col_idx in range(0, fake_num, chunk_size):
            distances.append(compute_pairwise_distance(
                real_features[row_idx:row_idx + chunk_size],
                fake_features[col_idx:col_idx + chunk_size]))
        distances = np.concatenate(distances, axis=1)
        predictions.append(np.any(distances <= thresholds, axis=1))
    predictions = np.concatenate(predictions, axis=0)
    assert predictions.shape == (real_num,)
    recall = predictions.astype(np.float32).mean()

    return float(precision), float(recall)
