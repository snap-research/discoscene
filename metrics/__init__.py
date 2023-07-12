# python3.7
"""Collects all metrics."""

from .gan_snapshot import GANSnapshot
from .fid import FIDMetric as FID
from .fid import FID50K
from .fid import FID50KFull
from .inception_score import ISMetric as IS
from .inception_score import IS50K
from .intra_class_fid import ICFIDMetric as ICFID
from .intra_class_fid import ICFID50K
from .intra_class_fid import ICFID50KFull
from .kid import KIDMetric as KID
from .kid import KID50K
from .kid import KID50KFull
from .gan_pr import GANPRMetric as GANPR
from .gan_pr import GANPR50K
from .gan_pr import GANPR50KFull
from .equivariance import EquivarianceMetric
from .equivariance import EQTMetric
from .equivariance import EQT50K
from .equivariance import EQTFracMetric
from .equivariance import EQTFrac50K
from .equivariance import EQRMetric
from .equivariance import EQR50K

__all__ = ['build_metric']

_METRICS = {
    'GANSnapshot': GANSnapshot,
    'FID': FID,
    'FID50K': FID50K,
    'FID50KFull': FID50KFull,
    'IS': IS,
    'IS50K': IS50K,
    'ICFID': ICFID,
    'ICFID50K': ICFID50K,
    'ICFID50KFull': ICFID50KFull,
    'KID': KID,
    'KID50K': KID50K,
    'KID50KFull': KID50KFull,
    'GANPR': GANPR,
    'GANPR50K': GANPR50K,
    'GANPR50KFull': GANPR50KFull,
    'Equivariance': EquivarianceMetric,
    'EQT': EQTMetric,
    'EQT50K': EQT50K,
    'EQTFrac': EQTFracMetric,
    'EQTFrac50K': EQTFrac50K,
    'EQR': EQRMetric,
    'EQR50K': EQR50K
}


def build_metric(metric_type, **kwargs):
    """Builds a metric evaluator based on its class type.

    Args:
        metric_type: Type of the metric, which is case sensitive.
        **kwargs: Configurations used to build the metric.

    Raises:
        ValueError: If the `metric_type` is not supported.
    """
    if metric_type not in _METRICS:
        raise ValueError(f'Invalid metric type: `{metric_type}`!\n'
                         f'Types allowed: {list(_METRICS)}.')
    return _METRICS[metric_type](**kwargs)
