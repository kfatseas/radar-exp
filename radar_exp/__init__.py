"""Top level package for the radar experiment framework.

This package contains submodules for signal processing (dsp),
detection, feature extraction, classification, metrics, visualisation
and experiment runners.  Users should typically not import from this
module directly; instead import from the subpackages, for example:

```python
from radar_exp.dsp.rd_map import compute_range_doppler_map
from radar_exp.detect.pointcloud import Point, Object
```
"""

__all__ = [
    "io",
    "dsp",
    "detect",
    "features",
    "models",
    "metrics",
    "viz",
    "exp",
]