"""DeNoiseRaw -- physics-aware noise reduction for DSLR and mirrorless RAW files.

Quick start::

    from denoiseraw import denoise_file, DenoiseSettings

    result = denoise_file("IMG_1234.CR2", DenoiseSettings(method="classical"))
    print(result.metrics)

See the README for the command-line interface and the training workflow.
"""

__version__ = "0.1.0"

from .noise.profile import NoiseProfile, ProfileBank
from .pipeline import DenoiseResult, DenoiseSettings, denoise_file, denoise_raw, write_outputs
from .rawio.loader import RawImage, load_raw

__all__ = [
    "DenoiseResult",
    "DenoiseSettings",
    "NoiseProfile",
    "ProfileBank",
    "RawImage",
    "__version__",
    "denoise_file",
    "denoise_raw",
    "load_raw",
    "write_outputs",
]
