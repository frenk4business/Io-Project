"""
models
------
Logistic regression training and spatial cross-validation.

⚠️  NEVER use random splits on this dataset.
    Always use SpatialLatitudeFoldCV. See CLAUDE.md.
"""

from models.spatial_cv import SpatialLatitudeFoldCV
from models.train import train_with_spatial_cv, save_model, load_model

__all__ = [
    "SpatialLatitudeFoldCV",
    "train_with_spatial_cv",
    "save_model",
    "load_model",
]
