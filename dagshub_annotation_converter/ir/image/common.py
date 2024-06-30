from enum import Enum


class NormalizationState(Enum):
    """Defines normalization of the annotation values"""

    NORMALIZED = (0,)
    """ Annotation values are in between 0 and 1 relative to the image dimensions """
    DENORMALIZED = (1,)
    """ Annotation values are in absolute pixel values """
