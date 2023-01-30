class BaseEx(Exception):
    """Base exception"""


class InvalidBackboneError(BaseEx):
    """Raised when the choice of backbone Convnet is invalid."""


class InvalidDatasetSelection(BaseEx):
    """Raised when the choice of dataset is invalid."""
