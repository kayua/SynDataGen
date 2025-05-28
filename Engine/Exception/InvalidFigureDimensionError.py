class InvalidFigureDimensionError(Exception):
    """Exception raised for invalid figure dimension."""

    def __init__(self, message="Invalid figure_dimension. It must be a tuple of two integers."):
        super().__init__(message)