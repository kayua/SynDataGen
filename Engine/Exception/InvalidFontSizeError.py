class InvalidFontSizeError(Exception):
    """Exception raised for invalid font size."""

    def __init__(self, message="Invalid font_size. It must be an integer."):
        super().__init__(message)