class InvalidLegendFontSizeError(Exception):
    """Exception raised for invalid legend font size."""

    def __init__(self, message="Invalid legend_font_size. It must be an integer."):
        super().__init__(message)
