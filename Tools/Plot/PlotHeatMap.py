#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Synthetic Ocean AI - Team'
__email__ = 'syntheticoceanai@gmail.com'
__version__ = '{1}.{0}.{1}'
__initial_data__ = '2022/06/01'
__last_update__ = '2025/03/29'
__credits__ = ['Synthetic Ocean AI']


# MIT License
#
# Copyright (c) 2025 Synthetic Ocean AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


try:
    import os
    import sys

    import numpy

    import seaborn
    import logging

    import matplotlib.pyplot as plt

    from scipy.ndimage import gaussian_filter

except ImportError as error:
    logging.error(error)
    sys.exit(-1)


DEFAULT_FIGURE_SIZE = (12, 6)
DEFAULT_CMAP = 'viridis'
DEFAULT_TITLES = ('Matrix 1', 'Matrix 2')
DEFAULT_LINE_WIDTHS = 0.5
DEFAULT_LINE_COLOR = 'gray'
DEFAULT_X_TICK_LABELS = 'auto'
DEFAULT_Y_TICK_LABELS = 'auto'
DEFAULT_ANNOTATIONS = False
DEFAULT_ANNOTATIONS_FORMAT = ".0f"
DEFAULT_FONT_SCALE = 1.0
DEFAULT_COLOR_BAR = True
DEFAULT_COLOR_BAR_LABEL = None
DEFAULT_NORMALIZE = False
DEFAULT_MASK = None
DEFAULT_GRID_SHAPE = (1, 2)
DEFAULT_EXPORT_PATH = None
DEFAULT_EXPORT_DPI = 300
DEFAULT_TIGHT_LAYOUT = True
DEFAULT_STYLE = "white"



class HeatmapComparator:
    """
    HeatmapComparator provides a convenient and customizable way to plot and compare multiple 2D matrices
    using seaborn heatmaps. It supports normalization, flexible layouts, annotations, export options, and more.

    Attributes are configured via the constructor, and matrices are plotted via the `plot(*matrices)` method.

    Example Usage:
    -------------
        >>> python3
        ... import numpy as np
        ...
        ... matrix1 = np.random.rand(10, 10)
        ... matrix2 = np.random.rand(10, 10)
        ...
        ... comparator = HeatmapComparator(
        ... titles=("Random A", "Random B"),
        ... annotations=True,
        ... annot_format=".2f",
        ... export_path="output/heatmaps.png"
        ... )
        ...
        >>> comparator.plot(matrix1, matrix2)

    Notes:
    ------
    - All matrices must be 2D and of equal shape.
    - Supports auto-labeling, normalization, and exporting to image files.
    """

    def __init__(self,
                 figure_size=DEFAULT_FIGURE_SIZE,
                 color_map=DEFAULT_CMAP,
                 titles=DEFAULT_TITLES,
                 linewidths=DEFAULT_LINE_WIDTHS,
                 linecolor=DEFAULT_LINE_COLOR,
                 x_tick_labels=DEFAULT_X_TICK_LABELS,
                 y_tick_labels=DEFAULT_Y_TICK_LABELS,
                 annotations=DEFAULT_ANNOTATIONS,
                 annot_format=DEFAULT_ANNOTATIONS_FORMAT,
                 font_scale=DEFAULT_FONT_SCALE,
                 color_bar=DEFAULT_COLOR_BAR,
                 color_bar_label=DEFAULT_COLOR_BAR_LABEL,
                 normalize=DEFAULT_NORMALIZE,
                 mask=DEFAULT_MASK,
                 grid_shape=DEFAULT_GRID_SHAPE,
                 export_path=DEFAULT_EXPORT_PATH,
                 export_dpi=DEFAULT_EXPORT_DPI,
                 tight_layout=DEFAULT_TIGHT_LAYOUT,
                 style=DEFAULT_STYLE,
                 show_difference_matrix=False):
        """
        Initializes the heatmap comparator with styling, layout, and plotting options.

        Parameters
        ----------
        figure_size : tuple
            Size of the figure in inches.
        color_map : str
            Colormap used for heatmaps.
        titles : tuple of str
            Titles for each subplot.
        linewidths : float
            Width of lines that divide the cells.
        linecolor : str
            Color of the cell lines.
        x_tick_labels : str or list
            Labels for x-axis ticks ('auto', list, or False).
        y_tick_labels : str or list
            Labels for y-axis ticks ('auto', list, or False).
        annotations : bool
            Whether to annotate cells with values.
        annot_format : str
            String formatting for annotations.
        font_scale : float
            Scaling factor for font size.
        color_bar : bool
            Whether to include colorbar.
        color_bar_label : str or None
            Label for the colorbar.
        normalize : bool
            Normalize all matrices to range [0, 1] based on max value.
        mask : ndarray or None
            Boolean array to mask certain values.
        grid_shape : tuple
            Shape of the subplot grid (rows, columns).
        export_path : str or None
            Path to save the figure, if desired.
        export_dpi : int
            Resolution in DPI for export.
        tight_layout : bool
            Whether to apply tight layout to reduce padding.
        style : str
            Seaborn style to apply (e.g., 'white', 'darkgrid').
        """

        self.figure_size = figure_size
        self.color_map = color_map
        self.titles = titles
        self.line_widths = linewidths
        self.line_color = linecolor
        self.x_tick_labels = x_tick_labels
        self.y_tick_labels = y_tick_labels
        self.annotations = annotations
        self.annotation_format = annot_format
        self.font_scale = font_scale
        self.color_bar = color_bar
        self.color_bar_label = color_bar_label
        self.normalize = normalize
        self.mask = mask
        self.grid_shape = grid_shape
        self.export_path = export_path
        self.export_dpi = export_dpi
        self.tight_layout = tight_layout
        self.style = style
        self.show_difference_matrix = show_difference_matrix

        self._setup_style()

    def _setup_style(self):
        """
        Configures the visual style and font scaling for the heatmaps.
        This affects all subsequent plots in the current session.

        - `set_context` defines the base font scaling across elements.
        - `set_style` controls the background grid and aesthetics.

        Raises:
        -------
        No exceptions are expected here unless seaborn is misconfigured.
        """
        seaborn.set_context("notebook", font_scale=self.font_scale)
        seaborn.set_style(self.style)

    def _validate_and_prepare(self, matrices):
        """
        Validates that all input matrices are 2D, NumPy-compatible, and of equal shape.
        Optionally normalizes each matrix individually to [0, 1].

        Parameters
        ----------
        matrices : list of arrays
            A variable-length list of array-like objects to be plotted.

        Returns
        -------
        list of ndarray
            Cleaned, validated, and possibly normalized matrices.

        Raises
        ------
        TypeError if the input cannot be converted to NumPy arrays.
        ValueError if matrix dimensions are not 2D or shapes are mismatched.
        """
        matrices = [numpy.array(m) for m in matrices]
        shapes = [m.shape for m in matrices]

        if len(set(shapes)) != 1:
            raise ValueError("All matrices must have the same shape.")

        if self.normalize:
            matrices = [m / numpy.max(m) if numpy.max(m) != 0 else m for m in matrices]

        return matrices

    def _create_figure_and_axes(self, n_matrices):
        """
        Initializes the matplotlib Figure and Axis objects needed to display the matrices.

        Parameters
        ----------
        n_matrices : int
            Number of matrices to be plotted.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object to contain subplots.
        axs : np.ndarray of Axes
            Flattened list of axes for plotting.

        Raises
        ------
        ValueError if the grid shape is too small for the number of matrices.
        """
        rows, cols = self.grid_shape
        if rows * cols < n_matrices:
            raise ValueError("Grid shape is insufficient for the number of matrices.")

        fig, axs = plt.subplots(rows, cols, figsize=self.figure_size)
        axs = numpy.array(axs).flatten() if isinstance(axs, numpy.ndarray) else numpy.array([axs])
        return fig, axs

    def _plot_matrix_on_axis(self, ax, matrix, title):
        """
        Renders a single matrix on the given axis using seaborn's heatmap API.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The subplot axis to draw on.
        matrix : np.ndarray
            The matrix to visualize.
        title : str
            Title displayed above the subplot.

        Notes
        -----
        - Tick labels and annotations depend on user configuration.
        - Colorbar is attached if enabled.
        """
        seaborn.heatmap(matrix,
                        ax=ax,
                        cmap=self.color_map,
                        linewidths=self.line_widths,
                        linecolor=self.line_color,
                        xticklabels=self.x_tick_labels,
                        yticklabels=self.y_tick_labels,
                        annot=self.annotations,
                        fmt=self.annotation_format,
                        cbar=self.color_bar,
                        mask=self.mask,
                        cbar_kws={'label': self.color_bar_label} if self.color_bar_label else None)

        ax.set_title(title)

        if not self.x_tick_labels and not self.y_tick_labels:
            ax.axis('off')

    def _finalize_plot(self, fig, axs, n_used_axes):
        """
        Handles the final layout adjustments, export to file (if enabled), and display.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The top-level figure.
        axs : np.ndarray of Axes
            All subplot axes created.
        n_used_axes : int
            Number of actual data plots used. Remaining axes will be hidden.

        Raises
        ------
        IOError if exporting the figure fails due to file or directory issues.
        """
        for ax in axs[n_used_axes:]:
            ax.axis('off')

        if self.tight_layout:
            fig.tight_layout()

        if self.export_path:
            os.makedirs(os.path.dirname(self.export_path), exist_ok=True)
            fig.savefig(self.export_path, dpi=self.export_dpi)


    def plot(self, *matrices):
        """
        Plots multiple matrices side-by-side in a single figure using the configured settings.

        Parameters
        ----------
        *matrices : array-like
            Any number of 2D arrays (all must have the same shape).

        Behavior
        --------
            - Validates input shapes and types.
            - Optionally normalizes values.
            - Creates subplots and plots each matrix.
            - Applies labels, annotations, and colorbars.
            - Displays the plot and optionally saves to disk.

        Raises
        ------
            ValueError if input validation fails.
            IOError if exporting fails.
        """

        matrices = self._validate_and_prepare(matrices)

        if self.show_difference_matrix:

            if len(matrices) != 2:
                raise ValueError("Difference matrix plot requires exactly two input matrices.")

            diff_matrix = numpy.abs(matrices[0] - matrices[1])
            diff_matrix = gaussian_filter(diff_matrix, sigma=1)

            matrices = [*matrices, diff_matrix]
            self.titles = list(self.titles) + ["Contrast Map (Smoothed)"]

            if len(matrices) > self.grid_shape[0] * self.grid_shape[1]:

                new_cols = self.grid_shape[1]
                new_rows = (len(matrices) + new_cols - 1) // new_cols
                self.grid_shape = (new_rows, new_cols)

        fig, axs = self._create_figure_and_axes(len(matrices))

        for idx, (matrix, ax) in enumerate(zip(matrices, axs)):
            title = self.titles[idx] if idx < len(self.titles) else f'Matrix {idx+1}'
            self._plot_matrix_on_axis(ax, matrix, title)

        self._finalize_plot(fig, axs, len(matrices))

    @property
    def figure_size(self):
        """tuple: Width and height in inches of the output figure (e.g., (12, 6))."""
        return self._figure_size

    @figure_size.setter
    def figure_size(self, value):
        if not (isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)):
            raise ValueError("figure_size must be a tuple of two numeric values, e.g., (12, 6).")
        self._figure_size = value

    @property
    def color_map(self):
        """str: Colormap used for the heatmaps (e.g., 'viridis', 'plasma', 'coolwarm')."""
        return self._color_map

    @color_map.setter
    def color_map(self, value):
        if not isinstance(value, str):
            raise ValueError("color_map must be a string.")
        self._color_map = value

    @property
    def font_scale(self):
        """float: Scaling factor for all font elements in the plot."""
        return self._font_scale

    @font_scale.setter
    def font_scale(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("font_scale must be a positive number.")
        self._font_scale = value

    @property
    def annotations(self):
        """bool: Whether to show cell values in the heatmap."""
        return self._annotations

    @annotations.setter
    def annotations(self, value):
        if not isinstance(value, bool):
            raise ValueError("annotations must be a boolean.")
        self._annotations = value

    @property
    def titles(self):
        """tuple or list: Titles for each subplot."""
        return self._titles

    @titles.setter
    def titles(self, value):
        if not isinstance(value, (list, tuple)):
            raise ValueError("titles must be a list or tuple of strings.")
        self._titles = value

    @property
    def grid_shape(self):
        """tuple: Grid shape (rows, cols) for subplot arrangement."""
        return self._grid_shape

    @grid_shape.setter
    def grid_shape(self, value):
        if not (isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) and v > 0 for v in value)):
            raise ValueError("grid_shape must be a tuple of two positive integers (rows, cols).")
        self._grid_shape = value

    @property
    def export_path(self):
        """str or None: Optional file path to export the figure as an image."""
        return self._export_path

    @export_path.setter
    def export_path(self, value):
        if value is not None and not isinstance(value, str):
            raise ValueError("export_path must be a string or None.")
        self._export_path = value


