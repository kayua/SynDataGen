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
    import sys
    import plotly.io as pio
    from itertools import cycle
    from sklearn.metrics import auc
    import plotly.graph_objects as go
    from sklearn.metrics import roc_curve

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_COLORS_ROC = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
DEFAULT_FONT_SIZE = 12
DEFAULT_TITLE_ROC_PLOT = "Multiclass ROC Curve"

class PlotROCMulticlass:

    def __init__(self, colors_roc=None, font_size=DEFAULT_FONT_SIZE, plot_title=DEFAULT_TITLE_ROC_PLOT):

        if colors_roc is None:
            colors_roc = DEFAULT_COLORS_ROC

        self.colors_roc = colors_roc
        self.font_size = font_size
        self.plot_title = plot_title

    def plot_roc_curve(self, y_true, y_score, number_classes, plot_filename):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(number_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First plot
        fig = go.Figure()

        # Plot ROC curve for each class
        colors = cycle(self.colors_roc)
        for i, color in zip(range(number_classes), colors):
            fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode='lines',
                                     name=f'Class {i} (area = {roc_auc[i]:0.2f})',
                                     line=dict(color=color, width=2)))

        # Plot micro-average ROC curve
        fig.add_trace(go.Scatter(x=fpr["micro"], y=tpr["micro"], mode='lines',
                                 name=f'Micro-average ROC (area = {roc_auc["micro"]:0.2f})',
                                 line=dict(color='black', dash='dash', width=2)))

        # Plot diagonal
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 line=dict(color='gray', dash='dash'),
                                 showlegend=False))

        # Customize layout
        fig.update_layout(
            title=self.plot_title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            font=dict(size=self.font_size),
            plot_bgcolor='white'
        )

        # Save plot to file
        pio.write_image(fig, plot_filename)

    def set_colors_roc(self, colors_roc):
        self.colors_roc = colors_roc

    def set_font_size(self, font_size):
        self.font_size = font_size

    def set_plot_title(self, plot_title):
        self.plot_title = plot_title