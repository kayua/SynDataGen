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
    import json
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

except ImportError as error:
    print(error)
    sys.exit(-1)


DEFAULT_WIDTH_BAR = 0.2
DEFAULT_FONT_SIZE = 12
DEFAULT_MATRIX_CONFUSION_ROTATION_LEGENDS = 45
DEFAULT_PLOT_CLASSIFIER_METRICS_LABELS = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']

DEFAULT_COLOR_MAP = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                     '#17becf']

class PlotClassificationMetrics:

    def __init__(self, labels_bar_metrics=None, color_map_bar=None, width_bar=DEFAULT_WIDTH_BAR,
                 font_size=DEFAULT_FONT_SIZE):

        if color_map_bar is None:
            color_map_bar = DEFAULT_COLOR_MAP

        if labels_bar_metrics is None:
            labels_bar_metrics = DEFAULT_PLOT_CLASSIFIER_METRICS_LABELS

        self.labels_bar_metrics = labels_bar_metrics
        self.color_map_bar = color_map_bar
        self.width_bar = width_bar
        self.font_size = font_size

    def plot_classifier_metrics(self,
                                classifier_type,
                                accuracy_list,
                                precision_list,
                                recall_list,
                                f1_score_list,
                                plot_filename,
                                plot_title,
                                type_of_classifier):

        list_all_metrics = [accuracy_list, precision_list, recall_list, f1_score_list]
        new_plot_bars = go.Figure()
        color_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        if type_of_classifier == "TR_As":
            metrics_name = ['Acurácia-TR_As', 'Precisão-TR_As', 'Recall-TR_As', 'F1-Score-TR_As']

        else:
            metrics_name = ['Acurácia-TS_Ar', 'Precisão-TS_Ar', 'Recall-TS_Ar', 'F1-Score-TS_Ar']

        for metric, metric_values, color in zip(metrics_name, list_all_metrics, color_map):

            try:
                metric_mean = np.mean(metric_values)
                metric_std = np.std(metric_values)
                new_plot_bars.add_trace(go.Bar(
                    x=[metric], y=[metric_mean], name=metric, marker=dict(color=color),
                    error_y=dict(type='constant', value=metric_std, visible=True),
                    width=self.width_bar
                ))

                new_plot_bars.add_annotation(
                    x=metric, y=metric_mean + metric_std, xref="x", yref="y",
                    text=f' {metric_std:.4f}', showarrow=False,
                    font=dict(color='black', size=self.font_size),
                    xanchor='center', yanchor='bottom'
                )
            except Exception as e:
                print(f"Metric {metric} error: {e}")

        y_label_dictionary = dict(
            title=f'Média {len(accuracy_list)} dobras', tickmode='linear', tick0=0.0, dtick=0.1,
            gridcolor='black', gridwidth=.05
        )

        new_plot_bars.update_layout(
            barmode='group', title=plot_title, yaxis=y_label_dictionary,
            xaxis=dict(title=f'Desempenho com {classifier_type}'), showlegend=False,
            plot_bgcolor='white'
        )

        pio.write_image(new_plot_bars, plot_filename)

        return new_plot_bars

    @staticmethod
    def export_plot_to_json(figure, json_filename):
        plot_dict = figure.to_plotly_json()

        with open(json_filename, 'w') as f:
            json.dump(plot_dict, f, indent=4)
        print(f"Gráfico exportado para {json_filename}.")

    def plot_and_export_classifier_metrics(self, classifier_type, accuracy_list, precision_list, recall_list, f1_score_list,
                                           plot_filename, json_filename, plot_title, type_of_classifier):
        figure = self.plot_classifier_metrics(classifier_type, accuracy_list, precision_list, recall_list, f1_score_list,
                                              plot_filename, plot_title, type_of_classifier)

        self.export_plot_to_json(figure, json_filename)

    def set_labels_bar_metrics(self, labels_bar_metrics):
        self.labels_bar_metrics = labels_bar_metrics

    def set_color_map_bar(self, color_map_bar):
        self.color_map_bar = color_map_bar

    def set_width_bar(self, width_bar):
        self.width_bar = width_bar

    def set_font_size(self, font_size):
        self.font_size = font_size
