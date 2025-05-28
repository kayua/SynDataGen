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
    import numpy

    import plotly.io as pio

    import plotly.graph_objects as go

except ImportError as error:
    print(error)
    sys.exit(-1)

DEFAULT_WIDTH_BAR = 0.2
DEFAULT_FONT_SIZE = 12
DEFAULT_TITLE_COMPARATIVE_PLOTS = "Comparativo entre dados sintéticos e reais (Média)"
DEFAULT_PLOT_FIDELITY_METRICS_LABELS = ['Similaridade de Cossenos',
                                        'Erro Médio Quadrático',
                                        'Máxima Discrepância Média']
DEFAULT_COLOR_MAP_REGRESSIVE = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


class PlotFidelityMetrics:

    def __init__(self, labels_plot_fidelity_metrics=None, color_map_bar=None, width_bar=DEFAULT_WIDTH_BAR,
                 font_size=DEFAULT_FONT_SIZE, plot_title=DEFAULT_TITLE_COMPARATIVE_PLOTS):

        if color_map_bar is None:
            color_map_bar = DEFAULT_COLOR_MAP_REGRESSIVE

        if labels_plot_fidelity_metrics is None:
            labels_plot_fidelity_metrics = DEFAULT_PLOT_FIDELITY_METRICS_LABELS

        self.labels_plot_fidelity_metrics = labels_plot_fidelity_metrics
        self.color_map_bar = color_map_bar
        self.width_bar = width_bar
        self.plot_title_axis_x = plot_title
        self.font_size = font_size

    def plot_fidelity_metrics(self, mean_squared_error_list, list_cosine_similarity, list_max_mean_discrepancy,
                              plot_filename, plot_title):

        list_metrics = [list_cosine_similarity, mean_squared_error_list, list_max_mean_discrepancy]
        new_plot_bars = go.Figure()

        for metric, metric_values, color in zip(self.labels_plot_fidelity_metrics, list_metrics, self.color_map_bar):
            try:

                metric_mean = numpy.mean(metric_values)
                metric_std = numpy.std(metric_values)
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
                print(f"Metric: {metric} Exception: {e}")

        y_label_dictionary = dict(
            title=f'Média {len(mean_squared_error_list)} dobras', tickmode='linear', tick0=0.0, dtick=0.1,
            gridcolor='black', gridwidth=.05
        )

        new_plot_bars.update_layout(
            barmode='group', title=plot_title, yaxis=y_label_dictionary,
            xaxis=dict(title=self.plot_title_axis_x), showlegend=False,
            plot_bgcolor='white'
        )

        pio.write_image(new_plot_bars, plot_filename)

    def export_fidelity_metrics_to_json(self, mean_squared_error_list, list_cosine_similarity,
                                        list_max_mean_discrepancy, json_filename):
        metrics_data = []

        for metric, metric_values, color in zip(self.labels_plot_fidelity_metrics,
                                                [list_cosine_similarity, mean_squared_error_list,
                                                 list_max_mean_discrepancy],
                                                self.color_map_bar):
            metric_mean = numpy.mean(metric_values)
            metric_std = numpy.std(metric_values)

            metrics_data.append({
                "metric": metric,
                "mean": metric_mean,
                "std": metric_std,
                "color": color
            })

        json_data = {
            "title": self.plot_title_axis_x,
            "y_axis_title": f'Média {len(mean_squared_error_list)} dobras',
            "metrics": metrics_data
        }

        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"Dados do gráfico exportados para {json_filename}.")

    def set_labels_bar_metrics(self, labels_bar_metrics):
        self.labels_plot_fidelity_metrics = labels_bar_metrics

    def set_color_map_bar(self, color_map_bar):
        self.color_map_bar = color_map_bar

    def set_width_bar(self, width_bar):
        self.width_bar = width_bar

    def set_font_size(self, font_size):
        self.font_size = font_size
