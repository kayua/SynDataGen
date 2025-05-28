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
    import logging

    from typing import List
    from abc import abstractmethod
    import Tools.config as config

except ImportError as error:
    logging.error(error)
    sys.exit(-1)

class Plot:
    """
    Abstract base class for plot generation.

    Provides core functionality for:
    - Managing plot output directory
    - Reading and parsing JSON data files
    - Template method pattern for value extraction

    Subclasses must implement the _get_values() method to specify how to extract
    relevant data from the JSON structure for their specific plot type.
    """

    def __init__(self):
        """
        Initialize the Plot instance.

        Sets up the default plots directory from the module configuration.
        """
        self.plots_dir = config.PLOTS_DIRECTORY

    def _read_data(self, input_files: List[str]):
        """
        Read and process multiple JSON data files.

        Args:
            input_files: List of paths to JSON files containing plot data

        Returns:
            Dictionary mapping each file path to its extracted values.
            The structure of the values is determined by the _get_values() implementation.

        Example:
            {
                'data/file1.json': [values1],
                'data/file2.json': [values2]
            }
        """
        values = {}
        for file_path in input_files:
             
            with open(file_path, 'r') as file:
                data = json.load(file)
                values[file_path] = self._get_values(data)
                 
        return values

    @abstractmethod
    def _get_values(self, data):
        """
        Abstract method for extracting plot values from JSON data.

        Args:
            data: Parsed JSON data as a dictionary

        Returns:
            List of values to be plotted. The exact structure depends on
            the specific plot implementation.

        Note:
            Must be implemented by concrete subclasses to specify how to
            extract relevant data from the input JSON structure.
        """
        pass