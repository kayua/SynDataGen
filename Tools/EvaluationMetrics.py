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
    import numpy

    from scipy.special import rel_entr
    from sklearn.metrics import r2_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import log_loss
    from sklearn.metrics import pairwise

    from sklearn.metrics import recall_score
    from sklearn.metrics import hamming_loss
    from sklearn.metrics import jaccard_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import zero_one_loss

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import v_measure_score
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import brier_score_loss

    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import cohen_kappa_score
    from sklearn.metrics import mutual_info_score
    from sklearn.metrics import homogeneity_score

    from scipy.spatial.distance import euclidean
    from scipy.spatial.distance import cityblock
    from scipy.spatial.distance import chebyshev
    from scipy.spatial.distance import correlation
    from scipy.spatial.distance import jensenshannon

    from sklearn.metrics import completeness_score
    from sklearn.metrics import mean_squared_error

    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics import mean_absolute_error

    from sklearn.metrics import davies_bouldin_score
    from sklearn.metrics import calinski_harabasz_score
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import explained_variance_score

    from sklearn.metrics import adjusted_mutual_info_score
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.metrics import precision_recall_fscore_support


except ImportError as error:
    print(error)
    sys.exit(-1)

class EvaluationMetrics:
    """
    A utility class providing static methods for calculating various machine learning evaluation metrics.
    This class includes metrics for regression, classification, and clustering tasks.

    All methods are static and can be used without instantiating the class.
    """

    def __init__(self):
        """
        Initializer for the EvaluationMetrics class.
        Note: This class is not meant to be instantiated as all methods are static.
        """
        pass

    @staticmethod
    def get_mean_squared_error(real_label, predicted_label):
        """
        Calculates the Mean Squared Error (MSE) between real and predicted values.

        MSE measures the average of the squares of the errors between predicted and actual values.
        Lower values indicate better model performance.

        Parameters:
            real_label (array-like): Ground truth (correct) target values
            predicted_label (array-like): Estimated target values

        Returns:
            float: The mean squared error value
        """
        return mean_squared_error(real_label, predicted_label)

    @staticmethod
    def get_mean_absolute_error(real_label, predicted_label):
        """
        Calculates the Mean Absolute Error (MAE) between real and predicted values.

        MAE measures the average magnitude of the errors in a set of predictions,
        without considering their direction.

        Parameters:
            real_label (array-like): Ground truth (correct) target values
            predicted_label (array-like): Estimated target values

        Returns:
            float: The mean absolute error value
        """
        return mean_absolute_error(real_label, predicted_label)

    @staticmethod
    def get_r2_score(real_label, predicted_label):
        """
        Calculates the R² score, also known as the coefficient of determination.

        R² indicates the proportion of the variance in the dependent variable that is
        predictable from the independent variable(s). Best possible score is 1.0.

        Parameters:
            real_label (array-like): Ground truth (correct) target values
            predicted_label (array-like): Estimated target values

        Returns:
            float: The R² score
        """
        return r2_score(real_label, predicted_label)

    @staticmethod
    def get_explained_variance(real_label, predicted_label):
        """
        Calculates the explained variance regression score.

        Measures the proportion to which a mathematical model accounts for
        the variation of a given data set.

        Parameters:
            real_label (array-like): Ground truth (correct) target values
            predicted_label (array-like): Estimated target values

        Returns:
            float: The explained variance score
        """
        return explained_variance_score(real_label, predicted_label)

    @staticmethod
    def get_cosine_similarity(real_label, predicted_label):
        """
        Calculates the mean cosine similarity between real and predicted values.

        Cosine similarity measures the cosine of the angle between two vectors,
        indicating their orientation similarity regardless of magnitude.

        Parameters:
            real_label (array-like): Ground truth (correct) target values
            predicted_label (array-like): Estimated target values

        Returns:
            float: The mean cosine similarity score
        """
        return numpy.mean(pairwise.cosine_similarity(real_label, predicted_label))

    @staticmethod
    def get_kl_divergence(real_label, predicted_label):
        """
        Calculates the Kullback-Leibler (KL) divergence between real and predicted distributions.

        KL divergence measures how one probability distribution diverges from a second,
        expected probability distribution.

        Parameters:
            real_label (array-like): Ground truth (correct) target values
            predicted_label (array-like): Estimated target values

        Returns:
            float: The KL divergence value
        """
        real_label = numpy.asarray(real_label, dtype=numpy.float32) + 1e-10
        predicted_label = numpy.asarray(predicted_label, dtype=numpy.float32) + 1e-10
        return sum(rel_entr(real_label, predicted_label))

    @staticmethod
    def get_jensen_shannon_divergence(real_label, predicted_label):
        """
        Calculates the Jensen-Shannon divergence between real and predicted distributions.

        A symmetric and smoothed version of the KL divergence, bounded between 0 and 1.

        Parameters:
            real_label (array-like): Ground truth (correct) target values
            predicted_label (array-like): Estimated target values

        Returns:
            float: The Jensen-Shannon divergence value
        """
        return jensenshannon(real_label, predicted_label)

    @staticmethod
    def get_euclidean_distance(real_label, predicted_label):
        """
        Calculates the Euclidean distance between real and predicted values.

        The straight-line distance between two points in Euclidean space.

        Parameters:
            real_label (array-like): Ground truth (correct) target values
            predicted_label (array-like): Estimated target values

        Returns:
            float: The Euclidean distance
        """
        return euclidean(real_label, predicted_label)

    @staticmethod
    def get_cityblock_distance(real_label, predicted_label):
        """
        Calculates the Manhattan distance (city block distance) between real and predicted values.

        The sum of absolute differences between coordinates.

        Parameters:
            real_label (array-like): Ground truth (correct) target values
            predicted_label (array-like): Estimated target values

        Returns:
            float: The Manhattan distance
        """
        return cityblock(real_label, predicted_label)

    @staticmethod
    def get_correlation_distance(real_label, predicted_label):
        """
        Calculates the correlation distance between real and predicted values.

        Measures the distance based on the correlation between the vectors.

        Parameters:
            real_label (array-like): Ground truth (correct) target values
            predicted_label (array-like): Estimated target values

        Returns:
            float: The correlation distance
        """
        return correlation(real_label, predicted_label)

    @staticmethod
    def get_chebyshev_distance(real_label, predicted_label):
        """
        Calculates the Chebyshev distance between real and predicted values.

        The maximum distance between any coordinate pair.

        Parameters:
            real_label (array-like): Ground truth (correct) target values
            predicted_label (array-like): Estimated target values

        Returns:
            float: The Chebyshev distance
        """
        return chebyshev(real_label, predicted_label)

    @staticmethod
    def get_maximum_mean_discrepancy(real_label, predicted_label):
        """
        Calculates the Maximum Mean Discrepancy (MMD) between real and predicted values.

        A kernel-based statistical test to determine if two samples are from different distributions.

        Parameters:
            real_label (array-like): Ground truth (correct) target values
            predicted_label (array-like): Estimated target values

        Returns:
            float: The MMD value
        """
        delta = real_label.mean(0) - predicted_label.mean(0)
        return delta.dot(delta.T)

    @staticmethod
    def get_accuracy(real_label, predicted_label):
        """
        Calculates the accuracy classification score.

        The fraction of correctly classified samples.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The accuracy score
        """
        return accuracy_score(real_label, predicted_label)

    @staticmethod
    def get_balanced_accuracy(real_label, predicted_label):
        """
        Calculates the balanced accuracy score.

        The average of recall obtained on each class, useful for imbalanced datasets.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The balanced accuracy score
        """
        return balanced_accuracy_score(real_label, predicted_label)

    @staticmethod
    def get_precision(real_label, predicted_label):
        """
        Calculates the precision score.

        The ratio of true positives to the sum of true and false positives.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The precision score
        """
        return precision_score(real_label, predicted_label)

    @staticmethod
    def get_recall(real_label, predicted_label):
        """
        Calculates the recall score (sensitivity, true positive rate).

        The ratio of true positives to the sum of true positives and false negatives.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The recall score
        """
        return recall_score(real_label, predicted_label)

    @staticmethod
    def get_f1_score(real_label, predicted_label):
        """
        Calculates the F1 score.

        The harmonic mean of precision and recall.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The F1 score
        """
        return f1_score(real_label, predicted_label)

    @staticmethod
    def get_roc_auc_score(real_label, predicted_label):
        """
        Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

        Measures the entire two-dimensional area underneath the ROC curve.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted probabilities or decision function

        Returns:
            float: The ROC AUC score
        """
        return roc_auc_score(real_label, predicted_label)

    @staticmethod
    def get_log_loss(real_label, predicted_label):
        """
        Calculates the logarithmic loss (cross-entropy loss).

        Measures the performance of a classification model where the prediction
        is a probability value between 0 and 1.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted probabilities

        Returns:
            float: The log loss value
        """
        return log_loss(real_label, predicted_label)

    @staticmethod
    def get_brier_score_loss(real_label, predicted_label):
        """
        Calculates the Brier score loss.

        Measures the mean squared difference between predicted probabilities
        and the actual outcomes.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted probabilities

        Returns:
            float: The Brier score loss
        """
        return brier_score_loss(real_label, predicted_label)

    @staticmethod
    def get_hamming_loss(real_label, predicted_label):
        """
        Calculates the Hamming loss.

        The fraction of labels that are incorrectly predicted.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The Hamming loss
        """
        return hamming_loss(real_label, predicted_label)

    @staticmethod
    def get_jaccard_score(real_label, predicted_label):
        """
        Calculates the Jaccard similarity coefficient score.

        The size of the intersection divided by the size of the union of label sets.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The Jaccard score
        """
        return jaccard_score(real_label, predicted_label)

    @staticmethod
    def get_zero_one_loss(real_label, predicted_label):
        """
        Calculates the zero-one classification loss.

        The fraction of misclassifications.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The zero-one loss
        """
        return zero_one_loss(real_label, predicted_label)

    @staticmethod
    def get_precision_recall_fscore(real_label, predicted_label):
        """
        Calculates precision, recall, F-measure and support for each class.

        Returns weighted averages for precision, recall and F-measure.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            tuple: (precision, recall, fbeta_score, support)
        """
        return precision_recall_fscore_support(real_label, predicted_label, average='weighted')

    @staticmethod
    def get_matthews_corrcoef(real_label, predicted_label):
        """
        Calculates the Matthews correlation coefficient (MCC).

        A correlation coefficient between observed and predicted binary classifications.
        Returns value between -1 and +1.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The MCC value
        """
        return matthews_corrcoef(real_label, predicted_label)

    @staticmethod
    def get_cohen_kappa_score(real_label, predicted_label):
        """
        Calculates Cohen's kappa score.

        Measures inter-annotator agreement for categorical items.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The kappa score
        """
        return cohen_kappa_score(real_label, predicted_label)

    @staticmethod
    def get_adjusted_mutual_info_score(real_label, predicted_label):
        """
        Calculates the adjusted mutual information score.

        Measures the agreement of two clusterings, adjusted for chance.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The adjusted mutual information score
        """
        return adjusted_mutual_info_score(real_label, predicted_label)

    @staticmethod
    def get_mutual_info_score(real_label, predicted_label):
        """
        Calculates the mutual information score.

        Measures the agreement of two clusterings, not adjusted for chance.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The mutual information score
        """
        return mutual_info_score(real_label, predicted_label)

    @staticmethod
    def get_normalized_mutual_info_score(real_label, predicted_label):
        """
        Calculates the normalized mutual information score.

        Normalized version of mutual information to scale results between 0 and 1.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The normalized mutual information score
        """
        return normalized_mutual_info_score(real_label, predicted_label)

    @staticmethod
    def get_adjusted_rand_score(real_label, predicted_label):
        """
        Calculates the adjusted Rand index.

        Measures the similarity between two clusterings, adjusted for chance.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The adjusted Rand score
        """
        return adjusted_rand_score(real_label, predicted_label)

    @staticmethod
    def get_completeness_score(real_label, predicted_label):
        """
        Calculates the completeness score.

        Measures whether all members of a given class are assigned to the same cluster.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The completeness score
        """
        return completeness_score(real_label, predicted_label)

    @staticmethod
    def get_homogeneity_score(real_label, predicted_label):
        """
        Calculates the homogeneity score.

        Measures whether all clusters contain only members of a single class.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The homogeneity score
        """
        return homogeneity_score(real_label, predicted_label)

    @staticmethod
    def get_v_measure_score(real_label, predicted_label):
        """
        Calculates the V-measure score.

        Harmonic mean between homogeneity and completeness.

        Parameters:
            real_label (array-like): Ground truth (correct) labels
            predicted_label (array-like): Predicted labels

        Returns:
            float: The V-measure score
        """
        return v_measure_score(real_label, predicted_label)

    @staticmethod
    def get_silhouette_score(data, labels):
        """
        Calculates the silhouette score.

        Measures how similar an object is to its own cluster compared to other clusters.
        Ranges from -1 to +1, where higher values indicate better clustering.

        Parameters:
            data (array-like): Input data samples
            labels (array-like): Cluster labels for each sample

        Returns:
            float: The silhouette score
        """
        return silhouette_score(data, labels)

    @staticmethod
    def get_calinski_harabasz_score(data, labels):
        """
        Calculates the Calinski-Harabasz score.

        The ratio of between-clusters dispersion to within-cluster dispersion.
        Higher values indicate better clustering.

        Parameters:
            data (array-like): Input data samples
            labels (array-like): Cluster labels for each sample

        Returns:
            float: The Calinski-Harabasz score
        """
        return calinski_harabasz_score(data, labels)

    @staticmethod
    def get_davies_bouldin_score(data, labels):
        """
        Calculates the Davies-Bouldin score.

        Measures the average similarity between each cluster and its most similar cluster.
        Lower values indicate better clustering.

        Parameters:
            data (array-like): Input data samples
            labels (array-like): Cluster labels for each sample

        Returns:
            float: The Davies-Bouldin score
        """
        return davies_bouldin_score(data, labels)