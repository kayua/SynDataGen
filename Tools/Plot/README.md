# Synthetic Data Generation Report - reduced_balanced_kronodroid_emulator.csv

**Date:** 2025-05-25  
**Model Type:** AUTOENCODER  

---

## 1. Dataset Information
| **Characteristic**       | **Value**                          |
|--------------------------|------------------------------------|
| **Name**                 | reduced_balanced_kronodroid_emulator.csv             |
| **Format**               | CSV           |
| **Samples per class**    | Class 0: 1000, Class 1: 1000 |
| **Number of classes**    | 2 |
| **Model type**           | autoencoder       |

---

## 2. Model Configuration
| **Parameter**            | **Value**                          |
|--------------------------|------------------------------------|
| Latent dimension         | 128   |
| Training epochs          | 1             |
| Batch size               | 128         |
| Learning rate            | 0.001      |
| Activation function      | Relu         |

---

## 3. Performance Metrics Summary

### 3.1 Training on Synthetic, Testing on Real (TS-TR)

#### RandomForest
- **Accuracy:** 0.7429 ± 0.0402
- **Precision:** 0.7291 ± 0.1030
- **Recall:** 0.8437 ± 0.1216
- **F1 Score:** 0.7663 ± 0.0146

#### SupportVectorMachine
- **Accuracy:** 0.6605 ± 0.0541
- **Precision:** 0.6770 ± 0.1156
- **Recall:** 0.7818 ± 0.2525
- **F1 Score:** 0.6770 ± 0.1117

#### KNN
- **Accuracy:** 0.7427 ± 0.0362
- **Precision:** 0.7191 ± 0.0842
- **Recall:** 0.8430 ± 0.0920
- **F1 Score:** 0.7665 ± 0.0158

#### DecisionTree
- **Accuracy:** 0.7451 ± 0.0396
- **Precision:** 0.7190 ± 0.0867
- **Recall:** 0.8590 ± 0.1041
- **F1 Score:** 0.7716 ± 0.0147

#### NaiveBayes
- **Accuracy:** 0.7052 ± 0.0601
- **Precision:** 0.7019 ± 0.1259
- **Recall:** 0.8379 ± 0.1449
- **F1 Score:** 0.7404 ± 0.0202

#### GradientBoosting
- **Accuracy:** 0.7411 ± 0.0414
- **Precision:** 0.7200 ± 0.0975
- **Recall:** 0.8549 ± 0.1143
- **F1 Score:** 0.7678 ± 0.0148

#### StochasticGradientDescent
- **Accuracy:** 0.7392 ± 0.0437
- **Precision:** 0.7176 ± 0.0942
- **Recall:** 0.8521 ± 0.1076
- **F1 Score:** 0.7663 ± 0.0144

### 3.2 Training on Real, Testing on Synthetic (TR-TS)

#### RandomForest
- **Accuracy:** 0.5481 ± 0.0999
- **Precision:** 0.1994 ± 0.3988
- **Recall:** 0.0989 ± 0.1979
- **F1 Score:** 0.1323 ± 0.2645

#### SupportVectorMachine
- **Accuracy:** 0.5008 ± 0.0048
- **Precision:** 0.4000 ± 0.4899
- **Recall:** 0.0016 ± 0.0029
- **F1 Score:** 0.0032 ± 0.0058

#### KNN
- **Accuracy:** 0.4974 ± 0.0081
- **Precision:** 0.2743 ± 0.3903
- **Recall:** 0.0080 ± 0.0154
- **F1 Score:** 0.0145 ± 0.0279

#### DecisionTree
- **Accuracy:** 0.5805 ± 0.2047
- **Precision:** 0.3957 ± 0.4847
- **Recall:** 0.2010 ± 0.3935
- **F1 Score:** 0.2034 ± 0.3902

#### NaiveBayes
- **Accuracy:** 0.5008 ± 0.0048
- **Precision:** 0.6003 ± 0.1999
- **Recall:** 0.8001 ± 0.3998
- **F1 Score:** 0.5338 ± 0.2664

#### GradientBoosting
- **Accuracy:** 0.6070 ± 0.1899
- **Precision:** 0.3916 ± 0.4796
- **Recall:** 0.2189 ± 0.3870
- **F1 Score:** 0.2352 ± 0.3820

#### StochasticGradientDescent
- **Accuracy:** 0.6750 ± 0.1957
- **Precision:** 0.7958 ± 0.3980
- **Recall:** 0.3528 ± 0.4011
- **F1 Score:** 0.3973 ± 0.4123

---

## 4. Distance Metrics (Real vs Synthetic)
| **Metric**              | **Value**                          |
|--------------------------|------------------------------------|
| Euclidean Distance       | 4.1917 ± 0.0177 |
| Hellinger Distance       | 194.1721 ± 1.1365 |
| Manhattan Distance       | 377.0410 ± 4.4186 |
| Hamming Distance         | 1.8852 ± 0.0221 |
| Jaccard Distance         | 0.4376 ± 0.0074 |

---

## 5. Efficiency Metrics

| **Metric**              | **Value**                          |
|--------------------------|------------------------------------|
| Training Time (ms)       | 969.81 ± 95.71 |
| Generation Time (ms)     | 554.10 ± 16.62 |
| Process CPU Usage (%)    | 10.03 ± 0.11 |
| Process Memory (MB)      | 104.37 ± 91.42 |

---

## 6. Visualizations
*(Note: In a complete implementation, this section would include paths to generated plots)*

- Confusion matrices for each classifier
- Training curves
- Feature distribution comparisons
- Distance metric visualizations

---

## 7. Conclusion
- **Best performing classifier:** [To be determined based on metrics]
- **Data similarity:** The distance metrics indicate [low/moderate/high] similarity between real and synthetic data
- **Efficiency:** The model shows [good/poor] computational efficiency with average training time of 969.81ms

