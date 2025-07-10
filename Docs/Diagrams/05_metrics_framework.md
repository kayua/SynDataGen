# Metrics Framework

This diagram shows the comprehensive metrics framework used in MalDataGen for evaluating synthetic data quality and model performance.

```mermaid
graph LR
    subgraph "Binary Classification Metrics"
        A1[Accuracy]
        A2[Precision]
        A3[Recall]
        A4[F1-Score]
        A5[Specificity]
        A6[True Positive Rate]
        A7[False Positive Rate]
        A8[True Negative Rate]
        A9[False Negative Rate]
        A10[ROC-AUC]
    end
    
    subgraph "Regression Metrics"
        B1[Mean Squared Error]
        B2[Mean Absolute Error]
    end
    
    subgraph "Distance Metrics"
        C1[Euclidean Distance]
        C2[Hellinger Distance]
        C3[Manhattan Distance]
        C4[Hamming Distance]
        C5[Jaccard Distance]
    end
    
    subgraph "Efficiency Metrics"
        D1[Process CPU %]
        D2[Process Memory MB]
        D3[System CPU %]
        D4[System Memory MB]
        D5[System Memory %]
        D6[Training Time ms]
        D7[Generation Time ms]
    end
    
    subgraph "SDV Integration Metrics"
        E1[Diagnostic Score]
        E2[Quality Score]
    end
    
    subgraph "Statistical Tests"
        F1[Permutation Test]
    end
    
    style A1 fill:#e8f5e8
    style A2 fill:#e8f5e8
    style A3 fill:#e8f5e8
    style A4 fill:#e8f5e8
    style A5 fill:#e8f5e8
    style A6 fill:#e8f5e8
    style A7 fill:#e8f5e8
    style A8 fill:#e8f5e8
    style A9 fill:#e8f5e8
    style A10 fill:#e8f5e8
    style B1 fill:#fff3e0
    style B2 fill:#fff3e0
    style C1 fill:#e3f2fd
    style C2 fill:#e3f2fd
    style C3 fill:#e3f2fd
    style C4 fill:#e3f2fd
    style C5 fill:#e3f2fd
    style D1 fill:#f3e5f5
    style D2 fill:#f3e5f5
    style D3 fill:#f3e5f5
    style D4 fill:#f3e5f5
    style D5 fill:#f3e5f5
    style D6 fill:#f3e5f5
    style D7 fill:#f3e5f5
    style E1 fill:#ffebee
    style E2 fill:#ffebee
    style F1 fill:#e0f2f1
```

## Description

MalDataGen implements a comprehensive metrics framework covering multiple evaluation dimensions:

### Binary Classification Metrics
- **Core Metrics**: Accuracy, Precision, Recall, F1-Score, Specificity
- **Rate Metrics**: TPR, FPR, TNR, FNR for detailed performance analysis
- **Advanced Metrics**: ROC-AUC for overall model performance assessment

### Regression Metrics
- **Error Metrics**: MSE and MAE for continuous value prediction evaluation
- **Purpose**: Assess model performance on regression tasks within the framework

### Distance Metrics
- **Distribution Comparison**: Euclidean, Hellinger, Manhattan distances
- **Categorical Comparison**: Hamming and Jaccard distances for discrete data
- **Purpose**: Measure similarity between real and synthetic data distributions

### Efficiency Metrics
- **Resource Monitoring**: CPU and memory usage tracking
- **Performance Timing**: Training and generation time measurements
- **Purpose**: Assess computational efficiency and resource requirements

### SDV Integration Metrics
- **Diagnostic Score**: Overall synthetic data quality assessment
- **Quality Score**: Statistical quality evaluation
- **Purpose**: Leverage industry-standard synthetic data evaluation

### Statistical Tests
- **Permutation Test**: Statistical significance testing
- **Purpose**: Validate the statistical validity of results

This multi-dimensional approach ensures comprehensive evaluation of synthetic data quality, model performance, and computational efficiency. 