# Evaluation Strategy Flow

This diagram illustrates the two main evaluation strategies used in MalDataGen: TS-TR (Train Synthetic - Test Real) and TR-TS (Train Real - Test Synthetic).

```mermaid
flowchart TD
    A[Input Dataset] --> B[Stratified K-Fold Split]
        B --> C1[Train Generative Model on Real Data]
        C1 --> D1[Generate Synthetic Data]
    subgraph "TS-TR Strategy (Train Synthetic - Test Real)"
        D1 --> E1[Train Classifier on Synthetic Data]
        E1 --> F1[Test Classifier on Real Data]
        F1 --> G1[Calculate Performance Metrics]
    end
    
    subgraph "TR-TS Strategy (Train Real - Test Synthetic)"
        B --> C2[Train Classifier on Real Data]
        C2 --> D2[Train Generative Model on Real Data]
        D2 --> E2[Generate Synthetic Data]
        E2 --> F2[Test Classifier on Synthetic Data]
        F2 --> G2[Calculate Performance Metrics]
    end
    
    G1 --> H[Performance Comparison]
    G2 --> H
    H --> I[Generate Comprehensive Report]
    I --> J[Save Results to JSON]
    J --> K[Create Visualizations]
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C1 fill:#f3e5f5
    style D1 fill:#f3e5f5
    style E1 fill:#e8f5e8
    style F1 fill:#e8f5e8
    style G1 fill:#fff3e0
    style C2 fill:#e8f5e8
    style D2 fill:#f3e5f5
    style E2 fill:#f3e5f5
    style F2 fill:#e8f5e8
    style G2 fill:#fff3e0
    style H fill:#ffebee
    style I fill:#e0f2f1
    style J fill:#e0f2f1
    style K fill:#e0f2f1
```

## Description

MalDataGen implements two complementary evaluation strategies to comprehensively assess synthetic data quality:

### TS-TR Strategy (Train Synthetic - Test Real)
- **Purpose**: Measures the generalization ability of synthetic data
- **Process**: 
  1. Train generative model on real data
  2. Generate synthetic samples
  3. Train classifier on synthetic data
  4. Test classifier on real data
- **Interpretation**: High performance indicates synthetic data preserves real data patterns

### TR-TS Strategy (Train Real - Test Synthetic)
- **Purpose**: Assesses the realism and quality of generated data
- **Process**:
  1. Train classifier on real data
  2. Train generative model on real data
  3. Generate synthetic samples
  4. Test classifier on synthetic data
- **Interpretation**: High performance indicates synthetic data is realistic and indistinguishable from real data

### Cross-Validation Integration
Both strategies are executed using stratified k-fold cross-validation to ensure robust and reliable evaluation results across multiple data splits. 
