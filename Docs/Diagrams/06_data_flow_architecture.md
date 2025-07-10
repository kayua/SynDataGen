# Data Flow Architecture

This diagram illustrates the complete data flow through the MalDataGen framework, from raw data ingestion to final results generation.

```mermaid
graph TD
    A[Raw Data Files] --> B[Data Ingestion Layer]
    B --> C[Data Preprocessing]
    C --> D[Stratified K-Fold Split]
    
    subgraph "Data Processing Pipeline"
        B --> B1[CSVLoader]
        B --> B2[XLSLoader]
        B1 --> C
        B2 --> C
    end
    
    D --> E[Training Set]
    D --> F[Test Set]
    
    subgraph "Generative Model Training"
        E --> G[Model Initialization]
        G --> H[Hyperparameter Configuration]
        H --> I[Model Training]
        I --> J[Model Persistence]
    end
    
    subgraph "Synthetic Data Generation"
        J --> K[Latent Space Sampling]
        K --> L[Data Generation]
        L --> M[Synthetic Dataset]
    end
    
    subgraph "Evaluation Pipeline"
        M --> N[TS-TR Evaluation]
        F --> N
        E --> O[Classifier Training]
        O --> P[TR-TS Evaluation]
        M --> P
    end
    
    subgraph "Metrics Calculation"
        N --> Q[Binary Metrics]
        N --> R[Distance Metrics]
        P --> Q
        P --> R
        Q --> S[Performance Analysis]
        R --> S
    end
    
    subgraph "Results & Visualization"
        S --> T[JSON Results Storage]
        T --> U[Statistical Analysis]
        U --> V[Performance Plots]
        U --> W[Comparison Charts]
        U --> X[Heatmaps]
    end
    
    subgraph "Resource Monitoring"
        Y[CPU Usage Tracking]
        Z[Memory Usage Tracking]
        AA[Training Time Measurement]
        BB[Generation Time Measurement]
    end
    
    I --> Y
    I --> Z
    I --> AA
    L --> BB
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style B1 fill:#e3f2fd
    style B2 fill:#e3f2fd
    style C fill:#e3f2fd
    style D fill:#e3f2fd
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style G fill:#e8f5e8
    style H fill:#e8f5e8
    style I fill:#e8f5e8
    style J fill:#e8f5e8
    style K fill:#fff3e0
    style L fill:#fff3e0
    style M fill:#fff3e0
    style N fill:#ffebee
    style O fill:#ffebee
    style P fill:#ffebee
    style Q fill:#e0f2f1
    style R fill:#e0f2f1
    style S fill:#e0f2f1
    style T fill:#fce4ec
    style U fill:#fce4ec
    style V fill:#fce4ec
    style W fill:#fce4ec
    style X fill:#fce4ec
    style Y fill:#fff8e1
    style Z fill:#fff8e1
    style AA fill:#fff8e1
    style BB fill:#fff8e1
```

## Description

The MalDataGen data flow architecture follows a systematic pipeline:

### 1. Data Ingestion & Preprocessing
- **Input**: Raw CSV/XLS files containing tabular data
- **Loaders**: Specialized CSVLoader and XLSLoader for format handling
- **Preprocessing**: Data cleaning, normalization, and encoding
- **Splitting**: Stratified k-fold split maintaining class distribution

### 2. Generative Model Pipeline
- **Initialization**: Model-specific configuration and setup
- **Training**: Iterative training with hyperparameter optimization
- **Persistence**: Model saving for reproducibility
- **Generation**: Synthetic data creation through latent space sampling

### 3. Evaluation Framework
- **TS-TR**: Train on synthetic, test on real data
- **TR-TS**: Train on real, test on synthetic data
- **Cross-validation**: Robust evaluation across multiple folds

### 4. Metrics & Analysis
- **Binary Metrics**: Classification performance evaluation
- **Distance Metrics**: Distribution similarity assessment
- **Performance Analysis**: Comprehensive statistical analysis

### 5. Results & Visualization
- **Storage**: JSON format for structured results
- **Visualization**: Performance plots, comparison charts, heatmaps
- **Reporting**: Publication-ready visualizations and analysis

### 6. Resource Monitoring
- **Efficiency Tracking**: CPU, memory, and timing measurements
- **Performance Optimization**: Resource usage analysis for scalability

This architecture ensures data integrity, reproducibility, and comprehensive evaluation throughout the entire pipeline. 