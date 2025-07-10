# System Architecture Diagram

This diagram shows the high-level architecture of the MalDataGen framework for synthetic data generation and evaluation.

```mermaid
graph TB
    subgraph "Input Layer"
        A[CSV/XLS Data] --> B[Data Preprocessing]
        B --> C[Stratified K-Fold Split]
    end
    
    subgraph "Generative Models"
        D[Adversarial GAN] 
        E[Wasserstein GAN]
        F[Autoencoder]
        G[Variational Autoencoder]
        H[Denoising Diffusion]
        I[Latent Diffusion]
        J[VQ-VAE]
        K[SMOTE]
    end
    
    subgraph "Third-Party Models"
        L[CTGAN]
        M[TVAE]
        N[Copula]
    end
    
    subgraph "Evaluation Framework"
        O[TS-TR Strategy]
        P[TR-TS Strategy]
        Q[Cross-Validation]
    end
    
    subgraph "Metrics & Analysis"
        R[Binary Metrics]
        S[Distance Metrics]
        T[Efficiency Metrics]
        U[Visualization]
    end
    
    C --> D
    C --> E
    C --> F
    C --> G
    C --> H
    C --> I
    C --> J
    C --> K
    C --> L
    C --> M
    C --> N
    
    D --> O
    E --> O
    F --> O
    G --> O
    H --> O
    I --> O
    J --> O
    K --> O
    L --> O
    M --> O
    N --> O
    
    O --> P
    P --> Q
    Q --> R
    Q --> S
    Q --> T
    R --> U
    S --> U
    T --> U
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style D fill:#f3e5f5
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style G fill:#f3e5f5
    style H fill:#f3e5f5
    style I fill:#f3e5f5
    style J fill:#f3e5f5
    style K fill:#f3e5f5
    style L fill:#fff3e0
    style M fill:#fff3e0
    style N fill:#fff3e0
    style O fill:#e8f5e8
    style P fill:#e8f5e8
    style Q fill:#e8f5e8
    style R fill:#fff8e1
    style S fill:#fff8e1
    style T fill:#fff8e1
    style U fill:#fff8e1
```

## Description

The MalDataGen framework follows a modular architecture with four main layers:

1. **Input Layer**: Handles data ingestion from CSV/XLS formats, preprocessing, and stratified k-fold splitting
2. **Generative Models**: Core implementation of 8 different generative algorithms
3. **Third-Party Models**: Integration with SDV library models (CTGAN, TVAE, Copula)
4. **Evaluation Framework**: Implements TS-TR and TR-TS evaluation strategies with cross-validation
5. **Metrics & Analysis**: Comprehensive evaluation metrics and visualization capabilities

The framework supports both native implementations and third-party integrations, providing a comprehensive evaluation pipeline for synthetic data generation. 