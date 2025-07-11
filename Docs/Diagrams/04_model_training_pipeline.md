# Model Training Pipeline

This sequence diagram shows the complete workflow of the MalDataGen framework from data loading to final results generation.

```mermaid
sequenceDiagram
    participant U as User
    participant RC as run_campaign_sbseg.py
    participant M as main.py
    participant GM as GenerativeModels
    participant E as Evaluation
    participant Met as Metrics
    participant C as Classifiers
    participant P as plots.py
    
    U->>RC: Execute campaign
    RC->>RC: Parse arguments & config
    RC->>M: run_experiments()
    
    M->>M: Load dataset (CSV/XLS)
    M->>M: Preprocess data
    M->>M: Stratified K-Fold split
    
    loop For each fold (1 to K)
        M->>GM: train_model()
        GM->>GM: Initialize model instance
        GM->>GM: Configure hyperparameters
        GM->>GM: Train generative model
        GM->>M: Return trained model
        
        M->>GM: synthesize_data()
        GM->>GM: Generate synthetic samples
        GM->>M: Return synthetic data
        
        
        M->>E: TS-TR Evaluation
        E->>C: Train classifiers on synthetic
        E->>C: Test on real data
        E->>Met: Calculate metrics
        Met->>E: Return TS-TR results
        
        M->>E: TR-TS Evaluation
        E->>C: Train classifiers on real data
        E->>C: Test classifiers on synthetic
        E->>Met: Calculate metrics
        Met->>E: Return TR-TS results
        
        M->>Met: Update fold results
    end
    
    M->>Met: Calculate mean/std across folds
    M->>Met: save_dictionary_to_json()
    Met->>M: Save results to file
    
    M->>RC: Return to campaign script
    RC->>P: Generate visualizations
    P->>P: Create performance plots
    P->>P: Create comparison charts
    P->>U: Final report & visualizations
```

## Description

The MalDataGen training pipeline follows a systematic approach:

### 1. Campaign Initialization
- User executes `run_campaign_sbseg.py` with specific configuration
- Campaign script parses arguments and sets up experiment parameters
- Calls the main experiment runner

### 2. Data Preparation
- Loads dataset from CSV/XLS format
- Applies preprocessing (normalization, encoding, etc.)
- Performs stratified k-fold splitting to maintain class distribution

### 3. Model Training Loop
For each fold:
- **Model Training**: Initializes and trains the specified generative model
- **Data Generation**: Creates synthetic samples using the trained model
- **Classifier Training**: Trains multiple classifiers on real data
- **TS-TR Evaluation**: Tests synthetic data quality by training on synthetic, testing on real
- **TR-TS Evaluation**: Tests synthetic data realism by training on real, testing on synthetic

### 4. Results Aggregation
- Calculates mean and standard deviation across all folds
- Saves comprehensive results to JSON format
- Generates publication-ready visualizations

### 5. Output Generation
- Creates performance comparison plots
- Generates statistical analysis reports
- Provides both numerical and visual results

This pipeline ensures robust evaluation through cross-validation and provides comprehensive insights into synthetic data quality. 
