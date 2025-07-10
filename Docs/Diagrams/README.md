# MalDataGen Architecture Diagrams

We provide a comprehensive visual overview (8 diagrams) of the MalDataGen framework, covering its architecture, design principles, data processing flow, and evaluation strategies. Developed using Mermaid notation, these diagrams support understanding of both the structural and functional aspects of the system. They include high-level system architecture, object-oriented class relationships, evaluation workflows, training pipelines, metric frameworks, and data flow. Together, they offer a detailed and cohesive view of how MalDataGen enables the generation and assessment of synthetic data in cybersecurity contexts.

## Diagram Overview

### 1. [System Architecture](01_system_architecture.md)
High-level overview of the MalDataGen framework showing the complete pipeline from data input to results generation.

### 2. [Core Class Hierarchy](02_core_class_hierarchy.md)
Object-oriented architecture showing class relationships, inheritance, and key methods in the framework.

### 3. [Evaluation Strategy](03_evaluation_strategy.md)
Flow diagram illustrating the TS-TR and TR-TS evaluation strategies used for comprehensive synthetic data assessment.

### 4. [Model Training Pipeline](04_model_training_pipeline.md)
Sequence diagram showing the complete workflow from campaign execution to final results generation.

### 5. [Metrics Framework](05_metrics_framework.md)
Comprehensive overview of all evaluation metrics including binary classification, distance metrics, and efficiency measurements.

### 6. [Data Flow Architecture](06_data_flow_architecture.md)
Detailed data flow showing how information moves through the system from raw data to final visualizations.

### 7. [Generative Models Comparison](07_generative_models_comparison.md)
Comparison of all supported generative models highlighting their characteristics and use cases.

## Usage Instructions

### For Academic Papers

1. **Copy the Mermaid code** from any diagram file
2. **Use Mermaid Live Editor** (https://mermaid.live) to preview and refine
3. **Export as SVG/PNG** for publication
4. **Include in your paper** using standard academic formats

### For Documentation

1. **View diagrams directly** in GitHub (renders automatically)
2. **Use in Markdown files** by including the Mermaid code blocks
3. **Generate static images** for presentations or reports

### For Development

1. **Reference diagrams** for understanding system architecture
2. **Update diagrams** when making architectural changes
3. **Use as design documentation** for new features

## Diagram Standards

- **Color Coding**: Consistent colors for different component types
- **Naming Convention**: Clear, descriptive names for all elements
- **Documentation**: Each diagram includes detailed descriptions
- **Modularity**: Separate diagrams for different aspects of the system

## Tools and Resources

- **Mermaid Live Editor**: https://mermaid.live
- **Mermaid Documentation**: https://mermaid.js.org/
- **GitHub Mermaid Support**: Native rendering in markdown files
- **VS Code Extension**: Mermaid Preview for local development

## Contributing

When adding new diagrams or modifying existing ones:

1. Follow the established naming convention
2. Include comprehensive descriptions
3. Use consistent styling and color schemes
4. Update this README with new diagram information
5. Ensure diagrams are accurate and up-to-date with the codebase

## Version Control

All diagrams are version-controlled with the codebase, ensuring:
- **Synchronization**: Diagrams stay current with code changes
- **History**: Track architectural evolution over time
- **Collaboration**: Multiple contributors can update diagrams
- **Reproducibility**: Diagrams are always available with the code 
