# Deployment Architecture

This diagram shows the deployment options and execution modes available for the MalDataGen framework, including Docker containerization and local execution.

```mermaid
graph TB
    subgraph "User Interface"
        A[Command Line Interface]
        B[Campaign Scripts]
        C[Docker Commands]
    end
    
    subgraph "Execution Modes"
        D[Local Python Execution]
        E[Docker Container Execution]
        F[Demo Mode]
        G[Full Experiment Mode]
    end
    
    subgraph "Docker Infrastructure"
        H[Docker Image]
        I[Container Runtime]
        J[Volume Mounts]
        K[Network Configuration]
    end
    
    subgraph "Local Environment"
        L[Python Virtual Environment]
        M[System Dependencies]
        N[GPU Support]
        O[Local File System]
    end
    
    subgraph "Data & Results"
        P[Input Datasets]
        Q[Generated Models]
        R[Results JSON]
        S[Visualization Plots]
    end
    
    A --> D
    A --> E
    B --> D
    B --> E
    C --> E
    
    D --> L
    D --> M
    D --> N
    D --> O
    
    E --> H
    E --> I
    E --> J
    E --> K
    
    F --> D
    F --> E
    G --> D
    G --> E
    
    D --> P
    D --> Q
    D --> R
    D --> S
    
    E --> P
    E --> Q
    E --> R
    E --> S
    
    J --> P
    J --> Q
    J --> R
    J --> S
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#e3f2fd
    style D fill:#e8f5e8
    style E fill:#e8f5e8
    style F fill:#fff3e0
    style G fill:#fff3e0
    style H fill:#f3e5f5
    style I fill:#f3e5f5
    style J fill:#f3e5f5
    style K fill:#f3e5f5
    style L fill:#ffebee
    style M fill:#ffebee
    style N fill:#ffebee
    style O fill:#ffebee
    style P fill:#e0f2f1
    style Q fill:#e0f2f1
    style R fill:#e0f2f1
    style S fill:#e0f2f1
```

## Deployment Options

### Local Execution
- **Python Environment**: Virtual environment with pip dependencies
- **System Requirements**: Python 3.8+, CUDA 11+ (optional)
- **Execution**: Direct Python script execution
- **Advantages**: Full system access, GPU acceleration, easy debugging

### Docker Containerization
- **Base Image**: Ubuntu 22.04 with Python 3.8+
- **Dependencies**: Pre-installed via requirements.txt
- **Volume Mounts**: Data and results persistence
- **Advantages**: Reproducible environment, isolation, easy distribution

## Execution Modes

### Demo Mode
- **Purpose**: Quick testing and demonstration
- **Duration**: ~3 minutes
- **Scope**: Reduced dataset, single model type
- **Command**: `python3 run_campaign_sbseg.py -c sf`

### Full Experiment Mode
- **Purpose**: Complete research evaluation
- **Duration**: ~7 hours
- **Scope**: All models, comprehensive evaluation
- **Command**: `python3 run_campaign_sbseg.py`

## Docker Commands

```bash
# Demo execution
./run_demo_docker.sh

# Full experiments
./run_experiments_docker.sh

# Manual Docker execution
docker build -t maldatagen .
docker run -v $(pwd)/datasets:/MalDataGen/datasets maldatagen
```

## File Structure

```
MalDataGen/
├── main.py                    # Core framework
├── run_campaign_sbseg.py      # Campaign orchestrator
├── Dockerfile                 # Container definition
├── requirements.txt           # Python dependencies
├── run_demo_docker.sh         # Demo execution script
├── run_experiments_docker.sh  # Full experiment script
├── Datasets/                  # Input data directory
├── Results/                   # Output results directory
└── Docs/Diagrams/             # Architecture documentation
```

## Resource Requirements

### Minimum Requirements
- **CPU**: Any x86_64 processor
- **RAM**: 4 GB
- **Storage**: 10 GB
- **GPU**: Optional (NVIDIA with CUDA 11+)

### Recommended Requirements
- **CPU**: Multi-core (i5/Ryzen 5+)
- **RAM**: 8 GB+
- **Storage**: 20 GB SSD
- **GPU**: NVIDIA with CUDA 11+ for acceleration

## Security Considerations

- **Local Execution**: No security concerns
- **Docker Execution**: Requires sudo permissions for Docker engine
- **Data Privacy**: All processing occurs locally or within container
- **Network**: No external network access required 