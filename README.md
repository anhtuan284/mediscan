# MediScan

> Advanced Medical Image Analysis Platform powered by YOLO models

## Table of Contents
- [Core Features](#core-features)
- [Application UI Demo](#application-ui-demo)
  - [Chest X-Ray Analysis Interface](#chest-x-ray-analysis-interface)
  - [Skin Condition Analysis](#skin-condition-analysis)
- [System Overview](#system-overview)
  - [System Flow Architecture](#system-flow-architecture)
  - [Technology Stack](#technology-stack)
  - [Content Management System](#content-management-system)
- [System Architecture](#system-architecture)
- [Development Setup](#development-setup)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Contributing](#contributing)
- [License](#license)

## Core Features

- 🔍 **Advanced Image Analysis**
  - Chest X-ray abnormality detection
  - Skin condition assessment
  - Multi-model support
  
- ⚡ **Performance**
  - Real-time image processing
  - Optimized YOLO implementations
  - Scalable architecture

- 🛠 **Technical Capabilities**
  - RESTful API architecture
  - Comprehensive metrics monitoring

## Application UI Demo

### Chest X-Ray Analysis Interface
![Demo 1](./assets/images/demo1.png)
*Chest X-Ray DenseNen121 prediction and heatmap by Grad-CAM*

![Demo 2](./assets/images/demo2.png)
*Highest disease possibility*

![Demo 3](./assets/images/demo3.png)
*X-Ray disease detection by YOLO*

### Skin Condition Analysis
![Demo 4](./assets/images/demo4.png)
*Acne detection using YoloV8*

## System Overview

### System Flow Architecture
![System Flow](./assets/images/systemflow.png)
*End-to-end system architecture and data flow*

### Technology Stack
![Tech Stack](./assets/images/techstack.png)
*Complete technology stack overview*

#### Tech Stack Breakdown

##### Data Processing & Model Training 🧮
- **Libraries**: 
  - TensorFlow & Keras
  - PyTorch
  - scikit-learn
  - Ultralytics YOLO
  - seaborn
- **Purpose**: Advanced data preprocessing, visualization, and deep learning model training/testing

##### AI Server 🤖
- **Core Frameworks**:
  - FastAPI: High-performance API server
  - MLflow: Experiment tracking
  - GitHub Actions: CI/CD automation
- **Models**:
  - TensorFlow: DenseNet implementation
  - Ultralytics YOLO: Object detection
- **Purpose**: Robust model serving and automated workflows

##### Front-end Interface 🎨
- **Platform**: Appsmith
- **Features**:
  - Interactive UI components
  - Real-time AI result visualization
  - Image annotation display
- **Purpose**: User-friendly interface for medical professionals

##### Document Backend 📁
- **Technologies**:
  - Strapi: Headless CMS
  - SQLite: Data storage
- **Purpose**: Secure patient record management and document API integration

##### Monitoring & Analytics 📊
- **Tools**:
  - Grafana: Metric visualization
  - Prometheus: Data collection
- **Purpose**: Real-time performance monitoring and system analytics

---

### Content Management System
![Strapi CMS](./assets/images/strapi.png)
*Strapi CMS interface for content management*

## System Architecture

```ascii
mediscan/
├── .github/                    # GitHub-related configurations (CI/CD, issues, PRs)
├── assets/                     # Static assets (e.g., images, icons, documentation)
│
├── be-fastapi/                 # Core Analysis Engine
│   ├── main.py                 # Application entrypoint
│   ├── utils/                   # Core utilities
│   │   ├── models.py            # Model management & YOLO implementations
│   │   ├── image_processing.py  # Image preprocessing & augmentation
│   │   └── metrics.py           # Performance & inference metrics
│   ├── tests/                   # Test suites
│   │   ├── unit/                # Unit tests
│   │   └── integration/         # Integration tests
│   ├── models/                  # Pre-trained model storage
│   │   ├── xray/                # X-ray analysis models
│   │   └── skin/                # Skin condition models
│   └── requirements.txt         # Python dependencies
│
├── be-fastapi-densenet/        # DenseNet Model Service
│   ├── main.py                 # DenseNet application entry
│   ├── models/                 # DenseNet model files
│   │   └── DenseNet121_epoch_30.keras
│   ├── services/               # Service Layer
│   │   ├── __init__.py
│   │   └── image_service.py
│   ├── utils/                  # DenseNet utilities
│   │   ├── __init__.py
│   │   ├── gradcam.py          # Grad-CAM visualization
│   ├── .dockerignore
│   ├── .gitignore
│   ├── config.py
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt        # DenseNet dependencies
│   ├── schemas.py              # API schemas
│
├── be-strapi/                  # Content Management System
│   ├── api/                    # API definitions & routes
│   ├── config/                 # CMS configurations
│   ├── scripts/                # Utility scripts
│   │   ├── seed.js             # Database seeding
│   │   └── backup.js           # Backup utilities
│   ├── data/                   # CMS data and content
│   │   ├── uploads/            # Media storage
│   │   └── exports/            # Data exports
│   └── package.json            # Node.js dependencies
│
├── dataset/                    # Dataset storage and preprocessing
│
├── fe-appsmith/                # Frontend Appsmith integration
│   └── PatientManagementApp.json  # Appsmith configurations
│
├── grafana/                    # Analytics & Monitoring
│   ├── dashboards/             # Custom dashboard definitions
│   │   ├── system.json         # System metrics dashboard
│   │   └── model.json          # Model performance dashboard
│   └── provisioning/           # Grafana configurations
│       ├── datasources/        # Data source configs
│       └── notifications/      # Alert configurations
│
├── notebooks/                  # Jupyter notebooks for experimentation
│
├── prometheus/                 # Monitoring metrics collection
│   └── prometheus.yaml         # Prometheus data source configurations
├── .gitattributes
├── CODE_OF_CONDUCT.md          # Code of conduct guidelines
├── docker-compose.yml          # Docker orchestration
├── LICENSE                     # Open-source license
└── README.md                   # Project documentation

```

## Development Setup

### Requirements

- Python 3.8+
- Node.js 18+
- Docker & Docker Compose
- GPU support (recommended)

### Quick Start

1. **Environment Setup**
   ```bash
   git clone https://github.com/your-org/mediscan.git
   cd mediscan
   ```

2. **Backend & Monitoring services**
   ```bash
   # FastAPI Backend
   cd be-fastapi
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   
   # Start API Server
   uvicorn main:app --reload --port 8000

   # FastAPI Backend for DenseNet121
   cd be-fastapi-densenet
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   
   # Start API Server
   uvicorn main:app --reload --port 5000
   ```
   OR with Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. **CMS**
   ```bash
   # Strapi CMS
   cd be-strapi
   npm install
   npm run develop
   ```

## API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Generic prediction pipeline |
| `/yolo_predict` | POST | X-ray analysis |
| `/acne-yolo-predict` | POST | Skin condition analysis |
| `/metrics` | GET | System metrics |
| `/health` | GET | Service health |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CORS_ORIGINS` | Allowed origins | `*` |
| `MODEL_PATH` | Model directory | `./models` |
| `PORT` | Service port | `8000` |

## Monitoring

- Real-time performance metrics
- Model inference tracking
- System resource monitoring
- Custom Grafana dashboards

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push branch (`git push origin feature/enhancement`)
5. Open a Pull Request


## License

MIT License - See [LICENSE](LICENSE) for details