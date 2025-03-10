# 🏥 MediScan - Medical Image Analysis Platform

A modern medical image analysis platform using YOLO models for detecting abnormalities in medical images, particularly focusing on chest X-rays and skin conditions.

## 🌟 Features

- **Chest X-Ray Analysis**: Detect and analyze abnormalities in chest X-ray images
- **Acne Detection**: Analyze skin conditions and detect acne patterns
- **Real-time Processing**: Fast and efficient image processing with YOLO models
- **API Integration**: RESTful API endpoints for easy integration
- **Metrics Monitoring**: Built-in metrics endpoint for monitoring system performance
- **Cross-Origin Support**: Full CORS support for web applications

## 📁 Project Structure

```ascii
mediscan/
├── be-fastapi/                 # Backend FastAPI service
│   ├── main.py                 # Main application entry point
│   ├── utils/                  # Utility modules
│   │   ├── models.py          # ML model management
│   │   ├── image_processing.py # Image processing utilities
│   │   └── metrics.py         # Monitoring metrics
│   └── requirements.txt        # Python dependencies
│
├── be-strapi/                  # Strapi CMS Backend
│   ├── scripts/               # Utility scripts
│   │   └── seed.js           # Database seeding
│   └── data/                  # CMS data and content
│
└── grafana/                    # Monitoring & Analytics
    └── provisioning/          # Grafana configuration
        └── datasources/       # Data source configs
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18+
- Docker (optional)

### Installation

1. **FastAPI Backend**

```bash
cd be-fastapi
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Start the API Server**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🔌 API Endpoints

- `POST /predict/` - Get detailed detection results
- `POST /yolo_predict` - Get annotated chest X-ray images
- `POST /acne-yolo-predict` - Get annotated skin condition images
- `GET /metrics` - Monitor system metrics
- `GET /health` - API health check

## 🔧 Configuration

The application supports various configuration options through environment variables:

- `CORS_ORIGINS` - Configure allowed origins
- `MODEL_PATH` - Custom model path
- `PORT` - API server port

## 📊 Monitoring

The project includes Grafana dashboards for monitoring:

- System metrics
- API performance
- Model inference statistics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [Strapi Documentation](https://docs.strapi.io)
