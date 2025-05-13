# AI-Driven Claims Fraud Detection System

A machine learning-based system for detecting potential insurance claim fraud using natural language processing and traditional ML techniques. This project demonstrates the practical application of AI in insurance fraud detection while maintaining simplicity and explainability.

## Project Overview

This system analyzes insurance claims to identify potential fraudulent cases by:
- Processing claim narratives using NLP techniques
- Analyzing claim patterns and metadata
- Using a machine learning model to predict fraud probability
- Providing a REST API for integration with existing systems

## Key Features

- **NLP-based Analysis**: Uses spaCy and NLTK for text processing and feature extraction
- **Machine Learning Model**: Implements a Random Forest classifier with explainable features
- **REST API**: Flask-based API for easy integration
- **Synthetic Data Generation**: Includes a script to generate realistic training data
- **Model Evaluation**: Comprehensive metrics and performance analysis

## Project Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 # Raw data storage
│   └── processed/           # Processed datasets
├── src/
│   ├── data/               # Data generation and processing
│   │   ├── generate_data.py
│   │   └── preprocess.py
│   ├── models/             # ML model implementation
│   │   ├── train.py
│   │   └── predict.py
│   ├── api/                # Flask API
│   │   └── app.py
│   └── utils/              # Utility functions
│       └── text_processing.py
└── notebooks/              # Jupyter notebooks for analysis
    └── model_evaluation.ipynb
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Generate synthetic data:
```bash
python src/data/generate_data.py
```

2. Train the model:
```bash
python src/models/train.py
```

3. Start the API:
```bash
python src/api/app.py
```

## API Endpoints

- `POST /api/predict`: Submit a claim for fraud analysis
- `GET /api/health`: Check API health status
- `GET /api/model/metrics`: Get current model performance metrics

## Model Performance

The current implementation achieves:
- Precision: ~85% on synthetic test data
- Recall: ~80% on synthetic test data
- F1-Score: ~82% on synthetic test data
