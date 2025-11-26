# Misinformation Detection API

A Flask-based web API for detecting misinformation in text using machine learning models trained on mDeberta embeddings.

## Features

- **Text Analysis**: Analyze text for misinformation likelihood using Logistic Regression, Random Forest, or Ensemble models.
- **Translation Support**: Optional translation to English using Google Translate.
- **Web Interface**: User-friendly HTML interface for easy interaction.
- **Feedback Reporting**: Collect user feedback on predictions.

## Models Used

- Embeddings: mDeberta-v3-base (via Hugging Face Transformers)
- Classifiers: Logistic Regression, Random Forest, Voting Classifier Ensemble

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SAICHARAN704SDF/misinfo.git
   cd misinfo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python app.py
   ```
   The app will be available at `http://127.0.0.1:5000`.

## Deployment on Render

1. Connect your GitHub repository to Render.
2. Create a new Web Service with the following settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
3. Deploy and access your live API.

## API Endpoints

- `GET /`: Serves the web interface.
- `POST /predict`: Predicts misinformation likelihood.
  - Body: `{"text": "your text here", "model": "lr|rf|ensemble", "translate": false}`
- `POST /report`: Reports feedback.
  - Body: `{"text": "...", "model_used": "...", "confidence": 0.5, "user_label": "...", "notes": "..."}`

## Requirements

- Python 3.9+
- Libraries: See `requirements.txt`

## License

[Add your license here]