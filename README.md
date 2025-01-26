# FastAPI Multi-Label Classification with Entity Extraction

A FastAPI-based application for multi-label text classification and entity extraction using an XGBoost model and a TF-IDF vectorizer. The app allows users to classify text snippets into multiple labels and extract domain-specific entities using fuzzy matching and spaCy's Named Entity Recognition (NER).

## Features
- Multi-label classification using an XGBoost model.
- TF-IDF vectorization for text feature extraction.
- Domain-specific entity extraction with:
  - Exact matching for domain-specific terms.
  - Fuzzy matching to handle slight variations.
  - Integration with spaCy NER for enhanced entity recognition.
- RESTful API for predictions and entity extraction.
- Visualization tools like confusion matrix and label co-occurrence heatmap.

## Project Structure
```
├── main.py                  # The main FastAPI application
├── calls_dataset.csv        # The dataset for training and testing
├── domain_knowledge.json    # Domain-specific keywords for entity extraction
├── xgboost_model.pkl        # Trained XGBoost model (saved)
├── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer
├── mlb.pkl                  # Saved MultiLabelBinarizer for label encoding
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Docker (for containerized deployment)

### Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Prepare Files
Ensure the following files are in the project directory:
- `calls_dataset.csv`: Dataset for training/testing.
- `domain_knowledge.json`: JSON file containing domain-specific keywords.

### Run the FastAPI Application
Run the application locally:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
The app will be available at Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

## Usage

### Endpoint: `/analyze`

Analyze a text snippet for classification labels and entities.

- **Method**: POST
- **URL**: [http://localhost:8000/analyze](http://localhost:8000/analyze)

**Request Body:**
```json
{
    "text": "CompetitorX offers better pricing options and analytics."
}
```

**Response:**
```json
{
    "labels": [
        ["Pricing Discussion", "Positive"]
    ],
    "entities": {
        "competitors": ["CompetitorX"],
        "features": ["analytics"],
        "pricing_keywords": ["better pricing options"]
    },
    "summary": "CompetitorX offers better pricing options and ana..."
}
```

## Visualization
The application generates the following visualizations during training:
- **Classification Report**: Displays precision, recall, and F1 scores for each label.
- **Confusion Matrix**: Per-label confusion matrices to assess predictions.
- **Label Co-Occurrence Heatmap**: Visualizes relationships between labels.

## Entity Extraction
The application extracts entities using:

### Domain Knowledge (`domain_knowledge.json`):
- **Categories**: competitors, features, pricing_keywords, etc.
- **Techniques**: Supports exact and fuzzy matching.

### spaCy NER:
- Extracts standard entities like organizations, dates, etc.

## Model and Files

### Trained Model Files
- `xgboost_model.pkl`: Saved XGBoost classifier.
- `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer.
- `mlb.pkl`: Saved MultiLabelBinarizer for encoding labels.

### Dataset
- `calls_dataset.csv`: Dataset for training and testing the model.

## How to Train the Model
The model is pre-trained and saved, but you can retrain it by following these steps:

1. Modify `main.py` to include a retraining script.
2. Use the `calls_dataset.csv` file to preprocess, train, and save the updated model:

```python
# Train the model
model = OneVsRestClassifier(XGBClassifier(eval_metric="logloss"))
model.fit(X_train, y_train)

# Save the updated model
with open("xgboost_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
```

## Deployment

### Docker Deployment

**Build the Docker image:**
```bash
docker build -t fastapi-xgboost-app .
```

**Run the Docker container:**
```bash
docker run -p 8000:8000 fastapi-xgboost-app
```
Access the app at [http://localhost:8000](http://localhost:8000).

### Deploy to Cloud
You can deploy this app to platforms like:
- **Heroku** (via Container Registry)
- **AWS ECS** or **Google Cloud Run**

## Future Enhancements
- Add more categories and refine `domain_knowledge.json`.
- Implement logging for production use.
- Use advanced embedding techniques like BERT for feature extraction.

## Contributing
Contributions are welcome! To contribute:
1. Fork this repository.
2. Create a feature branch.
3. Submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
