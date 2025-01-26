# FastAPI-Multi-Label-Classification-with-Entity-Extraction
A FastAPI-based application for multi-label text classification and entity extraction using an XGBoost model and a TF-IDF vectorizer. The app allows users to classify text snippets into multiple labels and extract domain-specific entities using fuzzy matching and spaCy's Named Entity Recognition (NER).

**Features**
Multi-label classification using an XGBoost model.
TF-IDF vectorization for text feature extraction.
Domain-specific entity extraction with:
Exact matching for domain-specific terms.
Fuzzy matching to handle slight variations.
Integration with spaCy NER for enhanced entity recognition.
RESTful API for predictions and entity extraction.
Visualization tools like confusion matrix and label co-occurrence heatmap.

**Project Structure**
.
├── main.py                  # The main FastAPI application
├── calls_dataset.csv        # The dataset for training and testing
├── domain_knowledge.json    # Domain-specific keywords for entity extraction
├── xgboost_model.pkl        # Trained XGBoost model (saved)
├── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer
├── mlb.pkl                  # Saved MultiLabelBinarizer for label encoding
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

**Setup and Installation**
**Prerequisites:**
Python 3.8 or higher
pip (Python package installer)
Docker (optional, for containerized deployment)



























