
# Detecting Spam YouTube Comments

## Overview  
YouTube’s immense popularity—with over 2.7 billion monthly users—has attracted spammers and bots that post irrelevant or promotional comments, degrading user experience and engagement. This project applies natural language processing and machine learning to automatically detect and filter spam comments, achieving high accuracy on a Kaggle benchmark dataset. :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

## Data  
- **Source:** Kaggle “Detecting Spam YouTube Comments” dataset  
- **Samples:** 1,369 comments  
- **Features used:** `CONTENT` (comment text) and `CLASS` (spam vs. not spam) :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}  
- **Class distribution:** ~51.9 % spam, ~48.1 % not spam :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}  
- **Comment length:** 3–1,200 characters (mean ≈ 96)  
- **Vocabulary size after TF-IDF:** 884 features (with `min_df=2`, default `max_df=1`) :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}

## Methods  
1. **Preprocessing**  
   - Lowercasing, removal of non-alphabetic characters, digits, and extra whitespace  
   - Stop-word filtering using English stop-word list :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}  
2. **Feature Extraction**  
   - **TF-IDF** vectorization (chosen over Bag-of-Words and Word2Vec for short-text suitability) :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}  
3. **Modeling**  
   - **Logistic Regression** and **Support Vector Machine** (SVM) compared via 5-fold cross-validation  
   - Outlier analysis using Z-score and Isolation Forest :contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}  
   - Hyperparameter tuning with `GridSearchCV`  
4. **Evaluation Metrics**  
   - Accuracy, F1-score, and ROC AUC  

## Results  
- **Final model:** SVM with RBF kernel (`C=1`, `gamma=0.5`)  
- **Kaggle test accuracy:** 93.66 % (ranked 27th) :contentReference[oaicite:14]{index=14}:contentReference[oaicite:15]{index=15}  
- **Test set performance:**  
  - Accuracy: 94.7 %  
  - F1 Score: 93.33 %  
  - AUC: 0.9842 :contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17}  

## Repository Structure  


├── Dataset/               # Raw and output CSV files
├── Scripts/               # Model training and evaluation scripts
│   └── classifier.py      # Main classifier implementation
├── README.md              # Project summary and instructions
└── requirements.txt       # Python dependencies



## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare training and test CSV files under `Dataset/`.
2. Run the classifier script:

   ```bash
   python Scripts/classifier.py
   ```
3. Output predictions will be saved to `Dataset/output/`.

## Future Work

* Incorporate ensemble tree methods (e.g., Random Forest, Gradient Boosting)
* Expand training data to include diverse video categories
* Enhance robustness to misspellings and semantic variations

## References

* Kaggle dataset “Detecting Spam YouTube Comments”
* Word2Vec pre-trained vectors (Google News)
* Scikit-learn, NLTK, Gensim documentation

```



