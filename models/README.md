# models/

Place your trained model files here before running the app:

| File              | Description                                      |
|-------------------|--------------------------------------------------|
| `text_model.pkl`  | Trained sklearn classifier (e.g. LogisticRegression, SVM, RandomForest) |
| `vectorizer.pkl`  | Fitted TF-IDF or CountVectorizer                 |
| `vision_model.pt` | Trained PyTorch CNN (full model or state-dict)   |

> **Note**: These files are excluded from version control via `.gitignore`.  
> For Streamlit Cloud deployment, use [Git LFS](https://git-lfs.github.com/)
> or upload via the Streamlit Secrets / file-hosting approach described in the README.
