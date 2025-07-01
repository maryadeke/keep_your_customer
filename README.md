# 🧠 Keep Your Customer Model_ML | Group 6
Link to the application: https://maryadeke-keep-your-customer-app-mary-2pmt76.streamlit.app/

## 📌 Group Members
- Adeke Mary  
- Mugumya Timothy Kitulazi  
- Atukwase Godson  
- Muhawenimana Calorine  

---

## 🎯 Problem Statement

**Predicting Customer Churn in Online Retail Using Machine Learning**

Customer churn poses a significant challenge in online retail, impacting both revenue and customer retention. This project aims to build a binary classification machine learning model to predict whether a customer is likely to churn (`Target_Churn`) based on demographic, behavioral, and engagement data.

Our goal is to help the business **identify at-risk customers** early and take proactive retention actions such as personalized offers and better customer support.

---

## 🎯 Objectives

- Explore and analyze customer behavior data to identify churn patterns.
- Build and evaluate a binary classification model to predict churn.
- Determine the most influential factors contributing to customer churn.
- Provide actionable insights to reduce churn and improve retention.

---

## 📊 Dataset

We used a dataset from Kaggle containing **1,000 rows and 15 columns** with details such as customer demographics, purchase behavior, and engagement history.

### Key Columns:

- `Customer_ID`: Unique identifier
- `Age`, `Gender`, `Annual_Income`
- `Total_Spend`, `Years_as_Customer`, `Num_of_Purchases`
- `Average_Transaction_Amount`, `Num_of_Returns`, `Num_of_Support_Contacts`
- `Satisfaction_Score`, `Last_Purchase_Days_Ago`
- `Email_Opt_In`, `Promotion_Response`
- `Target_Churn`: Our target column (True/False)

---

## 🧹 Data Cleaning & Preprocessing

- Removed outliers and irrelevant columns (e.g., `Customer_ID`)
- No missing values were present
- Used **One-Hot Encoding** for categorical variables (`Gender`, `Promotion_Response`)
- Converted binary features (`Email_Opt_In`) to numeric
- Applied **MinMaxScaler** to normalize numeric features before feeding them into models

---

## 🤖 Model Training

We:
- Split data into features (`X`) and target (`y`)
- Used `train_test_split()` for model validation
- Trained multiple models: **Logistic Regression, Decision Tree, Random Forest, XGBoost**
- Performed **hyperparameter tuning** using PyCaret suggestions
- Evaluated performance using metrics like accuracy, precision, recall, AUC

---

## 🧠 Final Model: Ensemble Pipeline

To improve accuracy, we combined all strong classifiers using a **VotingClassifier** in an ensemble:

```python
VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(...)),
    ('lr', LogisticRegression(...)),
    ('dt', DecisionTreeClassifier(...)),
    ('xgb', XGBClassifier(...))
], voting='soft')
```

We wrapped this ensemble into a **Pipeline**, added scaling, trained it on the final dataset, and saved it using `joblib`:

```python
import joblib
joblib.dump(pipeline, 'ensemble_pipeline.pkl')
```

---

## 🖥️ Deployment with Streamlit

We created a **Streamlit web app** to:

- Accept user input (age, income, gender, etc.)
- One-hot encode and scale inputs
- Use the saved pipeline to predict churn
- Display prediction and probability with friendly feedback

### Key Streamlit Features:
- 🎛️ Intuitive user input forms
- ✅ Real-time churn prediction
- 📈 Confidence score (% probability)
- 🧠 Model powered by a stacked ensemble

---

## 🚀 Running the App

To run locally:

```bash
streamlit run app.py
```

To deploy online:
- Push `app.py`, `ensemble_pipeline.pkl`, and README to GitHub
- Deploy via [Streamlit Cloud](https://streamlit.io/cloud) (Free)

---

## 🧾 Technologies Used

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- PyCaret (for model comparison)
- Streamlit (for app interface)
- Joblib (for model persistence)

---

## 💡 Future Work

- Add explainability using SHAP or LIME
- Improve accuracy with stacking or deep learning
- Add customer segmentation features
- Automate retraining with new data

---

## 📬 Contact

Made with ❤️ by **Group 6 (MUST)**  
