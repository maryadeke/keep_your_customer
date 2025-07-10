
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

#Define the custom class
from sklearn.base import BaseEstimator, TransformerMixin

class DropIDTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop('Customer_ID', axis=1)

# Load model
pipeline = joblib.load('ensemble_pipeline.pkl')

st.markdown("## 🔮 Customer Churn Prediction")
st.markdown("Predict whether a customer will churn based on various features.")

# Sidebar for navigation
option = st.sidebar.radio("Choose input mode:", ["🔘 Manual Entry", "📂 Upload CSV"])

# --- Manual Input Form ---
if option == "🔘 Manual Entry":
    with st.form("manual_form"):
        st.subheader("📝 Manual Customer Data Input")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0)
            income = st.number_input("Annual Income")
            transaction_amount = st.number_input("Average Transaction Amount")
            days_ago = st.number_input("Last Purchase Days Ago")
            spend = st.number_input("Spend per Year")
            return_rate = st.number_input("Return Rate")
            engagement = st.number_input("Engagement Score", min_value=0)
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            email_opt_in = st.checkbox("Email Opt-In")
            promotion_response = st.selectbox("Promotion Response", ["None", "Responded", "Unsubscribed"])

        submit = st.form_submit_button("Predict")

        if submit:
            gender_cols = {"Female": [1, 0, 0], "Male": [0, 1, 0], "Other": [0, 0, 1]}
            promo_cols = {"None": [0, 0], "Responded": [1, 0], "Unsubscribed": [0, 1]}

            input_df = pd.DataFrame([[
                age, income, transaction_amount, days_ago,
                *gender_cols[gender],
                int(email_opt_in),
                *promo_cols[promotion_response],
                spend, return_rate, engagement
            ]], columns=[
                "Age", "Annual_Income", "Average_Transaction_Amount", "Last_Purchase_Days_Ago",
                "Gender_Female", "Gender_Male", "Gender_Other",
                "Email_Opt_In_True",
                "Promotion_Response_Responded", "Promotion_Response_Unsubscribed",
                "Spend_per_Year", "Return_Rate", "Engagement_Score"
            ])

            prediction = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.error(f"⚠️ Customer is likely to churn! Probability: {probability:.2%}")
            else:
                st.success(f"✅ Customer is unlikely to churn. Probability: {probability:.2%}")

# --- File Upload for Batch Prediction ---
if option == "📂 Upload CSV":
    st.subheader("📄 Upload Customer Data File")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Preprocessing: Add missing one-hot columns if not present
        for col in ["Gender_Female", "Gender_Male", "Gender_Other",
                    "Promotion_Response_Responded", "Promotion_Response_Unsubscribed",
                    "Email_Opt_In_True"]:
            if col not in data.columns:
                data[col] = 0

        # Predict
        preds = pipeline.predict(data)
        probs = pipeline.predict_proba(data)[:, 1]
        data['Churn_Prediction'] = preds
        data['Churn_Probability'] = probs

        # Display results
        st.success("✅ Prediction complete!")
        st.dataframe(data.head(10))

        # Summary
        churn_rate = data['Churn_Prediction'].mean()
        st.metric(label="📉 Predicted Churn Rate", value=f"{churn_rate:.2%}")

        # Download option
        csv = data.to_csv(index=False).encode()
        st.download_button("📥 Download Predictions", csv, file_name="churn_predictions.csv", mime="text/csv")

        # --- Visualizations ---
        st.subheader("📊 Visual Insights")

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            sns.histplot(data['Churn_Probability'], kde=True, bins=20, ax=ax1, color="orange")
            ax1.set_title("Churn Probability Distribution")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.countplot(x='Churn_Prediction', data=data, palette="Set2", ax=ax2)
            ax2.set_title("Churn vs Non-Churn Count")
            ax2.set_xticklabels(['No Churn', 'Churn'])
            st.pyplot(fig2)

        st.markdown("✅ **Interpretation:**")
        st.markdown("- High churn probability implies a customer is likely to leave.")
        #st.markdown("- Use features like 'Engagement Score' or 'Return Rate' to target retention efforts.")

        # --- Feature Importance & Suggestions ---
        st.subheader("📌 Feature Insights & Retention Suggestions")

        # Extract trained classifier from pipeline
        classifier = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocess']
        feature_names = preprocessor.get_feature_names_out()

        # Feature importances (works with tree-based models)
        if hasattr(classifier.named_estimators_['rf'], 'feature_importances_'):
            importances = classifier.named_estimators_['rf'].feature_importances_
            importance_df = pd.DataFrame({
                'Feature': preprocessor.get_feature_names_out(),
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            st.dataframe(importance_df.head(10))

            # Dynamic suggestion logic
            top_feature = importance_df.iloc[0]['Feature']
            st.info(f"🚨 **Top feature causing churn:** `{top_feature}`")

            st.markdown("### 💡 Suggested Actions:")
            if "Engagement" in top_feature:
                st.markdown("- Consider launching loyalty programs or exclusive offers to increase customer engagement.")
            elif "Return" in top_feature:
                st.markdown("- Review product quality and streamline the return process.")
            elif "Email_Opt_In" in top_feature:
                st.markdown("- Encourage customers to subscribe to emails with clear value propositions.")
            elif "Promotion_Response" in top_feature:
                st.markdown("- Run more personalized promotions based on purchase history.")
            elif "Last_Purchase" in top_feature:
                st.markdown("- Target customers who haven’t purchased recently with win-back campaigns.")
            elif "Average_Transaction" in top_feature:
                st.markdown("- Provide discounts or bundles to increase purchase amounts.")
            else:
                st.markdown("- Analyze top features and design campaigns focused on them.")

            # Show total churn count
            churn_count = data['Churn_Prediction'].sum()
            total = len(data)
            st.metric("🔢 Total Churning Customers", f"{churn_count} out of {total}")
