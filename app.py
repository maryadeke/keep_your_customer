
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import docx
from io import BytesIO

def load_uploaded_data(file) -> pd.DataFrame:
    """Read and return a DataFrame from various file formats."""
    try:
        file_type = file.name.split('.')[-1].lower()

        if file_type == 'csv':
            return pd.read_csv(file)

        elif file_type in ['xls', 'xlsx']:
            return pd.read_excel(file)

        elif file_type == 'docx':
            doc = docx.Document(file)
            table = doc.tables[0]  # Use the first table

            # Extract headers
            headers = [cell.text.strip() for cell in table.rows[0].cells]
            records = []

            for row in table.rows[1:]:
                row_data = [cell.text.strip() for cell in row.cells]
                records.append(dict(zip(headers, row_data)))

            data = pd.DataFrame(records)

            # Strip whitespace and replace empty strings
            data.replace('', pd.NA, inplace=True)
            data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            # Try to convert numeric columns
            for col in data.columns:
                try:
                    data[col] = pd.to_numeric(data[col])
                except:
                    pass  # Leave non-numeric columns unchanged
            # Fix common categorical fields (if needed)
            if 'Gender' in data.columns:
                data['Gender'] = data['Gender'].str.title().str.strip()

            if 'Promotion_Response' in data.columns:
                data['Promotion_Response'] = data['Promotion_Response'].str.title().str.strip()
            return data


        elif file_type in ['db', 'sqlite']:
            with open("temp.db", "wb") as f:
                f.write(file.read())
            con = sqlite3.connect("temp.db")
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", con)
            table_names = tables['name'].tolist()
            if table_names:
                selected = st.selectbox("Choose a table", table_names)
                return pd.read_sql_query(f"SELECT * FROM {selected}", con)
            else:
                st.warning("No tables found in database.")
                return pd.DataFrame()

        else:
            st.error("Unsupported file format.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return pd.DataFrame()
    


# Set page config
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

#Define the custom class
from sklearn.base import BaseEstimator, TransformerMixin

class DropIDTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop('Customer_ID', axis=1, errors='ignore')

# Load model
pipeline = joblib.load('ensemble_pipeline.pkl')

st.markdown("## 🔮 Customer Churn Prediction")
st.markdown("Predict whether a customer will churn based on various features.")

# Sidebar for navigation
option = st.sidebar.radio("Choose input mode:", ["🔘 Manual Entry", "📂 Upload File"])

# --- Manual Input Form ---
if option == "🔘 Manual Entry":
    with st.form("manual_form"):
        st.subheader("📝 Manual Customer Data Input")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0)
            income = st.number_input("Annual Income")
            total_spend = st.number_input("Total Spend")
            avg_transaction = st.number_input("Average Transaction Amount")
            last_purchase_days = st.number_input("Last Purchase Days Ago")
            years_as_customer = st.number_input("Years as Customer", min_value=0)
        with col2:
            satisfaction_score = st.number_input("Satisfaction Score (1-5)", min_value=1, max_value=5)
            num_purchases = st.number_input("Number of Purchases", min_value=0)
            num_returns = st.number_input("Number of Returns", min_value=0)
            num_support_contacts = st.number_input("Number of Support Contacts", min_value=0)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            promotion_response = st.selectbox("Promotion Response", ["None", "Responded", "Unsubscribed"])

        submit = st.form_submit_button("Predict")

        if submit:
            input_df = pd.DataFrame([{
                "Age": age,
                "Annual_Income": income,
                "Gender": gender,
                "Promotion_Response": promotion_response,
                "Total_Spend": total_spend,
                "Years_as_Customer": years_as_customer,
                "Satisfaction_Score": satisfaction_score,
                "Num_of_Purchases": num_purchases,
                "Num_of_Returns": num_returns,
                "Num_of_Support_Contacts": num_support_contacts,
                "Average_Transaction_Amount": avg_transaction,
                "Last_Purchase_Days_Ago": last_purchase_days
            }])

            prediction = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.error(f"⚠️ Customer is likely to churn! Probability: {probability:.2%}")
            else:
                st.success(f"✅ Customer is unlikely to churn. Probability: {probability:.2%}")



# --- File Upload for Batch Prediction ---
# --- File Upload for Batch Prediction ---
if option == "📂 Upload File":
    st.subheader("📄 Upload Customer Data File")
    uploaded_file = st.file_uploader("Upload CSV, Excel, Word (.docx), or SQLite (.db)",
                                     type=["csv", "xls", "xlsx", "docx", "db", "sqlite"])

    if uploaded_file:
        data = load_uploaded_data(uploaded_file)

        if not data.empty:
            # Fill missing expected features with zeros
            required_features = pipeline.named_steps['preprocess'].feature_names_in_
            for col in required_features:
                if col not in data.columns:
                    data[col] = 0
            data.fillna(0, inplace=True)

            # Predict
            preds = pipeline.predict(data)
            
            data = data.replace('', pd.NA)
            data = data.dropna(how='all')  # remove fully empty rows
            data.fillna(0, inplace=True)   # convert NA to 0

            probs = pipeline.predict_proba(data)[:, 1]
            data['Churn_Prediction'] = preds
            data['Churn_Probability'] = probs

            st.success("✅ Prediction complete!")
            st.dataframe(data.head(10))

            churn_rate = data['Churn_Prediction'].mean()
            st.metric(label="📉 Predicted Churn Rate", value=f"{churn_rate:.2%}")

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

            st.subheader("📌 Feature Insights & Retention Suggestions")

            # Get feature importance
            classifier = pipeline.named_steps['classifier']
            preprocessor = pipeline.named_steps['preprocess']
            feature_names = preprocessor.get_feature_names_out()

            # Only works for tree-based models like RandomForest
            if hasattr(classifier.named_estimators_['rf'], 'feature_importances_'):
                importances = classifier.named_estimators_['rf'].feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)

                st.markdown("### 🔍 Feature Importance Ranking")
                st.dataframe(importance_df.head(10), use_container_width=True)

                st.markdown("### 💡 Insights & Suggested Actions")

                for _, row in importance_df.head(4).iterrows():
                    feature = row['Feature']
                    score = row['Importance']

                    # Display the feature and its contribution
                    st.markdown(f"**🔹 {feature}** – Importance: `{score:.3f}`")

                    # Dynamic suggestions based on partial keyword matches
                    if "Engagement" in feature:
                        st.markdown("""
                        - **Low Engagement Score** often indicates a lack of customer interaction with your platform.
                        - 📉 **Churn Insight**: Disengaged users are more likely to leave as they find little value or interest.
                        - 💡 **Retention Strategy**: Launch loyalty programs, offer exclusive content, and use push notifications or gamification to keep users involved.
                        """)

                    elif "Return" in feature:
                        st.markdown("""
                        - **High Return Rate** usually signals dissatisfaction with product quality or service.
                        - 📉 **Churn Insight**: Repeated returns reduce trust and satisfaction, pushing customers away.
                        - 💡 **Retention Strategy**: Improve product quality, provide better sizing/fit info, and simplify the return process to build trust.
                        """)

                    elif "Email_Opt_In" in feature:
                        st.markdown("""
                        - **Email Opt-Out** suggests users don’t find your communications helpful or relevant.
                        - 📉 **Churn Insight**: Lack of communication reduces opportunities to engage, upsell, or build loyalty.
                        - 💡 **Retention Strategy**: Personalize emails, send fewer but more valuable messages, and clearly highlight benefits.
                        """)

                    elif "Promotion_Response" in feature:
                        st.markdown("""
                        - **Lack of Promotion Response** indicates offers are not compelling or poorly targeted.
                        - 📉 **Churn Insight**: If users ignore your promotions, they may not see enough value to stay.
                        - 💡 **Retention Strategy**: Use purchase history and segmentation to craft personalized, time-sensitive offers.
                        """)

                    elif "Last_Purchase" in feature:
                        st.markdown("""
                        - **Many Days Since Last Purchase** shows reduced buying behavior.
                        - 📉 **Churn Insight**: Inactivity often precedes churn — customers lose interest or switch to competitors.
                        - 💡 **Retention Strategy**: Trigger win-back campaigns with incentives, new arrivals, or reminders based on last purchase behavior.
                        """)

                    elif "Transaction" in feature:
                        st.markdown("""
                        - **Low Average Transaction Amount** means small, infrequent purchases.
                        - 📉 **Churn Insight**: Low spenders are easier to lose and often less loyal.
                        - 💡 **Retention Strategy**: Offer volume discounts, bundles, or suggest complementary items to increase cart value.
                        """)

                    elif "Total_Spend" in feature:
                        st.markdown("""
                        - **Low Total Spend** suggests limited customer value and engagement.
                        - 📉 **Churn Insight**: Customers who never ramp up their spend may churn without ever becoming profitable.
                        - 💡 **Retention Strategy**: Create loyalty tiers, provide spend-based rewards, or offer onboarding incentives to build value.
                        """)

                    elif "Support" in feature:
                        st.markdown("""
                        - **High Number of Support Contacts** may point to unresolved issues or poor user experience.
                        - 📉 **Churn Insight**: Frustrated users are highly likely to leave, especially if issues recur.
                        - 💡 **Retention Strategy**: Improve self-service support, reduce resolution time, and follow up on support tickets with satisfaction surveys.
                        """)

                    elif "Satisfaction" in feature:
                        st.markdown("""
                        - **Low Satisfaction Score** is a direct churn indicator.
                        - 📉 **Churn Insight**: Unhappy customers are at high risk of switching to competitors.
                        - 💡 **Retention Strategy**: Collect detailed feedback, fix pain points, and offer personal apologies or credits for bad experiences.
                        """)

                    elif "Years_as_Customer" in feature:
                        st.markdown("""
                        - **Few Years as Customer** often means a fragile relationship still in development.
                        - 📉 **Churn Insight**: New customers are less loyal and more likely to leave if not impressed early on.
                        - 💡 **Retention Strategy**: Build strong first impressions with welcome bonuses, guided onboarding, and quick value delivery.
                        """)

                    elif "Num_of_Purchases" in feature:
                        st.markdown("""
                        - **Low Purchase Count** reflects minimal engagement with your offerings.
                        - 📉 **Churn Insight**: Infrequent buyers feel less connected and are easier to lose.
                        - 💡 **Retention Strategy**: Encourage repurchase with personalized suggestions, seasonal sales, or limited-time loyalty perks.
                        """)
                    elif "Annual_Income" in feature:
                        st.markdown("""
                        - **Low Annual Income** may reflect limited purchasing power or price sensitivity.
                        - 📉 **Churn Insight**: Customers with lower income might churn if your offerings don't match their budget.
                        - 💡 **Retention Strategy**: Offer budget-friendly bundles, payment plans, or highlight cost-effective options to appeal to this segment.
                        """)

                    elif "Gender" in feature:
                        st.markdown("""
                        - **Gender Correlation** may reveal differences in churn rates between male/female/other customers.
                        - 📉 **Churn Insight**: Gender-specific patterns can emerge from marketing mismatch or product relevance.
                        - 💡 **Retention Strategy**: Tailor communication tone, product selection, and ad visuals to better engage each segment.
                        """)
                    else:
                        st.markdown("### ☹️ No Insights Available")
                        st.markdown("Please try again with different input data or select a different mode.")

                    st.markdown("---")


            # Show total churn count
            churn_count = data['Churn_Prediction'].sum()
            total = len(data)
            st.metric("🔢 Total Churning Customers", f"{churn_count} out of {total}")
