import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.markdown("# Welcome to Bassma's Application")
st.markdown("# LinkedIn Usage Prediction App")
st.write("Based on your social media habits, we will predict if you are a LinkedIn user or NOT!!!")
st.write("Let's get started!")


@st.cache_resource
def load_and_prepare_data():
    df = pd.read_csv("social_media_usage.csv")

    def clean_sm(x):
        return np.where(x == 1, 1, 0)

    df = df[['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()
    df.columns = ['sm_li', 'income', 'education', 'parent', 'marital_status', 'gender', 'age']

    df['sm_li'] = clean_sm(df['sm_li'])

    df = df[
        (df['income'] <= 9) &
        (df['education'] <= 8) &
        (df['age'] <= 97)
    ].dropna()

    return df

@st.cache_resource
def train_model(df):
    X = df[['income', 'education', 'parent', 'marital_status', 'gender', 'age']]
    y = df['sm_li']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

# Main app questions
def main():
    # Question 1: Social Media Usage
    web1h = st.radio(
        "Please tell me if you ever use any of the following: Twitter, Instagram, Facebook, Snapchat, YouTube, WhatsApp, Pinterest, LinkedIn, Reddit, TikTok, or Nextdoor",
        options=[1, 2, 8, 9],
        format_func=lambda x: {
            1: "Yes, do this",
            2: "No, do not do this",
            8: "Don't know",
            9: "Refused"
        }[x]
    )

    # Question 2: Income Level
    income = st.selectbox(
        "What is your household income level?",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 98, 99],
        format_func=lambda x: {
            1: "Less than $10,000",
            2: "10 to under $20,000",
            3: "20 to under $30,000",
            4: "30 to under $40,000",
            5: "40 to under $50,000",
            6: "50 to under $75,000",
            7: "75 to under $100,000",
            8: "100 to under $150,000",
            9: "$150,000 or more",
            98: "Don't know",
            99: "Refused"
        }[x]
    )

    # Question 3: Education Level
    education = st.selectbox(
        "What is your highest level of school/degree completed?",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 98, 99],
        format_func=lambda x: {
            1: "Less than high school (Grades 1-8 or no formal schooling)",
            2: "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
            3: "High school graduate (Grade 12 with diploma or GED certificate)",
            4: "Some college, no degree (includes some community college)",
            5: "Two-year associate degree from a college or university",
            6: "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
            7: "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
            8: "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)",
            98: "Don't know",
            99: "Refused"
        }[x]
    )

    # Question 4: Are You a Parent?
    parent = st.radio(
        "Are you a parent of a child under 18 living in your home?",
        options=[1, 2, 8, 9],
        format_func=lambda x: {
            1: "Yes",
            2: "No",
            8: "Don't know",
            9: "Refused"
        }[x]
    )

    # Question 5: Marital Status
    marital_status = st.selectbox(
        "What is your current marital status?",
        options=[1, 2, 3, 4, 5, 6, 8, 9],
        format_func=lambda x: {
            1: "Married",
            2: "Living with a partner",
            3: "Divorced",
            4: "Separated",
            5: "Widowed",
            6: "Never been married",
            8: "Don't know",
            9: "Refused"
        }[x]
    )

    # Question 6: Gender
    gender = st.radio(
        "What is your gender?",
        options=[1, 2, 3, 98, 99],
        format_func=lambda x: {
            1: "Male",
            2: "Female",
            3: "Other",
            98: "Don't know",
            99: "Refused"
        }[x]
    )

    # Question 7: Age
    age = st.slider("What is your age?", min_value=18, max_value=97, value=30)

    # Prepare user input as a DataFrame
    user_input = pd.DataFrame({
        'income': [income],
        'education': [education],
        'parent': [1 if parent == 1 else 0],
        'marital_status': [1 if marital_status == 1 else 0],
        'gender': [1 if gender == 2 else 0],  
        'age': [age]
    })

    # Load data and train the model
    df = load_and_prepare_data()
    model = train_model(df)

    # Predict and display results
    if st.button("Predict"):
        prediction = model.predict(user_input)[0]
        probability = model.predict_proba(user_input)[0][1]
        st.subheader(f"Prediction: {"Congratulations! You're a LinkedIn User!" if prediction == 1 else "Unfortunately, it looks like you're not a LinkedIn User"}")
        st.subheader(f"Probability of LinkedIn Usage: {probability * 100:.2f}%")

if __name__ == "__main__":
    main()
