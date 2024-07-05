import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna

# Disable Optuna logging
optuna.logging.disable_default_handler()

# Function to annotate percentages on count plots
def annotate_percent(ax, total):
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', fontsize=12, weight='bold')

# Streamlit App
st.title('Telco Customer Churn Analysis')

# Upload CSV
st.sidebar.header('Upload Your CSV File')
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display first few rows of the dataframe
    st.subheader('First 10 Rows of the Dataset')
    st.write(df.head(10))

    # Display basic information
    st.subheader('Basic Information')
    st.write(f"Shape: {df.shape}")
    st.write(f"Size: {df.size}")
    st.write(f"Columns: {df.columns.tolist()}")
    st.write(f"Null Values: \n{df.isnull().sum()}")

    # Check if all customer IDs are unique
    st.write(f"All customer IDs are unique: {df['customerID'].nunique() == df.shape[0]}")

    # Drop the customerID column
    df = df.drop('customerID', axis=1)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Numerical and categorical columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

    # Summary statistics for numerical columns
    st.subheader('Summary Statistics for Numerical Columns')
    st.write(df[num_cols].describe())

    # Plot histograms for numerical columns
    st.subheader('Histograms for Numerical Columns')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    sns.histplot(data=df, x='tenure', bins=30, kde=True, label='Total', ax=axes[0])
    sns.histplot(data=df[df['Churn'] == 'Yes'], x='tenure', bins=30, kde=True, label='Churn', ax=axes[0])
    axes[0].legend()
    axes[0].set_title('Tenure Distribution')

    sns.histplot(data=df, x='MonthlyCharges', bins=30, kde=True, label='Total', ax=axes[1])
    sns.histplot(data=df[df['Churn'] == 'Yes'], x='MonthlyCharges', bins=30, kde=True, label='Churn', ax=axes[1])
    axes[1].legend()
    axes[1].set_title('Monthly Charges Distribution')

    sns.histplot(data=df, x='TotalCharges', bins=30, kde=True, label='Total', ax=axes[2])
    sns.histplot(data=df[df['Churn'] == 'Yes'], x='TotalCharges', bins=30, kde=True, label='Churn', ax=axes[2])
    axes[2].legend()
    axes[2].set_title('Total Charges Distribution')

    st.pyplot(fig)

    # Plot boxplots for numerical columns against churn
    st.subheader('Boxplots for Numerical Columns vs. Churn')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    sns.boxplot(data=df, x='Churn', y='tenure', ax=axes[0])
    axes[0].set_title('Churn vs Tenure')
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1])
    axes[1].set_title('Churn vs Monthly Charges')
    sns.boxplot(data=df, x='Churn', y='TotalCharges', ax=axes[2])
    axes[2].set_title('Churn vs Total Charges')
    st.pyplot(fig)

    # Distribution of churn
    st.subheader('Churn Distribution')
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.countplot(data=df, x='Churn', ax=ax)
    annotate_percent(ax, df.shape[0])
    st.pyplot(fig)

    # Distribution of categorical columns
    st.subheader('Distribution of Categorical Columns')
    fig, axes = plt.subplots(2, 4, figsize=(32, 16))
    cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService']
    for idx, col in enumerate(cat_cols):
        ax = axes[idx // 4, idx % 4]
        sns.countplot(data=df, x=col, ax=ax)
        annotate_percent(ax, df.shape[0])
        ax.set_title(f'{col} Distribution')
    st.pyplot(fig)

    # Distribution of contract type and payment method
    st.subheader('Distribution of Contract Type and Payment Method')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    sns.countplot(data=df, x='Contract', ax=axes[0])
    annotate_percent(axes[0], df.shape[0])
    axes[0].set_title('Contract Type Distribution')

    sns.countplot(data=df, x='PaperlessBilling', ax=axes[1])
    annotate_percent(axes[1], df.shape[0])
    axes[1].set_title('Paperless Billing Distribution')

    sns.countplot(data=df, x='PaymentMethod', ax=axes[2])
    annotate_percent(axes[2], df.shape[0])
    axes[2].set_title('Payment Method Distribution')
    st.pyplot(fig)

    # Distribution of internet services
    st.subheader('Distribution of Internet Services')
    fig, axes = plt.subplots(2, 4, figsize=(32, 16))
    internet_services = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 
                         'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for idx, service in enumerate(internet_services):
        ax = axes[idx // 4, idx % 4]
        sns.countplot(data=df, x=service, ax=ax)
        annotate_percent(ax, df.shape[0])
        ax.set_title(f'{service} Distribution')
    st.pyplot(fig)

    # Distribution of phone services
    st.subheader('Distribution of Phone Services')
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    phone_services = ['PhoneService', 'MultipleLines']
    for idx, service in enumerate(phone_services):
        ax = axes[idx]
        sns.countplot(data=df, x=service, ax=ax)
        annotate_percent(ax, df.shape[0])
        ax.set_title(f'{service} Distribution')
    st.pyplot(fig)

    # Tenure vs churn by contract type
    st.subheader('Tenure vs Churn by Contract Type')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    sns.histplot(data=df[df['Contract'] == 'Month-to-month'], x='tenure', hue='Churn', bins=30, kde=True, ax=axes[0])
    axes[0].set_title('Tenure for Month-to-Month Contracts')

    sns.histplot(data=df[df['Contract'] == 'One year'], x='tenure', hue='Churn', bins=30, kde=True, ax=axes[1])
    axes[1].set_title('Tenure for One-Year Contracts')

    sns.histplot(data=df[df['Contract'] == 'Two year'], x='tenure', hue='Churn', bins=30, kde=True, ax=axes[2])
    axes[2].set_title('Tenure for Two-Year Contracts')
    st.pyplot(fig)

    # Encode InternetService
    st.subheader('Internet Service Encoding')
    label_encode = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
    df['InternetService'] = df['InternetService'].map(label_encode)
    st.write(df['InternetService'].value_counts())

else:
    st.info('Please upload a CSV file.')
