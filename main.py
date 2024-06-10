import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
optuna.logging.disable_default_handler()


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


print(df.head(10))
print(f"Shape: {df.shape}")
print(f"Size: {df.size}")
print(f"Columns: {df.columns.tolist()}")
print(f"Null Values: \n{df.isnull().sum()}")


print(f"All customer IDs are unique: {df['customerID'].nunique() == df.shape[0]}")


df = df.drop('customerID', axis=1)


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
print(df[num_cols].describe())


plt.figure(figsize=(24, 8))
sns.set(style='whitegrid', palette='deep')


plt.subplot(1, 3, 1)
sns.histplot(data=df, x='tenure', bins=30, kde=True, label='Total')
sns.histplot(data=df[df['Churn'] == 'Yes'], x='tenure', bins=30, kde=True, label='Churn')
plt.legend()
plt.title('Tenure Distribution')

plt.subplot(1, 3, 2)
sns.histplot(data=df, x='MonthlyCharges', bins=30, kde=True, label='Total')
sns.histplot(data=df[df['Churn'] == 'Yes'], x='MonthlyCharges', bins=30, kde=True, label='Churn')
plt.legend()
plt.title('Monthly Charges Distribution')


plt.subplot(1, 3, 3)
sns.histplot(data=df, x='TotalCharges', bins=30, kde=True, label='Total')
sns.histplot(data=df[df['Churn'] == 'Yes'], x='TotalCharges', bins=30, kde=True, label='Churn')
plt.legend()
plt.title('Total Charges Distribution')

plt.show()


plt.figure(figsize=(24, 8))
sns.set(style='whitegrid')

plt.subplot(1, 3, 1)
sns.boxplot(data=df, x='Churn', y='tenure')
plt.title('Churn vs Tenure')

plt.subplot(1, 3, 2)
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Churn vs Monthly Charges')

plt.subplot(1, 3, 3)
sns.boxplot(data=df, x='Churn', y='TotalCharges')
plt.title('Churn vs Total Charges')

plt.show()


def annotate_percent(ax, total):
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() / 2
        ax.annotate(percentage, (x, y), ha='center', fontsize=15, weight='bold')

plt.figure(figsize=(8, 8))
sns.set(style='whitegrid')
ax = sns.countplot(data=df, x='Churn')
annotate_percent(ax, df.shape[0])
plt.title('Churn Distribution')
plt.show()

plt.figure(figsize=(32, 8))
sns.set(style='whitegrid')

plt.subplot(1, 4, 1)
ax1 = sns.countplot(data=df, x='gender')
annotate_percent(ax1, df.shape[0])
plt.title('Gender Distribution')

plt.subplot(1, 4, 2)
ax2 = sns.countplot(data=df, x='SeniorCitizen')
annotate_percent(ax2, df.shape[0])
plt.title('Senior Citizen Distribution')

plt.subplot(1, 4, 3)
ax3 = sns.countplot(data=df, x='Partner')
annotate_percent(ax3, df.shape[0])
plt.title('Partner Distribution')

plt.subplot(1, 4, 4)
ax4 = sns.countplot(data=df, x='Dependents')
annotate_percent(ax4, df.shape[0])
plt.title('Dependents Distribution')

plt.show()


plt.figure(figsize=(24, 8))
sns.set(style='whitegrid')

plt.subplot(1, 3, 1)
ax1 = sns.countplot(data=df, x='Contract')
annotate_percent(ax1, df.shape[0])
plt.title('Contract Type Distribution')

plt.subplot(1, 3, 2)
ax2 = sns.countplot(data=df, x='PaperlessBilling')
annotate_percent(ax2, df.shape[0])
plt.title('Paperless Billing Distribution')

plt.subplot(1, 3, 3)
ax3 = sns.countplot(data=df, x='PaymentMethod')
annotate_percent(ax3, df.shape[0])
plt.title('Payment Method Distribution')

plt.show()


plt.figure(figsize=(32, 16))
sns.set(style='whitegrid')

internet_services = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for idx, service in enumerate(internet_services):
    plt.subplot(2, 4, idx + 1)
    ax = sns.countplot(data=df, x=service)
    annotate_percent(ax, df.shape[0])
    plt.title(f'{service} Distribution')

plt.show()


plt.figure(figsize=(16, 8))
sns.set(style='whitegrid')

phone_services = ['PhoneService', 'MultipleLines']

for idx, service in enumerate(phone_services):
    plt.subplot(1, 2, idx + 1)
    ax = sns.countplot(data=df, x=service)
    annotate_percent(ax, df.shape[0])
    plt.title(f'{service} Distribution')

plt.show()


plt.figure(figsize=(24, 8))
sns.set(style='whitegrid')

plt.subplot(1, 3, 1)
ax1 = sns.histplot(data=df[df['Contract'] == 'Month-to-month'], x='tenure', hue='Churn', bins=30, kde=True)
plt.title('Tenure for Month-to-Month Contracts')

plt.subplot(1, 3, 2)
ax2 = sns.histplot(data=df[df['Contract'] == 'One year'], x='tenure', hue='Churn', bins=30, kde=True)
plt.title('Tenure for One-Year Contracts')

plt.subplot(1, 3, 3)
ax3 = sns.histplot(data=df[df['Contract'] == 'Two year'], x='tenure', hue='Churn', bins=30, kde=True)
plt.title('Tenure for Two-Year Contracts')

plt.show()


label_encode = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
df['InternetService'] = df['InternetService'].map(label_encode)
print(df['InternetService'].value_counts())
