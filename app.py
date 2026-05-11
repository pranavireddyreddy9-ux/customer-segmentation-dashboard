import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Page settings
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# Page title
st.title("Customer Segmentation & Business Insights Dashboard")

# Sidebar
st.sidebar.title("Dashboard Menu")

option = st.sidebar.selectbox(
    "Choose Section",
    ["Dataset", "Clusters", "Insights"]
)

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Train model
kmeans = KMeans(n_clusters=5, random_state=42)

# Predict clusters
df['Cluster'] = kmeans.fit_predict(X)

# KPI Metrics
total_customers = len(df)

average_income = round(df['Annual Income (k$)'].mean(), 2)

average_score = round(df['Spending Score (1-100)'].mean(), 2)

# Create KPI columns
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", total_customers)

col2.metric("Average Income", average_income)

col3.metric("Average Spending Score", average_score)

# DATASET SECTION
if option == "Dataset":

    st.subheader("Customer Dataset")

    st.write(df.head())

# CLUSTER SECTION
if option == "Clusters":

    st.subheader("Clustered Customers")

    st.write(df)

    # Scatter plot
    fig, ax = plt.subplots()

    scatter = ax.scatter(
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=df['Cluster']
    )

    ax.set_xlabel("Annual Income")

    ax.set_ylabel("Spending Score")

    ax.set_title("Customer Segments")

    st.pyplot(fig)

# INSIGHTS SECTION
if option == "Insights":

    st.subheader("Business Insights")

    st.write("Cluster 0 → Balanced Customers")

    st.write("Cluster 1 → High Spending Customers")

    st.write("Cluster 2 → Budget Customers")

    st.write("Cluster 3 → Premium Customers")

    st.write("Cluster 4 → Low Engagement Customers")

# CUSTOMER PREDICTION SECTION

st.sidebar.subheader("Predict Customer Cluster")

income = st.sidebar.slider(
    "Enter Annual Income",
    0,
    150,
    50
)

score = st.sidebar.slider(
    "Enter Spending Score",
    0,
    100,
    50
)

# Predict cluster
prediction = kmeans.predict([[income, score]])

cluster_names = {
    0: "Balanced Customers",
    1: "High Spending Customers",
    2: "Budget Customers",
    3: "Premium Customers",
    4: "Low Engagement Customers"
}

result = cluster_names[prediction[0]]

st.sidebar.subheader("Prediction Result")

st.sidebar.success(
    f"Customer Type: {result}"
)

# PIE CHART SECTION

st.subheader("Customer Cluster Distribution")

cluster_names_chart = {
    0: "Balanced",
    1: "High Spending",
    2: "Budget",
    3: "Premium",
    4: "Low Engagement"
}

cluster_counts = df['Cluster'].map(cluster_names_chart).value_counts()

fig2, ax2 = plt.subplots()

ax2.pie(
    cluster_counts,
    labels=cluster_counts.index,
    autopct='%1.1f%%'
)

st.pyplot(fig2)