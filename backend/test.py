import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your cleaned DataFrame

# Set the style for the plots
sns.set(style="whitegrid")

# 1. Overview of Data Distribution with Histograms for Numerical Columns Only
def plot_histograms(df):
    numerical_columns = df.select_dtypes(include=['number']).columns
    df[numerical_columns].hist(bins=20, figsize=(14, 10))
    plt.suptitle("Histograms of Numerical Columns", fontsize=16)
    plt.savefig('histograms.png')  # Save the plot as an image
    plt.close()

# 2. Correlation Matrix Heatmap for Numerical Columns Only
def plot_correlation_matrix(df):
    numerical_columns = df.select_dtypes(include=['number']).columns
    corr = df[numerical_columns].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix (Numerical Features)', fontsize=16)
    plt.savefig('correlation_matrix.png')  # Save the plot as an image
    plt.close()

# 3. Pairplot for Relationships between Numerical Features Only
def plot_pairplot(df):
    numerical_columns = df.select_dtypes(include=['number']).columns
    sns.pairplot(df[numerical_columns], diag_kind='kde', height=2.5)
    plt.suptitle('Pairplot of Numerical Features', y=1.02, fontsize=16)
    plt.savefig('pairplot.png')  # Save the plot as an image
    plt.close()

# 4. Boxplot to Visualize Outliers for Numerical Features Only
def plot_boxplot(df):
    numerical_columns = df.select_dtypes(include=['number']).columns
    plt.figure(figsize=(14, 10))
    sns.boxplot(data=df[numerical_columns], orient='h', palette='Set2')
    plt.title('Boxplot of Numerical Features to Identify Outliers', fontsize=16)
    plt.savefig('boxplot.png')  # Save the plot as an image
    plt.close()

# 5. Heatmap of Missing Values (for all columns)
def plot_missing_values(df):
    plt.figure(figsize=(10, 7))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', cbar_kws={'label': 'Missing Data'})
    plt.title('Missing Data Heatmap', fontsize=16)
    plt.savefig('missing_data_heatmap.png')  # Save the plot as an image
    plt.close()

# 6. Bar Graph for Categorical Columns
def plot_categorical_bars(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=column, palette='Set3')
        plt.title(f'Bar Plot for {column}', fontsize=16)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()
        plt.savefig(f'barplot_{column}.png')  # Save the plot as an image
        plt.close()


df = pd.read_csv('../iris.csv')
# Example usage:
plot_histograms(df)
plot_correlation_matrix(df)
plot_pairplot(df)
plot_boxplot(df)
plot_missing_values(df)
plot_categorical_bars(df)
