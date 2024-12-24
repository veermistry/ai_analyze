import pandas as pd
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from scipy.stats import skew
import json

# Load environment variables
load_dotenv()

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = api_key

# Initialize OpenAI Model
model = 'gpt-4o-mini'
llm = ChatOpenAI(
    model_name=model,
    temperature=0
)

# Function to create a dynamic dataframe agent
def create_dynamic_dataframe_agent(df: pd.DataFrame):
    """
    Creates a dynamic data analysis agent for any dataframe.
    """
    return create_pandas_dataframe_agent(
        llm, 
        df, 
        agent_type=AgentType.OPENAI_FUNCTIONS,
        suffix="Always analyze based on the provided dataframe structure and return ONLY JSON arrays containing relevant cleaned data.",
        verbose=True,
        allow_dangerous_code=True
    )

def handle_missing_values(df):
    df = df.copy()  # Avoid modifying the original DataFrame
    for column in df.columns:
        if df[column].dtype == "object" or df[column].dtype.name == "category":
            # Mode imputation for categorical data
            mode_value = df[column].mode()[0] if not df[column].mode().empty else None
            df[column].fillna(mode_value, inplace=True)
        elif np.issubdtype(df[column].dtype, np.number):
            # Check skewness for numerical columns
            column_skew = skew(df[column].dropna(), nan_policy='omit')
            if -0.5 <= column_skew <= 0.5:
                # Normal distribution -> Mean imputation
                mean_value = df[column].mean()
                df[column].fillna(mean_value, inplace=True)
            else:
                # Skewed distribution -> Median imputation
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
    return df

# Load any dataset (example placeholder)
df = pd.read_csv('../iris.csv')

# Display the dataframe to ensure correct loading
print(df.head())

df = handle_missing_values(df)

df.to_csv("cleaned_data.csv", index=False)

# Create an agent for the loaded dataframe
data_analysis_agent = create_dynamic_dataframe_agent(df)

# Run a query dynamically based on the dataframe's features
query = (
    "Encode categorical columns in the dataframe only if it's meaningful."
    "Generate new meaningful features if possible (e.g., age groups, interaction terms)."
    "Return the updated dataframe as a JSON array without any additional text or explanation."
)

response = data_analysis_agent.invoke(query)

# Print the agent response
# print("Agent Response:")
# print(response['output'])

# Check and process the agent's output
raw_output = response['output']

# Find the index of the first opening bracket/brace
start_index = raw_output.find('[') if '[' in raw_output else raw_output.find('{')

# Find the index of the last closing bracket/brace
end_index = raw_output.rfind('}') if '}' == raw_output[start_index] else raw_output.rfind(']')

# Trim the output to include from the first opening bracket/brace to the last closing bracket/brace
raw_output = raw_output[start_index:end_index + 1]

if not raw_output.strip():
    print("Agent response is empty or not provided.")
else:
    try:
        # Parse JSON
        json_data = json.loads(raw_output)
        print("\nParsed JSON data:\n")

        # Convert to DataFrame
        if isinstance(json_data, list):  # List of records
            modified_df = pd.DataFrame(json_data)
        else:  # Nested dictionary or object
            modified_df = pd.json_normalize(json_data)

        print("Modified DataFrame:")
        print(modified_df.head())

        # Save the DataFrame
        modified_df.to_csv("cleaned_or_featured_data.csv", index=False)
        print("Modified DataFrame saved as 'cleaned_or_featured_data.csv'")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}. Response output was:\n{raw_output}")

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

# Example usage:
plot_histograms(df)
plot_correlation_matrix(df)
plot_pairplot(df)
plot_boxplot(df)
plot_missing_values(df)
plot_categorical_bars(df)
