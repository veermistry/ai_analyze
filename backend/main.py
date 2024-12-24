from flask import Flask, request, jsonify, send_file
import os
import pandas as pd
from werkzeug.utils import secure_filename
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from scipy.stats import skew

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = api_key

# Initialize OpenAI model
model = 'gpt-4o-mini'
llm = ChatOpenAI(model_name=model, temperature=0)

def create_dynamic_dataframe_agent(df):
    return create_pandas_dataframe_agent(
        llm, 
        df, 
        agent_type=AgentType.OPENAI_FUNCTIONS,
        suffix="Always analyze based on the provided dataframe structure and return ONLY JSON arrays containing relevant cleaned data.",
        verbose=True,
        allow_dangerous_code=True
    )

def handle_missing_values(df):
    df = df.copy()
    for column in df.columns:
        if df[column].dtype == "object" or df[column].dtype.name == "category":
            mode_value = df[column].mode()[0] if not df[column].mode().empty else None
            df[column].fillna(mode_value, inplace=True)
        elif pd.api.types.is_numeric_dtype(df[column]):
            column_skew = skew(df[column].dropna(), nan_policy='omit')
            if -0.5 <= column_skew <= 0.5:
                df[column].fillna(df[column].mean(), inplace=True)
            else:
                df[column].fillna(df[column].median(), inplace=True)
    return df

def save_plot(plot_func, df, filename):
    plot_func(df)
    path = os.path.join(UPLOAD_FOLDER, filename)
    plt.savefig(path)
    plt.close()
    return path

# Routes
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return jsonify({'message': 'File uploaded successfully', 'file_path': file_path}), 200

@app.route('/clean', methods=['POST'])
def clean_data():
    file_path = request.json.get('file_path')
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    df = pd.read_csv(file_path)
    cleaned_df = handle_missing_values(df)
    cleaned_path = os.path.join(UPLOAD_FOLDER, 'cleaned_data.csv')
    cleaned_df.to_csv(cleaned_path, index=False)
    return jsonify({'message': 'Data cleaned', 'file_path': cleaned_path}), 200

@app.route('/feature_engineering', methods=['POST'])
def feature_engineering():
    file_path = request.json.get('file_path')
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    df = pd.read_csv(file_path)
    agent = create_dynamic_dataframe_agent(df)
    query = (
        "Encode categorical columns in the dataframe only if it's meaningful."
        "Generate new meaningful features if possible (e.g., age groups, interaction terms)."
        "Return the updated dataframe as a JSON array without any additional text or explanation."
    )
    response = agent.invoke(query)
    raw_output = response['output']
    start_index = raw_output.find('[')
    end_index = raw_output.rfind(']')
    raw_output = raw_output[start_index:end_index + 1]

    try:
        json_data = json.loads(raw_output)
        modified_df = pd.DataFrame(json_data) if isinstance(json_data, list) else pd.json_normalize(json_data)
        modified_path = os.path.join(UPLOAD_FOLDER, 'feature_engineered_data.csv')
        modified_df.to_csv(modified_path, index=False)
        return jsonify({'message': 'Feature engineering complete', 'file_path': modified_path}), 200
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Error parsing JSON: {e}'}), 500

@app.route('/explore', methods=['POST'])
def explore_data():
    file_path = request.json.get('file_path')
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    df = pd.read_csv(file_path)
    plot_paths = {
        'histograms': save_plot(lambda df: df.hist(bins=20, figsize=(14, 10)), df, 'histograms.png'),
        'correlation_matrix': save_plot(lambda df: sns.heatmap(df.corr(), annot=True, cmap='coolwarm'), df, 'correlation_matrix.png'),
        'pairplot': save_plot(lambda df: sns.pairplot(df.select_dtypes(include=['number'])), df, 'pairplot.png'),
    }
    return jsonify({'message': 'Exploration complete', 'plots': plot_paths}), 200

@app.route('/download', methods=['GET'])
def download_file():
    file_path = request.args.get('file_path')
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(file_path, as_attachment=True)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
