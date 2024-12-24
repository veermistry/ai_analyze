# Flask Data Processing Application

This Flask application allows users to upload datasets, process them through multiple phases (data cleaning, feature engineering, and data exploration) using LangChain, and download the results at each stage. It will provide a user-friendly interface for interacting with the data processing pipeline and offers visualization and structured data exploration.

## Features

### 1. Upload Dataset
- Users can upload datasets in CSV format.
- The uploaded dataset is stored for further processing.

### 2. Data Cleaning
- Handles missing values by:
  - Imputing categorical columns with the mode.
  - Imputing numerical columns based on skewness:
    - Mean for normally distributed data.
    - Median for skewed data.
- Saves the cleaned dataset for download.

### 3. Feature Engineering
- Encodes categorical columns.
- Generates new meaningful features (e.g., interaction terms, age groups).
- Saves the updated dataset for download.

### 4. Data Exploration
- Provides visualizations, including:
  - Histograms for numerical columns.
  - Correlation matrix heatmaps.
  - Pairplots for numerical features.
  - Boxplots for identifying outliers.
  - Heatmaps of missing values.
  - Bar plots for categorical features.
- Allows users to download the visualization results.

### 5. API Routes
The application provides the following API routes:

#### `/upload`
- **Method**: `POST`
- **Description**: Upload a dataset in CSV format.
- **Response**: Confirms the successful upload of the dataset.

#### `/clean`
- **Method**: `POST`
- **Description**: Cleans the uploaded dataset by handling missing values.
- **Response**: Returns the cleaned dataset as a downloadable file.

#### `/feature-engineer`
- **Method**: `POST`
- **Description**: Performs feature engineering on the cleaned dataset.
- **Response**: Returns the updated dataset as a downloadable file.

#### `/explore`
- **Method**: `GET`
- **Description**: Generates visualizations for data exploration.
- **Response**: Returns downloadable visualization files.

#### `/download/<phase>`
- **Method**: `GET`
- **Description**: Allows users to download the dataset for a specific phase (`cleaned`, `featured`, etc.).
- **Response**: Provides the requested dataset as a downloadable file.

## Requirements
- Python 3.8+
- Flask
- LangChain
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scipy

## Setup

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (e.g., OpenAI API key):
   - Create a `.env` file:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     ```

5. Run the application:
   ```bash
   flask run
   ```

6. Open the application in your browser:
   - Navigate to `http://127.0.0.1:5000`

## Usage

- Navigate through the web interface or use API routes via tools like Postman or cURL.
- Upload your dataset, process it through the pipeline, and download the results at each stage.

## Example Workflow

1. Upload a dataset to `/upload`.
2. Clean the data by sending a request to `/clean`.
3. Perform feature engineering via `/feature-engineer`.
4. Generate visualizations using `/explore`.
5. Download processed datasets and visualizations from `/download/<phase>`.

## Future Enhancements
- Add database integration (e.g., SQLite, PostgreSQL) for storing datasets and processing metadata.
- Implement real-time progress tracking for processing tasks.
- Support additional file formats (e.g., Excel, JSON).
- Include user authentication for secure dataset handling.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

