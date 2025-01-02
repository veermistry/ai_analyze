# AI-Automated Data Processing Application

This project provides a web-based dashboard for uploading, cleaning, feature engineering, exploring, and downloading CSV data files. The system consists of a Flask-based backend for handling file uploads and processing, and a React frontend for user interaction.

## Backend Overview (Flask)
The backend is built using Flask, with the following features:
- **File Upload**: Allows users to upload CSV files.
- **Data Cleaning**: Cleans missing values using different strategies (mean, median, mode, etc.).
- **Feature Engineering**: Uses LangChain agents to perform feature engineering (e.g., encoding categorical variables and generating new features).
- **Data Exploration**: Generates histograms, correlation matrices, and pairplots to visualize data.
- **File Download**: Allows users to download cleaned or processed files.

### API Endpoints
1. **POST /upload**: Uploads a CSV file.
   - Request: `multipart/form-data` with a file.
   - Response: A success message and the file path.
   
2. **POST /clean**: Cleans the uploaded data by handling missing values.
   - Request: JSON with `file_path`.
   - Response: A success message and the cleaned file path.

3. **POST /feature_engineering**: Performs feature engineering on the cleaned data.
   - Request: JSON with `file_path`.
   - Response: A success message and the file path for feature-engineered data.

4. **POST /explore**: Explores the data by generating histograms, correlation matrices, and pairplots.
   - Request: JSON with `file_path`.
   - Response: A success message and the generated plots.

5. **GET /download**: Downloads a processed file (cleaned, feature-engineered, or explored data).
   - Request: Query parameter `file_path`.
   - Response: The requested file.

### Setup
1. Clone this repository.
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```
4. Run the backend:
   ```bash
   python app.py
   ```

## Frontend Overview (React)
The frontend is a React app that interacts with the Flask backend. It allows users to:
- Upload CSV files.
- View the status of data processing (file upload, cleaning, feature engineering, and exploration).
- Download the processed files.
- View data plots (histograms, correlation matrices, and pairplots).

### Features:
- **File Upload**: Select and upload CSV files.
- **Processing Status**: Shows the current processing step with an animated message.
- **Download Files**: Download the processed data files and view generated plots.
- **Error Handling**: Displays error messages if any step fails.

### Setup
1. Clone this repository.
2. Install required npm packages:
   ```bash
   npm install
   ```
3. Run the frontend:
   ```bash
   npm start
   ```

## Usage
1. Open the frontend in a browser.
2. Upload a CSV file to begin processing.
3. The app will display progress messages while performing data cleaning, feature engineering, and exploration.
4. Once completed, you can download the processed files and view the generated plots.

## Example Workflow
1. **Upload File**: The user uploads a CSV file.
2. **Data Cleaning**: The backend processes the file and cleans missing data.
3. **Feature Engineering**: New features are created using LangChain agents.
4. **Data Exploration**: Visualizations such as histograms, correlation matrices, and pairplots are generated.
5. **Download**: The user can download the cleaned, feature-engineered data or view the plots.

## Dependencies

- Flask
- Flask-CORS
- Pandas
- Seaborn
- Matplotlib
- LangChain
- OpenAI API
- React
- Axios

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
