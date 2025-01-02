import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [csvStates, setCsvStates] = useState([]);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [currentStep, setCurrentStep] = useState("");

  useEffect(() => {
    let interval;
    if (loading) {
      const messages = ["Processing.", "Processing..", "Processing..."];
      let index = 0;
      interval = setInterval(() => {
        setStatusMessage(messages[index]);
        index = (index + 1) % messages.length;
      }, 500);
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [loading]);

  const uploadFile = async () => {
    if (!file) return alert("Please select a file to upload.");
    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      setCurrentStep("Uploading file...");
      const uploadResponse = await axios.post("http://127.0.0.1:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const filePath = uploadResponse.data.file_path;
      console.log("Upload Response:", uploadResponse.data);

      // Data cleaning
      setCurrentStep("Data Cleaning");
      const cleanResponse = await axios.post("http://127.0.0.1:5000/clean", { file_path: filePath });
      const cleanedFilePath = cleanResponse.data.file_path;
      setCsvStates((prev) => [...prev, { step: "Data Cleaning", filePath: cleanedFilePath }]);
      console.log("Clean Response:", cleanResponse.data);

      // Feature engineering
      setCurrentStep("Feature Engineering");
      const featureResponse = await axios.post("http://127.0.0.1:5000/feature_engineering", { file_path: cleanedFilePath });
      const featureFilePath = featureResponse.data.file_path;
      setCsvStates((prev) => [...prev, { step: "Feature Engineering", filePath: featureFilePath }]);
      console.log("Feature Engineering Response:", featureResponse.data);

      // Data exploration
      setCurrentStep("Data Exploration");
      const exploreResponse = await axios.post("http://127.0.0.1:5000/explore", { file_path: featureFilePath });
      const plots = exploreResponse.data.plots;
      setCsvStates((prev) => [...prev, { step: "Data Exploration", plots }]);
      console.log("Explore Response:", exploreResponse.data);

      setStatusMessage("Processing completed successfully!");
    } catch (error) {
      console.error("Error:", error);
      setStatusMessage("Unable to process, please try again.");
    } finally {
      setLoading(false);
      setCurrentStep("");
    }
  };

  const downloadFile = (filePath) => {
    window.open(`http://127.0.0.1:5000/download?file_path=${filePath}`, "_blank");
  };

  return (
    <div className="App">
      <h1>Data Processing Dashboard</h1>
      <div className="upload-section">
        <input
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
          accept=".csv"
        />
        <button onClick={uploadFile} disabled={loading}>
          {loading ? "Processing..." : "Upload and Start"}
        </button>
      </div>
      <div className="results-section">
        {csvStates.map((state, index) => (
          <div key={index} className="result-step">
            <h2>{state.step}</h2>
            {state.filePath && (
              <>
                <button onClick={() => downloadFile(state.filePath)}>Download CSV</button>
                <p>File Path: {state.filePath}</p>
              </>
            )}
            {state.plots &&
              Object.entries(state.plots).map(([key, value]) => (
                <div key={key}>
                  <h3>{key}</h3>
                  <img src={`http://localhost:5000/${value}`} alt={`${key} plot`} />
                </div>
              ))}
          </div>
        ))}
        {loading && currentStep && (
          <div className="processing-message">
            <p>{statusMessage}</p>
          </div>
        )}
      </div>
      {statusMessage === "Unable to process, please try again." && (
        <div className="error-message" style={{ color: "red", marginTop: "20px" }}>
          <p>{statusMessage}</p>
        </div>
      )}
    </div>
  );
}

export default App;
