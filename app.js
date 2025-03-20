import React, { useState } from "react";
import axios from "axios";

const BASE_URL = "http://127.0.0.1:5000"; // Flask backend URL

function App() {
  const [inputData, setInputData] = useState({
    PM10: "",
    NO: "",
    NO2: "",
    NOx: "",
    NH3: "",
    CO: "",
    SO2: "",
    O3: "",
    Benzene: "",
    Toluene: "",
    Xylene: "",
  });

  const [prediction, setPrediction] = useState(null);

  // Handle input change
  const handleChange = (e) => {
    setInputData({ ...inputData, [e.target.name]: e.target.value });
  };

  // Send data to backend
  const handlePredict = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post(`${BASE_URL}/predict`, inputData);
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Prediction error:", error);
      setPrediction("Error making prediction");
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h2>Air Pollution Prediction</h2>
      <form onSubmit={handlePredict}>
        {Object.keys(inputData).map((key) => (
          <div key={key} style={{ marginBottom: "10px" }}>
            <input
              type="text"
              name={key}
              value={inputData[key]}
              placeholder={`${key} Level`}
              onChange={handleChange}
              required
            />
          </div>
        ))}
        <button type="submit">Predict</button>
      </form>

      {prediction && (
        <h3 style={{ marginTop: "20px" }}>Predicted PM2.5: {prediction}</h3>
      )}
    </div>
  );
}

export default App;
