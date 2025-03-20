import axios from "axios";

export const predictAirPollution = async (data) => {
    try {
        const response = await axios.post("http://127.0.0.1:5000/predict", data);
        return response.data;
    } catch (error) {
        console.error("Error predicting air pollution:", error);
        return null;
    }
};
