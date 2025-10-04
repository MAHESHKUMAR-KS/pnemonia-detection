import { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(
        `${import.meta.env.VITE_API_URL}/predict`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during prediction');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <div className="bg-white shadow-xl rounded-lg overflow-hidden">
          {/* Header */}
          <div className="px-6 py-4 bg-blue-600">
            <h1 className="text-2xl font-bold text-white text-center">
              Pneumonia Detection System
            </h1>
          </div>

          {/* Main Content */}
          <div className="p-6">
            {/* File Upload */}
            <div className="mb-6">
              <label className="block text-gray-700 text-sm font-bold mb-2">
                Upload Chest X-ray Image
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
            </div>

            {/* Image Preview */}
            {preview && (
              <div className="mb-6">
                <img
                  src={preview}
                  alt="Preview"
                  className="max-w-full h-auto rounded-lg shadow-md"
                />
              </div>
            )}

            {/* Predict Button */}
            <button
              onClick={handlePredict}
              disabled={isLoading || !selectedFile}
              className={`w-full py-2 px-4 rounded-md text-white font-semibold ${
                isLoading || !selectedFile
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {isLoading ? 'Analyzing...' : 'Predict'}
            </button>

            {/* Error Message */}
            {error && (
              <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-md">
                {error}
              </div>
            )}

            {/* Prediction Results */}
            {prediction && (
              <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                <h3 className="text-lg font-semibold mb-2">Results:</h3>
                <div className="space-y-2">
                  <p className="text-gray-700">
                    Prediction:{' '}
                    <span
                      className={`font-bold ${
                        prediction.prediction === 'Pneumonia'
                          ? 'text-red-600'
                          : 'text-green-600'
                      }`}
                    >
                      {prediction.prediction}
                    </span>
                  </p>
                  <p className="text-gray-700">
                    Confidence:{' '}
                    <span className="font-bold">
                      {(prediction.confidence * 100).toFixed(2)}%
                    </span>
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
