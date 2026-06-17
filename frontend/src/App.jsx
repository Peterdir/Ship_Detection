import React, { useState } from 'react';
import Header from './components/Header';
import UploadForm from './components/UploadForm';
import ResultCard from './components/ResultCard';
import { predictImage } from './services/api';
import { Loader2 } from 'lucide-react';

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleUpload = async (file, previewUrl) => {
    setLoading(true);
    setResult(null);
    setOriginalImage(previewUrl);
    setUploadProgress(0);

    try {
      const response = await predictImage(file, (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(percentCompleted);
      });
      setResult(response);
    } catch (error) {
      console.error('Failed to predict image:', error);
      alert('Error connecting to the backend. Is FastAPI running on port 8000?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      <Header />
      
      <main className="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <UploadForm onUpload={handleUpload} />
        
        {loading && (
          <div className="w-full max-w-2xl mx-auto mt-8 flex flex-col items-center p-8 bg-white dark:bg-gray-800 rounded-2xl shadow-sm">
            <Loader2 className="w-10 h-10 text-blue-600 animate-spin mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">
              Processing Image...
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
              Faster R-CNN model is analyzing the image for ships.
            </p>
            
            {uploadProgress < 100 && (
              <div className="w-full max-w-xs mt-4">
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
                  <span>Uploading</span>
                  <span>{uploadProgress}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        )}

        {!loading && result && (
          <ResultCard result={result} originalImage={originalImage} />
        )}
      </main>
      
      <footer className="bg-white dark:bg-gray-800 py-6 border-t border-gray-200 dark:border-gray-700 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>Ship Detection System using PyTorch, FastAPI, and React</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
