import axios from 'axios';

// Ensure this matches your FastAPI backend URL and port
const API_BASE_URL = 'http://localhost:8000';

export const predictImage = async (imageFile, onUploadProgress) => {
  const formData = new FormData();
  formData.append('image', imageFile);

  try {
    const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress,
    });
    return response.data;
  } catch (error) {
    console.error('Error uploading image:', error);
    throw error;
  }
};

export const getResultImageUrl = (imagePath) => {
  if (!imagePath) return null;
  return `${API_BASE_URL}${imagePath}`;
};
