import React from 'react';
import { Download, AlertCircle, CheckCircle2, Info } from 'lucide-react';
import { getResultImageUrl } from '../services/api';

const ResultCard = ({ result, originalImage }) => {
  if (!result) return null;

  const { has_ship, ship_count, average_confidence, result_image, ships } = result;
  const imageUrl = getResultImageUrl(result_image);

  return (
    <div className="w-full max-w-5xl mx-auto mt-8 bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden transition-all duration-300">
      <div className="p-6 sm:p-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 border-b border-gray-200 dark:border-gray-700 pb-4">
          Detection Results
        </h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Status Panel */}
          <div className="space-y-6">
            <div className={`p-6 rounded-xl border ${has_ship ? 'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800' : 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800'}`}>
              <div className="flex items-center mb-2">
                {has_ship ? (
                  <CheckCircle2 className="w-8 h-8 text-green-600 dark:text-green-400 mr-3" />
                ) : (
                  <AlertCircle className="w-8 h-8 text-red-600 dark:text-red-400 mr-3" />
                )}
                <h3 className={`text-xl font-bold ${has_ship ? 'text-green-800 dark:text-green-300' : 'text-red-800 dark:text-red-300'}`}>
                  {has_ship ? 'Có tàu' : 'Không phát hiện tàu'}
                </h3>
              </div>
              <p className={`text-sm ${has_ship ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                Mô hình Faster R-CNN {has_ship ? 'đã xác định được mục tiêu.' : 'không tìm thấy mục tiêu nào thỏa mãn.'}
              </p>
            </div>

            {has_ship && (
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 dark:bg-gray-700/50 p-5 rounded-xl border border-gray-100 dark:border-gray-600 text-center">
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-1 font-medium">Số lượng tàu</p>
                  <p className="text-3xl font-bold text-blue-600 dark:text-blue-400">{ship_count}</p>
                </div>
                <div className="bg-gray-50 dark:bg-gray-700/50 p-5 rounded-xl border border-gray-100 dark:border-gray-600 text-center">
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-1 font-medium">Độ tin cậy trung bình</p>
                  <p className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                    {(average_confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            )}

            {has_ship && ships && ships.length > 0 && (
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
                <div className="px-4 py-3 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600 flex items-center">
                  <Info className="w-4 h-4 text-gray-500 mr-2" />
                  <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Chi tiết phát hiện</h4>
                </div>
                <div className="max-h-48 overflow-y-auto">
                  <ul className="divide-y divide-gray-100 dark:divide-gray-700">
                    {ships.map((ship, index) => (
                      <li key={index} className="px-4 py-3 flex justify-between items-center hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Tàu #{index + 1}</span>
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300">
                          Confidence: {(ship.confidence * 100).toFixed(1)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>

          {/* Image Display */}
          <div className="flex flex-col space-y-4">
            <div className="relative rounded-xl overflow-hidden border-2 border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-900 group">
              <img 
                src={imageUrl || originalImage} 
                alt="Detection Result" 
                className="w-full h-auto max-h-[500px] object-contain"
              />
              {imageUrl && (
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                  <a 
                    href={imageUrl} 
                    download="detection_result.jpg"
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center px-4 py-2 bg-white text-gray-900 rounded-lg font-medium shadow-lg hover:bg-gray-100 transition-colors"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Tải ảnh kết quả
                  </a>
                </div>
              )}
            </div>
            <p className="text-center text-sm text-gray-500 dark:text-gray-400">
              {imageUrl ? 'Ảnh sau nhận diện (có Bounding Box)' : 'Ảnh gốc'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultCard;
