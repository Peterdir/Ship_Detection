import React from 'react';
import { Anchor } from 'lucide-react';

const Header = () => {
  return (
    <header className="bg-white dark:bg-gray-800 shadow-sm transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Anchor className="h-8 w-8 text-blue-600 dark:text-blue-400" />
            <h1 className="ml-3 text-xl font-bold text-gray-900 dark:text-white">
              Ship Detection System using Faster R-CNN
            </h1>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
