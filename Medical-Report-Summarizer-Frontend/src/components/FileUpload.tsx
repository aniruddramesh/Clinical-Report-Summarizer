
import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { FaUpload, FaFilePdf, FaSpinner } from 'react-icons/fa';
import { MdCloudUpload } from 'react-icons/md';

const FileUpload = ({ mode, onFileUpload, isProcessing }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (file) => {
    setSelectedFile(file);
  };

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleSubmit = () => {
    if (selectedFile) {
      onFileUpload(selectedFile);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -50 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.6, delay: 0.2 }}
      className="bg-white rounded-xl shadow-lg p-6"
    >
      <h2 className="text-2xl font-semibold text-gray-800 mb-6">
        {mode === 'patient' ? 'Upload Medical Report' : 'Upload Research Paper'}
      </h2>

      {/* Drag and Drop Area */}
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
          dragActive 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleInputChange}
          className="hidden"
        />

        <motion.div
          animate={{ scale: dragActive ? 1.05 : 1 }}
          transition={{ duration: 0.2 }}
        >
          <MdCloudUpload className="mx-auto text-4xl text-gray-400 mb-4" />
          <p className="text-lg font-medium text-gray-700 mb-2">
            {dragActive ? 'Drop your PDF here' : 'Click to upload or drag and drop'}
          </p>
          <p className="text-sm text-gray-500">
            PDF files only (max 10MB)
          </p>
        </motion.div>
      </div>

      {/* Selected File Display */}
      {selectedFile && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200"
        >
          <div className="flex items-center space-x-3">
            <FaFilePdf className="text-red-500 text-xl" />
            <div>
              <p className="font-medium text-gray-800">{selectedFile.name}</p>
              <p className="text-sm text-gray-600">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Submit Button */}
      <motion.button
        onClick={handleSubmit}
        disabled={!selectedFile || isProcessing}
        className={`w-full mt-6 py-3 px-6 rounded-lg font-medium transition-all duration-300 ${
          selectedFile && !isProcessing
            ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl'
            : 'bg-gray-300 text-gray-500 cursor-not-allowed'
        }`}
        whileHover={selectedFile && !isProcessing ? { scale: 1.02 } : {}}
        whileTap={selectedFile && !isProcessing ? { scale: 0.98 } : {}}
      >
        {isProcessing ? (
          <div className="flex items-center justify-center space-x-2">
            <FaSpinner className="animate-spin" />
            <span>Processing...</span>
          </div>
        ) : (
          <div className="flex items-center justify-center space-x-2">
            <FaUpload />
            <span>Create Summary</span>
          </div>
        )}
      </motion.button>
    </motion.div>
  );
};

export default FileUpload;
