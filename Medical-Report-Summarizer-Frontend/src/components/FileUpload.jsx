import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { FaUpload, FaFilePdf, FaSpinner, FaTimes } from 'react-icons/fa';

const FileUpload = ({ mode, onFileUpload, isProcessing }) => {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      setSelectedFile(files[0]);
    }
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      setSelectedFile(files[0]);
    }
  };

  const handleUpload = () => {
    if (selectedFile) {
      onFileUpload(selectedFile);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    fileInputRef.current.value = null;
  };

  const getButtonClass = () => {
    if (isProcessing) return 'bg-gray-300 cursor-not-allowed';
    if (mode === 'doctor') return 'bg-blue-600 hover:bg-blue-700 text-white';
    return 'bg-blue-600 hover:bg-blue-700 text-white';
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
      className={`p-6 rounded-lg shadow-lg border-2 transition-all duration-500 ${
        mode === 'patient'
          ? 'bg-white border-blue-100'
          : 'bg-slate-50 border-slate-200'
      }`}
    >
      <h2
        className={`text-2xl font-semibold mb-4 transition-colors duration-500 ${
          mode === 'patient' ? 'text-gray-800' : 'text-slate-800'
        }`}
      >
        {mode === 'patient' ? 'Upload Medical Report' : 'Upload Research Paper'}
      </h2>

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
          dragOver
            ? mode === 'patient'
              ? 'border-blue-400 bg-blue-50'
              : 'border-teal-400 bg-teal-50'
            : mode === 'patient'
            ? 'border-blue-200 bg-blue-50/50 hover:bg-blue-50'
            : 'border-slate-300 bg-slate-100/50 hover:bg-slate-100'
        }`}
      >
        <FaUpload
          className={`mx-auto text-4xl mb-4 transition-colors duration-300 ${
            mode === 'patient' ? 'text-blue-400' : 'text-teal-400'
          }`}
        />

        <p
          className={`text-lg mb-4 transition-colors duration-500 ${
            mode === 'patient' ? 'text-gray-700' : 'text-slate-700'
          }`}
        >
          Drag and drop your PDF here, or
        </p>

        <button
          onClick={() => fileInputRef.current?.click()}
          className={`px-6 py-2 rounded-lg transition-all duration-300 font-medium ${
            mode === 'patient' 
              ? 'bg-blue-500 hover:bg-blue-600 text-white'
              : 'bg-blue-500 hover:bg-blue-600 text-white'
          }`}
        >
          Choose File
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileSelect}
          className="hidden"
        />
      </div>

      {selectedFile && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-4 bg-gray-50 rounded-lg"
        >
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <FaFilePdf className="text-red-500" />
              <span className="text-sm font-medium text-gray-700">
                {selectedFile.name}
              </span>
            </div>
            <button
              onClick={handleRemoveFile}
              className="text-red-500 hover:text-red-700 transition-colors"
              title="Remove file"
            >
              <FaTimes />
            </button>
          </div>

          <button
            onClick={handleUpload}
            disabled={isProcessing}
            className={`w-full py-3 px-4 rounded-lg font-medium transition-all duration-300 ${getButtonClass()}`}
          >
            {isProcessing ? (
              <div className="flex items-center justify-center space-x-2">
                <FaSpinner className="animate-spin" />
                <span>Processing...</span>
              </div>
            ) : (
              'Create Summary'
            )}
          </button>
        </motion.div>
      )}
    </motion.div>
  );
};

export default FileUpload;
