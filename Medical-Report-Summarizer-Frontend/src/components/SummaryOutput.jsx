
import { motion } from 'framer-motion';
import { FaDownload, FaCopy, FaSpinner } from 'react-icons/fa';

const SummaryOutput = ({ summary, mode, isProcessing }) => {
  const handleCopy = () => {
    navigator.clipboard.writeText(summary);
  };

  const handleDownload = () => {
    const blob = new Blob([summary], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${mode}-summary.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className={`p-6 rounded-lg shadow-lg border-2 transition-all duration-500 ${
        mode === 'patient' 
          ? 'bg-white border-blue-100' 
          : 'bg-slate-50 border-slate-200'
      }`}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className={`text-2xl font-semibold transition-colors duration-500 ${
          mode === 'patient' ? 'text-gray-800' : 'text-slate-800'
        }`}>
          {mode === 'patient' ? 'Your Summary' : 'Research Summary'}
        </h2>
        
        {summary && (
          <div className="flex space-x-2">
            <button
              onClick={handleCopy}
              className={`p-2 rounded-lg transition-all duration-300 ${
                mode === 'patient'
                  ? 'bg-blue-100 hover:bg-blue-200 text-blue-600'
                  : 'bg-teal-100 hover:bg-teal-200 text-teal-600'
              }`}
              title="Copy to clipboard"
            >
              <FaCopy />
            </button>
            <button
              onClick={handleDownload}
              className={`p-2 rounded-lg transition-all duration-300 ${
                mode === 'patient'
                  ? 'bg-blue-100 hover:bg-blue-200 text-blue-600'
                  : 'bg-teal-100 hover:bg-teal-200 text-teal-600'
              }`}
              title="Download summary"
            >
              <FaDownload />
            </button>
          </div>
        )}
      </div>
      
      <div className={`min-h-[300px] rounded-lg p-4 transition-all duration-500 ${
        mode === 'patient' 
          ? 'bg-blue-50/50 border border-blue-100' 
          : 'bg-slate-100/50 border border-slate-200'
      }`}>
        {isProcessing ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <FaSpinner className={`animate-spin text-4xl mb-4 mx-auto ${
                mode === 'patient' ? 'text-blue-500' : 'text-teal-500'
              }`} />
              <p className={`text-lg transition-colors duration-500 ${
                mode === 'patient' ? 'text-gray-600' : 'text-slate-600'
              }`}>
                Analyzing your document...
              </p>
            </div>
          </div>
        ) : summary ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className={`prose max-w-none transition-colors duration-500 ${
              mode === 'patient' ? 'prose-blue' : 'prose-slate'
            }`}
          >
            <div className={`whitespace-pre-wrap text-sm leading-relaxed ${
              mode === 'patient' ? 'text-gray-700' : 'text-slate-700'
            }`}>
              {summary}
            </div>
          </motion.div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className={`text-lg transition-colors duration-500 ${
              mode === 'patient' ? 'text-gray-500' : 'text-slate-500'
            }`}>
              {mode === 'patient' 
                ? 'Your simplified summary will appear here'
                : 'Research analysis will appear here'
              }
            </p>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default SummaryOutput;
