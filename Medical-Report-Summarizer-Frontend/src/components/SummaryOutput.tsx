
import { motion } from 'framer-motion';
import { FaDownload, FaPrint, FaSpinner } from 'react-icons/fa';
import { MdSummarize } from 'react-icons/md';

const SummaryOutput = ({ summary, mode, isProcessing }) => {
  const handleDownload = () => {
    const element = document.createElement('a');
    const file = new Blob([summary], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `${mode}-summary.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handlePrint = () => {
    const printWindow = window.open('', '_blank');
    printWindow.document.write(`
      <html>
        <head>
          <title>${mode === 'patient' ? 'Medical Report' : 'Research Paper'} Summary</title>
          <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #2563eb; }
            pre { white-space: pre-wrap; }
          </style>
        </head>
        <body>
          <h1>${mode === 'patient' ? 'Medical Report' : 'Research Paper'} Summary</h1>
          <pre>${summary}</pre>
        </body>
      </html>
    `);
    printWindow.document.close();
    printWindow.print();
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 50 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.6, delay: 0.4 }}
      className="bg-white rounded-xl shadow-lg p-6 h-fit"
    >
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-gray-800">
          Summary Output
        </h2>
        {summary && (
          <div className="flex space-x-2">
            <motion.button
              onClick={handleDownload}
              className="p-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <FaDownload />
            </motion.button>
            <motion.button
              onClick={handlePrint}
              className="p-2 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <FaPrint />
            </motion.button>
          </div>
        )}
      </div>

      <div className="min-h-[400px] border-2 border-dashed border-gray-200 rounded-lg p-6">
        {isProcessing ? (
          <div className="flex flex-col items-center justify-center h-full space-y-4">
            <FaSpinner className="text-4xl text-blue-500 animate-spin" />
            <p className="text-gray-600">Analyzing your document...</p>
            <div className="flex space-x-1">
              <motion.div
                className="w-2 h-2 bg-blue-500 rounded-full"
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
              />
              <motion.div
                className="w-2 h-2 bg-blue-500 rounded-full"
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
              />
              <motion.div
                className="w-2 h-2 bg-blue-500 rounded-full"
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
              />
            </div>
          </div>
        ) : summary ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="prose max-w-none"
          >
            <div className="whitespace-pre-wrap text-gray-700 leading-relaxed">
              {summary}
            </div>
          </motion.div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <MdSummarize className="text-6xl mb-4" />
            <p className="text-lg font-medium">
              {mode === 'patient' 
                ? 'Your simplified summary will appear here'
                : 'Your clinical summary will appear here'
              }
            </p>
            <p className="text-sm mt-2">
              Upload a PDF to get started
            </p>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default SummaryOutput;
