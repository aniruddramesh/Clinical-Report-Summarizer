
import { motion } from 'framer-motion';
import { FaUser, FaUserMd } from 'react-icons/fa';

const ToggleModeSwitch = ({ mode, onModeChange }) => {
  return (
    <div className="relative">
      <motion.button
        onClick={() => onModeChange(mode === 'patient' ? 'doctor' : 'patient')}
        className={`relative w-16 h-8 rounded-full p-1 focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-300 ${
          mode === 'patient' 
            ? 'bg-blue-100 focus:ring-blue-500' 
            : 'bg-teal-100 focus:ring-teal-500'
        }`}
        whileTap={{ scale: 0.95 }}
      >
        <motion.div
          className="flex items-center justify-center w-6 h-6 bg-white rounded-full shadow-md"
          animate={{
            x: mode === 'patient' ? 0 : 32,
            backgroundColor: mode === 'patient' ? '#EBF8FF' : '#F0FDFA'
          }}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
        >
          {mode === 'patient' ? (
            <FaUser className="text-xs text-blue-600" />
          ) : (
            <FaUserMd className="text-xs text-teal-600" />
          )}
        </motion.div>
      </motion.button>
      
      {/* Mobile labels */}
      <div className="sm:hidden absolute top-10 left-1/2 transform -translate-x-1/2 text-xs text-gray-600 whitespace-nowrap">
        {mode === 'patient' ? 'Patient Mode' : 'Doctor Mode'}
      </div>
    </div>
  );
};

export default ToggleModeSwitch;
