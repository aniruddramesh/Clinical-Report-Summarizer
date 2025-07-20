
import { motion } from 'framer-motion';
import { FaUserMd, FaUser } from 'react-icons/fa';
import { MdLocalHospital } from 'react-icons/md';
import ToggleModeSwitch from './ToggleModeSwitch';

const Navbar = ({ mode, onModeChange }) => {
  return (
    <motion.nav 
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className={`shadow-lg border-b-2 transition-all duration-700 ${
        mode === 'patient' 
          ? 'bg-white border-blue-100' 
          : 'bg-slate-50 border-slate-200'
      }`}
    >
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <div className="flex items-center space-x-3">
            <MdLocalHospital className={`text-3xl transition-colors duration-500 ${
              mode === 'patient' ? 'text-blue-600' : 'text-teal-600'
            }`} />
            <span className={`text-xl font-bold transition-colors duration-500 ${
              mode === 'patient' ? 'text-gray-800' : 'text-slate-800'
            }`}>
              MedSummary
            </span>
          </div>

          {/* Mode Toggle */}
          <div className="flex items-center space-x-4">
            <div className={`hidden sm:flex items-center space-x-2 text-sm transition-colors duration-500 ${
              mode === 'patient' ? 'text-gray-600' : 'text-slate-600'
            }`}>
              <FaUser className={`transition-colors duration-300 ${
                mode === 'patient' ? 'text-blue-600' : 'text-gray-400'
              }`} />
              <span className={`transition-colors duration-300 ${
                mode === 'patient' ? 'text-blue-600 font-medium' : ''
              }`}>
                Patient
              </span>
            </div>
            
            <ToggleModeSwitch 
              mode={mode} 
              onModeChange={onModeChange} 
            />
            
            <div className={`hidden sm:flex items-center space-x-2 text-sm transition-colors duration-500 ${
              mode === 'patient' ? 'text-gray-600' : 'text-slate-600'
            }`}>
              <span className={`transition-colors duration-300 ${
                mode === 'doctor' ? 'text-teal-600 font-medium' : ''
              }`}>
                Doctor
              </span>
              <FaUserMd className={`transition-colors duration-300 ${
                mode === 'doctor' ? 'text-teal-600' : 'text-gray-400'
              }`} />
            </div>
          </div>
        </div>
      </div>
    </motion.nav>
  );
};

export default Navbar;
