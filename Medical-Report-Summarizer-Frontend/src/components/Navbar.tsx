
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
      className="bg-white shadow-lg border-b-2 border-blue-100"
    >
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <div className="flex items-center space-x-3">
            <MdLocalHospital className="text-3xl text-blue-600" />
            <span className="text-xl font-bold text-gray-800">
              MedSummary
            </span>
          </div>

          {/* Mode Toggle */}
          <div className="flex items-center space-x-4">
            <div className="hidden sm:flex items-center space-x-2 text-sm text-gray-600">
              <FaUser className={mode === 'patient' ? 'text-blue-600' : 'text-gray-400'} />
              <span className={mode === 'patient' ? 'text-blue-600 font-medium' : ''}>
                Patient
              </span>
            </div>
            
            <ToggleModeSwitch 
              mode={mode} 
              onModeChange={onModeChange} 
            />
            
            <div className="hidden sm:flex items-center space-x-2 text-sm text-gray-600">
              <span className={mode === 'doctor' ? 'text-blue-600 font-medium' : ''}>
                Doctor
              </span>
              <FaUserMd className={mode === 'doctor' ? 'text-blue-600' : 'text-gray-400'} />
            </div>
          </div>
        </div>
      </div>
    </motion.nav>
  );
};

export default Navbar;
