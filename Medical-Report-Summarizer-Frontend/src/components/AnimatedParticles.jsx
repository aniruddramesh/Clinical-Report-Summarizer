
import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

const AnimatedParticles = () => {
  const [particles, setParticles] = useState([]);

  useEffect(() => {
    // Generate random particles
    const newParticles = Array.from({ length: 30 }, (_, i) => ({
      id: i,
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      size: Math.random() * 8 + 4,
      color: ['#3B82F6', '#60A5FA', '#93C5FD', '#DBEAFE'][Math.floor(Math.random() * 4)],
      delay: Math.random() * 0.5,
    }));
    setParticles(newParticles);
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none z-50">
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute rounded-full"
          style={{
            width: particle.size,
            height: particle.size,
            backgroundColor: particle.color,
            left: particle.x,
            top: particle.y,
          }}
          initial={{
            scale: 0,
            opacity: 1,
            y: particle.y,
          }}
          animate={{
            scale: [0, 1, 0],
            opacity: [0, 1, 0],
            y: particle.y - 200,
          }}
          transition={{
            duration: 2,
            delay: particle.delay,
            ease: "easeOut",
          }}
        />
      ))}
      
      {/* Celebration burst effect */}
      <motion.div
        className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"
        initial={{ scale: 0 }}
        animate={{ scale: [0, 1.5, 0] }}
        transition={{ duration: 1, ease: "easeOut" }}
      >
        <div className="w-32 h-32 border-4 border-blue-400 rounded-full opacity-30" />
      </motion.div>
      
      <motion.div
        className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"
        initial={{ scale: 0 }}
        animate={{ scale: [0, 2, 0] }}
        transition={{ duration: 1.5, delay: 0.2, ease: "easeOut" }}
      >
        <div className="w-24 h-24 border-2 border-blue-300 rounded-full opacity-20" />
      </motion.div>
    </div>
  );
};

export default AnimatedParticles;
