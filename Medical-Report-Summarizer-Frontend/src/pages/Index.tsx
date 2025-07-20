import { useState } from 'react';
import Navbar from '../components/Navbar';
import FileUpload from '../components/FileUpload';
import SummaryOutput from '../components/SummaryOutput';
import AnimatedParticles from '../components/AnimatedParticles';
import { useToast } from '@/hooks/use-toast';

const Index = () => {
  const [mode, setMode] = useState<'patient' | 'doctor'>('patient'); // ✅ Accept both modes
  const [summary, setSummary] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [showParticles, setShowParticles] = useState(false);
  const { toast } = useToast();

  const handleFileUpload = async (file: File) => {
    if (!file) {
      toast({
        title: "Error",
        description: "Please select a file to upload.",
        variant: "destructive",
      });
      return;
    }

    if (file.type !== 'application/pdf') {
      toast({
        title: "Invalid File Type",
        description: "Please upload a PDF file only.",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setSummary('');

    try {
      const formData = new FormData();
      formData.append("file", file);

      // ✅ Choose backend URL based on mode
      const BACKEND_URL =
        mode === 'patient'
          ? "http://localhost:5000/summarize-patient"
          : "http://localhost:5000/summarize-doctor";

      const response = await fetch(BACKEND_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setShowParticles(true);
        setSummary(data.summary);

        toast({
          title: "Success!",
          description: "Your summary has been generated successfully.",
        });

        setTimeout(() => setShowParticles(false), 3000);
      } else {
        toast({
          title: "Error",
          description: data.error || "Something went wrong while summarizing.",
          variant: "destructive",
        });
      }
    } catch (error: any) {
      toast({
        title: "Network Error",
        description: error.message || "Unable to connect to the server.",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50">
      <Navbar mode={mode} onModeChange={setMode} />

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
              Medical Report Summarizer
            </h1>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              {mode === 'patient'
                ? "Get clear, easy-to-understand summaries of your medical reports"
                : "Generate precise, clinical summaries of research papers and medical documents"}
            </p>
          </div>

          {/* Main Content */}
          <div className="grid lg:grid-cols-2 gap-8 items-start">
            <FileUpload
              mode={mode}
              onFileUpload={handleFileUpload}
              isProcessing={isProcessing}
            />

            <SummaryOutput
              summary={summary}
              mode={mode}
              isProcessing={isProcessing}
            />
          </div>
        </div>
      </main>

      {showParticles && <AnimatedParticles />}
    </div>
  );
};

export default Index;
