import { useState } from 'react';
import DocumentInputPage from './Pages/DocumentInputPage';
import ResumePage from './Pages/ResumePage';

function App() {
  const [dark, setDark] = useState(true);
  const [currentPage, setCurrentPage] = useState('document'); // 'document' or 'resume'
  const [resumeData, setResumeData] = useState(null);

  const handleResumeGenerated = (data) => {
    setResumeData(data);
    setCurrentPage('resume');
  };

  return (
    <>
      {currentPage === 'document' && (
        <DocumentInputPage 
          onNext={handleResumeGenerated} 
          dark={dark}
        />
      )}
      {currentPage === 'resume' && (
        <ResumePage 
          onBack={() => setCurrentPage('document')} 
          dark={dark}
          resumeData={resumeData}
        />
      )}
    </>
  );
}

export default App;
