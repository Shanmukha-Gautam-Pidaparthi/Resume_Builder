import { useState } from 'react';
import DocumentInputPage from './Pages/DocumentInputPage';
import ResumePage from './Pages/ResumePage';

function App() {
  const [currentPage, setCurrentPage] = useState('document'); // 'document' or 'resume'
  const [resumeData, setResumeData] = useState(null);
  const [dark, setDark] = useState(false);

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
          onToggleDark={() => setDark(!dark)}
        />
      )}
      {currentPage === 'resume' && (
        <ResumePage 
          onBack={() => setCurrentPage('document')} 
          dark={dark}
          onToggleDark={() => setDark(!dark)}
          resumeData={resumeData}
        />
      )}
    </>
  );
}

export default App;
