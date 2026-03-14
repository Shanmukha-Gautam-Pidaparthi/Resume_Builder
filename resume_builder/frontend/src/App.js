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

  const onToggleDark = () => setDark(p => !p);

  return (
    <>
      {currentPage === 'document' && (
        <DocumentInputPage 
          onNext={handleResumeGenerated} 
          dark={dark}
          onToggleDark={onToggleDark}
        />
      )}
      {currentPage === 'resume' && (
        <ResumePage 
          onBack={() => setCurrentPage('document')} 
          dark={dark}
          onToggleDark={onToggleDark}
          resumeData={resumeData}
        />
      )}
    </>
  );
}

export default App;
