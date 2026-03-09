import { useState } from 'react';
import DocumentInputPage from './Pages/DocumentInputPage';
import ResumePage from './Pages/ResumePage';

function App() {
  const [dark, setDark] = useState(true);
  const [currentPage, setCurrentPage] = useState('document'); // 'document' or 'resume'

  return (
    <>
      {currentPage === 'document' && (
        <DocumentInputPage 
          onNext={() => setCurrentPage('resume')} 
          dark={dark}
        />
      )}
      {currentPage === 'resume' && (
        <ResumePage 
          onBack={() => setCurrentPage('document')} 
          dark={dark}
        />
      )}
    </>
  );
}

export default App;
