import { BrowserRouter, Routes, Route } from 'react-router-dom'
import TopNavBar from './components/TopNavBar'
import Footer from './components/Footer'
import Landing from './pages/Landing'
import Predict from './pages/Predict'
import BatchUpload from './pages/BatchUpload'
import TrainConfig from './pages/TrainConfig'
import TrainMonitor from './pages/TrainMonitor'
import Results from './pages/Results'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-background text-on-surface">
        <TopNavBar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/predict/batch" element={<BatchUpload />} />
            <Route path="/train/configure" element={<TrainConfig />} />
            <Route path="/train/monitor" element={<TrainMonitor />} />
            <Route path="/results" element={<Results />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  )
}
