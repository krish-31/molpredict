import { useState, useRef } from 'react'
import { Link } from 'react-router-dom'
import { apiClient } from '../api/client'

const ALL_TASKS = [
  { key: 'NR-AR', label: 'NR-AR', desc: 'Nuclear Receptor — Androgen Receptor' },
  { key: 'NR-AR-LBD', label: 'NR-AR-LBD', desc: 'Androgen Receptor Ligand Binding Domain' },
  { key: 'NR-AhR', label: 'NR-AhR', desc: 'Aryl Hydrocarbon Receptor' },
  { key: 'NR-Aromatase', label: 'NR-Aromatase', desc: 'Aromatase Enzyme Inhibition' },
  { key: 'NR-ER', label: 'NR-ER', desc: 'Estrogen Receptor Alpha' },
  { key: 'NR-ER-LBD', label: 'NR-ER-LBD', desc: 'Estrogen Receptor Ligand Binding Domain' },
  { key: 'NR-PPAR-gamma', label: 'NR-PPAR-γ', desc: 'Peroxisome Proliferator-Activated Receptor Gamma' },
  { key: 'SR-ARE', label: 'SR-ARE', desc: 'Antioxidant Response Element' },
  { key: 'SR-ATAD5', label: 'SR-ATAD5', desc: 'ATPase Family AAA Domain Containing 5' },
  { key: 'SR-HSE', label: 'SR-HSE', desc: 'Heat Shock Element Pathway' },
  { key: 'SR-MMP', label: 'SR-MMP', desc: 'Mitochondrial Membrane Potential' },
  { key: 'SR-p53', label: 'SR-p53', desc: 'p53 Tumor Suppressor Pathway' }
]

function cellColor(v, threshold = 0.20) {
  if (v === undefined || v === null) return 'text-outline'
  if (v >= threshold) return 'text-red-400 font-bold'
  if (v >= threshold * 0.6) return 'text-yellow-400'
  return 'text-primary'
}

function getRiskBadge(risk) {
  if (!risk) return <span className="px-2 py-0.5 rounded-full text-[10px] font-label-caps bg-outline/10 text-outline border border-outline/20">LOW</span>
  const r = risk.toUpperCase()
  if (r === 'HIGH') return <span className="px-2 py-0.5 rounded-full text-[10px] font-label-caps bg-red-500/10 text-red-400 border border-red-500/20">HIGH</span>
  if (r === 'MODERATE') return <span className="px-2 py-0.5 rounded-full text-[10px] font-label-caps bg-yellow-500/10 text-yellow-400 border border-yellow-500/20">MODERATE</span>
  return <span className="px-2 py-0.5 rounded-full text-[10px] font-label-caps bg-primary/10 text-primary border border-primary/20">LOW</span>
}

export default function BatchUpload() {
  const [dragOver, setDragOver] = useState(false)
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [processing, setProcessing] = useState(false)
  const [done, setDone] = useState(false)
  const [results, setResults] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [error, setError] = useState(null)
  const [totalCount, setTotalCount] = useState(0)
  const [threshold, setThreshold] = useState(0.20)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const fileInputRef = useRef()

  const [searchQuery, setSearchQuery] = useState('')
  const [riskFilter, setRiskFilter] = useState('ALL')
  const [flaggedFilter, setFlaggedFilter] = useState('ALL')
  const [sortConfig, setSortConfig] = useState({ key: 'compound_name', direction: 'asc' })
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize, setPageSize] = useState(25)

  const handleSort = (key) => {
    let direction = 'asc'
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc'
    }
    setSortConfig({ key, direction })
    setCurrentPage(1)
  }

  // Filter and Search
  const filteredResults = (results || []).filter(row => {
    // 1. Search Query
    const nameMatch = (row.compound_name || '').toLowerCase().includes(searchQuery.toLowerCase())
    const smilesMatch = (row.smiles || '').toLowerCase().includes(searchQuery.toLowerCase())
    if (!nameMatch && !smilesMatch) return false

    // 2. Risk Filter
    if (riskFilter !== 'ALL') {
      if ((row.risk_level || 'LOW').toUpperCase() !== riskFilter) return false
    }

    // 3. Flagged Filter
    if (flaggedFilter !== 'ALL') {
      const flagged = parseInt(row.flagged_endpoints || 0, 10)
      if (flaggedFilter === 'SAFE' && flagged > 0) return false
      if (flaggedFilter === 'TOXIC' && flagged === 0) return false
      if (flaggedFilter === 'GE1' && flagged < 1) return false
      if (flaggedFilter === 'GE3' && flagged < 3) return false
    }

    return true
  })

  // Sorting
  const sortedResults = [...filteredResults].sort((a, b) => {
    let aVal = a[sortConfig.key]
    let bVal = b[sortConfig.key]

    // Special cases
    if (sortConfig.key === 'compound_name') {
      aVal = aVal || 'Unknown Compound'
      bVal = bVal || 'Unknown Compound'
      return sortConfig.direction === 'asc' 
        ? aVal.localeCompare(bVal) 
        : bVal.localeCompare(aVal)
    }
    
    if (sortConfig.key === 'risk_level') {
      const riskOrder = { 'LOW': 0, 'MODERATE': 1, 'HIGH': 2 }
      const aOrder = riskOrder[(a.risk_level || 'LOW').toUpperCase()] ?? 0
      const bOrder = riskOrder[(b.risk_level || 'LOW').toUpperCase()] ?? 0
      return sortConfig.direction === 'asc' ? aOrder - bOrder : bOrder - aOrder
    }

    // Default numeric sort
    aVal = parseFloat(aVal) || 0
    bVal = parseFloat(bVal) || 0
    return sortConfig.direction === 'asc' ? aVal - bVal : bVal - aVal
  })

  // Pagination
  const totalPages = Math.ceil(sortedResults.length / pageSize)
  const paginatedResults = sortedResults.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  )

  function handleFile(f) {
    if (!f) return
    setFile(f)
    setDone(false)
    setResults(null)
    setProgress(0)
    setProcessing(true)
    setError(null)

    apiClient.createBatchJob(f, threshold)
      .then(jobData => {
        const activeId = jobData.job_id
        setJobId(activeId)
        
        // Start polling
        const interval = setInterval(() => {
          apiClient.getBatchJobStatus(activeId)
            .then(statusData => {
              setProgress(statusData.progress_pct)
              if (statusData.total_molecules) {
                setTotalCount(statusData.total_molecules)
              }
              
              if (statusData.status === 'completed') {
                clearInterval(interval)
                // Fetch preview
                apiClient.getBatchJobPreview(activeId)
                  .then(preview => {
                    setResults(preview)
                    setProcessing(false)
                    setDone(true)
                  })
                  .catch(err => {
                    console.error("Preview error: ", err)
                    setError("Failed to fetch results preview.")
                    setProcessing(false)
                  })
              } else if (statusData.status === 'failed') {
                clearInterval(interval)
                setError("Batch prediction failed on the server.")
                setProcessing(false)
              }
            })
            .catch(err => {
              clearInterval(interval)
              console.error(err)
              setError("Error checking job status.")
              setProcessing(false)
            })
        }, 1500)
      })
      .catch(err => {
        console.error(err)
        setError("Failed to create batch prediction job. Make sure the backend is online.")
        setProcessing(false)
      })
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f && f.name.endsWith('.csv')) handleFile(f)
  }

  function downloadTemplate() {
    const csv = 'smiles\nCC(=O)Oc1ccccc1C(=O)O\nCn1cnc2c1c(=O)n(c(=O)n2C)C\nCC(=O)Nc1ccc(cc1)O\n'
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'template.csv'; a.click()
  }

  function downloadResults() {
    if (!jobId) return
    window.open(apiClient.getBatchJobDownloadUrl(jobId), '_blank')
  }


  return (
    <div className="pt-24 pb-16 min-h-screen">
      <div className="max-w-container-max mx-auto px-4 md:px-16">
        {/* Header */}
        <div className="mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <h1 className="font-headline-lg text-headline-lg-mobile md:text-headline-md text-primary mb-2">
              Batch Prediction
            </h1>
            <p className="font-body-md text-body-md text-on-surface-variant">
              Upload a CSV file with a <code className="font-code-sm text-primary bg-surface-container px-1 py-0.5 rounded">smiles</code> column to predict all 12 properties at scale.
            </p>
          </div>
          <Link
            to="/predict"
            className="flex items-center gap-2 font-label-caps text-label-caps text-on-surface-variant hover:text-primary transition-colors"
          >
            <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>arrow_back</span>
            Single Prediction
          </Link>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-gutter">
          {/* Upload zone */}
          <div className="lg:col-span-1 flex flex-col gap-6">
            {/* Drop zone */}
            <div
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`relative border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center gap-4 cursor-pointer transition-all duration-300 ${dragOver ? 'border-primary bg-primary/10' : 'border-outline-variant hover:border-primary/50 bg-surface-container'}`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                className="hidden"
                onChange={e => handleFile(e.target.files[0])}
              />
              <div className={`w-16 h-16 rounded-full border-2 flex items-center justify-center transition-all duration-300 ${dragOver ? 'border-primary bg-primary/20' : 'border-outline-variant'}`}>
                <span className={`material-symbols-outlined text-3xl ${dragOver ? 'text-primary' : 'text-outline'}`}>cloud_upload</span>
              </div>
              <div className="text-center">
                <p className="font-body-md text-body-md text-on-surface mb-1">
                  {dragOver ? 'Drop to upload' : 'Drag & drop CSV file'}
                </p>
                <p className="font-label-caps text-label-caps text-on-surface-variant">
                  or <span className="text-primary underline underline-offset-2">browse files</span>
                </p>
              </div>
              {file && (
                <div className="mt-2 px-4 py-2 bg-primary/10 border border-primary/20 rounded-lg w-full text-center animate-fade-in">
                  <span className="material-symbols-outlined text-primary" style={{ fontSize: '16px' }}>description</span>
                  <span className="font-code-sm text-code-sm text-primary ml-2">{file.name}</span>
                </div>
              )}
            </div>

            {/* Advanced options */}
            <div className="bg-surface-container border border-outline-variant rounded-xl overflow-hidden">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="w-full px-6 py-4 flex items-center justify-between text-on-surface-variant hover:text-on-surface transition-colors"
              >
                <span className="font-label-caps text-label-caps">ADVANCED OPTIONS</span>
                <span className="material-symbols-outlined transition-transform duration-200" style={{ transform: showAdvanced ? 'rotate(180deg)' : 'rotate(0)' }}>
                  expand_more
                </span>
              </button>
              {showAdvanced && (
                <div className="px-6 pb-6 animate-slide-up">
                  <div className="mb-4">
                    <label className="font-label-caps text-label-caps text-on-surface-variant block mb-2">
                      CLASSIFICATION THRESHOLD: {threshold.toFixed(2)}
                    </label>
                    <input
                      type="range" min="0.1" max="0.9" step="0.05"
                      value={threshold}
                      onChange={e => setThreshold(parseFloat(e.target.value))}
                      className="w-full accent-primary"
                    />
                    <div className="flex justify-between font-code-sm text-code-sm text-outline mt-1">
                      <span>0.1</span><span>0.5</span><span>0.9</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {error && (
              <div className="p-3 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg text-xs font-code-sm text-center">
                {error}
              </div>
            )}

            {/* Template download */}
            <button
              onClick={downloadTemplate}
              className="flex items-center justify-center gap-2 py-3 px-4 rounded-xl border border-outline-variant text-on-surface-variant font-label-caps text-label-caps hover:border-primary hover:text-primary transition-all duration-200"
            >
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>download</span>
              Download Template CSV
            </button>

            {/* Format hint */}
            <div className="bg-surface-container-low border border-outline-variant rounded-xl p-5">
              <div className="font-label-caps text-label-caps text-on-surface-variant mb-3">EXPECTED FORMAT</div>
              <div className="bg-surface-container-lowest rounded-lg p-3 font-code-sm text-code-sm text-primary">
                <div className="text-outline mb-1"># CSV with SMILES column</div>
                <div>smiles</div>
                <div>CC(=O)Oc1ccccc1C(=O)O</div>
                <div>Cn1cnc2c1c(=O)n2C</div>
                <div>c1ccccc1</div>
              </div>
            </div>

            {/* Status */}
            <div className="bg-surface-container border border-outline-variant rounded-xl p-5">
              <div className="font-label-caps text-label-caps text-on-surface-variant mb-3">PROCESSING STATUS</div>
              <div className="space-y-3">
                {[
                  { label: 'Upload', done: !!file },
                  { label: 'Featurize', done: progress > 30 },
                  { label: 'GIN Inference', done: progress > 70 },
                  { label: 'Aggregate', done: done },
                ].map(({ label, done: isDone }) => (
                  <div key={label} className="flex items-center gap-3">
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center border ${isDone ? 'border-primary bg-primary/10' : 'border-outline-variant'}`}>
                      {isDone && <span className="material-symbols-outlined text-primary" style={{ fontSize: '14px' }}>check</span>}
                    </div>
                    <span className={`font-label-caps text-label-caps ${isDone ? 'text-primary' : 'text-outline'}`}>{label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Progress + Results */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            {/* Progress */}
            {(processing || done) && file && (
              <div className="bg-surface-container border border-outline-variant rounded-xl p-6 animate-fade-in">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <div className="font-label-caps text-label-caps text-on-surface">{file.name}</div>
                    <div className="font-code-sm text-code-sm text-outline mt-1">
                      {done ? `${totalCount} molecules · Completed` : `Processing… ${Math.round(progress)}%`}
                    </div>
                  </div>
                  {done && (
                    <span className="flex items-center gap-1 px-2 py-1 bg-primary/10 border border-primary/20 rounded-full font-label-caps text-label-caps text-primary">
                      <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>check_circle</span>
                      Done
                    </span>
                  )}
                </div>

                {/* Progress bar */}
                <div className="w-full h-2 bg-surface-container-high rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>

                {!done && (
                  <div className="mt-2 font-code-sm text-code-sm text-outline">
                    ETA: ~{Math.max(0, Math.round((100 - progress) / 15))}s
                  </div>
                )}
              </div>
            )}

            {/* Preview table */}
            {results && (
              <div className="bg-surface-container border border-outline-variant rounded-xl p-6 animate-fade-in flex flex-col">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
                  <div>
                    <div className="font-label-caps text-label-caps text-on-surface-variant">SCREENING DASHBOARD</div>
                    <div className="font-code-sm text-code-sm text-outline mt-1">
                      Showing {filteredResults.length} of {results.length} molecules · 12 toxicity endpoints
                    </div>
                  </div>
                  <button
                    onClick={downloadResults}
                    className="flex items-center justify-center gap-2 px-4 py-2 bg-primary text-on-primary rounded-lg font-label-caps text-label-caps hover:opacity-90 transition-all duration-200 active:scale-95 whitespace-nowrap self-end md:self-auto"
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>download</span>
                    Download CSV
                  </button>
                </div>

                {/* Dashboard Controls */}
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 mb-6 bg-surface-container-low p-4 rounded-xl border border-outline-variant/40">
                  {/* Search */}
                  <div className="flex flex-col gap-1.5">
                    <label className="font-label-caps text-label-caps text-outline text-[10px]">Search Compound</label>
                    <div className="flex items-center gap-2 rounded-lg border border-outline-variant bg-surface-container-high px-3 py-2 text-xs">
                      <span className="material-symbols-outlined text-outline text-base">search</span>
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={e => { setSearchQuery(e.target.value); setCurrentPage(1) }}
                        placeholder="Name or SMILES..."
                        className="bg-transparent text-on-surface focus:outline-none w-full font-code-sm"
                      />
                    </div>
                  </div>

                  {/* Risk Level */}
                  <div className="flex flex-col gap-1.5">
                    <label className="font-label-caps text-label-caps text-outline text-[10px]">Risk level</label>
                    <select
                      value={riskFilter}
                      onChange={e => { setRiskFilter(e.target.value); setCurrentPage(1) }}
                      className="rounded-lg border border-outline-variant bg-surface-container-high px-3 py-2.5 text-xs text-on-surface focus:outline-none accent-primary cursor-pointer"
                    >
                      <option value="ALL">All Risk Levels</option>
                      <option value="LOW">Low Risk</option>
                      <option value="MODERATE">Moderate Risk</option>
                      <option value="HIGH">High Risk</option>
                    </select>
                  </div>

                  {/* Toxicity Status */}
                  <div className="flex flex-col gap-1.5">
                    <label className="font-label-caps text-label-caps text-outline text-[10px]">Toxicity status</label>
                    <select
                      value={flaggedFilter}
                      onChange={e => { setFlaggedFilter(e.target.value); setCurrentPage(1) }}
                      className="rounded-lg border border-outline-variant bg-surface-container-high px-3 py-2.5 text-xs text-on-surface focus:outline-none accent-primary cursor-pointer"
                    >
                      <option value="ALL">All Compounds</option>
                      <option value="SAFE">Safe Only (0 flagged)</option>
                      <option value="TOXIC">{"Flagged Only (>= 1 flagged)"}</option>
                      <option value="GE1">{">= 1 Flagged"}</option>
                      <option value="GE3">{">= 3 Flagged (High Toxin)"}</option>
                    </select>
                  </div>

                  {/* Page Size */}
                  <div className="flex flex-col gap-1.5">
                    <label className="font-label-caps text-label-caps text-outline text-[10px]">Molecules per page</label>
                    <select
                      value={pageSize}
                      onChange={e => { setPageSize(parseInt(e.target.value)); setCurrentPage(1) }}
                      className="rounded-lg border border-outline-variant bg-surface-container-high px-3 py-2.5 text-xs text-on-surface focus:outline-none accent-primary cursor-pointer"
                    >
                      <option value="10">10 rows</option>
                      <option value="25">25 rows</option>
                      <option value="50">50 rows</option>
                      <option value="100">100 rows</option>
                    </select>
                  </div>
                </div>

                {/* Table Container with Sticky Headers and Frozen Columns */}
                <div className="overflow-x-auto relative rounded-lg border border-outline-variant/60 max-h-[600px] overflow-y-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-outline-variant bg-surface-container-high">
                        <th 
                          onClick={() => handleSort('compound_name')}
                          className="sticky left-0 top-0 z-30 bg-surface-container-high border-b border-outline-variant text-left py-3 px-3 font-label-caps text-label-caps text-on-surface-variant hover:text-primary transition-colors cursor-pointer min-w-[160px] shadow-[3px_0_6px_-3px_rgba(0,0,0,0.4)]"
                        >
                          <div className="flex items-center gap-1">
                            <span>Compound Name</span>
                            {sortConfig.key === 'compound_name' && (
                              <span className="material-symbols-outlined text-[12px]">{sortConfig.direction === 'asc' ? 'arrow_upward' : 'arrow_downward'}</span>
                            )}
                          </div>
                        </th>
                        <th 
                          onClick={() => handleSort('smiles')}
                          className="sticky left-[160px] top-0 z-30 bg-surface-container-high border-b border-outline-variant text-left py-3 px-3 font-label-caps text-label-caps text-on-surface-variant hover:text-primary transition-colors cursor-pointer min-w-[180px] shadow-[3px_0_6px_-3px_rgba(0,0,0,0.4)]"
                        >
                          <div className="flex items-center gap-1">
                            <span>SMILES</span>
                            {sortConfig.key === 'smiles' && (
                              <span className="material-symbols-outlined text-[12px]">{sortConfig.direction === 'asc' ? 'arrow_upward' : 'arrow_downward'}</span>
                            )}
                          </div>
                        </th>
                        <th 
                          onClick={() => handleSort('formula')}
                          className="sticky top-0 bg-surface-container-high border-b border-outline-variant text-center py-3 px-2 font-label-caps text-label-caps text-on-surface-variant hover:text-primary transition-colors cursor-pointer whitespace-nowrap min-w-[100px]"
                        >
                          <div className="flex items-center justify-center gap-1">
                            <span>Formula</span>
                            {sortConfig.key === 'formula' && (
                              <span className="material-symbols-outlined text-[12px]">{sortConfig.direction === 'asc' ? 'arrow_upward' : 'arrow_downward'}</span>
                            )}
                          </div>
                        </th>
                        <th 
                          onClick={() => handleSort('molecular_weight')}
                          className="sticky top-0 bg-surface-container-high border-b border-outline-variant text-center py-3 px-2 font-label-caps text-label-caps text-on-surface-variant hover:text-primary transition-colors cursor-pointer whitespace-nowrap min-w-[100px]"
                        >
                          <div className="flex items-center justify-center gap-1">
                            <span>Mol. Wt</span>
                            {sortConfig.key === 'molecular_weight' && (
                              <span className="material-symbols-outlined text-[12px]">{sortConfig.direction === 'asc' ? 'arrow_upward' : 'arrow_downward'}</span>
                            )}
                          </div>
                        </th>
                        <th 
                          onClick={() => handleSort('flagged_endpoints')}
                          className="sticky top-0 bg-surface-container-high border-b border-outline-variant text-center py-3 px-2 font-label-caps text-label-caps text-on-surface-variant hover:text-primary transition-colors cursor-pointer whitespace-nowrap min-w-[90px]"
                        >
                          <div className="flex items-center justify-center gap-1">
                            <span>Flagged</span>
                            {sortConfig.key === 'flagged_endpoints' && (
                              <span className="material-symbols-outlined text-[12px]">{sortConfig.direction === 'asc' ? 'arrow_upward' : 'arrow_downward'}</span>
                            )}
                          </div>
                        </th>
                        <th 
                          onClick={() => handleSort('risk_level')}
                          className="sticky top-0 bg-surface-container-high border-b border-outline-variant text-center py-3 px-2 font-label-caps text-label-caps text-on-surface-variant hover:text-primary transition-colors cursor-pointer whitespace-nowrap min-w-[110px]"
                        >
                          <div className="flex items-center justify-center gap-1">
                            <span>Risk Level</span>
                            {sortConfig.key === 'risk_level' && (
                              <span className="material-symbols-outlined text-[12px]">{sortConfig.direction === 'asc' ? 'arrow_upward' : 'arrow_downward'}</span>
                            )}
                          </div>
                        </th>
                        {ALL_TASKS.map(t => (
                          <th 
                            key={t.key} 
                            onClick={() => handleSort(t.key)}
                            className="sticky top-0 bg-surface-container-high border-b border-outline-variant text-center py-3 px-2 font-label-caps text-label-caps text-on-surface-variant hover:text-primary transition-colors cursor-pointer whitespace-nowrap group min-w-[90px]"
                          >
                            <div className="flex items-center justify-center gap-1">
                              <span>{t.label}</span>
                              {sortConfig.key === t.key && (
                                <span className="material-symbols-outlined text-[12px]">{sortConfig.direction === 'asc' ? 'arrow_upward' : 'arrow_downward'}</span>
                              )}
                            </div>
                            
                            {/* Hover Tooltip */}
                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block bg-surface-container-highest border border-outline-variant rounded-lg p-2.5 w-48 shadow-xl z-50 text-[10px] font-normal normal-case text-on-surface whitespace-normal leading-relaxed text-center">
                              {t.desc}
                            </div>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedResults.map((row, i) => (
                        <tr key={i} className="border-b border-outline-variant/40 hover:bg-surface-container-high transition-colors group">
                          {/* Frozen Compound Name */}
                          <td className="sticky left-0 z-10 bg-surface-container group-hover:bg-surface-container-high py-3 px-3 font-code-sm text-code-sm text-primary font-semibold truncate max-w-[160px] shadow-[3px_0_6px_-3px_rgba(0,0,0,0.4)] transition-colors">
                            {row.compound_name || 'Unknown Compound'}
                          </td>
                          {/* Frozen SMILES */}
                          <td className="sticky left-[160px] z-10 bg-surface-container group-hover:bg-surface-container-high py-3 px-3 font-code-sm text-code-sm text-on-surface-variant truncate max-w-[180px] shadow-[3px_0_6px_-3px_rgba(0,0,0,0.4)] transition-colors" title={row.smiles}>
                            {row.smiles}
                          </td>
                          
                          <td className="py-3 px-2 text-center font-code-sm text-code-sm text-on-surface-variant">
                            {row.formula || '—'}
                          </td>
                          <td className="py-3 px-2 text-center font-metric-display text-on-surface-variant" style={{ fontSize: '13px' }}>
                            {row.molecular_weight ? `${row.molecular_weight} Da` : '—'}
                          </td>
                          <td className="py-3 px-2 text-center">
                            <span className={`px-2 py-0.5 rounded font-metric-display ${row.flagged_endpoints > 0 ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-primary/10 text-primary border border-primary/20'}`} style={{ fontSize: '12px' }}>
                              {row.flagged_endpoints || 0}
                            </span>
                          </td>
                          <td className="py-3 px-2 text-center">
                            {getRiskBadge(row.risk_level)}
                          </td>

                          {ALL_TASKS.map(t => (
                            <td key={t.key} className="py-3 px-2 text-center">
                              <span className={`font-metric-display ${cellColor(row[t.key], threshold)}`} style={{ fontSize: '13px' }}>
                                {row[t.key] !== undefined && row[t.key] !== null ? `${(row[t.key] * 100).toFixed(0)}%` : '—'}
                              </span>
                              {row[t.key] >= threshold && (
                                <span className="ml-1 text-red-400 material-symbols-outlined align-middle" style={{ fontSize: '12px' }}>warning</span>
                              )}
                            </td>
                          ))}
                        </tr>
                      ))}
                      {paginatedResults.length === 0 && (
                        <tr>
                          <td colSpan={6 + ALL_TASKS.length} className="py-10 text-center font-body-md text-body-md text-outline">
                            No compounds match the current filter criteria.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                {/* Pagination Controls */}
                {totalPages > 1 && (
                  <div className="flex flex-col sm:flex-row items-center justify-between gap-4 mt-6 pt-4 border-t border-outline-variant">
                    <div className="text-xs text-outline font-label-caps">
                      Showing {Math.min(filteredResults.length, (currentPage - 1) * pageSize + 1)} to {Math.min(filteredResults.length, currentPage * pageSize)} of {filteredResults.length} matching compounds
                    </div>
                    <div className="flex items-center gap-3">
                      <button
                        onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                        disabled={currentPage === 1}
                        className="px-3 py-1.5 rounded-lg border border-outline-variant text-[11px] font-label-caps text-on-surface-variant hover:text-primary hover:border-primary disabled:opacity-30 disabled:pointer-events-none transition-colors duration-200"
                      >
                        Previous
                      </button>
                      <div className="text-xs font-metric-display text-on-surface">
                        Page {currentPage} of {totalPages}
                      </div>
                      <button
                        onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                        disabled={currentPage === totalPages}
                        className="px-3 py-1.5 rounded-lg border border-outline-variant text-[11px] font-label-caps text-on-surface-variant hover:text-primary hover:border-primary disabled:opacity-30 disabled:pointer-events-none transition-colors duration-200"
                      >
                        Next
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Empty state */}
            {!file && (
              <div className="flex-1 bg-surface-container border border-dashed border-outline-variant rounded-xl flex flex-col items-center justify-center py-24 gap-4 text-outline">
                <span className="material-symbols-outlined text-5xl opacity-30">table_chart</span>
                <p className="font-body-md text-body-md text-center max-w-xs">Upload a CSV file to see batch predictions appear here</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
