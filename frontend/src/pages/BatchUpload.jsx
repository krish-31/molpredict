import { useState, useRef } from 'react'
import { Link } from 'react-router-dom'

const MOCK_RESULTS = [
  { smiles: 'CC(=O)Oc1ccccc1C(=O)O', 'NR-AR': 0.12, 'NR-AhR': 0.45, 'SR-ARE': 0.52, 'SR-MMP': 0.11, 'SR-p53': 0.19 },
  { smiles: 'Cn1cnc2c1c(=O)n(c(=O)n2C)C', 'NR-AR': 0.03, 'NR-AhR': 0.08, 'SR-ARE': 0.11, 'SR-MMP': 0.04, 'SR-p53': 0.06 },
  { smiles: 'CC(=O)Nc1ccc(cc1)O', 'NR-AR': 0.07, 'NR-AhR': 0.15, 'SR-ARE': 0.23, 'SR-MMP': 0.09, 'SR-p53': 0.14 },
  { smiles: 'CC(C)Cc1ccc(cc1)C(C)C(=O)O', 'NR-AR': 0.18, 'NR-AhR': 0.31, 'SR-ARE': 0.44, 'SR-MMP': 0.22, 'SR-p53': 0.28 },
  { smiles: 'c1ccccc1', 'NR-AR': 0.05, 'NR-AhR': 0.62, 'SR-ARE': 0.28, 'SR-MMP': 0.08, 'SR-p53': 0.12 },
]

const SHOWN_TASKS = ['NR-AR', 'NR-AhR', 'SR-ARE', 'SR-MMP', 'SR-p53']

function cellColor(v) {
  if (v >= 0.5) return 'text-red-400'
  if (v >= 0.3) return 'text-yellow-400'
  return 'text-primary'
}

export default function BatchUpload() {
  const [dragOver, setDragOver] = useState(false)
  const [file, setFile] = useState(null)
  const [progress, setProgress] = useState(0)
  const [processing, setProcessing] = useState(false)
  const [done, setDone] = useState(false)
  const [results, setResults] = useState(null)
  const fileInputRef = useRef()

  function handleFile(f) {
    if (!f) return
    setFile(f)
    setDone(false)
    setResults(null)
    setProgress(0)
    setProcessing(true)

    // Simulate progressive processing
    let p = 0
    const interval = setInterval(() => {
      p += Math.random() * 18 + 5
      if (p >= 100) {
        p = 100
        clearInterval(interval)
        setTimeout(() => {
          setProcessing(false)
          setDone(true)
          setResults(MOCK_RESULTS)
        }, 400)
      }
      setProgress(Math.min(p, 100))
    }, 300)
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
    if (!results) return
    const header = ['smiles', ...SHOWN_TASKS, '...+7']
    const rows = results.map(r => [r.smiles, ...SHOWN_TASKS.map(t => r[t].toFixed(4)), ''])
    const csv = [header, ...rows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'batch_results.csv'; a.click()
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
                      {done ? `${MOCK_RESULTS.length} molecules · Completed` : `Processing… ${Math.round(progress)}%`}
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
              <div className="bg-surface-container border border-outline-variant rounded-xl p-6 animate-fade-in">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <div className="font-label-caps text-label-caps text-on-surface-variant">RESULTS PREVIEW</div>
                    <div className="font-code-sm text-code-sm text-outline mt-1">First {results.length} molecules · 12 tasks each</div>
                  </div>
                  <button
                    onClick={downloadResults}
                    className="flex items-center gap-2 px-4 py-2 bg-primary text-on-primary rounded-lg font-label-caps text-label-caps hover:opacity-90 transition-all duration-200 active:scale-95"
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>download</span>
                    Download CSV
                  </button>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full min-w-[600px]">
                    <thead>
                      <tr className="border-b border-outline-variant">
                        <th className="text-left py-3 px-3 font-label-caps text-label-caps text-on-surface-variant">SMILES</th>
                        {SHOWN_TASKS.map(t => (
                          <th key={t} className="text-center py-3 px-2 font-label-caps text-label-caps text-on-surface-variant whitespace-nowrap">{t}</th>
                        ))}
                        <th className="text-center py-3 px-2 font-label-caps text-label-caps text-outline">+7</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((row, i) => (
                        <tr key={i} className="border-b border-outline-variant/40 hover:bg-surface-container-high transition-colors">
                          <td className="py-3 px-3 font-code-sm text-code-sm text-on-surface-variant max-w-[180px] truncate">
                            {row.smiles}
                          </td>
                          {SHOWN_TASKS.map(t => (
                            <td key={t} className="py-3 px-2 text-center">
                              <span className={`font-metric-display ${cellColor(row[t])}`} style={{ fontSize: '13px' }}>
                                {(row[t] * 100).toFixed(0)}%
                              </span>
                              {row[t] >= 0.5 && (
                                <span className="ml-1 text-red-400 material-symbols-outlined" style={{ fontSize: '12px' }}>warning</span>
                              )}
                            </td>
                          ))}
                          <td className="py-3 px-2 text-center font-label-caps text-label-caps text-outline">…</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
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
