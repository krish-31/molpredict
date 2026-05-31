import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate, Link } from 'react-router-dom'
import PropertyBar from '../components/PropertyBar'
import MoleculeViewer from '../components/MoleculeViewer'

const TASK_INFO = [
  { key: 'NR-AR',        name: 'NR-AR',         desc: 'Nuclear Receptor — Androgen Receptor' },
  { key: 'NR-AR-LBD',   name: 'NR-AR-LBD',     desc: 'Androgen Receptor Ligand Binding Domain' },
  { key: 'NR-AhR',      name: 'NR-AhR',         desc: 'Aryl Hydrocarbon Receptor' },
  { key: 'NR-Aromatase',name: 'NR-Aromatase',   desc: 'Aromatase Enzyme Inhibition' },
  { key: 'NR-ER',       name: 'NR-ER',          desc: 'Estrogen Receptor Alpha' },
  { key: 'NR-ER-LBD',  name: 'NR-ER-LBD',      desc: 'Estrogen Receptor Ligand Binding Domain' },
  { key: 'NR-PPAR-γ',  name: 'NR-PPAR-γ',      desc: 'Peroxisome Proliferator-Activated Receptor Gamma' },
  { key: 'SR-ARE',      name: 'SR-ARE',          desc: 'Antioxidant Response Element' },
  { key: 'SR-ATAD5',   name: 'SR-ATAD5',        desc: 'ATPase Family AAA Domain Containing 5' },
  { key: 'SR-HSE',     name: 'SR-HSE',          desc: 'Heat Shock Element Pathway' },
  { key: 'SR-MMP',     name: 'SR-MMP',          desc: 'Mitochondrial Membrane Potential' },
  { key: 'SR-p53',     name: 'SR-p53',          desc: 'p53 Tumor Suppressor Pathway' },
]

// Mock prediction function — in production, calls POST /predict
function mockPredict(smiles) {
  const seed = smiles.split('').reduce((a, c) => a + c.charCodeAt(0), 0)
  const rand = (i) => ((seed * (i + 1) * 2654435761) >>> 0) / 4294967296
  return Object.fromEntries(TASK_INFO.map((t, i) => [t.key, Math.min(0.95, Math.max(0.02, rand(i)))]))
}

function isValidSmiles(s) {
  return s.length > 2 && /^[A-Za-z0-9@+\-\[\]()=#$/\\.%]+$/.test(s)
}

const MOL_PROPS = {
  'CC(=O)Oc1ccccc1C(=O)O': { formula: 'C₉H₈O₄', mw: '180.04 Da', hbd: 1, hba: 4, logp: 1.19, name: 'Aspirin' },
  'Cn1cnc2c1c(=O)n(c(=O)n2C)C': { formula: 'C₈H₁₀N₄O₂', mw: '194.19 Da', hbd: 0, hba: 3, logp: -0.07, name: 'Caffeine' },
  'CC(=O)Nc1ccc(cc1)O': { formula: 'C₈H₉NO₂', mw: '151.16 Da', hbd: 2, hba: 2, logp: 0.46, name: 'Paracetamol' },
}

export default function Predict() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const [smiles, setSmiles] = useState(searchParams.get('smiles') || '')
  const [draftSmiles, setDraftSmiles] = useState(searchParams.get('smiles') || '')
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [threshold, setThreshold] = useState(0.5)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const valid = isValidSmiles(draftSmiles)

  useEffect(() => {
    if (smiles && isValidSmiles(smiles)) runPredict(smiles)
  }, [])

  function runPredict(s = smiles) {
    if (!isValidSmiles(s)) return
    setLoading(true)
    setPredictions(null)
    setTimeout(() => {
      setPredictions(mockPredict(s))
      setLoading(false)
    }, 900)
  }

  const molProps = MOL_PROPS[smiles] || null
  const toxicCount = predictions ? Object.values(predictions).filter(v => v >= threshold).length : 0

  function downloadCSV() {
    if (!predictions) return
    const rows = [['Task', 'Probability', 'Label']]
    TASK_INFO.forEach(({ key }) => {
      const p = predictions[key]
      rows.push([key, p.toFixed(4), p >= threshold ? 1 : 0])
    })
    const csv = rows.map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'predictions.csv'; a.click()
  }

  return (
    <div className="pt-24 pb-16 min-h-screen">
      <div className="max-w-container-max mx-auto px-4 md:px-16">
        {/* Page header */}
        <div className="mb-8">
          <h1 className="font-headline-lg text-headline-lg-mobile md:text-headline-md text-primary mb-2">
            Single Molecule Prediction
          </h1>
          <p className="font-body-md text-body-md text-on-surface-variant">
            Enter a SMILES string to predict all 12 Tox21 toxicity endpoints simultaneously.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-gutter">
          {/* ── Left panel ── */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            {/* SMILES input card */}
            <div className="bg-surface-container border border-outline-variant rounded-xl p-6">
              <label className="font-label-caps text-label-caps text-on-surface-variant block mb-3">SMILES INPUT</label>
              <div className={`flex items-center gap-2 rounded-lg border px-3 py-2.5 transition-colors duration-200 ${valid ? 'border-primary/60 bg-primary/5' : draftSmiles ? 'border-red-500/40 bg-red-500/5' : 'border-outline-variant bg-surface-container-high'}`}>
                <span className={`material-symbols-outlined text-base ${valid ? 'text-primary' : draftSmiles ? 'text-red-400' : 'text-outline'}`}>
                  {valid ? 'check_circle' : 'science'}
                </span>
                <input
                  type="text"
                  value={draftSmiles}
                  onChange={e => setDraftSmiles(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') { setSmiles(draftSmiles); runPredict(draftSmiles) } }}
                  placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
                  className="flex-1 bg-transparent font-code-sm text-code-sm text-on-surface placeholder-outline focus:outline-none"
                />
              </div>
              {draftSmiles && (
                <p className={`font-label-caps text-label-caps mt-2 ${valid ? 'text-primary' : 'text-red-400'}`}>
                  {valid ? '✓ Valid SMILES format' : '✗ Invalid SMILES format'}
                </p>
              )}

              <button
                onClick={() => { setSmiles(draftSmiles); runPredict(draftSmiles) }}
                disabled={!valid || loading}
                className="mt-4 w-full py-3 bg-primary text-on-primary rounded-lg font-label-caps text-label-caps hover:opacity-90 transition-all duration-200 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <span className="material-symbols-outlined animate-spin" style={{ fontSize: '18px' }}>autorenew</span>
                    Predicting…
                  </>
                ) : (
                  <>
                    <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>bolt</span>
                    Predict Properties
                  </>
                )}
              </button>

              {/* Quick picks */}
              <div className="mt-4 flex flex-wrap gap-2">
                {[
                  { label: 'Aspirin', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
                  { label: 'Caffeine', smiles: 'Cn1cnc2c1c(=O)n(c(=O)n2C)C' },
                  { label: 'Paracetamol', smiles: 'CC(=O)Nc1ccc(cc1)O' },
                ].map(({ label, smiles: s }) => (
                  <button
                    key={label}
                    onClick={() => { setDraftSmiles(s); setSmiles(s); runPredict(s) }}
                    className="px-2.5 py-1 rounded-full border border-outline-variant text-on-surface-variant font-label-caps text-label-caps hover:border-primary hover:text-primary transition-all duration-200"
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Molecule viewer */}
            <div className="bg-surface-container border border-outline-variant rounded-xl p-6">
              <label className="font-label-caps text-label-caps text-on-surface-variant block mb-3">2D STRUCTURE</label>
              <MoleculeViewer smiles={smiles} isValid={!!smiles && isValidSmiles(smiles)} />
            </div>

            {/* Molecular properties */}
            {molProps && (
              <div className="bg-surface-container border border-outline-variant rounded-xl p-6 animate-fade-in">
                <label className="font-label-caps text-label-caps text-on-surface-variant block mb-3">MOLECULAR PROPERTIES</label>
                <div className="font-label-caps text-label-caps text-primary mb-3">{molProps.name}</div>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    ['Formula', molProps.formula],
                    ['Mol. Weight', molProps.mw],
                    ['HBD', molProps.hbd],
                    ['HBA', molProps.hba],
                    ['LogP', molProps.logp],
                  ].map(([k, v]) => (
                    <div key={k} className="bg-surface-container-high rounded-lg px-3 py-2">
                      <div className="font-label-caps text-label-caps text-outline">{k}</div>
                      <div className="font-metric-display text-on-surface mt-0.5" style={{ fontSize: '16px' }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

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
          </div>

          {/* ── Right panel ── */}
          <div className="lg:col-span-3 flex flex-col gap-6">
            {/* Results card */}
            <div className="bg-surface-container border border-outline-variant rounded-xl p-6 flex-1">
              <div className="flex items-center justify-between mb-6">
                <label className="font-label-caps text-label-caps text-on-surface-variant">PREDICTION RESULTS</label>
                {predictions && (
                  <div className="flex items-center gap-3">
                    <span className={`font-label-caps text-label-caps px-2 py-1 rounded-full ${toxicCount > 0 ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-primary/10 text-primary border border-primary/20'}`}>
                      {toxicCount} / 12 flagged
                    </span>
                  </div>
                )}
              </div>

              {!predictions && !loading && (
                <div className="flex flex-col items-center justify-center py-20 gap-4 text-outline">
                  <span className="material-symbols-outlined text-5xl opacity-40">biotech</span>
                  <p className="font-body-md text-body-md text-center max-w-xs">
                    Enter a SMILES string and click Predict to see all 12 property predictions
                  </p>
                </div>
              )}

              {loading && (
                <div className="flex flex-col items-center justify-center py-20 gap-4">
                  <div className="w-12 h-12 rounded-full border-2 border-primary/20 border-t-primary animate-spin" />
                  <p className="font-label-caps text-label-caps text-on-surface-variant">Running GIN inference…</p>
                  <div className="flex gap-1">
                    {[0,1,2,3,4].map(i => (
                      <div key={i} className="w-1 h-4 bg-primary rounded-full animate-pulse" style={{ animationDelay: `${i * 0.1}s` }} />
                    ))}
                  </div>
                </div>
              )}

              {predictions && (
                <div className="animate-fade-in">
                  {/* Summary bar */}
                  <div className="grid grid-cols-3 gap-3 mb-6">
                    {[
                      { label: 'TOTAL TASKS', val: '12', icon: 'analytics' },
                      { label: 'FLAGGED', val: toxicCount, icon: 'warning', warn: toxicCount > 0 },
                      { label: 'THRESHOLD', val: threshold.toFixed(2), icon: 'tune' },
                    ].map(({ label, val, icon, warn }) => (
                      <div key={label} className={`rounded-xl p-3 border text-center ${warn ? 'bg-red-500/10 border-red-500/20' : 'bg-surface-container-high border-outline-variant'}`}>
                        <span className={`material-symbols-outlined text-base block mb-1 ${warn ? 'text-red-400' : 'text-primary'}`}>{icon}</span>
                        <div className={`font-metric-display mb-0.5 ${warn ? 'text-red-400' : 'text-on-surface'}`} style={{ fontSize: '20px' }}>{val}</div>
                        <div className="font-label-caps text-label-caps text-outline">{label}</div>
                      </div>
                    ))}
                  </div>

                  <div className="card-header-line mb-4" />

                  {/* Property bars */}
                  <div className="space-y-1">
                    {TASK_INFO.map(({ key, name, desc }, idx) => (
                      <div key={key} title={desc}>
                        <PropertyBar
                          name={name}
                          probability={predictions[key]}
                          description={desc}
                          delay={idx * 60}
                        />
                      </div>
                    ))}
                  </div>

                  {/* Actions */}
                  <div className="flex gap-3 mt-6 pt-6 border-t border-outline-variant">
                    <button
                      onClick={downloadCSV}
                      className="flex-1 flex items-center justify-center gap-2 py-3 rounded-lg border border-outline-variant text-on-surface-variant font-label-caps text-label-caps hover:border-primary hover:text-primary transition-all duration-200"
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>download</span>
                      Download CSV
                    </button>
                    <Link
                      to="/predict/batch"
                      className="flex-1 flex items-center justify-center gap-2 py-3 rounded-lg border border-outline-variant text-on-surface-variant font-label-caps text-label-caps hover:border-primary hover:text-primary transition-all duration-200"
                    >
                      <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>upload_file</span>
                      Batch Upload
                    </Link>
                  </div>
                </div>
              )}
            </div>

            {/* Inference info */}
            {predictions && (
              <div className="bg-surface-container-low border border-outline-variant rounded-xl p-4 flex items-center justify-between animate-fade-in">
                <div className="flex items-center gap-2 text-outline">
                  <span className="material-symbols-outlined" style={{ fontSize: '16px' }}>speed</span>
                  <span className="font-code-sm text-code-sm">Inference: ~47ms · Model: GIN-5L-300d · Split: Scaffold</span>
                </div>
                <span className="font-label-caps text-label-caps text-primary">v1.0.0</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
