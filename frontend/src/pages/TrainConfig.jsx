import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { apiClient } from '../api/client'

const POOLING = ['sum', 'mean', 'max']
const DATASETS = ['tox21', 'bace', 'bbbp', 'hiv', 'esol', 'lipophilicity']
const SPLITS = ['scaffold', 'random', 'stratified']

export default function TrainConfig() {
  const navigate = useNavigate()
  const [config, setConfig] = useState({
    runName: 'my_run_001',
    ginLayers: 5,
    hiddenDim: 300,
    dropout: 0.5,
    pooling: 'sum',
    dataset: 'tox21',
    split: 'scaffold',
    trainFrac: 80,
    valFrac: 10,
    testFrac: 10,
    batchSize: 128,
    maxEpochs: 200,
    lr: 0.001,
    weightDecay: 1e-5,
    patience: 20,
    usePCGrad: true,
    useUncertainty: true,
    useTaskGrouping: false,
  })
  const [saved, setSaved] = useState(false)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState(null)

  const set = (k, v) => setConfig(c => ({ ...c, [k]: v }))

  function startTraining() {
    setStarting(true)
    setError(null)
    apiClient.startTraining(config)
      .then(data => {
        setStarting(false)
        navigate('/train/monitor', { state: { config, runId: data.run_id } })
      })
      .catch(err => {
        console.error(err)
        setError("Failed to start training run on the backend. Make sure the server is online.")
        setStarting(false)
      })
  }

  function saveConfig() {
    const json = JSON.stringify(config, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `${config.runName}.json`; a.click()
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const inputCls = "w-full bg-surface-container-high border border-outline-variant rounded-lg px-3 py-2.5 text-on-surface font-code-sm text-code-sm focus:outline-none focus:border-primary transition-colors duration-200"
  const selectCls = inputCls + " cursor-pointer"
  const labelCls = "font-label-caps text-label-caps text-on-surface-variant block mb-2"
  const sectionCls = "bg-surface-container border border-outline-variant rounded-xl p-6"

  return (
    <div className="pt-24 pb-16 min-h-screen">
      <div className="max-w-container-max mx-auto px-4 md:px-16">
        {/* Header */}
        <div className="mb-8">
          <h1 className="font-headline-lg text-headline-lg-mobile md:text-headline-md text-primary mb-2">
            Configure Training Run
          </h1>
          <p className="font-body-md text-body-md text-on-surface-variant">
            Configure the GIN backbone, training hyperparameters, and multi-task strategy.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-gutter">
          {/* Left column */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            {/* Run name */}
            <div className={sectionCls}>
              <label className={labelCls}>RUN NAME</label>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-outline" style={{ fontSize: '18px' }}>label</span>
                <input
                  type="text"
                  value={config.runName}
                  onChange={e => set('runName', e.target.value)}
                  className={inputCls + ' pl-10'}
                />
              </div>
            </div>

            {/* Model architecture */}
            <div className={sectionCls}>
              <h3 className="font-headline-md text-headline-md text-on-surface mb-6 flex items-center gap-2">
                <span className="material-symbols-outlined text-primary">hub</span>
                Model Architecture
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-gutter">
                <div>
                  <label className={labelCls}>GIN LAYERS</label>
                  <div className="flex items-center gap-3">
                    <input
                      type="range" min="2" max="8" step="1"
                      value={config.ginLayers}
                      onChange={e => set('ginLayers', parseInt(e.target.value))}
                      className="flex-1 accent-primary"
                    />
                    <span className="font-metric-display text-primary w-8 text-right" style={{ fontSize: '20px' }}>{config.ginLayers}</span>
                  </div>
                </div>
                <div>
                  <label className={labelCls}>HIDDEN DIM</label>
                  <select className={selectCls} value={config.hiddenDim} onChange={e => set('hiddenDim', parseInt(e.target.value))}>
                    {[64, 128, 256, 300, 512].map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                </div>
                <div>
                  <label className={labelCls}>DROPOUT: {config.dropout.toFixed(2)}</label>
                  <input
                    type="range" min="0" max="0.9" step="0.05"
                    value={config.dropout}
                    onChange={e => set('dropout', parseFloat(e.target.value))}
                    className="w-full accent-primary"
                  />
                </div>
                <div>
                  <label className={labelCls}>GRAPH POOLING</label>
                  <div className="flex gap-2">
                    {POOLING.map(p => (
                      <button
                        key={p}
                        onClick={() => set('pooling', p)}
                        className={`flex-1 py-2.5 rounded-lg font-label-caps text-label-caps transition-all duration-200 border ${config.pooling === p ? 'bg-primary text-on-primary border-primary' : 'border-outline-variant text-on-surface-variant hover:border-primary hover:text-primary'}`}
                      >
                        {p.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Training config */}
            <div className={sectionCls}>
              <h3 className="font-headline-md text-headline-md text-on-surface mb-6 flex items-center gap-2">
                <span className="material-symbols-outlined text-primary">settings</span>
                Training Configuration
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-gutter">
                <div>
                  <label className={labelCls}>DATASET</label>
                  <select className={selectCls} value={config.dataset} onChange={e => set('dataset', e.target.value)}>
                    {DATASETS.map(d => <option key={d} value={d}>{d.toUpperCase()}</option>)}
                  </select>
                </div>
                <div>
                  <label className={labelCls}>SPLIT STRATEGY</label>
                  <div className="flex gap-2">
                    {SPLITS.map(s => (
                      <button
                        key={s}
                        onClick={() => set('split', s)}
                        className={`flex-1 py-2.5 rounded-lg font-label-caps text-label-caps transition-all duration-200 border text-xs ${config.split === s ? 'bg-primary text-on-primary border-primary' : 'border-outline-variant text-on-surface-variant hover:border-primary hover:text-primary'}`}
                      >
                        {s.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className={labelCls}>SPLIT RATIO</label>
                  <div className="flex gap-2 items-center">
                    {[
                      { k: 'trainFrac', label: 'Train' },
                      { k: 'valFrac', label: 'Val' },
                      { k: 'testFrac', label: 'Test' },
                    ].map(({ k, label }) => (
                      <div key={k} className="flex-1">
                        <div className="font-code-sm text-code-sm text-outline mb-1">{label}</div>
                        <input
                          type="number" min="5" max="90" step="5"
                          value={config[k]}
                          onChange={e => set(k, parseInt(e.target.value))}
                          className={inputCls + ' text-center'}
                        />
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <label className={labelCls}>BATCH SIZE</label>
                  <select className={selectCls} value={config.batchSize} onChange={e => set('batchSize', parseInt(e.target.value))}>
                    {[32, 64, 128, 256].map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                </div>
                <div>
                  <label className={labelCls}>MAX EPOCHS</label>
                  <input type="number" min="10" max="500" value={config.maxEpochs} onChange={e => set('maxEpochs', parseInt(e.target.value))} className={inputCls} />
                </div>
                <div>
                  <label className={labelCls}>LEARNING RATE</label>
                  <select className={selectCls} value={config.lr} onChange={e => set('lr', parseFloat(e.target.value))}>
                    {[0.01, 0.001, 0.0005, 0.0001].map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                </div>
                <div>
                  <label className={labelCls}>EARLY STOPPING PATIENCE</label>
                  <input type="number" min="5" max="50" value={config.patience} onChange={e => set('patience', parseInt(e.target.value))} className={inputCls} />
                </div>
              </div>
            </div>

            {/* MTL strategy */}
            <div className={sectionCls}>
              <h3 className="font-headline-md text-headline-md text-on-surface mb-6 flex items-center gap-2">
                <span className="material-symbols-outlined text-primary">account_tree</span>
                Multi-Task Strategy
              </h3>
              <div className="space-y-4">
                {[
                  {
                    key: 'usePCGrad', label: 'PCGrad — Projecting Conflicting Gradients',
                    desc: 'Detect gradient conflicts between task pairs and project them to eliminate destructive interference.',
                    badge: 'Recommended',
                  },
                  {
                    key: 'useUncertainty', label: 'Learned Uncertainty Weighting (Kendall et al.)',
                    desc: 'Adaptive task loss balancing via learnable log-uncertainty σᵢ per task.',
                    badge: 'Recommended',
                  },
                  {
                    key: 'useTaskGrouping', label: 'Task Grouping (Experimental)',
                    desc: 'Group tasks by gradient similarity and apply separate shared heads per group.',
                    badge: 'Experimental',
                  },
                ].map(({ key, label, desc, badge }) => (
                  <div
                    key={key}
                    onClick={() => set(key, !config[key])}
                    className={`flex items-start gap-4 p-5 rounded-xl border cursor-pointer transition-all duration-200 ${config[key] ? 'border-primary/40 bg-primary/5' : 'border-outline-variant hover:border-primary/30'}`}
                  >
                    <div className={`flex-shrink-0 w-6 h-6 rounded border-2 flex items-center justify-center transition-all duration-200 mt-0.5 ${config[key] ? 'border-primary bg-primary' : 'border-outline-variant'}`}>
                      {config[key] && <span className="material-symbols-outlined text-on-primary" style={{ fontSize: '14px' }}>check</span>}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-body-md text-body-md text-on-surface">{label}</span>
                        <span className={`font-label-caps text-label-caps px-2 py-0.5 rounded-full border ${badge === 'Recommended' ? 'text-primary border-primary/30 bg-primary/10' : 'text-yellow-400 border-yellow-400/30 bg-yellow-400/10'}`}>
                          {badge}
                        </span>
                      </div>
                      <p className="font-body-md text-body-md text-on-surface-variant" style={{ fontSize: '14px' }}>{desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right column — summary + actions */}
          <div className="lg:col-span-1 flex flex-col gap-6">
            {/* Config summary */}
            <div className="sticky top-24 flex flex-col gap-6">
              <div className={sectionCls}>
                <h3 className="font-headline-md text-on-surface mb-5">Run Summary</h3>
                <div className="space-y-3">
                  {[
                    ['Run', config.runName],
                    ['Dataset', config.dataset.toUpperCase()],
                    ['Split', config.split],
                    ['GIN Layers', config.ginLayers],
                    ['Hidden Dim', config.hiddenDim],
                    ['Dropout', config.dropout.toFixed(2)],
                    ['Pooling', config.pooling],
                    ['Batch Size', config.batchSize],
                    ['Max Epochs', config.maxEpochs],
                    ['LR', config.lr],
                    ['Patience', config.patience],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between items-center py-1.5 border-b border-outline-variant/30">
                      <span className="font-label-caps text-label-caps text-on-surface-variant">{k}</span>
                      <span className="font-metric-display text-on-surface" style={{ fontSize: '14px' }}>{v}</span>
                    </div>
                  ))}
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  {config.usePCGrad && (
                    <span className="px-2 py-1 bg-primary/10 border border-primary/20 rounded-full font-label-caps text-label-caps text-primary">PCGrad</span>
                  )}
                  {config.useUncertainty && (
                    <span className="px-2 py-1 bg-secondary/10 border border-secondary/20 rounded-full font-label-caps text-label-caps text-secondary">Uncertainty</span>
                  )}
                  {config.useTaskGrouping && (
                    <span className="px-2 py-1 bg-yellow-400/10 border border-yellow-400/20 rounded-full font-label-caps text-label-caps text-yellow-400">Grouping</span>
                  )}
                </div>
              </div>

              {error && (
                <div className="p-3 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg text-xs font-code-sm text-center">
                  {error}
                </div>
              )}

              {/* Action buttons */}
              <button
                onClick={startTraining}
                disabled={starting}
                className="w-full py-4 bg-primary text-on-primary rounded-xl font-label-caps text-label-caps hover:opacity-90 transition-all duration-200 active:scale-95 disabled:opacity-60 flex items-center justify-center gap-2"
              >
                {starting ? (
                  <>
                    <span className="material-symbols-outlined animate-spin" style={{ fontSize: '18px' }}>autorenew</span>
                    Starting…
                  </>
                ) : (
                  <>
                    <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>play_arrow</span>
                    Start Training
                  </>
                )}
              </button>

              <button
                onClick={saveConfig}
                className={`w-full py-3 rounded-xl border font-label-caps text-label-caps transition-all duration-200 flex items-center justify-center gap-2 ${saved ? 'border-primary text-primary bg-primary/10' : 'border-outline-variant text-on-surface-variant hover:border-primary hover:text-primary'}`}
              >
                <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>{saved ? 'check' : 'save'}</span>
                {saved ? 'Saved!' : 'Save Config JSON'}
              </button>

              {/* Estimated time */}
              <div className="bg-surface-container-low border border-outline-variant rounded-xl p-4">
                <div className="font-label-caps text-label-caps text-on-surface-variant mb-2">ESTIMATED TRAINING TIME</div>
                <div className="font-metric-display text-primary" style={{ fontSize: '20px' }}>
                  ~{Math.round(config.ginLayers * config.hiddenDim * config.maxEpochs / 150000 * 2 + 0.5)}h
                </div>
                <div className="font-code-sm text-code-sm text-outline mt-1">on GPU · Tox21 dataset</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
