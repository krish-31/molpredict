import { useState, useEffect, useRef } from 'react'
import { useLocation, Link } from 'react-router-dom'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  BarChart, Bar, Cell,
} from 'recharts'

const TASK_NAMES = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-γ','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

function generateEpochData(epoch) {
  const t = epoch / 200
  const noise = () => (Math.random() - 0.5) * 0.02
  return {
    epoch,
    trainLoss: Math.max(0.08, 0.85 * Math.exp(-3 * t) + 0.12 + noise()),
    valLoss: Math.max(0.10, 0.9 * Math.exp(-2.5 * t) + 0.15 + noise()),
    avgAUC: Math.min(0.90, 0.55 + 0.32 * (1 - Math.exp(-4 * t)) + noise()),
    conflictRate: Math.max(0.05, 0.32 * Math.exp(-2 * t) + 0.05 + noise()),
  }
}

function generateTaskAUC(epoch) {
  const t = epoch / 200
  return Object.fromEntries(TASK_NAMES.map((name, i) => {
    const base = 0.70 + (i % 5) * 0.03
    return [name, Math.min(0.95, base + 0.18 * (1 - Math.exp(-4 * t)) + (Math.random() - 0.5) * 0.015)]
  }))
}

function generateUncertainty(epoch) {
  const t = epoch / 200
  return Object.fromEntries(TASK_NAMES.map((name, i) => {
    const base = 0.3 + (i % 4) * 0.12
    return [name, Math.max(0.05, base * (1 - 0.4 * t) + (Math.random() - 0.5) * 0.02)]
  }))
}

const CHART_TOOLTIP_STYLE = {
  backgroundColor: '#1a1f2f',
  border: '1px solid #3b4a46',
  borderRadius: '8px',
  color: '#dee1f7',
  fontSize: '12px',
  fontFamily: 'JetBrains Mono',
}

export default function TrainMonitor() {
  const location = useLocation()
  const config = location.state?.config || { runName: 'demo_run', maxEpochs: 200 }

  const [epoch, setEpoch] = useState(0)
  const [running, setRunning] = useState(true)
  const [metrics, setMetrics] = useState([])
  const [taskAUC, setTaskAUC] = useState({})
  const [uncertainty, setUncertainty] = useState({})
  const [bestAUC, setBestAUC] = useState(0)
  const [bestEpoch, setBestEpoch] = useState(0)
  const intervalRef = useRef(null)

  useEffect(() => {
    if (!running) return
    intervalRef.current = setInterval(() => {
      setEpoch(prev => {
        const next = prev + 1
        const data = generateEpochData(next)
        setMetrics(m => [...m.slice(-60), data])
        const auc = generateTaskAUC(next)
        setTaskAUC(auc)
        setUncertainty(generateUncertainty(next))
        const avg = data.avgAUC
        setBestAUC(b => { if (avg > b) { setBestEpoch(next); return avg } return b })
        if (next >= config.maxEpochs) {
          clearInterval(intervalRef.current)
          setRunning(false)
        }
        return next
      })
    }, 300)
    return () => clearInterval(intervalRef.current)
  }, [running, config.maxEpochs])

  const lastMetric = metrics[metrics.length - 1] || {}
  const progress = Math.round((epoch / config.maxEpochs) * 100)

  return (
    <div className="pt-24 pb-16 min-h-screen">
      <div className="max-w-container-max mx-auto px-4 md:px-16">
        {/* Header */}
        <div className="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="font-headline-lg text-headline-lg-mobile md:text-headline-md text-primary mb-1">
              Training Monitor
            </h1>
            <p className="font-code-sm text-code-sm text-on-surface-variant">Run: {config.runName}</p>
          </div>
          <div className="flex items-center gap-3">
            <span className={`flex items-center gap-2 px-3 py-1.5 rounded-full border font-label-caps text-label-caps ${running ? 'border-primary/30 bg-primary/10 text-primary' : 'border-outline-variant text-outline'}`}>
              <span className={`w-2 h-2 rounded-full ${running ? 'bg-primary animate-pulse' : 'bg-outline'}`} />
              {running ? 'RUNNING' : 'COMPLETED'}
            </span>
            <button
              onClick={() => { setRunning(r => { if (r) clearInterval(intervalRef.current); return !r }) }}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg border font-label-caps text-label-caps transition-all duration-200 ${running ? 'border-red-500/40 text-red-400 hover:bg-red-500/10' : 'border-primary/40 text-primary hover:bg-primary/10'}`}
            >
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>{running ? 'stop' : 'play_arrow'}</span>
              {running ? 'Pause' : 'Resume'}
            </button>
            <Link
              to="/results"
              className="flex items-center gap-2 px-4 py-2 bg-primary text-on-primary rounded-lg font-label-caps text-label-caps hover:opacity-90 transition-all duration-200"
            >
              <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>open_in_new</span>
              View Results
            </Link>
          </div>
        </div>

        {/* Progress bar */}
        <div className="bg-surface-container border border-outline-variant rounded-xl p-6 mb-6">
          <div className="flex items-center justify-between mb-3">
            <span className="font-label-caps text-label-caps text-on-surface">
              Epoch <span className="text-primary font-metric-display" style={{ fontSize: '20px' }}>{epoch}</span>
              <span className="text-outline"> / {config.maxEpochs}</span>
            </span>
            <span className="font-code-sm text-code-sm text-on-surface-variant">
              {running ? `ETA: ~${Math.round((config.maxEpochs - epoch) * 0.3)}s` : 'Training complete'}
            </span>
          </div>
          <div className="w-full h-2 bg-surface-container-high rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="flex justify-between mt-2 font-code-sm text-code-sm text-outline">
            <span>0</span>
            <span className="text-primary">{progress}%</span>
            <span>{config.maxEpochs}</span>
          </div>
        </div>

        {/* Live metric cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-gutter mb-6">
          {[
            { label: 'AVG VAL AUC', val: lastMetric.avgAUC?.toFixed(4) ?? '—', icon: 'analytics', hi: true },
            { label: 'BEST AUC', val: bestAUC.toFixed(4), icon: 'star', sub: `Epoch ${bestEpoch}` },
            { label: 'CONFLICT RATE', val: `${((lastMetric.conflictRate ?? 0) * 100).toFixed(1)}%`, icon: 'sync_problem', warn: (lastMetric.conflictRate ?? 0) > 0.2 },
            { label: 'TRAIN LOSS', val: lastMetric.trainLoss?.toFixed(4) ?? '—', icon: 'trending_down' },
          ].map(({ label, val, icon, hi, warn, sub }) => (
            <div key={label} className={`bg-surface-container border rounded-xl p-5 ${hi ? 'border-primary/40' : warn ? 'border-red-500/30' : 'border-outline-variant'}`}>
              <div className="flex items-start justify-between mb-2">
                <span className="font-label-caps text-label-caps text-on-surface-variant">{label}</span>
                <span className={`material-symbols-outlined ${hi ? 'text-primary' : warn ? 'text-red-400' : 'text-outline'}`} style={{ fontSize: '18px' }}>{icon}</span>
              </div>
              <div className={`font-metric-display ${hi ? 'text-primary' : warn ? 'text-red-400' : 'text-on-surface'}`} style={{ fontSize: '28px' }}>{val}</div>
              {sub && <div className="font-code-sm text-code-sm text-outline mt-1">{sub}</div>}
            </div>
          ))}
        </div>

        {/* Charts row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-gutter mb-6">
          {/* Loss chart */}
          <div className="bg-surface-container border border-outline-variant rounded-xl p-6">
            <h3 className="font-headline-md text-on-surface mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-primary" style={{ fontSize: '20px' }}>trending_down</span>
              Loss Curves
            </h3>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={metrics} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#3b4a46" strokeOpacity={0.4} />
                <XAxis dataKey="epoch" stroke="#84948f" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                <YAxis stroke="#84948f" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} domain={['auto', 'auto']} />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'Inter' }} />
                <Line type="monotone" dataKey="trainLoss" stroke="#46f1d3" strokeWidth={2} dot={false} name="Train Loss" />
                <Line type="monotone" dataKey="valLoss" stroke="#c6c0ff" strokeWidth={2} dot={false} strokeDasharray="5 5" name="Val Loss" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* AUC chart */}
          <div className="bg-surface-container border border-outline-variant rounded-xl p-6">
            <h3 className="font-headline-md text-on-surface mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-primary" style={{ fontSize: '20px' }}>analytics</span>
              Average Validation AUC
            </h3>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={metrics} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#3b4a46" strokeOpacity={0.4} />
                <XAxis dataKey="epoch" stroke="#84948f" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                <YAxis stroke="#84948f" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} domain={[0.5, 1]} />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                <Line type="monotone" dataKey="avgAUC" stroke="#46f1d3" strokeWidth={2.5} dot={false} name="Avg Val AUC" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Conflict rate + Uncertainty */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-gutter mb-6">
          {/* Conflict rate */}
          <div className="bg-surface-container border border-outline-variant rounded-xl p-6">
            <h3 className="font-headline-md text-on-surface mb-1 flex items-center gap-2">
              <span className="material-symbols-outlined text-primary" style={{ fontSize: '20px' }}>sync_problem</span>
              Gradient Conflict Rate
            </h3>
            <p className="font-body-md text-body-md text-on-surface-variant mb-4" style={{ fontSize: '13px' }}>
              Fraction of task pairs (i,j) with conflicting gradients. PCGrad actively reduces this.
            </p>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={metrics} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#3b4a46" strokeOpacity={0.4} />
                <XAxis dataKey="epoch" stroke="#84948f" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                <YAxis stroke="#84948f" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} domain={[0, 0.4]} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} formatter={v => `${(v*100).toFixed(1)}%`} />
                <Line type="monotone" dataKey="conflictRate" stroke="#f87171" strokeWidth={2} dot={false} name="Conflict Rate" />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-3 flex items-center gap-2 text-on-surface-variant">
              <span className="material-symbols-outlined text-primary" style={{ fontSize: '16px' }}>arrow_downward</span>
              <span className="font-code-sm text-code-sm">
                {epoch > 5 ? `Reduced from ${(metrics[0]?.conflictRate * 100).toFixed(0)}% → ${(lastMetric.conflictRate * 100).toFixed(1)}%` : 'Monitoring…'}
              </span>
            </div>
          </div>

          {/* Uncertainty weights */}
          <div className="bg-surface-container border border-outline-variant rounded-xl p-6">
            <h3 className="font-headline-md text-on-surface mb-1 flex items-center gap-2">
              <span className="material-symbols-outlined text-primary" style={{ fontSize: '20px' }}>tune</span>
              Learned Uncertainty σᵢ
            </h3>
            <p className="font-body-md text-body-md text-on-surface-variant mb-4" style={{ fontSize: '13px' }}>
              Per-task loss scaling. Higher σᵢ → lower effective weight for that task.
            </p>
            <div className="space-y-2">
              {TASK_NAMES.slice(0, 6).map(name => {
                const val = uncertainty[name] ?? 0.3
                return (
                  <div key={name} className="flex items-center gap-3">
                    <span className="font-code-sm text-code-sm text-on-surface-variant w-24 flex-shrink-0" style={{ fontSize: '11px' }}>{name}</span>
                    <div className="flex-1 h-2 bg-surface-container-high rounded-full overflow-hidden">
                      <div
                        className="h-full bg-secondary rounded-full transition-all duration-500"
                        style={{ width: `${val * 200}%` }}
                      />
                    </div>
                    <span className="font-metric-display text-secondary w-12 text-right" style={{ fontSize: '12px' }}>
                      {val.toFixed(3)}
                    </span>
                  </div>
                )
              })}
              <div className="font-label-caps text-label-caps text-outline mt-2 text-center">
                + {TASK_NAMES.length - 6} more tasks
              </div>
            </div>
          </div>
        </div>

        {/* Per-task AUC table */}
        <div className="bg-surface-container border border-outline-variant rounded-xl p-6">
          <h3 className="font-headline-md text-on-surface mb-4 flex items-center gap-2">
            <span className="material-symbols-outlined text-primary" style={{ fontSize: '20px' }}>table_chart</span>
            Per-Task Validation AUC — Epoch {epoch}
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {TASK_NAMES.map(name => {
              const val = taskAUC[name] ?? 0
              const isGood = val >= 0.8
              const isMed = val >= 0.7 && val < 0.8
              return (
                <div key={name} className={`rounded-xl border p-3 transition-all duration-300 ${isGood ? 'border-primary/30 bg-primary/5' : isMed ? 'border-yellow-400/30 bg-yellow-400/5' : 'border-outline-variant'}`}>
                  <div className="font-label-caps text-label-caps text-on-surface-variant mb-1">{name}</div>
                  <div className={`font-metric-display ${isGood ? 'text-primary' : isMed ? 'text-yellow-400' : 'text-on-surface-variant'}`} style={{ fontSize: '22px' }}>
                    {val.toFixed(3)}
                  </div>
                  <div className="mt-2 h-1 bg-surface-container-high rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${isGood ? 'bg-primary' : isMed ? 'bg-yellow-400' : 'bg-outline'}`}
                      style={{ width: `${val * 100}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
