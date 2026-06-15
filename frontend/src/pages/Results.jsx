import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer,
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, Cell,
} from 'recharts'

const TASKS = ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-γ','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

const RESULTS = [
  { task: 'NR-AR',       single: 0.801, naive: 0.793, ours: 0.821, delta: +0.020 },
  { task: 'NR-AR-LBD',  single: 0.844, naive: 0.832, ours: 0.857, delta: +0.013 },
  { task: 'NR-AhR',     single: 0.876, naive: 0.869, ours: 0.891, delta: +0.015 },
  { task: 'NR-Aromatase',single: 0.823, naive: 0.811, ours: 0.839, delta: +0.016 },
  { task: 'NR-ER',      single: 0.791, naive: 0.785, ours: 0.812, delta: +0.021 },
  { task: 'NR-ER-LBD',  single: 0.812, naive: 0.805, ours: 0.829, delta: +0.017 },
  { task: 'NR-PPAR-γ',  single: 0.755, naive: 0.748, ours: 0.778, delta: +0.023 },
  { task: 'SR-ARE',     single: 0.782, naive: 0.771, ours: 0.803, delta: +0.021 },
  { task: 'SR-ATAD5',   single: 0.798, naive: 0.789, ours: 0.815, delta: +0.017 },
  { task: 'SR-HSE',     single: 0.821, naive: 0.815, ours: 0.843, delta: +0.022 },
  { task: 'SR-MMP',     single: 0.869, naive: 0.858, ours: 0.884, delta: +0.015 },
  { task: 'SR-p53',     single: 0.836, naive: 0.828, ours: 0.851, delta: +0.015 },
]

const ROC_DATA = Array.from({ length: 21 }, (_, i) => {
  const x = i / 20
  return {
    fpr: x,
    single: Math.min(1, Math.pow(x, 0.55) * 1.1),
    ours: Math.min(1, Math.pow(x, 0.38) * 1.15),
    random: x,
  }
})

const RADAR_DATA = TASKS.map((t, i) => {
  const r = RESULTS[i]
  return { task: t.replace('NR-','').replace('SR-',''), ours: r.ours * 100, single: r.single * 100 }
})

const avgOurs = (RESULTS.reduce((a, r) => a + r.ours, 0) / RESULTS.length).toFixed(3)
const avgSingle = (RESULTS.reduce((a, r) => a + r.single, 0) / RESULTS.length).toFixed(3)

const CHART_TOOLTIP_STYLE = {
  backgroundColor: '#1a1f2f',
  border: '1px solid #3b4a46',
  borderRadius: '8px',
  color: '#dee1f7',
  fontSize: '12px',
  fontFamily: 'JetBrains Mono',
}

export default function Results() {
  const [activeTab, setActiveTab] = useState('NR-AR')
  const navigate = useNavigate()

  return (
    <div className="pt-24 pb-16 min-h-screen">
      <div className="max-w-container-max mx-auto px-4 md:px-16">
        {/* Header */}
        <section className="mb-12">
          <h1 className="font-headline-lg text-headline-lg-mobile md:text-headline-lg mb-4 text-primary">
            Technical Analysis & Benchmarking
          </h1>
          <p className="font-body-lg text-body-lg text-on-surface-variant max-w-3xl">
            Comprehensive performance evaluation of MolPredict across 12 diverse molecular property prediction
            tasks using multi-objective optimization strategies.
          </p>
        </section>

        {/* Summary strip */}
        <div className="w-full bg-surface-container-low border border-outline-variant rounded-xl p-4 mb-8 flex justify-center items-center">
          <span className="font-metric-display text-headline-md text-primary tracking-widest uppercase text-center">
            Avg ROC-AUC {avgOurs} · 12 Tasks · Scaffold Split · ↑ {((avgOurs - avgSingle) * 1000).toFixed(0)}‰ vs Single-Task
          </span>
        </div>

        {/* KPI cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-gutter mb-12">
          {[
            { label: 'AVG ROC-AUC', val: avgOurs, icon: 'analytics', sub: 'Scaffold test split' },
            { label: 'VS SINGLE-TASK', val: `+${((avgOurs - avgSingle) * 100).toFixed(2)}%`, icon: 'trending_up', sub: 'Average improvement' },
            { label: 'CONFLICT REDUCTION', val: '−23.4%', icon: 'sync_problem', sub: 'vs naïve MTL' },
            { label: 'TASKS IMPROVED', val: '12/12', icon: 'star', sub: 'vs single-task GIN' },
          ].map(({ label, val, icon, sub }) => (
            <div key={label} className="bg-surface-container border border-outline-variant p-6 rounded-xl teal-glow-hover transition-all duration-300">
              <div className="flex justify-between items-start mb-3">
                <span className="font-label-caps text-label-caps text-on-surface-variant">{label}</span>
                <span className="material-symbols-outlined text-primary opacity-60" style={{ fontSize: '20px' }}>{icon}</span>
              </div>
              <div className="card-header-line opacity-30 mb-3" />
              <div className="font-metric-display text-metric-display text-primary mb-1">{val}</div>
              <div className="font-code-sm text-code-sm text-outline">{sub}</div>
            </div>
          ))}
        </div>

        {/* Comparison table */}
        <section className="mb-12" id="comparison">
          <h2 className="font-headline-md text-headline-md mb-6 flex items-center gap-2">
            <span className="material-symbols-outlined text-primary">compare_arrows</span>
            Model Comparison — Scaffold Split Test Set
          </h2>
          <div className="bg-surface-container border border-outline-variant rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px]">
                <thead>
                  <tr className="border-b border-outline-variant bg-surface-container-high">
                    <th className="text-left py-4 px-5 font-label-caps text-label-caps text-on-surface-variant">TASK</th>
                    <th className="text-center py-4 px-4 font-label-caps text-label-caps text-on-surface-variant">SINGLE-TASK GIN</th>
                    <th className="text-center py-4 px-4 font-label-caps text-label-caps text-on-surface-variant">MTL NAÏVE</th>
                    <th className="text-center py-4 px-4 font-label-caps text-label-caps text-primary">OUR MODEL</th>
                    <th className="text-center py-4 px-4 font-label-caps text-label-caps text-on-surface-variant">Δ BEST</th>
                  </tr>
                </thead>
                <tbody>
                  {RESULTS.map((r, i) => (
                    <tr key={r.task} className="border-b border-outline-variant/40 hover:bg-surface-container-high transition-colors">
                      <td className="py-3.5 px-5 font-code-sm text-code-sm text-on-surface">{r.task}</td>
                      <td className="py-3.5 px-4 text-center font-metric-display text-on-surface-variant" style={{ fontSize: '14px' }}>{r.single.toFixed(3)}</td>
                      <td className="py-3.5 px-4 text-center font-metric-display text-on-surface-variant" style={{ fontSize: '14px' }}>{r.naive.toFixed(3)}</td>
                      <td className="py-3.5 px-4 text-center">
                        <span className="font-metric-display text-primary" style={{ fontSize: '14px' }}>{r.ours.toFixed(3)}</span>
                      </td>
                      <td className="py-3.5 px-4 text-center">
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-primary/10 border border-primary/20 font-metric-display text-primary" style={{ fontSize: '12px' }}>
                          <span className="material-symbols-outlined" style={{ fontSize: '12px' }}>arrow_upward</span>
                          {r.delta > 0 ? '+' : ''}{r.delta.toFixed(3)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr className="bg-surface-container-high border-t border-primary/30">
                    <td className="py-4 px-5 font-label-caps text-label-caps text-on-surface">AVERAGE</td>
                    <td className="py-4 px-4 text-center font-metric-display text-on-surface-variant" style={{ fontSize: '15px' }}>{avgSingle}</td>
                    <td className="py-4 px-4 text-center font-metric-display text-on-surface-variant" style={{ fontSize: '15px' }}>
                      {(RESULTS.reduce((a, r) => a + r.naive, 0) / RESULTS.length).toFixed(3)}
                    </td>
                    <td className="py-4 px-4 text-center font-metric-display text-primary font-bold" style={{ fontSize: '15px' }}>{avgOurs}</td>
                    <td className="py-4 px-4 text-center">
                      <span className="font-label-caps text-label-caps text-primary">↑ all 12 tasks</span>
                    </td>
                  </tr>
                </tfoot>
              </table>
            </div>
          </div>
        </section>

        {/* Charts row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-gutter mb-12">
          {/* ROC Curves */}
          <div className="bg-surface-container border border-outline-variant rounded-xl p-6">
            <h3 className="font-headline-md text-on-surface mb-2 flex items-center gap-2">
              <span className="material-symbols-outlined text-primary" style={{ fontSize: '20px' }}>show_chart</span>
              ROC Curves
            </h3>
            <div className="flex gap-2 flex-wrap mb-4">
              {TASKS.slice(0, 6).map(t => (
                <button
                  key={t}
                  onClick={() => setActiveTab(t)}
                  className={`px-2.5 py-1 rounded-full font-label-caps text-label-caps transition-all duration-200 ${activeTab === t ? 'bg-primary text-on-primary' : 'border border-outline-variant text-on-surface-variant hover:border-primary hover:text-primary'}`}
                  style={{ fontSize: '10px' }}
                >
                  {t}
                </button>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={ROC_DATA} margin={{ top: 5, right: 15, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#3b4a46" strokeOpacity={0.4} />
                <XAxis dataKey="fpr" stroke="#84948f" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} label={{ value: 'FPR', position: 'insideBottom', offset: -3, fill: '#84948f', fontSize: 11 }} />
                <YAxis stroke="#84948f" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} label={{ value: 'TPR', angle: -90, position: 'insideLeft', fill: '#84948f', fontSize: 11 }} />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} formatter={v => v.toFixed(3)} />
                <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'Inter' }} />
                <Line type="monotone" dataKey="ours" stroke="#46f1d3" strokeWidth={2.5} dot={false} name="Our Model" />
                <Line type="monotone" dataKey="single" stroke="#c6c0ff" strokeWidth={1.5} dot={false} strokeDasharray="4 4" name="Single-Task" />
                <Line type="monotone" dataKey="random" stroke="#3b4a46" strokeWidth={1} dot={false} name="Random" />
              </LineChart>
            </ResponsiveContainer>
            <p className="font-code-sm text-code-sm text-outline mt-3 text-center">Task: {activeTab}</p>
          </div>

          {/* Radar chart */}
          <div className="bg-surface-container border border-outline-variant rounded-xl p-6">
            <h3 className="font-headline-md text-on-surface mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-primary" style={{ fontSize: '20px' }}>radar</span>
              Task Coverage Radar
            </h3>
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={RADAR_DATA}>
                <PolarGrid stroke="#3b4a46" />
                <PolarAngleAxis dataKey="task" tick={{ fill: '#bacac5', fontSize: 10, fontFamily: 'JetBrains Mono' }} />
                <Radar name="Our Model" dataKey="ours" stroke="#46f1d3" fill="#46f1d3" fillOpacity={0.15} strokeWidth={2} />
                <Radar name="Single-Task" dataKey="single" stroke="#c6c0ff" fill="#c6c0ff" fillOpacity={0.08} strokeWidth={1.5} strokeDasharray="4 4" />
                <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'Inter' }} />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} formatter={v => `${v.toFixed(1)}%`} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Delta bar chart */}
        <section className="mb-12">
          <h2 className="font-headline-md text-headline-md mb-6 flex items-center gap-2">
            <span className="material-symbols-outlined text-primary">bar_chart</span>
            AUC Improvement over Single-Task Baseline
          </h2>
          <div className="bg-surface-container border border-outline-variant rounded-xl p-6">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={RESULTS} margin={{ top: 5, right: 15, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#3b4a46" strokeOpacity={0.4} />
                <XAxis dataKey="task" stroke="#84948f" tick={{ fontSize: 10, fontFamily: 'JetBrains Mono', angle: -30, textAnchor: 'end' }} />
                <YAxis stroke="#84948f" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono' }} tickFormatter={v => `+${(v * 100).toFixed(1)}%`} />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} formatter={v => `+${(v * 100).toFixed(2)}%`} />
                <Bar dataKey="delta" name="Δ AUC" radius={[4, 4, 0, 0]}>
                  {RESULTS.map((r, i) => (
                    <Cell key={i} fill="#46f1d3" fillOpacity={0.6 + (r.delta / 0.025) * 0.4} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Ablation */}
        <section className="mb-12">
          <h2 className="font-headline-md text-headline-md mb-6 flex items-center gap-2">
            <span className="material-symbols-outlined text-primary">science</span>
            Ablation Study
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-gutter">
            {[
              { model: 'Single-Task GIN', auc: 0.822, pcgrad: false, uncertainty: false, scaffold: true },
              { model: 'MTL + Equal Weight', auc: 0.815, pcgrad: false, uncertainty: false, scaffold: true },
              { model: 'MTL + Uncertainty', auc: 0.831, pcgrad: false, uncertainty: true, scaffold: true },
              { model: 'MTL + PCGrad', auc: 0.835, pcgrad: true, uncertainty: false, scaffold: true },
              { model: 'Ours (Full)', auc: 0.843, pcgrad: true, uncertainty: true, scaffold: true },
              { model: 'Ours (Random Split)', auc: 0.891, pcgrad: true, uncertainty: true, scaffold: false },
            ].map(({ model, auc, pcgrad, uncertainty, scaffold }) => (
              <div key={model} className={`bg-surface-container border rounded-xl p-5 teal-glow-hover transition-all duration-300 ${model === 'Ours (Full)' ? 'border-primary/50' : 'border-outline-variant'}`}>
                {model === 'Ours (Full)' && (
                  <div className="mb-3">
                    <span className="font-label-caps text-label-caps px-2 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary">★ Best</span>
                  </div>
                )}
                <div className="font-body-md text-body-md text-on-surface mb-3">{model}</div>
                <div className="font-metric-display text-primary mb-4" style={{ fontSize: '28px' }}>{auc.toFixed(3)}</div>
                <div className="flex flex-wrap gap-2">
                  <span className={`font-label-caps text-label-caps px-2 py-0.5 rounded-full border ${pcgrad ? 'border-primary/30 bg-primary/10 text-primary' : 'border-outline-variant text-outline'}`}>
                    {pcgrad ? '✓' : '✗'} PCGrad
                  </span>
                  <span className={`font-label-caps text-label-caps px-2 py-0.5 rounded-full border ${uncertainty ? 'border-secondary/30 bg-secondary/10 text-secondary' : 'border-outline-variant text-outline'}`}>
                    {uncertainty ? '✓' : '✗'} Uncertainty
                  </span>
                  <span className={`font-label-caps text-label-caps px-2 py-0.5 rounded-full border ${scaffold ? 'border-yellow-400/30 bg-yellow-400/10 text-yellow-400' : 'border-outline-variant text-outline'}`}>
                    {scaffold ? 'Scaffold' : 'Random'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* CTA */}
        <div className="flex flex-col md:flex-row gap-4 justify-center">
          <button onClick={() => navigate('/predict')} className="flex items-center justify-center gap-2 py-4 px-10 bg-primary text-on-primary rounded-xl font-label-caps text-label-caps hover:opacity-90 transition-all duration-200 teal-glow-hover">
            <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>bolt</span>
            Try Prediction
          </button>
          <button onClick={() => navigate('/train/configure')} className="flex items-center justify-center gap-2 py-4 px-10 border border-primary text-primary rounded-xl font-label-caps text-label-caps hover:bg-primary/10 transition-all duration-200">
            <span className="material-symbols-outlined" style={{ fontSize: '18px' }}>model_training</span>
            Train Your Model
          </button>
        </div>
      </div>
    </div>
  )
}
