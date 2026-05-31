import { useEffect, useRef } from 'react'

// Colors per toxicity probability
function barColor(prob) {
  if (prob < 0.3) return 'bg-primary'
  if (prob < 0.5) return 'bg-yellow-400'
  return 'bg-red-500'
}

function textColor(prob) {
  if (prob < 0.3) return 'text-primary'
  if (prob < 0.5) return 'text-yellow-400'
  return 'text-red-400'
}

export default function PropertyBar({ name, probability, description, delay = 0 }) {
  const barRef = useRef(null)

  useEffect(() => {
    if (!barRef.current) return
    const timer = setTimeout(() => {
      barRef.current.style.width = `${(probability * 100).toFixed(0)}%`
    }, delay)
    return () => clearTimeout(timer)
  }, [probability, delay])

  const pct = (probability * 100).toFixed(0)
  const label = probability >= 0.5 ? 1 : 0
  const col = barColor(probability)
  const txtCol = textColor(probability)

  return (
    <div className="group flex items-center gap-3 py-2.5 px-3 rounded-lg hover:bg-surface-container-high transition-colors duration-200">
      {/* Task name */}
      <div className="w-28 flex-shrink-0">
        <span className="font-code-sm text-code-sm text-on-surface-variant group-hover:text-on-surface transition-colors">
          {name}
        </span>
      </div>

      {/* Bar track */}
      <div className="flex-1 h-2 bg-surface-container-high rounded-full overflow-hidden">
        <div
          ref={barRef}
          className={`h-full ${col} rounded-full transition-all duration-700 ease-out`}
          style={{ width: '0%' }}
        />
      </div>

      {/* Probability */}
      <div className={`w-10 text-right font-metric-display text-sm ${txtCol} flex-shrink-0`}>
        {pct}%
      </div>

      {/* Label badge */}
      <div className={`w-16 flex-shrink-0 flex justify-end`}>
        {label === 1 ? (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-500/10 border border-red-500/30 text-red-400 font-label-caps text-label-caps">
            <span className="material-symbols-outlined text-xs" style={{ fontSize: '12px' }}>warning</span>
            TOXIC
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-primary/10 border border-primary/20 text-primary font-label-caps text-label-caps">
            <span className="material-symbols-outlined text-xs" style={{ fontSize: '12px' }}>check_circle</span>
            SAFE
          </span>
        )}
      </div>
    </div>
  )
}
