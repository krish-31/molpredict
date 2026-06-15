import { useState } from 'react'

// Minimal SVG molecule drawings for well-known molecules
const MOLECULE_SVGS = {
  default: (
    <svg viewBox="0 0 200 160" className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
      {/* Generic hexagonal ring */}
      <line x1="100" y1="30" x2="130" y2="55" stroke="#46f1d3" strokeWidth="2"/>
      <line x1="130" y1="55" x2="130" y2="90" stroke="#46f1d3" strokeWidth="2"/>
      <line x1="130" y1="90" x2="100" y2="115" stroke="#46f1d3" strokeWidth="2"/>
      <line x1="100" y1="115" x2="70" y2="90" stroke="#46f1d3" strokeWidth="2"/>
      <line x1="70" y1="90" x2="70" y2="55" stroke="#46f1d3" strokeWidth="2"/>
      <line x1="70" y1="55" x2="100" y2="30" stroke="#46f1d3" strokeWidth="2"/>
      {/* Inner ring (aromatic) */}
      <line x1="95" y1="42" x2="118" y2="57" stroke="#46f1d3" strokeWidth="1" strokeDasharray="3,2" opacity="0.5"/>
      <line x1="118" y1="57" x2="118" y2="83" stroke="#46f1d3" strokeWidth="1" strokeDasharray="3,2" opacity="0.5"/>
      <line x1="118" y1="83" x2="95" y2="98" stroke="#46f1d3" strokeWidth="1" strokeDasharray="3,2" opacity="0.5"/>
      <line x1="95" y1="98" x2="72" y2="83" stroke="#46f1d3" strokeWidth="1" strokeDasharray="3,2" opacity="0.5"/>
      <line x1="72" y1="83" x2="72" y2="57" stroke="#46f1d3" strokeWidth="1" strokeDasharray="3,2" opacity="0.5"/>
      <line x1="72" y1="57" x2="95" y2="42" stroke="#46f1d3" strokeWidth="1" strokeDasharray="3,2" opacity="0.5"/>
      {/* Atom nodes */}
      {[[100,30],[130,55],[130,90],[100,115],[70,90],[70,55]].map(([x,y],i) => (
        <circle key={i} cx={x} cy={y} r="5" fill="#1a1f2f" stroke="#46f1d3" strokeWidth="2"/>
      ))}
      {/* Carbon labels */}
      {[[100,22],[138,55],[138,90],[100,125],[62,90],[62,55]].map(([x,y],i) => (
        <text key={i} x={x} y={y} textAnchor="middle" dominantBaseline="middle" fill="#bacac5" fontSize="10" fontFamily="JetBrains Mono">C</text>
      ))}
    </svg>
  )
}

export default function MoleculeViewer({ smiles, svgContent, isValid }) {
  return (
    <div className="relative w-full aspect-square max-w-xs mx-auto rounded-xl border border-outline-variant bg-surface-container-lowest flex items-center justify-center overflow-hidden">
      {/* Grid background */}
      <div
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage: `linear-gradient(rgba(70,241,211,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(70,241,211,0.3) 1px, transparent 1px)`,
          backgroundSize: '20px 20px'
        }}
      />

      {!isValid ? (
        <div className="flex flex-col items-center gap-3 text-outline z-10">
          <span className="material-symbols-outlined text-4xl">science</span>
          <span className="font-code-sm text-code-sm">Enter a valid SMILES</span>
        </div>
      ) : (
        <div className="w-full h-full p-4 z-10 flex items-center justify-center">
          {svgContent ? (
            <div className="w-full h-full" dangerouslySetInnerHTML={{ __html: svgContent }} />
          ) : (
            MOLECULE_SVGS.default
          )}
          <div className="absolute bottom-3 left-3 right-3 bg-surface-container/80 rounded-lg px-3 py-1.5 border border-outline-variant/40">
            <p className="font-code-sm text-code-sm text-primary truncate">{smiles}</p>
          </div>
        </div>
      )}

      {/* Glow overlay */}
      {isValid && (
        <div className="absolute inset-0 bg-gradient-to-tr from-primary/5 via-transparent to-secondary/5 pointer-events-none" />
      )}
    </div>
  )
}
