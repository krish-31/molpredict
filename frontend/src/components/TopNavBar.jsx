import { useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'

const navLinks = [
  { to: '/', label: 'HOME' },
  { to: '/predict', label: 'PREDICT' },
  { to: '/predict/batch', label: 'BATCH' },
  { to: '/train/configure', label: 'TRAIN' },
  { to: '/results', label: 'RESULTS' },
]

export default function TopNavBar() {
  const [open, setOpen] = useState(false)
  const navigate = useNavigate()

  return (
    <header className="fixed top-0 left-0 w-full z-50 flex justify-between items-center px-4 md:px-16 h-16 bg-surface/80 backdrop-blur-md border-b border-outline-variant/30">
      {/* Logo */}
      <button
        onClick={() => navigate('/')}
        className="font-headline-md text-headline-md font-bold text-on-surface hover:text-primary transition-colors duration-200"
      >
        MolPredict
      </button>

      {/* Desktop nav */}
      <nav className="hidden md:flex gap-8 items-center">
        {navLinks.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `font-label-caps text-label-caps transition-colors duration-200 pb-1 ${
                isActive
                  ? 'text-primary border-b-2 border-primary'
                  : 'text-on-surface-variant hover:text-primary'
              }`
            }
          >
            {label}
          </NavLink>
        ))}

      </nav>

      {/* Mobile hamburger */}
      <button
        className="md:hidden text-primary"
        onClick={() => setOpen(!open)}
        aria-label="Toggle menu"
      >
        <span className="material-symbols-outlined">{open ? 'close' : 'menu'}</span>
      </button>

      {/* Mobile dropdown */}
      {open && (
        <div className="absolute top-16 left-0 w-full bg-surface-container border-b border-outline-variant flex flex-col p-4 gap-4 md:hidden animate-fade-in">
          {navLinks.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `font-label-caps text-label-caps px-2 py-1 rounded transition-colors ${
                  isActive ? 'text-primary bg-primary/10' : 'text-on-surface-variant hover:text-primary'
                }`
              }
            >
              {label}
            </NavLink>
          ))}

        </div>
      )}
    </header>
  )
}
