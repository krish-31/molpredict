export default function Footer() {
  return (
    <footer className="w-full bg-surface-container-lowest border-t border-outline-variant">
      <div className="max-w-container-max mx-auto px-4 md:px-16 flex flex-col md:flex-row justify-between items-center gap-4 py-8">
        <div className="font-headline-md text-headline-md text-on-surface">GraphMol</div>
        <div className="flex gap-6 md:gap-8 items-center">
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="font-label-caps text-label-caps text-on-surface-variant hover:text-primary transition-colors"
          >
            GitHub
          </a>
          <span className="text-outline-variant">·</span>
          <a href="#" className="font-label-caps text-label-caps text-on-surface-variant hover:text-primary transition-colors">
            arXiv
          </a>
          <span className="text-outline-variant">·</span>
          <a href="#" className="font-label-caps text-label-caps text-on-surface-variant hover:text-primary transition-colors">
            Dataset
          </a>
        </div>
        <div className="font-body-md text-body-md text-on-surface-variant opacity-60">
          © 2024 Molecular ML Research Group
        </div>
      </div>
    </footer>
  )
}
