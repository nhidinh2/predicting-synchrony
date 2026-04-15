export default function PortfolioSelector({ portfolios, value, onChange }) {
  return (
    <div className="portfolio-selector">
      {portfolios.map((p) => (
        <button key={p} className={p === value ? 'active' : ''} onClick={() => onChange(p)}>
          Portfolio {p}
        </button>
      ))}
    </div>
  )
}
