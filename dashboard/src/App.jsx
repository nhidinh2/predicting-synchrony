import { useEffect, useState } from 'react'
import PortfolioSelector from './components/PortfolioSelector.jsx'
import ForecastChart from './components/ForecastChart.jsx'
import DailyTotalsTable from './components/DailyTotalsTable.jsx'
import KpiTiles from './components/KpiTiles.jsx'
import ModelConfig from './components/ModelConfig.jsx'
import Heatmap from './components/Heatmap.jsx'
import PortfolioComparison from './components/PortfolioComparison.jsx'

export default function App() {
  const [data, setData] = useState(null)
  const [portfolio, setPortfolio] = useState('A')
  const [metric, setMetric] = useState('cv')
  const [date, setDate] = useState('2025-08-04')

  useEffect(() => {
    fetch('/dashboard.json')
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData({ error: 'dashboard.json not found — run `make pipeline` first.' }))
  }, [])

  if (!data) return <div className="loading">Loading…</div>
  if (data.error) return <div className="error">{data.error}</div>

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="stripe" />
          <div>
            <h1>Synchrony Interval Forecast</h1>
            <p className="subtitle">{data.meta.forecast_month} · {data.meta.portfolios.length} portfolios · 30-min granularity</p>
          </div>
        </div>
        <div className="gen">Generated {new Date(data.meta.generated_at).toLocaleString()}</div>
      </header>

      <section className="controls">
        <PortfolioSelector
          portfolios={data.meta.portfolios}
          value={portfolio}
          onChange={setPortfolio}
        />
        <div className="metric-toggle">
          {['cv', 'cct', 'abd_rate'].map((m) => (
            <button key={m} className={m === metric ? 'active' : ''} onClick={() => setMetric(m)}>
              {labelFor(m)}
            </button>
          ))}
        </div>
        <input
          type="date"
          value={date}
          min="2025-08-01"
          max="2025-08-31"
          onChange={(e) => setDate(e.target.value)}
        />
      </section>

      <KpiTiles intervals={data.intervals} daily={data.daily} portfolio={portfolio} />

      <section className="grid">
        <div className="card wide">
          <h2>Intraday Forecast — {labelFor(metric)}, Portfolio {portfolio}, {date}</h2>
          <ForecastChart
            records={data.intervals}
            portfolio={portfolio}
            date={date}
            metric={metric}
          />
        </div>

        <div className="card wide">
          <h2>Portfolio Comparison — {labelFor(metric)} daily totals</h2>
          <PortfolioComparison daily={data.daily} metric={metric} />
        </div>

        <div className="card">
          <h2>Model Configuration — Portfolio {portfolio}</h2>
          <ModelConfig portfolio={portfolio} />
        </div>

        <div className="card">
          <h2>Daily Totals — Portfolio {portfolio}</h2>
          <DailyTotalsTable daily={data.daily} portfolio={portfolio} />
        </div>

        <div className="card wide">
          <h2>Day × Hour Heatmap — {labelFor(metric)}, Portfolio {portfolio}</h2>
          <Heatmap intervals={data.intervals} portfolio={portfolio} metric={metric} />
        </div>
      </section>

      <footer>
        Multi-stage forecasting pipeline · nager.at holiday enrichment · Vite + React dashboard
      </footer>
    </div>
  )
}

function labelFor(m) {
  return { cv: 'Call Volume', cct: 'Customer Care Time', abd_rate: 'Abandoned Rate' }[m] ?? m
}
