export default function KpiTiles({ intervals, daily, portfolio }) {
  const iv = intervals.filter((r) => r.portfolio === portfolio)
  const dl = daily.filter((r) => r.portfolio === portfolio)

  const totalCV = iv.reduce((a, r) => a + (r.cv || 0), 0)
  const avgCCT = iv.length ? iv.reduce((a, r) => a + (r.cct || 0), 0) / iv.length : 0
  const totalAbd = iv.reduce((a, r) => a + (r.abd_calls || 0), 0)
  const abdRate = totalCV > 0 ? totalAbd / totalCV : 0

  let peak = { cv: -Infinity, time: '', date: '' }
  for (const r of iv) {
    if ((r.cv || 0) > peak.cv) peak = { cv: r.cv, time: idxToTime(r.interval_idx), date: r.date }
  }

  const peakDay = dl.reduce((best, r) => (r.cv_daily > (best?.cv_daily || 0) ? r : best), null)

  return (
    <div className="kpi-row">
      <Tile label="Total Call Volume" value={fmt0(totalCV)} sub={`Portfolio ${portfolio}, Aug 2025`} />
      <Tile label="Peak Interval" value={peak.time} sub={`${fmt0(peak.cv)} calls on ${peak.date}`} />
      <Tile label="Avg CCT" value={`${avgCCT.toFixed(1)}s`} sub="mean across all intervals" />
      <Tile label="Abandoned Rate" value={`${(abdRate * 100).toFixed(2)}%`} sub={`${fmt0(totalAbd)} abandoned`} />
      <Tile label="Busiest Day" value={peakDay?.date?.slice(5) ?? '—'} sub={`${fmt0(peakDay?.cv_daily)} daily calls`} />
    </div>
  )
}

function Tile({ label, value, sub }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      <div className="kpi-sub">{sub}</div>
    </div>
  )
}

function fmt0(v) {
  if (v == null || Number.isNaN(v)) return '—'
  return Math.round(v).toLocaleString()
}

function idxToTime(i) {
  const h = Math.floor(i / 2)
  const m = i % 2 === 0 ? '00' : '30'
  return `${h}:${m}`
}
