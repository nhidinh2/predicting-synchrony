export default function Heatmap({ intervals, portfolio, metric }) {
  const iv = intervals.filter((r) => r.portfolio === portfolio)
  if (iv.length === 0) return <div className="empty">No data.</div>

  const dates = [...new Set(iv.map((r) => r.date))].sort()
  const hours = Array.from({ length: 24 }, (_, h) => h)

  // sum counts, mean rates/times
  const aggMode = metric === 'cv' || metric === 'abd_calls' ? 'sum' : 'mean'
  const sums = new Map()
  const counts = new Map()
  for (const r of iv) {
    const h = Math.floor(r.interval_idx / 2)
    const key = `${r.date}|${h}`
    sums.set(key, (sums.get(key) || 0) + (r[metric] || 0))
    counts.set(key, (counts.get(key) || 0) + 1)
  }
  const grid = new Map()
  for (const [k, v] of sums.entries()) {
    grid.set(k, aggMode === 'sum' ? v : v / counts.get(k))
  }

  let vmin = Infinity, vmax = -Infinity
  for (const v of grid.values()) { if (v < vmin) vmin = v; if (v > vmax) vmax = v }
  if (vmin === vmax) vmax = vmin + 1

  return (
    <div className="heatmap-wrap">
      <div className="heatmap">
        <div className="heatmap-corner" />
        {hours.map((h) => (
          <div key={h} className="heatmap-colhead">{h}</div>
        ))}
        {dates.map((d) => (
          <Row key={d} date={d} hours={hours} grid={grid} vmin={vmin} vmax={vmax} metric={metric} />
        ))}
      </div>
      <div className="heatmap-legend">
        <span>{fmt(vmin, metric)}</span>
        <div className="scale" />
        <span>{fmt(vmax, metric)}</span>
      </div>
    </div>
  )
}

function Row({ date, hours, grid, vmin, vmax, metric }) {
  return (
    <>
      <div className="heatmap-rowhead">{date.slice(5)}</div>
      {hours.map((h) => {
        const v = grid.get(`${date}|${h}`) ?? 0
        const t = (v - vmin) / (vmax - vmin)
        return (
          <div
            key={h}
            className="heatmap-cell"
            style={{ background: color(t) }}
            title={`${date} ${h}:00 — ${fmt(v, metric)}`}
          />
        )
      })}
    </>
  )
}

function color(t) {
  t = Math.max(0, Math.min(1, t))
  const r = Math.round(15 + (232 - 15) * t)
  const g = Math.round(27 + (116 - 27) * t)
  const b = Math.round(48 + (44 - 48) * t)
  return `rgb(${r}, ${g}, ${b})`
}

function fmt(v, metric) {
  if (v == null) return '—'
  if (metric === 'abd_rate') return (v * 100).toFixed(2) + '%'
  if (metric === 'cct') return v.toFixed(1) + 's'
  return Math.round(v).toLocaleString()
}
