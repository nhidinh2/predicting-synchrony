export default function DailyTotalsTable({ daily, portfolio }) {
  const rows = daily.filter((r) => r.portfolio === portfolio).slice(0, 12)
  return (
    <table className="totals">
      <thead>
        <tr><th>Date</th><th>CV</th><th>CCT</th><th>Abd calls</th><th>Abd rate</th></tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.date}>
            <td>{r.date}</td>
            <td>{fmt(r.cv_daily, 0)}</td>
            <td>{fmt(r.cct_daily, 1)}</td>
            <td>{fmt(r.abd_calls_daily, 0)}</td>
            <td>{fmt(r.abd_rate_daily, 4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function fmt(v, d) {
  if (v == null || Number.isNaN(v)) return '—'
  return Number(v).toFixed(d)
}
