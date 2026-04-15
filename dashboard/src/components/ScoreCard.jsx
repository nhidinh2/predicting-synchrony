export default function ScoreCard({ scores }) {
  if (!scores) return <div className="empty">—</div>
  const entries = Object.entries(scores).filter(([, v]) => typeof v === 'number')
  if (entries.length === 0) {
    return <div className="empty"><p>{scores.note ?? 'No scores recorded.'}</p></div>
  }
  return (
    <ul className="score-list">
      {entries.map(([k, v]) => (
        <li key={k}>
          <span className="metric">{k.toUpperCase()}</span>
          <span className="value">{v.toFixed(4)}</span>
        </li>
      ))}
    </ul>
  )
}
