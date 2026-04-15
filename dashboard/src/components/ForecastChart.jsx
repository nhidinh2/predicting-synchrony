import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function ForecastChart({ records, portfolio, date, metric }) {
  const rows = records
    .filter((r) => r.portfolio === portfolio && r.date === date)
    .sort((a, b) => a.interval_idx - b.interval_idx)
    .map((r) => ({
      time: idxToTime(r.interval_idx),
      value: r[metric],
    }))

  if (rows.length === 0) return <div className="empty">No data for this selection.</div>

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid stroke="#22304a" strokeDasharray="3 3" />
        <XAxis dataKey="time" stroke="#9fb0cc" interval={5} />
        <YAxis stroke="#9fb0cc" />
        <Tooltip contentStyle={{ background: '#0f1b30', border: '1px solid #22304a' }} />
        <Line type="monotone" dataKey="value" stroke="#E8742C" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

function idxToTime(i) {
  const h = Math.floor(i / 2)
  const m = i % 2 === 0 ? '00' : '30'
  return `${h}:${m}`
}
