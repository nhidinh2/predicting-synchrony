import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const COLORS = { A: '#E8742C', B: '#6fa8dc', C: '#93c47d', D: '#c27ba0' }

export default function PortfolioComparison({ daily, metric }) {
  const key = { cv: 'cv_daily', cct: 'cct_daily', abd_rate: 'abd_rate_daily' }[metric] ?? 'cv_daily'
  const dates = [...new Set(daily.map((r) => r.date))].sort()

  const merged = dates.map((d) => {
    const row = { date: d.slice(5) }
    for (const p of ['A', 'B', 'C', 'D']) {
      const match = daily.find((r) => r.date === d && r.portfolio === p)
      row[p] = match ? match[key] : null
    }
    return row
  })

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={merged} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid stroke="#22304a" strokeDasharray="3 3" />
        <XAxis dataKey="date" stroke="#9fb0cc" />
        <YAxis stroke="#9fb0cc" />
        <Tooltip contentStyle={{ background: '#0f1b30', border: '1px solid #22304a' }} />
        <Legend wrapperStyle={{ color: '#9fb0cc' }} />
        {['A', 'B', 'C', 'D'].map((p) => (
          <Line key={p} type="monotone" dataKey={p} stroke={COLORS[p]} strokeWidth={2} dot={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
