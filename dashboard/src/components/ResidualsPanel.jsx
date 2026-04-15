import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

export default function ResidualsPanel({ residuals, portfolio }) {
  const rows = (residuals ?? []).filter((r) => r.portfolio === portfolio)
  if (rows.length === 0) {
    return <div className="empty">No holdout residuals exported — run <code>make backtest</code>.</div>
  }
  return (
    <ResponsiveContainer width="100%" height={220}>
      <ScatterChart>
        <CartesianGrid stroke="#22304a" strokeDasharray="3 3" />
        <XAxis dataKey="predicted" name="predicted" stroke="#9fb0cc" />
        <YAxis dataKey="residual" name="residual" stroke="#9fb0cc" />
        <ReferenceLine y={0} stroke="#E8742C" strokeDasharray="4 4" />
        <Tooltip contentStyle={{ background: '#0f1b30', border: '1px solid #22304a' }} />
        <Scatter data={rows} fill="#6fa8dc" />
      </ScatterChart>
    </ResponsiveContainer>
  )
}
