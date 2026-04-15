const CV_ALPHA = { A: 0.58, B: 0.60, C: 0.55, D: 0.62 }
const BLEND = { A: 0.75, B: 0.65, C: 0.95, D: 0.85 }

export default function ModelConfig({ portfolio }) {
  return (
    <div className="config">
      <Row label="Stage-1 CV loss" value={`reg:quantileerror @ α = ${CV_ALPHA[portfolio]}`} />
      <Row label="Stage-1 CCT loss" value="reg:quantileerror @ α = 0.57" />
      <Row label="Stage-1 ABD loss" value="count:poisson (+5% shift)" />
      <Row label="Asymmetric penalty" value="under × 1.5  /  over × 1.0" />
      <Row label="ABD blend weight" value={`${BLEND[portfolio]} × XGB  +  ${(1 - BLEND[portfolio]).toFixed(2)} × naive`} />
      <Row label="Stage-2 shape" value="DOW × 48 half-hour profile (Apr–Jun 2025)" />
      <Row label="Calibration" value="isotonic on 2024-08 holdout" />
      <Row label="External enrichment" value="nager.at US holidays (cached)" />
    </div>
  )
}

function Row({ label, value }) {
  return (
    <div className="config-row">
      <span className="config-label">{label}</span>
      <span className="config-value">{value}</span>
    </div>
  )
}
