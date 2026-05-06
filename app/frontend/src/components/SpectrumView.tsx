type Props = { eigenvalues: number[] };

export function SpectrumView({ eigenvalues }: Props) {
  const max = Math.max(...eigenvalues.map(Math.abs), 1e-6);
  const preview = eigenvalues.slice(0, 36);
  return (
    <div className="card wide spectrum-card">
      <h3>L1 spectrum</h3>
      <p className="muted">{eigenvalues.length} eigenvalues. First {preview.length} shown.</p>
      <div className="bars scroll-bars">
        {preview.map((v, i) => {
          const h = 10 + 120 * Math.abs(v) / max;
          const zero = Math.abs(v) < 1e-5;
          return <div className="bar-wrap" key={i}><div className={`bar ${zero ? "zero" : ""}`} style={{ height: h }} /><span>{v.toFixed(2)}</span></div>;
        })}
      </div>
    </div>
  );
}
