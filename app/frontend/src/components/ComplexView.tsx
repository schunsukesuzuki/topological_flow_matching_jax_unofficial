type Props = {
  title: string;
  nodes: number[][];
  edges: number[][];
  faces: number[][];
  signal: number[];
};

function scale([x, y]: number[]) {
  return [165 + x * 120, 160 + y * 120];
}

function edgeWidth(v: number) {
  return 1.2 + Math.min(8, Math.abs(v) * 4.5);
}

function edgeClass(v: number) {
  if (v > 0.05) return "edge positive";
  if (v < -0.05) return "edge negative";
  return "edge neutral";
}

export function ComplexView({ title, nodes, edges, faces, signal }: Props) {
  return (
    <div className="card">
      <h3>{title}</h3>
      <svg viewBox="0 0 330 320" className="complex-svg">
        {faces.map((face, i) => (
          <polygon key={i} points={face.map((idx) => scale(nodes[idx]).join(",")).join(" ")} className="face" />
        ))}
        {edges.map(([a, b], i) => {
          const [x1, y1] = scale(nodes[a]);
          const [x2, y2] = scale(nodes[b]);
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} className={edgeClass(signal[i])} strokeWidth={edgeWidth(signal[i])} />;
        })}
        {nodes.map((p, i) => {
          const [x, y] = scale(p);
          return <circle key={i} cx={x} cy={y} r="3.5" className="node" />;
        })}
      </svg>
      <p className="muted">{edges.length} edge signals on an annulus-shaped 2-simplicial complex.</p>
    </div>
  );
}
