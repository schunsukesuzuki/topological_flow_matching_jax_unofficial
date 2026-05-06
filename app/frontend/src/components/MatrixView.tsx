type Props = { title: string; matrix: number[][]; maxRows?: number; maxCols?: number };

export function MatrixView({ title, matrix, maxRows = 10, maxCols = 10 }: Props) {
  const rows = matrix.slice(0, maxRows);
  const originalRows = matrix.length;
  const originalCols = matrix[0]?.length ?? 0;
  return (
    <div className="card wide matrix-card">
      <h3>{title}</h3>
      <p className="muted">
        shape: {originalRows} × {originalCols}
        {originalRows > maxRows || originalCols > maxCols ? " / preview only" : ""}
      </p>
      <div className="table-scroll">
        <table className="matrix">
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                <th>{i}</th>
                {row.slice(0, maxCols).map((v, j) => <td key={j}>{Number(v).toFixed(1)}</td>)}
                {originalCols > maxCols && <td>…</td>}
              </tr>
            ))}
            {originalRows > maxRows && (
              <tr>
                <th>…</th>
                <td colSpan={Math.min(originalCols, maxCols) + 1}>truncated</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
