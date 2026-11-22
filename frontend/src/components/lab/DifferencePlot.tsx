import { signed } from "@/lib/format";

export interface Difference {
  label: string;
  mean: number;
  low: number;
  high: number;
  better: number;
  worse: number;
}

/**
 * Paired-bootstrap differences in MRR@10 with 95% intervals on one shared axis around zero.
 * Single series, so no legend; each row is direct-labelled. Native <title> gives hover text,
 * and the caller renders the same numbers as a table.
 */
export function DifferencePlot({ rows }: { rows: Difference[] }) {
  const W = 640;
  const left = 150;
  const right = 24;
  const rowH = 52;
  const top = 18;
  const H = top + rows.length * rowH + 34;
  const extent = Math.max(0.02, ...rows.flatMap((r) => [Math.abs(r.low), Math.abs(r.high)]));
  const max = Math.ceil(extent * 100) / 100;
  const x = (v: number) => left + ((v + max) / (2 * max)) * (W - left - right);
  const ticks = [-max, -max / 2, 0, max / 2, max];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-labelledby="diff-title diff-desc">
      <title id="diff-title">Differences in proxy MRR@10 with 95% bootstrap intervals</title>
      <desc id="diff-desc">
        {rows.map((r) => `${r.label}: ${signed(r.mean, 4)} (interval ${signed(r.low, 4)} to ${signed(r.high, 4)})`).join("; ")}
      </desc>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top - 6} y2={H - 30} stroke="var(--line)" strokeDasharray={t === 0 ? undefined : "2 4"} />
          <text x={x(t)} y={H - 12} textAnchor="middle" className="fill-fg-3 num" fontSize="12">
            {t === 0 ? "0" : signed(t, 2)}
          </text>
        </g>
      ))}
      <line x1={x(0)} x2={x(0)} y1={top - 6} y2={H - 30} stroke="var(--fg-3)" strokeWidth="1.25" />
      {rows.map((r, i) => {
        const cy = top + i * rowH + rowH / 2;
        return (
          <g key={r.label}>
            <title>{`${r.label}: ${signed(r.mean, 4)}, 95% interval ${signed(r.low, 4)} to ${signed(r.high, 4)}; ${r.better} queries better, ${r.worse} worse`}</title>
            <rect x={0} y={cy - rowH / 2} width={W} height={rowH} fill="transparent" />
            <text x={0} y={cy + 4} className="fill-fg" fontSize="14">
              {r.label}
            </text>
            <line x1={x(r.low)} x2={x(r.high)} y1={cy} y2={cy} stroke="var(--series-c)" strokeWidth="2" strokeLinecap="round" />
            <line x1={x(r.low)} x2={x(r.low)} y1={cy - 6} y2={cy + 6} stroke="var(--series-c)" strokeWidth="2" strokeLinecap="round" />
            <line x1={x(r.high)} x2={x(r.high)} y1={cy - 6} y2={cy + 6} stroke="var(--series-c)" strokeWidth="2" strokeLinecap="round" />
            <circle cx={x(r.mean)} cy={cy} r="5.5" fill="var(--series-c)" stroke="var(--surface)" strokeWidth="2" />
          </g>
        );
      })}
    </svg>
  );
}
