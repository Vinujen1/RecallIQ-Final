export interface Series {
  name: string;
  color: string; // CSS colour token
  dashed?: boolean;
  marker: "circle" | "square";
  points: { epoch: number; value: number }[];
}

/**
 * Proxy MRR@10 by epoch (epoch 0 = before fine-tuning) for two runs on one axis.
 * Identity is never colour alone: legend, direct end labels, marker shape and dash.
 */
export function TrainingCurve({ series, best }: { series: Series[]; best: Record<string, number> }) {
  const W = 640;
  const H = 320;
  const pad = { l: 56, r: 92, t: 16, b: 64 };
  const all = series.flatMap((s) => s.points);
  const maxEpoch = Math.max(...all.map((p) => p.epoch));
  const lo = Math.floor(Math.min(...all.map((p) => p.value)) * 100) / 100;
  const hi = Math.ceil(Math.max(...all.map((p) => p.value)) * 100) / 100;
  const x = (e: number) => pad.l + (e / maxEpoch) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b);
  const ticks = Array.from({ length: Math.round((hi - lo) / 0.01) + 1 }, (_, i) => lo + i * 0.01);

  return (
    <figure>
      <ul className="mb-3 flex flex-wrap gap-x-6 gap-y-1 text-sm text-fg-2" aria-hidden>
        {series.map((s) => (
          <li key={s.name} className="flex items-center gap-2">
            <svg width="28" height="10" aria-hidden>
              <line x1="0" x2="28" y1="5" y2="5" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? "5 4" : undefined} />
            </svg>
            {s.name}
          </li>
        ))}
      </ul>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Proxy MRR@10 by training epoch for Models B and C. Values are listed in the table below.">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="var(--line)" />
            <text x={pad.l - 10} y={y(t) + 4} textAnchor="end" fontSize="12" className="fill-fg-3 num">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {Array.from({ length: maxEpoch + 1 }, (_, e) => (
          <text key={e} x={x(e)} y={H - pad.b + 22} textAnchor="middle" fontSize="12" className="fill-fg-3 num">
            {e}
          </text>
        ))}
        <text x={(pad.l + W - pad.r) / 2} y={H - 6} textAnchor="middle" fontSize="12" className="fill-fg-3">
          Epoch (0 = pretrained, before fine-tuning)
        </text>
        {series.map((s) => {
          const d = s.points.map((p, i) => `${i ? "L" : "M"}${x(p.epoch)},${y(p.value)}`).join(" ");
          const last = s.points[s.points.length - 1];
          return (
            <g key={s.name}>
              <path d={d} fill="none" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? "6 5" : undefined} strokeLinejoin="round" />
              {s.points.map((p) => {
                const isBest = best[s.name] === p.epoch;
                const r = isBest ? 6 : 4.5;
                return (
                  <g key={p.epoch}>
                    <title>{`${s.name}, epoch ${p.epoch}: ${p.value.toFixed(4)}${isBest ? " (selected checkpoint)" : ""}`}</title>
                    {s.marker === "circle" ? (
                      <circle cx={x(p.epoch)} cy={y(p.value)} r={r} fill={isBest ? s.color : "var(--surface)"} stroke={s.color} strokeWidth="2" />
                    ) : (
                      <rect x={x(p.epoch) - r} y={y(p.value) - r} width={r * 2} height={r * 2} rx="1.5" fill={isBest ? s.color : "var(--surface)"} stroke={s.color} strokeWidth="2" />
                    )}
                  </g>
                );
              })}
              <text x={x(last.epoch) + 12} y={y(last.value) + 4} fontSize="13" className="fill-fg">
                {s.name}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="mt-2 text-sm text-fg-3">Filled marker: the checkpoint selected for serving.</figcaption>
    </figure>
  );
}
