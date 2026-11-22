/**
 * Hero illustration: a gold scan line passes over an example product label; as it crosses
 * the UPC and lot codes they light up and a connector draws to an official-notice card
 * marked "Code in notice". Orbital guides turn slowly behind it.
 * Pure SVG + CSS (see .scan-* in globals.css); static final frame under reduced motion.
 * The label is an illustration with placeholder codes, not a real product.
 */
export function LabelScanVisual() {
  const bars = [3, 1, 2, 1, 3, 2, 1, 1, 2, 3, 1, 2, 1, 1, 3, 1, 2, 2, 1, 3, 1, 1, 2, 1, 3, 2, 1, 2];
  let x = 102;
  const barcode = bars.map((w, i) => {
    const rect = i % 2 ? null : <rect key={i} x={x} y={286} width={w * 1.6} height={44} rx={0.4} />;
    x += w * 1.6 + 1.6;
    return rect;
  });

  return (
    <figure className="scan relative mx-auto w-full max-w-[560px]">
      <svg viewBox="0 0 560 520" className="h-auto w-full overflow-visible" aria-hidden="true" focusable="false">
        <defs>
          <radialGradient id="scan-light" cx="50%" cy="48%" r="50%">
            <stop offset="0%" stopColor="var(--accent-fill)" stopOpacity="0.22" />
            <stop offset="55%" stopColor="var(--accent-fill)" stopOpacity="0.05" />
            <stop offset="100%" stopColor="var(--accent-fill)" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="scan-beam" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent-fill)" stopOpacity="0" />
            <stop offset="100%" stopColor="var(--accent-fill)" stopOpacity="0.28" />
          </linearGradient>
          <linearGradient id="scan-plate" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="var(--raised)" />
            <stop offset="100%" stopColor="var(--surface)" />
          </linearGradient>
          <clipPath id="scan-clip">
            <rect x="70" y="70" width="300" height="370" rx="18" />
          </clipPath>
        </defs>

        <ellipse cx="280" cy="260" rx="270" ry="250" fill="url(#scan-light)" />

        {/* Orbital guides */}
        <g className="scan-orbits" fill="none" stroke="var(--accent-fill)">
          <ellipse cx="280" cy="260" rx="258" ry="92" strokeOpacity="0.28" transform="rotate(-18 280 260)" />
          <ellipse cx="280" cy="260" rx="222" ry="200" strokeOpacity="0.12" strokeDasharray="2 7" />
          <ellipse cx="280" cy="260" rx="246" ry="64" strokeOpacity="0.2" transform="rotate(24 280 260)" />
          <circle cx="58" cy="208" r="3" fill="var(--accent-fill)" fillOpacity="0.8" stroke="none" />
          <circle cx="512" cy="330" r="2.5" fill="var(--accent-fill)" fillOpacity="0.6" stroke="none" />
          <circle cx="430" cy="96" r="2" fill="var(--accent-fill)" fillOpacity="0.5" stroke="none" />
        </g>

        {/* Example product label */}
        <g>
          <rect x="70" y="70" width="300" height="370" rx="18" fill="url(#scan-plate)" stroke="var(--line-strong)" />
          <rect x="70.5" y="70.5" width="299" height="369" rx="17.5" fill="none" stroke="var(--accent-fill)" strokeOpacity="0.25" />
          <text x="102" y="122" className="scan-label-sm" fill="var(--fg-3)">EXAMPLE LABEL</text>
          <text x="102" y="160" className="scan-label-lg" fill="var(--fg)">Granola Bites</text>
          <text x="102" y="186" className="scan-label-md" fill="var(--fg-2)">Honey &amp; oat, 350 g</text>
          <line x1="102" y1="214" x2="338" y2="214" stroke="var(--line-strong)" />
          <text x="102" y="244" className="scan-label-sm" fill="var(--fg-3)">INGREDIENTS: ROLLED OATS, HONEY,</text>
          <text x="102" y="262" className="scan-label-sm" fill="var(--fg-3)">SUNFLOWER OIL, SESAME</text>

          <g fill="var(--fg-2)">{barcode}</g>

          <g className="scan-code scan-code-upc">
            <rect x="96" y="340" width="190" height="26" rx="6" className="scan-code-bg" />
            <text x="104" y="358" className="scan-mono">UPC 0 12345 67890 5</text>
          </g>
          <g className="scan-code scan-code-lot">
            <rect x="96" y="372" width="214" height="26" rx="6" className="scan-code-bg" />
            <text x="104" y="390" className="scan-mono">LOT 24A119  BB 2026 MR 14</text>
          </g>

          {/* Scan line */}
          <g clipPath="url(#scan-clip)">
            <g className="scan-line">
              <rect x="70" y="40" width="300" height="60" fill="url(#scan-beam)" />
              <line x1="70" y1="100" x2="370" y2="100" stroke="var(--accent-fill)" strokeWidth="1.5" />
            </g>
          </g>
        </g>

        {/* Connector and notice card */}
        <path className="scan-connector" d="M286 353 C 330 353, 330 300, 372 300" fill="none" stroke="var(--accent-fill)" strokeWidth="1.25" pathLength="1" />
        <g className="scan-notice">
          <rect x="372" y="228" width="168" height="150" rx="14" fill="var(--raised)" stroke="var(--line-strong)" />
          <text x="390" y="258" className="scan-label-sm" fill="var(--fg-3)">OFFICIAL NOTICE</text>
          <rect x="390" y="272" width="122" height="7" rx="3.5" fill="var(--fg-3)" fillOpacity="0.45" />
          <rect x="390" y="287" width="96" height="7" rx="3.5" fill="var(--fg-3)" fillOpacity="0.3" />
          <rect x="390" y="302" width="110" height="7" rx="3.5" fill="var(--fg-3)" fillOpacity="0.3" />
          <g className="scan-badge">
            <rect x="390" y="330" width="124" height="28" rx="6" fill="var(--alert-tint)" stroke="var(--alert)" strokeOpacity="0.6" />
            <path d="M402 351 l7 -12 l7 12 z M409 344.5 v2.5" fill="none" stroke="var(--alert)" strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round" />
            <text x="423" y="349" className="scan-badge-text" fill="var(--alert)">Code in notice</text>
          </g>
        </g>
      </svg>
      <figcaption className="sr-only">Illustration of an example product label being scanned; the codes are placeholders.</figcaption>
    </figure>
  );
}
