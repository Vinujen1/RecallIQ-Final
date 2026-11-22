import type { SVGProps } from "react";

const PATHS = {
  search: "M10.5 18a7.5 7.5 0 1 1 0-15 7.5 7.5 0 0 1 0 15Zm5.3-2.2L21 21",
  external: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  arrowLeft: "M19 12H5m6-6-6 6 6 6",
  alert: "M12 3 2.5 20h19L12 3Zm0 6.5v5m0 3v.01",
  cross: "M6 6l12 12M18 6 6 18",
  dash: "M6 12h12",
  check: "m5 12.5 4.5 4.5L19 7",
  filter: "M4 6h16M7 12h10M10 18h4",
  menu: "M4 7h16M4 12h16M4 17h16",
  sun: "M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0-13v2m0 14v2M3 12h2m14 0h2M5.6 5.6 7 7m10 10 1.4 1.4M5.6 18.4 7 17m10-10 1.4-1.4",
  moon: "M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z",
  monitor: "M3 5h18v11H3zM9 20h6m-3-4v4",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Zm0-10v5m0-8v.01",
  refresh: "M20 11a8 8 0 0 0-14.6-4.5L4 8m0-4v4h4m-4 5a8 8 0 0 0 14.6 4.5L20 16m0 4v-4h-4",
} as const;

export type IconName = keyof typeof PATHS;

/** 1.5px-stroke line icons. Decorative by default; pass `label` when the icon carries meaning alone. */
export function Icon({ name, label, size = 18, ...rest }: { name: IconName; label?: string; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      focusable="false"
      {...rest}
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
