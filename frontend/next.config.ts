import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Cache Components: pages prerender a static shell and stream dynamic parts (the recall
  // page relies on this; see src/proxy.ts for how missing ids still return a real 404).
  cacheComponents: true,
  partialPrefetching: true,
  // Tailwind CSS 4 runs as a Turbopack loader instead of a PostCSS config file.
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
