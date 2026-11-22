import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono, Manrope } from "next/font/google";
import { SiteFooter } from "@/components/layout/SiteFooter";
import { HeaderView, SiteHeader } from "@/components/layout/SiteHeader";
import { THEME_SCRIPT } from "@/components/layout/ThemeToggle";
import { Suspense } from "react";
import "./globals.css";

const manrope = Manrope({ variable: "--font-manrope", subsets: ["latin"], display: "swap" });
const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "RecallIQ: Canadian product recall search", template: "%s | RecallIQ" },
  description:
    "Explore Canadian product recalls using everyday descriptions, with evidence from official government notices.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#101112" },
    { media: "(prefers-color-scheme: light)", color: "#f8f4ea" },
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      data-theme="dark"
      suppressHydrationWarning
      className={`${manrope.variable} ${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body className="flex min-h-full flex-col">
        <a
          href="#main"
          className="sr-only z-50 rounded-md bg-accent-fill px-4 py-2 text-on-accent focus:not-sr-only focus:fixed focus:left-4 focus:top-3"
        >
          Skip to content
        </a>
        <Suspense fallback={<HeaderView pathname="" />}>
          <SiteHeader />
        </Suspense>
        <main id="main" tabIndex={-1} className="flex flex-1 flex-col outline-none">
          {children}
        </main>
        <SiteFooter />
      </body>
    </html>
  );
}
