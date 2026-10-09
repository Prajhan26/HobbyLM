import type { Metadata } from "next";
import localFont from "next/font/local";
import { SiteFooter } from "@/components/layout/SiteFooter";
import { SiteHeader } from "@/components/layout/SiteHeader";
import "./globals.css";

const siteUrl = "https://fuellabs.in";
const allowIndexing = process.env.VERCEL_ENV === "production";
const socialImage = {
  url: "/social/fuellabs-share-v1.jpg",
  width: 1200,
  height: 630,
  alt: "HobbyLM sparse architecture visual in the Fuellabs lime signal palette",
};

const generalSans = localFont({
  src: [
    { path: "./fonts/general-sans-400.woff2", weight: "400", style: "normal" },
    { path: "./fonts/general-sans-500.woff2", weight: "500", style: "normal" },
  ],
  display: "swap",
  variable: "--general-sans",
});
const montserrat = localFont({
  src: [
    { path: "./fonts/montserrat-400.ttf", weight: "400", style: "normal" },
    { path: "./fonts/montserrat-500.ttf", weight: "500", style: "normal" },
  ],
  display: "swap",
  variable: "--montserrat",
});
const plexSans = localFont({ src: "./fonts/ibm-plex-sans-500.ttf", weight: "500", display: "swap", variable: "--plex-sans" });
const plexMono = localFont({
  src: [
    { path: "./fonts/ibm-plex-mono-400.ttf", weight: "400", style: "normal" },
    { path: "./fonts/ibm-plex-mono-500.ttf", weight: "500", style: "normal" },
  ],
  display: "swap",
  variable: "--plex-mono",
});

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: { default: "fuellabs. — Independent AI research lab", template: "%s — fuellabs." },
  description: "An independent AI research lab building HobbyLM from scratch.",
  icons: {
    icon: [{ url: "/favicon-lime-v2.svg", type: "image/svg+xml" }],
    shortcut: "/favicon-lime-v2.svg",
  },
  openGraph: {
    title: "fuellabs.",
    description: "An independent AI research lab building HobbyLM from scratch.",
    url: siteUrl,
    siteName: "fuellabs.",
    images: [socialImage],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "fuellabs.",
    description: "An independent AI research lab building HobbyLM from scratch.",
    images: [{ url: socialImage.url, alt: socialImage.alt }],
  },
  robots: { index: allowIndexing, follow: allowIndexing },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${generalSans.variable} ${montserrat.variable} ${plexSans.variable} ${plexMono.variable}`}>
      <body>
        <a className="skip-link" href="#main">Skip to content</a>
        <SiteHeader />
        <main id="main" tabIndex={-1}>{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}
