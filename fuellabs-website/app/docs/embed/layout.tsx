import type { Metadata } from "next";
import localFont from "next/font/local";
import { DocsShell } from "@/components/docs/DocsShell";
const inter = localFont({
  src: "../../fonts/inter-variable.ttf",
  display: "swap",
  variable: "--font-docs",
});
export const metadata: Metadata = { title: "HobbyLM-Embed docs", description: "Documentation for HobbyLM-Embed from Fuel Labs." };
export default function EmbedDocsLayout({ children }: { children: React.ReactNode }) {
  return <div className={inter.variable}><DocsShell>{children}</DocsShell></div>;
}
