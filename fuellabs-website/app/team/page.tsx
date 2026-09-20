import type { Metadata } from "next";
import { Wordmark } from "@/components/brand/Wordmark";
import { FounderSelector } from "@/components/team/FounderSelector";
import { SignalHeading } from "@/components/typography/SignalHeading";
import styles from "./page.module.css";

export const metadata: Metadata = { title: "Team" };

export default function TeamPage() {
  return <section className={`section ${styles.team}`}><div className="container"><header className={styles.heading}><span className={styles.brand}>fuel<span>labs</span><i aria-hidden="true" /></span><SignalHeading as="h1">Team</SignalHeading></header><FounderSelector /><footer className={styles.note}><p>Two people building language models from first principles.</p><Wordmark /></footer></div></section>;
}
