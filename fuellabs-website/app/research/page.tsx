import type { Metadata } from "next";
import { ResearchList } from "@/components/research/ResearchList";
import { SignalHeading } from "@/components/typography/SignalHeading";
import styles from "./page.module.css";

export const metadata: Metadata = { title: "Research" };

export default function ResearchPage() {
  return <section className={`section ${styles.page}`}><div className="container"><span className="eyebrow">From the lab</span><SignalHeading as="h1" className="display">Research</SignalHeading><p className={`intro-copy ${styles.intro}`}>Technical accounts of how HobbyLM is built, trained, and evaluated. Articles publish when their evidence and review are complete.</p><ResearchList /></div></section>;
}
