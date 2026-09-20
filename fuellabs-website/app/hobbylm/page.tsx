import type { Metadata } from "next";
import { MotionFigure } from "@/components/figures/MotionFigure";
import { SignalHeading } from "@/components/typography/SignalHeading";
import styles from "./page.module.css";

export const metadata: Metadata = { title: "HobbyLM" };

export default function HobbyLMPage() {
  return <>
    <section className={`section ${styles.hero}`}><div className={`container ${styles.heroGrid}`}><div><span className="eyebrow">Our model</span><SignalHeading as="h1" className="display">HobbyLM</SignalHeading><p className="intro-copy">An open, compact language model built from scratch for instruction following, tool use, and small-model research.</p></div><dl className={styles.summary}><div><dt>Architecture</dt><dd>Sparse mixture of experts</dd></div><div><dt>Foundation</dt><dd>Trained from scratch</dd></div><div><dt>Release</dt><dd>Links and claims publish after approval</dd></div></dl></div></section>
    <section className="section section--ink" aria-labelledby="architecture-title"><div className={`container ${styles.chapter}`}><div><span className="eyebrow">Architecture / 01</span><SignalHeading as="h2" id="architecture-title">Selective routing</SignalHeading><p className="intro-copy">A conceptual view of tokens moving through selected experts before their outputs are recombined.</p></div><MotionFigure animatedSrc="/diagrams/sparse-routing.gif" stillSrc="/diagrams/sparse-routing-still.png" alt="Conceptual sparse routing diagram with selected experts highlighted in lime." caption="Conceptual illustration. Final technical corrections remain subject to model review." /></div></section>
    <section className="section" aria-labelledby="evidence-title"><div className={`container ${styles.evidence}`}><div><span className="eyebrow">Evidence / 02</span><SignalHeading as="h2" id="evidence-title">Context before claims</SignalHeading></div><p className="intro-copy">Architecture, training, post-training, and evaluation materials will appear here as evidence is approved. GitHub and Hugging Face links will be added with the release.</p></div></section>
  </>;
}
