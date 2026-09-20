import { MotionFigure } from "@/components/figures/MotionFigure";
import Link from "next/link";
import { CurtainGate } from "@/components/home/CurtainGate";
import { ResearchList } from "@/components/research/ResearchList";
import { SignalHeading } from "@/components/typography/SignalHeading";
import styles from "./page.module.css";

function Arrow() { return <svg width="22" height="18" viewBox="0 0 22 18" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M1 9h19M13 2l7 7-7 7" /></svg>; }

export default function Home() {
  return <>
    <CurtainGate />
    <section className={`section ${styles.hero}`} aria-labelledby="home-title"><div className={`container ${styles.heroInner}`}><SignalHeading as="h1" id="home-title" className={styles.heroTitle}><span>fuel</span><span className={styles.heroLabs}>labs</span></SignalHeading><p className={styles.heroCopy}>An independent AI research lab.</p><div className={styles.heroLower}><span>fuellabs.</span><Link className="arrow-link" href="/hobbylm">Discover the work <Arrow /></Link></div></div></section>
    <section className="section section--ink" aria-labelledby="model-title"><div className={`container ${styles.modelGrid}`}><div><span className="eyebrow">Our model / 01</span><SignalHeading as="h2" id="model-title">HobbyLM</SignalHeading><p className={styles.modelLead}>An open, compact language model built from scratch for instruction following, tool use, and small-model research.</p><dl className={styles.facts}><div><dt>Architecture</dt><dd>Sparse mixture of experts</dd></div><div><dt>Foundation</dt><dd>Trained from scratch</dd></div></dl><Link className={`arrow-link ${styles.limeLink}`} href="/hobbylm">Meet HobbyLM <Arrow /></Link></div><div className={styles.figure}><MotionFigure animatedSrc="/diagrams/sparse-routing.gif" stillSrc="/diagrams/sparse-routing-still.png" alt="Conceptual animation of a token being routed through selected experts and recombined." caption="Conceptual sparse-routing illustration." /></div></div></section>
    <section className="section" aria-labelledby="research-title"><div className="container"><div className={styles.researchHead}><div><span className="eyebrow">Selected research / 02</span><SignalHeading as="h2" id="research-title">Work in progress</SignalHeading></div><p>Technical accounts of how HobbyLM is built, trained, and evaluated. Publication follows evidence and review.</p></div><ResearchList preview /></div></section>
  </>;
}
