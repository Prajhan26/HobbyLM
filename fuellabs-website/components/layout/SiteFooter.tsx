import Link from "next/link";
import { Wordmark } from "@/components/brand/Wordmark";
import styles from "./SiteFooter.module.css";

export function SiteFooter() {
  return <footer className={styles.footer}><div className={styles.inner}><div className={styles.brand}><Wordmark large /></div><nav className={styles.links} aria-label="Footer navigation"><Link href="/hobbylm">HobbyLM</Link><Link href="/research">Research</Link><Link href="/team">Team</Link></nav><div className={styles.note}><span>Independent AI research.</span><span>© {new Date().getFullYear()} fuellabs.</span></div></div></footer>;
}
