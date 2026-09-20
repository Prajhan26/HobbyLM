import Link from "next/link";
import { researchArticles } from "@/content/site";
import styles from "./ResearchList.module.css";

export function ResearchList({ preview = false }: { preview?: boolean }) {
  const articles = preview ? researchArticles.slice(0, 3) : researchArticles;
  return <div className={styles.list}>{articles.map((article) => <Link className={styles.item} href={`/research/${article.slug}`} key={article.slug}><span className={styles.number}>{article.number}</span><span><strong>{article.title}</strong>{!preview && <small>{article.summary}</small>}</span><span className={styles.status}>{article.status}</span></Link>)}</div>;
}
