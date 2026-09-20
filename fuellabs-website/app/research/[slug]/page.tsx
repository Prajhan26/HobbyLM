import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { SignalHeading } from "@/components/typography/SignalHeading";
import { researchArticles } from "@/content/site";
import styles from "./page.module.css";

export function generateStaticParams() { return researchArticles.map(({ slug }) => ({ slug })); }

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const article = researchArticles.find((item) => item.slug === slug);
  return article ? { title: article.title } : {};
}

export default async function ArticlePage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const article = researchArticles.find((item) => item.slug === slug);
  if (!article) notFound();
  return <article className={`section ${styles.article}`}><header className="container"><span className="eyebrow">Research / {article.number}</span><SignalHeading as="h1" className="display">{article.title}</SignalHeading><p className={`intro-copy ${styles.deck}`}>{article.summary}</p><dl className={styles.meta}><div><dt>Status</dt><dd>{article.status}</dd></div><div><dt>Publication</dt><dd>After evidence and technical review</dd></div></dl></header><div className={`container ${styles.body}`}><p>This article route is ready for reviewed content. Drafts, internal claims, and placeholder results remain outside the public page until approval.</p></div></article>;
}
