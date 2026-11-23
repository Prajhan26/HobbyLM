import type { ReactNode } from "react";
import Link from "next/link";
import articles from "@/content/embedArticles.json";
import { SignalHeading, SignalMark } from "@/components/typography/SignalHeading";
import { ArticleContents } from "./ArticleContents";
import { EmbedFigure } from "./EmbedFigure";
import { EmbedCover } from "./EmbedCover";
import styles from "@/app/research/[slug]/page.module.css";
import study from "./EmbedArticle.module.css";

const figures = new Set(['B1-F1', 'B1-I1', 'B1-F2', 'B1-I2', 'B1-I3', 'B1-I4', 'B1-F3', 'B2-F1', 'B2-F2', 'B2-F3']);
function slugify(text: string) { return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""); }
function HeadingText({ text }: { text: string }) {
  const lastSpace = text.lastIndexOf(" ");
  return <>{lastSpace < 0 ? "" : text.slice(0, lastSpace + 1)}<span style={{ whiteSpace: "nowrap" }}>{text.slice(lastSpace + 1)}<SignalMark /></span></>;
}
function inline(text: string): ReactNode[] {
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\((?:https?:\/\/|\/(?!\/))[^)]+\))/g).map((part, i) => {
    if (part.startsWith('`')) return <code key={i}>{part.slice(1, -1)}</code>;
    if (part.startsWith('**')) return <strong key={i}>{part.slice(2, -2)}</strong>;
    const link = part.match(/^\[([^\]]+)\]\(((?:https?:\/\/|\/(?!\/))[^)]+)\)$/);
    return link ? <a key={i} href={link[2]}>{link[1]}</a> : part;
  });
}

export function EmbedArticle({ slug }: { slug: string }) {
  const article = articles.find(article => article.slug === slug)!;
  const lines = article.markdown.trim().split('\n');
  const deck = lines[2];
  const sections = lines.filter(line => line.startsWith('## ')).map(line => ({ id: slugify(line.slice(3)), label: line.slice(3) }));
  const blocks: ReactNode[] = [];
  for (let i = 4; i < lines.length;) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.startsWith('<!--')) {
      const id = line.match(/B[12]-[FI]\d/)?.[0];
      if (id && figures.has(id)) blocks.push(<EmbedFigure key={i} id={id} />);
      i++; continue;
    }
    if (line.startsWith('```')) {
      const code = []; const key = i++;
      while (i < lines.length && !lines[i].startsWith('```')) code.push(lines[i++]);
      blocks.push(<pre key={key} className={styles.codeBlock}><code>{code.join('\n')}</code></pre>); i++; continue;
    }
    if (line.startsWith('|')) {
      const rows: string[][] = []; const key = i;
      while (i < lines.length && lines[i].startsWith('|')) rows.push(lines[i++].split('|').slice(1,-1).map(cell => cell.trim()));
      blocks.push(<div key={key} className={styles.tableWrap} tabIndex={0} role="region" aria-label="HobbyLM-Embed evaluation results"><table><thead><tr>{rows[0].map((cell,j)=><th key={j}>{inline(cell)}</th>)}</tr></thead><tbody>{rows.slice(2).map((row,j)=><tr key={j}>{row.map((cell,k)=><td key={k}>{inline(cell)}</td>)}</tr>)}</tbody></table></div>); continue;
    }
    if (line.startsWith('## ')) { const title = line.slice(3); blocks.push(<h2 key={i} id={slugify(title)}><HeadingText text={title} /></h2>); i++; continue; }
    if (line.startsWith('### ')) { blocks.push(<h3 key={i}><HeadingText text={line.slice(4)} /></h3>); i++; continue; }
    if (line.startsWith('- ')) {
      const items = []; const key=i;
      while (i < lines.length && lines[i].startsWith('- ')) { items.push(<li key={i}>{inline(lines[i++].slice(2))}</li>); }
      blocks.push(<ul key={key}>{items}</ul>); continue;
    }
    const paragraph = []; const key = i;
    while (i < lines.length && lines[i].trim() && !/^(#|<!--|```|\||- )/.test(lines[i])) paragraph.push(lines[i++]);
    if (paragraph.length) blocks.push(<p key={key}>{inline(paragraph.join(' '))}</p>); else i++;
  }
  const readingMinutes = Math.max(1, Math.ceil(article.markdown.split(/\s+/).length / 220));
  return <article className={study.study}>
    <header className={study.header}>
      <Link className={study.back} href="/research"><span aria-hidden="true">‹</span> All posts</Link>
      <SignalHeading as="h1" signal={false}><HeadingText text={article.title} /></SignalHeading>
      <p className={study.deck}>{deck}</p>
      <div className={study.meta}><span>Fuel Labs</span><span aria-hidden="true">·</span><span>Research</span><span aria-hidden="true">·</span><span>{readingMinutes} min read</span></div>
      {slug === "building-hobbylm-embed" && <EmbedCover />}
    </header>
    <ArticleContents sections={sections} />
    <div className={`article-body-copy ${styles.body} ${study.body}`}>{blocks}</div>
  </article>;
}
