"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import styles from "@/app/docs/embed/docs.module.css";

const base = "/docs/embed";
const groups = [
  { label: "Get started", items: [["Overview", ""], ["Quickstart", "/quickstart"], ["Usage guide", "/usage"]] },
  { label: "Run locally", items: [["MLX (Apple Silicon)", "/run-locally/mlx"], ["GGUF", "/run-locally/gguf"]] },
  { label: "Integrations", items: [["LangChain", "/integrations/langchain"], ["LlamaIndex", "/integrations/llamaindex"], ["Chroma", "/integrations/chroma"]] },
  { label: "Reference", items: [["Benchmarks", "/benchmarks"]] },
  { label: "Guides", items: [["Fine-tune on your own data", "/fine-tuning"]] },
] as const;
const pages = groups.flatMap(group => group.items.map(([title, slug]) => ({ title, href: base + slug, group: group.label })));

export function DocsShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname().replace(/\/$/, "");
  const current = pages.findIndex(page => page.href === pathname);
  const page = pages[current] ?? pages[0];
  const article = useRef<HTMLDivElement>(null);
  const menu = useRef<HTMLDetailsElement>(null);
  const [headings, setHeadings] = useState<{ id: string; title: string }[]>([]);
  const [active, setActive] = useState("");
  useEffect(() => {
    const elements = Array.from(article.current?.querySelectorAll("h2") ?? []);
    const links = elements.map((element, index) => {
      const title = element.textContent ?? "";
      element.id = `section-${index}-${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
      return { id: element.id, title };
    });
    setHeadings(links);
    setActive(links[0]?.id ?? "");
    if (menu.current) menu.current.open = false;
    const observer = new IntersectionObserver(entries => {
      const visible = entries.find(entry => entry.isIntersecting);
      if (visible) setActive(visible.target.id);
    }, { rootMargin: "-100px 0px -55% 0px" });
    elements.forEach(element => observer.observe(element));
    return () => observer.disconnect();
  }, [pathname]);
  const navigation = <nav aria-label="HobbyLM-Embed documentation">{groups.map(group => <div className={styles.group} key={group.label}><p>{group.label}</p>{group.items.map(([title, slug]) => <Link key={slug} href={base + slug} aria-current={pathname === base + slug ? "page" : undefined}>{title}</Link>)}</div>)}</nav>;
  return <div className={styles.shell}>
    <aside className={styles.sidebar}>
      <Link className={styles.product} href={base}><span className={styles.square} aria-hidden="true" />HobbyLM-Embed<span className={styles.productSub}>Documentation</span></Link>
      {navigation}
      <div className={styles.sidebarFoot}><a href="https://huggingface.co/harims95/hobbylm-embed-140m">Hugging Face ↗</a><a href="https://github.com/fuellabs-ai/hobbylm-embed-140m">GitHub ↗</a></div>
    </aside>
    <details className={styles.mobileMenu} ref={menu}><summary>HobbyLM-Embed <span>Browse docs</span></summary>{navigation}</details>
    <div className={styles.main}>
      <div className={styles.utility}><Link href={base}>HobbyLM-Embed</Link><span aria-hidden="true">/</span><span>{page.group}</span></div>
      <div className={styles.readingGrid}><div ref={article} className={styles.articleColumn}>{children}
        <nav className={styles.pagination} aria-label="Documentation pages">{current > 0 ? <Link href={pages[current - 1].href}><small>Previous</small>{pages[current - 1].title}</Link> : <span />}{current < pages.length - 1 && <Link href={pages[current + 1].href}><small>Next</small>{pages[current + 1].title}</Link>}</nav>
        <footer className={styles.docsFooter}>Compute supported by <a href="https://jarvislabs.ai">Jarvis Labs</a>.</footer>
      </div><aside className={styles.contents} aria-label="On this page">{headings.length > 0 && <><p>On this page</p>{headings.map(heading => <a key={heading.id} href={`#${heading.id}`} aria-current={active === heading.id ? "location" : undefined}>{heading.title}</a>)}</>}</aside></div>
    </div>
  </div>;
}
