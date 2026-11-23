"use client";
import { useState } from "react";
import styles from "./CodeBlock.module.css";
export function CodeBlock({ code, language = "python" }: { code: string; language?: string }) {
  const [status, setStatus] = useState("Copy");
  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setStatus("Copied");
    } catch {
      setStatus("Select code to copy");
    }
    window.setTimeout(() => setStatus("Copy"), 2400);
  }
  return <div className={styles.wrap}><div className={styles.bar}><span>{language === "shell" ? "Terminal" : "Python"}</span><button type="button" onClick={copy} aria-label={`Copy ${language} code`}><span aria-live="polite">{status}</span></button></div><pre tabIndex={0} aria-label={`${language} code`}><code>{code}</code></pre></div>;
}
