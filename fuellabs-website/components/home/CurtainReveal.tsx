"use client";

import { useEffect, useState } from "react";
import { Wordmark } from "@/components/brand/Wordmark";
import styles from "./CurtainReveal.module.css";

export function CurtainReveal() {
  const [state, setState] = useState<"enter" | "leave" | "done">(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "done" : "enter");
  useEffect(() => {
    const leave = window.setTimeout(() => setState("leave"), 1800);
    const done = window.setTimeout(() => setState("done"), 2800);
    return () => { window.clearTimeout(leave); window.clearTimeout(done); };
  }, []);
  if (state === "done") return null;
  return <div className={`${styles.curtain} ${state === "leave" ? styles.leave : ""}`} aria-hidden="true"><div className={styles.logo}><Wordmark large tabIndex={-1} /></div></div>;
}
