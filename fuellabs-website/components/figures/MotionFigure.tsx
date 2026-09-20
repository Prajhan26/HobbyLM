"use client";

import Image from "next/image";
import { useEffect, useState } from "react";
import styles from "./MotionFigure.module.css";

export function MotionFigure({ animatedSrc, stillSrc, alt, caption }: { animatedSrc: string; stillSrc: string; alt: string; caption: string }) {
  const [playing, setPlaying] = useState(true);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const followPreference = () => setPlaying(!media.matches);
    const initial = window.setTimeout(followPreference, 0);
    media.addEventListener("change", followPreference);
    return () => { window.clearTimeout(initial); media.removeEventListener("change", followPreference); };
  }, []);
  return <figure className={styles.figure}><Image unoptimized src={playing ? animatedSrc : stillSrc} width={878} height={698} alt={alt} /><div className={styles.meta}><figcaption>{caption}</figcaption><button type="button" aria-pressed={!playing} onClick={() => setPlaying(!playing)}>{playing ? "Pause motion" : "Play motion"}</button></div></figure>;
}
