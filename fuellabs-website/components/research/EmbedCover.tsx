"use client";

import { useEffect, useRef, useState } from "react";
import { PauseIcon, PlayIcon, ReplayIcon } from "@/components/ui/ControlIcons";
import styles from "@/app/research/[slug]/page.module.css";

export function EmbedCover() {
  const video = useRef<HTMLVideoElement>(null);
  const wantsPlay = useRef(true);
  const visible = useRef(false);
  const [playing, setPlaying] = useState(false);
  function sync() {
    const element = video.current;
    if (!element) return;
    if (wantsPlay.current && visible.current && !document.hidden && !element.ended) void element.play().catch(() => {});
    else element.pause();
  }
  useEffect(() => {
    const element = video.current!;
    const motion = matchMedia("(prefers-reduced-motion: reduce)");
    wantsPlay.current = !motion.matches;
    const observer = new IntersectionObserver(entries => { visible.current = entries[0].isIntersecting; sync(); }, { threshold: 0.3 });
    observer.observe(element);
    const onMotion = () => { wantsPlay.current = !motion.matches; sync(); };
    motion.addEventListener("change", onMotion);
    document.addEventListener("visibilitychange", sync);
    return () => { observer.disconnect(); motion.removeEventListener("change", onMotion); document.removeEventListener("visibilitychange", sync); element.pause(); };
  }, []);
  function toggle() {
    const element = video.current!;
    if (element.ended) element.currentTime = 0;
    wantsPlay.current = element.paused;
    sync();
  }
  function replay() { video.current!.currentTime = 0; wantsPlay.current = true; sync(); }
  return <figure className={styles.cover} aria-label="Conceptual text embedding: a line of text turns into points that flow into a large network of passage representations. Positions are illustrative.">
    <video ref={video} muted playsInline preload="metadata" poster="/research/embed/B1-C-cover-poster-1920x1080.png" onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => { wantsPlay.current = false; setPlaying(false); }}>
      <source media="(max-width:640px)" src="/research/embed/B1-C-cover-square.webm" type="video/webm" />
      <source media="(max-width:640px)" src="/research/embed/B1-C-cover-square.mp4" type="video/mp4" />
      <source src="/research/embed/B1-C-cover.webm" type="video/webm" />
      <source src="/research/embed/B1-C-cover.mp4" type="video/mp4" />
    </video>
    <picture className={styles.coverStill}><source media="(max-width:640px)" srcSet="/research/embed/B1-C-cover-poster-square-1080.png" /><img src="/research/embed/B1-C-cover-poster-1920x1080.png" alt="Conceptual network of passage representations. Positions are illustrative." /></picture>
    <div className={styles.coverControls}><button type="button" onClick={toggle} aria-label={playing ? "Pause cover animation" : "Play cover animation"}>{playing ? <PauseIcon /> : <PlayIcon />}</button><button type="button" onClick={replay} aria-label="Replay cover animation"><ReplayIcon /></button></div>
  </figure>;
}
