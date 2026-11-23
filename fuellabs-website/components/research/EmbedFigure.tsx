"use client";

import { useEffect, useRef, useState } from "react";
import styles from "@/app/research/[slug]/page.module.css";

export function EmbedFigure({ id }: { id: string }) {
  const frame = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(600);
  function sizeOnLoad() {
    const body = frame.current?.contentDocument?.body;
    if (body) setHeight(Math.max(100, Math.min(4000, Math.ceil(body.getBoundingClientRect().height))));
  }
  useEffect(() => {
    const element = frame.current!;
    let observer: ResizeObserver | undefined;
    function observeContent() {
      const body = element.contentDocument?.body;
      if (!body || element.contentDocument?.URL === "about:blank") return;
      observer?.disconnect();
      observer = new ResizeObserver(sizeOnLoad);
      observer.observe(body);
      sizeOnLoad();
      void element.contentDocument?.fonts.ready.then(sizeOnLoad);
    }
    function resize(event: MessageEvent) {
      if (event.origin !== location.origin || event.source !== frame.current?.contentWindow) return;
      if (event.data?.type === "embed-figure-size" && event.data.id === id && Number.isFinite(event.data.height)) {
        setHeight(Math.max(100, Math.min(4000, event.data.height)));
      }
    }
    window.addEventListener("message", resize);
    element.addEventListener("load", observeContent);
    observeContent();
    return () => { window.removeEventListener("message", resize); element.removeEventListener("load", observeContent); observer?.disconnect(); };
  }, [id]);
  return <div className={styles.embedFigure}><iframe ref={frame} onLoad={sizeOnLoad} src={`/research/embed/${id}/index.html`} title={`HobbyLM-Embed ${id.startsWith("B1") ? "development" : "architecture"} figure ${id.split("-")[1]}`} loading="lazy" style={{ display: "block", border: 0, width: "100%", height }} /></div>;
}
