"use client";

import { useEffect, useRef, useState } from "react";

export function EmbedFigure({ id }: { id: string }) {
  const frame = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(600);
  useEffect(() => {
    function resize(event: MessageEvent) {
      if (event.origin !== location.origin || event.source !== frame.current?.contentWindow) return;
      if (event.data?.type === "embed-figure-size" && event.data.id === id && Number.isFinite(event.data.height)) {
        setHeight(Math.max(100, Math.min(4000, event.data.height)));
      }
    }
    window.addEventListener("message", resize);
    return () => window.removeEventListener("message", resize);
  }, [id]);
  return <div style={{ marginBlock: "3rem" }}><iframe ref={frame} src={`/research/embed/${id}/index.html`} title={`HobbyLM-Embed ${id.startsWith("B1") ? "development" : "architecture"} figure ${id.split("-")[1]}`} loading="lazy" style={{ display: "block", border: 0, width: "100%", height }} /></div>;
}
