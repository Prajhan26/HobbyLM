"use client";

import Image from "next/image";
import { useState } from "react";
import { localRuntimeLinks, modelAccessLinks } from "@/content/modelAccess";
import styles from "./ModelAccess.module.css";

function GitHubMark() {
  return (
    <svg className={styles.brandMark} viewBox="0 0 24 24" aria-hidden="true">
      <path
        fill="currentColor"
        d="M12 .7a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2.23c-3.22.7-3.9-1.36-3.9-1.36-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.16.08 1.78 1.2 1.78 1.2 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.74-1.55-2.57-.3-5.28-1.29-5.28-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.06 0 0 .97-.31 3.16 1.18a10.9 10.9 0 0 1 5.76 0c2.2-1.49 3.16-1.18 3.16-1.18.63 1.59.23 2.77.11 3.06.74.81 1.19 1.84 1.19 3.1 0 4.42-2.71 5.39-5.29 5.68.42.36.79 1.06.79 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z"
      />
    </svg>
  );
}

function HuggingFaceMark() {
  return <Image className={styles.brandMark} src="/brand/hugging-face.svg" width={18} height={18} alt="" aria-hidden="true" />;
}

export function ModelAccess() {
  const [localOpen, setLocalOpen] = useState(false);

  return (
    <div className={styles.accessList}>
      <div className={styles.accessGrid}>
        {modelAccessLinks.map((item) => {
          const content = (
            <span className={styles.destination}>
              <span>{item.destination}</span>
              {item.brand === "huggingface" ? <HuggingFaceMark /> : <GitHubMark />}
            </span>
          );
          return item.href ? (
            <a className={styles.accessTile} href={item.href} key={item.label} target="_blank" rel="noreferrer">
              {content}
            </a>
          ) : (
            <div className={`${styles.accessTile} ${styles.unavailable}`} key={item.label} role="link" aria-disabled="true">
              {content}
            </div>
          );
        })}
        <button
          className={`${styles.accessTile} ${styles.localTrigger}`}
          type="button"
          aria-expanded={localOpen}
          aria-controls="local-runtime-options"
          onClick={() => setLocalOpen((open) => !open)}
        >
          <span className={styles.destination}>
            <span>MLX · GGUF</span>
            <span className={styles.disclosureIcon} aria-hidden="true" />
          </span>
        </button>
      </div>

      <div className={styles.localOptions} id="local-runtime-options" hidden={!localOpen}>
        {localRuntimeLinks.map((item) =>
          item.href ? (
            <a href={item.href} key={item.label} target="_blank" rel="noreferrer">{item.label}</a>
          ) : (
            <span key={item.label} role="link" aria-disabled="true">{item.label}</span>
          ),
        )}
      </div>
    </div>
  );
}
