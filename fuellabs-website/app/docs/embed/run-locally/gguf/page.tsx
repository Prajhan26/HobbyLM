import styles from "../../docs.module.css";
export default function Page() { return <article className={styles.content}>
<p className={styles.kicker}>Run locally / GGUF and Ollama</p><h1>GGUF and Ollama</h1><p className={styles.lede}>Use the published GGUF model with llama.cpp, or load the Ollama release.</p>
<p>Follow the release guides for supported input settings and query/document handling. Ollama vectors differ from the reference model; use the same runtime for indexing and queries.</p>
<div className={styles.linkRow}><a href="https://huggingface.co/harims95/hobbylm-embed-140m-gguf">GGUF model ↗</a><a href="https://github.com/fuellabs-ai/hobbylm-embed-140m/blob/main/gguf/RUN-LOCALLY.md">llama.cpp guide ↗</a><a href="https://ollama.com/hariharanms95/hobbylm-embed-140m">Ollama model ↗</a></div>
</article>; }
