import styles from "../../docs.module.css";
export default function Page() { return <article className={styles.content}>
<p className={styles.kicker}>Run locally / MLX (Apple Silicon)</p><h1>MLX (Apple Silicon)</h1><p className={styles.lede}>Run HobbyLM-Embed locally on Apple silicon using the published MLX model.</p>
<p>512 tokens by default and up to 2,048 tokens opt-in. Follow the guide for the runtime-specific configuration.</p>
<div className={styles.linkRow}><a href="https://huggingface.co/harims95/hobbylm-embed-140m-mlx">MLX model ↗</a><a href="https://github.com/fuellabs-ai/hobbylm-embed-140m/tree/main/mlx">Installation and usage guide ↗</a></div>
</article>; }
