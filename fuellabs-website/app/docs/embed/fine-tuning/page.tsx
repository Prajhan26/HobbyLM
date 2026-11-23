import type { Metadata } from "next";
import { CodeBlock } from "@/components/docs/CodeBlock";
import styles from "../docs.module.css";

export const metadata: Metadata = { title: "Fine-tune on your own data | HobbyLM-Embed docs" };
const load = `from sentence_transformers import SentenceTransformer

model = SentenceTransformer("my-hobbylm-embed", trust_remote_code=True)
queries = model.encode(["Where is our leave policy?"], prompt_name="query")
passages = model.encode(["The leave policy is in the handbook."], prompt_name="passage")`;

export default function FineTuning() {
  return <article className={styles.content}>
    <p className={styles.kicker}>Guides / Fine-tuning</p><h1>Fine-tune on your own data.</h1>
    <p className={styles.lede}>Use your own question-passage pairs to create a separate local model with the public cookbook.</p>
    <h2>Get the cookbook</h2><CodeBlock language="shell" code={`git clone https://github.com/fuellabs-ai/cookbooks.git
cd cookbooks/fine-tune-on-your-data
pip install sentence-transformers accelerate datasets`} />
    <p>The script loads the base model with <code>trust_remote_code=True</code>, pinned to revision <code>ea708cee107818fa0c1871eebd26d82a1e3fb1aa</code>. Keep that pin when adapting the script.</p>
    <h2>Prepare your pairs</h2><p>Make a CSV with <code>query</code> and <code>passage</code> columns. Use raw text. The trainer maps those columns to the model&apos;s named prompts. Keep a separate held-out set for evaluation.</p>
    <h2>Run and save</h2><CodeBlock language="shell" code="python finetune.py --train my_pairs.csv --out my-hobbylm-embed" />
    <p>The script saves a new model folder. Keep the original release unchanged. The recipe uses 512-token inputs.</p>
    <h2>Check before switching</h2><p>Follow the cookbook&apos;s evaluation instructions with your own held-out questions and corpus. Decide whether to switch based on that check. Fine-tuning does not guarantee an accuracy improvement.</p>
    <h2>Load your local model</h2><CodeBlock code={load} />
    <p>512 tokens by default and up to 2,048 tokens opt-in. Set <code>model.max_seq_length = 2048</code> after loading to opt in. Re-embed the whole corpus with the model you use. Keep the original and fine-tuned vectors in separate indexes.</p>
    <div className={styles.linkRow}><a href="https://github.com/fuellabs-ai/cookbooks/tree/main/fine-tune-on-your-data">Full cookbook ↗</a></div>
  </article>;
}
