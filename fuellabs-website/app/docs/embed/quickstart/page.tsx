import { CodeBlock } from "@/components/docs/CodeBlock";
import styles from "../docs.module.css";
const install = `pip install sentence-transformers`;
const canonical = `from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "harims95/hobbylm-embed-140m",
    revision="ea708cee107818fa0c1871eebd26d82a1e3fb1aa",
    trust_remote_code=True,
)
emb = model.encode(["what is an embedding?"], prompt_name="query")
passages = model.encode(["An embedding represents text as a vector."], prompt_name="passage")
print(emb.shape)  # (1, 512)`;
const search = `${canonical.split("emb =")[0]}passages = ["A vector represents text as numbers.", "A tokenizer splits text into tokens."]
query = "How does text become a vector?"
passage_vectors = model.encode(passages, prompt_name="passage")
query_vector = model.encode(query, prompt_name="query")
scores = passage_vectors @ query_vector
ranking = scores.argsort()[::-1]
for index in ranking:
    print(float(scores[index]), passages[index])`;
export default function Quickstart() { return <article className={styles.content}><p className={styles.kicker}>Start here / Quickstart</p><h1>Run it locally.</h1><p className={styles.lede}>Load the open model with Sentence Transformers and encode your first query.</p><h2>Install</h2><CodeBlock code={install} language="shell" /><h2>Encode a query</h2><CodeBlock code={canonical} /><p>Expected shape: <code>(1, 512)</code>.</p><h2>Input length</h2><p>512 tokens by default and up to 2,048 tokens opt-in. Set <code>model.max_seq_length = 2048</code> after loading to opt in.</p><h2>Rank passages</h2><p>Use the passage prompt for documents and the query prompt for the search text. Dot product works because the vectors are already unit length.</p><CodeBlock code={search} /></article>; }
