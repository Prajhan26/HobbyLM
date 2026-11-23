import { CodeBlock } from "@/components/docs/CodeBlock";
import styles from "../../docs.module.css";
const code = `from langchain_hobbylm import HobbyLMEmbeddings

embeddings = HobbyLMEmbeddings(
    revision="ea708cee107818fa0c1871eebd26d82a1e3fb1aa",
    trust_remote_code=True,
    query_prompt_name="query",
    document_prompt_name="passage",
)
query_vector = embeddings.embed_query("How do I load HobbyLM-Embed?")
document_vectors = embeddings.embed_documents(["HobbyLM-Embed runs locally."])`;
const alternative = `from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="harims95/hobbylm-embed-140m",
    model_kwargs={
        "trust_remote_code": True,
        "revision": "ea708cee107818fa0c1871eebd26d82a1e3fb1aa",
    },
    query_encode_kwargs={"prompt_name": "query"},
    encode_kwargs={"prompt_name": "passage"},
)`;
export default function LangChain() {
  return <article className={styles.content}><p className={styles.kicker}>Integrations / LangChain</p><h1>LangChain.</h1>
    <p className={styles.lede}>Use langchain-hobbylm for exact vectors from the reference model while preserving line breaks.</p>
    <h2>Install</h2><CodeBlock code="pip install langchain-hobbylm" language="shell" />
    <h2>Encode queries and documents</h2><CodeBlock code={code} />
    <p>The wrapper uses <code>prompt_name=&quot;query&quot;</code> for queries and <code>prompt_name=&quot;passage&quot;</code> for documents. Pass raw text without adding prefixes.</p>
    <h2>Longer inputs</h2><p>512 tokens by default and up to 2,048 tokens opt-in. To opt in, add <code>{'model_kwargs={"tokenizer_kwargs": {"model_max_length": 2048}}'}</code> to the constructor.</p>
    <h2>Alternative: langchain-huggingface</h2><CodeBlock code="pip install langchain-huggingface sentence-transformers" language="shell" /><CodeBlock code={alternative} />
    <p>This alternative replaces line breaks with spaces before encoding. Use langchain-hobbylm when you need to preserve the original text for exact vectors.</p>
    <div className={styles.linkRow}><a href="https://pypi.org/project/langchain-hobbylm/">Package on PyPI ↗</a></div>
  </article>;
}
