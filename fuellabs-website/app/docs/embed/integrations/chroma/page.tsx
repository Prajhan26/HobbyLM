import { CodeBlock } from "@/components/docs/CodeBlock";
import styles from "../../docs.module.css";
const code = `import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

class HobbyLMEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self._model = SentenceTransformer(
            "harims95/hobbylm-embed-140m",
            revision="ea708cee107818fa0c1871eebd26d82a1e3fb1aa",
            trust_remote_code=True,
        )

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), prompt_name="passage").tolist()

    def embed_query(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), prompt_name="query").tolist()

client = chromadb.Client()
collection = client.create_collection(
    name="hobbylm_documents",
    embedding_function=HobbyLMEmbeddingFunction(),
    metadata={"hnsw:space": "cosine"},
)
collection.add(ids=["local_embeddings"], documents=["HobbyLM-Embed runs locally."])
results = collection.query(query_texts=["How do I load HobbyLM-Embed?"], n_results=1)`;
export default function Chroma() { return <article className={styles.content}>
  <p className={styles.kicker}>Integrations / Chroma</p><h1>Chroma.</h1>
  <p className={styles.lede}>Use separate named prompts for indexing documents and embedding queries.</p>
  <h2>Install</h2><CodeBlock code="pip install chromadb sentence-transformers" language="shell" />
  <h2>Index and query</h2><CodeBlock code={code} />
  <p>The function uses <code>prompt_name=&quot;passage&quot;</code> for documents and <code>prompt_name=&quot;query&quot;</code> for search. Pass raw text without adding prefixes.</p>
  <h2>Longer inputs</h2><p>512 tokens by default and up to 2,048 tokens opt-in. Set <code>self._model.max_seq_length = 2048</code> after loading the model inside the embedding function to opt in.</p>
  <p>Rebuild the document embeddings if you change the model or encoding configuration.</p>
  <div className={styles.linkRow}><a href="https://docs.trychroma.com/docs/embeddings/embedding-functions">Chroma embedding functions ↗</a></div>
</article>; }
