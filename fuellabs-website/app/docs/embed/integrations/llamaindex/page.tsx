import { CodeBlock } from "@/components/docs/CodeBlock";
import styles from "../../docs.module.css";
const code = `from pydantic import PrivateAttr
from sentence_transformers import SentenceTransformer
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding

class HobbyLMEmbedding(BaseEmbedding):
    _model: SentenceTransformer = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = SentenceTransformer(
            "harims95/hobbylm-embed-140m",
            revision="ea708cee107818fa0c1871eebd26d82a1e3fb1aa",
            trust_remote_code=True,
        )

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._model.encode(query, prompt_name="query").tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._model.encode(text, prompt_name="passage").tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

embed_model = HobbyLMEmbedding()
Settings.embed_model = embed_model
query_vector = embed_model.get_query_embedding("How do I load HobbyLM-Embed?")
document_vector = embed_model.get_text_embedding("HobbyLM-Embed runs locally.")`;
export default function LlamaIndex() { return <article className={styles.content}>
  <p className={styles.kicker}>Integrations / LlamaIndex</p><h1>LlamaIndex.</h1>
  <p className={styles.lede}>Use a local adapter that passes the model&apos;s named prompts directly to Sentence Transformers.</p>
  <h2>Install</h2><CodeBlock code="pip install llama-index-core sentence-transformers" language="shell" />
  <h2>Configure embeddings</h2><CodeBlock code={code} />
  <p>Queries use <code>prompt_name=&quot;query&quot;</code>. Documents use <code>prompt_name=&quot;passage&quot;</code>. Pass raw text without adding prefixes.</p>
  <h2>Longer inputs</h2><p>512 tokens by default and up to 2,048 tokens opt-in. Set <code>self._model.max_seq_length = 2048</code> after loading the model inside the adapter to opt in.</p>
  <div className={styles.linkRow}><a href="https://github.com/fuellabs-ai/hobbylm-embed-140m#integrations">Integration reference ↗</a></div>
</article>; }
