# Building HobbyLM-Embed from scratch

A text embedding model built on Fuel Labs' sparse architecture.

HobbyLM-Embed achieved 62.33 on MTEB(eng, v2), submitted to the official leaderboard. We trained it from scratch for text search and similarity and are releasing it as `hobbylm-embed-140m` under Apache-2.0.

The model has 140M total, 36M active per token (8 of 32 experts), and produces 512-dimensional embeddings. The evaluation covered all 41 tasks using `mteb 2.26.31`, fp32 and a maximum input length of 512 tokens. The model supports inputs up to 1,024 tokens.

Sparse routing limits expert computation per token. The active count excludes the shared embedding table; running cost also depends on the shared computation, implementation and hardware.

## Quickstart

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("harims95/hobbylm-embed-140m", trust_remote_code=True)
q = model.encode(["how do vaccines work?"], prompt_name="query")
d = model.encode(["Vaccines train the immune system to ..."], prompt_name="passage")
print(model.similarity(q, d))
```

The example encodes a query and passage, then compares their embeddings. The named prompts apply the retrieval prefixes.

## Building the encoder

The work began with an architecture and training codebase we already understood. The embedding project required a different attention pattern and a sequence of objectives that taught the model how texts relate to one another. It also required evaluation that could distinguish progress on a training objective from progress on retrieval.

That distinction affected our decisions throughout development. A pretrained checkpoint needed an embedding probe before we could assess its retrieval behaviour. Later, a fine-tuning run improved on familiar material while losing ground on unfamiliar search tasks. Additional task prefixes also failed to produce a consistent improvement. Those experiments shaped how we evaluated candidates and which changes we retained.

<!-- Cover B1-C: Text becoming a representation. See HOBBYLM-EMBED-VISUAL-PROMPTS-v0.1.md. -->
<!-- Figure 1 / B1-F1: Specification card. -->

## A representation for search

A search system needs a way to compare a question with passages that might answer it. Exact wording can help, but a relevant passage may express the answer in different terms. An embedding model maps each text to a vector so the system can compare their learned representations.

The encoder has to retain information that matters for those comparisons. Two passages can share vocabulary and answer different questions. A question and its answer can share little wording. Training the representation space means teaching the model which relationships should affect similarity.

For retrieval with HobbyLM-Embed, queries use the `query:` prefix and passages use `passage:`. A document collection can be embedded ahead of time and stored in a vector index. When a query arrives, the application embeds it and searches that index for related passages.

<!-- Inline explainer B1-I1: Retrieval prefixes, compact motion. -->

The resulting vectors allow the application to retrieve candidate passages. An application can show them directly or pass them to a language model as context. The embedding model produces a vector; the surrounding application handles indexing, retrieval and presentation.

This division makes the encoder's behaviour consequential. A passage that the retrieval system misses never reaches the next stage. We therefore treated retrieval evaluation as part of model development, alongside the signals that showed whether training was progressing.

<!-- Figure 2 / B1-F2: Text, vectors, conceptual point field and an illustrative search result. -->

## Starting with fresh weights

In this project, "from scratch" refers to initialization. HobbyLM-Embed began with fresh model weights. We reused engineering work from the HobbyLM codebase, including its sparse architecture and training infrastructure, and kept the GPT-2 tokenizer.

Reusing the codebase let us work with a training path whose components we could inspect. It did not give the new encoder a trained representation space. The input embeddings, attention and expert transformations still had to learn from the embedding project's training process.

The encoder reads text with bidirectional attention. Each token can use context from both sides, which matters when a later word changes the meaning of an earlier one. A sentence representation needs that context before it can be compared with another sentence or passage.

<!-- Inline explainer B1-I2: Context from both sides, compact motion. -->

This changes the information available at each position. With causal attention, an early token cannot use the end of the passage. Bidirectional attention lets that token's representation incorporate later context before the model pools the text into an embedding.

Pretraining used masked-token prediction. The model learned to recover hidden text from its surrounding context. This provided the base representations for the embedding stages that followed.

The objective matches the attention pattern: some text is hidden, and the visible context supplies evidence for reconstructing it. Once we use the encoder for embeddings, the application needs its representations rather than predictions over the tokenizer's vocabulary.

The sparse feed-forward blocks use a router to select experts for each token. The model's reported size is 140M total, 36M active per token (8 of 32 experts). Our active count excludes the shared embedding table. We explain that convention and the routing mechanism in the companion article, "Why we chose MoE for embeddings."

## How a passage becomes a vector

The GPT-2 tokenizer converts text into token IDs, which the shared embedding table maps to input representations. Attention and feed-forward computation build contextual hidden states across the passage. Mean pooling and normalization then produce a 512-dimensional text embedding.

The input table supplies token representations at the entrance to the encoder. A search index stores the final whole-text vectors produced after contextual processing and pooling.

The point-field illustration shows those whole-text representations in a conceptual view. Positions and the returned passage are illustrative. Actual retrieval compares the full embeddings; apparent distances in the picture do not establish rankings.

Output dimensionality stays fixed for supported input lengths. A longer passage still receives a 512-dimensional vector, so the representation must compress more text into the same space. Test whether it preserves the information needed for the application's queries and passage boundaries.

<!-- Inline explainer B1-I3: Text to token IDs, input embeddings, contextual states, mean pooling, normalization and a 512-dimensional output. Shallow-isometric motion with a complete static poster. -->

## From a pretrained encoder to embeddings

The completed pretraining run gave us contextual representations and a saved encoder to evaluate. To test its potential for retrieval, we applied a lightweight embedding probe. That adaptation allowed us to inspect behaviour beyond masked-token prediction before committing to the later embedding stages.

A probe answers a limited question: can this base learn a useful comparison between texts under a small adaptation? It does not measure the finished release. We kept that distinction when interpreting the early results.

### Pair training

Pair training supplied relationships between texts directly. It taught the encoder to bring matching texts together in its representation space and distinguish them from alternatives. This changed the objective from recovering missing tokens to producing vectors useful for comparison.

<!-- Inline explainer B1-I4: Matching and alternative texts in pair training, compact conceptual motion. -->

For retrieval, the relationship matters more than a superficial resemblance in wording. A useful representation should put an answer-bearing passage close to its query while separating passages that look related but do not answer it. Pair training gives the encoder examples of the distinction it must learn.

### Fine-tuning

Fine-tuning allowed us to evaluate further adjustments to embedding behaviour. At this point the model already produced text vectors, so the question was how an adjustment changed the relationships encoded in them.

During development, we used a development suite and, in later experiments, an additional held-out panel to assess changes. These checks exposed regressions as well as improvements. They helped us understand how an adjustment affected behaviour across tasks.

These stages had different purposes. A masked-token loss describes the pretraining objective. A retrieval score tests whether a query finds relevant passages. The two measurements informed different decisions during development.

<!-- Figure 3 / B1-F3: Pretraining, pair training and fine-tuning. Method overview. -->

## Testing the training and evaluation paths

The earlier HobbyLM project had made recovery and checkpoint handling part of our training process. We carried that discipline into the encoder work. Before committing to pretraining, we tested the distributed training path, including checkpoint save and resume behaviour. The preflight account records a successful kill-and-resume check and verified checkpoint uploads.

That work mattered when a run needed to restart. Recovering a saved state let us continue the experiment and inspect its results. Checkpoints also gave us candidates for evaluation, so their identity had to remain clear across training and scoring.

The evaluation runner exposed a separate problem. An earlier checkpoint check had been hard-coded into a script. When a new candidate reached the runner, that stale check rejected it before a task could complete. Another fresh-run path expected metadata that an earlier setup had already created.

Those failures happened around the model, but they prevented us from measuring it. They showed why an evaluation script needs testing with a fresh output directory and a new checkpoint, as well as with the files left by an earlier run. A runner that works in an existing workspace can still fail on the next candidate.

The later evaluation reports recorded the checkpoint identity and checked consistent task splits and subsets. That discipline made the resulting comparisons more useful: a change in the score could be examined alongside the model and evaluation configuration that produced it.

## Decisions from the experiments

### More training changed which behaviour improved

In one fine-tuning experiment, later checkpoints improved on familiar evaluation material while performance on unfamiliar retrieval tasks declined. Selecting the latest checkpoint would therefore have favored a different behaviour from the one we wanted.

The result was consistent with increasing specialization to the material used in that stage. It did not establish a general rule about training duration. It gave us a concrete reason to evaluate saved candidates against the intended retrieval objective instead of assuming that continuing the run would help.

### Prefix tweaks did not replace the existing model

The project also tested short classification and clustering prefixes. None of the prefix tweaks improved on the existing candidate sufficiently to replace its usage mode.

We retained retrieval prefixes and kept the tested classification and clustering modes plain. This was a decision about the candidate and protocol used in that experiment. It gives no basis for assuming that an arbitrary instruction will improve the release on a new task.

For retrieval with the released model, the interface remains `query:` for queries and `passage:` for passages. Those roles should be consistent when preparing a collection and encoding the queries that search it.

### Evaluation settings needed to stay explicit

Task coverage, numerical precision and input length are part of an evaluation result. We recorded those settings alongside the model being scored so the measurements could be interpreted in their original context.

The final release result below covers the complete benchmark under the stated settings.

## Evaluation

The result covers all 41 tasks. The headline is Mean (Task), which averages individual task scores. Mean (TaskType), shown second in the table, gives each category equal weight.

| Measure | Score |
|---|---:|
| Mean (Task) | 62.33 |
| Mean (TaskType) | 57.62 |
| Classification | 78.43 |
| Pair classification | 79.21 |
| Semantic textual similarity (STS) | 79.06 |
| Retrieval | 50.24 |
| Clustering | 45.17 |
| Reranking | 43.93 |
| Summarization | 27.30 |

The categories test different behaviours. Classification evaluates information available to a classifier using the representations. Pair classification and semantic textual similarity examine relationships between texts. Retrieval tests whether queries find relevant passages; reranking evaluates candidate ordering. Clustering tests grouping behaviour, while summarization evaluates a further kind of text relationship.

Each category uses its respective task metrics. The scores describe the evaluation profile; the numerical distance between two categories does not measure how much better the model is at one kind of work. For a retrieval application, inspect retrieval behaviour directly alongside the aggregate.

We ran the evaluation with `mteb 2.26.31` in fp32, with a maximum input length of 512 tokens. The supported maximum input length is 1,024 tokens, but that longer setting was not used for this result. Applications using longer passages should evaluate their own chunking and retrieval settings.

The result is submitted to the official leaderboard. Its description will be updated when the submission is merged.

<!-- Figure 4 / B1-F4: Approved evaluation profile. The table is its accessible text equivalent, not a second visual asset. -->

## Using the release

The release is named `hobbylm-embed-140m`. The model destination is [Hugging Face](https://huggingface.co/harims95/hobbylm-embed-140m), and the implementation destination is [GitHub](https://github.com/fuellabs-ai/hobbylm-embed-140m). The release licence is Apache-2.0.

HobbyLM-Embed is intended for English text. Its output has 512 dimensions, and its supported input limit is 1,024 GPT-2 tokens. Token counts are measured after tokenization; a character or word limit is not interchangeable with the token limit.

Passage boundaries affect what the encoder has to preserve. A passage should contain enough context to answer the queries it is intended to serve. Combining unrelated sections makes the representation responsible for several different subjects, while an isolated fragment can omit information needed to understand it. Chunking should be evaluated as part of the retrieval system.

Keep the `query:` and `passage:` prefixes consistent between document indexing and query encoding. If the model or document encoding changes, regenerate the affected document embeddings before comparing them with new queries. Mixing representations from different encoding configurations makes the index harder to interpret.

For an application, the next useful test is a collection of queries with known relevant passages. Examine whether those passages are retrieved and whether the returned context is useful to the application. Include queries whose wording differs from the answer passage and cases where related passages contain the wrong answer. These tests expose the distinctions the representation must preserve in that collection.

The benchmark gives a common evaluation setting. Application evaluation supplies the document lengths, query patterns and passage boundaries that the deployed system will encounter.

### Limitations

- The supported language is English; multilingual performance is not established here.
- The published score used maximum length 512, while the model supports up to 1,024 tokens. The score does not establish retrieval quality at the longer limit.
- Inputs use the GPT-2 tokenizer. Check passage lengths in tokens when setting chunk boundaries.
- Retrieval requires the `query:` and `passage:` prefixes. Keep their use consistent across indexing and queries.

We are publishing the model and its approved implementation. Training data sources, mixtures and detailed preparation procedures remain internal, along with training hyperparameters and proprietary optimization work.

We thank Jarvis Labs for providing compute for this work.

## Team and citation

HobbyLM-Embed was developed by Hariharan and Prabhu Rajhan.

```bibtex
@misc{hobbylm_embed,
  author = {Hariharan and Prabhu Rajhan},
  title = {HobbyLM-Embed},
  howpublished = {Fuel Labs model release},
  url = {https://huggingface.co/harims95/hobbylm-embed-140m}
}
```
