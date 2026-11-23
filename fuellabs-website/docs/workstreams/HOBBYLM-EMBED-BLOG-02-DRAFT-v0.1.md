# Why we chose MoE for embeddings

The routing mechanism behind HobbyLM-Embed, and how to read its parameter counts.

HobbyLM-Embed uses a sparse mixture-of-experts architecture: **140M total parameters**, with **36M active per token** (excluding embeddings; **8 of 32 experts**). The router selects part of the expert pool for each token, allowing different tokens to take different paths through the feed-forward computation.

That choice carried familiar engineering from our language-model work into an encoder. We could build on an existing sparse architecture while training fresh weights for text embeddings. The encoder reads its input with bidirectional attention and produces a **512-dimensional representation**.

MoE gives us a way to vary which parameters participate in a token's computation. Understanding that mechanism also means being precise about what the active count includes, how tokens are routed, and what parameter counts can tell a reader about deployment.

<!-- Cover B2-C: Sparse routing, remaining encoder and pooling, then a text representation in a conceptual embedding field. -->

## Capacity and computation

A sparse feed-forward block contains a pool of expert networks. A router examines a token's hidden state and selects the experts that will process it. Their outputs are weighted and combined before the token representation continues through the model.

The expert pool supplies several possible transformations at that point in the encoder. Selecting a subset lets the model make a conditional choice about which transformations to apply. Attention supplies context between positions; the expert block processes the contextual representation at each position.

In HobbyLM-Embed, the router selects eight of 32 experts per token. Different tokens can receive different selections. The routing decision belongs to a token at a particular layer; it is not a decision to assign an entire document to a single expert.

The model has **140M total parameters**, with **36M active per token** (excluding embeddings; **8 of 32 experts**). This follows the MTEB counting convention we use. The excluded shared embedding table contains about **26M parameters**. The table remains part of the model and is used to look up input-token representations.

The total and active figures therefore answer different questions. The total describes the parameter inventory. The active figure describes participation under the stated counting convention. Shared attention and the input lookup still contribute work, and a deployment has to account for the weights it keeps available, including experts that a particular token does not select.

<!-- Figure 1 / B2-F1: Total versus active parameter bars, with the shared-table exclusion visible beside the bars. -->

## What the router does

At the entrance to an MoE block, the router receives a hidden state that already contains contextual information about the token. It uses that state to choose experts. Each selected expert transforms the token representation, and the weighted outputs return to the shared model computation.

With bidirectional attention, the context can include words on either side of the token. The router therefore operates on a representation influenced by the surrounding text. The same token in different contexts can arrive at the expert block with different hidden states.

Expert identities describe networks in the model, not predefined subjects. Establishing specialization would require measurements across inputs. The diagrams accompanying this article use illustrative selections without assigning experts a topic or linguistic skill.

<!-- Figure 2 / B2-F2: Contextual hidden state, router, eight selected experts, weighted combination and output. -->

## Routing across a text

Tokens in the same input can take different expert paths. An expert can also be selected for tokens from many different inputs. The routing mechanism works over representations learned during training, so a token's path should not be read as a fixed dictionary assignment.

This distinction matters for the final embedding. The model builds token representations through attention and feed-forward computation, then pools them into a text representation. The expert choices contribute to that process. They are not the coordinates of the final embedding, and the vector dimensions do not correspond to individual experts.

Across a batch, different tokens can collectively use more of the expert pool than any individual token selects. A per-token count consequently does not specify the number of distinct experts used by a passage or batch. It also does not imply that unselected expert weights have been removed from the deployment.

HobbyLM-Embed produces a **512-dimensional vector** through its pooling and normalization path. The `query:` and `passage:` prefixes indicate the retrieval role of an input. Both inputs pass through the encoder, and the application compares the resulting representations to retrieve passages.

In a point-field visualization, each document point represents that completed text embedding. The points are separate from the expert grid: the grid describes computation inside a block, while the field depicts a collection of encoder outputs. Our field is conceptual, with illustrative positions rather than a measured projection.

<!-- Figure 3 / B2-F3: Step through illustrative token selections, then show the route back to the shared encoder and pooled embedding. -->

## Earlier work on sparse embedding models

Nomic Embed v2 is earlier work on applying sparse MoE architectures to text embeddings. Zach Nussbaum and Brandon Duderstadt describe its development in [Training Sparse Mixture Of Experts Text Embedding Models](https://arxiv.org/abs/2502.07972). The accompanying [model card](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) documents its architecture and use.

Nomic's work provides a published account of sparse routing in an embedding encoder. It is part of the research context for developing embeddings with conditional expert computation.

For Fuel Labs, the practical reason to pursue this direction was continuity with a codebase we could inspect and modify. Starting with that foundation let us focus the embedding project on bidirectional representations and the training stages that make text vectors useful for retrieval. This explains the choice; it does not establish an architectural advantage over a dense encoder under matched training conditions.

## The tradeoffs remain in the system

Sparse routing changes which expert computations run for a token. It also introduces work: the router must select experts, and the implementation must dispatch tokens and combine their outputs. The cost of those operations depends on the implementation and the workload.

A count of active parameters cannot establish query latency, throughput or memory use. Those depend on hardware, precision, input length and batching, as well as the model. Expert weights may remain resident even when an individual token does not use them.

Index construction and query serving also place different demands on the encoder. Index construction can process a collection in batches. A search request needs a query embedding before retrieval can begin. A deployment evaluation should measure both paths in the settings the application will use.

We evaluated the released model in fp32, with a maximum evaluation length of 512 tokens. That is the precision and length associated with the approved benchmark result in the companion article. The model supports inputs up to **1,024 tokens**; that input limit is separate from the benchmark setting.

Our reason for choosing MoE was to develop an embedding encoder with conditional expert computation on an architecture we already understood. The release makes that design available for inspection and use. The detailed data and optimization recipe remains internal.

The release destinations are [Hugging Face](https://huggingface.co/harims95/hobbylm-embed-140m) and [GitHub](https://github.com/fuellabs-ai/hobbylm-embed-140m), under Apache-2.0. We thank [Jarvis Labs](https://jarvislabs.ai) for providing compute for this work.
