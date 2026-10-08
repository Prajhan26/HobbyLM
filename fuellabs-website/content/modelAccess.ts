export const modelAccessLinks = [
  {
    label: "Try online",
    destination: "Hugging Face Space",
    brand: "huggingface",
    href: null,
  },
  {
    label: "Model weights",
    destination: "HobbyLM 1B",
    brand: "huggingface",
    href: "https://huggingface.co/harims95/hobbylm-1b-broad-sft-3450-hf/tree/ddf46d8f9c651ca3d0a73bb3189a7dc9a9112ce5",
  },
  {
    label: "Source code",
    destination: "GitHub",
    brand: "github",
    href: "https://github.com/Prajhan26/HobbyLM",
  },
] as const;

export const localRuntimeLinks = [
  { label: "MLX 4-bit", href: null },
  { label: "MLX 8-bit", href: null },
  { label: "GGUF", href: null },
  { label: "Ollama", href: null },
  { label: "llama.cpp", href: null },
] as const;
