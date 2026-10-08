type ModelAccessLink = {
  label: string;
  destination: string;
  brand: "huggingface" | "github";
  href: string | null;
};

export const modelAccessLinks: readonly ModelAccessLink[] = [
  {
    label: "Try online",
    destination: "Hugging Face Space",
    brand: "huggingface",
    href: "https://huggingface.co/spaces/harims95/hobbylm-1b-chat",
  },
  {
    label: "Model weights",
    destination: "HobbyLM 1B",
    brand: "huggingface",
    href: "https://huggingface.co/harims95/hobbylm-1B",
  },
  {
    label: "Source code",
    destination: "GitHub",
    brand: "github",
    href: "https://github.com/fuellabs-ai/hobbylm-1b",
  },
];

export const localRuntimeLinks = [
  { label: "MLX 4-bit", href: null },
  { label: "MLX 8-bit", href: null },
  { label: "GGUF", href: "https://huggingface.co/harims95/hobbylm-1B-gguf" },
  { label: "Ollama", href: null },
  { label: "llama.cpp", href: null },
] as const;
