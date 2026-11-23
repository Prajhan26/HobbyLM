import type { MetadataRoute } from "next";

const siteUrl = "https://fuellabs.in";

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date("2026-10-05");

  return [
    { url: siteUrl, lastModified, changeFrequency: "weekly", priority: 1 },
    { url: `${siteUrl}/hobbylm`, lastModified, changeFrequency: "weekly", priority: 0.8 },
    { url: `${siteUrl}/docs/embed`, lastModified, changeFrequency: "monthly", priority: 0.8 },
    { url: `${siteUrl}/docs/embed/quickstart`, lastModified, changeFrequency: "monthly", priority: 0.8 },
    { url: `${siteUrl}/docs/embed/usage`, lastModified, changeFrequency: "monthly", priority: 0.7 },
    { url: `${siteUrl}/docs/embed/integrations/langchain`, lastModified, changeFrequency: "monthly", priority: 0.6 },
    { url: `${siteUrl}/docs/embed/integrations/llamaindex`, lastModified, changeFrequency: "monthly", priority: 0.6 },
    { url: `${siteUrl}/docs/embed/integrations/chroma`, lastModified, changeFrequency: "monthly", priority: 0.6 },
    { url: `${siteUrl}/docs/embed/benchmarks`, lastModified, changeFrequency: "monthly", priority: 0.6 },
    { url: `${siteUrl}/docs/embed/fine-tuning`, lastModified: new Date("2026-11-23"), changeFrequency: "monthly", priority: 0.6 },
    { url: `${siteUrl}/research`, lastModified, changeFrequency: "weekly", priority: 0.8 },
    { url: `${siteUrl}/research/pretraining-hobbylm`, lastModified, changeFrequency: "monthly", priority: 0.9 },
    { url: `${siteUrl}/research/building-hobbylm-embed`, lastModified: new Date("2026-11-20"), changeFrequency: "monthly", priority: 0.9 },
    { url: `${siteUrl}/research/why-moe-for-embeddings`, lastModified: new Date("2026-11-20"), changeFrequency: "monthly", priority: 0.8 },
    { url: `${siteUrl}/team`, lastModified, changeFrequency: "monthly", priority: 0.6 },
  ];
}
