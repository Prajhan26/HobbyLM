import fs from 'node:fs';
import path from 'node:path';
import { withAutomaticPlayback } from './embed-figure-playback.mjs';

const root = process.cwd();
const source = path.join(root, 'design-studies/hobbylm-embed');
const dest = path.join(root, 'public/research/embed');
fs.mkdirSync(dest, { recursive: true });
const coverSource = path.join(source, 'B1-C/video');
for (const name of ['B1-C-cover.mp4', 'B1-C-cover.webm', 'B1-C-cover-square.mp4', 'B1-C-cover-square.webm', 'B1-C-cover-poster-1920x1080.png', 'B1-C-cover-poster-square-1080.png']) {
  fs.copyFileSync(path.join(coverSource, name), path.join(dest, name));
}
const articles = [
  ['building-hobbylm-embed', '01', 'Building HobbyLM-Embed from scratch'],
  ['why-moe-for-embeddings', '02', 'Why we chose MoE for embeddings'],
].map(([slug, number, title]) => {
  const markdown = fs.readFileSync(path.join(root, `docs/workstreams/HOBBYLM-EMBED-BLOG-${number}-DRAFT-v0.1.md`), 'utf8');
  return { slug, title, markdown };
});
const ids = ['B1-F1', 'B1-I1', 'B1-F2', 'B1-I2', 'B1-I3', 'B1-I4', 'B1-F3', 'B2-F1', 'B2-F2', 'B2-F3'];
for (const id of ids) {
  const dir = path.join(dest, id);
  fs.mkdirSync(dir, { recursive: true });
  fs.cpSync(path.join(source, id, 'fonts'), path.join(dir, 'fonts'), { recursive: true });
  const filename = id === 'B1-F2' ? `${id}-fan.html` : `${id}.html`;
  let html = fs.readFileSync(path.join(source, id, filename), 'utf8');
  html = withAutomaticPlayback(html, id);
  html = html.replace('</head>', `<style>.asset-id,.prose{display:none!important}.page{padding:16px 12px!important;box-sizing:border-box;margin:0 auto!important;max-width:100%!important}html,body{overflow-x:hidden}</style></head>`);
  html = html.replace('</body>', `<script>function reportSize(){parent.postMessage({type:'embed-figure-size',id:${JSON.stringify(id)},height:Math.ceil(document.body.getBoundingClientRect().height)},'*')}new ResizeObserver(reportSize).observe(document.body);document.fonts.ready.then(reportSize);window.addEventListener('load',reportSize);</script></body>`);
  fs.writeFileSync(path.join(dir, 'index.html'), html);
}
fs.writeFileSync(path.join(root, 'content/embedArticles.json'), JSON.stringify(articles, null, 2) + '\n');
console.log(`Prepared ${articles.length} approved articles and ${ids.length} figures.`);
