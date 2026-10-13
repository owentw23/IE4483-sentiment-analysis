// Run from the project root: node scripts/vault-lint.cjs
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..', 'Knowledge');
const exempt = new Set(['VAULT-RULES.md']);
function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    if (['.git', '.obsidian', 'Auto', 'Raw'].includes(entry.name)) return [];
    const full = path.join(dir, entry.name);
    return entry.isDirectory() ? walk(full) : full.endsWith('.md') ? [full] : [];
  });
}
const files = walk(root);
const errors = [];
for (const file of files) {
  const body = fs.readFileSync(file, 'utf8');
  if (!exempt.has(path.basename(file))) {
    const front = body.match(/^---\r?\n([\s\S]*?)\r?\n---/);
    if (!front) errors.push(`Missing frontmatter: ${file}`);
    else {
      for (const field of ['tags', 'status', 'created', 'modified']) {
        if (!new RegExp(`^${field}:`, 'm').test(front[1])) errors.push(`Missing ${field}: ${file}`);
      }
      for (const field of ['created', 'modified']) {
        const match = front[1].match(new RegExp(`^${field}:\\s*(.+)$`, 'm'));
        if (match && !/^\d{4}-\d{2}-\d{2}$/.test(match[1].trim())) errors.push(`Invalid ${field} date: ${file}`);
      }
    }
  }
  for (const match of body.matchAll(/\[\[([^\]]+)\]\]/g)) {
    const target = match[1].split('|')[0].split('#')[0];
    if (!target) continue;
    const candidates = files.filter(candidate =>
      path.relative(root, candidate).replaceAll('\\', '/').replace(/\.md$/, '') === target ||
      path.basename(candidate, '.md') === target);
    if (candidates.length !== 1) errors.push(`Missing or ambiguous wikilink ${target}: ${file}`);
  }
}
if (errors.length) { console.error(errors.join('\n')); process.exitCode = 1; }
else console.log(`PASS: ${files.length} Markdown files; frontmatter, dates, and wikilinks checked.`);
