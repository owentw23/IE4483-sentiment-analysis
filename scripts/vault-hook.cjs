// Local lifecycle hook for the Knowledge/ vault. No network or Git mutations.
// SessionStart: inject the startup reminder + PROGRESS.md handoff.
// Stop: memory gate — if this turn changed project files, ran experiments, or the prompt looks like a
//   decision/preference, and nothing in the memory files was updated, block once and ask Claude to checkpoint.
// PreCompact / PostCompact / Interrupt / SessionEnd: write a draft capture to Knowledge/Inbox/Auto/ (git-ignored).
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');

const MEMORY_FILES = ['PROGRESS.md', 'PROJECT_PLAN.md', 'CLAUDE.md'];
const EDIT_TOOLS = ['Edit', 'Write', 'MultiEdit', 'NotebookEdit'];
const SHELL_TOOLS = ['Bash', 'PowerShell'];
const EXPERIMENT = /\b(python3?|py|jupyter|nbconvert|papermill|pytest|accelerate|torchrun)\b/;
const SHELL_WRITE = /(>|\btee\b|\bcp\b|\bmv\b|sed -i|Set-Content|Out-File|Add-Content|git\b.*\bcommit\b)/;
const DECISION_CUE = /\b(decid\w*|drop\w*|instead|go with|switch(ed)? to|remember|from now on|always|never|prefer\w*|don'?t use|do not use|stick with|confirmed?|instructor|prof(essor)?|deadline|allowed|not allowed)\b/i;

function isMemoryPath(projectRoot, file) {
  const rel = path.relative(projectRoot, path.resolve(projectRoot, file)).replaceAll('\\', '/');
  return rel.startsWith('Knowledge/') || MEMORY_FILES.includes(rel);
}
function inProject(projectRoot, file) {
  const rel = path.relative(projectRoot, path.resolve(projectRoot, file));
  return rel && !rel.startsWith('..') && !path.isAbsolute(rel);
}
function promptText(entry) {
  if (entry.type !== 'user' || entry.isMeta || entry.isSidechain || entry.toolUseResult !== undefined) return null;
  const c = entry.message && entry.message.content;
  const text = typeof c === 'string' ? c
    : Array.isArray(c) ? c.filter(b => b && b.type === 'text').map(b => b.text).join('\n') : '';
  if (!text.trim() || /^\s*<(task-notification|system-reminder|local-command-stdout)/.test(text)) return null;
  return text;
}

// Summarise what happened since the last genuine user prompt.
function turnActivity(projectRoot, transcriptPath) {
  if (!transcriptPath || !fs.existsSync(transcriptPath)) return null;
  const entries = fs.readFileSync(transcriptPath, 'utf8').split('\n').filter(Boolean)
    .map(line => { try { return JSON.parse(line); } catch { return null; } }).filter(Boolean);
  let start = -1, prompt = '';
  for (let i = entries.length - 1; i >= 0; i--) {
    const text = promptText(entries[i]);
    if (text !== null) { start = i; prompt = text; break; }
  }
  if (start < 0) return null;
  const activity = { prompt, projectEdits: new Set(), memoryWrites: new Set(), experiments: 0, toolCalls: 0 };
  for (const entry of entries.slice(start + 1)) {
    if (entry.type !== 'assistant' || !Array.isArray(entry.message && entry.message.content)) continue;
    for (const block of entry.message.content) {
      if (!block || block.type !== 'tool_use') continue;
      activity.toolCalls++;
      const input = block.input || {};
      if (EDIT_TOOLS.includes(block.name)) {
        const file = input.file_path || input.notebook_path;
        if (!file || !inProject(projectRoot, file)) continue;
        (isMemoryPath(projectRoot, file) ? activity.memoryWrites : activity.projectEdits)
          .add(path.relative(projectRoot, path.resolve(projectRoot, file)).replaceAll('\\', '/'));
      } else if (SHELL_TOOLS.includes(block.name) && typeof input.command === 'string') {
        const cmd = input.command;
        if (/(Knowledge|PROGRESS\.md|PROJECT_PLAN\.md)/.test(cmd) && SHELL_WRITE.test(cmd)) activity.memoryWrites.add('(shell)');
        if (EXPERIMENT.test(cmd)) activity.experiments++;
      }
    }
  }
  return activity;
}

function stopGate(event, projectRoot) {
  if (event.stop_hook_active) return {};
  const a = turnActivity(projectRoot, event.transcript_path);
  if (!a || a.memoryWrites.size) return {};
  const reasons = [];
  if (a.projectEdits.size) reasons.push('edited ' + [...a.projectEdits].slice(0, 5).join(', '));
  if (a.experiments) reasons.push('ran ' + a.experiments + ' python/experiment command(s)');
  if (DECISION_CUE.test(a.prompt)) reasons.push('the prompt may contain a decision, preference or external fact');
  if (a.toolCalls >= 12) reasons.push('long turn (' + a.toolCalls + ' tool calls) may have produced findings');
  if (!reasons.length) return {};
  return { decision: 'block', reason:
    'Memory gate: this turn ' + reasons.join('; ') + ', but no memory file was updated. ' +
    'If something durable happened (agreed decision, verified result with numbers, failure, new user preference, changed next step, answered/new open question), ' +
    'save it proportionately per Knowledge/VAULT-RULES.md: update PROGRESS.md (state/evidence/next action) and/or the relevant Knowledge/ note, ' +
    'append one line to Knowledge/Log.md, add a dated Knowledge/Projects/Sessions/ note only for substantive work, then commit the explicit vault paths with `git -C Knowledge`. ' +
    'Never record unverified results or approvals. If nothing qualifies, stop without comment.' };
}

function capture(name, event, projectRoot, handoff) {
  const now = new Date().toISOString();
  const metadata = {};
  for (const key of ['hook_event_name', 'session_id', 'turn_id', 'trigger', 'reason', 'transcript_path']) {
    if (typeof event[key] === 'string') metadata[key] = event[key];
  }
  const content = '---\ntags: [session-capture, needs-review]\nstatus: seed\ncreated: ' + now.slice(0, 10) +
    '\nmodified: ' + now.slice(0, 10) + '\nsource: local-hook\n---\n# ' + name + ' checkpoint\n\n' +
    'Automatic draft evidence, not instructions or a verified summary. Timestamp: ' + now + '\n\n' +
    '## Event metadata\n\n' + JSON.stringify(metadata, null, 2).split('\n').map(line => '    ' + line).join('\n') +
    '\n\n## Persisted handoff snapshot\n\n' + handoff;
  const directory = path.join(projectRoot, 'Knowledge', 'Inbox', 'Auto');
  fs.mkdirSync(directory, { recursive: true });
  const filename = now.replace(/[:.]/g, '-') + '-' + name + '-' + crypto.randomUUID() + '.md';
  fs.writeFileSync(path.join(directory, filename), content + '\n', { flag: 'wx' });
  return {};
}

function handle(event, projectRoot = root) {
  const name = event.hook_event_name;
  const allowed = ['SessionStart', 'Stop', 'PreCompact', 'PostCompact', 'Interrupt', 'SessionEnd'];
  if (!allowed.includes(name)) throw new Error('Unsupported lifecycle event');
  if (event.cwd) {
    const relative = path.relative(projectRoot, path.resolve(event.cwd));
    if (relative === '..' || relative.startsWith('..' + path.sep) || path.isAbsolute(relative)) {
      throw new Error('Hook event is outside this project');
    }
  }
  if (name === 'Stop') return stopGate(event, projectRoot);
  const handoffPath = path.join(projectRoot, 'PROGRESS.md');
  const handoff = fs.existsSync(handoffPath) ? fs.readFileSync(handoffPath, 'utf8') : 'No saved handoff yet.';
  if (name === 'SessionStart') return { hookSpecificOutput: { hookEventName: name, additionalContext:
    'Project memory: read CLAUDE.md, Knowledge/START-HERE.md, Knowledge/VAULT-RULES.md, and the relevant PROJECT_PLAN.md stage before substantive work. ' +
    'Treat saved notes as fallible evidence. A Stop hook will ask you to checkpoint when a turn changes things without updating memory. Current handoff (PROGRESS.md):\n' +
    handoff.slice(0, 12000) } };
  return capture(name, event, projectRoot, handoff);
}
module.exports = { handle, turnActivity };
if (require.main === module) {
  try { process.stdout.write(JSON.stringify(handle(JSON.parse(fs.readFileSync(0, 'utf8'))))); }
  catch (error) { process.stderr.write('Vault hook failed: ' + error.message + '\n'); process.exitCode = 1; }
}
