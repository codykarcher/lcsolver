/**
 * Every id a page reaches for must exist in its own markup.
 *
 *     node check_ids.mjs
 *
 * A page asking for a control that is not there throws inside build(), which
 * aborts before anything reaches the scene, so the failure looks like a blank
 * screen. check_pages.mjs catches the same class of thing for userData fields;
 * this catches it for the DOM.
 *
 * It exists because a stale `num('cabinFlat')` survived a rename and was not
 * caught: the sweep at the time only looked at `el(...)`, and the reference was
 * through `num(...)`. Every accessor has to be listed or the check is decorative.
 */
import { readFileSync, readdirSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ACCESSORS = [/\bel\('([A-Za-z0-9_]+)'\)/g,
                   /\bnum\('([A-Za-z0-9_]+)'\)/g,
                   /getElementById\('([A-Za-z0-9_]+)'\)/g];

/**
 * Which controls the BUILD reads, and whether anything listens to them.
 *
 * A control that exists, is read when the model is built, and is attached to no
 * listener is the quietest fault a page can have: it looks right, it does
 * nothing, and every other check passes. That is what a `windscreen` checkbox
 * on the D8 page did -- present, read, and dead, because it was left out of the
 * array the listeners are registered from.
 *
 * Scoped to `build` deliberately. A control read every frame in the animation
 * loop -- the engine pages' `spin` -- takes effect the moment it is ticked and
 * needs no listener at all; asking those to be wired reports six faults that
 * are not there.
 */
function bodyOf(src, header) {
  const i = src.indexOf(header);
  if (i < 0) return '';
  let j = src.indexOf('{', i), depth = 0;
  for (let k = j; k < src.length; k++) {
    if (src[k] === '{') depth++;
    else if (src[k] === '}' && --depth === 0) return src.slice(j, k);
  }
  return '';
}

/**
 * Every identifier named near an `addEventListener`.
 *
 * Read as a WINDOW OF LINES rather than by matching the whole registration as
 * one pattern. The pattern version looked tidier and was wrong: given slack to
 * find the `addEventListener`, it backtracked the array's closing bracket and
 * swallowed unrelated statements, which quietly marked as wired the very
 * control it was written to catch.
 *
 * Three lines is enough for how these pages are written -- the array and the
 * call that consumes it sit together -- and being bounded, it cannot run away.
 * It also picks up the event names, which is harmless: no control is called
 * 'change'.
 */
function wiredIn(src) {
  const lines = src.split('\n');
  const wired = new Set();
  lines.forEach((line, i) => {
    if (!line.includes('addEventListener')) return;
    const window = lines.slice(Math.max(0, i - 3), i + 1).join('\n');
    for (const m of window.matchAll(/'([A-Za-z0-9_]+)'/g)) wired.add(m[1]);
  });
  // Spreads like [...DECK, ...SHAPE] name their ids in a const array above.
  for (const m of src.matchAll(/const\s+([A-Z][A-Z0-9_]*)\s*=\s*\[([^\]]*)\]/g)) {
    if (!new RegExp(`\\.\\.\\.${m[1]}\\b`).test(src)) continue;
    for (const q of m[2].matchAll(/'([A-Za-z0-9_]+)'/g)) wired.add(q[1]);
  }
  return wired;
}

let bad = 0;
for (const file of readdirSync(HERE).filter((f) => f.endsWith('_test.html'))) {
  const src = readFileSync(join(HERE, file), 'utf8');
  const have = new Set([...src.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));
  const used = new Set(ACCESSORS.flatMap((re) => [...src.matchAll(re)].map((m) => m[1])));
  const missing = [...used].filter((id) => !have.has(id));
  if (missing.length) { bad += missing.length; console.log(`  ${file}: MISSING ${missing.join(', ')}`); }

  const build = bodyOf(src, 'function build(');
  const readByBuild = new Set(ACCESSORS.flatMap((re) => [...build.matchAll(re)].map((m) => m[1])));
  const controls = new Set([...src.matchAll(/<(?:input|select)\b[^>]*\bid="([^"]+)"/g)].map((m) => m[1]));

  const wired = wiredIn(src);

  const dead = [...controls].filter((id) => readByBuild.has(id) && !wired.has(id));
  if (dead.length) {
    bad += dead.length;
    console.log(`  ${file}: DEAD CONTROL ${dead.join(', ')} -- the build reads it, nothing listens`);
  }
  console.log(`${file.padEnd(24)} ${String(used.size).padStart(3)} ids used, ` +
              `${String(readByBuild.size).padStart(2)} read by build, all wired`);
}
console.log(bad ? `FAIL: ${bad} problem(s)`
                : 'PASS: every id exists and every control the build reads is wired');
process.exit(bad ? 1 : 0);
