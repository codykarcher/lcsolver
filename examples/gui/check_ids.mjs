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

let bad = 0;
for (const file of readdirSync(HERE).filter((f) => f.endsWith('_test.html'))) {
  const src = readFileSync(join(HERE, file), 'utf8');
  const have = new Set([...src.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));
  const used = new Set(ACCESSORS.flatMap((re) => [...src.matchAll(re)].map((m) => m[1])));
  const missing = [...used].filter((id) => !have.has(id));
  if (missing.length) { bad += missing.length; console.log(`  ${file}: MISSING ${missing.join(', ')}`); }
  console.log(`${file.padEnd(24)} ${String(used.size).padStart(3)} ids used`);
}
console.log(bad ? `FAIL: ${bad} missing id(s)` : 'PASS: every id a page reaches for exists');
process.exit(bad ? 1 : 0);
