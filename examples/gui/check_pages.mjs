/**
 * Smoke test: every userData field a test page reads must exist on the object
 * that page builds.
 *
 *     node check_pages.mjs
 *
 * A page reading an absent field throws inside build(), which aborts before
 * anything reaches the scene -- so the failure looks like a blank screen with
 * no error anywhere obvious. That has now happened twice, once when the podded
 * turbofan did not carry the bare engine's properties forward. Cheap to check,
 * miserable to debug.
 *
 * Requires three: run from a directory where it resolves, or
 *     npm --prefix . i three && node check_pages.mjs
 */
import { readFileSync, readdirSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));
const E = await import(join(HERE, 'components/engines.js'));

/** page -> the variants it can build */
const PAGES = {
  'turbofan_test.html': [() => E.turbofan({}), () => E.bareTurbofan({})],
  'pylon_test.html': [() => E.turbofan({}), () => E.bareTurbofan({})],
  'turbojet_test.html': [() => E.turbojet({})],
  'turboshaft_test.html': [() => E.turboshaft({})],
  'piston_test.html': [() => E.pistonEngine({})],
  'motor_test.html': [() => E.electricMotor({})],
};

let bad = 0;
for (const file of readdirSync(HERE).filter((f) => f.endsWith('_test.html'))) {
  const builders = PAGES[file];
  if (!builders) { console.log(`${file.padEnd(24)} (no builder registered)`); continue; }
  const src = readFileSync(join(HERE, file), 'utf8');
  // Pages read either engine.userData.x or, after destructuring, d.x
  const fields = [...new Set([
    ...[...src.matchAll(/engine\.userData\.([A-Za-z_$][\w$]*)/g)].map((m) => m[1]),
    ...[...src.matchAll(/\bd\.([A-Za-z_$][\w$]*)/g)].map((m) => m[1]),
  ])];
  for (const make of builders) {
    const u = make().userData;
    const missing = fields.filter((f) => u[f] === undefined);
    if (missing.length) { bad++; console.log(`  ${file}: MISSING ${missing.join(', ')}`); }
  }
  console.log(`${file.padEnd(24)} ${String(fields.length).padStart(2)} fields, `
              + `${builders.length} variant(s)`);
}
console.log(bad ? `FAIL: ${bad} variant(s) missing fields`
                : 'PASS: every field the pages read exists');
process.exit(bad ? 1 : 0);
