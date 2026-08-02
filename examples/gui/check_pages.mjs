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
const F = await import(join(HERE, 'components/fuselage.js'));
const W = await import(join(HERE, 'components/wing.js'));

/**
 * page -> { build: the variants it can build, vars: what it calls the object }
 *
 * `vars` is needed because a page reads its object through whatever local name
 * it gave it -- `engine.userData.rFan` on an engine page, `u.length` after
 * destructuring on a fuselage page -- and the whole point of this check is to
 * see the same field names the page actually writes.
 */
const PAGES = {
  'turbofan_test.html': { build: [() => E.turbofan({}), () => E.bareTurbofan({})],
                          vars: ['engine.userData', 'd'] },
  'pylon_test.html':    { build: [() => E.turbofan({}), () => E.bareTurbofan({})],
                          vars: ['engine.userData', 'd'] },
  'turbojet_test.html': { build: [() => E.turbojet({})], vars: ['engine.userData', 'd'] },
  'turboshaft_test.html': { build: [() => E.turboshaft({})], vars: ['engine.userData', 'd'] },
  'piston_test.html':   { build: [() => E.pistonEngine({})], vars: ['engine.userData', 'd'] },
  'motor_test.html':    { build: [() => E.electricMotor({})], vars: ['engine.userData', 'd'] },
  'wing_test.html':     { build: [() => W.wing(), () => W.horizontalTail(),
                                  () => W.liftingSurface({ kink: null, mirror: false })],
                          vars: ['surface.userData', 'u'] },
  'd8_test.html':       { build: [() => F.d8Fuselage({}),
                                  () => F.d8Fuselage({ detail: true }),
                                  () => F.d8Fuselage({ radius: 1.1, fineness: 8,
                                                       shape: { bubble: 0.1, trough: 0 } })],
                          vars: ['body.userData', 'u'] },
  'fuselage_test.html': { build: [() => F.jetlinerFuselage({}),
                                  () => F.jetlinerFuselage({ detail: true }),
                                  () => F.jetlinerFuselage({
                                    radius: 3.2, fineness: 13, noseD: 2.6 })],
                          vars: ['body.userData', 'u', 'u.shapeParams'] },
};

let bad = 0;
for (const file of readdirSync(HERE).filter((f) => f.endsWith('_test.html'))) {
  const page = PAGES[file];
  if (!page) { console.log(`${file.padEnd(24)} (no builder registered)`); continue; }
  const src = readFileSync(join(HERE, file), 'utf8');
  const fields = [...new Set(page.vars.flatMap((v) => {
    const re = new RegExp(`\\b${v.replace('.', '\\.')}\\.([A-Za-z_$][\\w$]*)`, 'g');
    return [...src.matchAll(re)].map((m) => m[1]);
  }))];
  for (const make of page.build) {
    const u = make().userData;
    const missing = fields.filter((f) => u[f] === undefined);
    if (missing.length) { bad++; console.log(`  ${file}: MISSING ${missing.join(', ')}`); }
  }
  console.log(`${file.padEnd(24)} ${String(fields.length).padStart(2)} fields, `
              + `${page.build.length} variant(s)`);
}
console.log(bad ? `FAIL: ${bad} variant(s) missing fields`
                : 'PASS: every field the pages read exists');
process.exit(bad ? 1 : 0);
