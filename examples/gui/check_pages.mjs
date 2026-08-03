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
const A = await import(join(HERE, 'components/aircraft.js'));
const D = await import(join(HERE, 'components/d8_aircraft.js'));
const deckOf = (f) => A.deckFromSolve(
  JSON.parse(readFileSync(join(HERE, 'decks', f), 'utf8')));

/**
 * page -> { build: the variants it can build, vars: what it calls the object }
 *
 * `vars` is needed because a page reads its object through whatever local name
 * it gave it -- `engine.userData.rFan` on an engine page, `u.length` after
 * destructuring on a fuselage page -- and the whole point of this check is to
 * see the same field names the page actually writes.
 */
/**
 * The aeroplane pages read the DUCT's fields, and nothing was watching them.
 *
 * `u.parts.fuselage.userData.duct` is written by the carve and read by the page
 * to print what it did. When the cut was rebuilt those field names changed --
 * `floor`, `halfWidth`, `deepFrom` became `throatX`, `spacing`, `rThroat` --
 * and the page threw inside `build()` on the first one it touched. Nothing
 * reaches the scene after that, so the window comes up dead: no aeroplane, no
 * orbit, no error anywhere obvious. Exactly the failure this file exists for,
 * on the two pages it had no builder for.
 */
const PAGES = {
  'd8_aircraft_test.html': {
    build: [() => D.d8Aircraft(deckOf('b737_d8_solve.json'), { sitOnGround: false })],
    vars: ['u.parts.fuselage.userData.duct', 'dz'],
  },
  'b737_test.html': {
    build: [() => A.conventionalAircraft(deckOf('b737_conventional_solve.json'),
                                         { sitOnGround: false })],
    vars: ['u.parts.fuselage.userData.duct', 'dz'],
  },
  'turbofan_test.html': { build: [() => E.turbofan({}), () => E.bareTurbofan({})],
                          vars: ['engine.userData', 'd'] },
  'pylon_test.html':    { build: [() => E.turbofan({}), () => E.bareTurbofan({})],
                          vars: ['engine.userData', 'd'] },
  'turbojet_test.html': { build: [() => E.turbojet({})], vars: ['engine.userData', 'd'] },
  'turboshaft_test.html': { build: [() => E.turboshaft({})], vars: ['engine.userData', 'd'] },
  'piston_test.html':   { build: [() => E.pistonEngine({})], vars: ['engine.userData', 'd'] },
  'motor_test.html':    { build: [() => E.electricMotor({})], vars: ['engine.userData', 'd'] },
  'wing_test.html':     { build: [() => W.wing(), () => W.horizontalTail(),
                                  () => W.verticalTail(),
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

/**
 * Every name a page uses that a component exports must be IMPORTED by it.
 *
 * A page that uses `FUSELAGE_TITLES` without importing it throws a
 * ReferenceError the moment `build()` runs -- the same blank screen this file
 * was written for, and this file did not catch it, because the field check
 * cannot run a page it cannot load. It happened when a two-page edit matched
 * the import line on one page and not the other, so one page got the new name
 * and the other got the use without the declaration.
 *
 * Done by name rather than by parsing: collect what the components export, then
 * look for those names used in a page that does not import them. That misses a
 * misspelt local, but it catches exactly the class of mistake that a
 * search-and-replace across several pages makes.
 */
const MODULES = ['decals.js', 'fuselage.js', 'wing.js', 'engines.js',
                 'paint.js', 'materials.js', 'geom.js', 'aircraft.js',
                 'd8_aircraft.js', 'landing_gear.js'];
const exported = new Set();
for (const m of MODULES) {
  let mod;
  try { mod = await import(join(HERE, 'components', m)); } catch { continue; }
  for (const k of Object.keys(mod)) exported.add(k);
}
for (const file of readdirSync(HERE).filter((f) => f.endsWith('_test.html'))) {
  const whole = readFileSync(join(HERE, file), 'utf8');
  // Only the module script. The markup around it is full of the same words as
  // element ids and label text -- `wing`, `all`, `tire` -- and scanning it made
  // this fire on seven pages that were perfectly fine.
  const src = (whole.match(/<script[^>]*type=["']module["'][^>]*>([\s\S]*?)<\/script>/) || [, ''])[1];
  // What this page pulls in, from every import statement it has.
  const imported = new Set();
  for (const m of src.matchAll(/import\s*(?:\*\s*as\s*(\w+)|\{([^}]*)\}|(\w+))\s*from/g)) {
    if (m[1]) imported.add(m[1]);
    if (m[3]) imported.add(m[3]);
    if (m[2]) for (const n of m[2].split(',')) {
      const name = n.trim().split(/\s+as\s+/).pop().trim();
      if (name) imported.add(name);
    }
  }
  // Strip comments, strings and the import statements themselves, so a name
  // mentioned in prose or renamed on the way in does not count.
  const code = src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/\/\/[^\n]*/g, ' ')
    .replace(/import[\s\S]*?from\s*['"][^'"]*['"];?/g, ' ')
    .replace(/'[^'\n]*'|"[^"\n]*"|`[^`]*`/g, ' ');
  // A bare use, not `u.parts.wing` and not `{ wing: ... }`. Component exports
  // share plenty of names with ordinary properties -- `wing`, `dispose`, `all`
  // -- and counting those had this firing on all thirteen pages at once.
  // A name the page declares for itself is its own, not a missing import.
  const local = new Set();
  for (const m of code.matchAll(/\b(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)/g)) {
    local.add(m[1]);
  }
  const undeclared = [...exported].filter((name) =>
    !imported.has(name) && !local.has(name)
    && new RegExp(`(?<![.\\w$])${name}(?![\\w$])(?!\\s*:)`).test(code));
  if (undeclared.length) {
    bad++;
    console.log(`  ${file}: USES BUT DOES NOT IMPORT ${undeclared.join(', ')}`);
  }
}

for (const file of readdirSync(HERE).filter((f) => f.endsWith('_test.html'))) {
  const page = PAGES[file];
  if (!page) { console.log(`${file.padEnd(24)} (no builder registered)`); continue; }
  const src = readFileSync(join(HERE, file), 'utf8');
  const fields = [...new Set(page.vars.flatMap((v) => {
    const re = new RegExp(`\\b${v.replace(/\./g, '\\.')}\\.([A-Za-z_$][\\w$]*)`, 'g');
    return [...src.matchAll(re)].map((m) => m[1]);
  }))];
  for (const make of page.build) {
    const built = make().userData;
    // `dz` is the page's own name for the duct, so those fields are looked for
    // there rather than on the aeroplane.
    const u = page.vars.includes('dz')
      ? { ...built, ...(built.parts?.fuselage?.userData?.duct ?? {}) }
      : built;
    const missing = fields.filter((f) => u[f] === undefined);
    if (missing.length) { bad++; console.log(`  ${file}: MISSING ${missing.join(', ')}`); }
  }
  console.log(`${file.padEnd(24)} ${String(fields.length).padStart(2)} fields, `
              + `${page.build.length} variant(s)`);
}
console.log(bad ? `FAIL: ${bad} problem(s)`
                : 'PASS: every field the pages read exists, and every name they use is imported');
process.exit(bad ? 1 : 0);
