/**
 * A page's slider defaults must reproduce the component's own defaults.
 *
 *     node check_defaults.mjs
 *
 * This exists because of a specific failure. The D8's aft roof dish was turned
 * off in the component -- `trough: 0` -- but the page's slider was left at 0.30
 * and the page passes every slider back in on every build. So the model said
 * there was no channel, the check agreed, and the aeroplane on screen had a
 * channel down its back. Nothing was wrong with either half on its own.
 *
 * Two sources of truth for the same number is the whole bug, and it cannot be
 * fixed by being careful. So: read the page's defaults, build a body with them,
 * build a body with no arguments at all, and require the two to be the same
 * shape. Anything that drifts shows up on the next run rather than on screen.
 */
import { readFileSync, readdirSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));
const F = await import(join(HERE, 'components/fuselage.js'));
const W = await import(join(HERE, 'components/wing.js'));

/** page -> the builder it drives, and what it calls the body-size slider. */
const PAGES = {
  'fuselage_test.html': { build: F.jetlinerFuselage, size: 'rad' },
  'd8_test.html': { build: F.d8Fuselage, size: 'rad' },
  // A lifting surface has no single size slider and no `shape` object -- its
  // parameters are the planform itself -- so it is compared through its own
  // list rather than through the fuselage convention.
  // Two lists, because the page's wing state is described by both. Its tail
  // state is a different parameter set entirely and is check_toggle's job.
  'wing_test.html': { build: W.liftingSurface, flat: ['NUMS', 'WING_NUMS'] },
};

/** Pull `const NAME = [ 'a', 'b' ]` out of a page's script. */
function listOf(src, name) {
  const m = src.match(new RegExp(`const ${name} = \\[([^\\]]*)\\]`, 's'));
  return m ? [...m[1].matchAll(/'(\w+)'/g)].map((x) => x[1]) : [];
}

let bad = 0;
for (const file of readdirSync(HERE).filter((f) => f.endsWith('_test.html'))) {
  const page = PAGES[file];
  if (!page) continue;
  const src = readFileSync(join(HERE, file), 'utf8');

  const sliders = Object.fromEntries(
    [...src.matchAll(/<input type="range" id="(\w+)"[^>]*value="([-\d.]+)"/g)]
      .map((m) => [m[1], parseFloat(m[2])]));

  const deck = (Array.isArray(page.flat) ? page.flat : [page.flat ?? 'DECK'])
    .flatMap((n) => listOf(src, n));
  const shapeKeys = page.flat ? [] : listOf(src, 'SHAPE');
  const args = page.flat ? {} : { shape: {} };
  for (const k of deck) {
    if (sliders[k] === undefined) continue;
    if (k === page.size) args.radius = sliders[k]; else args[k] = sliders[k];
  }
  for (const k of shapeKeys) {
    if (sliders[k] !== undefined) args.shape[k] = sliders[k];
  }

  const fromPage = page.build(args).userData;
  const fromDefaults = page.build({}).userData;
  if (page.flat) {
    // Compared on the geometry, same as the bodies: sample the built surface at
    // a spread of spanwise stations and require the two to agree.
    let worstF = 0, whatF = '';
    for (let i = 0; i <= 40; i++) {
      const t = i / 40;
      const a = fromPage.at(t), b = fromDefaults.at(t);
      for (const k of ['chord', 'xLE', 'y', 'twist']) {
        if (Math.abs(a[k] - b[k]) > worstF) { worstF = Math.abs(a[k] - b[k]); whatF = `${k} at eta ${t.toFixed(2)}`; }
      }
    }
    if (worstF > 1e-9) {
      bad++;
      console.log(`  ${file}: page defaults differ from the component's ` +
                  `by ${worstF.toExponential(2)} (${whatF})`);
      for (const k of deck) {
        const mine = fromDefaults.planform?.[k];
        if (mine != null && sliders[k] !== undefined && Math.abs(mine - sliders[k]) > 1e-12) {
          console.log(`      ${k}: slider ${sliders[k]}, component ${mine}`);
        }
      }
    }
    console.log(`${file.padEnd(24)} ${String(deck.length).padStart(2)} defaults checked`);
    continue;
  }

  // Compared on the geometry rather than on the parameter lists, because that
  // is what actually differs -- a page can name a parameter the component has
  // renamed and quietly pass nothing at all.
  let worst = 0, worstAt = '';
  const L = fromDefaults.length;
  if (Math.abs(fromPage.length - L) > 1e-9) {
    worst = Math.abs(fromPage.length - L); worstAt = 'length';
  }
  for (let i = 0; i <= 120; i++) {
    const z = -L * i / 120;
    const a = fromPage.shapeAt(z), b = fromDefaults.shapeAt(z);
    for (const [what, d] of [['r', a.r - b.r], ['yc', a.yc - b.yc]]) {
      if (Math.abs(d) > worst) { worst = Math.abs(d); worstAt = `${what} at z ${z.toFixed(1)}`; }
    }
    for (let j = 0; j < 24; j++) {
      const th = (j / 24) * Math.PI * 2;
      const d = fromPage.section(th, z) - fromDefaults.section(th, z);
      if (Math.abs(d) > worst) { worst = Math.abs(d); worstAt = `section at z ${z.toFixed(1)}`; }
    }
  }

  const names = [...deck, ...shapeKeys].filter((k) => sliders[k] !== undefined);
  if (worst > 1e-9) {
    bad++;
    console.log(`  ${file}: page defaults differ from the component's ` +
                `by ${worst.toExponential(2)} (${worstAt})`);
    // Say WHICH, or the message is a puzzle rather than a report.
    for (const k of shapeKeys) {
      const mine = fromDefaults.shapeParams?.[k];
      if (mine !== undefined && sliders[k] !== undefined
          && Math.abs(mine - sliders[k]) > 1e-12) {
        console.log(`      ${k}: slider ${sliders[k]}, component ${mine}`);
      }
    }
  }
  console.log(`${file.padEnd(24)} ${String(names.length).padStart(2)} defaults checked`);
}

console.log(bad ? `FAIL: ${bad} page(s) disagree with their component`
                : 'PASS: page defaults reproduce the component defaults');
process.exit(bad ? 1 : 0);
