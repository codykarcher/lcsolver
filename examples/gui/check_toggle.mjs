/**
 * The wing page's kind toggle must build the component it names.
 *
 *     node check_toggle.mjs
 *
 * check_defaults covers the page's slider defaults against the component's, but
 * it only ever sees ONE state of the page. This page has three, and the second
 * was wrong when it was added: the toggle switched the topology -- crank off --
 * and nothing else, so picking "horizontal tail" gave a 34 m trapezoid with the
 * wing's dihedral and washout. Those values are locked rather than sliders, so
 * no amount of checking sliders would have found it.
 */
import { readFileSync } from 'fs';
import { liftingSurface, verticalTail, horizontalTail, wing, defaults }
  from './components/wing.js';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
const HERE = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(HERE, 'wing_test.html'), 'utf8');
const sliders = Object.fromEntries(
  [...src.matchAll(/<input type="range" id="(\w+)"[^>]*value="([-\d.]+)"/g)]
    .map((m) => [m[1], parseFloat(m[2])]));
const listOf = (name) =>
  [...src.match(new RegExp(`const ${name} = \\[([^\\]]*)\\]`, 's'))[1]
    .matchAll(/'(\w+)'/g)].map((m) => m[1]);
const NUMS = listOf('NUMS');
const KIND = { wing: listOf('WING_NUMS'), tail: listOf('TAIL_NUMS'), fin: listOf('FIN_NUMS') };

// The page loads the component's defaults into the sliders on toggle, so
// reproduce that rather than using the markup values for the tail.
let bad = 0;
for (const [kind, ref] of [['wing', wing()], ['tail', horizontalTail()],
                           ['fin', verticalTail()]]) {
  // The page's own rule: shared keys plus the selected kind's, loaded from the
  // component's defaults on toggle and overridden by the sliders.
  const d = defaults[kind];
  const keys = [...NUMS, ...KIND[kind]];
  const vals = {};
  for (const k of keys) vals[k] = kind === 'wing' ? sliders[k] : (d[k] ?? sliders[k]);
  const p = { ...d, ...vals };
  const got = (kind === 'fin' ? verticalTail(p) : liftingSurface(p)).userData;
  const want = ref.userData;
  let worst = 0, what = '';
  for (let i = 0; i <= 40; i++) {
    const t = i / 40, a = got.at(t), b = want.at(t);
    for (const k of ['chord', 'xLE', 'y', 'twist']) {
      if (Math.abs(a[k] - b[k]) > worst) { worst = Math.abs(a[k] - b[k]); what = `${k} at eta ${t.toFixed(2)}`; }
    }
  }
  if (worst > 1e-12) { bad++; console.log(`  ${kind}: differs by ${worst.toExponential(2)} (${what})`); }
  console.log(`${kind.padEnd(5)}: identical to the component  ` +
              `[dihedral ${got.dihedral}, tip twist ${got.planform.twistTip}, ` +
              `t/c ${got.rootThickness}/${got.tipThickness}, ` +
              `S ${got.area.toFixed(2)}, AR ${got.aspectRatio.toFixed(2)}` +
              `${got.isFin ? ' (h^2/S)' : ''}]`);
}
console.log(bad ? `FAIL: ${bad} toggle state(s) build the wrong surface`
                : 'PASS: every toggle state builds the component it names');
process.exit(bad ? 1 : 0);
