/**
 * Do the closed-form sections agree with the search they replaced?
 *
 * `channelRadius` used to answer by forty steps of bisection on a predicate.
 * That was the single most expensive thing in the build -- carving the duct
 * asks the body about a million times whether a point is inside it -- and it
 * was never necessary: the channel is a stadium cut by a half-plane, so it is
 * convex, contains the centre, and a ray leaves it exactly once through one of
 * four pieces, each of which is a line or a circle.
 *
 * The bisection is kept HERE, as the thing to be agreed with. A closed form
 * that is subtly wrong on some corner of the parameter space would show up as a
 * shape that is almost right, which is the hardest kind of fault to see, so it
 * is checked against its predecessor over the whole space rather than spot
 * checked -- and the check is shown to fail on an answer that is off by a
 * tenth of a per cent.
 */
import { channelRadius } from './components/fuselage.js';

let failures = 0;
const bad = (m) => { console.log(`  FAIL  ${m}`); failures++; };

/** The forty-step bisection, verbatim as it stood. */
function bisected(halfSpacing, radius, axisY, cap) {
  const inside = (x, y) => {
    if (y > cap) return false;
    const dy = y - axisY, ax = Math.abs(x);
    return ax <= halfSpacing
      ? Math.abs(dy) <= radius
      : (ax - halfSpacing) * (ax - halfSpacing) + dy * dy <= radius * radius;
  };
  const reach = Math.hypot(halfSpacing + radius, Math.abs(axisY) + radius) + 1;
  return (th) => {
    const c = Math.cos(th), sn = Math.sin(th);
    if (!inside(0, 0)) return 1e-4;
    let lo = 0, hi = reach;
    for (let i = 0; i < 40; i++) {
      const m = (lo + hi) / 2;
      if (inside(m * c, m * sn)) lo = m; else hi = m;
    }
    return Math.max(lo, 1e-4);
  };
}

// Deterministic, so a failure is reproducible rather than "it went red once".
let seed = 12345;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

function sweep(answer) {
  let worst = 0, rays = 0, degenerate = 0, off = 0;
  seed = 12345;
  for (let k = 0; k < 4000; k++) {
    // The whole space, including the degenerate channels the aft body passes
    // through as it closes: capped below the centre, axis outside the radius.
    const halfSpacing = rnd() * 2;
    const radius = 0.05 + rnd() * 1.5;
    const axisY = (rnd() * 2 - 1) * 1.5;
    const cap = rnd() < 0.25 ? Infinity : (rnd() * 2 - 0.4);
    const ref = bisected(halfSpacing, radius, axisY, cap);
    for (let j = 0; j < 73; j++) {
      const th = (j / 73) * Math.PI * 2;
      const a = answer(halfSpacing, radius, axisY, cap, th), b = ref(th);
      rays++;
      if (a === 1e-4 && b === 1e-4) { degenerate++; continue; }
      const e = Math.abs(a - b);
      if (e > 1e-8) off++;
      if (e > worst) worst = e;
    }
  }
  return { worst, rays, degenerate, off };
}

const r = sweep(channelRadius);
console.log(`${r.rays} rays over 4000 random channels, ${r.degenerate} of them degenerate`);
console.log(`worst disagreement with the bisection: ${r.worst.toExponential(2)} ` +
            `(the bisection's own remaining interval is about 1e-10)`);
if (r.off) bad(`${r.off} rays disagree by more than 1e-8`);

/* ---- negative control ---------------------------------------------------- */
// A closed form that is right to within a tenth of a per cent is still wrong,
// and has to be caught, or this sweep is decoration.
const nudged = (hs, rad, ay, cap, th) => channelRadius(hs, rad, ay, cap, th) * 1.001;
const c = sweep(nudged);
console.log(`\nnegative control`);
console.log(`  a 0.1% error disagrees on ${c.off} rays` + (c.off > 0 ? '  ok' : '  NOT DETECTED'));
if (!c.off) bad('control: a 0.1% error was not detected');

console.log(failures ? `\n${failures} FAILED` : '\nall checks passed');
process.exit(failures ? 1 : 0);
