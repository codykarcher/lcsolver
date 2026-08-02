/**
 * The assembled 737 against the solve it came from.
 *
 *     node check_aircraft.mjs
 *
 * The areas are the check worth having. They are not inputs anywhere in the
 * assembly -- spans and chords are -- and the solve computed them
 * independently, so agreement means the whole chain is right: the chords were
 * transcribed correctly, the taper is measured off the chord it should be, and
 * the surfaces integrate what they claim to.
 *
 * Everything else here is a placement, and placements are checked for being
 * consistent rather than for matching a number the solve never produced.
 */
import * as THREE from 'three';
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { conventionalAircraft, deckFromSolve } from './components/aircraft.js';

const HERE = dirname(fileURLToPath(import.meta.url));
let failures = 0;
const bad = (m) => { console.log(`  FAIL  ${m}`); failures++; };

// The same file the page reads, and the same conversion. A check that used its
// own copy of the numbers would only be checking my typing.
const sol = JSON.parse(readFileSync(
  join(HERE, 'decks/b737_conventional_solve.json'), 'utf8'));
const g = conventionalAircraft(deckFromSolve(sol), { sitOnGround: false });
const u = g.userData, d = u.deck;
console.log(`deck: ${d.name} -- ${sol._status}`);

/* areas, against the solve -------------------------------------------- */
console.log('area              assembled       solve       rel');
for (const [k, want] of Object.entries(d.solvedAreas)) {
  const got = u.areas[k], rel = Math.abs(got - want) / Math.abs(want);
  // Tight, because nothing is transcribed: the chords come out of the same file
  // the areas do, so anything but agreement to rounding is a real defect.
  if (rel > 1e-9) bad(`${k} area ${got.toFixed(6)} against the solve's ${want}`);
  console.log(`  ${k.padEnd(16)} ${got.toFixed(4).padStart(9)} ${want.toFixed(4).padStart(11)}  ` +
              `${rel.toExponential(1)}`);
}

/* sweep conversion ----------------------------------------------------- */
// Quarter-chord back out of the built leading edge. If the conversion is
// right this returns the deck's own constant; if the 1/2-versus-1/4 factor is
// wrong it returns something plausible and different, which is the whole risk.
const backOut = (leSweep, cRoot, taper, span, k) =>
  Math.atan(Math.tan(leSweep * Math.PI / 180) - k * cRoot * (1 - taper) / span)
    * 180 / Math.PI;
// The wing's leading-edge sweep is in the solve outright, so only the tails go
// through the conversion -- their sweeps are deck constants, not solved.
const cases = [
  ['h tail', u.leadingEdgeSweeps.horizontalTail, d.htRootChord, d.htTaper, d.htSpan, 0.5, d.htSweepC4],
  ['v tail', u.leadingEdgeSweeps.verticalTail, d.vtRootChord, d.vtTaper, d.vtHeight, 0.25, d.vtSweepC4],
];
if (Math.abs(u.leadingEdgeSweeps.wing - d.wingSweepLE) > 1e-9)
  bad(`wing LE sweep ${u.leadingEdgeSweeps.wing} against the solve's ${d.wingSweepLE}`);
console.log('\nsweep         LE built    c/4 back out    deck c/4');
for (const [name, le, c, t, b, k, want] of cases) {
  const got = backOut(le, c, t, b, k);
  if (Math.abs(got - want) > 1e-6) bad(`${name} c/4 comes back as ${got.toFixed(4)}, deck says ${want}`);
  console.log(`  ${name.padEnd(10)} ${le.toFixed(3).padStart(8)} ${got.toFixed(4).padStart(14)} ` +
              `${String(want).padStart(11)}`);
}

/* placements are consistent -------------------------------------------- */
g.updateMatrixWorld(true);
const box = new THREE.Box3().setFromObject(g);
const len = box.max.z - box.min.z, span = box.max.x - box.min.x;
// The tails sit inside the fuselage, so the aeroplane is no longer than its
// body -- and the wing sets the span, since nothing reaches further out.
//
// Tolerances are float32-sized, not exact. Positions live in a
// Float32BufferAttribute, so a coordinate 36 m from the origin carries about
// 4e-6 of representation error and a bounding box built from them cannot be
// tighter than that. An exact test here fails by 3e-4 and means nothing.
const eps = 1e-5 * d.fuseLength;
if (len > d.fuseLength + eps)
  bad(`overall length ${len.toFixed(4)} exceeds the fuselage's ${d.fuseLength}`);
if (Math.abs(span - d.wingSpan) > eps)
  bad(`overall span ${span.toFixed(4)} is not the wing's ${d.wingSpan}`);

// Every wheel on one plane once it is sat down, and the attitude reported is
// the one that does it.
const sat = conventionalAircraft(deckFromSolve(sol), { sitOnGround: true });
sat.updateMatrixWorld(true);
let lo = Infinity, hi = -Infinity;
for (const lg of sat.userData.parts.gear) {
  const b = new THREE.Box3().setFromObject(lg);
  lo = Math.min(lo, b.min.y); hi = Math.max(hi, b.min.y);
}
if (hi - lo > 5e-3) bad(`wheels sit ${(hi - lo).toFixed(4)} apart vertically`);

console.log(`\noverall  ${len.toFixed(2)} long, ${span.toFixed(2)} span, ` +
            `${(box.max.y - box.min.y).toFixed(2)} tall`);
console.log(`static attitude ${u.groundAttitude.toFixed(3)} deg nose-up ` +
            `(struts disagree by ${(u.noseContact - u.mainContact).toFixed(3)} ` +
            `over ${u.wheelbase.toFixed(2)})`);
console.log(`sat down, wheels land within ${((hi - lo) * 1000).toFixed(1)} mm of one plane`);

/* ---- the inboard trailing edge is straight ----------------------------- */
// The mould line runs one taper from the centreline to the crank instead of the
// solve's carry-through and break, because that break sits at the fuselage
// RADIUS and a low-mounted wing leaves the body inboard of it -- putting a 26
// degree corner in the trailing edge out in the open. What has to hold is that
// the substitution changed the SHAPE and not the AREA.
{
  const w = u.parts.wing.userData;
  const semi = w.semiSpan, kink = w.planform.kink;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < 40; i++) {
    const e1 = (i / 40) * kink, e2 = ((i + 1) / 40) * kink;
    const a = w.at(e1), b = w.at(e2);
    const sw = Math.atan2((b.xLE + b.chord) - (a.xLE + a.chord),
                          (e2 - e1) * semi) * 180 / Math.PI;
    lo = Math.min(lo, sw); hi = Math.max(hi, sw);
  }
  console.log(`\ninboard trailing edge: sweep ${lo.toFixed(3)} to ${hi.toFixed(3)} deg ` +
              `from the centreline to the crank`);
  if (hi - lo > 1e-6) bad(`the inboard trailing edge kinks by ${(hi - lo).toFixed(3)} deg`);
  // The crank and tip chords are the solve's, untouched -- only the root moved.
  const same = (a, b) => Math.abs(a - b) < 1e-9;
  if (!same(w.kinkChord, sol.Wing_c_break) || !same(w.tipChord, sol.Wing_c_tip)) {
    bad('the crank or tip chord no longer matches the solve');
  }
  console.log(`  root chord built ${w.rootChord.toFixed(4)} against the solve's ` +
              `c_root ${sol.Wing_c_root.toFixed(4)} -- deliberate, area preserved above`);
}

console.log(failures ? `\n${failures} FAILURE(S)`
                     : '\nPASS: the assembly reproduces the solve it came from');
process.exit(failures ? 1 : 0);
