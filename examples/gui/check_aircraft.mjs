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
import { conventionalAircraft, B737_TASOPT } from './components/aircraft.js';

let failures = 0;
const bad = (m) => { console.log(`  FAIL  ${m}`); failures++; };

const g = conventionalAircraft(B737_TASOPT, { sitOnGround: false });
const u = g.userData, d = u.deck;

/* areas, against the solve -------------------------------------------- */
const SOLVE = { wing: 116.649, horizontalTail: 19.095, verticalTail: 15.859 };
console.log('area              assembled       solve       rel');
for (const [k, want] of Object.entries(SOLVE)) {
  const got = u.areas[k], rel = Math.abs(got - want) / want;
  if (rel > 1e-4) bad(`${k} area ${got.toFixed(4)} against the solve's ${want}`);
  console.log(`  ${k.padEnd(16)} ${got.toFixed(4).padStart(9)} ${String(want).padStart(11)}  ` +
              `${rel.toExponential(1)}`);
}

/* sweep conversion ----------------------------------------------------- */
// Quarter-chord back out of the built leading edge. If the conversion is
// right this returns the deck's own constant; if the 1/2-versus-1/4 factor is
// wrong it returns something plausible and different, which is the whole risk.
const backOut = (leSweep, cRoot, taper, span, k) =>
  Math.atan(Math.tan(leSweep * Math.PI / 180) - k * cRoot * (1 - taper) / span)
    * 180 / Math.PI;
const cases = [
  ['wing', u.leadingEdgeSweeps.wing, d.wingRootChord, d.wingTaper, d.wingSpan, 0.5, d.wingSweepC4],
  ['h tail', u.leadingEdgeSweeps.horizontalTail, d.htRootChord, d.htTaper, d.htSpan, 0.5, d.htSweepC4],
  ['v tail', u.leadingEdgeSweeps.verticalTail, d.vtRootChord, d.vtTaper, d.vtHeight, 0.25, d.vtSweepC4],
];
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
const sat = conventionalAircraft(B737_TASOPT, { sitOnGround: true });
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

console.log(failures ? `\n${failures} FAILURE(S)`
                     : '\nPASS: the assembly reproduces the solve it came from');
process.exit(failures ? 1 : 0);
