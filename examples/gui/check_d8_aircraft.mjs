/**
 * Does the D8 assembly reproduce the solve it came from?
 *
 * The same question as the conventional aeroplane's check, and the same answer
 * matters most: areas are not inputs anywhere. They fall out of spans and
 * chords, the solve computed them independently, and agreement to machine
 * precision is what says the whole conversion is right.
 *
 * The fins are where a solve can be misread by a factor of two, so that reading
 * is checked rather than assumed: `VT_S_vt` describes ONE surface -- its own
 * span, root chord and taper reproduce it exactly -- and `n_vt` says how many
 * there are. The total is the product.
 */
import { readFileSync } from 'fs';
import * as THREE from 'three';
import { deckFromSolve } from './components/aircraft.js';
import { d8Aircraft } from './components/d8_aircraft.js';

const sol = JSON.parse(readFileSync(new URL('./decks/b737_d8_solve.json', import.meta.url), 'utf8'));
const deck = deckFromSolve(sol);
const g = d8Aircraft(deck, { sitOnGround: false });
const u = g.userData, d = u.deck;
const fu = u.parts.fuselage.userData;

let failures = 0;
const bad = (m) => { console.log(`  FAIL  ${m}`); failures++; };

/* ---- the body is the section the solve describes ----------------------- */
const halfW = 2.6105 && d.fuseHalfWidth, halfH = d.fuseHalfHeight;
const mid = -d.fuseLength * 0.45;
const gotW = fu.halfWidthAt(mid), gotH = (fu.crownAt(mid) - fu.keelAt(mid)) / 2;
console.log(`body ${(2 * gotW).toFixed(3)} wide by ${(2 * gotH).toFixed(3)} tall by ` +
            `${fu.length.toFixed(2)} long`);
console.log(`     solve says ${(2 * halfW).toFixed(3)} by ${(2 * halfH).toFixed(3)} by ` +
            `${d.fuseLength.toFixed(2)}`);
if (Math.abs(gotW - halfW) > 1e-3) bad(`half-width ${gotW.toFixed(4)}, solve ${halfW.toFixed(4)}`);
if (Math.abs(gotH - halfH) > 1e-3) bad(`half-height ${gotH.toFixed(4)}, solve ${halfH.toFixed(4)}`);
if (Math.abs(fu.length - d.fuseLength) > 1e-6) bad(`length ${fu.length}, solve ${d.fuseLength}`);

// It has to be a double bubble, and wider than it is tall, or it is not a D8.
if (!fu.isDoubleBubble) bad('the body is not a double bubble');
if (gotW <= gotH) bad(`the body is ${(gotW / gotH).toFixed(2)} wide for its height -- not a D8`);
console.log(`     width over height ${(gotW / gotH).toFixed(3)}, ` +
            `${d.seatsAbreast.toFixed(1)} abreast in ${d.cabinRows.toFixed(1)} rows`);

/* ---- areas ------------------------------------------------------------- */
console.log('\narea            built        solve         rel');
for (const [key, want] of Object.entries(d.solvedAreas)) {
  const got = u.areas[key];
  const rel = Math.abs(got - want) / want;
  console.log(`  ${key.padEnd(15)}${got.toFixed(4).padStart(9)}${want.toFixed(4).padStart(13)}` +
              `${rel.toExponential(1).padStart(12)}`);
  if (rel > 1e-9) bad(`${key} is ${got}, solve says ${want}`);
}

/* ---- the fins are two, halved ------------------------------------------ */
{
  const fins = u.parts.verticalTails;
  console.log(`\nfins: ${fins.length} (solve says n_vt = ${sol.n_vt}), each ` +
              `${fins[0].userData.height.toFixed(3)} tall by ` +
              `${fins[0].userData.rootChord.toFixed(3)} root, ` +
              `${fins[0].userData.area.toFixed(4)} m2`);
  if (fins.length !== sol.n_vt) bad(`${fins.length} fins, solve says ${sol.n_vt}`);
  // Each fin is the solve's own planform, unhalved -- the reading that makes
  // b_vt, c_root_vt and lambda_vt reproduce S_vt, which they do exactly.
  const planform = 0.5 * d.vtHeight * d.vtRootChord * (1 + d.vtTaper);
  console.log(`      planform gives ${planform.toFixed(4)} against VT_S_vt ` +
              `${sol.VT_S_vt.toFixed(4)}; total ${(fins.length * planform).toFixed(4)}`);
  if (Math.abs(planform - sol.VT_S_vt) > 1e-9) {
    bad(`b_vt x c_root x (1+lambda)/2 is ${planform}, not VT_S_vt ${sol.VT_S_vt}`);
  }
  for (const f of fins) {
    if (Math.abs(f.userData.area - sol.VT_S_vt) > 1e-9) {
      bad(`a fin is ${f.userData.area} m2, not the solve's ${sol.VT_S_vt}`);
    }
  }
  // They must splay apart, not both lean the same way.
  const [a, b] = fins.map((f) => new THREE.Box3().setFromObject(f));
  if (a.min.x * b.max.x > 0) bad('both fins are on the same side');
  if (Math.sign(a.max.x - a.min.x) === 0) bad('a fin has no width');
}

/* ---- the engines sit on the afterbody, not under a wing ---------------- */
{
  const pods = u.parts.engines;
  const crown = fu.crownAt(-d.engineX);
  const b = new THREE.Box3().setFromObject(pods[0]);
  console.log(`\nengines at x ${d.engineX.toFixed(2)} of a ${d.fuseLength.toFixed(2)} m body ` +
              `(${(100 * d.engineX / d.fuseLength).toFixed(0)}% aft), y +/-${d.engineY.toFixed(3)}`);
  console.log(`     nacelle spans y ${b.min.y.toFixed(2)}..${b.max.y.toFixed(2)}, ` +
              `body crown there ${crown.toFixed(2)}`);
  if (b.max.y < crown) bad('the nacelles are entirely below the crown -- buried in the body');
  if (d.engineX < 0.6 * d.fuseLength) bad('the engines are not on the afterbody');
  // Inboard of the fins, as the arrangement has them.
  const finX = Math.abs(new THREE.Box3().setFromObject(u.parts.verticalTails[0]).min.x);
  if (d.engineY > finX) bad(`engines at y ${d.engineY} are outboard of the fins at ${finX}`);
  console.log(`     and inboard of the fins, which stand at x ${finX.toFixed(2)}`);
}

/* ---- how it sits ------------------------------------------------------- */
{
  const sat = d8Aircraft(deck, { sitOnGround: true });
  sat.updateMatrixWorld(true);
  let lowest = Infinity;
  for (const lg of sat.userData.parts.gear) {
    lowest = Math.min(lowest, new THREE.Box3().setFromObject(lg).min.y);
  }
  console.log(`\nstatic attitude ${u.groundAttitude.toFixed(3)} deg, wheels within ` +
              `${(1000 * Math.abs(lowest)).toFixed(1)} mm of the ground`);
  if (Math.abs(lowest) > 0.01) bad(`sat down, the lowest wheel is ${lowest.toFixed(3)} off the ground`);
}

console.log(failures ? `\nFAIL: ${failures} problem(s)`
                     : '\nPASS: the D8 reproduces the solve it came from');
process.exit(failures ? 1 : 0);
