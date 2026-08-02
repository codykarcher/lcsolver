/**
 * Does the artwork lie ON the fin?
 *
 * A decal that is merely NEAR the surface is the failure this is built to
 * catch, and it is one that looks fine from most angles: a flat quad hung off a
 * curved fin touches in the middle and lifts at the corners, which reads as
 * correct until the camera comes round to a grazing view and the logo detaches.
 *
 * So the test is not "is it close" but "is every vertex the SAME distance from
 * the skin, on the outside, and is that distance the standoff it was asked
 * for". Constant standoff is the whole property; an average is not.
 */
import { readFileSync } from 'fs';
import * as THREE from 'three';
import { conventionalAircraft, deckFromSolve } from './components/aircraft.js';
import { surfaceArt, TAIL_ART, artNames } from './components/decals.js';

const sol = JSON.parse(readFileSync(new URL('./decks/b737_conventional_solve.json', import.meta.url), 'utf8'));
const craft = conventionalAircraft(deckFromSolve(sol), { sitOnGround: false });
const fin = craft.userData.parts.verticalTail;

let bad = 0;
const fail = (m) => { console.log('  FAIL ' + m); bad++; };

/* ---- the fin's own triangles, in its local frame ---------------------- */
const skin = fin.userData.skinMesh.geometry;
const P = skin.getAttribute('position').array;
const I = skin.getIndex().array;
const tris = [];
for (let t = 0; t < I.length; t += 3) {
  const v = [0, 1, 2].map((k) => new THREE.Vector3(
    P[3 * I[t + k]], P[3 * I[t + k] + 1], P[3 * I[t + k] + 2]));
  const c = v[0].clone().add(v[1]).add(v[2]).multiplyScalar(1 / 3);
  tris.push({ v, c });
}
console.log(`fin skin: ${tris.length} triangles, height ${fin.userData.height.toFixed(2)}`);

/** Distance from a point to a triangle, including its edges and corners. */
function distToTri(p, [a, b, c]) {
  const ab = b.clone().sub(a), ac = c.clone().sub(a), ap = p.clone().sub(a);
  const d1 = ab.dot(ap), d2 = ac.dot(ap);
  if (d1 <= 0 && d2 <= 0) return ap.length();
  const bp = p.clone().sub(b), d3 = ab.dot(bp), d4 = ac.dot(bp);
  if (d3 >= 0 && d4 <= d3) return bp.length();
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const t = d1 / (d1 - d3);
    return ap.clone().sub(ab.clone().multiplyScalar(t)).length();
  }
  const cp = p.clone().sub(c), d5 = ab.dot(cp), d6 = ac.dot(cp);
  if (d6 >= 0 && d5 <= d6) return cp.length();
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const t = d2 / (d2 - d6);
    return ap.clone().sub(ac.clone().multiplyScalar(t)).length();
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return p.clone().sub(b.clone().add(c.clone().sub(b).multiplyScalar(t))).length();
  }
  const den = 1 / (va + vb + vc);
  const q = a.clone()
    .add(ab.clone().multiplyScalar(vb * den))
    .add(ac.clone().multiplyScalar(vc * den));
  return p.clone().sub(q).length();
}

function nearestSkin(p) {
  let best = Infinity;
  for (const t of tris) {
    // Prune on the centroid first: a triangle whose centre is a metre away
    // cannot hold the nearest point when the standoff is millimetres.
    if (t.c.distanceToSquared(p) > 4) continue;
    const d = distToTri(p, t.v);
    if (d < best) best = d;
  }
  return best;
}

const OFFSET = 0.006;
for (const name of artNames) {
  const spec = TAIL_ART[name];
  if (!spec) { console.log(`\n${name}: nothing to place, as intended`); continue; }
  const art = surfaceArt(fin, { ...spec, texture: null, offset: OFFSET, name });
  const u = art.userData;
  console.log(`\n${name}  ${u.widthMetres.toFixed(2)} x ${u.heightMetres.toFixed(2)} m` +
              `  size ${u.requestedSize.toFixed(2)} -> ${u.size.toFixed(3)}` +
              (u.shrunk ? ' (shrunk to fit)' : ''));

  if (!u.fits) fail(`${name} runs ${u.overrun.toFixed(3)} chords off the fin`);
  if (art.children.length !== 2) fail(`${name} has ${art.children.length} faces, wanted 2`);

  // Centred on the fin's height, measured in the WORLD -- the fin stands up
  // through a quarter turn, so its own frame calls the vertical `x` and a
  // check written there would not be checking what anyone looks at.
  fin.updateWorldMatrix(true, false);
  const yAt = (x) => new THREE.Vector3(x, 0, 0).applyMatrix4(fin.matrixWorld).y;
  const finMid = (yAt(0) + yAt(fin.userData.height)) / 2;
  const box = new THREE.Box3().setFromObject(art);
  const artMid = (yAt(box.min.x) + yAt(box.max.x)) / 2;
  console.log(`  centre ${artMid.toFixed(3)} vs fin mid ${finMid.toFixed(3)}` +
              `   fore-aft anchor ${u.chord.toFixed(2)} of chord` +
              (u.requestedChord === 'auto' ? ' (solved)' : ''));
  if (Math.abs(artMid - finMid) > 1e-3) {
    fail(`${name} sits ${(artMid - finMid).toFixed(3)} m off the fin's mid height`);
  }

  for (const mesh of art.children) {
    const pos = mesh.geometry.getAttribute('position');
    const uv = mesh.geometry.getAttribute('uv');
    if (!uv) { fail(`${mesh.name} has no UVs`); continue; }

    // Every eleventh vertex: the standoff is a property of the construction,
    // so a sample that lands on all four edges and the interior is enough, and
    // the full cross-product against 6k triangles is not cheap.
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < pos.count; i += 11) {
      const p = new THREE.Vector3().fromBufferAttribute(pos, i);
      const d = nearestSkin(p);
      lo = Math.min(lo, d); hi = Math.max(hi, d);
    }
    const span = hi - lo;
    console.log(`  ${mesh.name.padEnd(22)} standoff ${lo.toFixed(4)}..${hi.toFixed(4)} m` +
                `  (spread ${(1000 * span).toFixed(2)} mm)`);
    if (Math.abs(lo - OFFSET) > 1e-3 || Math.abs(hi - OFFSET) > 1e-3) {
      fail(`${mesh.name} standoff ${lo.toFixed(4)}..${hi.toFixed(4)}, wanted ${OFFSET}`);
    }
    if (span > 5e-4) fail(`${mesh.name} standoff varies by ${(1000 * span).toFixed(2)} mm -- not conformal`);

    /* ---- does it read the right way round? ---------------------------- */
    // Stated in WORLD terms and in terms of what a viewer sees, because the
    // loft's own frame is exactly what misled me: the fin carries a quarter
    // turn, so its local +y face is the PORT one, and a check phrased in local
    // coordinates agreed with the code and with nothing else.
    //
    // The aeroplane's nose is +z. A viewer off the starboard side sees +z to
    // their left, so on a starboard face the image's left edge (u = 0) must be
    // the FORWARD one -- larger z. Off the port side, the reverse.
    fin.updateWorldMatrix(true, false);
    const world = (i) => new THREE.Vector3()
      .fromBufferAttribute(pos, i).applyMatrix4(fin.matrixWorld);

    let meanX = 0;
    for (let i = 0; i < pos.count; i++) meanX += world(i).x;
    meanX /= pos.count;
    const onStarboard = meanX > 0;
    if (onStarboard !== mesh.name.endsWith('Starboard')) {
      fail(`${mesh.name} is named for the wrong side: it sits at x = ${meanX.toFixed(3)}`);
    }

    // The two ends of the bottom row: u = 0 at one, u = 1 at the other.
    let iLeft = 0, iRight = 0;
    for (let i = 0; i < uv.count; i++) {
      if (uv.getY(i) > 1e-9) continue;                 // bottom row only
      if (uv.getX(i) < uv.getX(iLeft)) iLeft = i;
      if (uv.getX(i) > uv.getX(iRight)) iRight = i;
    }
    const zLeft = world(iLeft).z, zRight = world(iRight).z;
    const readsForward = zLeft > zRight;               // image left is further forward
    console.log(`    ${onStarboard ? 'starboard' : 'port     '}: image left at z ` +
                `${zLeft.toFixed(2)}, right at ${zRight.toFixed(2)} ` +
                `(nose is +z) -- ${readsForward === onStarboard ? 'reads correctly' : 'BACKWARDS'}`);
    if (readsForward !== onStarboard) {
      fail(`${mesh.name} reads backwards: from that side the image runs right to left`);
    }

    // UVs must cover the image exactly once, and the two faces must run
    // opposite ways, or the logo reads backwards from one side.
    const us = [], vs = [];
    for (let i = 0; i < uv.count; i++) { us.push(uv.getX(i)); vs.push(uv.getY(i)); }
    const eq = (a, b) => Math.abs(a - b) < 1e-6;
    if (!eq(Math.min(...us), 0) || !eq(Math.max(...us), 1)
     || !eq(Math.min(...vs), 0) || !eq(Math.max(...vs), 1)) {
      fail(`${mesh.name} UVs do not cover 0..1`);
    }
  }

  // The mirror: first vertex of each face must be at opposite ends of u.
  const [a, b] = art.children;
  if (Math.abs(a.geometry.getAttribute('uv').getX(0)
             + b.geometry.getAttribute('uv').getX(0) - 1) > 1e-6) {
    fail(`${name}: the two faces are not mirrored, so one reads backwards`);
  }
}

/* ---- a negative control ----------------------------------------------- */
// Prove the standoff test can fail. The same artwork asked to sit a quarter of
// a metre out should be caught, or the check above is measuring nothing.
{
  const art = surfaceArt(fin, { ...TAIL_ART['MIT'], texture: null, offset: 0.25 });
  const pos = art.children[0].geometry.getAttribute('position');
  const d = nearestSkin(new THREE.Vector3().fromBufferAttribute(pos, 0));
  console.log(`\ncontrol: artwork floated 0.25 m out measures ${d.toFixed(3)} m off the skin`);
  if (Math.abs(d - 0.25) > 2e-3) fail(`the control did not measure its own offset (${d})`);
}

// And that a flat quad, which is what this is built to avoid, would be caught.
{
  const art = surfaceArt(fin, { ...TAIL_ART['LB gold'], texture: null, offset: OFFSET });
  const pos = art.children[0].geometry.getAttribute('position');
  // Flatten it onto the plane of its own first vertex, as a naive decal would.
  const y0 = pos.getY(0);
  let worst = 0;
  for (let i = 0; i < pos.count; i += 11) {
    const p = new THREE.Vector3(pos.getX(i), y0, pos.getZ(i));
    worst = Math.max(worst, Math.abs(nearestSkin(p) - OFFSET));
  }
  console.log(`control: the same artwork flattened would sit up to ` +
              `${(1000 * worst).toFixed(1)} mm off the skin`);
  if (worst < 5e-4) fail('a flat quad was not distinguishable from the conformal one');
}

console.log(bad ? `\nFAIL: ${bad} problem(s)` : '\nPASS: artwork lies on the fin, both faces, right way round');
process.exit(bad ? 1 : 0);
