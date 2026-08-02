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
import { surfaceArt, fuselageTitles, cabinWindows, windscreen, windowLayout,
         TAIL_ART, TAIL_HEIGHT, artNames } from './components/decals.js';

const sol = JSON.parse(readFileSync(new URL('./decks/b737_conventional_solve.json', import.meta.url), 'utf8'));
const deck = deckFromSolve(sol);
const craft = conventionalAircraft(deck, { sitOnGround: false });
const fin = craft.userData.parts.verticalTail;

let bad = 0;
const fail = (m) => { console.log('  FAIL ' + m); bad++; };

/**
 * Do the triangles face OUT?
 *
 * The failure this catches is total rather than subtle: backface culling
 * removes an inward-facing patch completely, so the artwork does not look
 * wrong, it looks absent -- indistinguishable from never having been built.
 * Every other check here passed on decals that could not be seen at all,
 * because they all measured where the vertices were and none asked which way
 * the triangles between them pointed.
 */
function facesOutward(mesh, outwardAt, label) {
  const pos = mesh.geometry.getAttribute('position');
  const ix = mesh.geometry.getIndex();
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  let inward = 0;
  for (let t = 0; t < ix.count; t += 3) {
    a.fromBufferAttribute(pos, ix.getX(t));
    b.fromBufferAttribute(pos, ix.getX(t + 1));
    c.fromBufferAttribute(pos, ix.getX(t + 2));
    const n = b.clone().sub(a).cross(c.clone().sub(a));
    const centre = a.clone().add(b).add(c).multiplyScalar(1 / 3);
    if (n.dot(outwardAt(centre)) <= 0) inward++;
  }
  if (inward) fail(`${label} has ${inward} of ${ix.count / 3} triangles facing inward -- invisible`);
  return inward === 0;
}

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

  // At the intended fraction of the fin's height, measured in the WORLD -- the
  // fin stands up through a quarter turn, so its own frame calls the vertical
  // `x` and a check written there would not be checking what anyone looks at.
  //
  // Against TAIL_HEIGHT rather than the geometric middle: the artwork sits
  // below mid deliberately, because a fin tapers and the eye centres on the
  // visible panel rather than on the span.
  fin.updateWorldMatrix(true, false);
  const yAt = (x) => new THREE.Vector3(x, 0, 0).applyMatrix4(fin.matrixWorld).y;
  const root = yAt(0), tip = yAt(fin.userData.height);
  const want = root + (tip - root) * TAIL_HEIGHT;
  const box = new THREE.Box3().setFromObject(art);
  const artMid = (yAt(box.min.x) + yAt(box.max.x)) / 2;
  console.log(`  centre ${artMid.toFixed(3)} at ` +
              `${(100 * (artMid - root) / (tip - root)).toFixed(1)}% of height ` +
              `(wanted ${(100 * TAIL_HEIGHT).toFixed(0)}%)` +
              `   fore-aft anchor ${u.chord.toFixed(2)} of chord` +
              (u.requestedChord === 'auto' ? ' (solved)' : ''));
  if (Math.abs(artMid - want) > 1e-3) {
    fail(`${name} sits ${(artMid - want).toFixed(3)} m off ${(100 * TAIL_HEIGHT).toFixed(0)}% of the fin's height`);
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
    facesOutward(mesh, (p) => new THREE.Vector3(0, Math.sign(p.y) || 1, 0), mesh.name);

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

/* ---- the titles along the body ---------------------------------------- */
console.log('\n=== fuselage titles ===');
const fuse = craft.userData.parts.fuselage;
const R = craft.userData.deck.fuseRadius;
// The window row is placed at R sin 20 with a half-height of 0.0825, so this is
// where the glass actually reaches. Read from the same numbers the fuselage
// uses rather than restated as a constant.
const windowTop = R * Math.sin(20 * Math.PI / 180) + 0.0825;

const fuseTris = [];
{
  const g = fuse.userData.skinMesh.geometry;
  const p = g.getAttribute('position').array, ix = g.getIndex().array;
  for (let t = 0; t < ix.length; t += 3) {
    const v = [0, 1, 2].map((k) => new THREE.Vector3(
      p[3 * ix[t + k]], p[3 * ix[t + k] + 1], p[3 * ix[t + k] + 2]));
    fuseTris.push({ v, c: v[0].clone().add(v[1]).add(v[2]).multiplyScalar(1 / 3) });
  }
}
const nearestFuse = (q) => {
  let best = Infinity;
  for (const t of fuseTris) {
    if (t.c.distanceToSquared(q) > 1) continue;
    const d = distToTri(q, t.v);
    if (d < best) best = d;
  }
  return best;
};
console.log(`fuselage skin: ${fuseTris.length} triangles, radius ${R.toFixed(3)},` +
            ` glass reaches y ${windowTop.toFixed(3)}`);

const titles = fuselageTitles(fuse, { deck });
const LIFT = 0.008;
if (!titles.userData.fits) fail('a title ran off the top of the body');

// Front to back, as asked: Mitsubishi, MIT, gold BEACH.
const wanted = ['mhi', 'mit', 'beach-gold'];
const got = titles.children.map((c) => c.name);
if (got.join() !== wanted.join()) fail(`titles are ordered ${got.join(', ')}, wanted ${wanted.join(', ')}`);

let prevAft = Infinity;
for (const piece of titles.children) {
  const d = piece.userData;
  const box = new THREE.Box3().setFromObject(piece);
  console.log(`\n${piece.name}  ${d.widthMetres.toFixed(2)} x ${d.heightMetres.toFixed(2)} m` +
              `  z ${d.zRange[1].toFixed(2)} .. ${d.zRange[0].toFixed(2)}` +
              `  y ${box.min.y.toFixed(2)} .. ${box.max.y.toFixed(2)}`);

  // Above the windows, which is where they were asked to go.
  if (box.min.y <= windowTop) {
    fail(`${piece.name} reaches down to y ${box.min.y.toFixed(3)}, into the window line at ${windowTop.toFixed(3)}`);
  }
  // Nose to tail with no overlap: each starts aft of where the last ended.
  if (d.zRange[1] > prevAft) fail(`${piece.name} starts at ${d.zRange[1].toFixed(2)}, overlapping the one ahead`);
  prevAft = d.zRange[0];

  for (const mesh of piece.children) {
    const pos = mesh.geometry.getAttribute('position');
    const uv = mesh.geometry.getAttribute('uv');
    // Measured against the ANALYTIC surface, which is what the artwork was
    // built from, rather than against the triangles that approximate it.
    // Against the mesh a correct decal reads high over the middle of every
    // facet -- the tessellation is inscribed in the true surface -- and that
    // says something about the skin's resolution, not about the artwork.
    //
    // Exact here because the titles lie on the barrel, where the surface
    // normal is in the plane of the section.
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < pos.count; i += 3) {
      const p = new THREE.Vector3().fromBufferAttribute(pos, i);
      const s = fuse.userData.shapeAt(p.z);
      const th = Math.atan2(p.y - s.yc, p.x);
      const q = fuse.userData.surfaceAt(p.z, th);
      const d2 = Math.hypot(p.x, p.y - s.yc) - Math.hypot(q.x, q.y - s.yc);
      lo = Math.min(lo, d2); hi = Math.max(hi, d2);
    }
    console.log(`  ${mesh.name.padEnd(20)} standoff ${lo.toFixed(5)}..${hi.toFixed(5)} m` +
                `  (spread ${(1000 * (hi - lo)).toFixed(3)} mm)`);
    if (Math.abs(lo - LIFT) > 1e-4 || Math.abs(hi - LIFT) > 1e-4) {
      fail(`${mesh.name} standoff ${lo.toFixed(5)}..${hi.toFixed(5)}, wanted ${LIFT}`);
    }

    // And separately: it must never sink INTO the rendered skin, which is the
    // thing that would actually show. Facets can only put it further out.
    let worstIn = Infinity;
    for (let i = 0; i < pos.count; i += 23) {
      worstIn = Math.min(worstIn, nearestFuse(new THREE.Vector3().fromBufferAttribute(pos, i)));
    }
    if (worstIn < 1e-4) fail(`${mesh.name} touches or enters the skin (${worstIn.toFixed(5)})`);
    facesOutward(mesh, (p) => {
      const s2 = fuse.userData.shapeAt(p.z);
      return new THREE.Vector3(p.x, p.y - s2.yc, 0).normalize();
    }, mesh.name);

    // Reads the right way round, by the same rule as the fin: nose is +z, and
    // a viewer off the starboard side sees +z to their left.
    let meanX = 0;
    for (let i = 0; i < pos.count; i++) meanX += pos.getX(i);
    meanX /= pos.count;
    const onStarboard = meanX > 0;
    if (onStarboard !== mesh.name.endsWith('Starboard')) {
      fail(`${mesh.name} is named for the wrong side (x = ${meanX.toFixed(2)})`);
    }
    let iL = 0, iR = 0;
    for (let i = 0; i < uv.count; i++) {
      if (uv.getY(i) > 1e-9) continue;
      if (uv.getX(i) < uv.getX(iL)) iL = i;
      if (uv.getX(i) > uv.getX(iR)) iR = i;
    }
    if ((pos.getZ(iL) > pos.getZ(iR)) !== onStarboard) {
      fail(`${mesh.name} reads backwards`);
    }
  }
}

/* ---- is the artwork evenly spaced on the SKIN? ------------------------- */
// The property arc-length stepping exists for. Rows placed at equal HEIGHT, or
// at equal ANGLE, would bunch up as the band climbs towards the crown and the
// logo would come out squashed at the top. Measured on the built geometry, so
// it tests the result rather than the intention.
{
  const mesh = titles.children[0].children[0];
  const pos = mesh.geometry.getAttribute('position');
  const uv = mesh.geometry.getAttribute('uv');
  const nu = new Set(Array.from({ length: uv.count }, (_, i) => uv.getY(i).toFixed(6))).size;
  const perRow = uv.count / nu;
  let lo = Infinity, hi = -Infinity;
  const col = Math.floor(perRow / 2);
  for (let r = 0; r < nu - 1; r++) {
    const a = new THREE.Vector3().fromBufferAttribute(pos, r * perRow + col);
    const b = new THREE.Vector3().fromBufferAttribute(pos, (r + 1) * perRow + col);
    const d = a.distanceTo(b);
    lo = Math.min(lo, d); hi = Math.max(hi, d);
  }
  console.log(`\nrow spacing up the side: ${lo.toFixed(4)}..${hi.toFixed(4)} m` +
              `  (${(100 * (hi - lo) / lo).toFixed(2)}% variation)`);
  if ((hi - lo) / lo > 0.02) {
    fail(`artwork is stretched up the side: rows vary by ${(100 * (hi - lo) / lo).toFixed(1)}%`);
  }

  // Control: the same band stepped by equal HEIGHT instead of equal arc, which
  // is the obvious wrong thing and has to be distinguishable from the right one.
  const d0 = titles.children[0].userData;
  const yLo = d0.y - d0.heightMetres / 2, yHi = d0.y + d0.heightMetres / 2;
  let clo = Infinity, chi = -Infinity;
  const zMid = d0.z;
  for (let r = 0; r < nu - 1; r++) {
    const ya = yLo + (yHi - yLo) * (r / (nu - 1));
    const yb = yLo + (yHi - yLo) * ((r + 1) / (nu - 1));
    const pa = fuse.userData.surfaceAt(zMid, Math.asin(ya / R));
    const pb = fuse.userData.surfaceAt(zMid, Math.asin(yb / R));
    const d = pa.distanceTo(pb);
    clo = Math.min(clo, d); chi = Math.max(chi, d);
  }
  console.log(`control: stepped by equal height instead, rows vary by ` +
              `${(100 * (chi - clo) / clo).toFixed(1)}%`);
  if ((chi - clo) / clo < 0.05) fail('the equal-height control was not distinguishable');
}

/* ---- cabin windows ----------------------------------------------------- */
// The point of these is that they are not a styling choice: the solve says how
// many rows and where the shell is, so what is checked is that the geometry
// AGREES with the solve rather than that it looks plausible.
console.log('\n=== cabin windows ===');
const wins = cabinWindows(fuse, { deck });
const wu = wins.userData;
console.log(`${wu.rows} per side, pitch ${wu.pitch.toFixed(4)} m ` +
            `(${(wu.pitch / 0.0254).toFixed(1)} in), ${wu.passengers} pax ` +
            `${wu.seatsAbreast.toFixed(0)} abreast`);

if (Math.round(deck.cabinRows) !== wu.rows) {
  fail(`solve says ${deck.cabinRows} rows, built ${wu.rows}`);
}
// Derived two independent ways: the pitch times the rows must be the shell.
const shell = Math.abs(deck.cabinEnd - deck.cabinStart);
if (Math.abs(wu.pitch * wu.rows - shell) > 1e-6) {
  fail(`${wu.rows} rows at ${wu.pitch} does not fill the ${shell} shell`);
}

const wbox = new THREE.Box3().setFromObject(wins);
console.log(`window band y ${wbox.min.y.toFixed(3)}..${wbox.max.y.toFixed(3)}` +
            `  z ${wbox.min.z.toFixed(2)}..${wbox.max.z.toFixed(2)}` +
            `  (shell ${(-deck.cabinStart).toFixed(2)}..${(-deck.cabinEnd).toFixed(2)})`);
if (wbox.max.z > -deck.cabinStart + 1e-6 || wbox.min.z < -deck.cabinEnd - 1e-6) {
  fail('windows run outside the pressure shell -- into the nose or the tailcone');
}

// Evenly spaced, and none doubled up. Measured from the built centres.
const zs = wins.children.map((c) => c.userData.z).sort((a, b) => b - a);
let glo = Infinity, ghi = -Infinity;
for (let i = 1; i < zs.length; i++) {
  const g2 = zs[i - 1] - zs[i];
  glo = Math.min(glo, g2); ghi = Math.max(ghi, g2);
}
console.log(`gaps ${glo.toFixed(5)}..${ghi.toFixed(5)} m`);
if (ghi - glo > 1e-9) fail(`window pitch is uneven: ${glo} to ${ghi}`);
if (Math.abs(glo - wu.pitch) > 1e-9) fail(`gap ${glo} is not the stated pitch ${wu.pitch}`);

// On the skin, facing out, both sides -- the same properties as everything else.
{
  const piece = wins.children[Math.floor(wins.children.length / 2)];
  for (const mesh of piece.children) {
    const pos = mesh.geometry.getAttribute('position');
    let lo2 = Infinity, hi2 = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      const p = new THREE.Vector3().fromBufferAttribute(pos, i);
      const s2 = fuse.userData.shapeAt(p.z);
      const th = Math.atan2(p.y - s2.yc, p.x);
      const q = fuse.userData.surfaceAt(p.z, th);
      const d2 = Math.hypot(p.x, p.y - s2.yc) - Math.hypot(q.x, q.y - s2.yc);
      lo2 = Math.min(lo2, d2); hi2 = Math.max(hi2, d2);
    }
    console.log(`  ${mesh.name.padEnd(22)} standoff ${lo2.toFixed(5)}..${hi2.toFixed(5)} m`);
    if (Math.abs(lo2 - 0.004) > 1e-4 || Math.abs(hi2 - 0.004) > 1e-4) {
      fail(`${mesh.name} standoff ${lo2.toFixed(5)}..${hi2.toFixed(5)}, wanted 0.004`);
    }
    facesOutward(mesh, (p) => {
      const s2 = fuse.userData.shapeAt(p.z);
      return new THREE.Vector3(p.x, p.y - s2.yc, 0).normalize();
    }, mesh.name);
  }
}

// Titles must clear the windows: they are two rows of artwork on one patch of
// skin, and overlapping decals fight for the depth buffer rather than stacking.
{
  const tb = new THREE.Box3().setFromObject(titles);
  const clear = tb.min.y - wbox.max.y;
  console.log(`titles sit ${clear.toFixed(3)} m above the window band`);
  if (clear <= 0) fail(`titles overlap the windows by ${(-clear).toFixed(3)} m`);
}

/* ---- the windscreen ---------------------------------------------------- */
// A band cut by two horizontal planes and wrapped onto the nose. What is
// checked is that the planes are where they were asked for and that the glass
// stops where the BODY stops offering surface between them -- the forward taper
// and the wrap over the crown are consequences of the shape, not of numbers
// typed in, so they are the things worth measuring.
console.log('\n=== windscreen ===');
{
  const scr = windscreen(fuse, {});
  const su = scr.userData;
  const fud = fuse.userData;
  console.log(`${(100 * su.low).toFixed(0)}% to ${(100 * su.high).toFixed(0)}% of a ` +
              `${su.bodyHeight.toFixed(3)} m body: y ${su.yLow.toFixed(3)} to ${su.yHigh.toFixed(3)}`);
  console.log(`upper edge ${su.topLength.toFixed(3)} m long from the centreline, ` +
              `${su.panesPerSide} panes a side, ${(1000 * su.post).toFixed(0)} mm posts`);

  const R = deck.fuseRadius;
  if (Math.abs(su.yLow - (-R + su.low * 2 * R)) > 1e-6
   || Math.abs(su.yHigh - (-R + su.high * 2 * R)) > 1e-6) {
    fail(`cut lines are not at ${su.low}/${su.high} of the body height`);
  }
  if (scr.children.length !== su.paneCount) {
    fail(`${scr.children.length} panes, wanted ${su.paneCount}`);
  }

  /* ---- a post crosses the upper edge SQUARE --------------------------- */
  /**
   * The property the posts are defined by, and the one every earlier attempt
   * failed. Seen down the surface normal where it starts, a post must meet the
   * upper edge at a right angle -- which is what makes it a post rather than
   * the trace of some plane through the body. Measured in the tangent plane at
   * the top of each post.
   */
  {
    let worst = 0;
    const arcs = [];
    for (const [s0, s1] of su.edges) { arcs.push(s0); arcs.push(s1); }
    for (const sArc of arcs) {
      const path = su.marchPost(sArc);
      if (path.length < 2) { fail(`the post at arc ${sArc} did not march`); continue; }
      const p0 = fud.surfaceAt(path[0].z, path[0].th);
      const dir = fud.surfaceAt(path[1].z, path[1].th).sub(p0).normalize();
      const e = 1e-3;
      const a0 = su.atArc(Math.max(0, sArc - e));
      const a1 = su.atArc(Math.min(su.topLength, sArc + e));
      const tan = fud.surfaceAt(a1.z, a1.th).sub(fud.surfaceAt(a0.z, a0.th)).normalize();
      const ang = Math.acos(Math.min(1, Math.abs(dir.dot(tan)))) * 180 / Math.PI;
      worst = Math.max(worst, Math.abs(90 - ang));
    }
    console.log(`  posts meet the upper edge within ${worst.toFixed(2)} deg of square`);
    if (worst > 0.5) fail(`a post crosses the upper edge at ${(90 - worst).toFixed(1)} deg, not square`);
  }

  /* ---- and very nearly square to the lower edge ----------------------- */
  {
    let lo = 90, hi = 0;
    for (const [, s1] of su.edges) {
      const path = su.marchPost(s1);
      const n = path.length;
      if (n < 2) continue;
      const p0 = fud.surfaceAt(path[n - 2].z, path[n - 2].th);
      const dir = fud.surfaceAt(path[n - 1].z, path[n - 1].th).sub(p0).normalize();
      const zL = path[n - 1].z, dz = 1e-3;
      const t0 = su.thAtHeight(zL - dz, su.yLow), t1 = su.thAtHeight(zL + dz, su.yLow);
      if (t0 == null || t1 == null) continue;
      const tan = fud.surfaceAt(zL + dz, t1).sub(fud.surfaceAt(zL - dz, t0)).normalize();
      const ang = Math.acos(Math.min(1, Math.abs(dir.dot(tan)))) * 180 / Math.PI;
      lo = Math.min(lo, ang); hi = Math.max(hi, ang);
    }
    console.log(`  and the lower edge at ${lo.toFixed(1)}..${hi.toFixed(1)} deg`);
    // Not asserted at 90: the two edges are only nearly parallel, so this
    // follows rather than being imposed. Well off square would mean the band
    // is not the shape it is supposed to be.
    if (lo < 60) fail(`a post meets the lower edge at ${lo.toFixed(1)} deg -- far from square`);
  }

  /* ---- the posts, measured along the upper edge ------------------------ */
  {
    // The centre post is body left between the two sides.
    let minX = Infinity;
    for (const m of scr.children) {
      if (m.userData.side < 0) continue;
      minX = Math.min(minX, new THREE.Box3().setFromObject(m).min.x);
    }
    console.log(`  centre post ${(2000 * minX).toFixed(1)} mm`);
    if (minX <= 0) fail('the two sides meet -- there is no centre post');
    if (Math.abs(2 * minX - su.post) > 4 * su.lift) {
      fail(`centre post is ${(2000 * minX).toFixed(1)} mm, wanted ${(1000 * su.post).toFixed(0)}`);
    }
    for (let k = 0; k < su.edges.length - 1; k++) {
      const gap = su.edges[k + 1][0] - su.edges[k][1];
      console.log(`  post between panes ${k + 1} and ${k + 2}: ${(1000 * gap).toFixed(1)} mm`);
      if (Math.abs(gap - su.post) > 1e-6) {
        fail(`post reads ${(1000 * gap).toFixed(1)} mm, wanted ${(1000 * su.post).toFixed(0)}`);
      }
    }
  }

  /* ---- a post is the same thickness top to bottom ---------------------- */
  // Measured down each divider's own walk. Two walks launched half a post apart
  // diverge across the nose and the gap opens out towards the bottom, which is
  // what it used to do; one walk offset either way cannot.
  {
    let lo = Infinity, hi = 0;
    for (let d = 0; d < su.dividers.length; d++) {
      const A = su.offsetPath(su.dividerPaths[d], su.post / 2);
      const B = su.offsetPath(su.dividerPaths[d], -su.post / 2);
      for (let i = 0; i < Math.min(A.length, B.length); i++) {
        const w = fud.surfaceAt(A[i].z, A[i].th).distanceTo(fud.surfaceAt(B[i].z, B[i].th));
        lo = Math.min(lo, w); hi = Math.max(hi, w);
      }
    }
    console.log(`  post thickness down its whole length: ` +
                `${(1000 * lo).toFixed(1)}..${(1000 * hi).toFixed(1)} mm`);
    if (hi - lo > 1e-3) {
      fail(`a post varies by ${(1000 * (hi - lo)).toFixed(1)} mm along its length`);
    }
    if (Math.abs(lo - su.post) > 1e-3) {
      fail(`posts measure ${(1000 * lo).toFixed(1)} mm, wanted ${(1000 * su.post).toFixed(0)}`);
    }
  }

  /* ---- widest against the centre post --------------------------------- */
  {
    const w = su.panes.map((p) => p.width);
    console.log(`  pane widths from the centre out: ${w.map((v) => v.toFixed(3)).join(', ')} m ` +
                `(along the upper edge)`);
    // Non-increasing, not strictly decreasing: 40/30/30 puts two equal panes
    // outboard on purpose, and only the innermost has to be the widest.
    for (let i = 1; i < w.length; i++) {
      if (w[i] > w[i - 1] + 1e-9) {
        fail(`pane ${i + 1} is wider than pane ${i}`);
      }
    }
  }

  /* ---- the carved lower edge ------------------------------------------ */
  // The upper edge is the cut line and never moves; only the bottom is carved.
  // What is checked is that each pane's corners lift by what was asked, and
  // that the lifts AGREE across the dividers -- the lower edge is meant to be
  // one unbroken line, and a mismatched pair either side of a post would read
  // as a step.
  {
    const bandH = su.yHigh - su.yLow;
    const panes = scr.children.filter((m) => m.userData.side > 0)
      .sort((a, b) => a.userData.pane - b.userData.pane);
    let worst = 0;
    panes.forEach((m, k) => {
      const pos = m.geometry.getAttribute('position');
      const nvv = m.userData.nv, rows = pos.count / nvv;
      const gotIn = (pos.getY((rows - 1) * nvv) - su.yLow) / bandH;
      const gotOut = (pos.getY((rows - 1) * nvv + nvv - 1) - su.yLow) / bandH;
      const [wIn, wOut] = su.lowerRaise[k];
      console.log(`  pane ${k + 1} lower edge lifted ` +
                  `${(100 * gotIn).toFixed(1)}% / ${(100 * gotOut).toFixed(1)}% ` +
                  `(asked ${(100 * wIn).toFixed(0)} / ${(100 * wOut).toFixed(0)})`);
      // A per-cent of tolerance: the lift is applied by walking a fraction less
      // far down a marched column, and the standoff moves the last vertex too.
      if (Math.abs(gotIn - wIn) > 0.01 || Math.abs(gotOut - wOut) > 0.01) {
        fail(`pane ${k + 1} lower edge lifted ${(100 * gotIn).toFixed(1)}/` +
             `${(100 * gotOut).toFixed(1)}%, wanted ${(100 * wIn).toFixed(0)}/${(100 * wOut).toFixed(0)}`);
      }
      if (k) {
        const prev = panes[k - 1];
        const pp = prev.geometry.getAttribute('position');
        const pr = pp.count / prev.userData.nv;
        const yPrev = pp.getY((pr - 1) * prev.userData.nv + prev.userData.nv - 1);
        worst = Math.max(worst, Math.abs(yPrev - pos.getY((rows - 1) * nvv)));
      }
    });
    console.log(`  lower edge is continuous across the posts to ${(1000 * worst).toFixed(1)} mm`);
    if (worst > 2e-3) fail(`the lower edge steps ${(1000 * worst).toFixed(1)} mm at a post`);
  }

  /* ---- inside its own cut lines --------------------------------------- */
  const sbox = new THREE.Box3().setFromObject(scr);
  const tol = su.lift + 1e-4;
  console.log(`  spans y ${sbox.min.y.toFixed(4)}..${sbox.max.y.toFixed(4)}, ` +
              `cut lines ${su.yLow.toFixed(4)}..${su.yHigh.toFixed(4)} ` +
              `(+${(1000 * su.lift).toFixed(0)} mm standoff allowed)`);
  if (sbox.min.y < su.yLow - tol || sbox.max.y > su.yHigh + tol) {
    fail(`glass reaches y ${sbox.min.y.toFixed(4)}..${sbox.max.y.toFixed(4)}, ` +
         `outside its cut lines by more than the ${su.lift} standoff`);
  }
  if (sbox.min.z < -fud.noseLength || sbox.max.z > 0) {
    fail(`glass is not on the nose: z ${sbox.min.z.toFixed(3)}..${sbox.max.z.toFixed(3)}`);
  }

  /* ---- on the skin, facing out, no slivers ---------------------------- */
  for (const mesh of scr.children) {
    const pos = mesh.geometry.getAttribute('position');
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < pos.count; i += 5) {
      const p = new THREE.Vector3().fromBufferAttribute(pos, i);
      let bz = p.z; const s0 = fud.shapeAt(p.z);
      let bt = Math.atan2(p.y - s0.yc, p.x), best = Infinity;
      for (let step = 0.15; step > 1e-5; step *= 0.45) {
        let mz = bz, mt = bt;
        for (const dz of [-step, 0, step]) for (const dt of [-step, 0, step]) {
          const q = fud.surfaceAt(bz + dz, bt + dt);
          const d = Math.hypot(q.x - p.x, q.y - p.y, q.z - p.z);
          if (d < best) { best = d; mz = bz + dz; mt = bt + dt; }
        }
        bz = mz; bt = mt;
      }
      lo = Math.min(lo, best); hi = Math.max(hi, best);
    }
    if (lo < 1e-4) fail(`${mesh.name} touches or enters the nose`);
    if (Math.abs(hi - su.lift) > 6e-4) fail(`${mesh.name} standoff reaches ${hi.toFixed(5)}`);
    facesOutward(mesh, (p) => {
      const s2 = fud.shapeAt(p.z);
      return new THREE.Vector3(p.x, p.y - s2.yc, 0).normalize();
    }, mesh.name);

    let collapsed = 0;
    for (let i = 0; i < pos.count; i++) {
      if (Math.abs(pos.getX(i)) < 1e-9 && Math.abs(pos.getY(i)) < 1e-9
       && Math.abs(pos.getZ(i)) < 1e-9) collapsed++;
    }
    if (collapsed) fail(`${mesh.name} has ${collapsed} vertices collapsed to the origin`);
  }
  console.log(`  all ${scr.children.length} panes lie on the skin and face out`);

  if (sbox.min.z < wbox.max.z) {
    fail(`glass reaches ${sbox.min.z.toFixed(2)}, aft of the first window at ${wbox.max.z.toFixed(2)}`);
  }
}

/* ---- the titles start at the window line ------------------------------ */
{
  const wl = windowLayout(deck);
  const d = Math.abs(titles.userData.startZ - wl.zFwd);
  console.log(`\ntitles start at ${titles.userData.startZ.toFixed(3)}, ` +
              `window line front is ${wl.zFwd.toFixed(3)}`);
  if (d > 1e-9) fail(`titles start ${d.toFixed(3)} m off the front of the window line`);
}

console.log(bad ? `\nFAIL: ${bad} problem(s)` : '\nPASS: artwork, windows and glass lie on the skin, placed as the solve says');
process.exit(bad ? 1 : 0);
