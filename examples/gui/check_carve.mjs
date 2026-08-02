/**
 * Does the duct carve do what it claims, and only that?
 *
 * Three questions, and the third matters most. The carve exists to expose the
 * engines, and it has to leave a closed shell -- but the binding condition is
 * that NOTHING ELSE MOVES. That body was arrived at by hand over many attempts
 * and every earlier way of making a duct changed it, so "the rest is untouched"
 * is not a nicety here, it is the requirement, and it is checked literally:
 * every triangle of the original that lies clear of the duct must still be in
 * the carved mesh, the same three vertices, bit for bit.
 *
 * Each check is run against deliberately broken geometry as well. Twice this
 * session a check passed on a body that was visibly wrong, so a check that has
 * not been shown to fail is not evidence.
 */
import { readFileSync } from 'fs';
import * as THREE from 'three';
import { deckFromSolve } from './components/aircraft.js';
import { d8Aircraft } from './components/d8_aircraft.js';
import { ductVolume, clipTriangles } from './components/carve.js';

const sol = JSON.parse(readFileSync(new URL('./decks/b737_d8_solve.json', import.meta.url), 'utf8'));
const deck = deckFromSolve(sol);

let failures = 0;
const bad = (m) => { console.log(`  FAIL  ${m}`); failures++; };

const plain = d8Aircraft(deck, { sitOnGround: false, ductCarve: false });
const cut = d8Aircraft(deck, { sitOnGround: false, ductCarve: true });
const fu = cut.userData.parts.fuselage.userData;
const duct = ductVolume(fu.duct);

/** Every triangle of a mesh, as sorted vertex triples rounded to the micron. */
function triangles(geo) {
  const pos = geo.getAttribute('position');
  const index = geo.getIndex();
  const n = index ? index.count : pos.count;
  const at = index ? (i) => index.getX(i) : (i) => i;
  const v = new THREE.Vector3();
  const out = [];
  for (let t = 0; t + 2 < n; t += 3) {
    const c = [];
    for (let k = 0; k < 3; k++) {
      v.fromBufferAttribute(pos, at(t + k));
      c.push(`${v.x.toFixed(6)},${v.y.toFixed(6)},${v.z.toFixed(6)}`);
    }
    out.push({ key: c.slice().sort().join('|'), pts: c });
  }
  return out;
}

/* ---- 1. everything clear of the duct is untouched ----------------------- */
// The test is on triangles wholly clear of the duct by a margin, since one
// sitting exactly on the boundary is legitimately allowed to be split.
const before = triangles(plain.userData.parts.fuselage.userData.skinMesh.geometry);
const after = new Set(triangles(cut.userData.parts.fuselage.userData.skinMesh.geometry)
  .map((t) => t.key));
const p = new THREE.Vector3();
const clearOf = (t) => t.pts.every((s) => {
  const [x, y, z] = s.split(',').map(Number);
  return duct.depth(p.set(x, y, z)) < -0.02;
});
const clear = before.filter(clearOf);
const lost = clear.filter((t) => !after.has(t.key));
console.log(`skin ${before.length} triangles, ${clear.length} clear of the duct`);
if (lost.length) bad(`${lost.length} triangles clear of the duct were changed`);

/* ---- 2. the engines are no longer buried -------------------------------- */
/**
 * Sampled on the nacelles' own outer surface, in aircraft coordinates, and
 * asked whether that point is still in solid material -- inside the body's
 * section AND outside the duct. Sampling the axis instead would pass on a duct
 * far too narrow to clear the fan.
 */
const d = cut.userData.deck;
const rN = d.nacelleDia / 2, axis = cut.userData.engineAxisY;
// The engine's extent is MEASURED off the built pod. The turbofan is not
// centred on its own origin, so `engineX +/- length/2` misses it by over a
// metre, and the sampling would then miss the part that is actually buried.
const pod = cut.userData.parts.engines[0];
pod.updateMatrixWorld(true);
const box = new THREE.Box3().setFromObject(pod);
const [xNose, xTail] = [-box.max.z, -box.min.z];
console.log(`nacelle runs x ${xNose.toFixed(2)} to ${xTail.toFixed(2)}, duct level from ` +
            `${fu.duct.deepFrom.toFixed(2)} aft`);
/**
 * How far the nacelle reaches into solid material, at its worst.
 *
 * DEPTH, not a count of points. Counting says 1.0% of the nacelle is buried
 * where the truth is that the duct's wall stands exactly on the engine's
 * equator and touches it along a line -- 0.00 mm deep. That is the clearance
 * the configuration has, not a fault, and a check that cannot tell tangency
 * from burial would either fail on a correct duct or have to be loosened until
 * it stopped catching real ones.
 */
function deepestBurial(volume) {
  let worst = -Infinity;
  for (let i = 0; i <= 80; i++) {
    const x = xNose + (xTail - xNose) * (i / 80);
    for (let j = 0; j < 96; j++) {
      const th = (j / 96) * Math.PI * 2;
      const px = d.engineY + rN * Math.cos(th), py = axis + rN * Math.sin(th);
      p.set(px, py, -x);
      if (fu.depthInside(px, py, -x) <= 0) continue;      // not in the body
      // Inside the body, how far from getting out through the duct.
      worst = Math.max(worst, volume ? -volume.depth(p) : fu.depthInside(px, py, -x));
    }
  }
  return worst === -Infinity ? 0 : worst;
}
const wasBuried = deepestBurial(null), nowBuried = deepestBurial(duct);
console.log(`nacelle reaches ${(1000 * wasBuried).toFixed(0)} mm into solid material before, ` +
            `${(1000 * nowBuried).toFixed(2)} mm after`);
if (nowBuried > 1e-3) bad(`the nacelle is still ${(1000 * nowBuried).toFixed(0)} mm inside the body`);
if (wasBuried < 0.1) bad('nothing was buried to begin with -- the test proves nothing');

/* ---- 3. the shell is closed along the cut ------------------------------- */
/**
 * The skin's new boundary and the floor's boundary have to be the same curve.
 * Counted as open edges: an edge used once is a boundary, and every boundary
 * edge of one sheet should land on the other. Compared by proximity rather than
 * identity, because the two sheets cut on DIFFERENT surfaces -- the skin on the
 * duct, the floor on the body -- and meet on the curve where those cross.
 */
function boundary(geo) {
  const pos = geo.getAttribute('position');
  const index = geo.getIndex();
  const n = index ? index.count : pos.count;
  const at = index ? (i) => index.getX(i) : (i) => i;
  const v = new THREE.Vector3();
  const key = (i) => {
    v.fromBufferAttribute(pos, at(i));
    return `${v.x.toFixed(5)},${v.y.toFixed(5)},${v.z.toFixed(5)}`;
  };
  const uses = new Map();
  const seg = new Map();
  for (let t = 0; t + 2 < n; t += 3) {
    const k = [key(t), key(t + 1), key(t + 2)];
    for (let e = 0; e < 3; e++) {
      const a = k[e], b = k[(e + 1) % 3];
      const id = a < b ? `${a}/${b}` : `${b}/${a}`;
      uses.set(id, (uses.get(id) || 0) + 1);
      if (!seg.has(id)) {
        const pa = a.split(',').map(Number), pb = b.split(',').map(Number);
        seg.set(id, {
          a: new THREE.Vector3(...pa), b: new THREE.Vector3(...pb),
          mid: new THREE.Vector3(...pa).lerp(new THREE.Vector3(...pb), 0.5),
        });
      }
    }
  }
  return [...uses].filter(([, c]) => c === 1).map(([id]) => seg.get(id));
}

/**
 * Distance from a point to a segment.
 *
 * Point-to-POINT between edge midpoints was the first measure and it reads
 * half an edge length even when two boundaries lie exactly on each other --
 * 66 mm on this mesh, which is indistinguishable from a real 66 mm crack. The
 * two sheets are meshed at different resolutions and their vertices have no
 * reason to coincide, so only distance to the other CURVE means anything.
 */
const toSegment = (q, s) => {
  const ab = s.b.clone().sub(s.a);
  const l2 = ab.lengthSq();
  const t = l2 < 1e-18 ? 0 : Math.min(1, Math.max(0, q.clone().sub(s.a).dot(ab) / l2));
  return q.distanceTo(s.a.clone().addScaledVector(ab, t));
};
/**
 * Drop edges that are only topologically open.
 *
 * Cutting one surface at a time means a triangle can be split while its
 * neighbour, already outside a different surface, is kept whole. The shared
 * edge is then one segment on one side and two collinear segments on the
 * other, so edge-counting calls all three a boundary -- and there is no hole,
 * the two sides lie exactly on each other. Counting those as gaps put 26 false
 * open edges in a mesh with none, which is the kind of number that sends you
 * looking for a fault that is not there.
 *
 * Detected by the thing that defines them: the midpoint of a T-junction edge
 * lies exactly on another boundary edge.
 */
function realHoles(edges) {
  return edges.filter((s, i) =>
    !edges.some((o, j) => j !== i && toSegment(s.mid, o) < 1e-4));
}
const skinEdge = realHoles(boundary(cut.userData.parts.fuselage.userData.skinMesh.geometry))
  .filter((s) => duct.depth(s.mid) > -0.05);    // only the edge the carve made
const floorEdge = boundary(fu.duct.mesh.geometry);
console.log(`cut edge: ${skinEdge.length} open edges on the skin, ${floorEdge.length} on the floor`);
const nearest = (q, set) => set.reduce((m, s) => Math.min(m, toSegment(q, s)), Infinity);
const TOL = 0.01;                                // 10 mm on a 30 m aeroplane
const gaps = skinEdge.map((s) => nearest(s.mid, floorEdge));
const strays = gaps.filter((g) => g > TOL);
console.log(`worst gap ${gaps.length ? Math.max(...gaps).toFixed(4) : 'n/a'} m, ` +
            `${strays.length} open edges further than ${TOL * 1000} mm from the duct`);
if (!skinEdge.length) bad('the carve left no open edge at all -- it cut nothing');
if (strays.length > skinEdge.length * 0.02) bad(`${strays.length} of ${skinEdge.length} cut edges are unclosed`);

/* ---- 4. the duct's surface faces into the duct --------------------------- */
/**
 * Every triangle of the floor and walls must face the void, which is up off the
 * floor and inboard off the walls.
 *
 * Worth its own check because the failure is silent: backface culling makes a
 * wrongly wound sheet INVISIBLE, not inside-out, so the duct would simply look
 * as though the carve had left a hole -- which sends you looking at the
 * clipping, where nothing is wrong.
 */
{
  const pos = fu.duct.mesh.geometry.getAttribute('position');
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const n = new THREE.Vector3(), cen = new THREE.Vector3();
  let wrong = 0, total = 0;
  for (let i = 0; i + 2 < pos.count; i += 3) {
    a.fromBufferAttribute(pos, i);
    b.fromBufferAttribute(pos, i + 1);
    c.fromBufferAttribute(pos, i + 2);
    cen.copy(a).add(b).add(c).multiplyScalar(1 / 3);
    n.crossVectors(b.clone().sub(a), c.clone().sub(a));
    if (n.lengthSq() < 1e-18) continue;
    // A step into the void from the face's own centre: up, and inboard.
    const into = new THREE.Vector3(-Math.sign(cen.x) * 0.01, 0.01, 0);
    total++;
    if (n.dot(into) < 0) wrong++;
  }
  console.log(`duct surface: ${total} triangles, ${wrong} facing away from the void`);
  if (wrong > total * 0.02) bad(`${wrong} of ${total} duct faces are wound inside out`);
}

/* ---- negative controls -------------------------------------------------- */
/**
 * Each check, run against geometry it is supposed to reject. A check that has
 * never been seen to fail is not evidence that anything is right.
 */
console.log('\nnegative controls');
{
  // A carve that moves the untouched region: nudge one far-forward vertex.
  const g = plain.userData.parts.fuselage.userData.skinMesh.geometry.clone();
  g.getAttribute('position').setY(0, g.getAttribute('position').getY(0) + 0.5);
  const moved = new Set(triangles(g).map((t) => t.key));
  const n = clear.filter((t) => !moved.has(t.key)).length;
  console.log(`  a moved vertex loses ${n} clear triangles` + (n ? '  ok' : '  NOT DETECTED'));
  if (!n) bad('control: a moved vertex was not detected as a change');
}
{
  // A duct too shallow to reach the engines leaves them buried.
  const shallow = ductVolume({ ...fu.duct, floor: axis + rN * 0.9 });
  const f = deepestBurial(shallow);
  console.log(`  a duct stopping short leaves ${(1000 * f).toFixed(0)} mm buried` +
              (f > 1e-3 ? '  ok' : '  NOT DETECTED'));
  if (!(f > 1e-3)) bad('control: a too-shallow duct was not detected');
}
{
  // A floor built for a duct 30 cm narrower cannot close the skin's cut.
  const wrong = ductVolume({ ...fu.duct, halfWidth: fu.duct.halfWidth - 0.3 });
  const g = clipTriangles(
    fu.duct.mesh.geometry, (q) => wrong.halfWidth - Math.abs(q.x));
  const e = boundary(g);
  const n = skinEdge.filter((s) => nearest(s.mid, e) > TOL).length;
  console.log(`  a floor 300 mm narrow leaves ${n} of ${skinEdge.length} edges open` +
              (n > skinEdge.length * 0.02 ? '  ok' : '  NOT DETECTED'));
  if (!(n > skinEdge.length * 0.02)) bad('control: a narrow floor was not detected');
}

{
  // Reversing the winding must be caught, or the check is decoration.
  const g = fu.duct.mesh.geometry.clone();
  const arr = g.getAttribute('position').array;
  for (let i = 0; i + 8 < arr.length; i += 9) {
    for (let k = 0; k < 3; k++) {
      const t = arr[i + 3 + k]; arr[i + 3 + k] = arr[i + 6 + k]; arr[i + 6 + k] = t;
    }
  }
  const pos = g.getAttribute('position');
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  const n = new THREE.Vector3(), cen = new THREE.Vector3();
  let wrong = 0, total = 0;
  for (let i = 0; i + 2 < pos.count; i += 3) {
    a.fromBufferAttribute(pos, i); b.fromBufferAttribute(pos, i + 1); c.fromBufferAttribute(pos, i + 2);
    cen.copy(a).add(b).add(c).multiplyScalar(1 / 3);
    n.crossVectors(b.clone().sub(a), c.clone().sub(a));
    if (n.lengthSq() < 1e-18) continue;
    total++;
    if (n.dot(new THREE.Vector3(-Math.sign(cen.x) * 0.01, 0.01, 0)) < 0) wrong++;
  }
  console.log(`  reversing the winding turns ${wrong} of ${total} faces the wrong way` +
              (wrong > total * 0.02 ? '  ok' : '  NOT DETECTED'));
  if (!(wrong > total * 0.02)) bad('control: a reversed winding was not detected');
}

console.log(failures ? `\n${failures} FAILED` : '\nall checks passed');
process.exit(failures ? 1 : 0);
