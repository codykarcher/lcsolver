/**
 * Paint schemes are cosmetic and must stay that way.
 *
 *     node check_paint.mjs
 *
 * Two things to hold. A scheme must repaint everything that was white and
 * NOTHING else -- tyres, blades, the bare inlet lip and the black inside a
 * nozzle are all distinguished by carrying their own materials, and a livery
 * that reached them would undo that distinction. And it must not touch the
 * geometry, because the moment a scheme can move a vertex it stops being a
 * decision about appearance.
 */
import * as THREE from 'three';
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { conventionalAircraft, deckFromSolve } from './components/aircraft.js';
import { paintRoles, paint, schemeNames, SCHEMES, ROLES } from './components/paint.js';
import { skin } from './components/materials.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const sol = JSON.parse(readFileSync(join(HERE, 'decks/b737_conventional_solve.json'), 'utf8'));
const build = () => conventionalAircraft(deckFromSolve(sol));

let failures = 0;
const bad = (m) => { console.log(`  FAIL  ${m}`); failures++; };

/* what counts as paintable, and what does not ------------------------- */
const ref = build();
const refRoles = paintRoles(ref);
const paintable = new Set(Object.values(refRoles).flat());
const others = [];
ref.traverse((o) => { if (o.isMesh && !paintable.has(o)) others.push(o); });
if (!paintable.size) bad('no paintable meshes found at all');
for (const role of ROLES) {
  if (!refRoles[role].length) bad(`role ${role} matched no meshes`);
}
console.log(`paintable: ${paintable.size} meshes across ${ROLES.length} roles ` +
            `(${ROLES.map((r) => `${r} ${refRoles[r].length}`).join(', ')})`);
console.log(`untouched: ${others.length} meshes`);

/* every scheme ---------------------------------------------------------- */
// Positions are snapshotted before and after. A scheme is a material swap, so
// a single vertex moving is a scheme doing something it has no business doing.
const snapshot = (g) => {
  const out = [];
  g.traverse((o) => {
    if (o.isMesh) out.push(o.geometry.getAttribute('position').array.slice(0, 24));
  });
  return out;
};

console.log('\nscheme            body      fin       nacelle   metalness  repainted');
for (const name of schemeNames) {
  const g = build();
  const roles = paintRoles(g);
  const before = snapshot(g);
  const untouchedBefore = [];
  g.traverse((o) => { if (o.isMesh && o.material !== skin) untouchedBefore.push([o, o.material]); });

  paint(roles, name);

  const after = snapshot(g);
  for (let i = 0; i < before.length; i++) {
    for (let j = 0; j < before[i].length; j++) {
      if (before[i][j] !== after[i][j]) { bad(`${name} moved geometry`); i = before.length; break; }
    }
  }
  for (const [o, m] of untouchedBefore) {
    if (o.material !== m) { bad(`${name} repainted a non-white part (${m.name || 'unnamed'})`); break; }
  }
  let repainted = 0;
  for (const mesh of Object.values(roles).flat()) {
    if (mesh.material === skin) bad(`${name} left a paintable mesh on the base skin`);
    else repainted++;
  }
  const hex = (r) => '#' + roles[r][0].material.color.getHexString();
  console.log(`  ${name.padEnd(16)}${hex('body')}   ${hex('fin')}   ${hex('nacelle')}   ` +
              `${roles.body[0].material.metalness.toFixed(2).padStart(6)}     ${repainted}`);
  if (!SCHEMES[name].note) bad(`${name} has no note for the page to show`);
}

console.log(failures ? `\n${failures} FAILURE(S)`
                     : '\nPASS: every scheme paints the white parts and only those');
process.exit(failures ? 1 : 0);
