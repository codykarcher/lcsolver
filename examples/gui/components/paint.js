/**
 * Paint schemes.
 *
 * Cosmetic only: nothing here moves a vertex. A scheme reassigns materials on
 * the parts that are currently PAINTED -- that is, the ones sharing the base
 * `skin` material -- and leaves everything else exactly as it was. So tyres
 * stay rubber, the inlet lip stays bare metal, and the inside of a nozzle stays
 * a hole, without any of them having to be listed here or know this file
 * exists. "The white parts" is a set the model already defines; this only has
 * to look it up.
 *
 * Parts are grouped into ROLES rather than named individually, because a livery
 * is a statement about roles -- the fin is one colour, the engines another --
 * and a scheme that had to name meshes would break the first time a component
 * gained one.
 */
import * as THREE from 'three';
import { skin } from './materials.js';

/**
 * The roles a scheme can paint. Anything a scheme leaves out keeps the body
 * colour, so a two-colour livery is two lines rather than six.
 */
export const ROLES = ['body', 'wing', 'stabiliser', 'fin', 'nacelle', 'pylon'];

/**
 * Five schemes.
 *
 * `metal` turns a colour into bare polished alloy rather than paint -- high
 * metalness, low roughness -- which is a different surface and not a darker
 * shade of one. Painted aluminium and unpainted aluminium look nothing alike
 * under a moving light, and a scheme that tried to fake one with the other
 * would read as grey plastic.
 */
export const SCHEMES = {
  'house white': {
    note: 'the base skin everywhere',
    body: '#e9eaec',
  },
  'navy tail': {
    note: 'white body, deep blue fin and engines',
    body: '#eef0f2',
    fin: '#16305c', nacelle: '#16305c', pylon: '#eef0f2',
  },
  'polished metal': {
    note: 'bare alloy body, painted fin',
    body: { color: '#c6ccd4', metal: true },
    wing: { color: '#bcc3cb', metal: true },
    pylon: { color: '#c6ccd4', metal: true },
    nacelle: { color: '#ced4db', metal: true },
    fin: '#e9eaec', stabiliser: { color: '#bcc3cb', metal: true },
  },
  'crimson': {
    note: 'white body, red tail surfaces and engines',
    body: '#f0eeec',
    fin: '#a81f2b', stabiliser: '#a81f2b', nacelle: '#a81f2b',
  },
  'slate': {
    note: 'low-visibility grey, darker surfaces',
    body: '#5b636e',
    wing: '#474f5a', stabiliser: '#474f5a',
    fin: '#2c333c', nacelle: '#4f5761', pylon: '#474f5a',
  },
};

/**
 * Materials are cached by their full specification, not by colour.
 *
 * Two schemes asking for the same grey -- one painted, one polished -- want two
 * different materials, and keying on colour alone would silently hand the
 * second whichever the first happened to create.
 */
const cache = new Map();
export const painted = {};          // registry, so a page can find them all

function materialFor(spec) {
  const s = typeof spec === 'string' ? { color: spec } : spec;
  const key = JSON.stringify(s);
  if (cache.has(key)) return cache.get(key);
  const m = new THREE.MeshStandardMaterial({
    color: new THREE.Color(s.color),
    roughness: s.roughness ?? (s.metal ? 0.16 : 0.42),
    metalness: s.metalness ?? (s.metal ? 1.0 : 0.05),
  });
  m.name = `paint ${key}`;
  cache.set(key, m);
  painted[key] = m;
  return m;
}

/**
 * Which meshes belong to which role.
 *
 * Only meshes still carrying the BASE skin are collected. That is what makes
 * "the white parts" precise: the model already distinguishes painted surfaces
 * from tyres, hot metal and cavities by giving them different materials, so
 * this needs no list of exceptions and cannot fall out of date with one.
 *
 * It also means the lookup has to run against a FRESHLY built aircraft. Once a
 * scheme has been applied the parts no longer carry the base skin, which is why
 * `paint` takes the roles it was given rather than finding them again.
 */
export function paintRoles(aircraft) {
  const roles = Object.fromEntries(ROLES.map((r) => [r, []]));
  const collect = (obj, role) => {
    obj?.traverse?.((o) => { if (o.isMesh && o.material === skin) roles[role].push(o); });
  };
  const p = aircraft.userData.parts ?? {};
  collect(p.fuselage, 'body');
  collect(p.wing, 'wing');
  collect(p.horizontalTail, 'stabiliser');
  collect(p.verticalTail, 'fin');
  for (const pod of p.engines ?? []) {
    for (const child of pod.children) {
      // A pod holds an engine and a pylon. The pylon is one mesh named for
      // itself; everything else paintable in there is nacelle.
      collect(child, child.name === 'underMountPylon' ? 'pylon' : 'nacelle');
    }
  }
  return roles;
}

/** Apply a scheme to roles collected earlier. Unlisted roles take the body. */
export function paint(roles, schemeName) {
  const scheme = SCHEMES[schemeName] ?? SCHEMES['house white'];
  const body = materialFor(scheme.body ?? '#e9eaec');
  for (const role of ROLES) {
    const m = scheme[role] === undefined ? body : materialFor(scheme[role]);
    for (const mesh of roles[role] ?? []) mesh.material = m;
  }
  return scheme;
}

export const schemeNames = Object.keys(SCHEMES);
