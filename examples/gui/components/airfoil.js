/**
 * Aerofoil sections.
 *
 * A section is held as **camber and half-thickness on a shared chordwise
 * distribution**, not as a list of surface points:
 *
 *   x[i]       chord fraction, 0 at the leading edge to 1 at the trailing
 *   camber[i]  the mean line
 *   half[i]    half the local thickness, measured normal to the mean line
 *
 * Surface points are reconstructed on demand. That split is what makes
 * thickness a PARAMETER rather than a property of whichever aerofoil was named:
 * scaling `half` changes the thickness and leaves the camber alone, whereas
 * scaling the y of a surface point list scales the camber with it and turns a
 * 12% section into a differently-cambered 10% one. It also makes blending two
 * sections exact, since camber and thickness interpolate independently.
 *
 * Every section is built on the SAME distribution, which is what lets a wing
 * blend root to crank to tip term for term. Two sections with the same point
 * count but different x stations cannot be blended at all -- the result is a
 * shape neither of them has -- so the distribution lives here, not in callers.
 *
 * It is cosine in x, clustered at both ends, and deliberately not the
 * fuselage's even-arc spacing: a fuselage ring has no distinguished points and
 * an aerofoil has two. Nearly all of an aerofoil's curvature is in the first few
 * per cent of chord, and a trailing edge has to be resolved rather than
 * averaged over.
 */

/** Chordwise stations, cosine-clustered at leading and trailing edges. */
function chordStations(n) {
  const xs = [];
  for (let i = 0; i < n; i++) xs.push((1 - Math.cos(Math.PI * i / (n - 1))) / 2);
  return xs;
}

/**
 * A NACA 4-digit section: `naca4('2412')`.
 *
 *   first digit   maximum camber, in per cent of chord
 *   second        where that camber is, in tenths of chord
 *   last two      maximum thickness, in per cent of chord
 *
 * The thickness polynomial's last coefficient is -0.1036 rather than the
 * original -0.1015. The original is correct to the definition and leaves the
 * trailing edge 0.21% of chord thick, which means the section is not a closed
 * loop and a wing built from it has a slot down its back. -0.1036 closes it and
 * moves nothing else by more than a thousandth of chord.
 */
export function naca4(code, n = 80) {
  const s = String(code).padStart(4, '0');
  const m = parseInt(s[0], 10) / 100;
  const p = parseInt(s[1], 10) / 10;
  const t = parseInt(s.slice(2), 10) / 100;

  const xs = chordStations(n);
  const camber = [], half = [];
  for (const x of xs) {
    half.push(5 * t * (0.2969 * Math.sqrt(x) - 0.1260 * x - 0.3516 * x * x
      + 0.2843 * x ** 3 - 0.1036 * x ** 4));
    camber.push(m === 0 || p === 0 ? 0
      : x < p ? m / (p * p) * (2 * p * x - x * x)
              : m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x * x));
  }
  return { x: xs, camber, half, n, name: `NACA ${s}`, thickness: t, camberMax: m };
}

/**
 * Put an arbitrary section onto the standard distribution, in the same form.
 *
 * Takes surface points in Selig order -- trailing edge, over the top, round the
 * nose, back along the bottom -- and reads them at this file's own chordwise
 * stations. Without this, a measured aerofoil and a NACA one cannot appear on
 * the same wing.
 */
export function resample(points, n = 80) {
  const lead = points.reduce((best, q, i) => (q[0] < points[best][0] ? i : best), 0);
  const at = (chain, x) => {
    for (let i = 0; i < chain.length - 1; i++) {
      const [x0, y0] = chain[i], [x1, y1] = chain[i + 1];
      if ((x - x0) * (x - x1) <= 0 && Math.abs(x1 - x0) > 1e-12) {
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
      }
    }
    return chain[chain.length - 1][1];
  };
  const upper = points.slice(0, lead + 1).slice().reverse();
  const lower = points.slice(lead);
  const xs = chordStations(n), camber = [], half = [];
  for (const x of xs) {
    const yu = at(upper, x), yl = at(lower, x);
    camber.push((yu + yl) / 2);
    half.push((yu - yl) / 2);
  }
  return { x: xs, camber, half, n, name: 'resampled',
           thickness: 2 * Math.max(...half), camberMax: Math.max(...camber) };
}

/** Whatever the caller gave us, as a section: a NACA code, or surface points. */
export function asSection(spec, n = 80) {
  if (Array.isArray(spec)) return resample(spec, n);
  return naca4(spec, n);
}

/** Maximum thickness of a section, as a fraction of chord. */
export function thicknessOf(section) {
  return 2 * Math.max(...section.half);
}

/**
 * Surface points, Selig order: trailing edge, over the top, round the nose,
 * back along the bottom. 2n - 2 of them -- the trailing edge is one point, not
 * two, because the section is closed there.
 *
 * `scale` multiplies the half-thickness only, so a section can be taken to a
 * given t/c without touching its camber. The surfaces are offset along the
 * mean line's NORMAL, not vertically, which is what the definition says and
 * what puts the leading edge of a cambered section slightly ahead of x = 0.
 */
export function sectionPoints(section, scale = 1) {
  const { x, camber, half, n } = section;
  const slope = (i) => {
    const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
    return (camber[b] - camber[a]) / Math.max(x[b] - x[a], 1e-12);
  };
  const surf = (i, sign) => {
    const th = Math.atan(slope(i)), h = half[i] * scale;
    return [x[i] - sign * h * Math.sin(th), camber[i] + sign * h * Math.cos(th)];
  };
  const pts = [];
  for (let i = n - 1; i >= 0; i--) pts.push(surf(i, 1));    // TE -> LE, upper
  for (let i = 1; i <= n - 2; i++) pts.push(surf(i, -1));   // LE -> TE, lower
  return pts;
}
