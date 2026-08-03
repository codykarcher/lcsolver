/**
 * A D8 assembled from a sizing solve.
 *
 * The same parts as the conventional aeroplane, arranged the way the D8 puts
 * them: a wide flat double-bubble body, a low wing, TWIN fins on the aft
 * upper body with the engines between them, and the horizontal tail carried on
 * top of the fins.
 *
 * Everything the solve determines is read from it and nothing is retyped. What
 * the solve does not determine -- where the wing sits on the body's height, how
 * far apart the fins stand, how deeply the nacelles are let into the upper
 * skin -- is a choice, and is collected in D8_CHOICES where it can be seen to
 * be one.
 */
import * as THREE from 'three';
import { d8Fuselage } from './fuselage.js';
import { liftingSurface, verticalTail } from './wing.js';
import { bareTurbofan, turbofan, embeddedTurbofan } from './engines.js';
import { landingGear } from './landing_gear.js';
import { carveInto, wouldCarve, carveOut,
         nacelleDuct, nacelleDuctSurface } from './carve.js';

const DEG = Math.PI / 180;
const IN = 0.0254;

/**
 * Clearance the fuselage's own channel section leaves round what it holds.
 *
 * `fuselage.js` owns the default; it is repeated here because the afterbody's
 * width is worked back from where the fins stand, and that arithmetic has to
 * know the same number.
 */
const CHANNEL_GAP = 0.02;

/**
 * The numbers a sizing solve is two-dimensional about.
 *
 * Read against the general arrangement rather than invented: the fins stand
 * outboard of the nacelles, the nacelles sit half into the upper skin, and the
 * tailplane bridges the fin tips.
 */
export const D8_CHOICES = {
  wingRootY:     -0.55,   // wing root chord height, in body half-heights
  /**
   * Fin cant, degrees outboard from vertical.
   *
   * ZERO. The 14 degrees this carried was read off the general arrangement
   * drawing, and it is not what the solve does: the solve has the ability and
   * leaves it unused, so the fins stand upright. It is kept as a parameter
   * because the machinery for it is built and a later deck may want it.
   *
   * Not cosmetic -- the cant moved the fin tip 0.79 m outboard, and the
   * tailplane sits on those tips, so it carried the whole tailplane with it.
   */
  finCant:        0.0,
  /**
   * Engine axis height, as a fraction of the BODY's height up from its keel.
   *
   * Referred to the cabin's height rather than the local one: the afterbody is
   * closing where the engines sit, and a fraction of a collapsing number is not
   * a place. Set against the arrangement drawing, not derived.
   */
  engineHeight:   0.70,
  /**
   * The afterbody's trailing-edge half-width, in engine outer extents.
   *
   * One puts the walls of the channel exactly on the outside of the engines.
   * The cradle needs this: where the body runs wider than what it holds, a ray
   * out of the section crosses skin, the opening, then skin again, and a
   * section written as one radius per angle cannot say that.
   */
  tailSpan:       1.00,
  planTaperA:     1.0,    // the plan narrows EARLY -- the depth does not
  planTaperB:     2.0,
  runStart:       1.5,    // where the run begins, in tailcone lengths off the tail
  tailHold:       4.0,    // how squarely the afterbody holds its depth aft
  tailTrough:     0.30,   // valley between the lobes, in local half-heights
  troughWidth:    0.55,   // angular half-width of that valley, radians
  htOnFins:       true,   // tailplane carried on the fin tips, not the body
  wingDihedral:   2.0,
  wingSweepC4:   20.0,
  htSweepC4:     22.0,
  vtSweepC4:     30.0,
  tcRoot: 0.145, tcCrank: 0.125, tcTip: 0.105, tcTail: 0.09,
  /**
   * Carve the engine duct out of the afterbody.
   *
   * A toggle, and defaulted on, because the shape underneath it is the one
   * that was arrived at by hand and is not to be disturbed: turning this off
   * gives back exactly that body, vertex for vertex, rather than some earlier
   * approximation of it. See `carve.js` for why this cannot be a section.
   */
  ductCarve:      true,
  /** Clearance between the duct's walls and the engines they hold. */
  ductGap:        0.04,
  /**
   * How deep into the body the duct's lip is blended, in metres.
   *
   * Zero gives the bare cut, which meets the skin at a right angle along the
   * median of the rim and folds back on itself at 158 degrees at the worst.
   *
   * Small, because the thing it is rounding is small: where the walls come
   * through the upper skin they are a few centimetres tall, so a radius of the
   * 0.30 m that first looked right swallowed the wall whole and came out worse
   * than no blend at all.
   *
   * 0.08 is the largest that still closes. The flare moves the cut outboard, and
   * past this the duct's own sheet starts running out of body to be clipped
   * against before the skin's cut edge does: at 0.10 ten of the seam's edges
   * open up and the worst gap goes from 18 to 45 mm.
   */
  ductBlend:      0.0,
  /**
   * Put the engines in nacelles, and cut away the part the body already fills.
   *
   * A toggle because the bare engine is what the afterbody was shaped against,
   * and it is worth being able to see the two side by side.
   */
  nacelles:       true,
  /**
   * Target triangle edge, in metres, for the body and the duct together.
   *
   * 0.18 puts about 170 stations along a 30 m body, against the 140 this ran
   * at, and gives 276 mm triangles where they were 337.
   *
   * Chosen off the measured trade rather than picked. Going to 0.13 with a
   * matching sheet buys 39 mm triangles on the duct but costs 3.9 seconds a
   * build against 1.4, and the lip -- the thing the density is FOR -- reads no
   * better: 52 degrees against 47. The body's own curvature is metres, so past
   * about here the extra triangles are spent on a surface that was already
   * smooth. The duct's sheet is meshed on its own, finer, because what it
   * carries is small. `quality` scales both.
   */
  ductEdge:       0.18,
  /**
   * The duct sheet's triangle edge, as a fraction of the lip's radius.
   *
   * Its own number because the sheet and the skin carry different things: the
   * body's features are metres, the lip's round is 80 mm. Measured at 0.027,
   * 0.035, 0.045 and 0.060 m the lip sits between 52 and 67 degrees with no
   * trend, so there is nothing to buy below about half the radius -- and
   * plenty to pay, since the sheet is where nearly all the build time goes.
   * 0.625 of the radius puts the sheet's triangles at 71 mm -- finer than the
   * 101 mm they were before any of this, and square where they were 2.5:1.
   */
  ductSheetGrain: 0.625,
  /**
   * Mesh density, as a multiplier on every count that drives it.
   *
   * One knob rather than four, because the four have to move together: a fine
   * skin carved against a coarse duct sheet is not a better aeroplane, it is
   * the same seam mismatch with more triangles on one side of it.
   *
   * 1.0 is the density everything above was judged at. It costs about half a
   * second a build, nearly all of it the carve, which is fine for looking at
   * one aeroplane and too slow to sit inside a solver's loop.
   */
  quality:        1.0,
};

/** Leading-edge sweep from a quarter-chord one. `k` is 1/2 tip-to-tip, 1/4 for a fin. */
function leadingEdgeSweep(sweepC4, rootChord, taper, span, mirrored = true) {
  const k = mirrored ? 0.5 : 0.25;
  return Math.atan(Math.tan(sweepC4 * DEG) + (k * rootChord * (1 - taper)) / span) / DEG;
}

/**
 * Leading-edge sweep from the SPAR AXIS sweep the solve publishes.
 *
 * `tan_Lambda` is measured about the box axis at 0.40 chord, so the conversion
 * is the same one term as above with p in place of the quarter: for a straight
 * taper, tan(L_axis) = tan(L_LE) + p (c_tip - c_root) / s.
 *
 * `span` is the SEMI-span for a mirrored surface and the full height for a fin,
 * because that is the length the chord actually tapers over in each case.
 */
function leadingEdgeFromAxis(tanAxis, rootChord, taper, span, p = 0.40) {
  return Math.atan(tanAxis + (p * rootChord * (1 - taper)) / span) / DEG;
}

export function d8Aircraft(deck, opts = {}) {
  const d = { ...D8_CHOICES, ...deck };
  /**
   * How finely everything is meshed, from one number.
   *
   * The skin's stations and segments, and the duct sheet's grid, all scale
   * together. Rounded to even numbers because the section's meshing pairs
   * points across the symmetry plane and an odd count puts a seam down the
   * middle of the aeroplane.
   */
  const q = Math.max(0.15, opts.quality ?? d.quality);
  const grain = (n) => 2 * Math.max(3, Math.round((n * q) / 2));
  /**
   * The size of a triangle, in metres, which is what actually decides how a
   * surface reads. Everything that meshes here is derived from it.
   *
   * Counts were the wrong thing to set. 140 stations by 64 segments gave the
   * skin 337 mm edges while the duct's sheet ran at 67 mm, so the trough was
   * a finely meshed hole in a coarse body -- and the seam between them was
   * limited by the coarse side, which is also the side that cannot be moved.
   */
  const edge = (opts.edge ?? d.ductEdge) / q;
  const g = new THREE.Group();
  const parts = {};

  /* ---- fuselage ------------------------------------------------------- */
  // `radius` is the body's HALF-HEIGHT and `cabinWidth` its width over that,
  // which is exactly the pair the solve reports.
  const halfH = d.fuseHalfHeight, halfW = d.fuseHalfWidth;
  // 55% of the body's height up from its keel.
  const engineAxisY = -halfH + d.engineHeight * 2 * halfH;
  /**
   * Where the engine's nose actually falls, worked out before the body is
   * built, because the body's run has to finish there.
   *
   * MEASURED, not `engineX - length/2`: the turbofan is not centred on its own
   * origin, so half its length forward of the station is 1.28 m ahead of its
   * real leading edge -- and the run finished that far early, flattening off
   * before it reached the thing it was climbing to meet.
   */
  const probe0 = bareTurbofan({ rFan: 1, bypassRatio: 9 });
  const rFanFor = (d.nacelleDia / 2) / probe0.userData.rMax;
  const engineNoseX = d.engineX - new THREE.Box3()
    .setFromObject(bareTurbofan({ rFan: rFanFor, bypassRatio: 9 })).max.z;
  const fuse = d8Fuselage({
    // Stations along the body and segments around it, both from the edge
    // target, so the skin's triangles come out square at any size of aeroplane.
    nStation: 2 * Math.max(8, Math.round(d.fuseLength / edge / 2)),
    nSeg: 2 * Math.max(8, Math.round(
      (2 * Math.PI * Math.sqrt((halfW * halfW + halfH * halfH) / 2)) / edge / 2)),
    radius: halfH,
    length: d.fuseLength,
    noseD: d.noseLength / (2 * halfH),
    shape: {
      cabinWidth: halfW / halfH,
      noseWidth: halfW / halfH,
      tailD: d.coneLength / (2 * halfH),
      /**
       * The afterbody narrows in plan until its trailing edge reaches the FINS.
       *
       * Left at the cabin's width -- which is what the standalone body does,
       * and is right for a body with nothing on the back of it -- the planform
       * is a constant-width slab and the engines sit on a shelf that runs on
       * past them to either side. So it is taken from something on the back of
       * the aeroplane, and the right something is the fin station.
       *
       * It used to be the engines' outer extent, which was the best available
       * while the fins were placed by this file too -- the two agreed because
       * both came from the same guess. Now the solve names `y_vt` and they do
       * not: the fins stand at 1.952 and the engines reach 1.847, so a body
       * sized to the engines ends 87 mm inboard of its own fins and they hang
       * off the corner in mid-air.
       *
       * The fins are what the body has to carry, so the body ends where they
       * stand. The engines then sit inboard of the trailing edge rather than
       * flush with it, which is the correct way round: the trough holds them
       * and the corner holds the fins.
       */
      tailWidth: (d.finY ?? d.tailSpan * (d.engineY + d.nacelleDia / 2)) / halfH,
      /**
       * The afterbody holds its DEPTH back to the engines, then closes.
       *
       * The smoothstep the bare body uses falls away from the moment the cabin
       * ends, which is right when there is nothing to carry: it had the body
       * 0.23 m deep where the engines sit, 12 per cent of full, so they perched
       * on an edge and the dish they are supposed to sit in had no room to
       * exist. A power law holds it at 77 per cent there and does its closing
       * in the last fifth instead.
       */
      tailLaw: 'power', tailA: d.tailHold, tailB: 0.70,
      /**
       * Where the afterbody's centreline ends up: in the CHANNEL.
       *
       * A section is one radius per angle about that centreline, so the
       * centreline has to be inside the shape it describes. Left at its own
       * default the body closes at 0.35 of its height while the channel, with
       * the engines at 0.70, floors at 0.47 -- the origin ends up below the
       * floor, outside the section, and the whole afterbody collapses to
       * nothing. Aiming it at the middle of the channel is not a tuning
       * choice; it is what keeps the description valid.
       */
      tailEdgeHeight: ((engineAxisY - (d.nacelleDia / 2 + 0.02) / 2) + halfH) / (2 * halfH),
      // The plan narrows on its own law, and earlier than the depth, so the
      // afterbody is no wider than the engines by the time it has to hold them.
      tailWidthA: d.planTaperA, tailWidthB: d.planTaperB,
      /**
       * And the afterbody closes into a channel: a flat floor with sides
       * rounding up at the engines' own radius.
       *
       * The SPACING is set so the hull reaches the fins, not so the rounds sit
       * on the engines. This is the section that decides the body's width at
       * the trailing edge -- `tailWidth` above is overridden here, since a
       * station this far aft is all channel -- and the width it has to make is
       * the fin station, because that is what the corner carries.
       *
       * The rounds no longer sitting exactly on the engines costs nothing now:
       * the trough the engines lie in is CARVED and follows the nacelle's own
       * inner line. This section only has to be the outer shape.
       */
      channel: {
        x: d.finY != null ? d.finY - (d.nacelleDia / 2 + CHANNEL_GAP) : d.engineY,
        y: engineAxisY,
        r: d.nacelleDia / 2,
      },
      /**
       * The run starts 1.5 tailcone-lengths off the tail -- so it begins well
       * forward of the cone, where there is length to climb in -- and runs
       * straight through to the body's TRAILING EDGE. Every metre it is given
       * is a degree it does not have to climb at: over the cone alone the keel
       * swept up at 48 degrees, to the engines' nose at 23, and to the trailing
       * edge at less again.
       */
      channelFromX: d.fuseLength - d.runStart * d.coneLength,
      channelToX: d.fuseLength,
    },
    detail: opts.detail ?? false,
  });
  g.add(fuse); parts.fuselage = fuse;
  const u = fuse.userData;

  /* ---- wing ----------------------------------------------------------- */
  // One straight taper inboard, area preserved, for the same reason as the
  // conventional aeroplane: the solve's wing-box break sits outboard of where a
  // low wing actually leaves a body, and the corner would be in the open.
  const cKink = d.wingRootChord * d.wingCrankRatio;
  const inboard = d.wingRootChord * d.wingEtaRoot
    + 0.5 * (d.wingRootChord + cKink) * (d.wingKink - d.wingEtaRoot);
  const straightRoot = (2 * inboard) / d.wingKink - cKink;

  const wing = liftingSurface({
    span: d.wingSpan, rootChord: straightRoot, etaRoot: 0,
    kink: d.wingKink, crankRatio: cKink / straightRoot, tipRatio: d.wingTipRatio,
    sweep: d.wingSweepLE, dihedral: d.wingDihedral,
    twistRoot: 0, twistTip: -3,
    rootThickness: d.tcRoot, crankThickness: d.tcCrank, tipThickness: d.tcTip,
    root: '2412', kinkFoil: null, tip: '2410',
  });
  const wingRootLE = d.wingQuarterX - 0.25 * d.wingRootChord;
  const wingY = d.wingRootY * halfH;
  wing.position.set(0, wingY, -wingRootLE);
  g.add(wing); parts.wing = wing;

  /* ---- the fins ------------------------------------------------------- */
  /**
   * Each fin is the solve's planform, unhalved, and there are `n_vt` of them.
   *
   * The solve settles this itself and it is worth being precise about, because
   * the two readings differ by a factor of two: b_vt, c_root_vt and lambda_vt
   * reproduce S_vt exactly on their own, so S_vt describes ONE surface and n_vt
   * says how many there are. The total is the product. Splitting the area
   * instead -- which is what this did before the solve carried n_vt -- builds
   * two fins that together come to what one was supposed to be.
   */
  const fin = {
    height: d.vtHeight, rootChord: d.vtRootChord, taper: d.vtTaper,
    area: 0.5 * d.vtHeight * d.vtRootChord * (1 + d.vtTaper),
  };
  // The solve gives the leading edge outright, which is the station that
  // actually places the surface; the arm is a moment length measured to a
  // reference this file would have to guess at.
  const vtRootLE = d.vtLE ?? (d.wingQuarterX + d.vtArm - 0.25 * fin.rootChord);
  const vtQuarterX = vtRootLE + 0.25 * fin.rootChord;
  /**
   * The fins are hung by their root TRAILING EDGE, on the body's back upper
   * corners.
   *
   * Which is where the two actually meet: the solve puts the fin's trailing
   * edge on the body's own trailing edge, and the channel puts the body's upper
   * corners there too, so the corner is a single point both of them already
   * own. Hanging them at the crown over their QUARTER chord instead -- which is
   * a station a good way forward, where the afterbody is still much deeper --
   * floated them well above it.
   *
   * The root's leading edge ends up inside the body, which is correct: the fin
   * is faired in forward and emerges aft.
   */
  const teCornerY = u.crownAt(-d.fuseLength);
  /**
   * Laterally, where the solve puts them -- `y_vt` -- and not the body's own
   * corner.
   *
   * The corner was the reading while the deck was silent, and it is 1.87 here
   * against the solve's 1.952. That 85 mm is not cosmetic: the tailplane sits
   * on the fin tips, and the solve aligns its 0.40c spar with the fin tip's at
   * one spanwise station. Standing the fins anywhere else slides the tailplane
   * along its own sweep -- at the corner the join was open by 140 mm.
   */
  // The name used to belong to a CHOICE here -- 0.62, a fraction of the body's
  // half-width -- which nothing read. Left in place it would have shadowed the
  // deck's metres with a fraction and stood the fins at 0.62 m, inside the
  // nacelles, on any deck that predates `y_vt`. Deleted rather than renamed:
  // the solve owns this number now.
  const teCornerX = d.finY ?? u.halfWidthAt(-d.fuseLength);
  parts.verticalTails = [];
  // Placed symmetrically about the centreline, whatever the count.
  const sides = d.finCount >= 2 ? [1, -1] : [0];
  for (const side of sides) {
    const vt = verticalTail({
      height: fin.height, rootChord: fin.rootChord, taperRatio: fin.taper,
      // The solve's own spar-axis sweep when it publishes one; the hard-coded
      // quarter-chord choice only as a fallback for a deck that does not.
      sweep: d.vtSweepAxisTan != null
        ? leadingEdgeFromAxis(d.vtSweepAxisTan, fin.rootChord, fin.taper, fin.height)
        : leadingEdgeSweep(d.vtSweepC4, fin.rootChord, fin.taper, fin.height, false),
      thickness: d.tcTail,
      // Positive cant leans the tip to +x, so the port fin takes the negative
      // of it and the pair splay outboard rather than both leaning one way.
      cant: side * d.finCant,
    });
    vt.position.set(side * teCornerX, teCornerY, -vtRootLE);
    vt.name = side > 0 ? 'starboardFin' : 'portFin';
    vt.userData.side = side;
    g.add(vt); parts.verticalTails.push(vt);
  }

  /* ---- tailplane, carried on the fin tips ----------------------------- */
  const ht = liftingSurface({
    kink: null, span: d.htSpan, rootChord: d.htRootChord, taperRatio: d.htTaper,
    sweep: d.htSweepAxisTan != null
      ? leadingEdgeFromAxis(d.htSweepAxisTan, d.htRootChord, d.htTaper, d.htSpan / 2)
      : leadingEdgeSweep(d.htSweepC4, d.htRootChord, d.htTaper, d.htSpan),
    dihedral: 0, twistRoot: 0, twistTip: 0,
    thickness: d.tcTail, symmetric: true,
  });
  const htRootLE = d.htLE ?? (d.wingQuarterX + d.htArm - 0.25 * d.htRootChord);
  const htQuarterX = htRootLE + 0.25 * d.htRootChord;
  // Level with the fin tips when it rides on them, which is what makes the
  // empennage a pi rather than a cross.
  const finTipY = teCornerY + fin.height * Math.cos(d.finCant * DEG);
  const htY = d.htOnFins ? finTipY : u.crownAt(-htQuarterX);
  ht.position.set(0, htY, -htRootLE);
  g.add(ht); parts.horizontalTail = ht;

  /* ---- engines, on top of the body between the fins ------------------- */
  /**
   * Nacelled, and then cut back to whatever the body does not already contain.
   *
   * Bare was the earlier reading, and the reasoning was that on a D8 the
   * afterbody IS the fairing, so a cowl would read as an engine parked on the
   * fuselage. That is right about the buried part and wrong about the rest: the
   * half standing proud of the trough is a nacelle like any other, and without
   * one the fan case simply ends in mid-air.
   *
   * So the whole nacelle is built and the part inside solid body is removed --
   * which is the same carve the duct uses, run the other way round.
   *
   * Sized on the BARE engine either way, so the cowl WRAPS what is already
   * there rather than replacing it.
   *
   * Sizing the nacelle to `nacelleDia` instead looks defensible -- that is what
   * the solve calls the number, and it is what the duct's cradle holds -- but
   * it shrinks the fan to fit a cowl inside the same envelope, which moves the
   * engine. The engine's position and size are the deck's, and the duct was
   * shaped around them; the cowl is an addition. So the fan stays exactly where
   * it was and the nacelle stands 1.177 times its radius, 0.99 m against 0.84,
   * which is why a good deal of it ends up inside the body.
   */
  const podded = opts.nacelles ?? d.nacelles;
  const probe = bareTurbofan({ rFan: 1, bypassRatio: 9 });
  const rFan = (d.nacelleDia / 2) / probe.userData.rMax;
  // Length from the deck too, not left to the component's own proportions. A
  // D8's propulsor is short and fat -- 1.24 m long on a 1.68 m diameter, an
  // aspect of 0.74 where a podded engine is nearer 1.8 -- because the duct is
  // short when the fan is fed off the body. Letting the component choose gives
  // a 3.2 m engine that hangs a metre and a half past the tail.
  const rNac = d.nacelleDia / 2;
  parts.engines = [];
  for (const side of [1, -1]) {
    const pod = new THREE.Group();
    const eng = podded ? embeddedTurbofan({ rFan, bypassRatio: 9 })
                       : bareTurbofan({ rFan, bypassRatio: 9 });
    pod.add(eng);
    pod.position.set(side * d.engineY, engineAxisY, -d.engineX);
    pod.userData.isEnginePod = true;
    pod.userData.side = side;
    pod.name = side > 0 ? 'starboardPod' : 'portPod';
    g.add(pod); parts.engines.push(pod);
  }

  /**
   * The cowl's outer line, as a floor for the trough to follow.
   *
   * Read off the nacelle that was actually built rather than recomputed, so the
   * trough cannot drift from the thing it is cradling. `cowlOuter` is published
   * in the engine's own frame, running from the lip forward to the tail aft;
   * the pod sits at `engineX`, so a station converts straight across.
   */
  const nacelleTrough = (() => {
    if (!podded) return {};
    const cowl = parts.engines[0]?.children[0]?.userData?.cowlInner;
    if (!cowl?.length) return {};
    const zOf = (x) => d.engineX - x;
    const rAt = (x) => {
      const z = zOf(x);
      if (z >= cowl[0][0]) return cowl[0][1];
      for (let i = 0; i < cowl.length - 1; i++) {
        const a = cowl[i], b = cowl[i + 1];
        if (z <= a[0] && z >= b[0]) {
          const t = (z - a[0]) / ((b[0] - a[0]) || 1);
          return a[1] + (b[1] - a[1]) * t;
        }
      }
      return cowl[cowl.length - 1][1];
    };
    return {
      axisY: engineAxisY,
      spacing: d.engineY,                         // half the distance between them
      noseFrom: d.engineX - cowl[0][0],           // the inlet lip's station
      radiusAt: (x) => rAt(x) + d.ductGap,
    };
  })();

  /* ---- the cutout the nacelles sit in --------------------------------- */
  /**
   * The convex hull of the two ducts aft, a straight rectangle forward.
   *
   * Aft of the nacelles' narrowest station the cut is the hull of their two
   * INNER circles -- a stadium, since that is what two circles side by side
   * hull to -- so at every station it is exactly their section and the flat
   * between them. Forward of that station it is the section the hull had there,
   * run straight to the end of the cabin.
   *
   * Cut out of the MESH, not out of the body's section law: a section here is
   * one radius per angle about one centre, and this shape needs two spans of
   * material on the same ray. See `carve.js`.
   */
  if (opts.ductCarve ?? d.ductCarve) {
    const cowl = podded && parts.engines[0]?.children[0]?.userData?.cowlInner;
    if (cowl?.length) {
      const zOf = (x) => d.engineX - x;
      const rRaw = (x) => {
        const z = zOf(x);
        if (z >= cowl[0][0]) return cowl[0][1];
        for (let i = 0; i < cowl.length - 1; i++) {
          const a = cowl[i], b = cowl[i + 1];
          if (z <= a[0] && z >= b[0]) {
            return a[1] + (b[1] - a[1]) * ((z - a[0]) / ((b[0] - a[0]) || 1));
          }
        }
        return cowl[cowl.length - 1][1];
      };
      /**
       * The narrowest station, found by looking rather than assumed.
       *
       * An inlet contracts from the highlight to a throat and opens out again,
       * so the smallest section is inside the lip, not at it -- and where it
       * falls depends on the mean line, which is the nacelle's business and not
       * this file's.
       */
      let throatX = d.engineX - cowl[0][0], rMin = Infinity;
      for (let i = 0; i < cowl.length; i++) {
        const x = d.engineX - cowl[i][0];
        if (x < d.engineX && cowl[i][1] < rMin) { rMin = cowl[i][1]; throatX = x; }
      }
      const duct = nacelleDuct({
        axisY: engineAxisY,
        spacing: d.engineY,
        radiusAt: (x) => rRaw(x) + d.ductGap,
        throatX,
        fromX: -u.cabinZ[1],
        toX: d.fuseLength,
        crown: u.crownAt(u.cabinZ[1]),
      });
      const skinMesh = fuse.userData.skinMesh;
      const triangles = (g2) => {
        const ix = g2.getIndex(), pp = g2.getAttribute('position');
        return (ix ? ix.count : pp.count) / 3;
      };
      const before0 = triangles(skinMesh.geometry);
      for (const fields of duct.passes) {
        const before = skinMesh.geometry;
        skinMesh.geometry = carveOut(before, fields);
        before.dispose();
      }
      const after0 = triangles(skinMesh.geometry);
      const { wall, front } = nacelleDuctSurface(duct, u.depthInside);
      const wallMesh = new THREE.Mesh(wall, skinMesh.material);
      const frontMesh = new THREE.Mesh(front, skinMesh.material);
      wallMesh.name = 'ductWall';
      frontMesh.name = 'ductFront';
      fuse.add(wallMesh); fuse.add(frontMesh);
      fuse.userData.duct = {
        skin: { before: before0, after: after0 },
        throatX, fromX: duct.fromX, toX: duct.toX, spacing: d.engineY,
        axisY: engineAxisY, rThroat: duct.rT, radiusAt: duct.radiusAt,
        inside: duct.inside, depth: duct.depth, passes: duct.passes,
        mesh: wallMesh, front: frontMesh,
      };
      const body = (p) => u.depthInside(p.x, p.y, p.z);
      /**
       * The fins lose what stands IN the cut; the nacelles lose what is left
       * outside it. Opposite rules, and they were briefly the same one.
       *
       * A fin's job is to stand on the body. Where it crosses into the open
       * trough it is a slab hanging in the middle of the duct, so what goes is
       * `in the body AND in the cut` -- one pass per region, since a pass
       * removes an intersection and that is exactly what this is.
       */
      for (const vt of parts.verticalTails) {
        for (const fields of duct.passes) carveInto(vt, [body, ...fields]);
      }
      for (const pod of parts.engines) {
        for (let round = 0; round < 2; round++) {
          for (const fields of duct.passes) {
            for (const f of fields) {
              const fs = [body, (p) => -f(p)];
              if (wouldCarve(pod, fs)) carveInto(pod, fs);
            }
          }
        }
      }
    }
  } else {
    // No cut, so the body is whole and everything inside it is solid: the
    // nacelles lose whatever the body already fills, in one pass.
    const body = (p) => u.depthInside(p.x, p.y, p.z);
    for (const pod of parts.engines) {
      if (wouldCarve(pod, [body])) carveInto(pod, [body]);
    }
  }

  /* ---- undercarriage --------------------------------------------------- */
  const mainTyre = d.mainTyreIn * IN / 2;
  const mkMain = () => landingGear({
    wheels: 2, rTire: mainTyre, rWheel: mainTyre * 0.55,
    lStrut: d.mainStrut, rStrut: 0.10,
  });
  /**
   * The main legs hang wherever puts the wheels in one plane -- SOLVED, not
   * chosen from a shortlist.
   *
   * Which datum the solve sized its struts against is not in the solve, and
   * guessing it has now been wrong three times running: one solve wanted the
   * wing (its main leg reached 0.881 m further than the nose), the next wanted
   * the keel (0.158 m less), and this one wants the wing again (0.871 m more).
   * Each reload was a scramble to re-pick, and picking from two fixed options
   * only ever lands close by luck -- on this solve the wing, the better of the
   * two, still leaves 371 mm.
   *
   * What is not in doubt is that an aeroplane stands on its wheels. That is one
   * equation and the attachment height is the one unknown in it, so it is
   * solved rather than selected: put the main attachment exactly as far above
   * the nose attachment as its leg is longer. The wheels are then coplanar by
   * construction, whatever the solve does next.
   *
   * Reported against the keel and the wing below, because a height that comes
   * out somewhere structurally silly is worth seeing.
   */
  const noseAttachY = u.keelAt(-d.noseX);
  const mainReach = d.mainStrut + (d.mainTyreIn * IN) / 2;
  const noseReach = d.noseStrut + (d.noseTyreIn * IN) / 2;
  const mainAttachY = noseAttachY + mainReach - noseReach;
  const mainContact = mainAttachY + mkMain().userData.contactY;
  parts.gear = [];
  for (const side of [1, -1]) {
    const lg = mkMain();
    lg.position.set(side * d.mainY, mainAttachY, -d.mainX);
    lg.userData.gearKind = 'main';
    lg.userData.side = side;
    lg.userData.retract = { axis: 'z', angle: -side * Math.PI / 2 };
    lg.name = side > 0 ? 'starboardGear' : 'portGear';
    g.add(lg); parts.gear.push(lg);
  }
  const noseTyre = d.noseTyreIn * IN / 2;
  const nose = landingGear({
    wheels: 2, rTire: noseTyre, rWheel: noseTyre * 0.55,
    lStrut: d.noseStrut, rStrut: 0.075,
  });
  nose.position.set(0, noseAttachY, -d.noseX);
  nose.userData.gearKind = 'nose';
  nose.userData.side = 0;
  nose.userData.retract = { axis: 'x', angle: -Math.PI / 2 };
  nose.name = 'noseGear';
  g.add(nose); parts.gear.push(nose);
  const noseContact = noseAttachY + nose.userData.contactY;

  /* ---- how it sits ----------------------------------------------------- */
  const wheelbase = d.mainX - d.noseX;
  const attitude = Math.atan((noseContact - mainContact) / wheelbase) / DEG;
  const ground = Math.min(noseContact, mainContact);
  if (opts.sitOnGround ?? true) g.position.y = -ground;

  Object.assign(g.userData, {
    isD8: true, deck: d, parts,
    areas: {
      wing: wing.userData.area,
      horizontalTail: ht.userData.area,
      // Every fin together, which is what the deck's solvedAreas counts.
      verticalTail: parts.verticalTails.reduce((t, f) => t + f.userData.area, 0),
    },
    fin, finCount: parts.verticalTails.length, engineAxisY,
    /**
     * Where the solved main-gear attachment landed, against the two datums it
     * used to be picked from. A height well outside the body would mean the
     * deck's struts do not describe this aeroplane.
     */
    mainGearAttach: {
      y: mainAttachY,
      aboveKeel: mainAttachY - u.keelAt(-d.mainX),
      aboveWing: mainAttachY - (wingY + d.mainY * Math.tan(d.wingDihedral * DEG)),
      bodyDepth: u.crownAt(-d.mainX) - u.keelAt(-d.mainX),
    },
    wheelbase, groundAttitude: attitude, ground,
    noseContact, mainContact,
    leadingEdgeSweeps: {
      wing: wing.userData.sweep,
      horizontalTail: ht.userData.sweep,
      verticalTail: parts.verticalTails[0].userData.sweep,
    },
  });
  return g;
}
