"""Import shim for EDI's pre-solve chain.

The package these live in has been renamed twice mid-session
(edi.structure/edi.units -> edi.preconditioner -> edi.presolve), each time
breaking every import in this study at once. This tries them in reverse
chronological order so a rename costs one edit here rather than thirty-one
across the tree.
"""
_LAYOUTS = (
    ("edi.presolve.structureDetector", "edi.presolve.unitCorrector"),
    ("edi.preconditioner.structureDetector", "edi.preconditioner.unitCorrector"),
    ("edi.structure.structureDetector", "edi.units.unitCorrector"),
)

structure_detector = unit_corrector = None
for _sd, _uc in _LAYOUTS:
    try:
        import importlib
        structure_detector = importlib.import_module(_sd).structure_detector
        unit_corrector = importlib.import_module(_uc).unit_corrector
        break
    except Exception:
        continue
if structure_detector is None:
    raise ImportError("could not locate EDI's structure detector in any known "
                      f"layout; tried {[l[0] for l in _LAYOUTS]}")

__all__ = ["structure_detector", "unit_corrector"]
