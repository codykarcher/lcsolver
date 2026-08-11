#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""The master list of LCsolver message codes.

Every warning or error LCsolver raises carries one of these codes, embedded
at the front of its message as ``[LC-Wxxx]`` / ``[LC-Exxx]``, so callers can
detect a condition without parsing prose::

    sol = lcsolver.solve(f)
    from lcsolver.core import codes
    if any(codes.HOLOGRAPHIC_ACTIVE in m for m in sol.messages):
        ...

Numbering: ``E`` codes are errors (raised), ``W`` codes are warnings
(captured into ``sol.messages`` by default). The hundreds digit groups by
pipeline stage: 0xx/1xx pre-solve, 2xx solve-time, 3xx post-solve.
"""

# --- errors (raised) -------------------------------------------------------
ILL_POSED = 'LC-E001'            #: pre-solve gate: the stated problem is
                                 #: ill-posed (empty/unbounded variables)
UNIT_MISMATCH = 'LC-E002'        #: units do not balance (UnitMismatch)
UNBUILT_BLOCK = 'LC-E003'        #: a model block never received all its
                                 #: inputs, so it posted no constraints
INFEASIBLE = 'LC-E101'           #: the problem was proven infeasible

# --- pre-solve warnings ----------------------------------------------------
PRESOLVE_FINDINGS = 'LC-W101'    #: gate findings demoted to a warning
                                 #: (diagnostics='warn')
DETECTION_FAILED = 'LC-W102'     #: structure detection failed; solving as a
                                 #: raw NLP instead

# --- solve-time warnings ---------------------------------------------------
BACKEND_FAILED = 'LC-W201'       #: the structured backend failed; fell back
NO_IPOPT_FALLBACK = 'LC-W202'    #: no usable IPOPT; solved with cvxopt
SIA_NOT_CONVERGED = 'LC-W203'    #: SIA returned a best iterate, not a
                                 #: certified optimum (remedies attached)
FALSE_OPTIMUM = 'LC-W204'        #: a GP form reported optimality that its
                                 #: verification solve contradicted
BACKEND_NONOPTIMAL = 'LC-W205'   #: cvxopt returned a non-optimal status
                                 #: without raising
WRITEBACK_FAILED = 'LC-W206'     #: solve succeeded but the solution could
                                 #: not be written onto the model

# --- post-solve warnings ---------------------------------------------------
HOLOGRAPHIC_ACTIVE = 'LC-W301'   #: holographic (should-never-bind)
                                 #: constraints are active at the solution
SENSITIVITIES_UNRELIABLE = 'LC-W302'  #: recovered duals fail stationarity;
                                      #: sensitivities untrustworthy
SENSITIVITIES_AMBIGUOUS = 'LC-W303'   #: degenerate active set; some
                                      #: sensitivities are not determined
AT_FLOOR = 'LC-W304'             #: variables resting on the solver's
                                 #: positivity floor

#: Every code, with a one-line meaning. The authoritative list.
CODES = {
    ILL_POSED: 'pre-solve gate: the stated problem is ill-posed',
    UNIT_MISMATCH: 'units do not balance',
    INFEASIBLE: 'the problem was proven infeasible',
    PRESOLVE_FINDINGS: 'pre-solve findings (demoted to a warning)',
    DETECTION_FAILED: 'structure detection failed; solved as a raw NLP',
    BACKEND_FAILED: 'the structured backend failed; fell back',
    NO_IPOPT_FALLBACK: 'no usable IPOPT; solved with cvxopt',
    SIA_NOT_CONVERGED: 'SIA returned a best iterate, not a certified optimum',
    FALSE_OPTIMUM: 'a GP form reported optimality its verification solve '
                   'contradicted',
    BACKEND_NONOPTIMAL: 'cvxopt returned a non-optimal status',
    WRITEBACK_FAILED: 'solution write-back onto the model failed',
    HOLOGRAPHIC_ACTIVE: 'holographic constraints active at the solution',
    SENSITIVITIES_UNRELIABLE: 'recovered duals fail stationarity',
    SENSITIVITIES_AMBIGUOUS: 'degenerate active set; sensitivities not '
                             'determined',
    AT_FLOOR: "variables resting on the solver's positivity floor",
}


def tag(code, message):
    """``'[LC-W301] <message>'`` -- the standard prefix form."""
    return f'[{code}] {message}'
