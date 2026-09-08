#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""What a solve produced, held on its own and printed on request.

Pyomo reloads a solution onto the model, so ``pyo.value(m.x)`` answers after a
solve. That is convenient and LCsolver keeps doing it, but it leaves the result with
nowhere to live: to see what happened you must already know which variables to
ask about, one at a time, and which model object to ask -- and asking the wrong
one returns the initial guess with no indication anything is wrong, because
``unit_corrector`` cloned the model before detection.

A :class:`Solution` is the answer as an object. It carries the objective, every
variable and constant with its units and description, and the sensitivities
when they have been computed, and it prints them as a table. ``f.solution``
builds one from the model's current values.

The layout follows the ``OptimizationOutput.result`` table from corsair:
objective first, then variables, then constants, then sensitivities, each
column width computed from its contents so the colons line up.
"""
from __future__ import annotations

import re

import numpy as np

__all__ = ["Solution"]

_DIGITS = re.compile(r'(\d+)')
_ELEMENT = re.compile(r'^(.*)\[([0-9, ]+)\]$')


def _split_element(name):
    """``'V[2]'`` -> ``('V', (2,))``, ``'x[0,1]'`` -> ``('x', (0, 1))``.

    None when the name carries no index. This is the reverse of how Pyomo
    names the element of an indexed component, and is what lets ``sol['V[2]']``
    keep working after the vector is stacked into one array-valued entry.
    """
    m = _ELEMENT.match(name)
    if not m:
        return None
    return m.group(1), tuple(int(p) for p in m.group(2).split(','))


def _element_name(name, index):
    """The name Pyomo gives an element: ``V[2]``, ``x[0,1]``."""
    return name + '[' + ','.join(str(i) for i in index) + ']'


def _alphabetical(name):
    """Sort key for a displayed name: how a reader would alphabetize it.

    Two things a plain string sort gets wrong in a table someone is scanning
    for a name they already know.

    CASE. ASCII orders every capital ahead of every lowercase, so ``Re_station``
    files before ``area_disk`` rather than between ``radius`` and ``rho``. An
    engineering model capitalizes on the convention of the quantity -- Re, CT,
    M -- not to signal precedence, so case is folded away.

    DIGITS. A vector prints one row per element, and lexicographically
    ``[10]`` sits between ``[1]`` and ``[2]``: a twenty-segment chain reads
    0, 10, 11, ... 19, 1, 2. Runs of digits are compared as numbers so the
    elements come out in the order they are indexed.

    Each part is tagged with its kind so a numeric run is never compared
    against a text one, and the untouched name is appended by the caller to
    break ties -- otherwise ``Re`` and ``re`` would order arbitrarily.
    """
    return tuple((1, int(p)) if p.isdigit() else (0, p.lower())
                 for p in _DIGITS.split(name))


def _fmt(value, ndecimal):
    """A number as corsair printed it: integers bare, everything else fixed."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:
        return 'nan'
    if v.is_integer() and abs(v) < 1e15:
        return str(int(v)) + ' ' * (1 + ndecimal)
    if v != 0 and (abs(v) >= 10 ** 6 or abs(v) < 10 ** -(ndecimal + 1)):
        return f'%.{ndecimal}e' % v
    return f'%.{ndecimal}f' % v


def _units(u):
    s = str(u) if u is not None else ''
    return '[-]' if s in ('dimensionless', 'None', '') else f'[{s}]'


class Entry:
    """One named quantity: its value, units and description.

    A vector or array variable is ONE entry, its value a numpy array in the
    declared shape, rather than one entry per element. That is how it was
    declared and how a caller wants it back -- to save, to slice, to hand to
    the next model as a guess -- and printing it per element is the table's
    job, not the data's.
    """

    __slots__ = ('name', 'value', 'units', 'description')

    def __init__(self, name, value, units=None, description=''):
        self.name = name
        self.value = value
        self.units = units
        self.description = description or ''

    @property
    def is_array(self):
        return isinstance(self.value, np.ndarray)

    @property
    def shape(self):
        return self.value.shape if self.is_array else ()

    def element(self, index):
        """The scalar :class:`Entry` for one element of an array entry."""
        if not self.is_array:
            raise KeyError(f'{self.name!r} is a scalar; it has no element '
                           f'{index}')
        return Entry(_element_name(self.name, index),
                     self.value[tuple(index)].item(), self.units,
                     self.description)

    def __repr__(self):
        return f'<{self.name} = {self.value!r} {_units(self.units)}>'


class EntryMap(dict):
    """``{name: Entry}`` that also answers for the elements of an array.

    Iterating gives the stacked names -- ``V``, not ``V[0]``, ``V[1]``,
    ``V[2]`` -- so a loop over the solution sees each quantity once, in the
    shape it was declared. But a sensitivity is computed per element and keyed
    by the element's Pyomo name, and a reader who knows a name from the printed
    table may well ask for ``sol['V[2]']``; both resolve here to a scalar
    :class:`Entry` cut from the array.
    """

    def __missing__(self, name):
        split = _split_element(name)
        if split is None or not dict.__contains__(self, split[0]):
            raise KeyError(name)
        base = dict.__getitem__(self, split[0])
        try:
            return base.element(split[1])
        except (IndexError, KeyError) as exc:
            raise KeyError(name) from exc

    def __contains__(self, name):
        if dict.__contains__(self, name):
            return True
        try:
            self[name]
        except KeyError:
            return False
        return True

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default


class Solution:
    """The result of a solve: values, units, descriptions, sensitivities.

    Indexing gives a value, so ``sol['Wing_AR']`` reads as one would expect,
    and ``print(sol)`` gives the table. ``variables`` and ``constants`` hold
    :class:`Entry` objects when the units and description are wanted.
    """

    def __init__(self, objective=None, objective_units=None, variables=None,
                 constants=None, sensitivities=None, status=None,
                 solver=None, structure=None, groups=None, ambiguous=None,
                 holographic=None, holographic_total=0, report=None,
                 messages=None):
        self.objective = objective
        self.objective_units = objective_units
        self.variables = EntryMap(variables or {})
        self.constants = EntryMap(constants or {})
        self.sensitivities = dict(sensitivities) if sensitivities else None
        #: Constants whose sensitivity the problem does not determine, because
        #: the active set is degenerate. Hidden from the table by default: the
        #: value returned for one of these is a property of which dual vector
        #: was recovered, not of the design.
        self.ambiguous = set(ambiguous or ())
        #: Holographic constraints found ACTIVE at this solution. Non-empty
        #: means the answer is on a limit that was declared never to bind.
        self.holographic = list(holographic or [])
        #: How many were DECLARED, so the report can say "2 of 3" rather than
        #: "2 of 2" and understate how much of the model was being watched.
        self.holographic_total = holographic_total or len(self.holographic)
        self.status = status
        self.solver = solver
        self.structure = structure
        #: How the solve went: detected/prescribed structure, route, status.
        #: A dict stashed by solve() on the model (``_solve_report``); None
        #: when the model was solved some other way.
        self.report = dict(report) if report else None
        #: Code-tagged messages ([LC-Wxxx] ...) the solve captured instead of
        #: printing. Rendered in the summary's Post Solve Report.
        self.messages = list(messages or [])
        #: ``[(flat_prefix, dotted_path)]``, longest first, for display only.
        self.groups = sorted(groups or [], key=lambda p: -len(p[0]))

    # -- access ------------------------------------------------------------
    def __getitem__(self, name):
        if name in self.variables:
            return self.variables[name].value
        if name in self.constants:
            return self.constants[name].value
        raise KeyError(f'{name!r} is not a variable or constant of this '
                       'solution')

    def __contains__(self, name):
        return name in self.variables or name in self.constants

    def __len__(self):
        return len(self.variables)

    def get(self, name, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def to_dict(self):
        """``{name: value}`` for every variable and constant."""
        out = {k: e.value for k, e in self.variables.items()}
        out.update({k: e.value for k, e in self.constants.items()})
        return out

    # -- presentation ------------------------------------------------------
    def display_name(self, name):
        """``wing_box_t_cap`` shown as ``wing.box.t_cap``.

        Groups namespace by flat prefix so that nothing downstream has to know
        about them, but a reader wants the hierarchy back. The dotted path is
        carried from the group rather than derived by swapping underscores for
        dots, which would turn a group named ``landing_gear`` into
        ``landing.gear``.
        """
        for prefix, path in self.groups:
            # An empty prefix is a group that namespaces nothing -- a builder
            # shared between a standalone model and a larger one that mounts it
            # under a prefix. Every name starts with '', so matching on it
            # would file the entire model under that group.
            if prefix and name.startswith(prefix):
                return path + '.' + name[len(prefix):]
        return name

    def _order(self, names):
        """Ungrouped first, then grouped, each alphabetically.

        A model's own quantities are the ones its author is looking for, and
        they get buried when a hundred namespaced ones sort in among them.

        Within each of the two, `_alphabetical` orders the way a reader would:
        case folded and vector indices numeric. The raw name rides along last
        so that names differing only in case still have a settled order.
        """
        return sorted(names, key=lambda n: (self.display_name(n) != n,
                                            _alphabetical(self.display_name(n)),
                                            self.display_name(n)))

    @staticmethod
    def _rows(entries):
        """``{row_name: Entry}`` with every array broken out per element.

        The table prints ``V[0]``, ``V[1]``, ``V[2]`` -- a row that is a whole
        vector cannot be read -- but the entries themselves stay stacked.
        """
        out = {}
        for n, e in entries.items():
            if e.is_array:
                for ix in np.ndindex(*e.shape):
                    el = e.element(ix)
                    out[el.name] = el
            else:
                out[n] = e
        return out

    def _table(self, entries, ndecimal):
        if not entries:
            return []
        entries = self._rows(entries)
        names = self._order(entries)
        shown = [self.display_name(n) for n in names]
        vals = [_fmt(entries[n].value, ndecimal) for n in names]
        uts = [_units(entries[n].units) for n in names]
        des = [entries[n].description for n in names]
        w = (max(len(s) for s in shown), max(len(s) for s in vals),
             max(len(s) for s in uts), max((len(s) for s in des), default=0))
        return ['   ' + n.ljust(w[0]) + '  :  ' + v.rjust(w[1]) + '   '
                + u.center(w[2]) + ('   ' + d.ljust(w[3]) if d else '')
                for n, v, u, d in zip(shown, vals, uts, des)]

    _STRUCTURE_NAMES = {
        'linear_program': 'linear program (LP)',
        'quadratic_program': 'quadratic program (QP)',
        'geometric_program': 'geometric program (GP)',
        'signomial_program_sia': 'signomial program (SP)',
        'signomial_program_pccp': 'signomial program (SP)',
        'nonlinear_program': 'general nonlinear program (NLP)',
    }
    _SOLVER_NAMES = {
        'pyomo': 'IPOPT (executable, AMPL interface)',
        'cyipopt': 'IPOPT (in-process, cyipopt)',
        'cvxopt': 'cvxopt',
    }

    def _report_lines(self):
        """How the problem was classified and solved, stated accurately.

        'auto-detected' is only claimed when the router actually chose;
        a prescribed solver or backend is reported as prescribed.
        """
        r = self.report or {}
        structure = self._STRUCTURE_NAMES.get(r.get('structure'),
                                              r.get('structure') or
                                              'unknown structure')
        solver = self._SOLVER_NAMES.get(r.get('solver'),
                                        r.get('solver') or 'unknown solver')
        requested = r.get('requested_solver')
        n_bb = r.get('greybox') or 0
        bb = (f', with {n_bb} black-box constraint'
              + ('s' if n_bb != 1 else '') if n_bb else '')

        lines = []
        if requested in (None, 'auto'):
            lines.append(f'   Problem auto-detected as a {structure}{bb}')
        else:
            lines.append(f"   Solver prescribed (solver={requested!r}), "
                         f'bypassing auto-detection; run as a '
                         f'{structure}{bb}')
        solved = f'   Solved with {solver}'
        if (r.get('convex_backend') not in (None, 'ipopt')
                and r.get('structure') != 'nonlinear_program'
                and requested in (None, 'auto')):
            solved += f" (convex backend prescribed: {r['convex_backend']})"
        if r.get('gp_form'):
            solved += f" [gp form: {r['gp_form']}]"
        lines.append(solved)
        # Status is reported in the Post Solve Report at the end of the
        # summary, next to the captured messages, not here.
        return lines

    def summary(self, ndecimal=2, sensitivity_tol=1e-8, top=None,
                show_ambiguous=False):
        """The table, as a string.

        ``top`` keeps only the ``n`` largest sensitivities by magnitude, and
        ``sensitivity_tol`` drops everything below a threshold. A model with
        two hundred constants prints two hundred rows otherwise, and the ones
        worth reading are the handful at the top.

        ``show_ambiguous`` includes the sensitivities the problem does not
        determine, marked with ``?``. They are hidden by default because they
        look exactly like answers.
        """
        L = ['']
        if self.report:
            L += ['Report', '------'] + self._report_lines() + ['']
        if self.objective is not None:
            L += ['Objective', '---------',
                  '   ' + _fmt(self.objective, ndecimal) + ' '
                  + _units(self.objective_units).strip('[]'), '']
        if self.variables:
            L += ['Variables', '---------'] + self._table(self.variables,
                                                          ndecimal) + ['']
        if self.constants:
            L += ['Constants', '---------'] + self._table(self.constants,
                                                          ndecimal) + ['']

        if self.holographic:
            from lcsolver.postsolve.holographic import format_holographic
            L += [format_holographic(self.holographic,
                                     total=self.holographic_total), '']

        L += ['Sensitivities', '-------------']
        if not self.constants:
            L += ['   No constants in the formulation', '']
        elif self.sensitivities is None:
            L += ['   Not computed. Call f.sensitivities(), or '
                  'f.solution_with_sensitivities().', '']
        else:
            hidden_ambiguous = 0
            items = []
            for n, v in self.sensitivities.items():
                if v != v or abs(v) < sensitivity_tol:
                    continue
                if n in self.ambiguous and not show_ambiguous:
                    hidden_ambiguous += 1
                    continue
                items.append((n, v))
            if not items:
                L += [f'   all below {sensitivity_tol:g}', '']
            else:
                # RANKED GLOBALLY BY MAGNITUDE, and deliberately not split
                # ungrouped-first the way the tables above are. There the
                # reader is looking up a name they already know, and a model's
                # own quantities get buried among a hundred namespaced ones.
                # Here the ranking IS the content: the question a sensitivity
                # table answers is "what is this design most sensitive to",
                # and filing every sub-model's constants below every top-level
                # one answers a different question -- it puts a 0.001 in the
                # assembly above a 1.5 in a block, so the largest number in
                # the model lands twenty rows down.
                #
                # Ties break alphabetically so that constants of genuinely
                # equal sensitivity -- a pair that always appears as a product
                # -- keep a settled order across runs instead of falling back
                # on dict insertion.
                items.sort(key=lambda kv: (-abs(kv[1]),
                                           _alphabetical(self.display_name(kv[0])),
                                           self.display_name(kv[0])))
                shown = items if top is None else items[:top]
                wn = max(len(self.display_name(n)) for n, _ in shown)
                for n, v in shown:
                    bar = ('+' if v > 0 else '-') * min(int(abs(v) * 20) + 1, 24)
                    mark = ' ?' if n in self.ambiguous else ''
                    L.append('   ' + self.display_name(n).ljust(wn) + '  :  '
                             + f'{v:+.4f}'.rjust(9) + '   ' + bar + mark)
                if top is not None and len(items) > top:
                    L.append(f'   ({len(items) - top} smaller not shown; '
                             f'pass top=None for all)')
                omitted = (len(self.sensitivities) - len(items)
                           - hidden_ambiguous)
                if omitted:
                    L.append(f'   ({omitted} below {sensitivity_tol:g} omitted)')
                if hidden_ambiguous:
                    L.append(f'   ({hidden_ambiguous} not determined by the '
                             f'problem -- degenerate active set -- hidden; '
                             f'pass show_ambiguous=True)')
                L.append('')
        if self.status or self.solver or self.structure:
            bits = [b for b in (self.structure, self.solver, self.status) if b]
            L.append('(' + '; '.join(str(b) for b in bits) + ')')
        if (self.report and self.report.get('status')) or self.messages:
            import textwrap
            L += ['', 'Post Solve Report', '-----------------']
            if self.report and self.report.get('status'):
                L.append(f"   Status: {self.report['status']}")
            for msg in self.messages:
                L += textwrap.wrap(msg, width=76, initial_indent='   ',
                                   subsequent_indent='       ')
            L.append('')
        return '\n'.join(L)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        obj = ('' if self.objective is None
               else f'objective={self.objective:.6g}, ')
        return (f'<Solution: {obj}{len(self.variables)} variables, '
                f'{len(self.constants)} constants'
                + (', sensitivities' if self.sensitivities else '') + '>')

    # -- construction ------------------------------------------------------
    @staticmethod
    def _group_paths(groups):
        """``[(flat_prefix, dotted_path)]`` for every group, recursively."""
        out = []
        for g in (groups or {}).values():
            out.append((g.prefix, g.path))
            out.extend(Solution._group_paths(getattr(g, '_groups', {})))
        return out

    @classmethod
    def from_model(cls, model, sensitivities=None, status=None, solver=None,
                   structure=None, ambiguous=None, holographic=None,
                   report=None):
        """Read the model's current values into a Solution.

        Deliberately reads the model handed to it rather than any structure
        detected from it: a detected structure holds variables belonging to the
        unit-corrected clone, which was never solved, and reading those returns
        the initial guess.
        """
        import pyomo.environ as pyo

        from lcsolver.postsolve.holographic import (
            holographic_total as _holographic_total,
        )

        def scalar(v):
            try:
                return pyo.value(v)
            except Exception:
                return None

        def stacked(v):
            """An indexed component as one array in its declared shape.

            The shape is the one the declaration recorded (``size=[3, 4]``);
            a component that carries none is read as a flat vector of its
            keys in order. An element that cannot be evaluated is NaN, so the
            array keeps its shape and the rest of the elements stay usable.
            """
            shape = getattr(v, '_edi_shape', None)
            keys = list(v.keys())
            if shape is None:
                shape = (len(keys),)
            out = np.full(tuple(shape), np.nan, dtype=float)
            for k in keys:
                ix = k if isinstance(k, tuple) else (k,)
                val = scalar(v[k])
                if val is not None:
                    try:
                        out[ix] = float(val)
                    except (TypeError, ValueError):
                        pass
            return out

        def entries(components):
            out = {}
            for v in components:
                value = stacked(v) if v.is_indexed() else scalar(v)
                out[v.name] = Entry(v.name, value,
                                    getattr(v, 'get_units', lambda: None)(),
                                    getattr(v, 'doc', '') or '')
            return out

        variables = entries(model.get_variables())
        constants = entries(model.get_constants())

        objective, objective_units = None, None
        try:
            objs = list(model.component_data_objects(pyo.Objective,
                                                     active=True))
            if objs:
                objective = float(pyo.value(objs[0]))
                objective_units = pyo.units.get_units(objs[0].expr)
        except Exception:
            pass

        return cls(objective=objective, objective_units=objective_units,
                   variables=variables, constants=constants,
                   sensitivities=sensitivities, status=status, solver=solver,
                   structure=structure, ambiguous=ambiguous,
                   holographic=holographic,
                   holographic_total=_holographic_total(model),
                   report=report or getattr(model, '_solve_report', None),
                   messages=getattr(model, '_solve_messages', None),
                   groups=cls._group_paths(getattr(model, '_groups', {})))
