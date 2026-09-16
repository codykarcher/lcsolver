#  ___________________________________________________________________________
#
#  LCsolver: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""What a solve produced, held on its own and printed on request.

Pyomo reloads the solution onto the model, but the result then has nowhere to
live: you must know which variables to ask about and which model object to
ask -- the wrong one silently returns the initial guess, because
unit_corrector cloned the model before detection.

A :class:`Solution` is the answer as an object: objective, every variable and
constant with units and description, sensitivities when computed, printed as
a table. ``f.solution`` builds one from the model's current values. Layout
follows corsair's ``OptimizationOutput.result``: objective, variables,
constants, sensitivities, column widths computed so the colons line up.
"""
from __future__ import annotations

import re

import numpy as np

__all__ = ["Solution"]

_DIGITS = re.compile(r'(\d+)')
_ELEMENT = re.compile(r'^(.*)\[([0-9, ]+)\]$')


def split_element(name):
    """``'V[2]'`` -> ``('V', (2,))``; None when the name carries no index.

    Reverses Pyomo's element naming so ``sol['V[2]']`` keeps working after
    the vector is stacked into one array-valued entry.
    """
    m = _ELEMENT.match(name)
    if not m:
        return None
    return m.group(1), tuple(int(p) for p in m.group(2).split(','))


def element_name(name, index):
    """The name Pyomo gives an element: ``V[2]``, ``x[0,1]``."""
    return name + '[' + ','.join(str(i) for i in index) + ']'


def alphabetical_key(name):
    """Sort key: case folded, digit runs compared as numbers.

    A plain string sort files ``Re_station`` before ``area_disk`` (ASCII
    case) and puts ``[10]`` between ``[1]`` and ``[2]``. Parts are
    kind-tagged so numbers never compare against text; the caller appends
    the raw name to break ties like ``Re`` vs ``re``.
    """
    return tuple((1, int(p)) if p.isdigit() else (0, p.lower())
                 for p in _DIGITS.split(name))


def format_number(value, ndecimal):
    """A number as corsair printed it: integers bare, everything else fixed."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:
        return 'nan'
    if v.is_integer() and abs(v) < 1e15:
        return str(int(v)) + ' ' * (1 + ndecimal)
    # scientific below the last printable decimal -- 0.0035 at ndecimal=2
    # used to print as '0.00', which reads as an exact zero at a bound
    if v != 0 and (abs(v) >= 10 ** 6 or abs(v) < 10 ** -ndecimal):
        return f'%.{ndecimal}e' % v
    return f'%.{ndecimal}f' % v


def units_label(u):
    s = str(u) if u is not None else ''
    return '[-]' if s in ('dimensionless', 'None', '') else f'[{s}]'


class Entry:
    """One named quantity: its value, units and description.

    An array variable is ONE entry holding a numpy array in its declared
    shape -- how a caller wants it back (to save, slice, or hand to the next
    model as a guess). Printing per element is the table's job, not the data's.
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
        return Entry(element_name(self.name, index),
                     self.value[tuple(index)].item(), self.units,
                     self.description)

    def __repr__(self):
        return f'<{self.name} = {self.value!r} {units_label(self.units)}>'


class EntryMap(dict):
    """``{name: Entry}`` that also answers for the elements of an array.

    Iterating gives the stacked names (``V``, not ``V[0]``...), so each
    quantity appears once. But sensitivities are keyed per element and a
    reader may ask for ``sol['V[2]']``; both resolve here to a scalar
    :class:`Entry` cut from the array.
    """

    def __missing__(self, name):
        split = split_element(name)
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
        # Constants whose sensitivity the problem does not determine
        # (degenerate active set); hidden from the table by default
        self.ambiguous = set(ambiguous or ())
        # Holographic constraints found ACTIVE at this solution: the answer
        # sits on a limit declared never to bind
        self.holographic = list(holographic or [])
        # how many were DECLARED, so the report can say "2 of 3"
        self.holographic_total = holographic_total or len(self.holographic)
        self.status = status
        self.solver = solver
        self.structure = structure
        # How the solve went (dict stashed by solve() as _solve_report);
        # None when the model was solved some other way
        self.report = dict(report) if report else None
        # code-tagged [LC-Wxxx] messages captured instead of printed;
        # rendered in the summary's Post Solve Report
        self.messages = list(messages or [])
        # [(flat_prefix, dotted_path)], longest first, for display only
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

        The dotted path is carried from the group rather than derived by
        swapping underscores for dots, which would turn ``landing_gear``
        into ``landing.gear``.
        """
        for prefix, path in self.groups:
            # An empty prefix namespaces nothing; every name starts with '',
            # so matching on it would file the entire model under that group.
            if prefix and name.startswith(prefix):
                return path + '.' + name[len(prefix):]
        return name

    def ordered_names(self, names):
        """Ungrouped first, then grouped, each alphabetically.

        A model's own quantities get buried when a hundred namespaced ones
        sort in among them. The raw name rides along last so names differing
        only in case still have a settled order.
        """
        return sorted(names, key=lambda n: (self.display_name(n) != n,
                                            alphabetical_key(self.display_name(n)),
                                            self.display_name(n)))

    @staticmethod
    def rows_per_element(entries):
        """``{row_name: Entry}`` with every array broken out per element
        for printing; the entries themselves stay stacked."""
        out = {}
        for n, e in entries.items():
            if e.is_array:
                for ix in np.ndindex(*e.shape):
                    el = e.element(ix)
                    out[el.name] = el
            else:
                out[n] = e
        return out

    def table_lines(self, entries, ndecimal):
        if not entries:
            return []
        entries = self.rows_per_element(entries)
        names = self.ordered_names(entries)
        shown = [self.display_name(n) for n in names]
        vals = [format_number(entries[n].value, ndecimal) for n in names]
        uts = [units_label(entries[n].units) for n in names]
        des = [entries[n].description for n in names]
        w = (max(len(s) for s in shown), max(len(s) for s in vals),
             max(len(s) for s in uts), max((len(s) for s in des), default=0))
        return ['   ' + n.ljust(w[0]) + '  :  ' + v.rjust(w[1]) + '   '
                + u.center(w[2]) + ('   ' + d.ljust(w[3]) if d else '')
                for n, v, u, d in zip(shown, vals, uts, des)]

    STRUCTURE_NAMES = {
        'linear_program': 'linear program (LP)',
        'quadratic_program': 'quadratic program (QP)',
        'geometric_program': 'geometric program (GP)',
        'signomial_program_sia': 'signomial program (SP)',
        'signomial_program_pccp': 'signomial program (SP)',
        'nonlinear_program': 'general nonlinear program (NLP)',
    }
    SOLVER_NAMES = {
        'pyomo': 'IPOPT (executable, AMPL interface)',
        'cyipopt': 'IPOPT (in-process, cyipopt)',
        'cvxopt': 'cvxopt',
    }

    def report_lines(self):
        """How the problem was classified and solved, stated accurately.

        'auto-detected' is only claimed when the router actually chose;
        a prescribed solver or backend is reported as prescribed.
        """
        r = self.report or {}
        structure = self.STRUCTURE_NAMES.get(r.get('structure'),
                                              r.get('structure') or
                                              'unknown structure')
        solver = self.SOLVER_NAMES.get(r.get('solver'),
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
        if r.get('linear_solver'):
            solved += f" [linear solver: {r['linear_solver']}]"
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

    def summary(self, ndecimal=4, sensitivity_tol=1e-8, top=None,
                show_ambiguous=False):
        """The table, as a string.

        ``top`` keeps the n largest sensitivities, ``sensitivity_tol`` drops
        the rest -- two hundred constants otherwise print two hundred rows.
        ``show_ambiguous`` includes the ones the problem does not determine,
        marked ``?``; hidden by default because they look exactly like answers.
        """
        L = ['']
        if self.report:
            L += ['Report', '------'] + self.report_lines() + ['']
        if self.objective is not None:
            L += ['Objective', '---------',
                  '   ' + format_number(self.objective, ndecimal) + ' '
                  + units_label(self.objective_units).strip('[]'), '']
        if self.variables:
            L += ['Variables', '---------'] + self.table_lines(self.variables,
                                                          ndecimal) + ['']
        if self.constants:
            L += ['Constants', '---------'] + self.table_lines(self.constants,
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
                # RANKED GLOBALLY BY MAGNITUDE, deliberately not
                # ungrouped-first like the tables above: here the ranking IS
                # the content, and filing sub-model constants below top-level
                # ones would put a 0.001 in the assembly above a 1.5 in a
                # block. Ties break alphabetically so equal pairs keep a
                # settled order across runs.
                items.sort(key=lambda kv: (-abs(kv[1]),
                                           alphabetical_key(self.display_name(kv[0])),
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

        Reads the model handed to it, not a detected structure: detected
        variables belong to the never-solved unit-corrected clone and read
        as the initial guess.
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

            Shape from the declaration (``size=[3, 4]``); without one, a flat
            vector of keys in order. Unevaluable elements become NaN so the
            rest stay usable.
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
