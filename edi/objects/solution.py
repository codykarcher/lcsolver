#  ___________________________________________________________________________
#
#  EDI: The Engineering Design Interface
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""What a solve produced, held on its own and printed on request.

Pyomo reloads a solution onto the model, so ``pyo.value(m.x)`` answers after a
solve. That is convenient and EDI keeps doing it, but it leaves the result with
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

__all__ = ["Solution"]


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
    """One named quantity: its value, units and description."""

    __slots__ = ('name', 'value', 'units', 'description')

    def __init__(self, name, value, units=None, description=''):
        self.name = name
        self.value = value
        self.units = units
        self.description = description or ''

    def __repr__(self):
        return f'<{self.name} = {self.value!r} {_units(self.units)}>'


class Solution:
    """The result of a solve: values, units, descriptions, sensitivities.

    Indexing gives a value, so ``sol['Wing_AR']`` reads as one would expect,
    and ``print(sol)`` gives the table. ``variables`` and ``constants`` hold
    :class:`Entry` objects when the units and description are wanted.
    """

    def __init__(self, objective=None, objective_units=None, variables=None,
                 constants=None, sensitivities=None, status=None,
                 solver=None, structure=None, groups=None, ambiguous=None,
                 holographic=None):
        self.objective = objective
        self.objective_units = objective_units
        self.variables = dict(variables or {})
        self.constants = dict(constants or {})
        self.sensitivities = dict(sensitivities) if sensitivities else None
        #: Constants whose sensitivity the problem does not determine, because
        #: the active set is degenerate. Hidden from the table by default: the
        #: value returned for one of these is a property of which dual vector
        #: was recovered, not of the design.
        self.ambiguous = set(ambiguous or ())
        #: Holographic constraints found ACTIVE at this solution. Non-empty
        #: means the answer is on a limit that was declared never to bind.
        self.holographic = list(holographic or [])
        self.status = status
        self.solver = solver
        self.structure = structure
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
        """
        return sorted(names, key=lambda n: (self.display_name(n) != n,
                                            self.display_name(n)))

    def _table(self, entries, ndecimal):
        if not entries:
            return []
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
            from edi.solvers.holographic import format_holographic
            L += [format_holographic(self.holographic), '']

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
                # `top` means the n largest in the model, so select on
                # magnitude alone first. Sorting for display first and
                # truncating after would quietly return the n largest
                # *ungrouped* ones, which is a different question.
                items.sort(key=lambda kv: -abs(kv[1]))
                shown = items if top is None else items[:top]
                # Ungrouped first as elsewhere, then by magnitude within each
                # half, so the table reads the way the ones above it do.
                shown.sort(key=lambda kv: (self.display_name(kv[0]) != kv[0],
                                           -abs(kv[1])))
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
                   structure=None, ambiguous=None, holographic=None):
        """Read the model's current values into a Solution.

        Deliberately reads the model handed to it rather than any structure
        detected from it: a detected structure holds variables belonging to the
        unit-corrected clone, which was never solved, and reading those returns
        the initial guess.
        """
        import pyomo.environ as pyo

        def expand(components):
            """One entry per element, so a vector prints as its members.

            `get_variables` hands back components, and an indexed one is not a
            number: asking Pyomo to evaluate it raises, and Pyomo logs a page
            of ERROR lines on the way out. An indexed quantity is also the one
            a reader most wants broken out -- `V[0]`, `V[1]`, `V[2]` rather
            than a single row that cannot be printed at all.
            """
            for v in components:
                if v.is_indexed():
                    for ix in v:
                        yield v[ix]
                else:
                    yield v

        def entries(components):
            out = {}
            for v in expand(components):
                try:
                    value = pyo.value(v)
                except Exception:
                    value = None
                # The description is declared on the component, so an
                # element of a vector has to ask its parent for it.
                doc = getattr(v, 'doc', '') or ''
                if not doc:
                    parent = v.parent_component()
                    doc = (getattr(parent, 'doc', '') or '') if parent is not v else ''
                out[v.name] = Entry(v.name, value,
                                    getattr(v, 'get_units', lambda: None)(),
                                    doc)
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
                   groups=cls._group_paths(getattr(model, '_groups', {})))
