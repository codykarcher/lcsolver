#!/usr/bin/env python3
# ===========
# Description
# ===========
# submodel.py -- SubModel, the base class a reusable block of a model is
# written as.
#
# A block is a PACKAGE: its variables, its constants, and the constraints that
# tie them together, declared in one object that the formulation then owns by
# name,
#
#     f.rotor_beam_model = RotorBeamModel(n_stations=8)
#
# after which f.rotor_beam_model.thickness_skin reaches the handle, and the
# flat component is namespaced by the same name (rotor_beam_model_thickness
# _skin), so a deck key, a sensitivity row, and the attribute path all read the
# same.  The attribute the block is attached as IS its name -- there is no
# second place to spell it and therefore no way for the two to drift.
#
# A block declares only what it OWNS.  A quantity two blocks share -- a blade
# count, a root cutout -- is declared once by the assembly and the handle is
# assigned in, so there is one constant and one deck key for it rather than two
# that can drift apart.
#
# Declaring through the model (self.Variable / self.Constant) rather than
# through a bare group does two things beyond namespacing: every declared
# handle is also set as an attribute of the model, so the block's own rows and
# its caller reach quantities by name instead of by dict key; and the model
# keeps its declaration lists, so a block can be asked what it owns.

import types

__all__ = ['SubModel']


class _Inputs:
    """Read-only view of a block's declared inputs, handed out by
    :meth:`SubModel.bind_inputs`.

    Read-only because it is a VIEW, not state: writing through it would change
    nothing the formulation can see, so the write is refused rather than
    silently lost.  A missing name reports what the block actually declares,
    since a mistyped input is otherwise a bare AttributeError at build time.
    """

    __slots__ = ('_d',)

    def __init__(self, mapping):
        object.__setattr__(self, '_d', mapping)

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(
                f'{name!r} is not a declared input of this block; it declares '
                f'{sorted(self._d)}') from None

    def __setattr__(self, name, value):
        raise AttributeError(
            'inputs are read-only: assign to the block attribute instead')

    def __dir__(self):
        return sorted(self._d)

    def __repr__(self):
        return f'<inputs: {", ".join(sorted(self._d))}>'


def _referenced_names(code):
    """Every attribute/global name a code object mentions, nested ones too.

    ``m.area_disk`` and ``self.area_disk`` both put ``area_disk`` in
    ``co_names``, so this sees an input as used either way.
    """
    out = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):    # comprehensions, closures
            out |= _referenced_names(const)
    return out


def _wrap(text, width):
    """Break `text` on spaces at `width`. A status report is read in a
    terminal, so a long note wraps rather than running off the right."""
    out, row = [], ''
    for word in text.split():
        if row and len(row) + 1 + len(word) > width:
            out.append(row)
            row = word
        else:
            row = f'{row} {word}'.strip()
    if row:
        out.append(row)
    return out


class SubModel:
    """One packaged piece of a sizing problem: variables, constants, rows.

    Subclasses declare in :meth:`build` through ``self.Variable`` /
    ``self.Constant`` and post through ``self.ConstraintList``, or through
    ``self.HolographicConstraintList`` for rows that must hold but must not
    bind.

    THE FORMULATION IS NOT PASSED IN.  A block is attached to a formulation by
    assignment, and that assignment is what hands it both the formulation and
    its name::

        f.ferry_model = FerryModel(n_segments=20)

    The constructor takes only the SETTINGS -- the things that change which
    rows exist.

    INPUTS ARE ASSIGNED, NOT PASSED.  A block with thirty inputs makes a
    thirty-argument call that no reader can check against the signature, so
    they arrive one per line afterwards::

        f.ferry_model.area_disk   = area_disk
        f.ferry_model.weight_fuel = weight_fuel
        ...

    Each line names the input on the left and the assembly's quantity on the
    right, so a mismatch is visible where it happens rather than as a position
    in an argument list.

    ALL OF THE INPUTS OR NONE OF THEM.  Passing them to the constructor is
    still allowed -- a four-input block reads fine that way -- but a call that
    names SOME of them raises, listing what is missing.  That is the one
    reading which cannot be right: it looks like a complete construction, and
    it builds nothing.  Settings are not part of this, since they are what the
    constructor is for.

    THE BLOCK BUILDS WHEN ITS LAST INPUT ARRIVES.  :attr:`input_variables` and
    :attr:`input_constants` list what it needs; assigning the final one calls
    :meth:`build`.  Nothing is posted before then, so a block that never
    receives an input posts NO ROWS AT ALL -- a model that silently loses
    constraints and still solves.  That failure does not announce itself, so
    ``solve()`` refuses to run a formulation with an unbuilt block attached;
    see :func:`lcsolver.presolve.reductions.unbuilt_blocks_check`.

    The two input lists are kept apart because they are different promises to
    the assembly.  An input VARIABLE is something the optimizer moves -- the
    block writes rows that shape it and reads a solved value back.  An input
    CONSTANT is something the assembly fixed -- it appears in a deck, it earns
    a sensitivity, and nothing the block does can change it.  A quantity handed
    in under the wrong heading still solves, so :meth:`get_status` prints them
    separately and says what each one is actually wired to.

    :meth:`get_status` is the way to look at a block: what it needs, what is
    connected to each of those, what it declares, and what it hands back.
    """

    #: Names of input DESIGN VARIABLES that must be assigned before building.
    input_variables = ()

    #: Names of input CONSTANTS that must be assigned before building.
    input_constants = ()

    #: Inputs whose kind is not being declared. Prefer the two lists above;
    #: this exists so a block written before they did keeps working.
    inputs = ()

    #: Things this block hands BACK, beyond its declared components: ``{name:
    #: what a caller does with it}``. An expression a block exposes as an
    #: attribute -- Prouty's ``component_weights``, a power sum -- is invisible
    #: otherwise, since it is in no component list and no sensitivity table.
    provides = {}

    #: Names a subclass declares but deliberately never references in build()
    #: -- an input reached through getattr() with a computed name, say.  Listing
    #: one here exempts it from the staleness check below.
    _allow_unused_inputs = ()

    def __init_subclass__(cls, **kwargs):
        """Refuse a declaration no ``build()`` consumes.

        The two directions of drift are not symmetric.  Reading an input that
        was never declared fails loudly on its own -- ``self.foo`` raises the
        first time the block builds.  DECLARING one nothing reads is the silent
        half: the block goes on waiting for an input it does not need and the
        assembly goes on wiring it, and nothing ever complains.  That half is
        caught here, when the class is defined.

        The walk is over the MRO, not just this class: a subclass may inherit
        its tuples from a parent whose ``build()`` is what uses them.
        """
        super().__init_subclass__(**kwargs)
        used = set()
        for klass in cls.__mro__:
            fn = vars(klass).get('build')
            if fn is not None:
                used |= _referenced_names(fn.__code__)
        stale = sorted(set(cls.required_inputs())
                       - used - set(cls._allow_unused_inputs))
        if stale:
            raise TypeError(
                f'{cls.__name__} declares inputs no build() uses: {stale}.  '
                f'Drop them from input_variables/input_constants, use them, '
                f'or list them in _allow_unused_inputs.')

    @classmethod
    def required_inputs(cls):
        """Every input name this block waits for, variables then constants."""
        seen, out = set(), []
        for name in (tuple(cls.input_variables) + tuple(cls.input_constants)
                     + tuple(cls.inputs)):
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def bind_inputs(self):
        """This block's declared inputs, as one read-only namespace.

        Bound as ``m`` by convention, at the top of :meth:`build`::

            m = self.bind_inputs()
            ...
            CT * m.rho * m.area_disk * m.v_tip**2 >= m.k_download * m.weight_gross

        The tuples become the ONLY place an input is named.  Unpacking each one
        into a local rebuilt those tuples as a second, hand-maintained list that
        could quietly fall out of step; reaching them through the namespace
        cannot, and the prefix marks at a glance which symbols in a row arrived
        from the assembly and which the block owns.

        A SNAPSHOT, taken when build() runs.  That is not a constraint in
        practice: the block builds once, when its last input is assigned, and
        the handles it captures are mutated in place rather than rebound.

        Aliasing one back out (``solidity = m.solidity``) to keep a dense row
        readable is fine -- it is checked against the tuples, since dropping the
        name makes ``m.solidity`` raise.  Do it for the few that earn it; alias
        all of them and the second list is back.
        """
        return _Inputs({name: getattr(self, name)
                        for name in self.required_inputs()})

    def __init__(self, formulation=None, name=None, prefix=None, **kwargs):
        # ``Block('name')`` and ``Block(f, 'name')`` both still work: a block
        # written before the assignment syntax names itself, and one attached
        # by assignment takes the attribute as its name.
        if isinstance(formulation, str) and name is None:
            formulation, name = None, formulation
        required = self.required_inputs()
        # Inputs may be passed to the constructor instead of assigned, but it
        # is ALL OF THEM OR NONE.  A partial call is the one reading that
        # cannot be right: it looks like a complete construction and builds
        # nothing, and the block then sits there posting no rows.
        given = {key: kwargs.pop(key) for key in list(kwargs)
                 if key in required}
        settings = kwargs
        if given and len(given) < len(required):
            raise TypeError(self._partial_call_message(given, required))

        object.__setattr__(self, '_pending', set(required))
        # A block handed its formulation up front and declaring no inputs does
        # its work in its own __init__, the older way; it is built by
        # definition and never waits.  Constructed bare, the same block has
        # nothing to wait for either, so ATTACHING is what builds it.
        object.__setattr__(self, '_built',
                           formulation is not None and not required)
        object.__setattr__(self, '_prefix', prefix)
        self.formulation = formulation
        self.name = name
        self.group = None
        self.variables = {}
        self.constants = {}
        self.rows = []
        self.settings = settings
        for key, value in settings.items():
            setattr(self, key, value)
        if formulation is not None:
            self._open_group()
        # Assigned after the group exists, because the last one builds.
        for key, value in given.items():
            setattr(self, key, value)
        if formulation is not None and required and not self._pending \
                and not self._built:
            self._build_now()

    @classmethod
    def _partial_call_message(cls, given, required):
        """What a constructor call that names some inputs but not all says."""
        missing = [name for name in required if name not in given]
        lines = [
            f'{cls.__name__} was given {len(given)} of its {len(required)} '
            f'inputs. A block takes either ALL of its inputs in the '
            f'constructor or NONE of them, assigned one per line afterwards. '
            f'A partial call reads like a complete one, builds nothing, and '
            f'posts no rows.',
            '',
            f'  {len(missing)} input(s) are missing:',
            ]
        for title, names in (('input variables', cls.input_variables),
                             ('input constants', cls.input_constants),
                             ('inputs', cls.inputs)):
            absent = [n for n in names if n in missing]
            if not absent:
                continue
            lines.append(f'    {title}:')
            row = '     '
            for item in absent:
                if len(row) + len(item) + 2 > 74:
                    lines.append(row.rstrip(','))
                    row = '     '
                row += f' {item},'
            lines.append(row.rstrip(','))
        lines += [
            '',
            '  Either add the missing ones to the call, or drop the '
            f'{len(given)} that',
            '  are there and assign every input after attaching the block:',
            '',
            f'      f.<name> = {cls.__name__}(<settings only>)',
            f'      f.<name>.{missing[0]} = ...',
            ]
        return '\n'.join(lines)

    # ---- attachment ----
    def _attach(self, formulation, name):
        """Called by ``Formulation.__setattr__`` when this block is assigned.

        This is where a block constructed bare learns which formulation it
        belongs to and what it is called.  Re-assigning an already-attached
        block (to a second name, or to a second formulation) does nothing: its
        components are already declared, and declaring them again under a new
        namespace would be a copy rather than an alias.
        """
        if self.formulation is not None:
            return
        self.formulation = formulation
        if self.name is None:
            self.name = name
        self._open_group()
        if not self._pending and not self._built:
            # Nothing left to wait for, so attaching IS the last event.
            self._build_now()

    def _open_group(self):
        self.group = self.formulation.group(self.name, prefix=self._prefix)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        pending = self.__dict__.get('_pending')
        if pending and key in pending:
            pending.discard(key)
            if not pending and self.__dict__.get('formulation') is not None:
                self._build_now()

    def _build_now(self):
        object.__setattr__(self, '_built', True)
        self.build()

    def is_built(self):
        """True once :meth:`build` has run and this block's rows are posted."""
        return bool(self._built)

    def pending_inputs(self):
        """The inputs still unassigned, sorted.  Empty once built."""
        return sorted(self._pending)

    def build(self):
        """Declare this block's variables and post its rows.  Subclasses
        implement this; it runs once, when the last input has arrived."""
        raise NotImplementedError(
            f"{type(self).__name__} declares inputs but no build()")

    # ---- declaration, mirrored onto the model as attributes ----
    def Variable(self, name, guess, units, description, **kw):
        v = self.group.Variable(name, guess, units, description, **kw)
        self.variables[name] = v
        setattr(self, name, v)
        return v

    def Constant(self, name, value, units, description, **kw):
        c = self.group.Constant(name, value, units, description, **kw)
        self.constants[name] = c
        setattr(self, name, c)
        return c

    def Constraint(self, expr, holographic=False):
        """Post one row.  `ConstraintList` is the usual form; this is the
        one-at-a-time one underneath, and returns the component's name."""
        if expr is None:
            return None
        self.rows.append(expr)
        return self.group.Constraint(expr, holographic=holographic)

    def ConstraintList(self, rows, holographic=False):
        rows = [r for r in rows if r is not None]
        self.rows.extend(rows)
        self.group.ConstraintList(rows, holographic=holographic)
        return rows

    def HolographicConstraint(self, expr):
        """A row that must hold but must not *bind* -- a fit's validity
        envelope, a modelling limit, a box that keeps the problem well posed.
        Every solve checks it and reports any that came out active.  See
        `Formulation.HolographicConstraint` for why it is worth declaring."""
        return self.Constraint(expr, holographic=True)

    def HolographicConstraintList(self, rows):
        """`ConstraintList`, with every entry declared holographic."""
        return self.ConstraintList(rows, holographic=True)

    # ---- what this block needs, what it owns, what it hands back ----
    def get_status(self):
        """Print what this block needs, what is wired to it, and what it owns.

        Five sections: the input VARIABLES and what each is connected to, the
        input CONSTANTS and the same, the variables and constants this block
        DECLARES, and anything it :attr:`provides` beyond those.  An input not
        yet assigned is called out rather than omitted, because an input that
        never arrives is the one failure this syntax makes possible.

        Prints.  Use :meth:`status_text` for the same report as a string.
        """
        print(self.status_text())

    def status_text(self):
        """:meth:`get_status` as a string."""
        lines = []
        head = f"{type(self).__name__} '{self.name or '(unattached)'}'"
        if self.formulation is None:
            head += '  --  NOT ATTACHED to a formulation'
        elif self.is_built():
            head += f'  --  BUILT, {len(self.rows)} constraint statements'
        else:
            missing = self.pending_inputs()
            head += (f'  --  NOT BUILT, waiting on {len(missing)} input'
                     f'{"" if len(missing) == 1 else "s"}. It has posted '
                     f'nothing.')
        lines += [head, '=' * min(len(head), 78)]

        if self.settings:
            lines.append('')
            lines.append('Settings')
            for key in sorted(self.settings):
                lines.append(f'   {key:<28} = {self.settings[key]!r}')

        lines += self._input_section('Input variables', self.input_variables)
        lines += self._input_section('Input constants', self.input_constants)
        unclassified = [n for n in self.inputs
                        if n not in self.input_variables
                        and n not in self.input_constants]
        lines += self._input_section('Inputs (kind not declared)',
                                     unclassified)

        lines += self._declared_section('Variables declared here',
                                        self.variables)
        lines += self._declared_section('Constants declared here',
                                        self.constants)

        if self.provides:
            lines += ['', 'Also provides']
            for key in sorted(self.provides):
                value = getattr(self, key, None)
                state = (self._describe(value) if value is not None
                         else '(not available until the block builds)')
                lines.append(f'   {key:<28} {state}')
                for chunk in _wrap(str(self.provides[key]), 48):
                    lines.append(f'   {"":<28} {chunk}')
        return '\n'.join(lines)

    def _input_section(self, title, names):
        if not names:
            return []
        connected = [n for n in names if n not in self._pending]
        out = ['', f'{title} ({len(connected)} of {len(names)} connected)']
        for name in names:
            if name in self._pending:
                out.append(f'   {name:<28} <- NOT CONNECTED')
            else:
                out.append(f'   {name:<28} <- '
                           f'{self._describe(getattr(self, name, None))}')
        return out

    def _declared_section(self, title, declared):
        if not declared:
            if self.is_built():
                return []
            return ['', f'{title}: none yet, the block has not built']
        out = ['', f'{title} ({len(declared)})']
        for name, component in declared.items():
            out.append(f'   {name:<28} {self._describe(component, doc=True)}')
        return out

    @staticmethod
    def _describe(thing, doc=False):
        """One line for a handle: what it is, its shape, its units, its doc."""
        import pyomo.environ as pyo

        if thing is None:
            return '(none)'
        bits = []
        name = getattr(thing, 'name', None)
        expression = False
        try:
            expression = (thing.is_expression_type()
                          and not thing.is_variable_type()
                          and not thing.is_parameter_type())
        except Exception:
            pass
        if expression:
            # A Pyomo expression reports its OPERATOR as its name ('sum'),
            # which reads like a component and is not one.
            bits.append('<expression>')
        elif isinstance(name, str):
            bits.append(name)
        elif isinstance(thing, (int, float)):
            bits.append(repr(thing))
        else:
            text = str(thing)
            bits.append(text if len(text) <= 40 else text[:37] + '...')
        try:
            if thing.is_indexed():
                bits.append(f'[{len(thing)}]')
        except Exception:
            shape = getattr(thing, 'shape', None)
            if shape:
                bits.append(f'{tuple(shape)}')
        try:
            unit = pyo.units.get_units(thing)
            if unit is not None and str(unit) != 'dimensionless':
                bits.append(f'[{unit}]')
        except Exception:
            pass
        if doc:
            text = getattr(thing, 'doc', '') or ''
            if not text:
                parent = getattr(thing, 'parent_component', lambda: None)()
                text = (getattr(parent, 'doc', '') or '') if parent is not None else ''
            if text:
                bits.append(f'-- {text}')
        return ' '.join(bits)

    def __repr__(self):
        if not self._built:
            return (f"<{type(self).__name__} {self.name!r}: unbuilt, waiting "
                    f"for {', '.join(self.pending_inputs()) or 'attachment'}>")
        return (f"<{type(self).__name__} {self.name!r}: "
                f"{len(self.variables)} variables, {len(self.constants)} "
                f"constants, {len(self.rows)} rows>")
