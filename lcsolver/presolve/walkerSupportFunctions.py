#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2023
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base.var import ScalarVar, VarData, IndexedVar
from pyomo.core.base.param import ParamData, ScalarParam, IndexedParam
import copy


def propagate(dct):
    if dct['_changed']:
        active_list = [dct[key]['status'] == 'yes' for key in [ "constant", "monomial", "signomial", "signomial_fraction" ] ]
        if active_list not in [[ True,  True,  True,  True],
                                [False,  True,  True,  True],
                                [False, False,  True,  True],
                                [False, False, False,  True],
                                [ True, False, False, False],
                                [False,  True, False, False],
                                [False, False,  True, False],
                                ]:
            raise ValueError("Inconsistent structure dictionary: %s"%(dct))

        if   active_list in [ [ True,  True,  True,  True], [ True, False, False, False] ]:
            # a constant that needs to be propagated
            assert(dct['constant']['status'] == 'yes')
            assert(dct['constant']['value'] is not None)

            dct['monomial']['status'] = 'yes'
            dct['monomial']['leadingConstant'] = dct['constant']['value']
            dct['monomial']['bases'] = []
            dct['monomial']['exponents'] = []

            dct['signomial']['status'] = 'yes'
            dct['signomial']['leadingCoefficients'] = [dct['constant']['value']]
            dct['signomial']['bases'] = [[]]
            dct['signomial']['exponents'] = [[]]

            dct['signomial_fraction']['status'] = 'yes'
            dct['signomial_fraction']['numerator']['leadingCoefficients'] = [dct['constant']['value']]
            dct['signomial_fraction']['numerator']['bases'] = [[]]
            dct['signomial_fraction']['numerator']['exponents'] = [[]]
            dct['signomial_fraction']['denominator']['leadingCoefficients'] = [1.0]
            dct['signomial_fraction']['denominator']['bases'] = [[]]
            dct['signomial_fraction']['denominator']['exponents'] = [[]]
        
        elif active_list in [ [False,  True,  True,  True], [False,  True, False, False] ]:
            # a monomial that needs to be propagated
            assert(dct['monomial']['status'] == 'yes')
            assert(dct['monomial']['leadingConstant'] is not None)

            dct['signomial']['status'] = 'yes'
            dct['signomial']['leadingCoefficients'] = [dct['monomial']['leadingConstant']]
            dct['signomial']['bases'] = [dct['monomial']['bases']]
            dct['signomial']['exponents'] = [dct['monomial']['exponents']]

            dct['signomial_fraction']['status'] = 'yes'
            dct['signomial_fraction']['numerator']['leadingCoefficients'] = [dct['monomial']['leadingConstant']]
            dct['signomial_fraction']['numerator']['bases'] = [dct['monomial']['bases']]
            dct['signomial_fraction']['numerator']['exponents'] = [dct['monomial']['exponents']]
            dct['signomial_fraction']['denominator']['leadingCoefficients'] = [1.0]
            dct['signomial_fraction']['denominator']['bases'] = [[]]
            dct['signomial_fraction']['denominator']['exponents'] = [[]]

        elif active_list in [ [False, False,  True,  True], [False, False,  True, False] ]:
            # a signomial that needs to be propagated
            assert(dct['signomial']['status'] == 'yes')
            assert(dct['signomial']['leadingCoefficients'] is not None)

            dct['signomial_fraction']['status'] = 'yes'
            dct['signomial_fraction']['numerator']['leadingCoefficients'] = dct['signomial']['leadingCoefficients']
            dct['signomial_fraction']['numerator']['bases'] = dct['signomial']['bases']
            dct['signomial_fraction']['numerator']['exponents'] = dct['signomial']['exponents']
            dct['signomial_fraction']['denominator']['leadingCoefficients'] = [1.0]
            dct['signomial_fraction']['denominator']['bases'] = [[]]
            dct['signomial_fraction']['denominator']['exponents'] = [[]]

        elif active_list == [False, False, False,  True]:
            # Clean up the other structures
            dct['constant']  = {"status":"no", "value":None }
            dct['monomial']  = {"status":"no", "leadingConstant":None, "bases":[], "exponents":[]}
            dct['signomial'] = {"status":"no", "leadingCoefficients":None, "bases":[], "exponents":[]}

        else:
            raise ValueError("Inconsistent structure dictionary: %s"%(dct))

    dct['_changed'] = False

class SubStructureDictionary(object):
    def __init__(self,item_type, parent_address):
        super(SubStructureDictionary, self).__init__()
        self.type = item_type
        self.parent_copy = parent_address

    def __repr__(self):
        self.propagate()
        if self.parent_copy['_changed']:
            self.propagate()
            self.parent_copy['_changed'] = False
        return "%s"%(self.__dict__)
    
    def __str__(self):
        self.propagate()
        if self.parent_copy['_changed']:
            self.propagate()
            self.parent_copy['_changed'] = False
        return "%s"%(self.__dict__)

    def __getitem__(self, item):
        if self.parent_copy['_changed']:
            self.propagate()
            self.parent_copy['_changed'] = False
        return self.parent_copy[self.type].get(item, None)

    def __setitem__(self, key, value):
        self.parent_copy['_changed'] = True
        self.parent_copy[self.type][key] = value

    def propagate(self):
        if self.parent_copy['_changed']:  # Only propagate if actually changed
            propagate(self.parent_copy)


class StructureDictionary(object):
    def __init__(self, *args, **kwargs):
        super(StructureDictionary, self).__init__(*args, **kwargs)

        self._parent_dictionary = {
                "constant"           : {"status":"no", "value":None },
                "monomial"           : {"status":"no", "leadingConstant":None, "bases":[], "exponents":[]},
                "signomial"          : {"status":"no", "leadingCoefficients":None, "bases":[], "exponents":[]},
                "signomial_fraction" : {"status":"no",  "numerator":{"leadingCoefficients":None, "bases":[[]], "exponents":[[]]},
                                                        "denominator":{"leadingCoefficients":None, "bases":[[]], "exponents":[[]]} },
                "_changed"        : False,
            }

        self["constant"]           = SubStructureDictionary('constant', self._parent_dictionary)
        self["monomial"]           = SubStructureDictionary('monomial', self._parent_dictionary)
        self["signomial"]          = SubStructureDictionary('signomial', self._parent_dictionary)
        self["signomial_fraction"] = SubStructureDictionary('signomial_fraction', self._parent_dictionary)

    def __repr__(self):
        self.propagate()
        return "%s"%(self._parent_dictionary)
    
    def __str__(self):
        self.propagate()
        return "%s"%(self._parent_dictionary)

    def __getattr__(self, item):
        # self.propagate()
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def __getitem__(self, item):
        # self.propagate()
        ret_item = self.__dict__.get(item, None)
        return ret_item
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def propagate(self):
        if self._parent_dictionary['_changed']:  # Only propagate if actually changed
            propagate(self._parent_dictionary)

def no_structure_dict():
    return StructureDictionary()

def monomial_multiplication(lhm,rhm):
    mon = StructureDictionary()

    mon['monomial']['status'] = 'yes'
    mon['monomial']['leadingConstant'] = lhm['monomial']["leadingConstant"] * rhm['monomial']["leadingConstant"]
    
    lbases = lhm['monomial']['bases']
    lexps  = lhm['monomial']['exponents']
    rbases = rhm['monomial']['bases']
    rexps  = rhm['monomial']['exponents']
    
    # Use dictionary for O(1) lookup instead of nested loops
    base_to_exp = {}
    
    # Add left side bases and exponents
    for i, base in enumerate(lbases):
        base_to_exp[id(base)] = (base, lexps[i])
    
    # Process right side bases, combining with left if they match
    for i, base in enumerate(rbases):
        base_id = id(base)
        if base_id in base_to_exp:
            # Combine exponents for matching base
            existing_base, existing_exp = base_to_exp[base_id]
            base_to_exp[base_id] = (existing_base, existing_exp + rexps[i])
        else:
            # Add new base
            base_to_exp[base_id] = (base, rexps[i])
    
    # Extract results
    newBases = []
    newExps = []
    for base, exp in base_to_exp.values():
        newBases.append(base)
        newExps.append(exp)

    mon['monomial']['bases'] = newBases
    mon['monomial']['exponents'] = newExps
    mon.propagate()
    return mon

def signomial_multiplication(lhe,rhe):
    lhe_coeffs = lhe['signomial']['leadingCoefficients']
    lhe_bases = lhe['signomial']['bases']
    lhe_exps = lhe['signomial']['exponents']
    
    rhe_coeffs = rhe['signomial']['leadingCoefficients']
    rhe_bases = rhe['signomial']['bases']
    rhe_exps = rhe['signomial']['exponents']
    
    # Pre-allocate result lists with known size
    num_terms = len(lhe_coeffs) * len(rhe_coeffs)
    result_coeffs = []
    result_bases = []
    result_exps = []
    
    for i in range(len(lhe_coeffs)):
        lh_coeff = lhe_coeffs[i]
        lh_bases = lhe_bases[i]
        lh_exps = lhe_exps[i]
        
        for j in range(len(rhe_coeffs)):
            rh_coeff = rhe_coeffs[j]
            rh_bases = rhe_bases[j]
            rh_exps = rhe_exps[j]
            
            # Multiply coefficients
            new_coeff = lh_coeff * rh_coeff
            
            # Combine bases and exponents using the optimized approach
            base_to_exp = {}
            
            # Add left bases
            for k, base in enumerate(lh_bases):
                base_to_exp[id(base)] = (base, lh_exps[k])
            
            # Add/combine right bases
            for k, base in enumerate(rh_bases):
                base_id = id(base)
                if base_id in base_to_exp:
                    existing_base, existing_exp = base_to_exp[base_id]
                    base_to_exp[base_id] = (existing_base, existing_exp + rh_exps[k])
                else:
                    base_to_exp[base_id] = (base, rh_exps[k])
            
            # Extract combined results
            new_bases = []
            new_exps = []
            for base, exp in base_to_exp.values():
                new_bases.append(base)
                new_exps.append(exp)
            
            result_coeffs.append(new_coeff)
            result_bases.append(new_bases)
            result_exps.append(new_exps)

    elementDict = StructureDictionary()
    elementDict['signomial']['status'] = 'yes'
    elementDict['signomial']['leadingCoefficients'] = result_coeffs
    elementDict['signomial']['bases'] = result_bases
    elementDict['signomial']['exponents'] = result_exps
    elementDict.propagate()

    return elementDict


def signomial_fraction_multiplication(lhf,rhf):
    # print(lhf)
    # print(rhf)
    left_numerator   = StructureDictionary()
    left_numerator['signomial']['status']              = 'yes'
    left_numerator['signomial']['leadingCoefficients'] = lhf['signomial_fraction']['numerator']['leadingCoefficients']
    left_numerator['signomial']['bases']               = lhf['signomial_fraction']['numerator']['bases']
    left_numerator['signomial']['exponents']           = lhf['signomial_fraction']['numerator']['exponents']
    left_numerator.propagate()
    # print('left_numerator')

    left_denominator = StructureDictionary()
    left_denominator['signomial']['status']              = 'yes'
    left_denominator['signomial']['leadingCoefficients'] = lhf['signomial_fraction']['denominator']['leadingCoefficients']
    left_denominator['signomial']['bases']               = lhf['signomial_fraction']['denominator']['bases']
    left_denominator['signomial']['exponents']           = lhf['signomial_fraction']['denominator']['exponents']
    left_denominator.propagate()
    # print('left_denominator')

    right_numerator   = StructureDictionary()
    right_numerator['signomial']['status']              = 'yes'
    right_numerator['signomial']['leadingCoefficients'] = rhf['signomial_fraction']['numerator']['leadingCoefficients']
    right_numerator['signomial']['bases']               = rhf['signomial_fraction']['numerator']['bases']
    right_numerator['signomial']['exponents']           = rhf['signomial_fraction']['numerator']['exponents']
    right_numerator.propagate()
    # print('right_numerator')

    right_denominator = StructureDictionary()
    right_denominator['signomial']['status']              = 'yes'
    right_denominator['signomial']['leadingCoefficients'] = rhf['signomial_fraction']['denominator']['leadingCoefficients']
    right_denominator['signomial']['bases']               = rhf['signomial_fraction']['denominator']['bases']
    right_denominator['signomial']['exponents']           = rhf['signomial_fraction']['denominator']['exponents']
    right_denominator.propagate()
    # print('right_denominator')

    new_num = signomial_multiplication(left_numerator,right_numerator)
    new_dem = signomial_multiplication(left_denominator,right_denominator)

    new_num.propagate()
    new_dem.propagate()
    # print('new')

    elementDict = StructureDictionary()
    elementDict['signomial_fraction']['status']                             = 'yes'
    elementDict['signomial_fraction']['numerator']['leadingCoefficients']   = new_num['signomial']['leadingCoefficients']
    elementDict['signomial_fraction']['numerator']['bases']                 = new_num['signomial']['bases']
    elementDict['signomial_fraction']['numerator']['exponents']             = new_num['signomial']['exponents']
    elementDict['signomial_fraction']['denominator']['leadingCoefficients'] = new_dem['signomial']['leadingCoefficients']
    elementDict['signomial_fraction']['denominator']['bases']               = new_dem['signomial']['bases']
    elementDict['signomial_fraction']['denominator']['exponents']           = new_dem['signomial']['exponents']
    # print(elementDict)
    return elementDict

def signomial_power_evaluation(sig, expVal):
    if sig['signomial']['status'] != 'yes':
        raise ValueError("Signomial data is not flagged as 'yes'")
    
    sig_coeffs = sig['signomial']['leadingCoefficients']
    sig_bases = sig['signomial']['bases']
    sig_exps = sig['signomial']['exponents']
    
    if len(sig_coeffs) != len(sig_bases) or len(sig_coeffs) != len(sig_exps):
        raise ValueError("Signomial data has inconsistent lengths")

    if expVal == 0:
        sd = StructureDictionary()
        sd['constant']['status'] = 'yes'
        sd['constant']['value'] = 1.0
        sd.propagate()
        return sd

    if expVal < 0:
        sd = StructureDictionary()
        return sd

    if expVal == 1:
        # No need to multiply, just copy the signomial
        result = StructureDictionary()
        result['signomial']['status'] = 'yes'
        result['signomial']['leadingCoefficients'] = sig_coeffs[:]  # shallow copy
        result['signomial']['bases'] = [base[:] for base in sig_bases]  # copy lists
        result['signomial']['exponents'] = [exp[:] for exp in sig_exps]  # copy lists
        result.propagate()
        return result

    # Use exponentiation by squaring for better performance
    exp_int = int(expVal)
    result = StructureDictionary()
    result['signomial']['status'] = 'yes'
    result['signomial']['leadingCoefficients'] = sig_coeffs[:]
    result['signomial']['bases'] = [base[:] for base in sig_bases]
    result['signomial']['exponents'] = [exp[:] for exp in sig_exps]
    result.propagate()
    
    # Use the base signomial for multiplication
    base_sig = StructureDictionary()
    base_sig['signomial']['status'] = 'yes'
    base_sig['signomial']['leadingCoefficients'] = sig_coeffs[:]
    base_sig['signomial']['bases'] = [base[:] for base in sig_bases]
    base_sig['signomial']['exponents'] = [exp[:] for exp in sig_exps]
    base_sig.propagate()
    
    # Iterative multiplication (can be further optimized with binary exponentiation)
    for ii in range(exp_int - 1):
        result = signomial_multiplication(result, base_sig)
        result.propagate()

    return result

def processMonomial(ix, coeff, bases, exponents, N_vars_unwrapped, variableMap):
    gpRow = [0.0] * (N_vars_unwrapped + 2)
    gpRow[0] = ix
    gpRow[1] = coeff
    
    # Pre-compute type checks to avoid repeated isinstance calls
    for i in range(len(bases)):
        bs = bases[i]
        expt = exponents[i]
        
        if isinstance(bs, VarData):
            vr_ix = variableMap[bs]
            gpRow[2 + vr_ix] = expt
        elif isinstance(bs, ParamData):
            vl = bs.value
            gpRow[1] *= vl ** expt
        elif isinstance(bs, (int, float)):
            # Handle numeric values directly
            vl = float(bs)
            gpRow[1] *= vl ** expt
        else:
            raise ValueError(f'Unexpected type in the base: {type(bs)}')
    
    return gpRow