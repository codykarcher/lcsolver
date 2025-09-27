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
    matches = {}
    rskips  = []
    newBases = []
    newExps  = []
    for iii in range(0,len(lbases)):
        for jjj in range(0,len(rbases)):
            if lbases[iii] is rbases[jjj]:
                matches[iii] = jjj
                rskips.append(jjj)
    lkeys = list(matches.keys())
    for iii in range(0,len(lbases)):
        if iii in lkeys:
            newBases.append(lbases[iii])
            newExps.append(lexps[iii] + rexps[matches[iii]])
        else:
            newBases.append(lbases[iii])
            newExps.append(lexps[iii])
    for jjj in range(0,len(rbases)):
        if jjj not in rskips:
            newBases.append(rbases[jjj])
            newExps.append(rexps[jjj])

    mon['monomial']['bases'] = newBases
    mon['monomial']['exponents'] = newExps
    mon.propagate()
    return mon

def signomial_multiplication(lhe,rhe):
    rhe_new = StructureDictionary()
    rhe_new['signomial']['status'] = 'yes'
    rhe_new['signomial']['leadingCoefficients'] = []
    rhe_new['signomial']['bases'] = []
    rhe_new['signomial']['exponents'] = []

    for i in range(0,len(lhe['signomial']['leadingCoefficients'])):
        for j in range(0,len(rhe['signomial']['leadingCoefficients'])):
            # Create proper monomial dictionaries
            lhm = {'monomial': {'leadingConstant': lhe['signomial']['leadingCoefficients'][i], 
                               'bases': lhe['signomial']['bases'][i], 
                               'exponents': lhe['signomial']['exponents'][i]}}
            rhm = {'monomial': {'leadingConstant': rhe['signomial']['leadingCoefficients'][j], 
                               'bases': rhe['signomial']['bases'][j], 
                               'exponents': rhe['signomial']['exponents'][j]}}
            
            mon = monomial_multiplication(lhm, rhm)

            rhe_new['signomial']['leadingCoefficients'].append(mon['monomial']["leadingConstant"])
            rhe_new['signomial']['bases'].append(mon['monomial']['bases'])
            rhe_new['signomial']['exponents'].append(mon['monomial']['exponents'])

    elementDict = StructureDictionary()
    elementDict['signomial']['status'] = 'yes'
    elementDict['signomial']['leadingCoefficients'] = rhe_new['signomial']['leadingCoefficients']
    elementDict['signomial']['bases'] = rhe_new['signomial']['bases']
    elementDict['signomial']['exponents'] = rhe_new['signomial']['exponents']
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

def signomial_power_evaluation(sig,expVal):
    if sig['signomial']['status'] != 'yes':
        raise ValueError("Signomial data is not flagged as 'yes'")
    if len(sig['signomial']['leadingCoefficients']) != len(sig['signomial']['bases']) or len(sig['signomial']['leadingCoefficients']) != len(sig['signomial']['exponents']) :
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

    # Manual copy instead of deepcopy
    sig_lhe = StructureDictionary()
    sig_lhe['signomial']['status'] = 'yes'
    sig_lhe['signomial']['leadingCoefficients'] = sig['signomial']['leadingCoefficients'].copy()
    sig_lhe['signomial']['bases'] = [base.copy() for base in sig['signomial']['bases']]
    sig_lhe['signomial']['exponents'] = [exp.copy() for exp in sig['signomial']['exponents']]
    sig_lhe.propagate()
    
    sig_rhe = StructureDictionary()
    sig_rhe['signomial']['status'] = 'yes'
    sig_rhe['signomial']['leadingCoefficients'] = sig['signomial']['leadingCoefficients'].copy()
    sig_rhe['signomial']['bases'] = [base.copy() for base in sig['signomial']['bases']]
    sig_rhe['signomial']['exponents'] = [exp.copy() for exp in sig['signomial']['exponents']]
    sig_rhe.propagate()

    for ii in range(0,int(expVal)-1):
        sig_rhe_new = signomial_multiplication(sig_lhe,sig_rhe)
        sig_rhe = sig_rhe_new  ##TODO:  need to make this consistent for powers greater than 2 where the new RHE will be a full object ## I dont remember what this means
        sig_rhe.propagate()

    return sig_rhe

def processMonomial(ix,coeff,bases,exponents,N_vars_unwrapped,variableMap):
    gpRow = [0.0]*(N_vars_unwrapped+2)
    gpRow[0] = ix
    gpRow[1] = coeff
    for i in range(0,len(bases)):
        bs = bases[i]
        if isinstance(bs,VarData):
            vr_ix = variableMap[bs]
            gpRow[2+vr_ix] = exponents[i]
        elif isinstance(bs,ParamData):
            vl = bs.value
            expt = exponents[i]
            gpRow[1] *= vl**expt
        elif isinstance(bs, (int, float)):
            # Handle numeric values directly
            vl = float(bs)
            expt = exponents[i]
            gpRow[1] *= vl**expt
        else:
            raise ValueError(f'Unexpected type in the base: {type(bs)}')
    return gpRow