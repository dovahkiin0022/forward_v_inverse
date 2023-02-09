from pymongo import MongoClient
import json
import numpy as np
import rpy2.robjects as robjects

cred_file = 'credentials.json'
cred = json.load(open(cred_file))

client_string = 'mongodb+srv://' + cred['name'] + ':' + cred[
    'dbKey'] + '@testcluster.g3kud.mongodb.net/ULTREA_materials?retryWrites=true&w=majority'
database_name = 'ULTERA'
elemental_database_name = 'ELEMENTAL'

# Elemental properties
client = MongoClient(client_string)
elemental_database = client[elemental_database_name]
elemental_self = elemental_database['self']



def structLC(dLC, compD, structure):
    # Replacements ordered by highest priority
    if structure == 'BCC':
        replacements = ['FCC', 'HCP']
    elif structure == 'FCC':
        replacements = ['HCP', 'BCC']
    elif structure == 'HCP':
        replacements = ['FCC', 'BCC']
    else:
        print('Not-supported structure name. Use BCC, FCC, or HCP.')
        return

    # To be dictionary of selected properties
    selectedDicts = {}

    # For every element in the composition dictionary
    for element in compD:

        # If structure is present, take that property dictionary
        if structure in dLC[element]:
            selectedDicts.update({element: dLC[element]['common']})
            selectedDicts.update({element: dLC[element][structure]})
            selectedDicts[element].update({'replacementLevel': 0})

        # If not, look for a replacement
        else:

            # Iterate through replacements
            for s in replacements:
                if s in dLC[element]:
                    selectedDicts.update({element: dLC[element]['common']})
                    selectedDicts.update({element: dLC[element][s]})
                    selectedDicts[element].update({'replacementLevel': 1})
                    continue
    return selectedDicts

def findCommon(selectedD):
    # 2D list with properties avaialble for each element
    keyMatrix = []
    for element in selectedD:
        keyMatrix.append(list(selectedD[element].keys()))

    # Start with the first element
    common = set(keyMatrix[0])
    # If there is more elements, remove properties that are not available for the next ones
    for element in keyMatrix[1:]:
        common.intersection_update(element)

    return common

def get_LC_values(compDict, elemental_self = elemental_self):
    dataLC = {}
    eleSum = sum(compDict.values())
    for element in compDict:
        compDict[element] = compDict[element]/eleSum
    jpcmDataMissing = False
    for element in compDict:
        elementalFind = elemental_self.find_one({'symbol': element}, {'common': 1, 'BCC': 1, 'FCC': 1, 'HCP': 1})
        if 'BCC' in elementalFind or 'FCC' in elementalFind or 'HCP' in elementalFind:
            dataLC.update({element: elementalFind})
        else:
            jpcmDataMissing = True

    if jpcmDataMissing:
    #If JPCM is missing for some element, raise error
        raise FileNotFoundError

    entryUpdate = {}
    entryUpdateInternal = {}

    for s in ['BCC', 'FCC', 'HCP']:

        structDataLC = structLC(dataLC, compDict, s)
        commonProperties = findCommon(structDataLC)

        outLC = {}
        for prop in commonProperties:
            propAvg = sum([structDataLC[element][prop]*compDict[element] for element in compDict])
            outLC.update({prop: propAvg})
        entryUpdateInternal.update({'LC_' + s: outLC})
        entryUpdate.update({'properties.LC_' + s: outLC})

    return entryUpdate

def calculate_D_param(comp):
    compDict = comp.get_el_amt_dict()
    entryUpdate = get_LC_values(compDict)
    return entryUpdate['properties.LC_BCC']['Surf'] / entryUpdate['properties.LC_BCC']['USFE']

def calculate_b_g_ratio(comp):
    compDict = comp.get_el_amt_dict()
    entryUpdate = get_LC_values(compDict)
    return entryUpdate['properties.LC_BCC']['DFTBh']/entryUpdate['properties.LC_BCC']['DFTGh']

def calculate_FT_Rice92(comp):
    compDict = comp.get_el_amt_dict()
    entryUpdate = get_LC_values(compDict)
    Shear_Modulus = entryUpdate['properties.LC_BCC']['DFTGh']
    Unstable_Stacking_Fault_Energy = entryUpdate['properties.LC_BCC']['USFE']
    Poisson_Ratio = entryUpdate['properties.LC_BCC']['DFTpoisson'] 
    return np.sqrt(2*Shear_Modulus*Unstable_Stacking_Fault_Energy/(1-Poisson_Ratio))

def calculate_price(comp):
    compDict = comp.get_el_amt_dict()
    entryUpdate = get_LC_values(compDict)
    return entryUpdate['properties.LC_BCC']['price [$/kg]']

def calculate_density(comp):
    compDict = comp.get_el_amt_dict()
    entryUpdate = get_LC_values(compDict)
    return entryUpdate['properties.LC_BCC']['density0K [g/cm^3]']



def calculate_entropy_mixing(comp):
  delta = 0
  for v in comp.get_el_amt_dict().values():
    if v>0:
      delta += v*np.log(v)
  return delta

def calculate_D_param_from_eu(entryUpdate):
    return entryUpdate['properties.LC_BCC']['Surf'] / entryUpdate['properties.LC_BCC']['USFE']

def calculate_b_g_ratio_from_eu(entryUpdate):
    return entryUpdate['properties.LC_BCC']['DFTBh']/entryUpdate['properties.LC_BCC']['DFTGh']

def calculate_FT_Rice92_from_eu(entryUpdate):
    Shear_Modulus = entryUpdate['properties.LC_BCC']['DFTGh']
    Unstable_Stacking_Fault_Energy = entryUpdate['properties.LC_BCC']['USFE']
    Poisson_Ratio = entryUpdate['properties.LC_BCC']['DFTpoisson'] 
    return np.sqrt(2*Shear_Modulus*Unstable_Stacking_Fault_Energy/(1-Poisson_Ratio))

def calculate_price_from_eu(entryUpdate):
    return entryUpdate['properties.LC_BCC']['price [$/kg]']

def calculate_density_from_eu(entryUpdate):
    return entryUpdate['properties.LC_BCC']['density0K [g/cm^3]']


predict_file = 'hu_d_param_files/predict.R'
predet_order = ['Ti','Zr','Hf','V','Nb','Ta','Mo','W','Re','Ru']
def calculate_hu_d_param(comp, predet_order = predet_order):
    """Calculate d parameter from Hu's model from a pymatgen composition"""
    zero_vec = np.zeros(len(predet_order))
    el_dict = comp.get_el_amt_dict()
    for el in predet_order:
        ind = np.argwhere(np.array(predet_order) == el)
        if el in el_dict.keys():
            zero_vec[ind] = el_dict[el]
    atom_frac_str = ','.join([str(a) for a in zero_vec])
    with open(predict_file,'r') as fid:
        file_contents = fid.read()
    test = file_contents.format(test_comp = f'c({atom_frac_str})')
    temp_save_loc = 'hu_d_param_files/predict_temp.R'
    with open(temp_save_loc,'w') as fid:
        fid.write(test)
        
    params = robjects.r.source(temp_save_loc, encoding="utf-8")
    temp_dict = dict(zip(params.names, map(list,list(params))))
    return(temp_dict['value'][0])




