import numpy as np
import pymatgen.core as mg
import torch
import os


def pymatgen_comp(comp_list):
    iterable = True
    try:
        some_object_iterator = iter(comp_list)
    except TypeError as te:
        iterable = False
    if iterable:
        return [mg.Composition(x) for x in comp_list]
    else:
        return mg.Composition(comp_list)

class data_generator_vec(object):
    def __init__(self, comps, el_list = []):

        #with open(csv_file, 'r') as fid:
            #l = fid.readlines()
        #data = [x.strip().split(',')[1] for x in l]
        #data.remove('Composition')

        #remove single elements from dataset, want only HEAs. Also keep unqiue compositions
        comps = pymatgen_comp(comps)
        if len(el_list) == 0:
          all_eles = []
          for c in comps:
            all_eles += list(c.get_el_amt_dict().keys())
          eles = np.array(sorted(list(set(all_eles))))
        else:
          eles = np.array(el_list)
          
        self.elements = eles
        self.size = len(eles)
        self.length = len(comps)

        all_vecs = np.zeros([len(comps), len(self.elements)])
        for i, c in enumerate(comps):
            for k, v in c.get_el_amt_dict().items():
                j = np.argwhere(eles == k)
                all_vecs[i, j] = v
        all_vecs = all_vecs / np.sum(all_vecs, axis=1).reshape(-1, 1)
        self.real_data = np.array(all_vecs, dtype=np.float32)

    def sample(self, N):
        idx = np.random.choice(np.arange(self.length), N, replace=False)
        data = self.real_data[idx]

        return np.array(data, dtype=np.float32),idx
    
    def elements(self):
      return eles

def check_cuda():
  if torch.cuda.is_available():
    cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
  else:
    cuda = False
  return cuda


def get_comp(vec, elem_list, thresh=0.02):
    vec[vec<thresh] = 0
    vec /= vec.sum()
    comp = ''
    for i, x in enumerate(vec):
        if x > 0:
            comp += elem_list[i] + '{:.2f} '.format(x)
    return mg.Composition(comp)



def get_number_of_components(comp_list):
  count_list = []
  for c in comp_list:
    if not type(c) == mg.Composition:
      c = mg.Composition(c)
      count_list.append(len(list(c.get_el_amt_dict().keys())))
  return count_list