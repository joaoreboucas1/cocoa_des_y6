from cobaya.likelihoods.des_y6._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_des_y6_interface as ci
import numpy as np

class combo_xi_gg(_cosmolike_prototype_base):
  def initialize(self):
    super(combo_xi_gg,self).initialize(probe="xi_gg")