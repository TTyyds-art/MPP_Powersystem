import warnings

warnings.filterwarnings("ignore")

import sys, os

path_current = '/home/huzuntao/PycharmProjects/MPP_Powersystem/'
path_ = os.getcwd()
if path_current not in sys.path:
    sys.path.insert(1, '/home/huzuntao/PycharmProjects/MPP_Powersystem/')
elif path_ not in sys.path:
    sys.path.insert(1, path_)

from pypower_.pipsopf_solver import pipsopf_solver_gpu
import pandapower as pp


from pypower_pych_gpu.pipsopf_solver import pipsopf_solver_torch

net = pp.create_empty_network()

# create buses
bus1 = pp.create_bus(net, vn_kv=110.)
bus2 = pp.create_bus(net, vn_kv=110.)
bus3 = pp.create_bus(net, vn_kv=110.)
bus4 = pp.create_bus(net, vn_kv=110.)
bus5 = pp.create_bus(net, vn_kv=110.)
bus6 = pp.create_bus(net, vn_kv=110.)

# create 110 kV lines
pp.create_line(net, bus4, bus5, length_km=90., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus3, bus4, length_km=90., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus2, bus3, length_km=90., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus1, bus2, length_km=70., std_type='149-AL1/24-ST1A 110.0')
pp.create_line(net, bus6, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')

# create loads
p_load_1 = 10
p_load_2 = 30
pp.create_load(net, bus2, p_mw=p_load_1, controllable=False)
pp.create_load(net, bus4, p_mw=p_load_2/2, controllable=False)
pp.create_load(net, bus5, p_mw=p_load_2/2, controllable=False)
pp.create_load(net, bus6, p_mw=p_load_2/2, controllable=False)
# create generators
eg = pp.create_ext_grid(net, bus1, min_p_mw=0, max_p_mw=1000, vm_pu=1.05)
g0 = pp.create_gen(net, bus3, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.00, controllable=True)

costeg = pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=20)
costgen1 = pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=10)
costgen2 = pp.create_poly_cost(net, 1, 'gen', cp1_eur_per_mw=10)

net.bus["min_vm_pu"] = 0.96
net.bus["max_vm_pu"] = 1.04
net.line["max_loading_percent"] = 100

om, ppopt, raw = pp.runopp(net, delta=1e-16, RETURN_RAW_DER=1)
pipsopf_solver_torch(om, ppopt)