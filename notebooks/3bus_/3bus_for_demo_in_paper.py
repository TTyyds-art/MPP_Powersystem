import os.path
import pandapower.networks as pn
import pandapower as pp
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
# import sympy as sym
# import gurobipy as gp
# from gurobipy import GRB
from numpy import flatnonzero as find, ones, zeros, Inf, pi, exp, conj, r_, arange, array
from pypower_.makeYbus import makeYbus
from pypower_.idx_brch import F_BUS, T_BUS, RATE_A, PF, QF, PT, QT, MU_SF, MU_ST
from pypower_.idx_gen import GEN_BUS, PG, QG
from pypower_.idx_cost import MODEL, PW_LINEAR, NCOST, POLYNOMIAL
from pypower_.totcost import totcost
from pypower_.makeSbus import makeSbus
from pypower_.idx_bus import BUS_TYPE, REF, VM, VA, MU_VMAX, MU_VMIN, LAM_P, LAM_Q
import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix as sparse
from pypower_.idx_bus import PD, QD
from pypower_.idx_gen import GEN_BUS, PG, QG, GEN_STATUS

import pandapower as pp
net = pp.create_empty_network()

# create buses
# bus1 = pp.create_bus(net, vn_kv=110.)
# bus2 = pp.create_bus(net, vn_kv=110.)
# bus3 = pp.create_bus(net, vn_kv=110.)
# # bus4 = pp.create_bus(net, vn_kv=110.)
# # bus5 = pp.create_bus(net, vn_kv=110.)
#
# # create 110 kV lines
# pp.create_line(net, bus2, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')
# # pp.create_line(net, bus3, bus4, length_km=50., std_type="149-AL1/24-ST1A 110.0")
# # pp.create_line(net, bus4, bus2, length_km=40., std_type="149-AL1/24-ST1A 110.0")
# pp.create_line(net, bus1, bus2, length_km=70., std_type='149-AL1/24-ST1A 110.0')
#
# # create loads
# pp.create_load(net, bus1, p_mw=30., controllable=False)
# # pp.create_load(net, bus3, p_mw=70., controllable=False)
# # pp.create_load(net, bus4, p_mw=25., controllable=False)
#
# # create generators
# eg = pp.create_ext_grid(net, bus3, min_p_mw=0, max_p_mw=1000, vm_pu=1.05)
# g0 = pp.create_gen(net, bus2, p_mw=80, min_p_mw=0, max_p_mw=50, vm_pu=1.00, controllable=True)
# # g1 = pp.create_gen(net, bus4, p_mw=100, min_p_mw=0, max_p_mw=100, vm_pu=1.00, controllable=True)
#
#
# costeg = pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=20)
# costgen1 = pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=10)
# costgen2 = pp.create_poly_cost(net, 1, 'gen', cp1_eur_per_mw=10)

bus1 = pp.create_bus(net, vn_kv=110.)
bus2 = pp.create_bus(net, vn_kv=110.)
bus3 = pp.create_bus(net, vn_kv=110.)
# bus4 = pp.create_bus(net, vn_kv=110.)
# bus5 = pp.create_bus(net, vn_kv=110.)

# create 110 kV lines
# bus2,3 改成了 bus1,3; 又改了回去
pp.create_line(net, bus2, bus3, length_km=90., std_type='149-AL1/24-ST1A 110.0')
# pp.create_line(net, bus3, bus4, length_km=50., std_type="149-AL1/24-ST1A 110.0")
# pp.create_line(net, bus4, bus2, length_km=40., std_type="149-AL1/24-ST1A 110.0")
pp.create_line(net, bus1, bus2, length_km=70., std_type='149-AL1/24-ST1A 110.0')

# create loads
pp.create_load(net, bus2, p_mw=50., controllable=False)
# pp.create_load(net, bus3, p_mw=70., controllable=False)
# pp.create_load(net, bus4, p_mw=25., controllable=False)

# create generators
eg = pp.create_ext_grid(net, bus3, min_p_mw=0, max_p_mw=50, vm_pu=1.05)
g0 = pp.create_gen(net, bus1, p_mw=50, min_p_mw=0, max_p_mw=50, vm_pu=1.00, controllable=True)
# g1 = pp.create_gen(net, bus4, p_mw=50, min_p_mw=0, max_p_mw=50, vm_pu=1.00, controllable=True)


costeg = pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=20)
costgen1 = pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=10)
costgen2 = pp.create_poly_cost(net, 1, 'gen', cp1_eur_per_mw=10)

net.bus["min_vm_pu"] = 0.96
net.bus["max_vm_pu"] = 1.04
net.line["max_loading_percent"] = 100

net.line["max_loading_percent"] = 100
om,ppopt,_ = pp.runopp(net, delta=1e-16)   # solve the OPF problem

## unpack data
ppc = om.get_ppc()
baseMVA, bus, gen, branch, gencost = \
    ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"], ppc["gencost"]
vv, _, _, _ = om.get_idx()

## problem dimensions
nb = bus.shape[0]          ## number of buses
nl = branch.shape[0]       ## number of branches
ny = om.getN('var', 'y')   ## number of piece-wise linear costs, 'y' is a part of p-w linear costs

## bounds on optimization vars
x0, xmin, xmax = om.getv()

## build admittance matrices
Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
# 输出两位小数
np.set_printoptions(precision=2)
print(Ybus.todense())

# print(net.lines)


## Set the lower and upper bound for all variables
ll, uu = xmin.copy(), xmax.copy()
# print(f'll : {ll}; uu : {uu}')
ll[xmin == -Inf] = -1e10   ## replace Inf with numerical proxies
uu[xmax ==  Inf] =  1e10
Va_refs = bus[bus[:, BUS_TYPE]  == REF, VA] * (pi / 180)
# print(f"Va_refs: {Va_refs}")
ll[vv["i1"]["Va"]:vv["iN"]["Va"]] = -np.ones_like(bus[:, VA]) * (pi / 2) # Va lower bound 赋值
uu[vv["i1"]["Va"]:vv["iN"]["Va"]] = np.ones_like(bus[:, VA]) * (pi / 2) # Va upper bound 赋值
## deal with the Va_refs
ll[vv["i1"]["Va"]:vv["iN"]["Va"]][bus[:, BUS_TYPE]  == REF] = Va_refs  # Va_refs lower bound 赋值
uu[vv["i1"]["Va"]:vv["iN"]["Va"]][bus[:, BUS_TYPE]  == REF] = Va_refs # Va_refs upper bound 赋值
## deal with reactive power, whose ll and uu is 1e9
ll[ll<-1e4] = -100
uu[uu>1e4] = 100
v_max = uu[vv["i1"]["Vm"]:vv["iN"]["Vm"]][-1]
print(f'll"{ll};\n uu:{uu}')

x0[vv["i1"]["Vm"]:vv["iN"]["Vm"]] = bus[:, VM] # 赋值
Va_refs = bus[bus[:, BUS_TYPE]  == REF, VA] * (pi / 180)
x0[vv["i1"]["Va"]:vv["iN"]["Va"]] = bus[:, VA] * (pi / 180) # 赋值
## 赋值
x0[vv["i1"]["Pg"]:vv["iN"]["Pg"]] = gen[:,PG]
x0[vv["i1"]["Qg"]:vv["iN"]["Qg"]] = gen[:,QG]

ipol = find(gencost[:, MODEL] == POLYNOMIAL)   ## poly MW and MVAr costs
First_Or_Con = 4
print(f"{gencost[ipol, First_Or_Con], gencost[:, First_Or_Con]}")

## grab Pg & Qg
Pg = x0[vv["i1"]["Pg"]:vv["iN"]["Pg"]]  ## active generation in p.u.
Qg = x0[vv["i1"]["Qg"]:vv["iN"]["Qg"]]  ## reactive generation in p.u.

## put Pg & Qg back in gen
gen[:, PG] = Pg * baseMVA  ## active generation in MW
gen[:, QG] = Qg * baseMVA  ## reactive generation in MVAr

on = find(gen[:, GEN_STATUS] > 0)      ## which generators are on?
gbus = gen[on, GEN_BUS]                   ## what buses are they at?

## form net complex bus power injection vector
nb = bus.shape[0]
ngon = on.shape[0]
## connection matrix, element i, j is 1 if gen on(j) at bus i is ON
Cg = sparse((ones(ngon), (gbus, range(ngon))), (nb, ngon))

## power injected by gens plus power injected by loads converted to p.u.
Sbus = ( Cg * (gen[on, PG] + 1j * gen[on, QG]) - (bus[:, PD] + 1j * bus[:, QD]) ) / baseMVA

# Parameters
# obtain the complex load
Pd = bus[:, PD]
Qd = bus[:, QD]
flow_max = [8018.67,8018.67]
# slack_v = net.ext_grid['vm_pu']
Pg_lower, Pg_upper = ll[vv["i1"]["Pg"]:vv["iN"]["Pg"]], uu[vv["i1"]["Pg"]:vv["iN"]["Pg"]]*baseMVA
Qg_lower, Qg_upper = ll[vv["i1"]["Qg"]:vv["iN"]["Qg"]], uu[vv["i1"]["Qg"]:vv["iN"]["Qg"]]*baseMVA

def optimize_OPF(p_load):
    Pd = p_load
    Qd = 0
    # Pd = bus[:, PD]
    Qd = bus[:, QD]

    lb_update = -1e4

    # obtain the range of voltage; the range is from the setting
    V_range = 0.1
    V_ll,V_uu = 1.0-V_range, 1.0+V_range

    # create the model; set the model type: NonConvex.
    model = gp.Model("PowerGeneration");model.setParam('NonConvex', 2)
    model.setParam("LogToConsole", 0)

    # creat variables for the model; the numbers in the following codes should be replaced by parameters
    V_re = model.addMVar(3, lb=[-1.05,-1.05,-1.05], ub=[1.05,1.05,1.05], vtype=GRB.CONTINUOUS, name='V real')
    V_im = model.addMVar(3, lb=[-1.05,-1.05,-1.05],ub=[1.05,1.05,1.05], vtype=GRB.CONTINUOUS, name='V imag')
    # Pg = model.addMVar(2, lb=0, ub=[105,105], vtype=GRB.CONTINUOUS, name='Pg')
    # Qg = model.addMVar(2, lb=[0,0], ub=[105,105], vtype=GRB.CONTINUOUS, name='Qg')
    # temp1 = model.addMVar(3, lb=lb_update, vtype=GRB.CONTINUOUS, name='Temp 1')
    # temp2 = model.addMVar(3, lb=lb_update, vtype=GRB.CONTINUOUS, name='Temp 2')
    # Pg_net = model.addMVar(3, lb=lb_update, vtype=GRB.CONTINUOUS, name='Pg net')  # the number of buses, too
    # Qg_net = model.addMVar(3, lb=lb_update, vtype=GRB.CONTINUOUS, name='Qg net')

    Pg = model.addMVar(ngon, lb=Pg_lower, ub=Pg_upper, vtype=GRB.CONTINUOUS, name='Pg')
    Qg = model.addMVar(ngon, lb=Qg_lower, ub=Qg_upper, vtype=GRB.CONTINUOUS, name='Qg')
    temp1 = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='Temp 1')
    temp2 = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='Temp 2')
    Pg_net = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='Pg net')  # the number of buses, too
    Qg_net = model.addMVar(nb, lb=lb_update, vtype=GRB.CONTINUOUS, name='Qg net')
    Line_temp_1 = model.addMVar(nl, lb=lb_update, vtype=GRB.CONTINUOUS, name='Line temp 1')    # the shape should be line number
    Line_temp_2 = model.addMVar(nl, lb=lb_update, vtype=GRB.CONTINUOUS, name='Line temp 2')    # the shape should be line number
    Line_temp_3 = model.addMVar(nl, lb=lb_update, vtype=GRB.CONTINUOUS, name='Line temp 3')    # the shape should be line number
    Line_temp_4 = model.addMVar(nl, lb=lb_update, vtype=GRB.CONTINUOUS, name='Line temp 4')    # the shape should be line number


    # the prameters of branches
    G = np.real(Ybus)
    B = np.imag(Ybus)
    # the parameters for currents in branches
    G_yf, B_yf = np.real(Yf), np.imag(Yf)
    G_yt, B_yt = np.real(Yt), np.imag(Yt)

    # Constraints; insert the formulas
    Temp_c1 = model.addConstr(G@V_re - B@V_im == temp1, 'Temp Constr 1')
    Temp_c2 = model.addConstr(G @ V_im + B @ V_re == temp2, 'Temp Constr 2')
    Pg_net_c = model.addConstr(Cg @ Pg - Pd == Pg_net, 'Pg net Constr')
    Qg_net_c = model.addConstr(Cg @ Qg - Qd == Qg_net, 'Qg net Constr')
    # print(f'Pd:{Pd}\n Qd:{Qd}')
    CVL_CPL = model.addConstrs((V_re[i] * temp1[i] *baseMVA+ V_im[i] * temp2[i]*baseMVA == Pg_net[i]  for i in range(nb)), 'Cvl P')  # break into different parts
    CVL_CPL_Q = model.addConstrs((V_im[i] * temp1[i] *baseMVA - V_re[i] * temp2[i]*baseMVA == Qg_net[i] for i in range(nb)), 'Cvl Q')


    V_slack = model.addConstr(V_re[2]==1.05, 'V slack real')
    V_slack_a = model.addConstr(V_im[2]==0, 'V slack imag')

    V_upper = model.addConstrs((V_re[i]**2 + V_im[i]**2 <= V_uu**2 for i in [0,1]), 'V upper bound')  # 对 range 起始值的处理，是处理树
    V_lower = model.addConstrs((V_re[i]**2 + V_im[i]**2 >= V_ll**2 for i in [0,1]), 'V lower bound')
    # Yf * W * Yf; the number is equal to that of the lines/branches
    Line_temp_c_1 = model.addConstr(G_yf@V_re - B_yf@V_im == Line_temp_1, 'Branch Con 1')
    Line_temp_c_2 = model.addConstr(G_yf @ V_im + B_yf @ V_re == Line_temp_2, 'Branch Con 2')
    Line_temp_c_3 = model.addConstr(G_yt@V_re - B_yt@V_im == Line_temp_3, 'Branch Con 3')
    Line_temp_c_4 = model.addConstr(G_yt @ V_im + B_yt @ V_re == Line_temp_4, 'Branch Con 4')
    # 下式左侧是否应该乘上baseMVA
    Line_c_1 = model.addConstrs((Line_temp_1[i]**2 + Line_temp_2[i]**2 <= np.real(flow_max[i]) for i in range(nl)), 'Branch Con 1') # p.u.
    Line_c_2 = model.addConstrs((Line_temp_3[i]**2 + Line_temp_4[i]**2 <= np.real(flow_max[i]) for i in range(nl)), 'Branch Con 2') # p.u.

    model.setObjective(np.array([1, 1.5])@Pg, GRB.MINIMIZE)

    # calculate the optimization
    model.optimize()

    Pg_star, Qg_star = Pg.X, Qg.X
    optimized_obj = np.array([1, 1.5])@Pg_star

    print(f'Pg: {Pg_star}; Qg: {Qg_star}; opt obj: {optimized_obj}')
    print(f'Ve: {V_re.X}; Vim: {V_im.X}; Pd: {Pd}')
    return Pg_star

optimize_OPF(p_load=20)

