import json
from collections import namedtuple
from datetime import timedelta
import threading
import cvxpy as cp
import numpy as np
from itertools import zip_longest
from itertools import product
import math
from cachetools import TTLCache
import re

Shop = namedtuple("Shop", "path buys sells")
Commodity = namedtuple("Commodity", "name price stock refresh")
EPSILON = 0.001


class RoutePlanner:

    def __init__(self, shops):
        self.shops_idx = {s.path: i for i, s in enumerate(shops)}
        self.shops_rev_idx = {i: s.path for i, s in enumerate(shops)}
        self.commodities_idx = {}

        commodity_count = 0
        for s in shops:
            for b in s.buys:
                if b.name not in self.commodities_idx:
                    self.commodities_idx[b.name] = commodity_count
                    commodity_count = commodity_count + 1
            for sl in s.sells:
                if sl.name not in self.commodities_idx:
                    self.commodities_idx[sl.name] = commodity_count
                    commodity_count = commodity_count + 1

        self.commodities_rev_idx = {i: v for v, i in self.commodities_idx.items()}

        self.supply = np.zeros((len(self.commodities_idx), len(self.shops_idx)))
        self.demand = np.zeros((len(self.commodities_idx), len(self.shops_idx)))
        self.buy_price = np.zeros((len(self.commodities_idx), len(self.shops_idx)))
        self.sell_price = np.zeros((len(self.commodities_idx), len(self.shops_idx)))

        for s in shops:
            if s.path not in self.shops_idx:
                continue
            for b in s.buys:
                self.demand[self.commodities_idx[b.name], self.shops_idx[s.path]] = b.stock
                self.sell_price[self.commodities_idx[b.name], self.shops_idx[s.path]] = b.price
            for sl in s.sells:
                self.supply[self.commodities_idx[sl.name], self.shops_idx[s.path]] = sl.stock
                self.buy_price[self.commodities_idx[sl.name], self.shops_idx[s.path]] = sl.price

    def create_weights(self):
        buy_weight = np.divide(1, self.supply, out=np.zeros_like(self.supply), where=self.supply != 0)
        sell_weight = np.divide(1, self.demand, out=np.zeros_like(self.demand), where=self.demand != 0)
        return buy_weight, sell_weight

    def update_supply(self, good, location, amount):
        if good not in self.commodities_idx:
            raise KeyError("%s not found" % good)
        if location not in self.shops_idx:
            raise KeyError("%s not found" % location)
        if amount < 0:
            raise ValueError("amount cannot be negative")
        good_idx = self.commodities_idx[good]
        location_idx = self.shops_idx[location]
        self.supply[good_idx, location_idx] = amount

    def update_demand(self, good, location, amount):
        if good not in self.commodities_idx:
            raise KeyError("%s not found" % good)
        if location not in self.shops_idx:
            raise KeyError("%s not found" % location)
        if amount < 0:
            raise ValueError("amount cannot be negative")
        good_idx = self.commodities_idx[good]
        location_idx = self.shops_idx[location]
        self.demand[good_idx, location_idx] = amount


RoutePath = namedtuple("RoutePath", ["start", "end", "buy", "sell"])
Transaction = namedtuple("Transaction", ["loc_idx", "com_idx", "amount"])

HighLevelPlan = namedtuple("HighLevelPlan", ['cost', 'revenue', 'buy', 'sell'])


def compute_travel_cost(paths, path_idx):
    result = np.zeros((len(paths), len(paths)))
    max_level = -1
    for p in paths:
        p_pars = p.split(">")
        if len(p_pars) >= max_level:
            max_level = len(p_pars)

    for shop_a, shop_b in product(paths, paths):
        ix_a, ix_b = path_idx[shop_a], path_idx[shop_b]
        if shop_a == shop_b:
            result[ix_a, ix_b] = 0
        parts_a = shop_a.split(">")
        parts_b = shop_b.split(">")
        curr_level = max_level
        for a, b in zip_longest(parts_a, parts_b, fillvalue="-1"):
            if a.strip() != b.strip():
                result[ix_a, ix_b] = curr_level
                break
            curr_level = curr_level - 1
    return result


def build_idx(transactions):
    locations = set()
    commodities = set()
    shop_idx = {}
    shop_rev_idx = {}
    com_idx = {}
    com_rev_idx = {}
    for t in transactions:
        locations.add(t.loc_idx)
        commodities.add(t.com_idx)
    for i, p in enumerate(locations):
        shop_idx[p] = i
        shop_rev_idx[i] = p
    for i, p in enumerate(commodities):
        com_idx[p] = i
        com_rev_idx[i] = p

    return shop_idx, shop_rev_idx, com_idx, com_rev_idx


class HighLevelPlanner(RoutePlanner):

    def __init__(self, shops, solver=None, ignore_dpp=None):
        RoutePlanner.__init__(self, shops)
        self._buy_weight, self._sell_weight = self.create_weights()
        self.init_solve = self._formulate_step_one()
        self._trv_c = compute_travel_cost([s.path for s in shops], self.shops_idx)
        self.solver = solver
        self._init_supply = np.array(self.supply)
        self.ignore_dpp = ignore_dpp
        self._init_demand = np.array(self.demand)

    def plan_stage_one(self, cargo, max_percent=0.2, max_commodity=None, blk_locations=(),
                       max_com_loc=None, max_level=2, n_stop=3):
        self._init_basic(self.init_solve, cargo, max_percent=max_percent, max_commodity=max_commodity,
                         blk_locations=blk_locations, max_com_loc=max_com_loc)
        self.init_solve.param_dict["R"].value = self._trv_c
        self.init_solve.param_dict["ML"].value = max_level
        self.init_solve.param_dict["NS"].value = n_stop

        profit = self.init_solve.solve(solver=self.solver, ignore_dpp=self.ignore_dpp)
        if math.isfinite(profit):
            self.init_solve.var_dict["I"].value[np.where(self.init_solve.var_dict["I"].value < EPSILON)] = 0
            self.init_solve.var_dict["L"].value[np.where(self.init_solve.var_dict["L"].value < EPSILON)] = 0
            return profit, self._extract_plan(self.shops_rev_idx, self.commodities_rev_idx,
                                              self.init_solve.var_dict["I"], self.init_solve.var_dict["L"],
                                              self.init_solve.param_dict["S"],
                                              self.init_solve.param_dict["B"])
        return profit, None

    def plan_refinement(self, plan, cargo, max_percent=0.2, max_commodity=None, blk_locations=(),
                        max_com_loc=None):
        shop_idx, shop_rev_idx, com_idx, com_rev_idx = build_idx([t for t in plan.buy] +
                                                                 [t for t in plan.sell])

        if max_commodity is not None:
            max_commodity = {k: v for k, v in max_commodity.items() if k in com_idx}

        blk_locations = [l for l in blk_locations if l in shop_idx]

        if max_com_loc is not None:
            temp = {}
            for k, locs in max_com_loc.items():
                if k not in com_idx:
                    continue
                temp[k] = {}
                for l, amount in locs.items():
                    if l not in shop_idx:
                        continue
                    temp[k][l] = amount

            max_com_loc = temp

        refinement_prob = self._formulate_refinement(n_locs=len(shop_idx), n_coms=len(com_idx))

        shop_selector = [self.shops_idx[shop_rev_idx[i]] for i in range(len(shop_idx))]
        com_selector = [self.commodities_idx[com_rev_idx[i]] for i in range(len(com_idx))]
        self._init_basic(refinement_prob, cargo, max_percent=max_percent, max_commodity=max_commodity,
                         blk_locations=blk_locations,
                         max_com_loc=max_com_loc, com_idx=com_idx, shop_idx=shop_idx, rows=com_selector,
                         cols=shop_selector)
        profit = refinement_prob.solve(solver=self.solver)
        if math.isfinite(profit):
            refinement_prob.var_dict["I"].value[np.where(refinement_prob.var_dict["I"].value < EPSILON)] = 0
            refinement_prob.var_dict["L"].value[np.where(refinement_prob.var_dict["L"].value < EPSILON)] = 0
            return profit, self._extract_plan(shop_rev_idx, com_rev_idx,
                                              refinement_prob.var_dict["I"], refinement_prob.var_dict["L"],
                                              refinement_prob.param_dict["S"],
                                              refinement_prob.param_dict["B"])
        return profit, None

    def _init_basic(self, problem, cargo, max_percent=0.2, max_commodity=None, blk_locations=(),
                    max_com_loc=None, com_idx=None, shop_idx=None, rows=None, cols=None):
        buy_price = self.buy_price
        sell_price = self.sell_price
        demand = self.demand
        supply = self.supply
        buy_weight = self._buy_weight
        sell_weight = self._sell_weight

        if rows is not None:
            buy_price = buy_price[rows, :]
            sell_price = sell_price[rows, :]
            demand = demand[rows, :]
            supply = supply[rows, :]
            buy_weight = buy_weight[rows, :]
            sell_weight = sell_weight[rows, :]

        if cols is not None:
            buy_price = buy_price[:, cols]
            sell_price = sell_price[:, cols]
            demand = demand[:, cols]
            supply = supply[:, cols]
            buy_weight = buy_weight[:, cols]
            sell_weight = sell_weight[:, cols]

        if com_idx is None:
            com_idx = self.commodities_idx
        if shop_idx is None:
            shop_idx = self.shops_idx

        problem.param_dict["B"].value = buy_price
        problem.param_dict["S"].value = sell_price
        problem.param_dict["D"].value = demand
        problem.param_dict["P"].value = supply
        problem.param_dict["Wb"].value = buy_weight
        problem.param_dict["Ws"].value = sell_weight
        problem.param_dict["C"].value = cargo
        problem.param_dict["Q"].value = np.ones((len(com_idx), len(shop_idx))) * max_percent
        if max_commodity is not None:
            for k, percent in max_commodity.items():
                problem.param_dict["Q"].value[com_idx[k], :] = percent
        if max_com_loc is not None:
            for k, locs in max_com_loc.items():
                for l, amount in locs.items():
                    problem.param_dict["Q"].value[com_idx[k], shop_idx[l]] = amount
        for loc in blk_locations:
            loc_idx = shop_idx[loc]
            problem.param_dict["D"].value[:, loc_idx] = 0
            problem.param_dict["P"].value[:, loc_idx] = 0

    def _extract_plan(self, shop_rev_idx, com_rev_idx, I, L, S, B):
        buy_transactions = []
        sell_transactions = []

        for com, loc in zip(*np.nonzero(I.value)):
            buy_transactions.append(Transaction(shop_rev_idx[loc],
                                                com_rev_idx[com],
                                                I.value[com, loc]))
        for com, loc in zip(*np.nonzero(L.value)):
            sell_transactions.append(Transaction(shop_rev_idx[loc],
                                                 com_rev_idx[com],
                                                 L.value[com, loc]))
        revenue = np.sum(np.multiply(L.value, S.value))
        cost = np.sum(np.multiply(I.value, B.value))
        return HighLevelPlan(cost, revenue, buy_transactions, sell_transactions)

    def _formulate_step_one(self):
        C = cp.Parameter(nonneg=True, name="C")
        ML = cp.Parameter(name="ML", nonneg=True)
        NS = cp.Parameter(name="NS", nonneg=True)
        M = len(self.shops_idx)
        N = len(self.commodities_idx)
        Q = cp.Parameter((N, M), nonneg=True, name="Q")
        Wb = cp.Parameter((N, M), nonneg=True, name="Wb")
        Ws = cp.Parameter((N, M), nonneg=True, name="Ws")

        B = cp.Parameter((N, M), nonneg=True, name="B")
        S = cp.Parameter((N, M), nonneg=True, name="S")
        D = cp.Parameter((N, M), nonneg=True, name="D")
        P = cp.Parameter((N, M), nonneg=True, name="P")
        R = cp.Parameter((M, M), nonneg=True, name="R")

        I = cp.Variable((N, M), nonneg=True, name="I")
        L = cp.Variable((N, M), nonneg=True, name="L")
        X = cp.Variable(M, boolean=True, name="X")
        A = cp.Variable((M, M), boolean=True, name="A")

        objective = cp.Maximize(cp.sum(cp.multiply(L, S)) - cp.sum(cp.multiply(I, B)))

        constraints = []

        # (1)
        constraints.append(
            cp.sum(I, axis=1) == cp.sum(L, axis=1)
        )

        # (2)
        constraints.append(
            cp.sum(I, axis=0) <= C
        )

        # (3)
        constraints.append(
            cp.sum(L, axis=0) <= C
        )

        # (4)
        constraints.append(
            cp.sum(L, axis=0) + cp.sum(I, axis=0) <= 10 * C * X
        )

        for i in range(M):
            for j in range(i + 1, M):
                constraints.append((1 - A[i, j]) <= (1 - X[i]) + (1 - X[j]))

        constraints.append(
            cp.multiply(A, R) <= ML
        )

        # (5)
        constraints.append(
            cp.multiply(L, Ws) <= Q
        )
        constraints.append(
            cp.multiply(I, Wb) <= Q
        )

        # (6)
        constraints.append(I <= P)

        # (7)
        constraints.append(L <= D)

        # (9)
        constraints.append(
            cp.sum(X) == NS
        )

        return cp.Problem(objective, constraints)

    def _formulate_refinement(self, n_locs, n_coms):
        B = cp.Parameter((n_coms, n_locs), nonneg=True, name="B")
        S = cp.Parameter((n_coms, n_locs), nonneg=True, name="S")
        D = cp.Parameter((n_coms, n_locs), nonneg=True, name="D")
        P = cp.Parameter((n_coms, n_locs), nonneg=True, name="P")
        Q = cp.Parameter((n_coms, n_locs), nonneg=True, name="Q")
        Wb = cp.Parameter((n_coms, n_locs), nonneg=True, name="Wb")
        Ws = cp.Parameter((n_coms, n_locs), nonneg=True, name="Ws")
        C = cp.Parameter(nonneg=True, name="C")

        MCF = [cp.Variable((n_locs + 2, n_locs + 2), nonneg=True) for _ in
               range(n_locs + 2)]
        F = [cp.Variable((n_locs, n_locs), nonneg=True) for _ in range(n_coms)]
        X = cp.Variable((n_locs + 2, n_locs + 2), boolean=True, name="X")
        I = cp.Variable((n_coms, n_locs), nonneg=True, name="I")
        L = cp.Variable((n_coms, n_locs), nonneg=True, name="L")

        objective = cp.Maximize(cp.sum(cp.multiply(L, S)) - cp.sum(cp.multiply(I, B)))

        constraints = []

        # (1)
        constraints.append(
            cp.sum(X[:-1, :], axis=1) == 1
        )

        # (2)
        constraints.append(
            cp.sum(X[:, :-2], axis=0) == 1
        )

        constraints.append(cp.sum(X[:, -1]) == 1)
        constraints.append(cp.sum(X[-1, :]) == 0)
        constraints.append(cp.sum(X[-2, :]) == 1)
        constraints.append(cp.sum(X[:, -2]) == 0)

        k_range = list(range(0, n_locs + 2))
        k_range.remove(n_locs)
        for flow in MCF:
            constraints.append(X >= flow)

        for k in k_range:
            constraints.append(
                cp.sum(MCF[k][n_locs, :]) == 1
            )

        for k in k_range:
            constraints.append(
                cp.sum(MCF[k][:, k]) == 1
            )

        for k in k_range:
            j_range = list(range(0, n_locs))
            j_range.append(n_locs + 1)
            j_range.remove(k)
            for j in j_range:
                constraints.append(
                    cp.sum(MCF[k][:, j]) ==
                    cp.sum(MCF[k][j, :]) - MCF[k][j, n_locs]
                )

        # (3)
        for flow in F:
            constraints.append(10 * C * X[:-2, :-2] >= flow)

        # (4)
        for g, flow in enumerate(F):
            for j in range(n_locs):
                constraints.append(
                    cp.sum(flow[j, :]) - cp.sum(flow[:, j]) == I[g, j] - L[g, j]
                )

        # (5) (6)
        constraints.append(I <= P)
        constraints.append(L <= D)
        constraints.append(
            cp.multiply(L, Ws) <= Q
        )
        constraints.append(
            cp.multiply(I, Wb) <= Q
        )

        # (7)
        agg_flow = sum(F)
        constraints.append(
            cp.sum(agg_flow, axis=1) <= C
        )

        return cp.Problem(objective, constraints)


class LowLevelRoutePlanner:

    def __init__(self, solver=None):
        self.solver = solver
        self._S = None
        self._D = None
        self._R = None
        self._C = None
        self._F = []
        self._X = None
        self._I = None
        self._L = None
        self._prev_com_size = -1
        self._prev_loc_size = -1
        self._prev_visits = -1

    def plan_route(self, max_cargo, max_visits, high_level_plan):
        shop_idx, shop_rev_idx, com_idx, com_rev_idx = build_idx([t for t in high_level_plan.buy] +
                                                                 [t for t in high_level_plan.sell])
        n_locs = len(shop_idx)
        n_coms = len(com_idx)
        if n_locs != self._prev_loc_size or \
                n_coms != self._prev_com_size or \
                max_visits != self._prev_visits:
            self._formulate_problem(max_visits, n_locs, n_coms)
            self._prev_com_size = n_coms
            self._prev_loc_size = n_locs
            self._prev_visits = max_visits

        self._S.value = np.zeros((n_coms, n_locs))
        self._D.value = np.zeros((n_coms, n_locs))
        self._R.value = compute_travel_cost([shop_rev_idx[i] for i in range(n_locs)], shop_idx)
        self._R.value = np.power(10, self._R.value)
        self._C.value = max_cargo
        for t in high_level_plan.buy:
            self._S.value[com_idx[t.com_idx], shop_idx[t.loc_idx]] = t.amount
        for t in high_level_plan.sell:
            self._D.value[com_idx[t.com_idx], shop_idx[t.loc_idx]] = t.amount
        cost = self._prob.solve(solver=self.solver)
        if math.isfinite(cost):
            self._I.value[np.where(self._I.value < EPSILON)] = 0
            self._L.value[np.where(self._L.value < EPSILON)] = 0
            return cost, self._extract_route(max_visits, shop_rev_idx, com_rev_idx)
        return cost, None

    def _extract_route(self, max_visits, rev_shop_idx, rev_com_idx):
        cur_idx = np.nonzero(self._X.value[-2, :])[0][0]
        final_routes = []

        cur_start = "start"
        working_dest = ""
        working_buy = []
        working_sell = []
        while cur_idx != self._X.value.shape[0] - 1:
            shop_i = int(math.floor(cur_idx / max_visits))
            cur_end = rev_shop_idx[shop_i]
            new_route = cur_end != working_dest
            if new_route:
                working_dest = cur_end
                working_buy = []
                working_sell = []

            for com in np.nonzero(self._I.value[:, cur_idx])[0]:
                working_buy.append(Transaction(rev_shop_idx[shop_i],
                                               rev_com_idx[com],
                                               self._I.value[com, cur_idx]))
            for com in np.nonzero(self._L.value[:, cur_idx])[0]:
                working_sell.append(Transaction(rev_shop_idx[shop_i],
                                                rev_com_idx[com],
                                                self._L.value[com, cur_idx]))
            if new_route:
                final_routes.append(RoutePath(cur_start, working_dest, working_buy, working_sell))
            cur_start = cur_end
            cur_idx = np.nonzero(self._X.value[cur_idx, :])[0][0]
        return final_routes

    def _formulate_problem(self, max_visit, n_locs, n_coms):
        self._S = cp.Parameter((n_coms, n_locs), nonneg=True)
        self._D = cp.Parameter((n_coms, n_locs), nonneg=True)
        self._R = cp.Parameter((n_locs, n_locs), nonneg=True)
        self._C = cp.Parameter(nonneg=True)

        self._MCF = [cp.Variable((n_locs * max_visit + 2, n_locs * max_visit + 2), nonneg=True) for _ in
                     range(n_locs * max_visit + 2)]
        self._F = [cp.Variable((n_locs * max_visit, n_locs * max_visit), nonneg=True) for _ in range(n_coms)]
        self._X = cp.Variable((n_locs * max_visit + 2, n_locs * max_visit + 2), boolean=True)
        self._I = cp.Variable((n_coms, n_locs * max_visit), nonneg=True)
        self._L = cp.Variable((n_coms, n_locs * max_visit), nonneg=True)

        expanded_cost = 0
        for i in range(n_locs):
            for j in range(n_locs):
                i_start = i * max_visit
                i_end = i_start + max_visit
                j_start = j * max_visit
                j_end = j_start + max_visit
                expanded_cost = expanded_cost + cp.sum(self._X[i_start:i_end, j_start:j_end]) * self._R[i, j]

        objective = cp.Minimize(expanded_cost)

        constraints = []

        # (1)
        constraints.append(
            cp.sum(self._X[:-1, :], axis=1) == 1
        )

        # (2)
        constraints.append(
            cp.sum(self._X[:, :-2], axis=0) == 1
        )

        constraints.append(cp.sum(self._X[:, -1]) == 1)
        constraints.append(cp.sum(self._X[-1, :]) == 0)
        constraints.append(cp.sum(self._X[-2, :]) == 1)
        constraints.append(cp.sum(self._X[:, -2]) == 0)

        k_range = list(range(0, n_locs * max_visit + 2))
        k_range.remove(n_locs * max_visit)
        for flow in self._MCF:
            constraints.append(self._X >= flow)

        for k in k_range:
            constraints.append(
                cp.sum(self._MCF[k][(n_locs * max_visit), :]) == 1
            )

        for k in k_range:
            constraints.append(
                cp.sum(self._MCF[k][:, k]) == 1
            )

        for k in k_range:
            j_range = list(range(0, n_locs * max_visit))
            j_range.append(n_locs * max_visit + 1)
            j_range.remove(k)
            for j in j_range:
                constraints.append(
                    cp.sum(self._MCF[k][:, j]) ==
                    cp.sum(self._MCF[k][j, :]) - self._MCF[k][j, n_locs * max_visit]
                )

        # (3)
        for flow in self._F:
            constraints.append(10 * self._C * self._X[:-2, :-2] >= flow)

        # (4)
        for g, flow in enumerate(self._F):
            for j in range(n_locs * max_visit):
                constraints.append(
                    cp.sum(flow[j, :]) - cp.sum(flow[:, j]) == self._I[g, j] - self._L[g, j]
                )

        # (5) (6)
        for g in range(n_coms):
            for i in range(n_locs):
                start = i * max_visit
                end = start + max_visit
                constraints.append(cp.sum(self._I[g, start:end]) == self._S[g, i])
                constraints.append(cp.sum(self._L[g, start:end]) == self._D[g, i])

        # (7)
        agg_flow = sum(self._F)
        constraints.append(
            cp.sum(agg_flow, axis=1) <= self._C
        )

        self._prob = cp.Problem(objective, constraints)


shops = []
with open("shops.json", "r") as fp:
    temp = json.load(fp)
    for t in temp:
        path = t[0]
        buy_temp = t[1]
        sell_temp = t[2]
        buys = [Commodity(*b) for b in buy_temp]
        sells = [Commodity(*s) for s in sell_temp]

        shops.append(Shop(path, buys, sells))

shop_buy_index = {}
for i, s in enumerate(shops):
    shop_buy_index[s.path] = {}

    for j, c in enumerate(s.sells):
        if c.name not in shop_buy_index[s.path]:
            shop_buy_index[s.path][c.name] = (i, j)

shop_sell_index = {}
for i, s in enumerate(shops):
    shop_sell_index[s.path] = {}

    for j, c in enumerate(s.buys):
        if c.name not in shop_sell_index[s.path]:
            shop_sell_index[s.path][c.name] = (i, j)

local = threading.local()


def get_valid_shops(filter_regex):
    paths = set()
    try:
        filter_func = create_filter(filter_regex)
    except re.error:
        return []
    for s in shops:
        if filter_func(s.path):
            paths.add(s.path)
    return list(paths)


def get_valid_coms(filter_regex):
    coms = set()
    try:
        filter_func = create_filter(filter_regex)
    except re.error:
        return []
    for s in shops:
        if filter_func(s.path):
            for c in s.buys:
                coms.add(c.name)
            for c in s.sells:
                coms.add(c.name)
    return list(coms)


def create_filter(regex):
    pattern = re.compile(regex)
    return lambda text: pattern.search(text)


DEFAULT_RESULT = HighLevelPlan(0, 0, [], []), []


def null_solver(**kwargs):
    return DEFAULT_RESULT


def get_solver(filter_regex):
    try:
        _ = local.solver_cache
    except AttributeError:
        local.solver_cache = TTLCache(maxsize=128, ttl=timedelta(hours=12).seconds)
    try:
        filter_func = create_filter(filter_regex)
        if "ts_planner_" + filter_regex in local.solver_cache:
            ts_planner = local.solver_cache["ts_planner_" + filter_regex]
        else:
            filtered_shops = list(filter(lambda s: filter_func(s.path), shops))
            ts_planner = HighLevelPlanner(filtered_shops, solver="SCIP", ignore_dpp=True)
            local.solver_cache["ts_planner_" + filter_regex] = ts_planner
        if "tl_planner" in local.solver_cache:
            tl_planner = local.solver_cache["tl_planner"]
        else:
            tl_planner = LowLevelRoutePlanner(solver="SCIP")
            local.solver_cache["tl_planner"] = tl_planner
    except Exception as e:
        print(e)
        return null_solver

    def solve_problem(max_cargo, max_stops, max_range, blk_locs, com_restricts, restrictions):
        _, plan = ts_planner.plan_stage_one(max_cargo, max_percent=1,
                                            n_stop=max_stops,
                                            max_level=max_range,
                                            blk_locations=blk_locs,
                                            max_commodity=com_restricts,
                                            max_com_loc=restrictions)
        if plan is None or len(plan.buy) == 0:
            return DEFAULT_RESULT
        _, plan = ts_planner.plan_refinement(plan, max_cargo, max_percent=1,
                                             blk_locations=blk_locs,
                                             max_commodity=com_restricts,
                                             max_com_loc=restrictions)
        if plan is None or len(plan.buy) == 0:
            return DEFAULT_RESULT
        num_max = 1
        cost, routes = tl_planner.plan_route(max_cargo, num_max, plan)
        while routes is None:
            num_max = num_max + 1
            cost, routes = tl_planner.plan_route(max_cargo, num_max, plan)
        return plan, routes

    return solve_problem
