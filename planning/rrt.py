"""Örnekleme tabanlı planlama: RRT ve RRT*.

RRT  : rastgele bir nokta seç, ağaçtaki en yakın düğümden ona doğru sabit
       adım kadar uzan, çarpışma yoksa düğümü ekle. Hızlı ama yollar zikzaklı.
RRT* : yeni düğüm için komşuluktaki en ucuz ebeveyni seçer ve komşuları yeni
       düğüm üzerinden bağlamak daha ucuzsa yeniden bağlar (rewire). Örnek
       sayısı arttıkça yol optimal yola yakınsar.
"""
import math
import time

import numpy as np

from .search import SearchResult


class _Tree:
    def __init__(self, root):
        self.nodes = [np.asarray(root, float)]
        self.parent = [-1]
        self.cost = [0.0]

    def nearest(self, p):
        d = np.linalg.norm(np.asarray(self.nodes) - p, axis=1)
        return int(d.argmin())

    def near(self, p, radius):
        d = np.linalg.norm(np.asarray(self.nodes) - p, axis=1)
        return np.flatnonzero(d <= radius)

    def add(self, p, parent, cost):
        self.nodes.append(p)
        self.parent.append(parent)
        self.cost.append(cost)
        return len(self.nodes) - 1

    def path_to(self, i):
        out = []
        while i != -1:
            out.append(tuple(self.nodes[i]))
            i = self.parent[i]
        return out[::-1]


def rrt(grid, start, goal, star=False, step=3.0, goal_rate=0.08, max_iter=6000,
        goal_tol=2.5, gamma=30.0, seed=0, stop_at_first=None):
    """star=True ise RRT*. stop_at_first=None iken RRT ilk çözümde, RRT* ise
    max_iter boyunca iyileştirmeye devam eder."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if stop_at_first is None:
        stop_at_first = not star
    tree = _Tree(start)
    goal = np.asarray(goal, float)
    best_goal, best_cost = None, math.inf
    lo, hi = np.array([0.0, 0.0]), np.array([grid.h - 1.0, grid.w - 1.0])

    for it in range(max_iter):
        sample = goal if rng.random() < goal_rate else rng.uniform(lo, hi)
        i_near = tree.nearest(sample)
        p_near = tree.nodes[i_near]
        d = np.linalg.norm(sample - p_near)
        if d < 1e-9:
            continue
        p_new = p_near + (sample - p_near) * min(1.0, step / d)
        if not grid.line_free(p_near, p_new):
            continue

        parent, cost = i_near, tree.cost[i_near] + np.linalg.norm(p_new - p_near)
        if star:
            n = len(tree.nodes)
            radius = min(gamma * math.sqrt(math.log(n + 1) / (n + 1)), 4 * step)
            near = tree.near(p_new, radius)
            for j in near:  # en ucuz ebeveyni seç
                c = tree.cost[j] + np.linalg.norm(p_new - tree.nodes[j])
                if c < cost and grid.line_free(tree.nodes[j], p_new):
                    parent, cost = j, c
        i_new = tree.add(p_new, parent, cost)

        if star:
            for j in near:  # yeniden bağla
                c = cost + np.linalg.norm(tree.nodes[j] - p_new)
                if c < tree.cost[j] and grid.line_free(p_new, tree.nodes[j]):
                    delta = c - tree.cost[j]
                    tree.parent[j], tree.cost[j] = i_new, c
                    _propagate(tree, j, delta)

        if np.linalg.norm(p_new - goal) <= goal_tol and grid.line_free(p_new, goal):
            total = cost + np.linalg.norm(goal - p_new)
            if total < best_cost:
                best_goal, best_cost = i_new, total
            if stop_at_first:
                break

    secs = time.perf_counter() - t0
    if best_goal is None:
        return SearchResult([], math.inf, len(tree.nodes), secs), tree
    # rewire sonrası maliyet değişmiş olabilir: en iyi hedef düğümünü yeniden seç
    cands = [i for i in range(len(tree.nodes))
             if np.linalg.norm(tree.nodes[i] - goal) <= goal_tol and grid.line_free(tree.nodes[i], goal)]
    best_goal = min(cands, key=lambda i: tree.cost[i] + np.linalg.norm(goal - tree.nodes[i]))
    path = tree.path_to(best_goal) + [tuple(goal)]
    cost = tree.cost[best_goal] + float(np.linalg.norm(goal - tree.nodes[best_goal]))
    return SearchResult(path, cost, len(tree.nodes), secs), tree


def _propagate(tree, i, delta):
    """Yeniden bağlanan düğümün alt ağacındaki maliyetleri güncelle."""
    stack = [i]
    children = None
    while stack:
        k = stack.pop()
        if children is None:
            children = {}
            for c, p in enumerate(tree.parent):
                children.setdefault(p, []).append(c)
        for c in children.get(k, []):
            if c != i:
                tree.cost[c] += delta
                stack.append(c)
