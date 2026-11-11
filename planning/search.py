"""Izgara üzerinde graf arama: BFS, Dijkstra, Greedy Best-First ve A*.

Hepsi aynı öncelik kuyruğu iskeletini kullanır; fark yalnızca kuyruğa
hangi anahtarla girildiğinde:
    Dijkstra : g(n)              (başlangıçtan gelinen maliyet)
    Greedy   : h(n)              (hedefe sezgisel uzaklık)
    A*       : g(n) + w * h(n)   (w = 1 iken optimal; w > 1 hızlı ama optimal değil)
"""
import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    path: list
    cost: float
    expanded: int
    seconds: float
    visited: list = field(default_factory=list)  # genişletme sırası (animasyon için)

    @property
    def found(self):
        return bool(self.path)


def octile(a, b):
    """8-komşulu ızgarada kabul edilebilir (admissible) ve tutarlı sezgisel."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reconstruct(parent, goal):
    path, n = [goal], goal
    while parent[n] is not None:
        n = parent[n]
        path.append(n)
    return path[::-1]


def best_first(grid, start, goal, mode="astar", heuristic=octile, weight=1.0, diagonal=True):
    t0 = time.perf_counter()
    g = {start: 0.0}
    parent = {start: None}
    closed, visited = set(), []
    h0 = heuristic(start, goal)
    key = {"dijkstra": 0.0, "greedy": h0, "astar": weight * h0}[mode]
    pq, counter = [(key, 0, start)], 1  # sayaç: eşit anahtarlarda kararlı sıralama

    while pq:
        _, _, node = heapq.heappop(pq)
        if node in closed:
            continue  # kuyrukta eski (daha pahalı) bir kopya kalmış
        closed.add(node)
        visited.append(node)
        if node == goal:
            return SearchResult(_reconstruct(parent, goal), g[goal], len(closed),
                                time.perf_counter() - t0, visited)
        for nb, step in grid.neighbors(node, diagonal):
            ng = g[node] + step
            if nb in closed or ng >= g.get(nb, math.inf):
                continue
            g[nb], parent[nb] = ng, node
            h = heuristic(nb, goal)
            k = {"dijkstra": ng, "greedy": h, "astar": ng + weight * h}[mode]
            heapq.heappush(pq, (k, counter, nb))
            counter += 1
    return SearchResult([], math.inf, len(closed), time.perf_counter() - t0, visited)


def bfs(grid, start, goal):
    """Ağırlıksız genişlik öncelikli arama: en az adımlı yolu bulur (4-komşu)."""
    t0 = time.perf_counter()
    parent, q, visited = {start: None}, deque([start]), []
    while q:
        node = q.popleft()
        visited.append(node)
        if node == goal:
            path = _reconstruct(parent, goal)
            return SearchResult(path, float(len(path) - 1), len(visited), time.perf_counter() - t0, visited)
        for nb, _ in grid.neighbors(node, diagonal=False):
            if nb not in parent:
                parent[nb] = node
                q.append(nb)
    return SearchResult([], math.inf, len(visited), time.perf_counter() - t0, visited)


def dijkstra(grid, start, goal):
    return best_first(grid, start, goal, mode="dijkstra")


def greedy(grid, start, goal):
    return best_first(grid, start, goal, mode="greedy")


def astar(grid, start, goal, weight=1.0):
    return best_first(grid, start, goal, mode="astar", weight=weight)


def path_length(path):
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def shortcut(grid, path):
    """Görüş hattı (line-of-sight) ile gereksiz ara noktaları atar."""
    if len(path) < 3:
        return list(path)
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not grid.line_free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out
