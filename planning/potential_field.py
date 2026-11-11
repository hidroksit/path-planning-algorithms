"""Yapay potansiyel alan (artificial potential field) ile yerel planlama.

Hedef robotu çeken bir kuyu, engeller ise iten tepeler gibi davranır:

    U_att(q) = 0.5 * k_att * |q - q_goal|^2
    U_rep(q) = 0.5 * k_rep * (1/d(q) - 1/d0)^2   (d(q) <= d0 ise, değilse 0)

Robot toplam potansiyelin negatif gradyanı yönünde ilerler. Yöntem hızlı ve
basittir ama "yerel minimum" sorununa açıktır: U şeklindeki bir engelin
içinde çekme ve itme kuvvetleri dengelenir ve robot takılır. Burada takılma
algılanınca rastgele yürüyüşle kaçış denenir.
"""
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from .search import SearchResult


def potential_field(grid, start, goal, k_att=1.0, k_rep=300.0, d0=6.0, step=0.5,
                    max_steps=4000, goal_tol=1.0, escape=True, seed=0):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    # her hücrenin en yakın engele uzaklığı ve o uzaklığın gradyanı
    dist = distance_transform_edt(grid.occ == 0)
    gy, gx = np.gradient(dist)

    def force(q):
        r, c = int(round(q[0])), int(round(q[1]))
        f = -k_att * (q - goal)
        d = max(dist[r, c], 0.5)
        if d <= d0:
            mag = k_rep * (1.0 / d - 1.0 / d0) / d ** 2
            f = f + mag * np.array([gy[r, c], gx[r, c]])  # engelden uzağa
        return f

    q = np.asarray(start, float)
    goal = np.asarray(goal, float)
    path, stuck_events = [tuple(q)], 0
    for _ in range(max_steps):
        if np.linalg.norm(q - goal) <= goal_tol:
            return SearchResult(path + [tuple(goal)], _length(path + [tuple(goal)]), len(path),
                                time.perf_counter() - t0), stuck_events
        f = force(q)
        n = np.linalg.norm(f)
        q_new = q + step * f / n if n > 1e-9 else q
        if not grid.free((int(round(q_new[0])), int(round(q_new[1])))):
            q_new = q
        path.append(tuple(q_new))
        # takılma: son 40 adımda neredeyse hiç ilerleme yok
        if len(path) > 40 and np.linalg.norm(np.subtract(path[-1], path[-40])) < 1.0:
            stuck_events += 1
            if not escape:
                break
            for _ in range(60):  # rastgele yürüyüşle kaç
                cand = q_new + rng.normal(0, 1.5, size=2)
                if grid.line_free(q_new, cand):  # duvarın içinden atlamasın
                    q_new = cand
                    path.append(tuple(q_new))
        q = q_new
    return SearchResult([], float("inf"), len(path), time.perf_counter() - t0), stuck_events


def _length(path):
    p = np.asarray(path)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def guided_potential_field(grid, start, goal, global_planner, clearance=2, spacing=5,
                           k_rep=60.0, d0=3.0, **kw):
    """Pratikteki yaygın çözüm: global planlayıcı (ör. A*) engelleri `clearance`
    kadar şişirilmiş haritada yol bulur; yol `spacing` hücrede bir ara hedefe
    bölünür ve her ara hedefe potansiyel alanla gidilir. Global planlayıcı
    yerel minimumu, potansiyel alan ise engellere fazla yaklaşmayı önler."""
    t0 = time.perf_counter()
    plan = None
    for c in range(clearance, -1, -1):  # dar geçitlerde şişirmeyi azaltarak tekrar dene
        inflated = grid.inflate(c) if c > 0 else grid
        if inflated.free(tuple(start)) and inflated.free(tuple(goal)):
            plan = global_planner(inflated, start, goal)
            if plan.found:
                break
    else:
        return plan, 0
    waypoints = plan.path[spacing::spacing]
    if waypoints[-1] != plan.path[-1]:
        waypoints.append(plan.path[-1])
    path, stuck_total, q = [tuple(start)], 0, tuple(start)
    for wp in waypoints:
        seg, stuck = potential_field(grid, q, wp, k_rep=k_rep, d0=d0, goal_tol=1.0, escape=True, **kw)
        stuck_total += stuck
        if not seg.found:
            return SearchResult([], float("inf"), len(path), time.perf_counter() - t0), stuck_total
        path += seg.path[1:]
        q = seg.path[-1]
    return SearchResult(path, _length(path), plan.expanded, time.perf_counter() - t0), stuck_total
