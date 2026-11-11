import math

import numpy as np
import pytest

from planning import (GridMap, maze_map, random_map, bfs, dijkstra, greedy, astar, rrt,
                      potential_field, guided_potential_field, shortcut, path_length, octile)

START, GOAL = (30, 5), (30, 85)


def is_valid(grid, path):
    return all(grid.line_free(a, b) for a, b in zip(path, path[1:]))


@pytest.mark.parametrize("seed", range(5))
def test_astar_matches_dijkstra_cost(seed):
    grid = random_map(seed=seed, keep_free=[START, GOAL])
    a, d = astar(grid, START, GOAL), dijkstra(grid, START, GOAL)
    assert a.found and d.found
    assert math.isclose(a.cost, d.cost, rel_tol=1e-9)   # tutarlı sezgisel => optimal
    assert a.expanded <= d.expanded                      # ve daha az düğüm


def test_weighted_astar_is_bounded_suboptimal():
    grid = maze_map()
    opt, w = astar(grid, START, GOAL), astar(grid, START, GOAL, weight=2.0)
    assert opt.cost <= w.cost <= 2.0 * opt.cost


def test_paths_are_collision_free():
    grid = maze_map()
    for planner in (bfs, dijkstra, greedy, astar):
        r = planner(grid, START, GOAL)
        assert r.found and r.path[0] == START and r.path[-1] == GOAL
        assert all(grid.free(p) for p in r.path)


def test_no_path_when_blocked():
    occ = np.zeros((20, 20), dtype=np.uint8)
    occ[:, 10] = 1
    r = astar(GridMap(occ), (5, 2), (5, 18))
    assert not r.found and r.cost == math.inf


def test_no_corner_cutting():
    occ = np.zeros((3, 3), dtype=np.uint8)
    occ[0, 1] = occ[1, 0] = 1
    assert (0, 0) not in [q for q, _ in GridMap(occ).neighbors((1, 1))]


def test_octile_is_admissible():
    grid = random_map(seed=1, keep_free=[START, GOAL])
    assert octile(START, GOAL) <= dijkstra(grid, START, GOAL).cost


def test_shortcut_never_longer_and_stays_valid():
    grid = maze_map()
    p = astar(grid, START, GOAL).path
    s = shortcut(grid, p)
    assert path_length(s) <= path_length(p) + 1e-9
    assert is_valid(grid, s)


def test_rrt_star_improves_on_rrt():
    grid = random_map(seed=2, keep_free=[START, GOAL])
    r, _ = rrt(grid, START, GOAL, seed=0)
    rs, _ = rrt(grid, START, GOAL, star=True, max_iter=3000, seed=0)
    assert r.found and rs.found
    assert rs.cost < r.cost
    assert is_valid(grid, rs.path)


def test_potential_field_gets_stuck_in_u_trap_without_escape():
    r, stuck = potential_field(maze_map(), START, GOAL, escape=False)
    assert not r.found and stuck == 1


def test_guided_potential_field_reaches_goal():
    r, _ = guided_potential_field(maze_map(), START, GOAL, astar)
    assert r.found and is_valid(maze_map(), r.path)
