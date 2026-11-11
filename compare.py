"""Tüm planlayıcıları aynı harita üzerinde çalıştırıp karşılaştırır.

    python compare.py                 # U-tuzaklı labirent
    python compare.py --map random --seed 4
    python compare.py --gif           # A* ve Dijkstra'nın genişleme animasyonu
"""
import argparse
from pathlib import Path

import numpy as np

from planning import (maze_map, random_map, bfs, dijkstra, greedy, astar, rrt, potential_field,
                      guided_potential_field, shortcut, path_length)

ASSETS = Path(__file__).parent / "assets"


def run_all(grid, start, goal, seed=0):
    res = {}
    res["BFS (4-komşu)"] = bfs(grid, start, goal)
    res["Dijkstra"] = dijkstra(grid, start, goal)
    res["A*"] = astar(grid, start, goal)
    res["Ağırlıklı A* (w=2)"] = astar(grid, start, goal, weight=2.0)
    res["Greedy best-first"] = greedy(grid, start, goal)
    res["RRT"], _ = rrt(grid, start, goal, seed=seed)
    res["RRT*"], tree = rrt(grid, start, goal, star=True, max_iter=6000, seed=seed)
    res["Potansiyel alan"], stuck = potential_field(grid, start, goal, seed=seed)
    res["A* + potansiyel alan"], _ = guided_potential_field(grid, start, goal, astar, seed=seed)
    return res, tree, stuck


def table(res):
    rows = ["| Yöntem | Yol uzunluğu | Kısaltılmış yol | Genişletilen düğüm / örnek | Süre |",
            "|---|---|---|---|---|"]
    for name, r in res.items():
        if not r.found:
            rows.append(f"| {name} | bulunamadı | – | {r.expanded} | {r.seconds * 1000:.0f} ms |")
            continue
        if "otansiyel" in name:  # yerel planlayıcının yolunu kısaltmak yanıltıcı olur
            sc = "–"
        else:
            sc = f"{path_length(shortcut(GRID, [tuple(map(round, p)) for p in r.path])):.1f}"
        rows.append(f"| {name} | {path_length(r.path):.1f} | {sc} | {r.expanded} | {r.seconds * 1000:.0f} ms |")
    return "\n".join(rows)


def plot(grid, start, goal, res, tree, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["BFS (4-komşu)", "Dijkstra", "A*", "Greedy best-first",
             "RRT", "RRT*", "Potansiyel alan", "A* + potansiyel alan"]
    fig, axes = plt.subplots(2, 4, figsize=(19, 7.6))
    for ax, name in zip(axes.flat, names):
        r = res[name]
        ax.imshow(grid.occ, cmap="Greys", origin="upper")
        if r.visited:
            v = np.array(r.visited)
            ax.scatter(v[:, 1], v[:, 0], s=1.2, c="#9ecae1", marker="s", linewidths=0)
        if name == "RRT*":
            for i, p in enumerate(tree.parent):
                if p >= 0:
                    a, b = tree.nodes[p], tree.nodes[i]
                    ax.plot([a[1], b[1]], [a[0], b[0]], color="#c6dbef", lw=0.5)
        if r.found:
            p = np.array(r.path)
            ax.plot(p[:, 1], p[:, 0], color="#e6550d", lw=2)
        ax.plot(start[1], start[0], "o", color="#31a354", ms=8)
        ax.plot(goal[1], goal[0], "*", color="#d62728", ms=13)
        sub = f"uzunluk {path_length(r.path):.1f} · {r.expanded} düğüm" if r.found else "takıldı / bulunamadı"
        ax.set_title(f"{name}\n{sub}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(h_pad=2.5)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_gif(grid, start, goal, path, frames=45):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    runs = [("Dijkstra", dijkstra(grid, start, goal)), ("A*", astar(grid, start, goal))]
    n_max = max(len(r.visited) for _, r in runs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    arts = []
    for ax, (name, r) in zip(axes, runs):
        ax.imshow(grid.occ, cmap="Greys", origin="upper")
        sc = ax.scatter([], [], s=2, c="#6baed6", marker="s", linewidths=0)
        ln, = ax.plot([], [], color="#e6550d", lw=2)
        ax.plot(start[1], start[0], "o", color="#31a354", ms=7)
        ax.plot(goal[1], goal[0], "*", color="#d62728", ms=12)
        ax.set_xticks([]); ax.set_yticks([])
        title = ax.set_title(name)
        arts.append((r, sc, ln, title, name))
    fig.tight_layout()

    def update(f):
        k = int(n_max * min(1.0, (f + 1) / (frames - 8)))
        for r, sc, ln, title, name in arts:
            v = np.array(r.visited[:k])
            if len(v):
                sc.set_offsets(v[:, ::-1])
            done = k >= len(r.visited)
            if done:
                p = np.array(r.path)
                ln.set_data(p[:, 1], p[:, 0])
            title.set_text(f"{name}: {min(k, len(r.visited))} düğüm genişletildi")
        return []

    FuncAnimation(fig, update, frames=frames).save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


def main():
    global GRID
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", choices=["maze", "random"], default="maze")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    start, goal = (30, 5), (30, 85)
    GRID = maze_map() if args.map == "maze" else random_map(seed=args.seed, keep_free=[start, goal])
    res, tree, stuck = run_all(GRID, start, goal, seed=args.seed)
    md = table(res)
    print(md)
    print(f"\nPotansiyel alan {stuck} kez yerel minimuma takıldı (rastgele yürüyüşle kaçış denendi).")
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / f"results_{args.map}.md").write_text(md + "\n", encoding="utf-8")
    plot(GRID, start, goal, res, tree, ASSETS / f"compare_{args.map}.png")
    if args.gif:
        make_gif(GRID, start, goal, ASSETS / "astar_vs_dijkstra.gif")


if __name__ == "__main__":
    main()
