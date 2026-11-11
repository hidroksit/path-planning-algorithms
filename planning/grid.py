"""Doluluk ızgarası (occupancy grid): 0 = boş, 1 = engel. Koordinatlar (satır, sütun)."""
import math

import numpy as np

# 8-komşuluk: (dr, dc, maliyet)
NEIGHBORS_8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
NEIGHBORS_4 = NEIGHBORS_8[:4]


class GridMap:
    def __init__(self, occ):
        self.occ = np.asarray(occ, dtype=np.uint8)
        self.h, self.w = self.occ.shape

    def in_bounds(self, p):
        return 0 <= p[0] < self.h and 0 <= p[1] < self.w

    def free(self, p):
        return self.in_bounds(p) and self.occ[p[0], p[1]] == 0

    def neighbors(self, p, diagonal=True):
        r, c = p
        for dr, dc, cost in (NEIGHBORS_8 if diagonal else NEIGHBORS_4):
            q = (r + dr, c + dc)
            if not self.free(q):
                continue
            # köşe kesmeyi engelle: çapraz geçişte iki yan hücre de boş olmalı
            if dr and dc and not (self.free((r + dr, c)) and self.free((r, c + dc))):
                continue
            yield q, cost

    def line_free(self, a, b, step=0.25):
        """a ile b arasındaki doğru parçası engele çarpmadan geçiyor mu? (sürekli koordinat)"""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.linalg.norm(b - a) / step))
        for t in np.linspace(0, 1, n + 1):
            r, c = a + t * (b - a)
            if not self.free((int(round(r)), int(round(c)))):
                return False
        return True

    def inflate(self, radius):
        """Engelleri robot yarıçapı kadar büyüt (robotu nokta kabul edebilmek için)."""
        from scipy.ndimage import binary_dilation
        k = 2 * radius + 1
        yy, xx = np.mgrid[:k, :k] - radius
        disk = (yy ** 2 + xx ** 2) <= radius ** 2
        return GridMap(binary_dilation(self.occ, structure=disk).astype(np.uint8))


def random_map(h=60, w=90, n_obstacles=28, seed=0, keep_free=()):
    """Rastgele dikdörtgen engellerle harita üretir; keep_free içindeki hücreler boş bırakılır."""
    rng = np.random.default_rng(seed)
    occ = np.zeros((h, w), dtype=np.uint8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 1
    for _ in range(n_obstacles):
        rh, rw = rng.integers(3, 12), rng.integers(3, 12)
        r, c = rng.integers(1, h - rh - 1), rng.integers(1, w - rw - 1)
        occ[r:r + rh, c:c + rw] = 1
    for (r, c) in keep_free:
        occ[max(r - 2, 1):r + 3, max(c - 2, 1):c + 3] = 0
    return GridMap(occ)


def maze_map():
    """Duvarlı, U-tuzaklı sabit bir harita (potansiyel alanın yerel minimum sorununu göstermek için)."""
    occ = np.zeros((60, 90), dtype=np.uint8)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 1
    occ[10:50, 25:27] = 1          # dikey duvar
    occ[0:30, 50:52] = 1           # üstten inen duvar
    occ[38:60, 50:52] = 1          # alttan çıkan duvar
    occ[20:22, 62:80] = 1          # U-tuzak üst kolu
    occ[40:42, 62:80] = 1          # U-tuzak alt kolu
    occ[20:42, 78:80] = 1          # U-tuzak dibi
    return GridMap(occ)
