import sys
import os
from dataclasses import dataclass
from typing import Set, Tuple, Dict, List, Optional
from pathlib import Path
from collections import deque
import json
import argparse
import time
import math
from PIL import Image, ImageDraw

XY = Tuple[int, int]

# ====================== Costanti ======================
BOARD_W = 5
BOARD_H = 5

# Path constraint
START: XY = (0, 3)
TARGET: XY = (3, 0)

DEBUG = False

# ====================== Utility ======================

def dprint(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def xy_to_idx(x: int, y: int, W: int) -> int:
    return y * W + x

# ANSI helpers (use BACKGROUND tiles to ensure alignment)
ANSI_RESET = "\033[0m"

def fg_rgb(r,g,b):
    return f"\033[38;2;{r};{g};{b}m"

def bg_rgb(r,g,b):
    return f"\033[48;2;{r};{g};{b}m"

# Palette ad alto contrasto (come richiesto)
COLORS_RGB = {
    'A': (138, 43, 226),  # viola
    'B': (255, 182, 193), # rosa
    'C': (220, 20, 60),   # rosso
    'D': (230, 81, 0),    # arancione scuro
    'E': (30, 144, 255),  # blu
    'F': (255, 255, 0),   # giallo puro
    'G': (50, 205, 50),   # verde
}

# ====================== Board ======================
class Board:
    def __init__(self):
        self.W = BOARD_W
        self.H = BOARD_H
    def in_bounds(self, xy: XY) -> bool:
        x, y = xy
        return 0 <= x < self.W and 0 <= y < self.H
    def neighbors4(self, xy: XY) -> List[XY]:
        x, y = xy
        # Ordine fisso: O, E, S, N (sx, dx, giù, su) -> deterministico
        return [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]

# ====================== Pieces ======================
@dataclass(frozen=True)
class PieceVariant:
    name: str
    cells: Set[XY]
    gears: Set[XY]
    flipped: bool
    rot: int

@dataclass
class Piece:
    name: str
    base_cells: Set[XY]
    base_gears: Set[XY]
    allow_flip: bool = True
    def variants(self) -> List[PieceVariant]:
        seen = set(); out: List[PieceVariant] = []
        def norm(c: Set[XY], g: Set[XY]):
            minx = min(x for x,_ in c); miny = min(y for _,y in c)
            nc = {(x-minx, y-miny) for x,y in c}
            ng = {(x-minx, y-miny) for x,y in g}
            return nc, ng
        def rot(c: Set[XY]) -> Set[XY]:
            return {(y, -x) for x,y in c}
        def flip(c: Set[XY]) -> Set[XY]:
            return {(-x, y) for x,y in c}
        flip_opts = (False, True) if self.allow_flip else (False,)
        for f in flip_opts:
            c = flip(self.base_cells) if f else set(self.base_cells)
            g = flip(self.base_gears) if f else set(self.base_gears)
            for r in range(4):
                if r>0: c,g = rot(c), rot(g)
                nc, ng = norm(c,g)
                key = (frozenset(nc), frozenset(ng))
                if key not in seen:
                    seen.add(key)
                    out.append(PieceVariant(self.name, nc, ng, flipped=f, rot=r))
        # Ordine deterministico delle varianti (per eventuali usi futuri)
        out.sort(key=lambda v: (v.name, v.flipped, v.rot, sorted(v.cells), sorted(v.gears)))
        return out

def builtin_pieces() -> Dict[str, 'Piece']:
    # Dizionario in ordine di inserimento deterministico
    pieces: Dict[str, Piece] = {}
    pieces['A'] = Piece('A', {(0,0),(1,0),(1,1)}, {(1,0)}, True)
    pieces['B'] = Piece('B', {(0,0),(1,0),(1,1)}, {(0,0),(1,1)}, True)
    pieces['C'] = Piece('C', {(0,0),(1,0),(1,1),(2,1)}, {(0,0),(1,1)}, True)
    pieces['D'] = Piece('D', {(0,0),(1,0),(2,0)}, {(1,0),(2,0)}, True)
    pieces['E'] = Piece('E', {(0,0),(1,0),(2,0),(1,1)}, {(1,1)}, True)
    pieces['F'] = Piece('F', {(0,0),(1,0),(2,0),(1,1)}, {(0,0),(1,0),(2,0)}, True)
    pieces['G'] = Piece('G', {(0,0),(1,0),(2,0),(2,1)}, {(0,0),(1,0),(2,1)}, True)
    return pieces

# ====================== Challenge ======================
@dataclass
class FixedSlot:
    anchor: XY
    rel_shape: Set[XY]
    candidates: Dict[str, List[PieceVariant]]
    locked_name: Optional[str] = None

@dataclass
class Challenge:
    fixed_slots: List[FixedSlot]
    gear_clues: Set[XY]

def parse_grid(rows: List[str]) -> Set[XY]:
    if rows and (len(rows)!=BOARD_H or any(len(r)!=BOARD_W for r in rows)):
        raise AssertionError(f"Le griglie devono essere {BOARD_H}x{BOARD_W}.")
    out: Set[XY] = set()
    for i,row in enumerate(rows):
        for x,ch in enumerate(row):
            if ch!='.': out.add((x, BOARD_H-1-i))
    return out

def load_challenge(path: str, catalog: Dict[str, List[PieceVariant]]) -> Challenge:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    gear_clues = parse_grid(data.get('gear_grid', []))
    piece_grid = data.get('piece_grid', [])
    if piece_grid and (len(piece_grid)!=BOARD_H or any(len(r)!=BOARD_W for r in piece_grid)):
        raise AssertionError(f"'piece_grid' deve essere {BOARD_H}x{BOARD_W}.")
    letter_cells: Dict[str, Set[XY]] = {}
    for i,row in enumerate(piece_grid):
        for x,ch in enumerate(row):
            if ch!='.': letter_cells.setdefault(ch.upper(), set()).add((x, BOARD_H-1-i))
    fixed_slots: List[FixedSlot] = []
    for L in sorted(letter_cells.keys()):  # ordine deterministico sui blocchi
        abs_cells = letter_cells[L]
        minx = min(x for x,_ in abs_cells); miny = min(y for _,y in abs_cells)
        anchor = (minx, miny)
        rel = {(x-minx, y-miny) for x,y in abs_cells}
        clues_in_block = gear_clues & abs_cells
        if L not in catalog:
            raise ValueError(f"Nel JSON compare il pezzo '{L}', ma non è nel catalogo.")
        variants = catalog[L]
        vlist = [v for v in variants if v.cells == rel]
        if clues_in_block:
            def covers(v: PieceVariant) -> bool:
                abs_gears = {(anchor[0]+dx, anchor[1]+dy) for (dx,dy) in v.gears}
                return clues_in_block.issubset(abs_gears)
            vlist = [v for v in vlist if covers(v)]
        if not vlist:
            raise ValueError(f"Il pezzo '{L}' non ha varianti compatibili con la forma/clues del blocco.")
        # Variazioni ordinate stabilmente (poche comunque)
        vlist.sort(key=lambda vv: (vv.flipped, vv.rot, sorted(vv.gears)))
        fixed_slots.append(FixedSlot(anchor=anchor, rel_shape=rel, candidates={L: vlist}, locked_name=L))
    return Challenge(fixed_slots=fixed_slots, gear_clues=gear_clues)

# ====================== Solver ======================
class Solver:
    def __init__(self, board: Board, pieces: Dict[str, 'Piece'], challenge: Challenge):
        self.board = board
        self.challenge = challenge
        self.variants_cache = {k: p.variants() for k,p in pieces.items()}
        self.nodes = 0
        self.start_time = time.monotonic()
        self.visited: Set[Tuple] = set()

    def place(self, v: PieceVariant, a: XY) -> Optional[Tuple[Set[XY], Set[XY]]]:
        cells = {(a[0]+x, a[1]+y) for x,y in v.cells}
        if not all(self.board.in_bounds(c) for c in cells):
            return None
        gears = {(a[0]+x, a[1]+y) for x,y in v.gears}
        return cells, gears

    def _state_masks(self, used_cells: Set[XY], covered_clues: Set[XY], used_gears: Set[XY]) -> Tuple[int,int,int]:
        W = self.board.W
        um=0; cm=0; gm=0
        for (x,y) in sorted(used_cells): um |= 1<<xy_to_idx(x,y,W)
        for (x,y) in sorted(covered_clues): cm |= 1<<xy_to_idx(x,y,W)
        for (x,y) in sorted(used_gears): gm |= 1<<xy_to_idx(x,y,W)
        return um, cm, gm

    def solve(self) -> Optional[Dict[str, Tuple[PieceVariant, XY]]]:
        used_cells: Set[XY] = set()
        used_gears: Set[XY] = set()
        placed: Dict[str, Tuple[PieceVariant, XY]] = {}
        # Lista deterministica dei pezzi restanti
        remaining_pieces: List[str] = list(sorted(self.variants_cache.keys()))
        slot_assignments: List[Optional[Tuple[str, bool, int, Tuple[XY, ...]]]] = [None]*len(self.challenge.fixed_slots)

        def place_slots(k: int) -> Optional[Dict[str, Tuple[PieceVariant, XY]]]:
            nonlocal used_cells, used_gears, placed, remaining_pieces, slot_assignments
            used_mask, clue_mask, gear_mask = self._state_masks(used_cells, used_gears & self.challenge.gear_clues, used_gears)
            key = ('slots', k, used_mask, clue_mask, gear_mask,
                   tuple(slot_assignments), tuple(sorted(remaining_pieces)))
            if key in self.visited: return None
            self.visited.add(key)

            if DEBUG:
                elapsed = time.monotonic()-self.start_time
                uncovered = len(self.challenge.gear_clues - used_gears)
                dprint(f"nodes={self.nodes:,} slots={k}/{len(self.challenge.fixed_slots)} used={len(used_cells)}/{self.board.W*self.board.H} uncovered_clues={uncovered} elapsed={elapsed:.2f}s")

            if k == len(self.challenge.fixed_slots):
                # Ordine fisso: prima i pezzi con più gears, tie-break sul nome
                free_list = sorted(remaining_pieces,
                                   key=lambda nm: (-max(len(v.gears) for v in self.variants_cache[nm]), nm))
                return place_free(0, free_list)

            slot = self.challenge.fixed_slots[k]
            anchor = slot.anchor
            # Un solo pezzo bloccato: ordine deterministico già garantito
            candidates_items = [(slot.locked_name, slot.candidates.get(slot.locked_name, []))]
            for pname, variants in candidates_items:
                # Varianti: dal minor numero di gears a salire; tie-break stabile
                vv_list = sorted(variants, key=lambda vv: (len(vv.gears), vv.flipped, vv.rot, sorted(vv.gears)))
                for v in vv_list:
                    placed_try = self.place(v, anchor)
                    if placed_try is None: continue
                    c_abs, g_abs = placed_try
                    self.nodes += 1
                    if used_cells & c_abs: continue
                    placed[pname] = (v, anchor)
                    used_cells |= c_abs
                    used_gears |= g_abs
                    if pname in remaining_pieces:
                        remaining_pieces.remove(pname)
                    slot_assignments[k] = (pname, v.flipped, v.rot, tuple(sorted(v.gears)))
                    res = place_slots(k+1)
                    if res is not None: return res
                    slot_assignments[k] = None
                    # backtrack
                    remaining_pieces.append(pname)
                    remaining_pieces.sort()
                    used_cells -= c_abs
                    used_gears -= g_abs
                    placed.pop(pname, None)
            return None

        def place_free(i: int, free_list: List[str]) -> Optional[Dict[str, Tuple[PieceVariant, XY]]]:
            nonlocal used_cells, used_gears, placed, remaining_pieces, slot_assignments
            used_mask, clue_mask, gear_mask = self._state_masks(used_cells, used_gears & self.challenge.gear_clues, used_gears)
            placed_names = tuple(sorted(placed.keys()))
            key = ('free', i, used_mask, clue_mask, gear_mask, placed_names, tuple(free_list))
            if key in self.visited: return None
            self.visited.add(key)

            if i == len(free_list):
                full = (len(used_cells) == self.board.W * self.board.H)
                clues_ok = self.challenge.gear_clues.issubset(used_gears)
                path = self.gear_path(used_gears)
                path_ok = (path is not None)
                dprint(f"[END] full={full} clues_ok={clues_ok} path_ok={path_ok} path_len={(len(path) if path else 0)}")
                return dict(placed) if (full and clues_ok and path_ok) else None

            name = free_list[i]
            # Ordinamento deterministico delle varianti: più gears prima, tie-break stabili
            variants = sorted(self.variants_cache[name],
                              key=lambda v: (-len(v.gears), v.flipped, v.rot, sorted(v.cells), sorted(v.gears)))
            W, H = self.board.W, self.board.H
            uncovered = self.challenge.gear_clues - used_gears
            enforce_cover = len(uncovered) > 0

            for v in variants:
                if enforce_cover and v.gears:
                    anchors = set()
                    gear_rel_list = list(sorted(v.gears))
                    for (cx, cy) in sorted(uncovered):
                        for (gx, gy) in gear_rel_list:
                            a = (cx - gx, cy - gy)
                            ax, ay = a
                            if 0 <= ax < W and 0 <= ay < H:
                                placed_try = self.place(v, a)
                                if placed_try is not None:
                                    anchors.add(a)
                    anchors = sorted(anchors, key=lambda t: (t[1], t[0]))  # riga-colonna
                else:
                    anchors = [(x,y) for y in range(H) for x in range(W)]
                for a in anchors:
                    placed_try = self.place(v, a)
                    if placed_try is None: continue
                    c_abs, g_abs = placed_try
                    self.nodes += 1
                    if used_cells & c_abs: continue
                    if enforce_cover and g_abs.isdisjoint(uncovered): continue
                    placed[name] = (v, a)
                    used_cells |= c_abs
                    used_gears |= g_abs
                    if name in remaining_pieces:
                        remaining_pieces.remove(name)
                    res = place_free(i+1, free_list)
                    if res is not None: return res
                    # backtrack
                    remaining_pieces.append(name)
                    remaining_pieces.sort()
                    used_cells -= c_abs
                    used_gears -= g_abs
                    placed.pop(name, None)
            return None

        return place_slots(0)

    def gear_path(self, gears: Set[XY]) -> Optional[List[XY]]:
        if START not in gears or TARGET not in gears:
            return None
        q = deque([START])
        parent: Dict[XY, Optional[XY]] = {START: None}
        while q:
            cur = q.popleft()
            if cur == TARGET:
                path = [cur]
                while parent[cur] is not None:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            for nb in self.board.neighbors4(cur):
                if nb in gears and nb not in parent:
                    parent[nb] = cur
                    q.append(nb)
        return None

# ====================== ANSI Rendering (allineato) ======================
# Ogni cella è stampata come DUE caratteri a larghezza fissa:
#  - cella normale: background colore + due spazi
#  - cella con gear: background colore + '*' e spazio
# In entrambi i casi la larghezza visibile è 2. Usiamo uno spazio tra le celle.

def render_ansi_tiles(sol: Dict[str, Tuple[PieceVariant, XY]], board: Board) -> List[str]:
    # Costruisci mappe
    cell_piece = [[None for _ in range(board.W)] for _ in range(board.H)]
    gear_abs: Set[XY] = set()
    for piece, (v, a) in sol.items():
        for dx, dy in v.cells:
            X, Y = a[0]+dx, a[1]+dy
            cell_piece[Y][X] = piece
        for dx, dy in v.gears:
            X, Y = a[0]+dx, a[1]+dy
            gear_abs.add((X, Y))
    lines: List[str] = []
    for r_vis in range(board.H-1, -1, -1):  # stampa dall'alto
        parts: List[str] = []
        for x in range(board.W):
            y = r_vis
            piece = cell_piece[y][x]
            if piece is None:
                parts.append("  ")
            else:
                pr, pg, pb = COLORS_RGB.get(piece, (200,200,200))
                tile_bg = bg_rgb(pr, pg, pb)
                if (x, y) in gear_abs:
                    parts.append(f"{tile_bg}[97m*{ANSI_RESET}{tile_bg} {ANSI_RESET}")
                else:
                    parts.append(f"{tile_bg}  {ANSI_RESET}")
        lines.append(" ".join(parts))
    return lines

# ====================== PNG Rendering (pallini) ======================

def _build_maps(sol: Dict[str, Tuple[PieceVariant, XY]], board: Board):
    cell_piece = [[None for _ in range(board.W)] for _ in range(board.H)]
    gear_abs: Set[XY] = set()
    for piece, (v, a) in sol.items():
        for dx, dy in v.cells:
            X, Y = a[0]+dx, a[1]+dy
            cell_piece[Y][X] = piece
        for dx, dy in v.gears:
            X, Y = a[0]+dx, a[1]+dy
            gear_abs.add((X, Y))
    return cell_piece, gear_abs

def _cell_rect(x, y, cell_px, board_h):
    top = (board_h-1 - y) * cell_px
    left = x * cell_px
    return (left, top, left + cell_px, top + cell_px)

def _draw_asterisk(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, width: int = 2, color=(0,0,0)):
    for ang_deg in (0, 45, 90, 135):
        ang = math.radians(ang_deg)
        dx = r * math.cos(ang)
        dy = r * math.sin(ang)
        draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=color, width=width)

def save_png(sol: Dict[str, Tuple[PieceVariant, XY]], board: Board, path_prefix: str, cell_px: int = 96):
    W, H = board.W, board.H
    bg = (248, 248, 248)
    grid_color = (60, 60, 60)
    cell_piece, gear_abs = _build_maps(sol, board)
    img = Image.new('RGB', (W*cell_px, H*cell_px), bg)
    dr = ImageDraw.Draw(img)
    margin = max(2, int(cell_px * 0.12))
    outline_w = max(1, cell_px//24)
    gear_r = cell_px * 0.28
    gear_w = max(2, cell_px//18)
    for y in range(H):
        for x in range(W):
            rect = _cell_rect(x, y, cell_px, H)
            dr.rectangle(rect, outline=grid_color, width=outline_w)
            piece = cell_piece[y][x]
            if piece is not None:
                col = COLORS_RGB.get(piece, (200,200,200))
                dr.ellipse((rect[0]+margin, rect[1]+margin, rect[2]-margin, rect[3]-margin), fill=col)
            if (x, y) in gear_abs:
                cx = rect[0] + cell_px/2
                cy = rect[1] + cell_px/2
                _draw_asterisk(dr, cx, cy, gear_r, width=gear_w, color=(0,0,0))
    img.save(f"{path_prefix}.png")

# ====================== Main ======================

def main():
    global DEBUG
    ap = argparse.ArgumentParser(description='IQ Gears solver')
    ap.add_argument('--challenge', required=True, help='File JSON con piece_grid & gear_grid')
    ap.add_argument('--debug', action='store_true', help='Stampe di debug')
    ap.add_argument('--png', help='Prefisso file PNG (salva PREFIX.png)')
    ap.add_argument('--png-scale', type=int, default=96, help='Lato cella in pixel per il PNG (default 96)')
    args = ap.parse_args()

    DEBUG = args.debug

    board = Board()
    pieces = builtin_pieces()
    catalog = {k: p.variants() for k,p in pieces.items()}
    challenge = load_challenge(args.challenge, catalog)

    solver = Solver(board, pieces, challenge)
    sol = solver.solve()
    if not sol:
        print('Nessuna soluzione trovata.')
        if DEBUG:
            print(f"[DEBUG] nodes esplorati: {solver.nodes:,}")
        sys.exit(1)

    # ANSI allineato
    print("Griglia colorata (bg = cella; '*' = dentino)")
    for line in render_ansi_tiles(sol, board):
        print(line)
    print()

    # PNG opzionale
    if args.png:
        save_png(sol, board, args.png, cell_px=max(24, args.png_scale))
        print(f"PNG salvato: {args.png}.png")

if __name__ == '__main__':
    main()
