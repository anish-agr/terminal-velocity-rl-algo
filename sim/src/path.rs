//! Navigation: exact port of the official pathfinder (rust-algo pathfinding.rs / gamelib
//! navigation.py; algorithm confirmed by the competition rules page).
//!
//! Factoring: the expensive part (idealness BFS + validation BFS) is computed once per
//! (target edge, pocket, board layout) as a `NavField` of BFS pathlengths. Units then take
//! single steps against the field, carrying their own previous-move-axis memory across
//! repaths (⚠ MECHANICS.md: engine assumed to preserve direction memory when structures die).
//!
//! Step tie-break rules (rules page, verbatim order):
//!   min pathlength → first-ever move prefers vertical → prefer changing axis vs previous
//!   move → prefer the move toward the target edge.

use crate::geo::{self, N};

pub const BLOCKED: u32 = u32::MAX;
pub const UNSET: u32 = u32::MAX - 1;

#[derive(Clone)]
pub struct NavField {
    /// BFS distance to destination; BLOCKED for structures/out-of-bounds, UNSET for tiles
    /// not reachable from the destination side (different pocket).
    pub pathlength: Vec<u32>,
    /// True if the destination is the target edge; false if it is a self-destruct tile.
    pub reaches_edge: bool,
    pub target: u8,
}

#[inline]
pub fn idx(x: i32, y: i32) -> usize {
    (x * N + y) as usize
}

#[inline]
fn idealness(x: i32, y: i32, target: u8) -> u64 {
    if geo::is_on_edge(target, x, y) {
        return u64::MAX;
    }
    let a = match target {
        geo::TOP_LEFT | geo::TOP_RIGHT => 28 * y as u64,
        _ => 28 * (27 - y) as u64,
    };
    let b = match target {
        geo::TOP_RIGHT | geo::BOTTOM_RIGHT => x as u64,
        _ => (27 - x) as u64,
    };
    a + b
}

const NEIGHBORS: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

/// Compute the navigation field for the pocket containing (sx, sy), toward `target` edge.
pub fn compute_field(blocked: &dyn Fn(i32, i32) -> bool, sx: i32, sy: i32, target: u8) -> NavField {
    debug_assert!(!blocked(sx, sy), "field start must be unblocked");
    let mut pl = vec![UNSET; (N * N) as usize];
    for x in 0..N {
        for y in 0..N {
            if !geo::in_bounds(x, y) || blocked(x, y) {
                pl[idx(x, y)] = BLOCKED;
            }
        }
    }

    // Idealness BFS over the pocket (short-circuits when the target edge is reached).
    let mut visited = vec![false; (N * N) as usize];
    let mut queue: std::collections::VecDeque<(i32, i32)> = std::collections::VecDeque::new();
    queue.push_back((sx, sy));
    visited[idx(sx, sy)] = true;
    let mut best = (sx, sy);
    let mut best_ideal = idealness(sx, sy, target);
    let mut reaches_edge = geo::is_on_edge(target, sx, sy);
    'bfs: while let Some((cx, cy)) = queue.pop_front() {
        for (dx, dy) in NEIGHBORS {
            let (nx, ny) = (cx + dx, cy + dy);
            if !geo::in_bounds(nx, ny) || pl[idx(nx, ny)] == BLOCKED || visited[idx(nx, ny)] {
                continue;
            }
            visited[idx(nx, ny)] = true;
            if geo::is_on_edge(target, nx, ny) {
                reaches_edge = true;
                break 'bfs;
            }
            let ni = idealness(nx, ny, target);
            if ni > best_ideal {
                best_ideal = ni;
                best = (nx, ny);
            }
            queue.push_back((nx, ny));
        }
    }

    // Validation BFS from the destination set.
    let mut vq: std::collections::VecDeque<(i32, i32)> = std::collections::VecDeque::new();
    if reaches_edge {
        for &(ex, ey) in geo::edge_tiles(target).iter() {
            if pl[idx(ex, ey)] != BLOCKED {
                pl[idx(ex, ey)] = 0;
                vq.push_back((ex, ey));
            }
        }
    } else {
        pl[idx(best.0, best.1)] = 0;
        vq.push_back(best);
    }
    while let Some((cx, cy)) = vq.pop_front() {
        let cur = pl[idx(cx, cy)];
        for (dx, dy) in NEIGHBORS {
            let (nx, ny) = (cx + dx, cy + dy);
            if !geo::in_bounds(nx, ny) {
                continue;
            }
            let cell = &mut pl[idx(nx, ny)];
            if *cell == UNSET {
                *cell = cur + 1;
                vq.push_back((nx, ny));
            }
        }
    }

    NavField { pathlength: pl, reaches_edge, target }
}

/// Previous-move axis: 0 = never moved, 1 = horizontal, 2 = vertical (gamelib encoding).
pub type MoveAxis = u8;

/// One step for a unit standing at (cx, cy). None => nowhere to go (at destination or
/// trapped) => self-destruct condition per rules.
pub fn step(field: &NavField, cx: i32, cy: i32, move_axis: MoveAxis) -> Option<(i32, i32)> {
    let here = field.pathlength[idx(cx, cy)];
    if here == 0 {
        return None; // at destination
    }
    let mut best: Option<(i32, i32)> = None;
    let mut best_len = u32::MAX;
    for (dx, dy) in NEIGHBORS {
        let (nx, ny) = (cx + dx, cy + dy);
        if !geo::in_bounds(nx, ny) {
            continue;
        }
        let plen = field.pathlength[idx(nx, ny)];
        if plen == BLOCKED || plen == UNSET {
            continue;
        }
        if plen < best_len {
            best_len = plen;
            best = Some((nx, ny));
        } else if plen == best_len {
            let prev = best.unwrap();
            if prefer_new(cx, (nx, ny), prev, move_axis, field.target) {
                best = Some((nx, ny));
            }
        }
    }
    // A unit strictly follows decreasing pathlength; if the best neighbor does not improve,
    // the unit has nowhere useful to go (trapped in a pocket whose destination is itself).
    match best {
        Some(b) if best_len < here => Some(b),
        _ => None,
    }
}

/// True if `new` beats `prev` under the tie-break rules (both min-pathlength neighbors).
fn prefer_new(cx: i32, new: (i32, i32), prev: (i32, i32), move_axis: MoveAxis, target: u8) -> bool {
    let new_vertical = new.0 == cx;
    let prev_vertical = prev.0 == cx;
    if move_axis == 0 {
        if new_vertical != prev_vertical {
            return new_vertical; // first move prefers vertical
        }
    } else {
        let prev_was_vertical = move_axis == 2;
        let new_changes = new_vertical != prev_was_vertical;
        let prev_changes = prev_vertical != prev_was_vertical;
        if new_changes != prev_changes {
            return new_changes; // prefer changing axis
        }
    }
    if new_vertical == prev_vertical {
        if !new_vertical {
            let toward_right = matches!(target, geo::TOP_RIGHT | geo::BOTTOM_RIGHT);
            (new.0 > prev.0) == toward_right
        } else {
            let toward_up = matches!(target, geo::TOP_RIGHT | geo::TOP_LEFT);
            (new.1 > prev.1) == toward_up
        }
    } else {
        false // equal direction preference on mixed axes: keep first (strict-win replacement)
    }
}

/// Materialize a full path (start tile included) — used by tests, prediction features, and
/// cross-validation against gamelib. Fresh units: move_axis = 0.
pub fn pathfind(blocked: &dyn Fn(i32, i32) -> bool, sx: i32, sy: i32, target: u8) -> Vec<(i8, i8)> {
    let field = compute_field(blocked, sx, sy, target);
    let mut path = vec![(sx as i8, sy as i8)];
    let (mut cx, mut cy) = (sx, sy);
    let mut axis: MoveAxis = 0;
    while let Some((nx, ny)) = step(&field, cx, cy, axis) {
        axis = if nx == cx { 2 } else { 1 };
        cx = nx;
        cy = ny;
        path.push((cx as i8, cy as i8));
        debug_assert!(path.len() <= 800, "path runaway");
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(_x: i32, _y: i32) -> bool {
        false
    }

    #[test]
    fn empty_board_zigzag_starts_vertical() {
        let p = pathfind(&open, 13, 0, geo::TOP_RIGHT);
        assert_eq!(p[0], (13, 0));
        assert_eq!(p[1], (13, 1), "first move must be vertical");
        let last = *p.last().unwrap();
        assert!(geo::is_on_edge(geo::TOP_RIGHT, last.0 as i32, last.1 as i32));
        assert_eq!(p.len(), 29, "BFS-shortest route to TR edge from (13,0) is 28 steps");
        // strict zigzag in open space: consecutive moves alternate axis
        for w in p.windows(3) {
            let m1_vertical = w[1].0 == w[0].0;
            let m2_vertical = w[2].0 == w[1].0;
            assert_ne!(m1_vertical, m2_vertical, "expected alternation at {:?}", w);
        }
    }

    #[test]
    fn full_wall_forces_deepest_then_lateral_self_destruct_tile() {
        // Wall across the entire row y=14: bottom player can never reach the top.
        let blocked = |_x: i32, y: i32| y == 14;
        let p = pathfind(&blocked, 13, 0, geo::TOP_RIGHT);
        let last = *p.last().unwrap();
        // deepest reachable y = 13; idealness tie-break: toward x=27 for a TR target
        assert_eq!(last, (27, 13));
    }

    #[test]
    fn trapped_unit_paths_to_itself() {
        let blocked = |x: i32, y: i32| (x, y) == (13, 1) || (x, y) == (12, 0) || (x, y) == (14, 0);
        let p = pathfind(&blocked, 13, 0, geo::TOP_RIGHT);
        assert_eq!(p, vec![(13, 0)]);
    }

    #[test]
    fn rules_page_example_destination_choice() {
        // Rules: a TR-bound unit whose deepest reachable tiles are [13,26] and [14,26]
        // picks [14,26]; a TL-bound unit picks [13,26]. Build a chimney: the whole enemy
        // half blocked except the x in {13,14} column, capped at y=27 (the edge diagonals
        // are otherwise reachable at any depth).
        let blocked = |x: i32, y: i32| y == 27 || (y >= 14 && x != 13 && x != 14);
        let p_tr = pathfind(&blocked, 13, 0, geo::TOP_RIGHT);
        assert_eq!(*p_tr.last().unwrap(), (14, 26));
        let p_tl = pathfind(&blocked, 14, 0, geo::TOP_LEFT);
        assert_eq!(*p_tl.last().unwrap(), (13, 26));
    }
}
