//! Arena geometry: 28x28 diamond, edges, quadrants. Mirrors gamelib exactly.

pub const N: i32 = 28;
pub const HALF: i32 = 14;

// Edge indices follow gamelib: 0=TOP_RIGHT, 1=TOP_LEFT, 2=BOTTOM_LEFT, 3=BOTTOM_RIGHT.
pub const TOP_RIGHT: u8 = 0;
pub const TOP_LEFT: u8 = 1;
pub const BOTTOM_LEFT: u8 = 2;
pub const BOTTOM_RIGHT: u8 = 3;

#[inline]
pub fn in_bounds(x: i32, y: i32) -> bool {
    if x < 0 || x >= N || y < 0 || y >= N {
        return false;
    }
    if y < HALF {
        let row = y + 1;
        let startx = HALF - row;
        let endx = startx + 2 * row - 1;
        x >= startx && x <= endx
    } else {
        let row = N - y;
        let startx = HALF - row;
        let endx = startx + 2 * row - 1;
        x >= startx && x <= endx
    }
}

/// Edge tiles in gamelib order (n = 0..14).
pub fn edge_tiles(edge: u8) -> [(i32, i32); 14] {
    let mut out = [(0, 0); 14];
    for n in 0..14 {
        out[n as usize] = match edge {
            TOP_RIGHT => (HALF + n, N - 1 - n),
            TOP_LEFT => (HALF - 1 - n, N - 1 - n),
            BOTTOM_LEFT => (HALF - 1 - n, n),
            BOTTOM_RIGHT => (HALF + n, n),
            _ => unreachable!(),
        };
    }
    out
}

#[inline]
pub fn is_on_edge(edge: u8, x: i32, y: i32) -> bool {
    match edge {
        TOP_RIGHT => y >= HALF && x + y == 41,
        TOP_LEFT => y >= HALF && y - x == 14,
        BOTTOM_LEFT => y < HALF && x + y == 13,
        BOTTOM_RIGHT => y < HALF && x - y == 14,
        _ => false,
    }
}

/// The edge a mobile unit spawned at (x, y) paths toward (diagonal opposite quadrant).
#[inline]
pub fn target_edge_for(x: i32, y: i32) -> u8 {
    let left = x < HALF;
    let bottom = y < HALF;
    match (left, bottom) {
        (true, true) => TOP_RIGHT,
        (true, false) => BOTTOM_RIGHT,
        (false, true) => TOP_LEFT,
        (false, false) => BOTTOM_LEFT,
    }
}

#[inline]
pub fn dist2(ax: i32, ay: i32, bx: i32, by: i32) -> i32 {
    let dx = ax - bx;
    let dy = ay - by;
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diamond_has_420_tiles() {
        let mut count = 0;
        for x in 0..N {
            for y in 0..N {
                if in_bounds(x, y) {
                    count += 1;
                }
            }
        }
        assert_eq!(count, 420);
    }

    #[test]
    fn edges_match_bounds_and_flags() {
        for e in 0..4u8 {
            for &(x, y) in edge_tiles(e).iter() {
                assert!(in_bounds(x, y), "edge {} tile {},{} out of bounds", e, x, y);
                assert!(is_on_edge(e, x, y));
            }
        }
        // No overlap between the four edges except none (corners are distinct tiles).
        let mut all: Vec<(i32, i32)> = Vec::new();
        for e in 0..4u8 {
            all.extend_from_slice(&edge_tiles(e));
        }
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 56);
    }

    #[test]
    fn bottom_row_is_two_tiles() {
        assert!(in_bounds(13, 0) && in_bounds(14, 0));
        assert!(!in_bounds(12, 0) && !in_bounds(15, 0));
    }

    #[test]
    fn target_edges() {
        assert_eq!(target_edge_for(13, 0), TOP_RIGHT);
        assert_eq!(target_edge_for(14, 0), TOP_LEFT);
        assert_eq!(target_edge_for(1, 14), BOTTOM_RIGHT);
        assert_eq!(target_edge_for(26, 15), BOTTOM_LEFT);
    }
}
