//! Parity gate for the hand-written forward pass (ARCHITECTURE §0 gate 8):
//! nn.rs must match PyTorch to < 1e-4 on torso, value head, and every decoder
//! primitive of both heads.
//!
//! Fixtures come from `python scripts/gen_nn_fixtures.py` (writes
//! sim/target/nn_fixtures/{weights.bin, io.bin}). The test SKIPS cleanly when
//! fixtures are absent so plain `cargo test` stays green on a fresh clone;
//! run the generator first to arm it. After training, regenerate against the
//! real checkpoint for the final gate.

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use terminal_sim::nn::{NnNet, N_CELLS};

const TOL: f32 = 1e-4;
const N_PLANES: usize = 18;
const N_SCALARS: usize = 14;
const CH: usize = 64;
const HIDDEN: usize = 128;

struct Reader {
    buf: Vec<u8>,
    off: usize,
}

impl Reader {
    fn f32s(&mut self, n: usize) -> Vec<f32> {
        let raw = &self.buf[self.off..self.off + 4 * n];
        self.off += 4 * n;
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn u32s(&mut self, n: usize) -> Vec<u32> {
        let raw = &self.buf[self.off..self.off + 4 * n];
        self.off += 4 * n;
        raw.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn nn_matches_torch_reference() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/target/nn_fixtures");
    let wpath = format!("{}/weights.bin", dir);
    let iopath = format!("{}/io.bin", dir);
    if !std::path::Path::new(&iopath).exists() {
        eprintln!(
            "SKIP nn_parity: no fixtures — run `python scripts/gen_nn_fixtures.py`"
        );
        return;
    }

    let net = NnNet::load(&wpath).expect("weights.bin");
    let mut buf = Vec::new();
    File::open(&iopath).unwrap().read_to_end(&mut buf).unwrap();
    assert_eq!(&buf[..4], b"TVF1", "fixture magic");
    let n = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let mut r = Reader { buf, off: 8 };

    let boards = r.f32s(n * N_PLANES * N_CELLS);
    let scalars = r.f32s(n * N_SCALARS);
    let feat_ref = r.f32s(n * CH * N_CELLS);
    let g_ref = r.f32s(n * HIDDEN);
    let v_ref = r.f32s(n);

    let mut worst = 0.0f32;
    let mut feats: Vec<Vec<f32>> = Vec::new();
    let mut gs: Vec<Vec<f32>> = Vec::new();
    let t0 = Instant::now();
    for i in 0..n {
        let (feat, g) = net.forward_torso(
            &boards[i * N_PLANES * N_CELLS..(i + 1) * N_PLANES * N_CELLS],
            &scalars[i * N_SCALARS..(i + 1) * N_SCALARS],
        );
        let v = net.value(&g);
        worst = worst.max(max_abs_diff(
            &feat,
            &feat_ref[i * CH * N_CELLS..(i + 1) * CH * N_CELLS],
        ));
        worst = worst.max(max_abs_diff(&g, &g_ref[i * HIDDEN..(i + 1) * HIDDEN]));
        worst = worst.max((v - v_ref[i]).abs());
        feats.push(feat);
        gs.push(g);
    }
    let torso_ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;

    for head_name in ["policy", "predict"] {
        let dec = if head_name == "policy" {
            &net.policy
        } else {
            &net.predict
        };
        let c0_ref = r.f32s(n * HIDDEN);
        let keys_ref = r.f32s(n * CH * N_CELLS);
        let type_ref = r.f32s(n * 9);
        let loc_ref = r.f32s(n * N_CELLS);
        let locs = r.u32s(n);
        let count_ref = r.f32s(n * 8);
        let ttypes = r.u32s(n);
        let counts = r.u32s(n);
        let adv_ref = r.f32s(n * HIDDEN);

        for i in 0..n {
            let (c0, keys) = dec.init(&feats[i], &gs[i]);
            worst = worst.max(max_abs_diff(&c0, &c0_ref[i * HIDDEN..(i + 1) * HIDDEN]));
            worst = worst.max(max_abs_diff(
                &keys,
                &keys_ref[i * CH * N_CELLS..(i + 1) * CH * N_CELLS],
            ));
            let tl = dec.type_logits(&c0);
            worst = worst.max(max_abs_diff(&tl, &type_ref[i * 9..(i + 1) * 9]));
            let ll = dec.loc_logits(&c0, &keys);
            worst = worst.max(max_abs_diff(
                &ll,
                &loc_ref[i * N_CELLS..(i + 1) * N_CELLS],
            ));
            let cl = dec.count_logits(&c0, &feats[i], locs[i] as usize);
            worst = worst.max(max_abs_diff(&cl, &count_ref[i * 8..(i + 1) * 8]));
            let adv = dec.advance(
                &c0,
                &feats[i],
                ttypes[i] as usize,
                locs[i] as usize,
                counts[i] as usize,
            );
            worst = worst.max(max_abs_diff(&adv, &adv_ref[i * HIDDEN..(i + 1) * HIDDEN]));
        }
        eprintln!("head {}: cumulative worst {:.3e}", head_name, worst);
    }

    eprintln!(
        "nn parity worst |diff| = {:.3e} (tol {:.0e}); torso+value {:.1} ms/state",
        worst, TOL, torso_ms
    );
    assert!(worst < TOL, "parity {} > {}", worst, TOL);
}
