//! Hand-written TerminalNet forward pass (deployment inference ladder rung 1,
//! ARCHITECTURE §9.1 / §0 gate 8).
//!
//! Loads `weights.bin` (the train/export.py `TVW1` format) and reproduces
//! deploy/npforward.py — itself parity-gated against PyTorch to < 1e-4 —
//! primitive for primitive:
//!
//!   torso     3x3 stem conv + FiLM-lite scalar bias + 6 norm-free res blocks
//!   value     g -> 256 -> 1, tanh   (the ONLY head decisions consult)
//!   decoders  policy + predict: init / type_logits / loc_logits /
//!             count_logits / advance (GRUCell, torch gate order r, z, n)
//!
//! The aux head is intentionally NOT ported: it exists for representation
//! shaping during training and is never evaluated at match time (§4.2).
//!
//! Layout conventions (identical to the Python side):
//!   feature maps  [C][x*28+y]  row-major, x first — matches board_planes
//!   linear        torch nn.Linear: out[j] = b[j] + sum_i W[j][i] x[i]
//!   conv weight   torch nn.Conv2d: [O][C][ky][kx], ky shifts x, kx shifts y
//!
//! Verified by sim/tests/nn_parity.rs against fixtures generated with
//! scripts/gen_nn_fixtures.py (random-init net, torch reference outputs).
//! Pure std — no new dependencies, no feature gates.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

pub const GRID: usize = 28;
pub const N_CELLS: usize = GRID * GRID;
pub const PLANES: usize = 18;
pub const N_SCALARS: usize = 14;
pub const CH: usize = 64;
pub const HIDDEN: usize = 128;
pub const G_DIM: usize = 2 * CH;
pub const N_TYPES: usize = 9;
pub const N_BUCKETS: usize = 8;
pub const TYPE_EMB: usize = 16;
pub const COUNT_EMB: usize = 16;
pub const QK: usize = 64;
const BLOCKS: usize = 6;

// ---------------------------------------------------------------------------
// weights.bin (TVW1) loader
// ---------------------------------------------------------------------------

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct Weights {
    map: HashMap<String, Tensor>,
}

impl Weights {
    pub fn load(path: &str) -> Result<Weights, String> {
        let mut buf = Vec::new();
        File::open(path)
            .map_err(|e| format!("open {}: {}", path, e))?
            .read_to_end(&mut buf)
            .map_err(|e| format!("read {}: {}", path, e))?;
        let mut off = 0usize;
        let take = |off: &mut usize, n: usize| -> Result<&[u8], String> {
            if *off + n > buf.len() {
                return Err(format!("truncated weights.bin at offset {}", *off));
            }
            let s = &buf[*off..*off + n];
            *off += n;
            Ok(s)
        };
        if take(&mut off, 4)? != b"TVW1" {
            return Err("bad magic (want TVW1)".into());
        }
        let count = u32::from_le_bytes(take(&mut off, 4)?.try_into().unwrap());
        let mut map = HashMap::new();
        for _ in 0..count {
            let nlen =
                u16::from_le_bytes(take(&mut off, 2)?.try_into().unwrap()) as usize;
            let name = String::from_utf8(take(&mut off, nlen)?.to_vec())
                .map_err(|e| format!("tensor name utf8: {}", e))?;
            let ndim = take(&mut off, 1)?[0] as usize;
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                shape.push(
                    u32::from_le_bytes(take(&mut off, 4)?.try_into().unwrap()) as usize,
                );
            }
            let n: usize = shape.iter().product::<usize>().max(1);
            let raw = take(&mut off, 4 * n)?;
            let mut data = Vec::with_capacity(n);
            for chunk in raw.chunks_exact(4) {
                data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            map.insert(name, Tensor { shape, data });
        }
        Ok(Weights { map })
    }

    fn get(&self, name: &str, want: &[usize]) -> Result<&Tensor, String> {
        let t = self
            .map
            .get(name)
            .ok_or_else(|| format!("missing tensor {}", name))?;
        if t.shape != want {
            return Err(format!(
                "tensor {} shape {:?}, want {:?}",
                name, t.shape, want
            ));
        }
        Ok(t)
    }
}

// ---------------------------------------------------------------------------
// primitives
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// out[j] = b[j] + sum_i w[j*ni + i] * x[i]   (torch Linear, weight [no, ni])
fn linear(w: &[f32], b: &[f32], x: &[f32], no: usize, ni: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), ni);
    debug_assert_eq!(out.len(), no);
    for j in 0..no {
        let row = &w[j * ni..(j + 1) * ni];
        let mut acc = b[j];
        for i in 0..ni {
            acc += row[i] * x[i];
        }
        out[j] = acc;
    }
}

/// 3x3 pad-1 conv over [ci][784] -> [co][784]. Accumulation order (c, ky, kx)
/// with contiguous row-sliced inner loops so the compiler can vectorize.
fn conv3x3(w: &[f32], b: &[f32], x: &[f32], ci: usize, co: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), ci * N_CELLS);
    debug_assert_eq!(out.len(), co * N_CELLS);
    for o in 0..co {
        let dst = &mut out[o * N_CELLS..(o + 1) * N_CELLS];
        dst.fill(b[o]);
        for c in 0..ci {
            let src = &x[c * N_CELLS..(c + 1) * N_CELLS];
            let wbase = (o * ci + c) * 9;
            for ky in 0..3usize {
                for kx in 0..3usize {
                    let wv = w[wbase + ky * 3 + kx];
                    if wv == 0.0 {
                        continue;
                    }
                    // out[xo][yo] += wv * src[xo+ky-1][yo+kx-1]
                    let xo_lo = 1usize.saturating_sub(ky);
                    let xo_hi = GRID.min(GRID + 1 - ky); // xo+ky-1 < 28
                    let yo_lo = 1usize.saturating_sub(kx);
                    let yo_hi = GRID.min(GRID + 1 - kx);
                    for xo in xo_lo..xo_hi {
                        let xi = xo + ky - 1;
                        let d = &mut dst[xo * GRID + yo_lo..xo * GRID + yo_hi];
                        let s = &src[xi * GRID + yo_lo + kx - 1
                            ..xi * GRID + yo_hi + kx - 1];
                        for (dv, sv) in d.iter_mut().zip(s) {
                            *dv += wv * sv;
                        }
                    }
                }
            }
        }
    }
}

fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// the net
// ---------------------------------------------------------------------------

struct LinearW {
    w: Vec<f32>,
    b: Vec<f32>,
    no: usize,
    ni: usize,
}

impl LinearW {
    fn from(ws: &Weights, prefix: &str, no: usize, ni: usize) -> Result<Self, String> {
        Ok(LinearW {
            w: ws.get(&format!("{}.weight", prefix), &[no, ni])?.data.clone(),
            b: ws.get(&format!("{}.bias", prefix), &[no])?.data.clone(),
            no,
            ni,
        })
    }

    fn apply(&self, x: &[f32], out: &mut [f32]) {
        linear(&self.w, &self.b, x, self.no, self.ni, out);
    }
}

struct ConvW {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl ConvW {
    fn from(ws: &Weights, prefix: &str, co: usize, ci: usize) -> Result<Self, String> {
        Ok(ConvW {
            w: ws
                .get(&format!("{}.weight", prefix), &[co, ci, 3, 3])?
                .data
                .clone(),
            b: ws.get(&format!("{}.bias", prefix), &[co])?.data.clone(),
        })
    }
}

pub struct Decoder {
    fc_init: LinearW,
    gru_w_ih: Vec<f32>, // [3*HIDDEN, HIDDEN]
    gru_w_hh: Vec<f32>,
    gru_b_ih: Vec<f32>,
    gru_b_hh: Vec<f32>,
    fc_type: LinearW,
    fc_q: LinearW,
    wk_w: Vec<f32>, // [QK, CH] (1x1 conv flattened)
    wk_b: Vec<f32>,
    fc_count: LinearW,
    type_emb: Vec<f32>,  // [N_TYPES, TYPE_EMB]
    count_emb: Vec<f32>, // [N_BUCKETS, COUNT_EMB]
    fc_tok: LinearW,
}

impl Decoder {
    fn from(ws: &Weights, head: &str) -> Result<Self, String> {
        let p = |s: &str| format!("{}.{}", head, s);
        Ok(Decoder {
            fc_init: LinearW::from(ws, &p("fc_init"), HIDDEN, G_DIM)?,
            gru_w_ih: ws
                .get(&p("gru.weight_ih"), &[3 * HIDDEN, HIDDEN])?
                .data
                .clone(),
            gru_w_hh: ws
                .get(&p("gru.weight_hh"), &[3 * HIDDEN, HIDDEN])?
                .data
                .clone(),
            gru_b_ih: ws.get(&p("gru.bias_ih"), &[3 * HIDDEN])?.data.clone(),
            gru_b_hh: ws.get(&p("gru.bias_hh"), &[3 * HIDDEN])?.data.clone(),
            fc_type: LinearW::from(ws, &p("fc_type"), N_TYPES, HIDDEN)?,
            fc_q: LinearW::from(ws, &p("fc_q"), QK, HIDDEN)?,
            wk_w: ws.get(&p("wk.weight"), &[QK, CH, 1, 1])?.data.clone(),
            wk_b: ws.get(&p("wk.bias"), &[QK])?.data.clone(),
            fc_count: LinearW::from(ws, &p("fc_count"), N_BUCKETS, HIDDEN + CH)?,
            type_emb: ws
                .get(&p("type_emb.weight"), &[N_TYPES, TYPE_EMB])?
                .data
                .clone(),
            count_emb: ws
                .get(&p("count_emb.weight"), &[N_BUCKETS, COUNT_EMB])?
                .data
                .clone(),
            fc_tok: LinearW::from(
                ws,
                &p("fc_tok"),
                HIDDEN,
                TYPE_EMB + CH + COUNT_EMB,
            )?,
        })
    }

    /// torch GRUCell: r,z,n gate order; h' = (1-z)*n + z*h
    fn gru(&self, x: &[f32], h: &[f32], out: &mut [f32]) {
        let mut gi = [0f32; 3 * HIDDEN];
        let mut gh = [0f32; 3 * HIDDEN];
        linear(&self.gru_w_ih, &self.gru_b_ih, x, 3 * HIDDEN, HIDDEN, &mut gi);
        linear(&self.gru_w_hh, &self.gru_b_hh, h, 3 * HIDDEN, HIDDEN, &mut gh);
        for j in 0..HIDDEN {
            let r = sigmoid(gi[j] + gh[j]);
            let z = sigmoid(gi[HIDDEN + j] + gh[HIDDEN + j]);
            let n = (gi[2 * HIDDEN + j] + r * gh[2 * HIDDEN + j]).tanh();
            out[j] = (1.0 - z) * n + z * h[j];
        }
    }

    /// -> (c0 [HIDDEN], keys [QK*784])
    pub fn init(&self, feat: &[f32], g: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut c0 = vec![0f32; HIDDEN];
        self.fc_init.apply(g, &mut c0);
        for v in c0.iter_mut() {
            *v = v.tanh();
        }
        // 1x1 conv: keys[q][cell] = bk[q] + sum_c wk[q][c] * feat[c][cell]
        let mut keys = vec![0f32; QK * N_CELLS];
        for q in 0..QK {
            let dst = &mut keys[q * N_CELLS..(q + 1) * N_CELLS];
            dst.fill(self.wk_b[q]);
            for c in 0..CH {
                let wv = self.wk_w[q * CH + c];
                let src = &feat[c * N_CELLS..(c + 1) * N_CELLS];
                for (dv, sv) in dst.iter_mut().zip(src) {
                    *dv += wv * sv;
                }
            }
        }
        (c0, keys)
    }

    pub fn type_logits(&self, c: &[f32]) -> [f32; N_TYPES] {
        let mut out = [0f32; N_TYPES];
        self.fc_type.apply(c, &mut out);
        out
    }

    pub fn loc_logits(&self, c: &[f32], keys: &[f32]) -> Vec<f32> {
        let mut q = [0f32; QK];
        self.fc_q.apply(c, &mut q);
        let mut out = vec![0f32; N_CELLS];
        for (qi, &qv) in q.iter().enumerate() {
            let krow = &keys[qi * N_CELLS..(qi + 1) * N_CELLS];
            for (ov, kv) in out.iter_mut().zip(krow) {
                *ov += qv * kv;
            }
        }
        out
    }

    pub fn count_logits(&self, c: &[f32], feat: &[f32], loc: usize) -> [f32; N_BUCKETS] {
        let mut input = [0f32; HIDDEN + CH];
        input[..HIDDEN].copy_from_slice(c);
        for ch in 0..CH {
            input[HIDDEN + ch] = feat[ch * N_CELLS + loc];
        }
        let mut out = [0f32; N_BUCKETS];
        self.fc_count.apply(&input, &mut out);
        out
    }

    pub fn advance(
        &self,
        c: &[f32],
        feat: &[f32],
        ttype: usize,
        loc: usize,
        count: usize,
    ) -> Vec<f32> {
        let mut e = [0f32; TYPE_EMB + CH + COUNT_EMB];
        e[..TYPE_EMB].copy_from_slice(&self.type_emb[ttype * TYPE_EMB..(ttype + 1) * TYPE_EMB]);
        for ch in 0..CH {
            e[TYPE_EMB + ch] = feat[ch * N_CELLS + loc];
        }
        e[TYPE_EMB + CH..]
            .copy_from_slice(&self.count_emb[count * COUNT_EMB..(count + 1) * COUNT_EMB]);
        let mut x = [0f32; HIDDEN];
        self.fc_tok.apply(&e, &mut x);
        for v in x.iter_mut() {
            *v = v.tanh();
        }
        let mut out = vec![0f32; HIDDEN];
        self.gru(&x, c, &mut out);
        out
    }
}

pub struct NnNet {
    scalar_mlp0: LinearW, // 14 -> 64
    scalar_mlp2: LinearW, // 64 -> 64
    stem: ConvW,          // 18 -> 64
    blocks: Vec<(ConvW, ConvW)>,
    fc_value0: LinearW, // 128 -> 256
    fc_value2: LinearW, // 256 -> 1
    pub policy: Decoder,
    pub predict: Decoder,
}

impl NnNet {
    pub fn load(path: &str) -> Result<NnNet, String> {
        let ws = Weights::load(path)?;
        let mut blocks = Vec::with_capacity(BLOCKS);
        for i in 0..BLOCKS {
            blocks.push((
                ConvW::from(&ws, &format!("blocks.{}.c1", i), CH, CH)?,
                ConvW::from(&ws, &format!("blocks.{}.c2", i), CH, CH)?,
            ));
        }
        Ok(NnNet {
            scalar_mlp0: LinearW::from(&ws, "scalar_mlp.0", CH, N_SCALARS)?,
            scalar_mlp2: LinearW::from(&ws, "scalar_mlp.2", CH, CH)?,
            stem: ConvW::from(&ws, "stem", CH, PLANES)?,
            blocks,
            fc_value0: LinearW::from(&ws, "fc_value.0", 256, G_DIM)?,
            fc_value2: LinearW::from(&ws, "fc_value.2", 1, 256)?,
            policy: Decoder::from(&ws, "policy")?,
            predict: Decoder::from(&ws, "predict")?,
        })
    }

    /// board [18*784], scalars [14] -> (F [64*784], g [128])
    pub fn forward_torso(&self, board: &[f32], scalars: &[f32]) -> (Vec<f32>, Vec<f32>) {
        debug_assert_eq!(board.len(), PLANES * N_CELLS);
        debug_assert_eq!(scalars.len(), N_SCALARS);
        // FiLM-lite scalar encoding
        let mut s0 = [0f32; CH];
        self.scalar_mlp0.apply(scalars, &mut s0);
        relu_inplace(&mut s0);
        let mut s = [0f32; CH];
        self.scalar_mlp2.apply(&s0, &mut s);

        let mut x = vec![0f32; CH * N_CELLS];
        conv3x3(&self.stem.w, &self.stem.b, board, PLANES, CH, &mut x);
        for c in 0..CH {
            let bias = s[c];
            for v in x[c * N_CELLS..(c + 1) * N_CELLS].iter_mut() {
                *v = (*v + bias).max(0.0);
            }
        }

        let mut y = vec![0f32; CH * N_CELLS];
        let mut z = vec![0f32; CH * N_CELLS];
        for (c1, c2) in &self.blocks {
            conv3x3(&c1.w, &c1.b, &x, CH, CH, &mut y);
            relu_inplace(&mut y);
            conv3x3(&c2.w, &c2.b, &y, CH, CH, &mut z);
            for (xv, zv) in x.iter_mut().zip(&z) {
                *xv = (*xv + zv).max(0.0);
            }
        }

        let mut g = vec![0f32; G_DIM];
        for c in 0..CH {
            let row = &x[c * N_CELLS..(c + 1) * N_CELLS];
            let mut sum = 0f32;
            let mut mx = f32::NEG_INFINITY;
            for &v in row {
                sum += v;
                if v > mx {
                    mx = v;
                }
            }
            g[c] = sum / N_CELLS as f32;
            g[CH + c] = mx;
        }
        (x, g)
    }

    pub fn value(&self, g: &[f32]) -> f32 {
        let mut h = [0f32; 256];
        self.fc_value0.apply(g, &mut h);
        relu_inplace(&mut h);
        let mut out = [0f32; 1];
        self.fc_value2.apply(&h, &mut out);
        out[0].tanh()
    }

    /// Batch of states -> values (each torso is independent; deployment's
    /// value batches are the hot path, K_eff x M_eff states per decision).
    pub fn values(&self, boards: &[f32], scalars: &[f32], n: usize) -> Vec<f32> {
        debug_assert_eq!(boards.len(), n * PLANES * N_CELLS);
        debug_assert_eq!(scalars.len(), n * N_SCALARS);
        (0..n)
            .map(|i| {
                let (_, g) = self.forward_torso(
                    &boards[i * PLANES * N_CELLS..(i + 1) * PLANES * N_CELLS],
                    &scalars[i * N_SCALARS..(i + 1) * N_SCALARS],
                );
                self.value(&g)
            })
            .collect()
    }
}
