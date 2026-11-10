//! Function-preserving transforms that turn an exact compiled network into a
//! good *initialisation* for post-training.
//!
//! Every transform here leaves the computed function unchanged (up to
//! floating-point rounding):
//!
//! * [`normalize_inputs`]: fold the affine map from [-1, 1]^n to the declared
//!   domain into the first layer, so the network consumes standardised inputs.
//! * [`balance`]: use positive homogeneity relu(a x) = a relu(x) to equalise
//!   incoming and outgoing weight norms of every unit (well-conditioned
//!   gradients, no huge big-M rows next to tiny ones).
//! * [`widen`]: add spare units with random incoming weights and zero
//!   outgoing weights (and, for residual nets, spare stream dimensions):
//!   gradients reach them from the first step, giving free capacity.
//! * [`deepen`]: insert identity layers (relu(h) = h for h >= 0) / zero-output
//!   residual blocks, to match a target depth.

use crate::network::*;

/// Nearest power of two: rescaling by it is exact in binary floating point,
/// so transforms keep integer programs bit-exact (also in float32).
fn pow2(x: f64) -> f64 {
    2f64.powi(x.log2().round() as i32)
}

/// SplitMix64 + Box-Muller, to stay dependency-free and reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn random_row(rng: &mut Rng, n: usize, scale: f64) -> Vec<f64> {
    let std = scale / (n.max(1) as f64).sqrt();
    (0..n).map(|_| std * rng.normal()).collect()
}

/// Consume u in [-1, 1] instead of x in [lo, hi] (bounded inputs only).
pub fn normalize_inputs(net: &mut Network) {
    let n = net.n_features();
    // desired x = off + sc * u ; current weights consume u0 = (x - off0) / sc0
    // => u0 = (off - off0)/sc0 + (sc/sc0) u
    let mut shift = vec![0.0; n];
    let mut mult = vec![1.0; n];
    for i in 0..n {
        let (lo, hi) = net.input_bounds[i];
        let (off, sc) = if lo.is_finite() && hi.is_finite() && hi > lo {
            // midpoint, and a power-of-two half-width >= the true one: u stays
            // within [-1, 1] and every folded weight stays dyadic
            ((lo + hi) / 2.0, 2f64.powi(((hi - lo) / 2.0).log2().ceil() as i32))
        } else {
            (net.in_offset[i], net.in_scale[i])
        };
        shift[i] = (off - net.in_offset[i]) / net.in_scale[i];
        mult[i] = sc / net.in_scale[i];
        net.in_offset[i] = off;
        net.in_scale[i] = sc;
    }
    let firsts: Vec<&mut Dense> = match &mut net.body {
        Body::Mlp { hidden, out } => vec![hidden.first_mut().unwrap_or(out)],
        Body::Residual { embed, .. } => vec![embed],
        Body::Transformer { tok_embed, .. } => tok_embed.iter_mut().collect(),
    };
    for first in firsts {
        for r in 0..first.rows {
            let mut db = 0.0;
            for c in 0..n {
                let w = first.at(r, c);
                db += w * shift[c];
                *first.at_mut(r, c) = w * mult[c];
            }
            first.b[r] += db;
        }
    }
}

fn balance_block(b: &mut Block) {
    for u in 0..b.w_in.rows {
        let na = b.w_in.row_norm(u, true);
        let no = b.w_out.col_norm(u);
        if na == 0.0 || no == 0.0 {
            continue;
        }
        match &mut b.w_val {
            None => {
                let a = pow2((no / na).sqrt());
                b.w_in.scale_row(u, a);
                b.w_out.scale_col(u, 1.0 / a);
            }
            Some(wv) => {
                let nb = wv.row_norm(u, true);
                if nb == 0.0 {
                    continue;
                }
                let t = (na * nb * no).cbrt();
                let (a, c) = (pow2(t / na), pow2(t / nb));
                b.w_in.scale_row(u, a);
                wv.scale_row(u, c);
                b.w_out.scale_col(u, 1.0 / (a * c));
            }
        }
    }
}

/// Add spare units (random in, zero out) to a residual-style block.
fn widen_block(b: &mut Block, extra: usize, d: usize, scale: f64, rng: &mut Rng) {
    for _ in 0..extra {
        let row = random_row(rng, d, scale);
        b.w_in.push_row(&row, 0.0);
        if let Some(wv) = &mut b.w_val {
            let row = random_row(rng, d, scale);
            wv.push_row(&row, 1.0);
        }
    }
    b.w_out.push_cols(extra, |_, _| 0.0);
}

fn random_block(d: usize, width: usize, gated: bool, scale: f64, rng: &mut Rng) -> Block {
    let mut w_in = Dense::zeros(0, d);
    let mut w_val = Dense::zeros(0, d);
    for _ in 0..width {
        w_in.push_row(&random_row(rng, d, scale), 0.0);
        w_val.push_row(&random_row(rng, d, scale), 1.0);
    }
    Block { w_in, w_val: gated.then_some(w_val), w_out: Dense::zeros(d, width) }
}

fn ln_gain(anchor: f64, d: usize, eps: f64) -> f64 {
    (2.0 * anchor * anchor / d as f64 + eps).sqrt()
}

/// Rescale every hidden unit so its largest activation on the sample inputs
/// `xs` (flat, raw) is O(1): incoming weights / 2^j, outgoing * 2^j. Units are
/// only scaled down. Function-preserving and exact (powers of two). Large
/// activations (big-M constants, sentinel values) otherwise make every
/// gradient step on their outgoing weights huge. Returns false if the
/// architecture is not supported (transformer).
pub fn normalize_activations(net: &mut Network, xs: &[f64]) -> bool {
    let samples: Vec<Vec<f64>> = if net.n_in == 0 {
        vec![vec![]]
    } else {
        xs.chunks(net.n_in).map(|x| net.normalise_input(x)).collect()
    };
    let scale_of = |m: f64| pow2(m.max(1.0));
    match &mut net.body {
        Body::Mlp { hidden, out } => {
            let mut maxes: Vec<Vec<f64>> = hidden.iter().map(|d| vec![0.0; d.rows]).collect();
            for u in &samples {
                let mut h = u.clone();
                for (l, d) in hidden.iter().enumerate() {
                    h = d.apply(&h).into_iter().map(|v| v.max(0.0)).collect();
                    for (m, v) in maxes[l].iter_mut().zip(&h) {
                        *m = m.max(*v);
                    }
                }
            }
            for l in 0..hidden.len() {
                for i in 0..hidden[l].rows {
                    let k = scale_of(maxes[l][i]);
                    if k != 1.0 {
                        hidden[l].scale_row(i, 1.0 / k);
                        let nxt: &mut Dense = if l + 1 < hidden.len() { &mut hidden[l + 1] } else { &mut *out };
                        nxt.scale_col(i, k);
                    }
                }
            }
            true
        }
        Body::Residual { kind, embed, blocks, .. } => {
            let gated = *kind == BlockKind::Gated;
            let mut ma: Vec<Vec<f64>> = blocks.iter().map(|b| vec![0.0; b.hidden()]).collect();
            let mut mb: Vec<Vec<f64>> = ma.clone();
            for u in &samples {
                let mut st = embed.apply(u);
                for (j, b) in blocks.iter().enumerate() {
                    let a: Vec<f64> = b.w_in.apply(&st).into_iter().map(|v| v.max(0.0)).collect();
                    let h: Vec<f64> = match &b.w_val {
                        Some(wv) if gated => {
                            let bv = wv.apply(&st);
                            for (m, v) in mb[j].iter_mut().zip(&bv) {
                                *m = m.max(v.abs());
                            }
                            a.iter().zip(&bv).map(|(p, q)| p * q).collect()
                        }
                        _ => a.clone(),
                    };
                    for (m, v) in ma[j].iter_mut().zip(&a) {
                        *m = m.max(*v);
                    }
                    for (x, d) in st.iter_mut().zip(b.w_out.apply(&h)) {
                        *x += d;
                    }
                }
            }
            for (j, b) in blocks.iter_mut().enumerate() {
                for i in 0..b.w_in.rows {
                    let ka = scale_of(ma[j][i]);
                    let kb = if gated { scale_of(mb[j][i]) } else { 1.0 };
                    if ka != 1.0 {
                        b.w_in.scale_row(i, 1.0 / ka);
                    }
                    if kb != 1.0 {
                        if let Some(wv) = &mut b.w_val {
                            wv.scale_row(i, 1.0 / kb);
                        }
                    }
                    if ka * kb != 1.0 {
                        b.w_out.scale_col(i, ka * kb);
                    }
                }
            }
            true
        }
        Body::Transformer { .. } => false,
    }
}

/// Equalise in/out norms of every hidden unit, `sweeps` passes.
pub fn balance(net: &mut Network, sweeps: usize) {
    for _ in 0..sweeps {
        match &mut net.body {
            Body::Mlp { hidden, out } => {
                for l in 0..hidden.len() {
                    let (left, right) = hidden.split_at_mut(l + 1);
                    let cur = &mut left[l];
                    let nxt: &mut Dense = if right.is_empty() { &mut *out } else { &mut right[0] };
                    for u in 0..cur.rows {
                        let ni = cur.row_norm(u, true);
                        let no = nxt.col_norm(u);
                        if ni > 0.0 && no > 0.0 {
                            let a = pow2((no / ni).sqrt());
                            cur.scale_row(u, a);
                            nxt.scale_col(u, 1.0 / a);
                        }
                    }
                }
            }
            Body::Residual { blocks, .. } => blocks.iter_mut().for_each(balance_block),
            Body::Transformer { layers, .. } => {
                layers.iter_mut().filter_map(|l| l.mlp.as_mut()).for_each(balance_block)
            }
        }
    }
}

/// Add `extra` spare units to every hidden layer / block, and (residual only)
/// `extra_dims` spare stream dimensions. Spare units have random incoming
/// weights (std = scale / sqrt(fan_in)) and zero outgoing weights.
pub fn widen(net: &mut Network, extra: usize, extra_dims: usize, scale: f64, seed: u64) {
    let mut rng = Rng::new(seed);
    match &mut net.body {
        Body::Mlp { hidden, out } => {
            for l in 0..hidden.len() {
                let cols = hidden[l].cols;
                for _ in 0..extra {
                    let row = random_row(&mut rng, cols, scale);
                    hidden[l].push_row(&row, 0.0);
                }
                let nxt: &mut Dense = if l + 1 < hidden.len() { &mut hidden[l + 1] } else { &mut *out };
                nxt.push_cols(extra, |_, _| 0.0);
            }
        }
        Body::Transformer { tok_embed, layers, ln_f, readout, anchor, .. } => {
            let d = readout.cols;
            let d_new = d + extra_dims;
            for e in tok_embed.iter_mut() {
                for _ in 0..extra_dims {
                    e.push_row(&vec![0.0; e.cols], 0.0);
                }
            }
            readout.push_cols(extra_dims, |_, _| 0.0);
            let grow_ln = |l: &mut Option<LayerNorm>| {
                if let Some(l) = l {
                    l.gain.resize(d_new, 0.0);
                    l.bias.resize(d_new, 0.0);
                    let g = ln_gain(*anchor, d_new, l.eps);
                    if *anchor > 0.0 {
                        l.gain.iter_mut().for_each(|x| *x = g);
                    }
                }
            };
            grow_ln(ln_f);
            for l in layers.iter_mut() {
                grow_ln(&mut l.ln1);
                grow_ln(&mut l.ln2);
                if let Some(at) = &mut l.attn {
                    at.w_q.push_cols(extra_dims, |_, _| 0.0);
                    at.w_k.push_cols(extra_dims, |_, _| 0.0);
                    at.w_v.push_cols(extra_dims, |_, _| 0.0);
                    for _ in 0..extra_dims {
                        at.w_o.push_row(&vec![0.0; at.w_o.cols], 0.0);
                    }
                }
                if let Some(b) = &mut l.mlp {
                    let rows = b.w_in.rows;
                    b.w_in.push_cols(extra_dims, |_, _| 0.0);
                    for r in 0..rows {
                        for j in 0..extra_dims {
                            *b.w_in.at_mut(r, d + j) = scale / (d_new as f64).sqrt() * rng.normal();
                        }
                    }
                    if let Some(wv) = &mut b.w_val {
                        wv.push_cols(extra_dims, |_, _| 0.0);
                    }
                    for _ in 0..extra_dims {
                        b.w_out.push_row(&vec![0.0; b.w_out.cols], 0.0);
                    }
                    widen_block(b, extra, d_new, scale, &mut rng);
                }
            }
        }
        Body::Residual { embed, blocks, readout, .. } => {
            let d_new = embed.rows + extra_dims;
            // spare stream dims: never written, so they stay 0 and reading
            // them (with random weights) contributes nothing
            for _ in 0..extra_dims {
                embed.push_row(&vec![0.0; embed.cols], 0.0);
            }
            readout.push_cols(extra_dims, |_, _| 0.0);
            for b in blocks.iter_mut() {
                let rows_in = b.w_in.rows;
                b.w_in.push_cols(extra_dims, |_, _| 0.0);
                for r in 0..rows_in {
                    for j in 0..extra_dims {
                        let c = d_new - extra_dims + j;
                        *b.w_in.at_mut(r, c) = scale / (d_new as f64).sqrt() * rng.normal();
                    }
                }
                if let Some(wv) = &mut b.w_val {
                    wv.push_cols(extra_dims, |_, _| 0.0);
                }
                for _ in 0..extra_dims {
                    b.w_out.push_row(&vec![0.0; b.w_out.cols], 0.0);
                }
                widen_block(b, extra, d_new, scale, &mut rng);
            }
        }
    }
}

/// Insert `n` function-preserving layers, spread evenly through the network.
/// Residual nets get blocks with `width` random-input / zero-output units.
pub fn deepen(net: &mut Network, n: usize, width: usize, scale: f64, seed: u64) {
    if n == 0 {
        return;
    }
    let mut rng = Rng::new(seed);
    match &mut net.body {
        Body::Mlp { hidden, out } => {
            if hidden.is_empty() {
                // pair layer: x = relu(x) - relu(-x)
                let k = out.cols;
                let mut first = Dense::zeros(0, k);
                for i in 0..k {
                    let mut row = vec![0.0; k];
                    row[i] = 1.0;
                    first.push_row(&row, 0.0);
                    row[i] = -1.0;
                    first.push_row(&row, 0.0);
                }
                let mut o2 = Dense::zeros(out.rows, 2 * k);
                for r in 0..out.rows {
                    for i in 0..k {
                        *o2.at_mut(r, 2 * i) = out.at(r, i);
                        *o2.at_mut(r, 2 * i + 1) = -out.at(r, i);
                    }
                }
                o2.b = out.b.clone();
                *out = o2;
                hidden.push(first);
                if n == 1 {
                    return;
                }
                return deepen(net, n - 1, width, scale, seed);
            }
            let base = hidden.len();
            for j in 0..n {
                // insert after layer index pos (in the original numbering)
                let pos = ((j + 1) * base) / (n + 1);
                // never before hidden[0]: identity layers need non-negative inputs
                let at = (pos.max(1) + j).min(hidden.len());
                let w = hidden[at - 1].rows;
                hidden.insert(at, Dense::identity(w));
            }
        }
        Body::Transformer { kind, layers, ln_f, readout, .. } => {
            let d = readout.cols;
            let base = layers.len();
            let gated = *kind == BlockKind::Gated;
            for j in 0..n {
                let mlp = random_block(d, width, gated, scale, &mut rng);
                let layer = TxLayer { ln1: None, attn: None, ln2: ln_f.clone(), mlp: Some(mlp) };
                let pos = ((j + 1) * base) / (n + 1);
                layers.insert((pos + j).min(layers.len()), layer);
            }
        }
        Body::Residual { embed, blocks, .. } => {
            let d = embed.rows;
            let base = blocks.len();
            let gated = blocks.first().map_or(false, |b| b.w_val.is_some())
                || matches!(net.arch, crate::lower::Arch::Glu);
            for j in 0..n {
                let mut w_in = Dense::zeros(0, d);
                let mut w_val = Dense::zeros(0, d);
                for _ in 0..width {
                    w_in.push_row(&random_row(&mut rng, d, scale), 0.0);
                    w_val.push_row(&random_row(&mut rng, d, scale), 1.0);
                }
                let blk = Block { w_in, w_val: gated.then_some(w_val), w_out: Dense::zeros(d, width) };
                let pos = ((j + 1) * base) / (n + 1);
                blocks.insert((pos + j).min(blocks.len()), blk);
            }
        }
    }
}
