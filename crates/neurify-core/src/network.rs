//! Concrete network architectures produced by the assembler.

use crate::lower::{Arch, AttnMode};

/// Dense affine map y = W x + b, W row-major (rows x cols).
#[derive(Clone, Debug, PartialEq)]
pub struct Dense {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<f64>,
    pub b: Vec<f64>,
}

impl Dense {
    pub fn zeros(rows: usize, cols: usize) -> Dense {
        Dense { rows, cols, w: vec![0.0; rows * cols], b: vec![0.0; rows] }
    }
    pub fn identity(n: usize) -> Dense {
        let mut d = Dense::zeros(n, n);
        for i in 0..n {
            d.w[i * n + i] = 1.0;
        }
        d
    }
    #[inline]
    pub fn at(&self, r: usize, c: usize) -> f64 {
        self.w[r * self.cols + c]
    }
    #[inline]
    pub fn at_mut(&mut self, r: usize, c: usize) -> &mut f64 {
        &mut self.w[r * self.cols + c]
    }
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.cols);
        (0..self.rows)
            .map(|r| {
                let row = &self.w[r * self.cols..(r + 1) * self.cols];
                row.iter().zip(x).fold(self.b[r], |acc, (w, v)| acc + w * v)
            })
            .collect()
    }
    pub fn params(&self) -> usize {
        self.rows * self.cols + self.rows
    }
    /// non-zero weights and biases
    pub fn nnz(&self) -> usize {
        self.w.iter().chain(&self.b).filter(|&&x| x != 0.0).count()
    }
    pub fn push_row(&mut self, row: &[f64], bias: f64) {
        assert_eq!(row.len(), self.cols);
        self.w.extend_from_slice(row);
        self.b.push(bias);
        self.rows += 1;
    }
    /// Append `n` columns, filled by `f(row, new_col_index)`.
    pub fn push_cols(&mut self, n: usize, mut f: impl FnMut(usize, usize) -> f64) {
        let nc = self.cols + n;
        let mut w = Vec::with_capacity(self.rows * nc);
        for r in 0..self.rows {
            w.extend_from_slice(&self.w[r * self.cols..(r + 1) * self.cols]);
            for j in 0..n {
                w.push(f(r, j));
            }
        }
        self.w = w;
        self.cols = nc;
    }
    pub fn row_norm(&self, r: usize, with_bias: bool) -> f64 {
        let s: f64 = self.w[r * self.cols..(r + 1) * self.cols].iter().map(|v| v * v).sum();
        let b = if with_bias { self.b[r] * self.b[r] } else { 0.0 };
        (s + b).sqrt()
    }
    pub fn col_norm(&self, c: usize) -> f64 {
        (0..self.rows).map(|r| self.at(r, c).powi(2)).sum::<f64>().sqrt()
    }
    pub fn scale_row(&mut self, r: usize, k: f64) {
        for v in &mut self.w[r * self.cols..(r + 1) * self.cols] {
            *v *= k;
        }
        self.b[r] *= k;
    }
    pub fn scale_col(&mut self, c: usize, k: f64) {
        for r in 0..self.rows {
            *self.at_mut(r, c) *= k;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    /// x += W_out relu(W_in x + b_in) + b_out
    Relu,
    /// x += W_out (relu(W_in x + b_in) * (W_val x + b_val)) + b_out
    Gated,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub w_in: Dense,
    pub w_val: Option<Dense>,
    /// d_model x hidden
    pub w_out: Dense,
}

impl Block {
    pub fn hidden(&self) -> usize {
        self.w_in.rows
    }
    pub fn params(&self) -> usize {
        self.w_in.params() + self.w_val.as_ref().map_or(0, |d| d.params()) + self.w_out.params()
    }
}

/// Multi-head attention with d_v = 1 per head: head h reads q/k rows
/// [h*d_k, (h+1)*d_k), value row h, and writes column h of w_o.
#[derive(Clone, Debug)]
pub struct Attention {
    pub mode: AttnMode,
    pub heads: usize,
    pub d_k: usize,
    pub w_q: Dense,
    pub w_k: Dense,
    pub w_v: Dense,
    /// d_model x heads
    pub w_o: Dense,
}

#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub gain: Vec<f64>,
    pub bias: Vec<f64>,
    pub eps: f64,
}

impl LayerNorm {
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let r = 1.0 / (var + self.eps).sqrt();
        x.iter().enumerate().map(|(i, v)| self.gain[i] * (v - mean) * r + self.bias[i]).collect()
    }
}

#[derive(Clone, Debug)]
pub struct TxLayer {
    pub ln1: Option<LayerNorm>,
    pub attn: Option<Attention>,
    pub ln2: Option<LayerNorm>,
    pub mlp: Option<Block>,
}

#[derive(Clone, Debug)]
pub enum Body {
    /// relu(hidden[0]) -> ... -> relu(hidden[L-1]) -> out
    Mlp { hidden: Vec<Dense>, out: Dense },
    /// embed -> blocks (residual) -> readout
    Residual { kind: BlockKind, embed: Dense, blocks: Vec<Block>, readout: Dense },
    /// tokens t = 0..T: s_t = tok_embed[t] u; pre-LN blocks (attention, MLP);
    /// output = readout(ln_f(s_0)). `anchor` is the LayerNorm anchor value.
    Transformer {
        kind: BlockKind,
        tok_embed: Vec<Dense>,
        layers: Vec<TxLayer>,
        ln_f: Option<LayerNorm>,
        readout: Dense,
        anchor: f64,
    },
}

fn mlp_block(kind: BlockKind, b: &Block, x: &[f64]) -> Vec<f64> {
    let a: Vec<f64> = b.w_in.apply(x).into_iter().map(|v| v.max(0.0)).collect();
    let h: Vec<f64> = match (kind, &b.w_val) {
        (BlockKind::Gated, Some(wv)) => a.iter().zip(wv.apply(x)).map(|(p, q)| p * q).collect(),
        _ => a,
    };
    b.w_out.apply(&h)
}

/// Attention output for every token (d_model-dim update per token).
pub fn attention(at: &Attention, xs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let q: Vec<Vec<f64>> = xs.iter().map(|x| at.w_q.apply(x)).collect();
    let k: Vec<Vec<f64>> = xs.iter().map(|x| at.w_k.apply(x)).collect();
    let v: Vec<Vec<f64>> = xs.iter().map(|x| at.w_v.apply(x)).collect();
    let t = xs.len();
    let dk = at.d_k;
    let mut outs = vec![];
    for i in 0..t {
        let mut o = vec![0.0; at.heads];
        for h in 0..at.heads {
            let sc: Vec<f64> = (0..t)
                .map(|j| (0..dk).map(|c| q[i][h * dk + c] * k[j][h * dk + c]).sum::<f64>())
                .collect();
            let w: Vec<f64> = match at.mode {
                AttnMode::Relu => sc.iter().map(|s| s.max(0.0)).collect(),
                AttnMode::Hard => {
                    let m = (0..t).fold(0, |b, j| if sc[j] > sc[b] { j } else { b });
                    (0..t).map(|j| if j == m { 1.0 } else { 0.0 }).collect()
                }
                AttnMode::Softmax => {
                    let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
                    let z: f64 = e.iter().sum();
                    e.into_iter().map(|x| x / z).collect()
                }
            };
            o[h] = (0..t).map(|j| w[j] * v[j][h]).sum();
        }
        outs.push(at.w_o.apply(&o));
    }
    outs
}

/// A fixed (parameter-free) input feature: a raw input, or the indicator
/// `[round(x_i) == v]` (one-hot encoding of a small integer domain).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Feature {
    Raw(usize),
    Eq(usize, f64),
}

impl Feature {
    pub fn value(&self, x: &[f64]) -> f64 {
        match *self {
            Feature::Raw(i) => x[i],
            Feature::Eq(i, v) => (x[i].round() == v) as u8 as f64,
        }
    }
}

pub fn featurize(features: &[Feature], x: &[f64]) -> Vec<f64> {
    features.iter().map(|f| f.value(x)).collect()
}

#[derive(Clone, Debug)]
pub struct Network {
    pub arch: Arch,
    /// raw inputs
    pub n_in: usize,
    pub n_out: usize,
    /// fixed input features the weights read (one per first-layer column)
    pub features: Vec<Feature>,
    /// declared domain of every feature (used by normalisation / deepening)
    pub input_bounds: Vec<(f64, f64)>,
    /// the weights consume u = (feature - in_offset) / in_scale
    pub in_offset: Vec<f64>,
    pub in_scale: Vec<f64>,
    pub body: Body,
}

impl Network {
    pub fn normalise_input(&self, x: &[f64]) -> Vec<f64> {
        self.features
            .iter()
            .enumerate()
            .map(|(j, f)| (f.value(x) - self.in_offset[j]) / self.in_scale[j])
            .collect()
    }

    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    /// Forward pass on raw (un-normalised) inputs.
    ///
    /// Panics if `x` does not contain exactly `n_in` values.
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.n_in, "expected exactly n_in input values");
        let u = self.normalise_input(x);
        match &self.body {
            Body::Mlp { hidden, out } => {
                let mut h = u;
                for d in hidden {
                    h = d.apply(&h).into_iter().map(|v| v.max(0.0)).collect();
                }
                out.apply(&h)
            }
            Body::Residual { kind, embed, blocks, readout } => {
                let mut s = embed.apply(&u);
                for b in blocks {
                    let d = mlp_block(*kind, b, &s);
                    for (si, x) in s.iter_mut().zip(d) {
                        *si += x;
                    }
                }
                readout.apply(&s)
            }
            Body::Transformer { kind, tok_embed, layers, ln_f, readout, .. } => {
                let mut s: Vec<Vec<f64>> = tok_embed.iter().map(|e| e.apply(&u)).collect();
                let ln = |l: &Option<LayerNorm>, x: &Vec<f64>| l.as_ref().map_or_else(|| x.clone(), |l| l.apply(x));
                for l in layers {
                    if let Some(at) = &l.attn {
                        let xs: Vec<Vec<f64>> = s.iter().map(|x| ln(&l.ln1, x)).collect();
                        for (st, d) in s.iter_mut().zip(attention(at, &xs)) {
                            for (a, b) in st.iter_mut().zip(d) {
                                *a += b;
                            }
                        }
                    }
                    if let Some(b) = &l.mlp {
                        for st in s.iter_mut() {
                            let d = mlp_block(*kind, b, &ln(&l.ln2, st));
                            for (a, b) in st.iter_mut().zip(d) {
                                *a += b;
                            }
                        }
                    }
                }
                readout.apply(&ln(ln_f, &s[0]))
            }
        }
    }

    /// Evaluate complete rows of raw inputs. An empty input evaluates a
    /// zero-input (constant) network once.
    ///
    /// Panics if `xs` does not contain a whole number of input rows, or if
    /// values are supplied to a zero-input network.
    pub fn forward_batch(&self, xs: &[f64]) -> Vec<f64> {
        if self.n_in == 0 {
            assert!(xs.is_empty(), "a zero-input network expects no input values");
            return self.forward(&[]);
        }
        assert_eq!(xs.len() % self.n_in, 0, "batch must contain complete input rows");
        xs.chunks(self.n_in).flat_map(|x| self.forward(x)).collect()
    }

    pub fn params(&self) -> usize {
        match &self.body {
            Body::Mlp { hidden, out } => hidden.iter().map(|d| d.params()).sum::<usize>() + out.params(),
            Body::Residual { embed, blocks, readout, .. } => {
                embed.params() + blocks.iter().map(|b| b.params()).sum::<usize>() + readout.params()
            }
            Body::Transformer { tok_embed, layers, ln_f, readout, .. } => {
                let lnp = |l: &Option<LayerNorm>| l.as_ref().map_or(0, |l| 2 * l.gain.len());
                tok_embed.iter().map(|d| d.params()).sum::<usize>()
                    + layers
                        .iter()
                        .map(|l| {
                            lnp(&l.ln1)
                                + lnp(&l.ln2)
                                + l.attn.as_ref().map_or(0, |a| {
                                    a.w_q.params() + a.w_k.params() + a.w_v.params() + a.w_o.params()
                                })
                                + l.mlp.as_ref().map_or(0, |b| b.params())
                        })
                        .sum::<usize>()
                    + lnp(ln_f)
                    + readout.params()
            }
        }
    }

    /// Number of nonlinear layers (hidden layers / residual blocks).
    pub fn depth(&self) -> usize {
        match &self.body {
            Body::Mlp { hidden, .. } => hidden.len(),
            Body::Residual { blocks, .. } => blocks.len(),
            Body::Transformer { layers, .. } => layers.len(),
        }
    }

    /// Hidden widths per nonlinear layer.
    pub fn widths(&self) -> Vec<usize> {
        match &self.body {
            Body::Mlp { hidden, .. } => hidden.iter().map(|d| d.rows).collect(),
            Body::Residual { blocks, .. } => blocks.iter().map(|b| b.hidden()).collect(),
            Body::Transformer { layers, .. } => layers.iter().map(|l| l.mlp.as_ref().map_or(0, |b| b.hidden())).collect(),
        }
    }

    /// Attention heads per transformer layer.
    pub fn heads(&self) -> Vec<usize> {
        match &self.body {
            Body::Transformer { layers, .. } => layers.iter().map(|l| l.attn.as_ref().map_or(0, |a| a.heads)).collect(),
            _ => vec![],
        }
    }

    pub fn n_tokens(&self) -> usize {
        match &self.body {
            Body::Transformer { tok_embed, .. } => tok_embed.len(),
            _ => 1,
        }
    }

    pub fn d_model(&self) -> Option<usize> {
        match &self.body {
            Body::Residual { embed, .. } => Some(embed.rows),
            Body::Transformer { readout, .. } => Some(readout.cols),
            _ => None,
        }
    }
}
