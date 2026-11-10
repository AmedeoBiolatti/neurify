//! Assembly: neuron DAG -> layered network.
//!
//! * Units are scheduled into layers (ASAP or ALAP; the cheaper is kept).
//! * `mlp`: a value that must survive several layers is *carried* with one
//!   ReLU unit per layer using its lower bound, relu(v - lo) + lo (two units
//!   only if it is unbounded on both sides).
//! * `resmlp` / `glu`: every live signal owns a slot of the residual stream;
//!   blocks read the stream and write their units into their slots, so no
//!   carries are needed.

use crate::error::{err, ErrorKind, Result};
use crate::lower::{Arch, AttnMode, Lowered, Options};
use crate::network::*;
use crate::neuron::*;

struct Sched {
    layer: Vec<usize>,
    depth: usize,
}

fn live_sigs(dag: &Dag, outputs: &[Lin]) -> Vec<bool> {
    let mut live = vec![false; dag.sigs.len()];
    for o in outputs {
        for &(s, _) in &o.terms {
            live[s as usize] = true;
        }
    }
    for s in (dag.n_inputs..dag.sigs.len()).rev() {
        if live[s] {
            for t in dag.sigs[s].kind.deps() {
                live[t as usize] = true;
            }
        }
    }
    live
}

fn unit_args(dag: &Dag, s: usize) -> Vec<Sig> {
    dag.sigs[s].kind.deps()
}

fn asap(dag: &Dag, live: &[bool]) -> Sched {
    let mut layer = vec![0usize; dag.sigs.len()];
    let mut depth = 0;
    for s in dag.n_inputs..dag.sigs.len() {
        if !live[s] {
            continue;
        }
        let l = 1 + unit_args(dag, s).iter().map(|&t| layer[t as usize]).max().unwrap_or(0);
        layer[s] = l;
        depth = depth.max(l);
    }
    Sched { layer, depth }
}

fn alap(dag: &Dag, live: &[bool], outputs: &[Lin], depth: usize) -> Sched {
    let n = dag.sigs.len();
    let mut layer = vec![depth + 1; n];
    let mut latest = vec![depth + 1; n]; // min consumer layer
    for o in outputs {
        for &(s, _) in &o.terms {
            latest[s as usize] = latest[s as usize].min(depth + 1);
        }
    }
    for s in (dag.n_inputs..n).rev() {
        if !live[s] {
            continue;
        }
        layer[s] = latest[s] - 1;
        for t in unit_args(dag, s) {
            let t = t as usize;
            latest[t] = latest[t].min(layer[s]);
        }
    }
    for l in layer.iter_mut().take(dag.n_inputs) {
        *l = 0;
    }
    Sched { layer, depth }
}

/// Representation of a signal at some layer: affine form over that layer's units.
#[derive(Clone)]
struct Rep {
    terms: Vec<(usize, f64)>,
    c: f64,
}

fn subst(l: &Lin, rep: &[Option<Rep>], dim: usize) -> (Vec<f64>, f64) {
    let mut row = vec![0.0; dim];
    let mut c = l.c;
    for &(s, w) in &l.terms {
        let r = rep[s as usize].as_ref().expect("signal not available at this layer");
        c += w * r.c;
        for &(u, k) in &r.terms {
            row[u] += w * k;
        }
    }
    (row, c)
}

fn assemble_mlp(dag: &Dag, outputs: &[Lin], live: &[bool], sch: &Sched) -> (Vec<Dense>, Dense) {
    let n = dag.sigs.len();
    let d = sch.depth;
    // last layer at which each signal must be readable (consumers read layer-1)
    let mut need = vec![0usize; n];
    for s in dag.n_inputs..n {
        if live[s] {
            for t in unit_args(dag, s) {
                need[t as usize] = need[t as usize].max(sch.layer[s] - 1);
            }
        }
    }
    for o in outputs {
        for &(s, _) in &o.terms {
            need[s as usize] = d;
        }
    }
    let mut rep: Vec<Option<Rep>> = vec![None; n];
    for (i, r) in rep.iter_mut().enumerate().take(dag.n_inputs) {
        *r = Some(Rep { terms: vec![(i, 1.0)], c: 0.0 });
    }
    let mut dim = dag.n_inputs;
    let mut hidden = vec![];
    for k in 1..=d {
        let mut dense = Dense::zeros(0, dim);
        let mut next: Vec<Option<Rep>> = vec![None; n];
        for s in dag.n_inputs..n {
            if !live[s] || sch.layer[s] != k {
                continue;
            }
            let SigKind::Unit(Unit::Relu(a)) = &dag.sigs[s].kind else {
                panic!("gated units cannot be assembled into a plain MLP");
            };
            let (row, c) = subst(a, &rep, dim);
            dense.push_row(&row, c);
            next[s] = Some(Rep { terms: vec![(dense.rows - 1, 1.0)], c: 0.0 });
        }
        for s in 0..n {
            let is_live = live[s] || s < dag.n_inputs;
            if !is_live || sch.layer[s] >= k || need[s] < k || rep[s].is_none() {
                continue;
            }
            // carry s through layer k
            let one = Lin { terms: vec![(s as Sig, 1.0)], c: 0.0 };
            let (row, c) = subst(&one, &rep, dim);
            let (lo, hi) = (dag.sigs[s].lo, dag.sigs[s].hi);
            if lo.is_finite() {
                dense.push_row(&row, c - lo);
                next[s] = Some(Rep { terms: vec![(dense.rows - 1, 1.0)], c: lo });
            } else if hi.is_finite() {
                let neg: Vec<f64> = row.iter().map(|v| -v).collect();
                dense.push_row(&neg, hi - c);
                next[s] = Some(Rep { terms: vec![(dense.rows - 1, -1.0)], c: hi });
            } else {
                let neg: Vec<f64> = row.iter().map(|v| -v).collect();
                dense.push_row(&row, c);
                dense.push_row(&neg, -c);
                let p = dense.rows - 2;
                next[s] = Some(Rep { terms: vec![(p, 1.0), (p + 1, -1.0)], c: 0.0 });
            }
        }
        dim = dense.rows;
        hidden.push(dense);
        rep = next;
    }
    let mut out = Dense::zeros(0, dim);
    for o in outputs {
        let (row, c) = subst(o, &rep, dim);
        out.push_row(&row, c);
    }
    (hidden, out)
}

fn assemble_residual(dag: &Dag, outputs: &[Lin], live: &[bool], sch: &Sched, gated: bool) -> Body {
    let n = dag.sigs.len();
    let mut slot = vec![usize::MAX; n];
    let mut d_model = 0;
    for s in 0..n {
        if s < dag.n_inputs || live[s] {
            slot[s] = d_model;
            d_model += 1;
        }
    }
    let rep: Vec<Option<Rep>> =
        (0..n).map(|s| (slot[s] != usize::MAX).then(|| Rep { terms: vec![(slot[s], 1.0)], c: 0.0 })).collect();
    let mut embed = Dense::zeros(d_model, dag.n_inputs);
    for i in 0..dag.n_inputs {
        *embed.at_mut(slot[i], i) = 1.0;
    }
    let mut blocks = vec![];
    for k in 1..=sch.depth {
        let mut w_in = Dense::zeros(0, d_model);
        let mut w_val = Dense::zeros(0, d_model);
        let mut targets = vec![];
        for s in dag.n_inputs..n {
            if !live[s] || sch.layer[s] != k {
                continue;
            }
            match &dag.sigs[s].kind {
                SigKind::Unit(Unit::Relu(a)) => {
                    let (row, c) = subst(a, &rep, d_model);
                    w_in.push_row(&row, c);
                    w_val.push_row(&vec![0.0; d_model], 1.0);
                }
                SigKind::Unit(Unit::Gated(a, b)) => {
                    assert!(gated, "gated unit in a relu-only architecture");
                    let (row, c) = subst(a, &rep, d_model);
                    w_in.push_row(&row, c);
                    let (row, c) = subst(b, &rep, d_model);
                    w_val.push_row(&row, c);
                }
                SigKind::Input(_) | SigKind::Attn(_) => unreachable!("attention outside a transformer"),
            }
            targets.push(slot[s]);
        }
        let mut w_out = Dense::zeros(d_model, targets.len());
        for (j, &t) in targets.iter().enumerate() {
            *w_out.at_mut(t, j) = 1.0;
        }
        blocks.push(Block { w_in, w_val: gated.then_some(w_val), w_out });
    }
    let mut readout = Dense::zeros(0, d_model);
    for o in outputs {
        let (row, c) = subst(o, &rep, d_model);
        readout.push_row(&row, c);
    }
    Body::Residual { kind: if gated { BlockKind::Gated } else { BlockKind::Relu }, embed, blocks, readout }
}

struct Layout {
    d_c: usize,
    ln: bool,
}

impl Layout {
    fn d(&self) -> usize {
        if self.ln {
            2 * self.d_c + 2
        } else {
            self.d_c
        }
    }
    /// write v into content slot `slot` of column `col` (and -v into its mirror)
    fn put(&self, m: &mut Dense, slot: usize, col: usize, v: f64) {
        *m.at_mut(slot, col) += v;
        if self.ln {
            *m.at_mut(self.d_c + slot, col) -= v;
        }
    }
    fn put_bias(&self, m: &mut Dense, slot: usize, v: f64) {
        m.b[slot] += v;
        if self.ln {
            m.b[self.d_c + slot] -= v;
        }
    }
}

fn next_even(m: usize) -> usize {
    m + 2 - (m % 2)
}

fn next_odd(m: usize) -> usize {
    m + 1 + (m % 2)
}

fn bound_dot(row: &[f64], bias: f64, b: &[(f64, f64)]) -> f64 {
    row.iter().zip(b).fold(bias, |acc, (&w, &(lo, hi))| if w == 0.0 { acc } else { acc + (w * lo).max(w * hi) })
}

/// Transformer assembly. Returns (body, approximation notes).
fn assemble_transformer(dag: &Dag, outputs: &[Lin], live: &[bool], o: &Options) -> Result<(Body, Vec<String>)> {
    let n = dag.sigs.len();
    let n_in = dag.n_inputs;
    let mut token_of: Vec<Option<(usize, usize)>> = vec![None; n_in];
    for (&(f, p), &i) in &dag.tokens {
        token_of[i] = Some((f, p));
    }
    // half-step times: 0 = embedding, 2b+1 = attention of layer b, 2b+2 = MLP of layer b
    let mut t = vec![0usize; n];
    let mut max_t = 0;
    for s in n_in..n {
        if !live[s] {
            continue;
        }
        let m = dag.sigs[s].kind.deps().iter().map(|&d| t[d as usize]).max().unwrap_or(0);
        t[s] = match dag.sigs[s].kind {
            SigKind::Attn(_) => next_odd(m),
            _ => next_even(m),
        };
        max_t = max_t.max(t[s]);
    }
    for (i, tok) in token_of.iter().enumerate() {
        if live[i] && tok.is_some() {
            return err(ErrorKind::Invalid, "internal: sequence input used directly in a transformer");
        }
    }
    let n_layers = (max_t + 1) / 2;
    let n_pos = dag.tokens.keys().map(|k| k.1 + 1).max().unwrap_or(0);
    let n_feat = dag.tokens.keys().map(|k| k.0 + 1).max().unwrap_or(0);
    let mut fb = vec![(0.0f64, 0.0f64); n_feat];
    let mut seen = vec![false; n_feat];
    for (&(f, _), &i) in &dag.tokens {
        let (lo, hi) = (dag.sigs[i].lo, dag.sigs[i].hi);
        fb[f] = if seen[f] { (fb[f].0.min(lo), fb[f].1.max(hi)) } else { (lo, hi) };
        seen[f] = true;
    }

    // content slots
    let mut slot = vec![usize::MAX; n];
    let mut d_c = 0;
    for s in 0..n {
        let main_input = s < n_in && token_of[s].is_none();
        if main_input || (s >= n_in && live[s]) {
            slot[s] = d_c;
            d_c += 1;
        }
    }
    let feat_slot = |f: usize| d_c + f;
    let (pos_s, pos2_s, main_s, data_s) = (d_c + n_feat, d_c + n_feat + 1, d_c + n_feat + 2, d_c + n_feat + 3);
    let lay = Layout { d_c: d_c + n_feat + 4, ln: o.layernorm };
    let d = lay.d();
    let rep: Vec<Option<Rep>> =
        (0..n).map(|s| (slot[s] != usize::MAX).then(|| Rep { terms: vec![(slot[s], 1.0)], c: 0.0 })).collect();

    // bounds of every content slot on data tokens (MLP is gated off there)
    let attn_mode = o.attention;
    let mut data_b = vec![(0.0, 0.0); d];
    for f in 0..n_feat {
        data_b[feat_slot(f)] = fb[f];
    }
    let np = n_pos.max(1) as f64 - 1.0;
    data_b[pos_s] = (0.0, np);
    data_b[pos2_s] = (0.0, np * np);
    data_b[data_s] = (1.0, 1.0);
    for s in n_in..n {
        if live[s] {
            if let SigKind::Attn(a) = &dag.sigs[s].kind {
                if attn_mode != AttnMode::Relu {
                    let (lo, hi) = fb[a.feature];
                    data_b[slot[s]] = (lo.min(0.0), hi.max(0.0));
                }
            }
        }
    }

    // attention scale
    let vmax = fb.iter().fold(1.0f64, |m, &(lo, hi)| m.max(lo.abs()).max(hi.abs()));
    let tol = o.attn_tol;
    let (beta, big_m) = match attn_mode {
        AttnMode::Softmax => {
            let b = (2.0 * n_pos.max(2) as f64 * vmax / tol).ln().max(1.0) + 1.0;
            (b, b + (2.0 * vmax / tol).ln().max(0.0) + 1.0)
        }
        _ => (1.0, 1.0),
    };

    // token embeddings
    let mut tok_embed = vec![];
    let mut e0 = Dense::zeros(d, n_in);
    for i in 0..n_in {
        if token_of[i].is_none() {
            lay.put(&mut e0, slot[i], i, 1.0);
        }
    }
    lay.put_bias(&mut e0, main_s, 1.0);
    tok_embed.push(e0);
    for p in 0..n_pos {
        let mut e = Dense::zeros(d, n_in);
        for (i, tok) in token_of.iter().enumerate() {
            if let Some((f, q)) = tok {
                if *q == p {
                    lay.put(&mut e, feat_slot(*f), i, 1.0);
                }
            }
        }
        lay.put_bias(&mut e, pos_s, p as f64);
        lay.put_bias(&mut e, pos2_s, (p * p) as f64);
        lay.put_bias(&mut e, data_s, 1.0);
        tok_embed.push(e);
    }

    let gated = o.arch == Arch::Transformer && o.tx_glu;
    let mut layers = vec![];
    let mut approx = vec![];
    let mut n_heads = 0;
    for b in 0..n_layers {
        // ---- attention
        let heads: Vec<usize> = (n_in..n).filter(|&s| live[s] && t[s] == 2 * b + 1).collect();
        let attn = if heads.is_empty() {
            None
        } else {
            let h = heads.len();
            n_heads += h;
            let (mut wq, mut wk) = (Dense::zeros(4 * h, d), Dense::zeros(4 * h, d));
            let (mut wv, mut wo) = (Dense::zeros(h, d), Dense::zeros(d, h));
            for (j, &s) in heads.iter().enumerate() {
                let SigKind::Attn(a) = &dag.sigs[s].kind else { unreachable!() };
                let (qrow, qc) = subst(&a.query, &rep, d);
                for (r, ks) in [data_s, pos_s, pos2_s, main_s].iter().enumerate() {
                    *wk.at_mut(4 * j + r, *ks) = 1.0;
                }
                match attn_mode {
                    AttnMode::Softmax | AttnMode::Hard => {
                        for c in 0..d {
                            *wq.at_mut(4 * j + 1, c) = 2.0 * beta * qrow[c];
                        }
                        wq.b[4 * j + 1] = 2.0 * beta * qc;
                        wq.b[4 * j + 2] = -beta;
                        wq.b[4 * j + 3] = -big_m;
                    }
                    AttnMode::Relu => {
                        let (sqrow, sqc) = subst(a.query_sq.as_ref().expect("relu attention needs query^2"), &rep, d);
                        for c in 0..d {
                            *wq.at_mut(4 * j, c) = -sqrow[c];
                            *wq.at_mut(4 * j + 1, c) = 2.0 * qrow[c];
                        }
                        wq.b[4 * j] = 1.0 - sqc;
                        wq.b[4 * j + 1] = 2.0 * qc;
                        wq.b[4 * j + 2] = -1.0;
                        wq.b[4 * j + 3] = -1.0;
                        // queries issued by data tokens score <= 0 everywhere
                        let g = 2.0 + sqc.abs() + 2.0 * qc.abs() * n_pos as f64 + (n_pos * n_pos) as f64;
                        *wq.at_mut(4 * j, data_s) -= g;
                        *wq.at_mut(4 * j + 3, data_s) -= g;
                    }
                }
                *wv.at_mut(j, feat_slot(a.feature)) = 1.0;
                lay.put(&mut wo, slot[s], j, 1.0);
            }
            Some(Attention { mode: attn_mode, heads: h, d_k: 4, w_q: wq, w_k: wk, w_v: wv, w_o: wo })
        };
        // ---- MLP
        let units: Vec<usize> = (n_in..n)
            .filter(|&s| live[s] && t[s] == 2 * b + 2 && matches!(dag.sigs[s].kind, SigKind::Unit(_)))
            .collect();
        let mlp = if units.is_empty() {
            None
        } else {
            let mut w_in = Dense::zeros(0, d);
            let mut w_val = Dense::zeros(0, d);
            for &s in &units {
                let SigKind::Unit(u) = &dag.sigs[s].kind else { unreachable!() };
                let (a, bv) = match u {
                    Unit::Relu(a) => (a, None),
                    Unit::Gated(a, b) => (a, Some(b)),
                };
                let (mut row, c) = subst(a, &rep, d);
                if n_pos > 0 {
                    // gate the unit off on data tokens
                    let k = bound_dot(&row, c, &data_b).max(0.0) + 1.0;
                    row[data_s] -= k;
                }
                w_in.push_row(&row, c);
                match bv {
                    Some(bl) => {
                        let (r2, c2) = subst(bl, &rep, d);
                        w_val.push_row(&r2, c2);
                    }
                    None => w_val.push_row(&vec![0.0; d], 1.0),
                }
            }
            let mut w_out = Dense::zeros(d, units.len());
            for (j, &s) in units.iter().enumerate() {
                lay.put(&mut w_out, slot[s], j, 1.0);
            }
            Some(Block { w_in, w_val: gated.then_some(w_val), w_out })
        };
        layers.push(TxLayer { ln1: None, attn, ln2: None, mlp });
    }
    let mut readout = Dense::zeros(0, d);
    for out in outputs {
        let (row, c) = subst(out, &rep, d);
        readout.push_row(&row, c);
    }
    if attn_mode == AttnMode::Softmax && n_heads > 0 {
        approx.push(format!("softmax attention: each of {n_heads} reads within {tol:e}"));
    }

    // ---- opt-in LayerNorm: mirrored content (mean 0) + anchors (var ~ const)
    let mut anchor = 0.0;
    let mut ln_f = None;
    if o.layernorm {
        let mut c2_main = 1.0; // is_main
        let mut vmax_all = vmax;
        for s in 0..n {
            if slot[s] != usize::MAX {
                let m = dag.sigs[s].lo.abs().max(dag.sigs[s].hi.abs());
                c2_main += m * m;
                vmax_all = vmax_all.max(m);
            }
        }
        let c2_data: f64 = data_b.iter().take(lay.d_c).map(|&(lo, hi)| lo.abs().max(hi.abs()).powi(2)).sum();
        let c2 = c2_main.max(c2_data);
        if !c2.is_finite() {
            return err(ErrorKind::NeedsBounds, "layernorm needs bounded values: declare finite input domains");
        }
        let score_scale = 1.0 + 2.0 * beta * (n_pos * n_pos) as f64 + big_m;
        let tol_eff = o.ln_tol.min(0.01 / score_scale).min(0.1 / vmax_all.max(1.0));
        anchor = (c2 / (2.0 * tol_eff)).sqrt();
        let e_anchor = |e: &mut Dense| {
            e.b[2 * lay.d_c] = anchor;
            e.b[2 * lay.d_c + 1] = -anchor;
        };
        for e in tok_embed.iter_mut() {
            e_anchor(e);
        }
        let eps = 1e-5;
        let g = (2.0 * anchor * anchor / d as f64 + eps).sqrt();
        let ln = LayerNorm { gain: vec![g; d], bias: vec![0.0; d], eps };
        for l in layers.iter_mut() {
            if l.attn.is_some() {
                l.ln1 = Some(ln.clone());
            }
            if l.mlp.is_some() {
                l.ln2 = Some(ln.clone());
            }
        }
        ln_f = Some(ln);
        approx.push(format!("layernorm: reads within relative error {tol_eff:.1e} (anchor {anchor:.3e})"));
    }
    let kind = if gated { BlockKind::Gated } else { BlockKind::Relu };
    Ok((Body::Transformer { kind, tok_embed, layers, ln_f, readout, anchor }, approx))
}

pub fn assemble(low: &Lowered, o: &Options, input_bounds: Vec<(f64, f64)>) -> Result<(Network, Vec<String>)> {
    let arch = o.arch;
    let dag = &low.dag;
    let live = live_sigs(dag, &low.outputs);
    let s1 = asap(dag, &live);
    let o_method_sparse = o.method == crate::lower::Method::Sparse;
    let mut approx = vec![];
    let body = match arch {
        Arch::Mlp => {
            let s2 = alap(dag, &live, &low.outputs, s1.depth);
            let (h1, o1) = assemble_mlp(dag, &low.outputs, &live, &s1);
            let (h2, o2) = assemble_mlp(dag, &low.outputs, &live, &s2);
            let cost = |h: &[Dense], o: &Dense| -> usize {
                if o_method_sparse {
                    h.iter().map(|d| d.nnz()).sum::<usize>() + o.nnz()
                } else {
                    h.iter().map(|d| d.params()).sum::<usize>() + o.params()
                }
            };
            let (p1, p2) = (cost(&h1, &o1), cost(&h2, &o2));
            if p2 < p1 {
                Body::Mlp { hidden: h2, out: o2 }
            } else {
                Body::Mlp { hidden: h1, out: o1 }
            }
        }
        Arch::ResMlp => assemble_residual(dag, &low.outputs, &live, &s1, false),
        Arch::Glu => assemble_residual(dag, &low.outputs, &live, &s1, true),
        Arch::Transformer => {
            let (b, a) = assemble_transformer(dag, &low.outputs, &live, o)?;
            approx = a;
            b
        }
    };
    let nf = dag.n_inputs;
    let feat_bounds = low
        .features
        .iter()
        .map(|f| match *f {
            Feature::Raw(i) => input_bounds[i],
            Feature::Eq(..) => (0.0, 1.0),
        })
        .collect();
    let net = Network {
        arch,
        n_in: input_bounds.len(),
        n_out: low.outputs.len(),
        features: low.features.clone(),
        input_bounds: feat_bounds,
        in_offset: vec![0.0; nf],
        in_scale: vec![1.0; nf],
        body,
    };
    Ok((net, approx))
}

/// Number of live units in the DAG.
pub fn live_units(low: &Lowered) -> usize {
    let live = live_sigs(&low.dag, &low.outputs);
    (low.dag.n_inputs..low.dag.sigs.len()).filter(|&s| live[s]).count()
}
