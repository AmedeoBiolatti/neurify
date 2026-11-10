//! neurify-core: exact compilation of programs into neural networks.
//!
//! Pipeline: [`ir::Graph`] (typed, bounded dataflow) -> [`analysis`] (interval
//! + exhaustive value sets) -> [`lower`] (exact gadgets, neuron DAG) ->
//! [`assemble`] (layered architecture) -> [`train`] (function-preserving
//! transforms for post-training).

pub mod analysis;
pub mod assemble;
pub mod error;
pub mod ir;
pub mod lex;
pub mod lower;
pub mod network;
pub mod neuron;
pub mod train;

pub use error::{CompileError, ErrorKind, Result};
pub use ir::{BinOp, CmpOp, Graph, Kind, NodeId, OutputEnc, UnOp};
pub use lower::{Arch, AttnMode, Method, Options};
pub use network::Network;

#[derive(Clone, Debug)]
pub struct Report {
    pub arch: Arch,
    pub method: Method,
    pub depth: usize,
    pub widths: Vec<usize>,
    pub d_model: Option<usize>,
    pub params: usize,
    pub units: usize,
    /// number of real-valued discontinuities realised with a margin ramp
    pub margin_sites: usize,
    pub margin: f64,
    pub lookups: usize,
    /// size of the enumerated finite input domain (0 = not enumerated)
    pub exhaustive_points: usize,
    pub warnings: Vec<String>,
    /// architecture-level approximations (softmax attention, layernorm)
    pub approx: Vec<String>,
    pub heads: Vec<usize>,
    pub n_tokens: usize,
}

impl Report {
    /// True iff the network equals the program on the whole declared domain.
    pub fn exact(&self) -> bool {
        self.margin_sites == 0 && self.approx.is_empty()
    }
}

pub fn compile(g: &Graph, o: &Options) -> Result<(Network, Report)> {
    o.validate()?;
    if g.outputs.is_empty() {
        return error::err(ErrorKind::Invalid, "graph has no outputs");
    }
    let an = analysis::analyze(g, o.exhaustive_points, o.exhaustive_cells);
    let low = lower::lower(g, &an, o)?;
    let bounds = g.inputs.iter().map(|s| (s.ty.lo, s.ty.hi)).collect();
    let (net, approx) = assemble::assemble(&low, o, bounds)?;
    let report = Report {
        arch: o.arch,
        method: o.method,
        depth: net.depth(),
        widths: net.widths(),
        d_model: net.d_model(),
        params: net.params(),
        units: assemble::live_units(&low),
        margin_sites: low.stats.margin_sites,
        margin: o.margin,
        lookups: low.stats.lookups,
        exhaustive_points: an.n_samples,
        warnings: low.stats.warnings.clone(),
        approx,
        heads: net.heads(),
        n_tokens: net.n_tokens(),
    };
    Ok((net, report))
}

/// First IR node whose lowered value differs from the interpreter at `x`:
/// (node id, op, expected, got, analysis bounds).
pub fn debug_lowering(g: &Graph, o: &Options, x: &[f64]) -> Result<Option<(usize, String, f64, f64, (f64, f64))>> {
    o.validate()?;
    let an = analysis::analyze(g, o.exhaustive_points, o.exhaustive_cells);
    let low = lower::lower(g, &an, o)?;
    let (vals, _) = g.eval_all(x, None)?;
    let sig = low.dag.eval(&network::featurize(&low.features, x));
    for (i, l) in low.node_lins.iter().enumerate() {
        if let Some(l) = l {
            let got = neuron::Dag::eval_lin(l, &sig);
            if (got - vals[i]).abs() > 1e-6 * (1.0 + vals[i].abs()) {
                let mut desc = format!("{:?} lin={:?}", g.nodes[i].op, l);
                for &(s, _) in &l.terms {
                    let si = &low.dag.sigs[s as usize];
                    desc += &format!("\n   sig {s} = {:.4} bounds [{}, {}] {:?}", sig[s as usize], si.lo, si.hi, si.kind);
                }
                return Ok(Some((i, desc, vals[i], got, (an.lo[i], an.hi[i]))));
            }
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(g: &Graph, o: &Options, pts: &[Vec<f64>]) -> Report {
        let (net, rep) = compile(g, o).unwrap();
        for x in pts {
            let want = g.eval(x).unwrap();
            let got = net.forward(x);
            for (a, b) in want.iter().zip(&got) {
                assert!((a - b).abs() < 1e-9 * (1.0 + a.abs()), "{:?}: want {a} got {b} ({o:?})", x);
            }
        }
        rep
    }

    fn all_opts() -> Vec<Options> {
        let mut v = vec![];
        for arch in [Arch::Mlp, Arch::ResMlp, Arch::Glu] {
            for method in [Method::Depth, Method::Size, Method::Trainable, Method::Sparse] {
                for ex in [0usize, 1 << 16] {
                    for robust in [true, false] {
                        v.push(Options { arch, method, exhaustive_points: ex, robust, ..Default::default() });
                    }
                }
            }
        }
        v
    }

    #[test]
    fn int_program_exact_everywhere() {
        // f(a, b) = (a * b) % 7 if a >= b else max(a - 3, b) // 2
        let mut g = Graph::new();
        let a = g.input("a", Kind::Int, -5.0, 9.0).unwrap();
        let b = g.input("b", Kind::Int, 0.0, 12.0).unwrap();
        let ab = g.bin(BinOp::Mul, a, b).unwrap();
        let c7 = g.constant(7.0, Kind::Int).unwrap();
        let m = g.bin(BinOp::Mod, ab, c7).unwrap();
        let c3 = g.constant(3.0, Kind::Int).unwrap();
        let a3 = g.bin(BinOp::Sub, a, c3).unwrap();
        let mx = g.bin(BinOp::Max, a3, b).unwrap();
        let c2 = g.constant(2.0, Kind::Int).unwrap();
        let fd = g.bin(BinOp::FloorDiv, mx, c2).unwrap();
        let cond = g.cmp(CmpOp::Ge, a, b).unwrap();
        let r = g.select(cond, m, fd).unwrap();
        g.output(r).unwrap();
        let pts: Vec<Vec<f64>> =
            (-5..=9).flat_map(|x| (0..=12).map(move |y| vec![x as f64, y as f64])).collect();
        for o in all_opts() {
            let rep = check(&g, &o, &pts);
            assert!(rep.exact());
        }
    }

    #[test]
    fn real_polynomial_glu() {
        let mut g = Graph::new();
        let x = g.input("x", Kind::Real, -2.0, 3.0).unwrap();
        let y = g.input("y", Kind::Real, -1.0, 1.0).unwrap();
        let xy = g.bin(BinOp::Mul, x, y).unwrap();
        let xx = g.bin(BinOp::Mul, x, x).unwrap();
        let s = g.bin(BinOp::Sub, xx, xy).unwrap();
        g.output(s).unwrap();
        let o = Options { arch: Arch::Glu, ..Default::default() };
        let pts: Vec<Vec<f64>> = (0..50).map(|i| vec![-2.0 + 0.1 * i as f64, (i as f64 * 0.37).sin()]).collect();
        check(&g, &o, &pts);
        // relu archs must refuse
        let o = Options { arch: Arch::Mlp, ..Default::default() };
        assert_eq!(compile(&g, &o).unwrap_err().kind, ErrorKind::NotExact);
    }

    #[test]
    fn transformer_attention_modes() {
        // f(xs, i) = xs[i] * 2 + xs[0] - (xs[3] > xs[i]) ; xs on 5 tokens
        let mut g = Graph::new();
        let xs: Vec<NodeId> =
            (0..5).map(|p| g.input_token(&format!("xs[{p}]"), Kind::Int, -3.0, 6.0, 0, p).unwrap()).collect();
        let i = g.input("i", Kind::Int, 0.0, 4.0).unwrap();
        let xi = g.gather(xs.clone(), i).unwrap();
        let c2 = g.constant(2.0, Kind::Int).unwrap();
        let a = g.bin(BinOp::Mul, xi, c2).unwrap();
        let b = g.bin(BinOp::Add, a, xs[0]).unwrap();
        let c = g.cmp(CmpOp::Gt, xs[3], xi).unwrap();
        let r = g.bin(BinOp::Sub, b, c).unwrap();
        g.output(r).unwrap();
        let mut pts = vec![];
        let mut seed = 7u64;
        for _ in 0..400 {
            let mut x: Vec<f64> = (0..5)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((seed >> 33) % 10) as f64 - 3.0
                })
                .collect();
            x.push(((seed >> 40) % 5) as f64);
            pts.push(x);
        }
        for attention in [AttnMode::Softmax, AttnMode::Relu, AttnMode::Hard] {
            for layernorm in [false, true] {
                for tx_glu in [false, true] {
                    // exhaustive analysis off: 10^5 * 5 points would be too many anyway
                    let o = Options {
                        arch: Arch::Transformer,
                        attention,
                        layernorm,
                        tx_glu,
                        exhaustive_points: 0,
                        ..Default::default()
                    };
                    let (mut net, rep) = compile(&g, &o).unwrap();
                    assert_eq!(rep.n_tokens, 6);
                    assert_eq!(rep.exact(), attention != AttnMode::Softmax && !layernorm, "{:?}", rep.approx);
                    let tol = if rep.exact() { 1e-9 } else { 1e-4 };
                    for round in 0..2 {
                        for x in &pts {
                            let want = g.eval(x).unwrap()[0];
                            let got = net.forward(x)[0];
                            assert!((want - got).abs() < tol, "{attention:?} ln={layernorm} glu={tx_glu} r{round} {x:?}: {want} vs {got}");
                        }
                        // function-preserving transforms keep it
                        train::normalize_inputs(&mut net);
                        train::deepen(&mut net, 2, 8, 1.0, 3);
                        train::widen(&mut net, 8, 4, 1.0, 4);
                        train::balance(&mut net, 2);
                    }
                }
            }
        }
    }

    #[test]
    fn transforms_preserve_function() {
        let mut g = Graph::new();
        let a = g.input("a", Kind::Int, 0.0, 20.0).unwrap();
        let b = g.input("b", Kind::Int, 0.0, 20.0).unwrap();
        let c = g.cmp(CmpOp::Lt, a, b).unwrap();
        let d = g.bin(BinOp::Sub, b, a).unwrap();
        let s = g.select(c, d, a).unwrap();
        g.output(s).unwrap();
        let pts: Vec<Vec<f64>> = (0..=20).flat_map(|x| (0..=20).map(move |y| vec![x as f64, y as f64])).collect();
        for arch in [Arch::Mlp, Arch::ResMlp, Arch::Glu] {
            let o = Options { arch, ..Default::default() };
            let (mut net, _) = compile(&g, &o).unwrap();
            train::normalize_inputs(&mut net);
            train::balance(&mut net, 3);
            train::deepen(&mut net, 3, 8, 1.0, 1);
            train::widen(&mut net, 16, 4, 1.0, 2);
            train::balance(&mut net, 2);
            for x in &pts {
                let want = g.eval(x).unwrap()[0];
                let got = net.forward(x)[0];
                assert!((want - got).abs() < 1e-9, "{arch:?} {x:?}: {want} vs {got}");
            }
        }
    }

    #[test]
    fn lexicographic_rank_matches_direct_rank() {
        // s_i = select(legal_i, 1000*a_i - 10*b_i + c_i, -1e9), c in [0, 9], b in [0, 9]
        let mut g = Graph::new();
        let mut outs = vec![];
        for i in 0..4 {
            let l = g.input(&format!("l{i}"), Kind::Bool, 0.0, 1.0).unwrap();
            let a = g.input(&format!("a{i}"), Kind::Int, 0.0, 3.0).unwrap();
            let b = g.input(&format!("b{i}"), Kind::Int, 0.0, 9.0).unwrap();
            let c = g.input(&format!("c{i}"), Kind::Int, 0.0, 9.0).unwrap();
            let k1 = g.constant(1000.0, Kind::Int).unwrap();
            let k2 = g.constant(10.0, Kind::Int).unwrap();
            let ta = g.bin(BinOp::Mul, k1, a).unwrap();
            let tb = g.bin(BinOp::Mul, k2, b).unwrap();
            let s = g.bin(BinOp::Sub, ta, tb).unwrap();
            let s = g.bin(BinOp::Add, s, c).unwrap();
            let sent = g.constant(-1e9, Kind::Real).unwrap();
            outs.push(g.select(l, s, sent).unwrap());
        }
        for &o in &outs {
            g.output(o).unwrap();
        }
        let mut seed = 12345u64;
        let mut rnd = |n: u64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) % n) as f64
        };
        for enc in [OutputEnc::Rank, OutputEnc::OneHot] {
            let lex = g.encode_outputs(enc, true).unwrap();
            let direct = g.encode_outputs(enc, false).unwrap();
            let (net, rep) = compile(&lex, &Options::default()).unwrap();
            assert!(rep.exact());
            for _ in 0..3000 {
                let mut x = vec![];
                for _ in 0..4 {
                    // few distinct values, so ties are frequent
                    x.extend([rnd(2), rnd(2), rnd(3), rnd(3)]);
                }
                let want = direct.eval(&x).unwrap();
                assert_eq!(lex.eval(&x).unwrap(), want, "{x:?}");
                let got = net.forward(&x);
                for (w, v) in want.iter().zip(&got) {
                    assert!((w - v).abs() < 1e-9, "{x:?}: {want:?} vs {got:?}");
                }
            }
        }
    }
}
