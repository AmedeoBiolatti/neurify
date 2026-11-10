//! Lowering: IR graph -> neuron DAG, through a library of *exact gadgets*.
//!
//! Every gadget is exact on the declared input domain:
//!
//! | op                        | ReLU gadget                                         | units |
//! |---------------------------|-----------------------------------------------------|-------|
//! | affine (+, -, *c, /c)     | absorbed into weights                               | 0     |
//! | max(a,b), min, relu, abs  | b + relu(a-b), relu(x)+relu(-x)                     | 1-2   |
//! | [d >= 0], d integer       | relu(d+1) - relu(d)                                 | 2     |
//! | [d == 0], d integer       | relu(d+1) - 2 relu(d) + relu(d-1)                   | 3     |
//! | and / or / xor on {0,1}   | relu(a+b-1) (+ affine)                              | 1     |
//! | c * v, c in {0,1}         | relu(v - K(1-c)) - relu(-v - K(1-c))   (|v| <= K)   | 1-2   |
//! | f(n), n in finite set     | piecewise-linear interpolation through the set      | <=|S| |
//! | f(a,b), finite            | pair z = a*S + b (injective, affine) then lookup    | <=|S| |
//! | int * int                 | ((a+b)^2 - (a-b)^2)/4, or binary/thermometer split  |       |
//! | a * b (glu arch)          | relu(a)*b - relu(-a)*b                              | 1-2   |
//! | real comparison / floor   | exact on finite sets; else margin ramp (reported)   | 2     |

use crate::analysis::Analysis;
use crate::error::*;
use crate::ir::*;
use crate::neuron::*;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// plain feed-forward ReLU MLP
    Mlp,
    /// residual ReLU MLP: x += W2 relu(W1 x + b1)
    ResMlp,
    /// residual gated MLP (ReGLU): x += W2 (relu(Wa x + ba) * (Wb x + bb))
    Glu,
    /// transformer: token 0 carries the program, `Seq` inputs live on tokens
    /// 1..n and are read through attention; MLP blocks as in resmlp / glu
    Transformer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnMode {
    /// standard softmax attention; one-hot within `attn_tol` (approximate)
    Softmax,
    /// unnormalised ReLU attention, score relu(1 - (i - j)^2): exact
    Relu,
    /// argmax attention: exact, no gradient through the selection
    Hard,
}

impl AttnMode {
    pub fn parse(s: &str) -> Result<AttnMode> {
        match s {
            "softmax" => Ok(AttnMode::Softmax),
            "relu" => Ok(AttnMode::Relu),
            "hard" => Ok(AttnMode::Hard),
            _ => err(ErrorKind::Invalid, format!("unknown attention '{s}' (expected softmax, relu, hard)")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            AttnMode::Softmax => "softmax",
            AttnMode::Relu => "relu",
            AttnMode::Hard => "hard",
        }
    }
}

impl Arch {
    pub fn parse(s: &str) -> Result<Arch> {
        match s {
            "mlp" => Ok(Arch::Mlp),
            "resmlp" => Ok(Arch::ResMlp),
            "glu" | "reglu" => Ok(Arch::Glu),
            "transformer" => Ok(Arch::Transformer),
            _ => err(ErrorKind::Invalid, format!("unknown architecture '{s}' (expected mlp, resmlp, glu, transformer)")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Arch::Mlp => "mlp",
            Arch::ResMlp => "resmlp",
            Arch::Glu => "glu",
            Arch::Transformer => "transformer",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// minimise depth first, then units
    Depth,
    /// minimise units / parameters first, then depth
    Size,
    /// like Depth, but prefer well-conditioned, symmetric gadgets
    Trainable,
    /// minimise non-zero weights, then units, then depth: one-unit gadgets,
    /// shared wide affine forms factored into a unit, nnz-based schedule
    Sparse,
}

impl Method {
    pub fn parse(s: &str) -> Result<Method> {
        match s {
            "depth" => Ok(Method::Depth),
            "size" => Ok(Method::Size),
            "trainable" => Ok(Method::Trainable),
            "sparse" => Ok(Method::Sparse),
            _ => err(ErrorKind::Invalid, format!("unknown method '{s}' (expected depth, size, trainable, sparse)")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Method::Depth => "depth",
            Method::Size => "size",
            Method::Trainable => "trainable",
            Method::Sparse => "sparse",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub arch: Arch,
    pub method: Method,
    /// width of the band in which margin gadgets (real comparisons without a
    /// finite value set) may deviate from the exact function
    pub margin: f64,
    /// refuse margin gadgets instead of emitting them
    pub strict: bool,
    /// max units / knots a single lookup or staircase may use
    pub max_table: usize,
    /// max size of the joint finite input domain for exhaustive analysis
    pub exhaustive_points: usize,
    /// max (points x nodes) evaluated by exhaustive analysis
    pub exhaustive_cells: usize,
    /// noise-robust discrete gadgets: integer steps / lookups are flat within
    /// a quarter of the gap around every admissible value, so rounding errors
    /// are removed instead of amplified (needed for deep programs and float32)
    pub robust: bool,
    /// transformer: attention variant
    pub attention: AttnMode,
    /// transformer + softmax: max error of one attention read
    pub attn_tol: f64,
    /// transformer: gated (ReGLU) MLP blocks instead of ReLU blocks
    pub tx_glu: bool,
    /// transformer: pre-LayerNorm (opt-in, approximate)
    pub layernorm: bool,
    /// transformer + layernorm: max relative error of a LayerNorm read
    pub ln_tol: f64,
    /// one-hot value representation: every value with at most `onehot_max`
    /// distinct values (whose arguments are also small) is a vector of 0/1
    /// indicators; unary ops are free regroupings, n-ary ops one robust AND
    /// per reachable argument tuple. Every unit then sees an input range of
    /// about n_args, instead of the full value range (trainability).
    pub onehot: bool,
    pub onehot_max: usize,
    /// robust gadgets only: slope in [0, 1] at admissible values, as a
    /// fraction of the interpolating slope. 0 = flat plateaus (rounding errors
    /// removed, no gradient), 1 = interpolating. In between: still exact at
    /// every admissible value, errors damped by `slope` per gadget, gradients
    /// flow (trainability).
    pub slope: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            arch: Arch::Mlp,
            method: Method::Depth,
            margin: 1e-3,
            strict: false,
            max_table: 4096,
            exhaustive_points: 1 << 16,
            exhaustive_cells: 20_000_000,
            robust: true,
            attention: AttnMode::Softmax,
            attn_tol: 1e-6,
            tx_glu: false,
            layernorm: false,
            ln_tol: 1e-6,
            slope: 0.0,
            onehot: false,
            onehot_max: 64,
        }
    }
}

impl Options {
    /// Reject invalid numeric settings before analysis or gadget construction.
    pub fn validate(&self) -> Result<()> {
        if !self.margin.is_finite() || self.margin <= 0.0 {
            return err(ErrorKind::Invalid, "margin must be finite and positive");
        }
        if !(0.0..=1.0).contains(&self.slope) {
            return err(ErrorKind::Invalid, "slope must be in [0, 1]");
        }
        if self.arch == Arch::Transformer {
            if self.attention == AttnMode::Softmax
                && (!self.attn_tol.is_finite() || self.attn_tol <= 0.0 || self.attn_tol >= 1.0)
            {
                return err(ErrorKind::Invalid, "attn_tol must be finite and in (0, 1)");
            }
            if self.layernorm
                && (!self.ln_tol.is_finite() || self.ln_tol <= 0.0 || self.ln_tol >= 1.0)
            {
                return err(ErrorKind::Invalid, "ln_tol must be finite and in (0, 1)");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct LowerStats {
    pub margin_sites: usize,
    pub lookups: usize,
    pub warnings: Vec<String>,
}

pub struct Lowered {
    /// the DAG's inputs: fixed features of the raw inputs
    pub features: Vec<crate::network::Feature>,
    pub dag: Dag,
    pub outputs: Vec<Lin>,
    pub stats: LowerStats,
    /// lowered affine form of every live IR node (debugging / inspection)
    pub node_lins: Vec<Option<Lin>>,
}

type Samples = Option<Rc<Vec<f64>>>;

/// Operand view used by gadgets.
#[derive(Clone, Debug)]
struct V {
    lin: Lin,
    lo: f64,
    hi: f64,
    kind: Kind,
    s: Samples,
}

impl V {
    fn as_const(&self) -> Option<f64> {
        self.lin.as_const().or(if self.lo == self.hi && self.lo.is_finite() { Some(self.lo) } else { None })
    }
}

fn zip(a: &Samples, b: &Samples, f: impl Fn(f64, f64) -> f64) -> Samples {
    match (a, b) {
        (Some(x), Some(y)) => Some(Rc::new(x.iter().zip(y.iter()).map(|(&u, &v)| f(u, v)).collect())),
        _ => None,
    }
}

fn vsub(a: &V, b: &V) -> V {
    V {
        lin: a.lin.sub(&b.lin),
        lo: a.lo - b.hi,
        hi: a.hi - b.lo,
        kind: Kind::arith(a.kind, b.kind),
        s: zip(&a.s, &b.s, |u, v| u - v),
    }
}

fn vadd(a: &V, b: &V) -> V {
    V {
        lin: a.lin.add(&b.lin),
        lo: a.lo + b.lo,
        hi: a.hi + b.hi,
        kind: Kind::arith(a.kind, b.kind),
        s: zip(&a.s, &b.s, |u, v| u + v),
    }
}

/// k * a + c
fn vaff(a: &V, k: f64, c: f64) -> V {
    let (l, h) = (mulz(k, a.lo) + c, mulz(k, a.hi) + c);
    let integral = a.kind.integral() && k.fract() == 0.0 && c.fract() == 0.0;
    V {
        lin: a.lin.scale(k).add_c(c),
        lo: l.min(h),
        hi: l.max(h),
        kind: if integral { Kind::Int } else { Kind::Real },
        s: a.s.as_ref().map(|x| Rc::new(x.iter().map(|&u| k * u + c).collect())),
    }
}

fn bool_v(lin: Lin) -> V {
    V { lin, lo: 0.0, hi: 1.0, kind: Kind::Bool, s: None }
}

/// Sorted distinct values, merging values closer than a relative 1e-12.
fn distinct(xs: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut out: Vec<f64> = Vec::with_capacity(v.len().min(1024));
    for x in v {
        match out.last() {
            Some(&p) if (x - p).abs() <= 1e-12 * (1.0 + p.abs()) => {}
            _ => out.push(x),
        }
    }
    out
}

struct Lowerer<'a> {
    g: &'a Graph,
    o: &'a Options,
    dag: Dag,
    stats: LowerStats,
    /// one-hot representations: node -> [(value, indicator)]
    oh: HashMap<NodeId, OneHot>,
}

type OneHot = Rc<Vec<(f64, Lin)>>;

/// Estimated cost of a gadget alternative: (depth, units).
type Cost = (usize, usize);

impl<'a> Lowerer<'a> {
    fn gated(&self) -> bool {
        self.o.arch == Arch::Glu || (self.o.arch == Arch::Transformer && self.o.tx_glu)
    }

    fn better(&self, a: Cost, b: Cost) -> bool {
        match self.o.method {
            Method::Size | Method::Sparse => (a.1, a.0) < (b.1, b.0),
            _ => a < b,
        }
    }

    fn tight(&self, v: &V) -> (f64, f64) {
        let (mut lo, mut hi) = self.dag.bounds_with(&v.lin, Some((v.lo, v.hi)));
        if let Some(s) = &v.s {
            let (l, h) = hull(s);
            lo = lo.max(l);
            hi = hi.min(h);
        }
        (lo, hi)
    }

    fn relu(&mut self, a: &Lin, lo: f64, hi: f64) -> Lin {
        self.dag.relu(a, Some((lo, hi)))
    }

    /// Re-emit an affine form as a fresh unit when evaluating it would lose
    /// precision (coefficients >> value range), or (method=size) when it has
    /// many terms, which plain MLPs would otherwise carry layer after layer.
    /// `uses`: number of IR consumers (a lower bound on the units that will
    /// read this form). With method=sparse a form of k terms read by m units
    /// costs k*m weights inline and k + m (+1) as a unit.
    fn materialize(&mut self, lin: Lin, lo: f64, hi: f64, uses: usize) -> Lin {
        if lin.terms.len() <= 1 || !(lo.is_finite() || hi.is_finite()) {
            return lin;
        }
        let scale = lo.abs().max(hi.abs()).max(1.0);
        let ill = self.dag.magnitude(&lin) > 1024.0 * scale;
        let k = lin.terms.len();
        let wide = match self.o.method {
            Method::Size => k > 16,
            Method::Sparse => (k - 1) * uses.saturating_sub(1) > 2 && lo.is_finite() && hi.is_finite(),
            _ => false,
        };
        if !(ill || wide) {
            return lin;
        }
        if lo.is_finite() {
            self.dag.relu_force(&lin.add_c(-lo), 0.0, hi - lo).add_c(lo)
        } else {
            let neg = lin.scale(-1.0).add_c(hi);
            Lin::constant(hi).sub(&self.dag.relu_force(&neg, 0.0, hi - lo))
        }
    }

    fn relu_v(&mut self, v: &V) -> Lin {
        let (lo, hi) = self.tight(v);
        self.relu(&v.lin, lo, hi)
    }

    fn margin_site(&mut self, what: &str) -> Result<()> {
        if self.o.strict {
            return err(
                ErrorKind::NotExact,
                format!(
                    "{what} on real values has no exact ReLU realisation over a continuum; \
                     declare a finite domain (Int inputs) or compile with strict=False to \
                     accept a margin of {}",
                    self.o.margin
                ),
            );
        }
        self.stats.margin_sites += 1;
        Ok(())
    }

    fn knots(&self, v: &V, what: &str) -> Result<Vec<f64>> {
        if let Some(s) = &v.s {
            let d = distinct(s);
            if d.len() > self.o.max_table {
                return err(
                    ErrorKind::TableTooLarge,
                    format!("{what}: operand takes {} distinct values (> max_table={})", d.len(), self.o.max_table),
                );
            }
            return Ok(d);
        }
        let (lo, hi) = self.tight(v);
        if !v.kind.integral() {
            return err(
                ErrorKind::NotExact,
                format!("{what}: needs a finite value set, but the operand is real-valued over a continuum"),
            );
        }
        if !(lo.is_finite() && hi.is_finite()) {
            return err(ErrorKind::NeedsBounds, format!("{what}: operand is unbounded; declare input bounds"));
        }
        let n = (hi - lo) as usize + 1;
        if n > self.o.max_table {
            return err(
                ErrorKind::TableTooLarge,
                format!("{what}: operand range [{lo}, {hi}] has {n} values (> max_table={})", self.o.max_table),
            );
        }
        Ok((0..n).map(|i| lo + i as f64).collect())
    }

    /// (relu(x - a) - relu(x - b)) / (b - a): 0 for x <= a, 1 for x >= b.
    fn ramp(&mut self, x: &Lin, a: f64, b: f64, lo: f64, hi: f64) -> Lin {
        if lo >= b {
            return Lin::constant(1.0);
        }
        if hi <= a {
            return Lin::constant(0.0);
        }
        let k = 1.0 / (b - a);
        let p = self.relu(&x.add_c(-a), lo - a, hi - a);
        let q = self.relu(&x.add_c(-b), lo - b, hi - b);
        p.sub(&q).scale(k)
    }

    /// Robust ramp (a, b) between admissible values x0 < x1, blended with the
    /// interpolating ramp (x0, x1) by `slope`. Both are 0 at x0 and 1 at x1.
    fn soft_ramp(&mut self, x: &Lin, a: f64, b: f64, x0: f64, x1: f64, lo: f64, hi: f64) -> Lin {
        let s = self.o.slope;
        let r = self.ramp(x, a, b, lo, hi);
        if s <= 0.0 {
            return r;
        }
        let i = self.ramp(x, x0, x1, lo, hi);
        r.scale(1.0 - s).axpy(s, &i)
    }

    /// Exact lookup through sorted points (x_i, y_i): a staircase of ramps
    /// placed mid-gap (robust) or piecewise-linear interpolation.
    fn lookup(&mut self, x: &V, mut pts: Vec<(f64, f64)>) -> Lin {
        self.stats.lookups += 1;
        if pts.is_empty() {
            return Lin::constant(0.0);
        }
        // undefined entries take a neighbour's value (keeps bounds sound)
        let mut last = pts.iter().map(|p| p.1).find(|y| y.is_finite()).unwrap_or(0.0);
        for p in pts.iter_mut() {
            if p.1.is_finite() {
                last = p.1;
            } else {
                p.1 = last;
            }
        }
        if pts.len() == 1 {
            return Lin::constant(pts[0].1);
        }
        if self.o.robust {
            let (lo, hi) = self.tight(x);
            let mut out = Lin::constant(pts[0].1);
            for i in 1..pts.len() {
                let dy = pts[i].1 - pts[i - 1].1;
                if dy == 0.0 {
                    continue;
                }
                // power-of-two ramp width, centred mid-gap: dyadic weights, and
                // flat within >= gap/4 of every admissible value
                let g = pts[i].0 - pts[i - 1].0;
                let w = 2f64.powi((0.5 * g).log2().floor() as i32);
                let mid = 0.5 * (pts[i].0 + pts[i - 1].0);
                let r = self.soft_ramp(&x.lin, mid - 0.5 * w, mid + 0.5 * w, pts[i - 1].0, pts[i].0, lo, hi);
                out = out.axpy(dy, &r);
            }
            return out;
        }
        let m = pts.len();
        let slopes: Vec<f64> = (0..m - 1).map(|i| (pts[i + 1].1 - pts[i].1) / (pts[i + 1].0 - pts[i].0)).collect();
        let smax = slopes.iter().fold(0.0f64, |a, &s| a.max(s.abs()));
        let (x0, y0) = pts[0];
        // y0 + s0 (x - x0), and x >= x0 on the domain
        let mut out = x.lin.add_c(-x0).scale(slopes[0]).add_c(y0);
        let (_, xhi) = self.tight(x);
        for i in 1..m - 1 {
            let delta = slopes[i] - slopes[i - 1];
            if delta.abs() <= 1e-12 * smax {
                continue;
            }
            let xi = pts[i].0;
            let r = self.relu(&x.lin.add_c(-xi), x0 - xi, xhi - xi);
            out = out.axpy(delta, &r);
        }
        out
    }

    fn lookup_fn(&mut self, x: &V, what: &str, f: impl Fn(f64) -> f64) -> Result<Lin> {
        let ks = self.knots(x, what)?;
        let pts = ks.into_iter().map(|k| (k, f(k))).collect();
        Ok(self.lookup(x, pts))
    }

    /// f(a, b) on a finite joint domain via an injective affine pairing.
    fn lookup2(&mut self, a: &V, b: &V, what: &str, f: impl Fn(f64, f64) -> f64) -> Result<Lin> {
        let ka = self.knots(a, what)?;
        let kb = self.knots(b, what)?;
        let (alo, blo, bhi) = (ka[0], kb[0], *kb.last().unwrap());
        let gap = ka.windows(2).fold(f64::INFINITY, |m, w| m.min(w[1] - w[0]));
        let gap = if gap.is_finite() { gap } else { 1.0 };
        let scale = ((bhi - blo) + 1.0) / gap;
        let pairs: Vec<(f64, f64)> = match (&a.s, &b.s) {
            (Some(x), Some(y)) => x.iter().cloned().zip(y.iter().cloned()).collect(),
            _ => {
                if ka.len().saturating_mul(kb.len()) > self.o.max_table {
                    return err(
                        ErrorKind::TableTooLarge,
                        format!("{what}: joint domain {}x{} exceeds max_table={}", ka.len(), kb.len(), self.o.max_table),
                    );
                }
                ka.iter().flat_map(|&u| kb.iter().map(move |&v| (u, v))).collect()
            }
        };
        if self.o.method == Method::Trainable && a.kind.integral() && b.kind.integral() {
            return self.lookup2_onehot(a, b, &ka, &kb, &pairs, f);
        }
        let mut pts: Vec<(f64, f64)> =
            pairs.iter().map(|&(u, v)| ((u - alo) * scale + (v - blo), f(u, v))).collect();
        pts.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap());
        pts.dedup_by(|p, q| (p.0 - q.0).abs() <= 1e-12 * (1.0 + q.0.abs()));
        if pts.len() > self.o.max_table {
            return err(
                ErrorKind::TableTooLarge,
                format!("{what}: {} distinct operand pairs (> max_table={})", pts.len(), self.o.max_table),
            );
        }
        let z = vadd(&vaff(a, scale, -alo * scale), &vaff(b, 1.0, -blo));
        let z = V { s: None, lo: 0.0, hi: pts.last().unwrap().0, ..z };
        Ok(self.lookup(&z, pts))
    }

    /// f(a, b) = sum_ij f(i, j) * AND([a = i], [b = j]): every unit reads one
    /// operand (range |A| or |B|) or two indicators (range 2), instead of a
    /// packed index with range |A| * |B| -- far less precision demanded of
    /// each unit, so the network tolerates much larger weight perturbations.
    fn lookup2_onehot(
        &mut self,
        a: &V,
        b: &V,
        ka: &[f64],
        kb: &[f64],
        pairs: &[(f64, f64)],
        f: impl Fn(f64, f64) -> f64,
    ) -> Result<Lin> {
        self.stats.lookups += 1;
        let mut pairs: Vec<(f64, f64)> = pairs.to_vec();
        pairs.sort_by(|p, q| p.partial_cmp(q).unwrap());
        pairs.dedup();
        // the most common value becomes the constant term
        let vals: Vec<f64> = pairs.iter().map(|&(u, v)| f(u, v)).collect();
        let mut counts: Vec<(f64, usize)> = vec![];
        for &y in &vals {
            match counts.iter_mut().find(|c| c.0 == y) {
                Some(c) => c.1 += 1,
                None => counts.push((y, 1)),
            }
        }
        let base = counts.iter().max_by_key(|c| c.1).map_or(0.0, |c| c.0);
        let mut ea: Vec<Option<Lin>> = vec![None; ka.len()];
        let mut eb: Vec<Option<Lin>> = vec![None; kb.len()];
        let mut out = Lin::constant(base);
        for (&(u, v), &y) in pairs.iter().zip(&vals) {
            if y == base {
                continue;
            }
            let i = ka.iter().position(|&k| k == u).unwrap();
            let j = kb.iter().position(|&k| k == v).unwrap();
            if ea[i].is_none() {
                ea[i] = Some(self.eq0(&vaff(a, 1.0, -u))?);
            }
            if eb[j].is_none() {
                eb[j] = Some(self.eq0(&vaff(b, 1.0, -v))?);
            }
            // robust AND: flat within 1/4 of every 0/1 input
            let both = ea[i].as_ref().unwrap().add(eb[j].as_ref().unwrap());
            let and = self.ramp(&both, 1.25, 1.75, 0.0, 2.0);
            out = out.axpy(y - base, &and);
        }
        Ok(out)
    }

    // ------------------------------------------------------- indicators

    fn step_int(&mut self, d: &V) -> Lin {
        // [d >= 0] for integer d
        let (lo, hi) = self.tight(d);
        if lo >= 0.0 {
            return Lin::constant(1.0);
        }
        if hi < 0.0 {
            return Lin::constant(0.0);
        }
        if self.o.robust {
            return self.soft_ramp(&d.lin, -0.75, -0.25, -1.0, 0.0, lo, hi);
        }
        let r1 = self.relu(&d.lin.add_c(1.0), lo + 1.0, hi + 1.0);
        let r0 = self.relu(&d.lin, lo, hi);
        r1.sub(&r0)
    }

    fn ge0(&mut self, d: &V) -> Result<Lin> {
        let (lo, hi) = self.tight(d);
        if lo >= 0.0 {
            return Ok(Lin::constant(1.0));
        }
        if hi < 0.0 {
            return Ok(Lin::constant(0.0));
        }
        if d.kind.integral() {
            return Ok(self.step_int(d));
        }
        if d.s.is_some() {
            return self.lookup_fn(d, "comparison", |x| if x >= 0.0 { 1.0 } else { 0.0 });
        }
        self.margin_site("comparison")?;
        let u = d.lin.scale(1.0 / self.o.margin);
        let (ul, uh) = (lo / self.o.margin, hi / self.o.margin);
        let r1 = self.relu(&u.add_c(1.0), ul + 1.0, uh + 1.0);
        let r0 = self.relu(&u, ul, uh);
        Ok(r1.sub(&r0))
    }

    fn gt0(&mut self, d: &V) -> Result<Lin> {
        let (lo, hi) = self.tight(d);
        if lo > 0.0 {
            return Ok(Lin::constant(1.0));
        }
        if hi <= 0.0 {
            return Ok(Lin::constant(0.0));
        }
        if d.kind.integral() {
            return Ok(self.step_int(&vaff(d, 1.0, -1.0)));
        }
        if d.s.is_some() {
            return self.lookup_fn(d, "comparison", |x| if x > 0.0 { 1.0 } else { 0.0 });
        }
        self.margin_site("comparison")?;
        let u = d.lin.scale(1.0 / self.o.margin);
        let (ul, uh) = (lo / self.o.margin, hi / self.o.margin);
        let r0 = self.relu(&u, ul, uh);
        let r1 = self.relu(&u.add_c(-1.0), ul - 1.0, uh - 1.0);
        Ok(r0.sub(&r1))
    }

    fn eq0(&mut self, d: &V) -> Result<Lin> {
        let (lo, hi) = self.tight(d);
        if lo > 0.0 || hi < 0.0 {
            return Ok(Lin::constant(0.0));
        }
        if lo == 0.0 && hi == 0.0 {
            return Ok(Lin::constant(1.0));
        }
        if d.kind.integral() && self.o.robust {
            let up = self.soft_ramp(&d.lin, -0.75, -0.25, -1.0, 0.0, lo, hi);
            let down = self.soft_ramp(&d.lin, 0.25, 0.75, 0.0, 1.0, lo, hi);
            return Ok(up.sub(&down));
        }
        if d.kind.integral() {
            let a = self.relu(&d.lin.add_c(1.0), lo + 1.0, hi + 1.0);
            let b = self.relu(&d.lin, lo, hi);
            let c = self.relu(&d.lin.add_c(-1.0), lo - 1.0, hi - 1.0);
            return Ok(a.axpy(-2.0, &b).add(&c));
        }
        if d.s.is_some() {
            return self.lookup_fn(d, "equality", |x| if x == 0.0 { 1.0 } else { 0.0 });
        }
        let p = self.ge0(d)?;
        let q = self.ge0(&vaff(d, -1.0, 0.0))?;
        Ok(p.add(&q).add_c(-1.0))
    }

    // ------------------------------------------------------------ gating

    /// Would a gating constant of size `k` swallow the value in float32?
    /// (v + K - K loses the low bits of v once ulp(K) matters.)
    fn big_k(k: f64, v: &V) -> bool {
        k >= 65536.0 || (!v.kind.integral() && k >= 256.0)
    }

    /// Exact 0/1 units for c and 1 - c. Big constants then multiply an exact
    /// zero on the active path instead of cancelling against each other.
    fn bool_units(&mut self, c: &Lin) -> (Lin, Lin) {
        let single = c.c == 0.0 && c.terms.len() == 1 && c.terms[0].1 == 1.0;
        let cu = if single { c.clone() } else { self.dag.relu_force(c, 0.0, 1.0) };
        let nc = self.dag.relu_force(&Lin::constant(1.0).sub(c), 0.0, 1.0);
        (cu, nc)
    }

    /// c * v given c and nc = 1 - c (both in {0, 1}).
    fn gate_core(&mut self, c: &Lin, nc: &Lin, v: &V, allow_big_m: bool) -> Result<Lin> {
        if let Some(vc) = v.as_const() {
            if vc.abs() >= 256.0 && c.terms.len() > 1 {
                return Ok(self.bool_units(c).0.scale(vc));
            }
            return Ok(c.scale(vc));
        }
        let (lo, hi) = self.tight(v);
        if self.gated() {
            return Ok(self.dag.gated(c, &v.lin, Some((0.0, 1.0)), Some((lo, hi))));
        }
        if !(lo.is_finite() && hi.is_finite()) {
            return err(
                ErrorKind::NeedsBounds,
                "multiplying by a condition (if/else, select) needs finite bounds on the value in \
                 ReLU architectures; declare input bounds or use arch='glu'",
            );
        }
        if lo >= 0.0 {
            return Ok(self.relu(&v.lin.axpy(-hi, nc), lo - hi, hi));
        }
        if hi <= 0.0 {
            let r = self.relu(&v.lin.scale(-1.0).axpy(lo, nc), -hi + lo, -lo);
            return Ok(r.scale(-1.0));
        }
        if allow_big_m && matches!(self.o.method, Method::Size | Method::Sparse) {
            // big-M, one unit: relu(v - lo - (hi-lo)(1-c)) + lo c
            let arg = v.lin.add_c(-lo).axpy(-(hi - lo), nc);
            let r = self.relu(&arg, -(hi - lo), hi - lo);
            return Ok(r.axpy(lo, c));
        }
        // symmetric, two units, |weights| <= max(|lo|, |hi|)
        let k = (-lo).max(hi);
        let p = self.relu(&v.lin.axpy(-k, nc), lo - k, hi);
        let q = self.relu(&v.lin.scale(-1.0).axpy(-k, nc), -hi - k, -lo);
        Ok(p.sub(&q))
    }

    /// c * v with c in {0, 1}.
    fn gate(&mut self, c: &V, v: &V) -> Result<Lin> {
        if let Some(cc) = c.as_const() {
            return Ok(v.lin.scale(cc));
        }
        let (lo, hi) = self.tight(v);
        let k = lo.abs().max(hi.abs());
        if Self::big_k(k, v) && k.is_finite() {
            let (cu, nc) = self.bool_units(&c.lin);
            return self.gate_core(&cu, &nc, v, false);
        }
        let nc = Lin::constant(1.0).sub(&c.lin);
        self.gate_core(&c.lin, &nc, v, true)
    }

    /// select(c, a, b). For values of very different magnitude, c*a + (1-c)*b
    /// with exact 0/1 units avoids the cancellation in b + c*(a - b).
    fn select(&mut self, c: &V, a: &V, b: &V) -> Result<Lin> {
        if let Some(cc) = c.as_const() {
            return Ok(if cc != 0.0 { a.lin.clone() } else { b.lin.clone() });
        }
        let d = vsub(a, b);
        let (lo, hi) = self.tight(&d);
        let k = lo.abs().max(hi.abs());
        if Self::big_k(k, &d) && k.is_finite() {
            let (cu, nc) = self.bool_units(&c.lin);
            let ta = self.gate_core(&cu, &nc, a, false)?;
            let tb = self.gate_core(&nc, &cu, b, false)?;
            return Ok(ta.add(&tb));
        }
        let t = self.gate(c, &d)?;
        Ok(b.lin.add(&t))
    }

    // ------------------------------------------------------- arithmetic

    fn int_bits(r: f64) -> usize {
        let r = r as u64;
        (64 - r.leading_zeros()) as usize
    }

    /// n = base + sum_j coef_j * g_j with g_j in {0,1}.
    fn decompose(&mut self, n: &V, binary: bool) -> Result<(f64, Vec<(f64, V)>)> {
        let (lo, hi) = self.tight(n);
        if !(lo.is_finite() && hi.is_finite()) {
            return err(ErrorKind::NeedsBounds, "integer decomposition needs finite bounds");
        }
        let r = hi - lo;
        let z = vaff(n, 1.0, -lo);
        let mut parts = vec![];
        if binary {
            let k = Self::int_bits(r);
            let mut rem = V { lo: 0.0, hi: r, ..z };
            for i in (0..k).rev() {
                let p = (1u64 << i) as f64;
                let b = self.step_int(&vaff(&rem, 1.0, -p));
                let bv = bool_v(b);
                rem = V {
                    lin: rem.lin.axpy(-p, &bv.lin),
                    lo: 0.0,
                    hi: (p - 1.0).min(rem.hi),
                    kind: Kind::Int,
                    s: None,
                };
                parts.push((p, bv));
            }
        } else {
            if r as usize > self.o.max_table {
                return err(ErrorKind::TableTooLarge, "thermometer decomposition exceeds max_table");
            }
            for v in 1..=(r as usize) {
                let g = self.step_int(&vaff(&z, 1.0, -(v as f64)));
                parts.push((1.0, bool_v(g)));
            }
        }
        Ok((lo, parts))
    }

    fn mul_decomp(&mut self, x: &V, n: &V, binary: bool) -> Result<Lin> {
        let (base, parts) = self.decompose(n, binary)?;
        let mut out = x.lin.scale(base);
        for (coef, g) in parts {
            let t = self.gate(&g, x)?;
            out = out.axpy(coef, &t);
        }
        Ok(out)
    }

    fn range_len(&self, v: &V) -> Option<f64> {
        if let Some(s) = &v.s {
            return Some(distinct(s).len() as f64);
        }
        let (lo, hi) = self.tight(v);
        (v.kind.integral() && lo.is_finite() && hi.is_finite()).then_some(hi - lo + 1.0)
    }

    /// Number of distinct (a, b) pairs a pair-lookup would interpolate through.
    fn pair_count(&self, a: &V, b: &V) -> Option<usize> {
        if let (Some(x), Some(y)) = (&a.s, &b.s) {
            let mut p: Vec<(u64, u64)> = x.iter().zip(y.iter()).map(|(u, v)| (u.to_bits(), v.to_bits())).collect();
            p.sort_unstable();
            p.dedup();
            return Some(p.len());
        }
        let (ra, rb) = (self.range_len(a)?, self.range_len(b)?);
        Some((ra * rb).min(usize::MAX as f64) as usize)
    }

    /// k * v. A multi-term form scaled by a large constant would cancel big
    /// partial sums in float32; materialise exact 0/1 indicators and
    /// non-negative integers to a single unit first.
    fn scale_v(&mut self, v: &V, k: f64) -> Lin {
        if k.abs() >= 256.0 && v.lin.terms.len() > 1 {
            let (lo, hi) = self.tight(v);
            if v.kind == Kind::Bool {
                return self.bool_units(&v.lin).0.scale(k);
            }
            if v.kind.integral() && lo >= 0.0 && hi.is_finite() {
                return self.dag.relu_force(&v.lin, lo, hi).scale(k);
            }
        }
        v.lin.scale(k)
    }

    fn mul(&mut self, a: &V, b: &V, same: bool) -> Result<Lin> {
        if let Some(k) = a.as_const() {
            return Ok(self.scale_v(b, k));
        }
        if let Some(k) = b.as_const() {
            return Ok(self.scale_v(a, k));
        }
        if a.kind == Kind::Bool {
            return self.gate(a, b);
        }
        if b.kind == Kind::Bool {
            return self.gate(b, a);
        }
        if self.gated() {
            return Ok(self.product_glu(a, b));
        }
        let t = self.o.max_table as f64;
        let gw = if matches!(self.o.method, Method::Size | Method::Sparse) { 1 } else { 2 };
        let mut cands: Vec<(Cost, u8)> = vec![];
        if same {
            if let Some(r) = self.range_len(a).filter(|&r| r <= t) {
                cands.push(((1, r as usize), 0));
            }
        }
        if let Some(np) = self.pair_count(a, b).filter(|&n| n <= self.o.max_table) {
            cands.push(((1, np), 1));
        }
        if !same && a.kind.integral() && b.kind.integral() {
            if let (Some(rs), Some(rt)) = (self.range_len(&vadd(a, b)), self.range_len(&vsub(a, b))) {
                if rs <= t && rt <= t {
                    cands.push(((1, (rs + rt) as usize), 2));
                }
            }
        }
        // decompositions of an integer operand (binary: id, thermometer: id + 2)
        for (id, nv, xv) in [(3u8, a, b), (4u8, b, a)] {
            if !nv.kind.integral() {
                continue;
            }
            let (nl, nh) = self.tight(nv);
            let (xl, xh) = self.tight(xv);
            if !(nl.is_finite() && nh.is_finite() && xl.is_finite() && xh.is_finite()) {
                continue;
            }
            let r = nh - nl;
            let k = Self::int_bits(r).max(1);
            cands.push(((k + 1, 2 * k + gw * k), id));
            if r <= t {
                cands.push(((2, r as usize + 1 + gw * r as usize), id + 2));
            }
        }
        let mut best: Option<(Cost, u8)> = None;
        for (c, id) in cands {
            if best.map_or(true, |(bc, _)| self.better(c, bc)) {
                best = Some((c, id));
            }
        }
        let Some((_, id)) = best else {
            return err(
                ErrorKind::NotExact,
                "product of two real-valued variables is not piecewise linear: use arch='glu' \
                 (exact gated units) or give one operand a finite integer domain",
            );
        };
        match id {
            0 => self.lookup_fn(a, "square", |x| x * x),
            1 => self.lookup2(a, b, "product", |x, y| x * y),
            2 => {
                let s = self.lookup_fn(&vadd(a, b), "product", |x| x * x / 4.0)?;
                let d = self.lookup_fn(&vsub(a, b), "product", |x| x * x / 4.0)?;
                Ok(s.sub(&d))
            }
            3 => self.mul_decomp(b, a, true),
            4 => self.mul_decomp(a, b, true),
            5 => self.mul_decomp(b, a, false),
            6 => self.mul_decomp(a, b, false),
            _ => unreachable!(),
        }
    }

    fn product_glu(&mut self, a: &V, b: &V) -> Lin {
        let (al, ah) = self.tight(a);
        let (bl, bh) = self.tight(b);
        let neg = |v: &V| vaff(v, -1.0, 0.0);
        if al >= 0.0 {
            return self.dag.gated(&a.lin, &b.lin, Some((al, ah)), Some((bl, bh)));
        }
        if bl >= 0.0 {
            return self.dag.gated(&b.lin, &a.lin, Some((bl, bh)), Some((al, ah)));
        }
        if ah <= 0.0 {
            let na = neg(a);
            return self.dag.gated(&na.lin, &b.lin, Some((-ah, -al)), Some((bl, bh))).scale(-1.0);
        }
        let p = self.dag.gated(&a.lin, &b.lin, Some((al, ah)), Some((bl, bh)));
        let q = self.dag.gated(&neg(a).lin, &b.lin, Some((-ah, -al)), Some((bl, bh)));
        p.sub(&q)
    }

    /// sum_{j=from}^{to} [x >= j] for real x (margin ramps).
    fn staircase(&mut self, x: &V, from: f64, to: f64) -> Result<Lin> {
        if to - from + 1.0 > self.o.max_table as f64 {
            return err(ErrorKind::TableTooLarge, "floor/int() staircase exceeds max_table");
        }
        let mut out = Lin::constant(0.0);
        let mut j = from;
        while j <= to {
            let s = self.ge0(&vaff(x, 1.0, -j))?;
            out = out.add(&s);
            j += 1.0;
        }
        Ok(out)
    }

    fn bounded(&self, x: &V, what: &str) -> Result<(f64, f64)> {
        let (lo, hi) = self.tight(x);
        if !(lo.is_finite() && hi.is_finite()) {
            return err(ErrorKind::NeedsBounds, format!("{what} of an unbounded value; declare input bounds"));
        }
        Ok((lo, hi))
    }

    /// floor(x) for real x without a finite value set.
    fn floor_real(&mut self, x: &V) -> Result<Lin> {
        let (lo, hi) = self.bounded(x, "floor")?;
        Ok(self.staircase(x, lo.floor() + 1.0, hi.floor())?.add_c(lo.floor()))
    }

    /// trunc(x) = sum_{j>=1} [x >= j] - sum_{j>=1} [-x >= j]
    fn trunc_real(&mut self, x: &V) -> Result<Lin> {
        let (lo, hi) = self.bounded(x, "int()")?;
        let p = self.staircase(x, 1.0, hi.floor())?;
        let q = self.staircase(&vaff(x, -1.0, 0.0), 1.0, (-lo).floor())?;
        Ok(p.sub(&q))
    }

    /// floor(a / c) for integer a and constant c != 0.
    fn floordiv_int_const(&mut self, a: &V, c: f64) -> Result<Lin> {
        if c < 0.0 {
            return self.floordiv_int_const(&vaff(a, -1.0, 0.0), -c);
        }
        let (lo, hi) = self.tight(a);
        if !(lo.is_finite() && hi.is_finite()) {
            return err(ErrorKind::NeedsBounds, "// and % need bounded integer operands");
        }
        let (base, top) = ((lo / c).floor(), (hi / c).floor());
        let steps = top - base;
        // a finite value set may be cheaper
        if a.s.is_some() {
            let n = distinct(a.s.as_ref().unwrap()).len() as f64;
            if n < 2.0 * steps {
                return self.lookup_fn(a, "floordiv", |x| py_divmod(x, c).0);
            }
        }
        if steps as usize > self.o.max_table {
            return err(ErrorKind::TableTooLarge, "floordiv staircase exceeds max_table");
        }
        let mut out = Lin::constant(base);
        for j in (base as i64 + 1)..=(top as i64) {
            let t = j as f64 * c;
            let thr = t.ceil(); // a >= j*c  <=>  a >= ceil(j*c) for integer a
            out = out.add(&self.step_int(&vaff(a, 1.0, -thr)));
        }
        Ok(out)
    }

    fn floordiv(&mut self, a: &V, b: &V) -> Result<Lin> {
        if let Some(c) = b.as_const() {
            if c == 0.0 {
                return Ok(Lin::constant(0.0));
            }
            if a.kind.integral() && c.fract() == 0.0 {
                return self.floordiv_int_const(a, c);
            }
            if a.s.is_some() || a.kind.integral() {
                return self.lookup_fn(a, "floordiv", |x| py_divmod(x, c).0);
            }
            let q = vaff(a, 1.0 / c, 0.0);
            let q = V { kind: Kind::Real, ..q };
            return self.floor_real(&q);
        }
        self.lookup2(a, b, "floordiv", |x, y| py_divmod(x, y).0)
    }

    // ------------------------------------------------------------- nodes

    // ------------------------------------------------- one-hot values

    /// Distinct values of `v`, if there are at most `onehot_max`.
    fn domain(&self, v: &V) -> Option<Vec<f64>> {
        let k = self.o.onehot_max;
        if let Some(s) = &v.s {
            let d = distinct(s);
            return (d.len() <= k).then_some(d);
        }
        if !v.kind.integral() {
            return None;
        }
        let (lo, hi) = self.tight(v);
        if !(lo.is_finite() && hi.is_finite()) || hi - lo + 1.0 > k as f64 {
            return None;
        }
        Some((0..=(hi - lo) as usize).map(|i| lo + i as f64).collect())
    }

    /// One-hot form of a lowered node (robust equality hats if it is scalar).
    fn onehot_of(&mut self, id: NodeId, v: &V) -> Result<Option<OneHot>> {
        if let Some(o) = self.oh.get(&id) {
            return Ok(Some(o.clone()));
        }
        if let Some(c) = v.as_const() {
            return Ok(Some(Rc::new(vec![(c, Lin::constant(1.0))])));
        }
        let Some(dom) = self.domain(v) else {
            return Ok(None);
        };
        let ind: Vec<(f64, Lin)> = if v.kind == Kind::Bool {
            vec![(0.0, Lin::constant(1.0).sub(&v.lin)), (1.0, v.lin.clone())]
        } else {
            let mut out = vec![];
            for &k in &dom {
                out.push((k, self.eq0(&vaff(v, 1.0, -k))?));
            }
            out
        };
        let rc = Rc::new(ind);
        self.oh.insert(id, rc.clone());
        Ok(Some(rc))
    }

    /// Robust AND of 0/1 indicators: flat within 1/4 of every 0/1 input.
    fn and_all(&mut self, xs: &[&Lin]) -> Lin {
        let xs: Vec<&&Lin> = xs.iter().filter(|l| !(l.terms.is_empty() && l.c == 1.0)).collect();
        match xs.len() {
            0 => Lin::constant(1.0),
            1 => (*xs[0]).clone(),
            k => {
                let mut s = Lin::constant(0.0);
                for l in &xs {
                    s = s.add(l);
                }
                let k = k as f64;
                self.ramp(&s, k - 0.75, k - 0.25, 0.0, k)
            }
        }
    }

    /// Lower node `id` in one-hot form: (scalar form, indicators), or None if
    /// a domain is too large (the caller falls back to scalar gadgets).
    fn lower_onehot(&mut self, id: NodeId, me: &V, vals: &[Option<V>]) -> Result<Option<(Lin, OneHot)>> {
        let g = self.g;
        let get = |a: NodeId| vals[a].clone().expect("argument lowered before use");
        if matches!(g.nodes[id].op, Op::Input(_) | Op::Const(_)) {
            return Ok(None);
        }
        if self.domain(me).is_none() {
            return Ok(None);
        }
        let mut acc: Vec<(f64, Lin)> = vec![];
        let add = |acc: &mut Vec<(f64, Lin)>, r: f64, l: Lin| match acc.iter_mut().find(|e| e.0 == r) {
            Some(e) => e.1 = e.1.add(&l),
            None => acc.push((r, l)),
        };
        match g.nodes[id].op {
            Op::Select(c, a, b) => {
                let (Some(oc), Some(oa), Some(ob)) =
                    (self.onehot_of(c, &get(c))?, self.onehot_of(a, &get(a))?, self.onehot_of(b, &get(b))?)
                else {
                    return Ok(None);
                };
                let ind = |o: &OneHot, v: f64| o.iter().find(|e| e.0 == v).map(|e| e.1.clone());
                let (c1, c0) = (ind(&oc, 1.0), ind(&oc, 0.0));
                for (branch, o) in [(c1, oa), (c0, ob)] {
                    let Some(cl) = branch else { continue };
                    for (v, e) in o.iter() {
                        let and = self.and_all(&[&cl, e]);
                        add(&mut acc, *v, and);
                    }
                }
            }
            Op::Gather(k) => {
                let gt = &g.gathers[k];
                let Some(oi) = self.onehot_of(gt.index, &get(gt.index))? else {
                    return Ok(None);
                };
                for (p, e) in oi.iter() {
                    match gt.pick(*p) {
                        None => add(&mut acc, 0.0, e.clone()),
                        Some(q) => {
                            let el = gt.elems[q];
                            let Some(oe) = self.onehot_of(el, &get(el))? else {
                                return Ok(None);
                            };
                            for (v, f) in oe.iter() {
                                let and = self.and_all(&[e, f]);
                                add(&mut acc, *v, and);
                            }
                        }
                    }
                }
            }
            _ => {
                let mut args = g.args_of(id);
                args.sort();
                args.dedup();
                let mut ohs = vec![];
                for &a in &args {
                    match self.onehot_of(a, &get(a))? {
                        Some(o) => ohs.push(o),
                        None => return Ok(None),
                    }
                }
                // reachable argument tuples (joint samples) or the full product
                let all_s: Option<Vec<Rc<Vec<f64>>>> = args.iter().map(|&a| get(a).s).collect();
                let tuples: Vec<Vec<usize>> = match all_s {
                    Some(ss) if !ss.is_empty() => {
                        let mut seen = std::collections::HashSet::new();
                        let mut t = vec![];
                        for r in 0..ss[0].len() {
                            let key: Vec<usize> = ss
                                .iter()
                                .zip(&ohs)
                                .map(|(s, o)| o.iter().position(|e| e.0 == s[r]).unwrap_or(usize::MAX))
                                .collect();
                            if key.iter().all(|&x| x != usize::MAX) && seen.insert(key.clone()) {
                                t.push(key);
                                if t.len() > self.o.max_table {
                                    return Ok(None);
                                }
                            }
                        }
                        t
                    }
                    _ => {
                        let n: usize = ohs.iter().map(|o| o.len()).product();
                        if n > self.o.max_table {
                            return Ok(None);
                        }
                        let mut t = vec![vec![]];
                        for o in &ohs {
                            t = t.into_iter().flat_map(|p| (0..o.len()).map(move |i| [p.clone(), vec![i]].concat())).collect();
                        }
                        t
                    }
                };
                for t in tuples {
                    let r = g.eval_node(id, &|x| {
                        let j = args.iter().position(|&a| a == x).unwrap();
                        ohs[j][t[j]].0
                    });
                    let inds: Vec<&Lin> = t.iter().enumerate().map(|(j, &i)| &ohs[j][i].1).collect();
                    let and = self.and_all(&inds);
                    add(&mut acc, r, and);
                }
            }
        }
        if acc.len() > self.o.onehot_max.max(2) {
            return Ok(None);
        }
        acc.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut lin = Lin::constant(0.0);
        for (v, l) in &acc {
            lin = lin.axpy(*v, l);
        }
        Ok(Some((lin, Rc::new(acc))))
    }

    fn lower_node(&mut self, id: NodeId, vals: &[Option<V>]) -> Result<Lin> {
        let node = &self.g.nodes[id];
        let get = |i: NodeId| -> V { vals[i].clone().expect("operand lowered before use") };
        Ok(match node.op {
            Op::Input(k) => match self.g.inputs[k].token {
                Some((f, p)) if self.o.arch == Arch::Transformer => {
                    let t = self.g.inputs[k].ty;
                    let p = p as f64;
                    self.dag.attn(f, &Lin::constant(p), Some(Lin::constant(p * p)), t.lo, t.hi)
                }
                _ => Lin::sig(k as Sig),
            },
            Op::Gather(k) => {
                let gt = &self.g.gathers[k];
                let idx = get(gt.index);
                let (ilo, ihi) = self.tight(&idx);
                // a whole sequence feature, read by position -> one attention head
                let feature = gt.elems.iter().enumerate().try_fold(None, |acc: Option<usize>, (p, &e)| {
                    match self.g.nodes[e].op {
                        Op::Input(i) => match self.g.inputs[i].token {
                            Some((f, pos)) if pos == p && acc.map_or(true, |a| a == f) => Some(Some(f)),
                            _ => None,
                        },
                        _ => None,
                    }
                });
                if let (true, Some(Some(f)), true) = (self.o.arch == Arch::Transformer, feature, ilo >= 0.0) {
                    let (vlo, vhi) = (self.g.ty(id).lo, self.g.ty(id).hi);
                    let sq = if self.o.attention == AttnMode::Relu {
                        Some(self.lookup_fn(&idx, "attention index", |x| x * x)?)
                    } else {
                        None
                    };
                    self.dag.attn(f, &idx.lin, sq, vlo, vhi)
                } else {
                    // sum_k [idx == k (or k - n)] * elems[k]
                    let n = gt.elems.len() as f64;
                    let mut out = Lin::constant(0.0);
                    for (p, &e) in gt.elems.iter().enumerate() {
                        let p = p as f64;
                        let hit_pos = ilo <= p && p <= ihi;
                        let hit_neg = ilo <= p - n && p - n <= ihi;
                        if !(hit_pos || hit_neg) {
                            continue;
                        }
                        let mut c = Lin::constant(0.0);
                        if hit_pos {
                            c = c.add(&self.eq0(&vaff(&idx, 1.0, -p))?);
                        }
                        if hit_neg {
                            c = c.add(&self.eq0(&vaff(&idx, 1.0, -(p - n)))?);
                        }
                        let t = self.gate(&bool_v(c), &get(e))?;
                        out = out.add(&t);
                    }
                    out
                }
            }
            Op::Const(c) => Lin::constant(c),
            Op::Bin(op, a, b) => {
                let (va, vb) = (get(a), get(b));
                use BinOp::*;
                match op {
                    Add => va.lin.add(&vb.lin),
                    Sub => va.lin.sub(&vb.lin),
                    Mul => self.mul(&va, &vb, a == b)?,
                    Div => match vb.as_const() {
                        Some(c) if c == 0.0 => Lin::constant(0.0),
                        Some(c) => va.lin.scale(1.0 / c),
                        None => self.lookup2(&va, &vb, "division", |x, y| if y == 0.0 { 0.0 } else { x / y })?,
                    },
                    FloorDiv => self.floordiv(&va, &vb)?,
                    Mod => match vb.as_const() {
                        Some(c) if c == 0.0 => Lin::constant(0.0),
                        Some(c) if va.s.is_some() && !va.kind.integral() => {
                            self.lookup_fn(&va, "mod", |x| py_divmod(x, c).1)?
                        }
                        Some(c) => {
                            let q = self.floordiv(&va, &vb)?;
                            va.lin.axpy(-c, &q)
                        }
                        None => self.lookup2(&va, &vb, "mod", |x, y| py_divmod(x, y).1)?,
                    },
                    Max => {
                        let r = self.relu_v(&vsub(&va, &vb));
                        vb.lin.add(&r)
                    }
                    Min => {
                        let r = self.relu_v(&vsub(&va, &vb));
                        va.lin.sub(&r)
                    }
                    And | Or | Xor if self.o.method == Method::Trainable && self.o.robust => {
                        // flat within 1/4 of every 0/1 input: errors are absorbed
                        let s = va.lin.add(&vb.lin);
                        let and = self.ramp(&s, 1.25, 1.75, 0.0, 2.0);
                        let or = self.ramp(&s, 0.25, 0.75, 0.0, 2.0);
                        match op {
                            And => and,
                            Or => or,
                            _ => or.sub(&and),
                        }
                    }
                    And | Or | Xor => {
                        let s = va.lin.add(&vb.lin).add_c(-1.0);
                        let r = self.relu(&s, -1.0, 1.0);
                        match op {
                            And => r,
                            Or => va.lin.add(&vb.lin).sub(&r),
                            _ => va.lin.add(&vb.lin).axpy(-2.0, &r),
                        }
                    }
                }
            }
            Op::Un(op, a) => {
                let va = get(a);
                use UnOp::*;
                match op {
                    Neg => va.lin.scale(-1.0),
                    Not => Lin::constant(1.0).sub(&va.lin),
                    Relu => self.relu_v(&va),
                    Abs => {
                        let p = self.relu_v(&va);
                        let q = self.relu_v(&vaff(&va, -1.0, 0.0));
                        p.add(&q)
                    }
                    Floor | Ceil | Trunc => {
                        if va.s.is_some() {
                            self.lookup_fn(&va, "floor", |x| eval_un(op, x))?
                        } else {
                            match op {
                                Floor => self.floor_real(&va)?,
                                Ceil => self.floor_real(&vaff(&va, -1.0, 0.0))?.scale(-1.0),
                                _ => self.trunc_real(&va)?,
                            }
                        }
                    }
                }
            }
            Op::Cmp(op, a, b) => {
                let d = vsub(&get(a), &get(b));
                use CmpOp::*;
                match op {
                    Ge => self.ge0(&d)?,
                    Gt => self.gt0(&d)?,
                    Le => self.ge0(&vaff(&d, -1.0, 0.0))?,
                    Lt => self.gt0(&vaff(&d, -1.0, 0.0))?,
                    Eq => self.eq0(&d)?,
                    Ne => Lin::constant(1.0).sub(&self.eq0(&d)?),
                }
            }
            Op::Table(t) => {
                let t = &self.g.tables[t];
                // mixed-radix index z = sum (x_i - lo_i) * stride_i, then a lookup
                let st = t.strides();
                let mut z = V { lin: Lin::constant(0.0), lo: 0.0, hi: 0.0, kind: Kind::Int, s: None };
                let mut all_s = true;
                for (i, &a) in t.args.iter().enumerate() {
                    let va = get(a);
                    all_s &= va.s.is_some();
                    let term = vaff(&va, st[i] as f64, -t.lo[i] * st[i] as f64);
                    z = if i == 0 { term } else { vadd(&z, &term) };
                }
                if !all_s {
                    z.s = None;
                }
                let n = t.grid.len() as f64;
                z.lo = z.lo.max(0.0);
                z.hi = z.hi.min(n - 1.0);
                let grid = t.grid.clone();
                self.lookup_fn(&z, "table", move |x| grid.get(x as usize).copied().unwrap_or(0.0))?
            }
            Op::Select(c, a, b) => {
                let (vc, vthen, velse) = (get(c), get(a), get(b));
                self.select(&vc, &vthen, &velse)?
            }
        })
    }
}

pub fn lower(g: &Graph, an: &Analysis, o: &Options) -> Result<Lowered> {
    o.validate()?;
    let n = g.nodes.len();
    // reachability from outputs
    let mut live = vec![false; n];
    for &o in &g.outputs {
        live[o] = true;
    }
    for i in (0..n).rev() {
        if live[i] {
            for a in g.args_of(i) {
                live[a] = true;
            }
        }
    }
    let mut uses = vec![0usize; n];
    for &o in &g.outputs {
        uses[o] += 1;
    }
    for i in 0..n {
        if live[i] {
            for a in g.args_of(i) {
                uses[a] += 1;
            }
        }
    }
    // input features: small integer inputs are one-hot encoded (fixed, no
    // parameters) when values are one-hot, so no unit reads a raw range
    use crate::network::Feature;
    let mut features = vec![];
    let mut first_feature = vec![0usize; g.inputs.len()];
    for (i, inp) in g.inputs.iter().enumerate() {
        first_feature[i] = features.len();
        let t = inp.ty;
        let n_vals = t.hi - t.lo + 1.0;
        let encode = o.onehot
            && o.arch != Arch::Transformer
            && inp.token.is_none()
            && t.kind == Kind::Int
            && t.lo.is_finite()
            && t.hi.is_finite()
            && n_vals > 2.0
            && n_vals <= o.onehot_max as f64;
        if encode {
            for k in 0..n_vals as usize {
                features.push(Feature::Eq(i, t.lo + k as f64));
            }
        } else {
            features.push(Feature::Raw(i));
        }
    }
    let bounds: Vec<(f64, f64)> = features
        .iter()
        .map(|f| match *f {
            Feature::Raw(i) => (g.inputs[i].ty.lo, g.inputs[i].ty.hi),
            Feature::Eq(..) => (0.0, 1.0),
        })
        .collect();
    let mut lw = Lowerer { g, o, dag: Dag::new(&bounds), stats: LowerStats::default(), oh: HashMap::new() };
    for (i, inp) in g.inputs.iter().enumerate() {
        if let Some(t) = inp.token {
            lw.dag.tokens.insert(t, first_feature[i]);
        }
    }
    let mut vals: Vec<Option<V>> = vec![None; n];
    for i in 0..n {
        if !live[i] {
            continue;
        }
        let (lo, hi) = (an.lo[i], an.hi[i]);
        let kind = g.nodes[i].ty.kind;
        // inputs read through features (transformer token inputs: attention)
        let token_input = matches!(g.nodes[i].op, Op::Input(k) if g.inputs[k].token.is_some()) && o.arch == Arch::Transformer;
        if let (Op::Input(k), false) = (g.nodes[i].op, token_input) {
            let j = first_feature[k];
            if let Feature::Eq(..) = features[j] {
                let mut ind = vec![];
                let mut lin = Lin::constant(0.0);
                for (jj, f) in features.iter().enumerate().skip(j) {
                    match *f {
                        Feature::Eq(kk, v) if kk == k => {
                            let l = Lin::sig(jj as Sig);
                            lin = lin.axpy(v, &l);
                            ind.push((v, l));
                        }
                        _ => break,
                    }
                }
                lw.oh.insert(i, Rc::new(ind));
                vals[i] = Some(V { lin, lo, hi, kind, s: an.samples[i].clone() });
                continue;
            }
            vals[i] = Some(V { lin: Lin::sig(j as Sig), lo, hi, kind, s: an.samples[i].clone() });
            continue;
        }
        let onehot = if o.onehot && !(lo == hi && lo.is_finite()) {
            let me = V { lin: Lin::constant(0.0), lo, hi, kind, s: an.samples[i].clone() };
            lw.lower_onehot(i, &me, &vals)?
        } else {
            None
        };
        let lin = if lo == hi && lo.is_finite() {
            Lin::constant(lo)
        } else if let Some((l, oh)) = onehot {
            lw.oh.insert(i, oh);
            l
        } else {
            let l = lw.lower_node(i, &vals)?;
            let (tl, th) = lw.dag.bounds_with(&l, Some((lo, hi)));
            lw.materialize(l, tl, th, uses[i])
        };
        vals[i] = Some(V { lin, lo, hi, kind, s: an.samples[i].clone() });
    }
    let outputs = g.outputs.iter().map(|&o| vals[o].as_ref().unwrap().lin.clone()).collect();
    let node_lins = vals.into_iter().map(|v| v.map(|v| v.lin)).collect();
    Ok(Lowered { features, dag: lw.dag, outputs, stats: lw.stats, node_lins })
}
