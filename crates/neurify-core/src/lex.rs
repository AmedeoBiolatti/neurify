//! Lexicographic comparison of scores (range reduction for decisions).
//!
//! Heuristic scores often have the form `1e6*a + 1e2*b + c` (with `|c| < 1e2`
//! and so on) or `select(legal, score, -1e9)`. Comparing two such scores
//! directly needs integer steps over a huge range: large weights, and no
//! exactness at all in float32 once the range exceeds 2^24. If the score
//! decomposes into *tiers* whose scales dominate everything below them, then
//! `s_i > s_j` is a lexicographic comparison of small integers, one tier at a
//! time. This module finds such decompositions (with an exactness proof from
//! interval bounds) and emits the comparison in IR.

use std::collections::BTreeMap;

use crate::error::Result;
use crate::ir::{BinOp, CmpOp, Graph, Kind, NodeId, Op, UnOp};

/// Integer affine form `sum c_k * atom_k + c0` over integral atoms.
#[derive(Clone, Debug, Default)]
struct Affine {
    terms: BTreeMap<NodeId, f64>,
    c0: f64,
}

impl Affine {
    fn add(&mut self, o: &Affine, k: f64) {
        for (&a, &c) in &o.terms {
            *self.terms.entry(a).or_insert(0.0) += k * c;
        }
        self.c0 += k * o.c0;
        self.terms.retain(|_, c| *c != 0.0);
    }
}

const BIG: f64 = 9.0e15; // integers are exact in f64 below 2^53

fn integral(x: f64) -> bool {
    x.fract() == 0.0 && x.abs() < BIG
}

fn gcd(a: f64, b: f64) -> f64 {
    let (mut a, mut b) = (a.abs() as u64, b.abs() as u64);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a as f64
}

/// Walk Add / Sub / Neg / Mul-by-constant down to atoms.
fn walk(g: &Graph, id: NodeId, k: f64, out: &mut Affine) {
    match g.nodes[id].op {
        Op::Const(v) => out.c0 += k * v,
        Op::Bin(BinOp::Add, a, b) => {
            walk(g, a, k, out);
            walk(g, b, k, out);
        }
        Op::Bin(BinOp::Sub, a, b) => {
            walk(g, a, k, out);
            walk(g, b, -k, out);
        }
        Op::Un(UnOp::Neg, a) => walk(g, a, -k, out),
        Op::Bin(BinOp::Mul, a, b) if g.const_of(a).is_some() => walk(g, b, k * g.const_of(a).unwrap(), out),
        Op::Bin(BinOp::Mul, a, b) if g.const_of(b).is_some() => walk(g, a, k * g.const_of(b).unwrap(), out),
        _ => *out.terms.entry(id).or_insert(0.0) += k,
    }
}

fn affine_of(g: &Graph, id: NodeId) -> Option<Affine> {
    let mut a = Affine::default();
    walk(g, id, 1.0, &mut a);
    a.terms.retain(|_, c| *c != 0.0);
    let ok = integral(a.c0)
        && a.terms.iter().all(|(&x, &c)| {
            let t = g.ty(x);
            integral(c) && t.kind.integral() && t.lo.is_finite() && t.hi.is_finite()
        });
    ok.then_some(a)
}

/// Interval of `sum c * atom` (+ c0).
fn bounds(g: &Graph, terms: &[(NodeId, f64)], c0: f64) -> (f64, f64) {
    let (mut lo, mut hi) = (c0, c0);
    for &(x, c) in terms {
        let t = g.ty(x);
        let (a, b) = (c * t.lo, c * t.hi);
        lo += a.min(b);
        hi += a.max(b);
    }
    (lo, hi)
}

fn gcd_all(terms: &[(NodeId, f64)], c0: f64) -> f64 {
    terms.iter().fold(c0.abs(), |acc, &(_, c)| gcd(acc, c))
}

/// One tier: `sum c * atom + c0`, all multiples of `scale`.
#[derive(Clone, Debug)]
struct Tier {
    terms: Vec<(NodeId, f64)>,
    c0: f64,
}

/// Split terms (ascending |c|) at every level where `accept(high, low)` holds.
/// Tiers are returned most significant first.
fn split(terms: Vec<(NodeId, f64)>, accept: impl Fn(&[(NodeId, f64)], &[(NodeId, f64)]) -> bool) -> Vec<Vec<(NodeId, f64)>> {
    let mut terms = terms;
    terms.sort_by(|a, b| a.1.abs().total_cmp(&b.1.abs()).then(a.0.cmp(&b.0)));
    let mut cuts = vec![];
    for p in 1..terms.len() {
        if terms[p].1.abs() != terms[p - 1].1.abs() && accept(&terms[p..], &terms[..p]) {
            cuts.push(p);
        }
    }
    let mut tiers = vec![];
    let mut start = 0;
    for &c in cuts.iter().chain(std::iter::once(&terms.len())) {
        tiers.push(terms[start..c].to_vec());
        start = c;
    }
    tiers.reverse();
    tiers
}

/// Rigorous tiers of a difference `delta = terms + c0`: sign(delta) is the
/// lexicographic sign of the tier sums. None if there is only one tier.
fn tiers_of_difference(g: &Graph, d: &Affine) -> Option<Vec<Tier>> {
    let terms: Vec<_> = d.terms.iter().map(|(&a, &c)| (a, c)).collect();
    let maxabs = |t: &[(NodeId, f64)], c0: f64| {
        let (lo, hi) = bounds(g, t, c0);
        lo.abs().max(hi.abs())
    };
    // candidate tiers from the terms alone, then place the constant in the
    // most significant tier whose gcd divides it -- or, if none does, split
    // it into a multiple of the last tier's gcd plus a small remainder that
    // forms a constant-only bottom tier (e.g. a per-move bonus 0..3 under
    // coefficients 1000 / 100) -- and re-check every cut
    let mut groups = split(terms, |high, low| gcd_all(high, 0.0) > maxabs(low, 0.0));
    loop {
        if groups.is_empty() {
            return None;
        }
        let mut c0s = vec![0.0; groups.len()];
        let mut extra = 0.0;
        if d.c0 != 0.0 {
            match (0..groups.len()).find(|&k| d.c0 % gcd_all(&groups[k], 0.0) == 0.0) {
                Some(k) => c0s[k] = d.c0,
                None => {
                    let last = groups.len() - 1;
                    let gl = gcd_all(&groups[last], 0.0);
                    let mut r = d.c0 % gl;
                    if 2.0 * r.abs() > gl {
                        r -= gl * r.signum();
                    }
                    c0s[last] = d.c0 - r;
                    extra = r;
                }
            }
        }
        let mut tiers: Vec<(Vec<(NodeId, f64)>, f64)> = groups.iter().cloned().zip(c0s).collect();
        if extra != 0.0 {
            tiers.push((vec![], extra));
        }
        if tiers.len() <= 1 {
            return None;
        }
        // cut between tiers[k-1] (above) and tiers[k] (below)
        let bad = (1..tiers.len()).find(|&k| {
            let above: Vec<_> = tiers[..k].iter().flat_map(|t| t.0.clone()).collect();
            let below: Vec<_> = tiers[k..].iter().flat_map(|t| t.0.clone()).collect();
            let c_above: f64 = tiers[..k].iter().map(|t| t.1).sum();
            let c_below: f64 = tiers[k..].iter().map(|t| t.1).sum();
            gcd_all(&above, c_above) <= maxabs(&below, c_below)
        });
        match bad {
            None => {
                return Some(tiers.into_iter().map(|(terms, c0)| Tier { terms, c0 }).collect());
            }
            Some(k) if k >= groups.len() => return None, // the constant tier itself
            Some(k) => {
                let below = groups.remove(k);
                groups[k - 1].extend(below);
            }
        }
    }
}

fn int_const(g: &mut Graph, v: f64) -> Result<NodeId> {
    g.constant(v, Kind::Int)
}

/// IR node for `sum c * atom + c0` (integers).
fn emit(g: &mut Graph, terms: &[(NodeId, f64)], c0: f64) -> Result<NodeId> {
    let mut acc = int_const(g, c0)?;
    for &(x, c) in terms {
        let t = if c == 1.0 {
            x
        } else {
            let k = int_const(g, c)?;
            g.bin(BinOp::Mul, k, x)?
        };
        acc = g.bin(BinOp::Add, acc, t)?;
    }
    Ok(acc)
}

/// Order-preserving coarse form of one score, over small-range atoms.
#[derive(Clone, Debug)]
struct Coarse {
    form: Affine,
}

/// Scores prepared for pairwise lexicographic comparison.
pub struct LexScores {
    coarse: Vec<Option<Coarse>>,
    pub tiered: usize,
}

/// `select(c, a, K)` / `select(c, K, a)` with constant K: (flag, value, K).
fn sentinel(g: &mut Graph, s: NodeId) -> Result<Option<(NodeId, NodeId, f64)>> {
    if let Op::Select(c, a, b) = g.nodes[s].op {
        if let Some(k) = g.const_of(b) {
            return Ok(Some((c, a, k)));
        }
        if let Some(k) = g.const_of(a) {
            let nc = g.un(UnOp::Not, c)?;
            return Ok(Some((nc, b, k)));
        }
    }
    Ok(None)
}

/// Prepare scores: strip a common sentinel (`select(legal, x, -1e9)`), group
/// each score into tiers (`T_k`, new IR nodes) and mask them by the flag.
pub fn prepare(g: &mut Graph, scores: &[NodeId]) -> Result<LexScores> {
    let mut parts = vec![];
    for &s in scores {
        parts.push(sentinel(g, s)?);
    }
    // a common sentinel K below (or above) every non-sentinel value
    let values: Vec<NodeId> = scores.iter().zip(&parts).map(|(&s, p)| p.map_or(s, |(_, a, _)| a)).collect();
    let lo = values.iter().map(|&v| g.ty(v).lo).fold(f64::INFINITY, f64::min);
    let hi = values.iter().map(|&v| g.ty(v).hi).fold(f64::NEG_INFINITY, f64::max);
    let ks: Vec<f64> = parts.iter().flatten().map(|p| p.2).collect();
    let common = ks.first().copied().filter(|&k| ks.iter().all(|&x| x == k));
    // sigma = +1: K below all values, -1: above
    let sigma = match common {
        Some(k) if k < lo => 1.0,
        Some(k) if k > hi => -1.0,
        _ => 0.0,
    };
    if sigma == 0.0 {
        parts.iter_mut().for_each(|p| *p = None);
    }
    // per-score affine forms and tier groups (any grouping is exact; this
    // one keeps tiers small)
    let mut forms = vec![];
    for (i, &s) in scores.iter().enumerate() {
        let value = parts[i].map_or(s, |(_, a, _)| a);
        forms.push(affine_of(g, value).map(|aff| {
            let terms: Vec<_> = aff.terms.iter().map(|(&a, &c)| (a, c)).collect();
            let groups = split(terms, |high, low| {
                let (l, h) = bounds(g, low, 0.0);
                gcd_all(high, 0.0) > h - l
            });
            (aff.c0, groups)
        }));
    }
    let reach = lo.abs().max(hi.abs());
    // phi(K) = 0, phi(v) = v + sigma*S: strictly monotone, S dominates every
    // value, and S is a multiple of every tier scale (so the flag forms its
    // own tier without merging the tiers below it)
    let mut unit = 1.0f64;
    for (_, groups) in forms.iter().flatten() {
        for grp in groups {
            let sc = gcd_all(grp, 0.0);
            unit = unit / gcd(unit, sc) * sc;
        }
    }
    if !integral(unit) || unit > 1e12 {
        unit = 1.0;
    }
    let shift = if sigma != 0.0 && reach.is_finite() { ((2.0 * reach + 2.0) / unit).ceil() * unit } else { 0.0 };
    if !integral(shift) {
        return Ok(LexScores { coarse: vec![None; scores.len()], tiered: 0 });
    }
    let mut coarse = vec![];
    let mut tiered = 0;
    for (i, f) in forms.into_iter().enumerate() {
        let flag = parts[i].map(|(c, _, _)| c);
        let Some((c0, groups)) = f else {
            coarse.push(None);
            continue;
        };
        let mut form = Affine::default();
        if groups.len() > 1 {
            tiered += 1;
        }
        for grp in &groups {
            let scale = gcd_all(grp, 0.0);
            let unit: Vec<_> = grp.iter().map(|&(a, c)| (a, c / scale)).collect();
            let t = if unit.len() == 1 && unit[0].1 == 1.0 { unit[0].0 } else { emit(g, &unit, 0.0)? };
            let t = match flag {
                Some(c) => {
                    let z = int_const(g, 0.0)?;
                    g.select(c, t, z)?
                }
                None => t,
            };
            *form.terms.entry(t).or_insert(0.0) += scale;
        }
        match flag {
            Some(c) => {
                // the flag's coefficient must stay a multiple of every tier
                // scale: c0 = q * unit + r, and r (|r| <= unit/2) becomes its
                // own masked constant atom select(c, r, 0) in the lowest tier
                let mut r = c0 % unit;
                if 2.0 * r.abs() > unit {
                    r -= unit * r.signum();
                }
                *form.terms.entry(c).or_insert(0.0) += c0 - r + sigma * shift;
                if r != 0.0 {
                    let rk = int_const(g, r)?;
                    let z = int_const(g, 0.0)?;
                    let t = g.select(c, rk, z)?;
                    *form.terms.entry(t).or_insert(0.0) += 1.0;
                }
            }
            None => form.c0 += c0 + sigma * shift,
        }
        form.terms.retain(|_, c| *c != 0.0);
        coarse.push(Some(Coarse { form }));
    }
    Ok(LexScores { coarse, tiered })
}

/// `[s_a > s_b]` as a lexicographic comparison when a tier decomposition of
/// the difference exists; `None` means: use a direct comparison.
pub fn greater(g: &mut Graph, ls: &LexScores, a: usize, b: usize) -> Result<Option<NodeId>> {
    let (Some(ca), Some(cb)) = (&ls.coarse[a], &ls.coarse[b]) else {
        return Ok(None);
    };
    let mut d = ca.form.clone();
    d.add(&cb.form, -1.0);
    let Some(tiers) = tiers_of_difference(g, &d) else {
        return Ok(None);
    };
    // lexval = sum_k 3^(m-1-k) * sign(D_k); [delta > 0] = [lexval > 0]
    let zero = int_const(g, 0.0)?;
    let m = tiers.len();
    let mut lexval = zero;
    for (k, t) in tiers.iter().enumerate() {
        let scale = gcd_all(&t.terms, t.c0);
        let unit: Vec<_> = t.terms.iter().map(|&(x, c)| (x, c / scale)).collect();
        let dk = emit(g, &unit, t.c0 / scale)?;
        let gt = g.cmp(CmpOp::Gt, dk, zero)?;
        let w = int_const(g, 3f64.powi((m - 1 - k) as i32))?;
        let mut sgn = gt;
        if k + 1 < m {
            let lt = g.cmp(CmpOp::Lt, dk, zero)?;
            sgn = g.bin(BinOp::Sub, gt, lt)?;
        }
        let term = g.bin(BinOp::Mul, w, sgn)?;
        lexval = g.bin(BinOp::Add, lexval, term)?;
    }
    // the last tier only contributes its "greater" bit: lexval > 0 iff some
    // earlier tier is positive, or all earlier are 0 and the last is positive
    Ok(Some(g.cmp(CmpOp::Gt, lexval, zero)?))
}
