//! Neuron-level DAG.
//!
//! After lowering, a program is a DAG of *units* over *signals*. Signals are
//! the network inputs followed by unit outputs. Program values are affine
//! forms ([`Lin`]) over signals: affine arithmetic is free (it is absorbed
//! into weight matrices), only units cost depth and width.
//!
//! Unit kinds:
//! * `Relu(a)`       = max(0, a)
//! * `Gated(a, b)`   = max(0, a) * b      (ReGLU unit, `glu` architecture only)

use crate::ir::{imul, mulz};
use std::collections::HashMap;

pub type Sig = u32;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lin {
    /// sorted by signal, no duplicate signals, no zero coefficients
    pub terms: Vec<(Sig, f64)>,
    pub c: f64,
}

impl Lin {
    pub fn constant(c: f64) -> Lin {
        Lin { terms: vec![], c }
    }
    pub fn sig(s: Sig) -> Lin {
        Lin { terms: vec![(s, 1.0)], c: 0.0 }
    }
    pub fn as_const(&self) -> Option<f64> {
        self.terms.is_empty().then_some(self.c)
    }
    pub fn axpy(&self, k: f64, o: &Lin) -> Lin {
        // self + k * o
        let mut terms = Vec::with_capacity(self.terms.len() + o.terms.len());
        let (mut i, mut j) = (0, 0);
        while i < self.terms.len() || j < o.terms.len() {
            let take_left = j >= o.terms.len() || (i < self.terms.len() && self.terms[i].0 < o.terms[j].0);
            let take_right = i >= self.terms.len() || (j < o.terms.len() && o.terms[j].0 < self.terms[i].0);
            let (s, v) = if take_left {
                i += 1;
                self.terms[i - 1]
            } else if take_right {
                j += 1;
                (o.terms[j - 1].0, k * o.terms[j - 1].1)
            } else {
                i += 1;
                j += 1;
                (self.terms[i - 1].0, self.terms[i - 1].1 + k * o.terms[j - 1].1)
            };
            if v != 0.0 {
                terms.push((s, v));
            }
        }
        Lin { terms, c: self.c + k * o.c }
    }
    pub fn add(&self, o: &Lin) -> Lin {
        self.axpy(1.0, o)
    }
    pub fn sub(&self, o: &Lin) -> Lin {
        self.axpy(-1.0, o)
    }
    pub fn scale(&self, k: f64) -> Lin {
        if k == 0.0 {
            return Lin::constant(0.0);
        }
        Lin { terms: self.terms.iter().map(|&(s, v)| (s, v * k)).collect(), c: self.c * k }
    }
    pub fn add_c(&self, k: f64) -> Lin {
        Lin { terms: self.terms.clone(), c: self.c + k }
    }
    fn key(&self) -> Vec<u64> {
        let mut k = Vec::with_capacity(2 * self.terms.len() + 1);
        for &(s, v) in &self.terms {
            k.push(s as u64);
            k.push((v + 0.0).to_bits());
        }
        k.push((self.c + 0.0).to_bits());
        k
    }
}

#[derive(Clone, Debug)]
pub enum Unit {
    Relu(Lin),
    Gated(Lin, Lin),
}

impl Unit {
    pub fn args(&self) -> Vec<&Lin> {
        match self {
            Unit::Relu(a) => vec![a],
            Unit::Gated(a, b) => vec![a, b],
        }
    }
}

/// Attention read from the main token: value of sequence feature `feature`
/// at position round(query) (transformer architecture only). `query_sq` is
/// query^2 as a signal, needed by exact ReLU attention.
#[derive(Clone, Debug)]
pub struct AttnRead {
    pub feature: usize,
    pub query: Lin,
    pub query_sq: Option<Lin>,
}

#[derive(Clone, Debug)]
pub enum SigKind {
    Input(usize),
    Unit(Unit),
    Attn(AttnRead),
}

impl SigKind {
    /// Signals this one is computed from.
    pub fn deps(&self) -> Vec<Sig> {
        match self {
            SigKind::Input(_) => vec![],
            SigKind::Unit(u) => u.args().iter().flat_map(|a| a.terms.iter().map(|t| t.0)).collect(),
            SigKind::Attn(a) => a
                .query
                .terms
                .iter()
                .chain(a.query_sq.iter().flat_map(|q| q.terms.iter()))
                .map(|t| t.0)
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SigInfo {
    pub kind: SigKind,
    pub lo: f64,
    pub hi: f64,
}

#[derive(Clone, Debug)]
pub struct Dag {
    pub n_inputs: usize,
    pub sigs: Vec<SigInfo>,
    /// (feature, position) -> input index, for sequence-token inputs
    pub tokens: HashMap<(usize, usize), usize>,
    memo: HashMap<(u8, Vec<u64>), Sig>,
}

/// Positive normalisation a = k * a' with k a power of two chosen so the
/// unit's activation stays O(1) (from the value bound `mag`, falling back to
/// the largest coefficient). relu(k a) = k relu(a); powers of two keep every
/// weight dyadic, so integer programs evaluate exactly in floating point, and
/// O(1) activations prevent exponent drift through long chains.
fn normalise(a: &Lin, mag: f64) -> (f64, Lin) {
    let m = if mag.is_finite() && mag > 0.0 {
        mag
    } else {
        a.terms.iter().fold(0.0f64, |m, &(_, v)| m.max(v.abs()))
    };
    if m == 0.0 || !m.is_finite() {
        return (1.0, a.clone());
    }
    let k = 2f64.powi(m.log2().floor() as i32);
    if k == 1.0 {
        return (1.0, a.clone());
    }
    (k, a.scale(1.0 / k))
}

impl Dag {
    pub fn new(input_bounds: &[(f64, f64)]) -> Dag {
        let sigs = input_bounds
            .iter()
            .enumerate()
            .map(|(i, &(lo, hi))| SigInfo { kind: SigKind::Input(i), lo, hi })
            .collect();
        Dag { n_inputs: input_bounds.len(), sigs, tokens: HashMap::new(), memo: HashMap::new() }
    }

    pub fn n_units(&self) -> usize {
        self.sigs.len() - self.n_inputs
    }

    /// Sound bounds of an affine form from per-signal bounds.
    pub fn bounds(&self, a: &Lin) -> (f64, f64) {
        let (mut lo, mut hi) = (a.c, a.c);
        for &(s, v) in &a.terms {
            let si = &self.sigs[s as usize];
            let (l, h) = (mulz(v, si.lo), mulz(v, si.hi));
            lo += l.min(h);
            hi += l.max(h);
        }
        (lo, hi)
    }

    /// Bounds of `a`, intersected with externally known sound bounds.
    pub fn bounds_with(&self, a: &Lin, known: Option<(f64, f64)>) -> (f64, f64) {
        let (lo, hi) = self.bounds(a);
        match known {
            Some((l, h)) => (lo.max(l), hi.min(h)),
            None => (lo, hi),
        }
    }

    fn push_unit(&mut self, tag: u8, key: Vec<u64>, unit: Unit, lo: f64, hi: f64) -> Sig {
        if let Some(&s) = self.memo.get(&(tag, key.clone())) {
            let si = &mut self.sigs[s as usize];
            si.lo = si.lo.max(lo);
            si.hi = si.hi.min(hi);
            return s;
        }
        let s = self.sigs.len() as Sig;
        self.sigs.push(SigInfo { kind: SigKind::Unit(unit), lo, hi });
        self.memo.insert((tag, key), s);
        s
    }

    /// relu(a). `known`: sound bounds on `a` if the caller has better ones.
    pub fn relu(&mut self, a: &Lin, known: Option<(f64, f64)>) -> Lin {
        let (lo, hi) = self.bounds_with(a, known);
        if lo >= 0.0 {
            return a.clone();
        }
        if hi <= 0.0 {
            return Lin::constant(0.0);
        }
        let (k, an) = normalise(a, hi);
        let s = self.push_unit(0, an.key(), Unit::Relu(an), 0.0, hi.max(0.0) / k);
        Lin::sig(s).scale(k)
    }

    /// A unit relu(a) even when bounds say it is the identity (used to
    /// materialise ill-conditioned affine forms).
    pub fn relu_force(&mut self, a: &Lin, lo: f64, hi: f64) -> Lin {
        let (k, an) = normalise(a, hi);
        let s = self.push_unit(0, an.key(), Unit::Relu(an), lo.max(0.0) / k, hi.max(0.0) / k);
        Lin::sig(s).scale(k)
    }

    /// Sum of |coef| * max|signal| over the terms (scale of intermediate values
    /// when the form is evaluated).
    pub fn magnitude(&self, a: &Lin) -> f64 {
        a.terms.iter().fold(0.0, |m, &(s, v)| {
            let si = &self.sigs[s as usize];
            m + v.abs() * si.lo.abs().max(si.hi.abs())
        })
    }

    /// Attention read of `feature` at position `query` (value bounds lo..hi).
    pub fn attn(&mut self, feature: usize, query: &Lin, query_sq: Option<Lin>, lo: f64, hi: f64) -> Lin {
        let mut key = vec![feature as u64];
        key.extend(query.key());
        if let Some(&s) = self.memo.get(&(2, key.clone())) {
            return Lin::sig(s);
        }
        let s = self.sigs.len() as Sig;
        self.sigs.push(SigInfo { kind: SigKind::Attn(AttnRead { feature, query: query.clone(), query_sq }), lo, hi });
        self.memo.insert((2, key), s);
        Lin::sig(s)
    }

    /// relu(a) * b. Callers use it with a in {0,1} (gating) or for products.
    pub fn gated(&mut self, a: &Lin, b: &Lin, ka: Option<(f64, f64)>, kb: Option<(f64, f64)>) -> Lin {
        let (alo, ahi) = self.bounds_with(a, ka);
        if ahi <= 0.0 {
            return Lin::constant(0.0);
        }
        if let Some(bc) = b.as_const() {
            return self.relu(a, ka).scale(bc);
        }
        if let Some(ac) = a.as_const() {
            return b.scale(ac.max(0.0));
        }
        let (blo, bhi) = self.bounds_with(b, kb);
        let (k1, an) = normalise(a, ahi);
        let (k2, bn) = normalise(b, blo.abs().max(bhi.abs()));
        let mut key = an.key();
        key.push(u64::MAX);
        key.extend(bn.key());
        let ra = (alo.max(0.0) / k1, ahi.max(0.0) / k1);
        let (lo, hi) = imul(ra, (blo / k2, bhi / k2));
        let s = self.push_unit(1, key, Unit::Gated(an, bn), lo, hi);
        Lin::sig(s).scale(k1 * k2)
    }

    /// Evaluate all signals for one input vector (reference semantics).
    pub fn eval(&self, x: &[f64]) -> Vec<f64> {
        let mut v = vec![0.0; self.sigs.len()];
        let ev = |l: &Lin, v: &[f64]| l.terms.iter().fold(l.c, |acc, &(s, w)| acc + w * v[s as usize]);
        for (i, si) in self.sigs.iter().enumerate() {
            v[i] = match &si.kind {
                SigKind::Input(k) => x[*k],
                SigKind::Unit(Unit::Relu(a)) => ev(a, &v).max(0.0),
                SigKind::Unit(Unit::Gated(a, b)) => ev(a, &v).max(0.0) * ev(b, &v),
                SigKind::Attn(a) => {
                    let p = ev(&a.query, &v).round();
                    if p < 0.0 {
                        0.0
                    } else {
                        self.tokens.get(&(a.feature, p as usize)).map_or(0.0, |&k| x[k])
                    }
                }
            };
        }
        v
    }

    pub fn eval_lin(l: &Lin, v: &[f64]) -> f64 {
        l.terms.iter().fold(l.c, |acc, &(s, w)| acc + w * v[s as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lin_algebra() {
        let a = Lin::sig(0).add(&Lin::sig(1).scale(2.0)).add_c(1.0);
        let b = Lin::sig(1).scale(2.0);
        let d = a.sub(&b);
        assert_eq!(d.terms, vec![(0, 1.0)]);
        assert_eq!(d.c, 1.0);
    }

    #[test]
    fn relu_cse_with_scaling() {
        let mut d = Dag::new(&[(-1.0, 1.0)]);
        let x = Lin::sig(0);
        let r1 = d.relu(&x, None);
        let r2 = d.relu(&x.scale(4.0), None);
        assert_eq!(d.n_units(), 1);
        assert_eq!(r2, r1.scale(4.0));
    }
}
