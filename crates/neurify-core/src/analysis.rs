//! Static analysis feeding the gadget selection.
//!
//! * Interval bounds come from the IR builder.
//! * **Exhaustive value analysis**: every node that depends only on finite
//!   Int/Bool inputs is evaluated on the *whole* joint input domain (when it
//!   is small enough). This yields exact bounds and the exact set of values a
//!   node can take, which lets lookup gadgets interpolate only through
//!   reachable points. Consequences: real-valued comparisons become exact,
//!   any binary op becomes a depth-1 lookup, and bounds stop blowing up
//!   through loops.

use crate::ir::*;
use std::rc::Rc;

pub struct Analysis {
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
    /// Joint samples per node (same enumeration order for every node), or
    /// None when the node depends on a real / unbounded input or analysis
    /// was skipped.
    pub samples: Vec<Option<Rc<Vec<f64>>>>,
    pub n_samples: usize,
}

pub fn analyze(g: &Graph, max_points: usize, max_cells: usize) -> Analysis {
    let n = g.nodes.len();
    let mut lo: Vec<f64> = g.nodes.iter().map(|x| x.ty.lo).collect();
    let mut hi: Vec<f64> = g.nodes.iter().map(|x| x.ty.hi).collect();
    let mut samples: Vec<Option<Rc<Vec<f64>>>> = vec![None; n];

    // which nodes depend on a non-enumerable input
    let mut dep = vec![false; n];
    let mut enum_inputs: Vec<(usize, f64, usize)> = vec![]; // (input index, lo, size)
    for (i, node) in g.nodes.iter().enumerate() {
        dep[i] = match node.op {
            Op::Input(k) => {
                let t = node.ty;
                let ok = t.kind.integral() && t.finite();
                if ok {
                    enum_inputs.push((k, t.lo, (t.hi - t.lo) as usize + 1));
                }
                !ok
            }
            Op::Const(_) => false,
            _ => g.args_of(i).iter().any(|&a| dep[a]),
        };
    }
    let mut total: usize = 1;
    for &(_, _, s) in &enum_inputs {
        total = total.saturating_mul(s);
    }
    let n_enum_nodes = dep.iter().filter(|d| !**d).count();
    if max_points == 0 || total > max_points || total.saturating_mul(n_enum_nodes) > max_cells {
        return Analysis { lo, hi, samples, n_samples: 0 };
    }

    // mixed-radix enumeration of the enumerable inputs
    let mut strides = vec![1usize; enum_inputs.len()];
    for i in 1..enum_inputs.len() {
        strides[i] = strides[i - 1] * enum_inputs[i - 1].2;
    }
    let pos_of_input = |k: usize| enum_inputs.iter().position(|e| e.0 == k);

    for i in 0..n {
        if dep[i] {
            continue;
        }
        let col: Vec<f64> = match g.nodes[i].op {
            Op::Input(k) => {
                let p = pos_of_input(k).unwrap();
                let (_, l, s) = enum_inputs[p];
                (0..total).map(|j| l + ((j / strides[p]) % s) as f64).collect()
            }
            Op::Const(c) => vec![c; total],
            Op::Bin(op, a, b) => {
                let (x, y) = (samples[a].as_ref().unwrap(), samples[b].as_ref().unwrap());
                x.iter().zip(y.iter()).map(|(&u, &v)| eval_bin(op, u, v)).collect()
            }
            Op::Un(op, a) => samples[a].as_ref().unwrap().iter().map(|&u| eval_un(op, u)).collect(),
            Op::Cmp(op, a, b) => {
                let (x, y) = (samples[a].as_ref().unwrap(), samples[b].as_ref().unwrap());
                x.iter().zip(y.iter()).map(|(&u, &v)| eval_cmp(op, u, v)).collect()
            }
            Op::Select(c, a, b) => {
                let (cc, x, y) = (
                    samples[c].as_ref().unwrap(),
                    samples[a].as_ref().unwrap(),
                    samples[b].as_ref().unwrap(),
                );
                (0..total).map(|j| if cc[j] != 0.0 { x[j] } else { y[j] }).collect()
            }
            Op::Gather(k) => {
                let gt = &g.gathers[k];
                let idx = samples[gt.index].as_ref().unwrap();
                (0..total)
                    .map(|j| gt.pick(idx[j]).map_or(0.0, |p| samples[gt.elems[p]].as_ref().unwrap()[j]))
                    .collect()
            }
            Op::Table(t) => {
                let t = &g.tables[t];
                let cols: Vec<&Rc<Vec<f64>>> = t.args.iter().map(|&a| samples[a].as_ref().unwrap()).collect();
                (0..total).map(|j| t.get(&cols.iter().map(|c| c[j]).collect::<Vec<_>>())).collect()
            }
        };
        let (l, h) = hull(&col);
        lo[i] = lo[i].max(l);
        hi[i] = hi[i].min(h);
        samples[i] = Some(Rc::new(col));
    }
    Analysis { lo, hi, samples, n_samples: total }
}
