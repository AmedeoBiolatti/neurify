//! neurify-frontend: Python source -> neurify-core IR.
//!
//! [`trace`] parses a function with `rustpython-parser` and runs the
//! symbolic interpreter in [`interp`] over declared input domains. Names the
//! source does not define (globals, helper functions, modules, opaque
//! callables) are resolved through a [`Host`] — the Python process in the
//! `neurify` package.

pub mod interp;
pub mod value;

pub use interp::{batcher_pairs, trace, TraceError, TraceOptions, Traced};
pub use value::{Host, InSpec, Key, NoHost, OutSpec, Resolved, Value};

#[cfg(test)]
mod tests {
    use super::*;
    use neurify_core::{compile, Arch, Method, Options};

    fn run(src: &str, name: &str, params: &[(&str, InSpec)]) -> Traced {
        let params: Vec<(String, InSpec)> = params.iter().map(|(n, s)| (n.to_string(), s.clone())).collect();
        trace(src, name, "<test>", 1, 0, None, &params, &mut NoHost, TraceOptions::default())
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// compile under every arch/method, compare against `reference` on `pts`
    fn check(t: &Traced, pts: &[Vec<f64>], reference: impl Fn(&[f64]) -> Vec<f64>) {
        for arch in [Arch::Mlp, Arch::ResMlp, Arch::Glu] {
            for method in [Method::Depth, Method::Size, Method::Trainable, Method::Sparse] {
                let o = Options { arch, method, ..Default::default() };
                let (net, rep) = compile(&t.graph, &o).unwrap_or_else(|e| panic!("{arch:?}/{method:?}: {e}"));
                assert!(rep.exact(), "{arch:?}/{method:?} not exact");
                for x in pts {
                    let want = reference(x);
                    let ir = t.graph.eval(x).unwrap();
                    let got = net.forward(x);
                    for k in 0..want.len() {
                        assert!((want[k] - ir[k]).abs() < 1e-9, "IR mismatch at {x:?}: {want:?} vs {ir:?}");
                        assert!((want[k] - got[k]).abs() < 1e-7, "{arch:?}/{method:?} at {x:?}: {want:?} vs {got:?}");
                    }
                }
            }
        }
    }

    fn grid2(a: (i64, i64), b: (i64, i64)) -> Vec<Vec<f64>> {
        (a.0..=a.1).flat_map(|x| (b.0..=b.1).map(move |y| vec![x as f64, y as f64])).collect()
    }

    #[test]
    fn if_else_and_early_return() {
        let src = r#"
def f(a, b):
    if a > b:
        return a - b
    elif a == b:
        return 0
    m = max(a, b) // 2
    return m * 3 + (1 if a % 2 == 0 else -1)
"#;
        let t = run(src, "f", &[("a", InSpec::Int(-6.0, 9.0)), ("b", InSpec::Int(0.0, 7.0))]);
        check(&t, &grid2((-6, 9), (0, 7)), |x| {
            let (a, b) = (x[0] as i64, x[1] as i64);
            let r = if a > b {
                a - b
            } else if a == b {
                0
            } else {
                a.max(b).div_euclid(2) * 3 + if a.rem_euclid(2) == 0 { 1 } else { -1 }
            };
            vec![r as f64]
        });
    }

    #[test]
    fn loops_break_and_helpers() {
        let src = r#"
def helper(x, k=2):
    return x * k

def f(n, x):
    s = 0
    for i in range(n):
        if i * x > 20:
            break
        s += helper(i)
    while x > 0:
        x -= 3
        s += 1
    return s, x
"#;
        let t = run(src, "f", &[("n", InSpec::Int(0.0, 8.0)), ("x", InSpec::Int(0.0, 10.0))]);
        check(&t, &grid2((0, 8), (0, 10)), |v| {
            let (n, mut x) = (v[0] as i64, v[1] as i64);
            let mut s = 0;
            for i in 0..n {
                if i * x > 20 {
                    break;
                }
                s += 2 * i;
            }
            while x > 0 {
                x -= 3;
                s += 1;
            }
            vec![s as f64, x as f64]
        });
    }

    #[test]
    fn dicts_lists_and_sorting() {
        let src = r#"
def f(xs, cfg):
    d = {"lo": min(xs), "hi": max(xs)}
    d["span"] = d["hi"] - d["lo"]
    ys = sorted(xs)
    for i in range(len(ys)):
        if ys[i] > cfg["t"]:
            ys[i] = cfg["t"]
    return d, ys
"#;
        let t = run(
            src,
            "f",
            &[
                ("xs", InSpec::List(vec![InSpec::Int(0.0, 5.0); 4])),
                ("cfg", InSpec::Dict(vec![(Key::Str("t".into()), InSpec::Int(1.0, 4.0))])),
            ],
        );
        let mut pts = vec![];
        for a in 0..=5 {
            for b in 0..=5 {
                for c in [0, 3, 5] {
                    for t in 1..=4 {
                        pts.push(vec![a as f64, b as f64, c as f64, 2.0, t as f64]);
                    }
                }
            }
        }
        check(&t, &pts, |v| {
            let xs: Vec<f64> = v[..4].to_vec();
            let lo = xs.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut ys = xs.clone();
            ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ys: Vec<f64> = ys.iter().map(|&y| y.min(v[4])).collect();
            let mut out = vec![lo, hi, hi - lo];
            out.extend(ys);
            out
        });
    }

    #[test]
    fn symbolic_index_and_mutation_through_alias() {
        let src = r#"
def swap(a, i, j):
    a[i], a[j] = a[j], a[i]

def f(xs, k):
    for i in range(3):
        if xs[i] > xs[i + 1]:
            swap(xs, i, i + 1)
    return xs[k]
"#;
        let t = run(
            src,
            "f",
            &[("xs", InSpec::List(vec![InSpec::Int(0.0, 3.0); 4])), ("k", InSpec::Int(0.0, 3.0))],
        );
        let mut pts = vec![];
        for code in 0..256 {
            for k in 0..4 {
                let xs: Vec<f64> = (0..4).map(|i| ((code >> (2 * i)) & 3) as f64).collect();
                let mut p = xs.clone();
                p.push(k as f64);
                pts.push(p);
            }
        }
        check(&t, &pts, |v| {
            let mut xs = v[..4].to_vec();
            for i in 0..3 {
                if xs[i] > xs[i + 1] {
                    xs.swap(i, i + 1);
                }
            }
            vec![xs[v[4] as usize]]
        });
    }

    #[test]
    fn nested_symbolic_store_and_count() {
        // drop a piece into column c of a 3x2 grid (row found by a symbolic scan)
        let src = r#"
def f(cells, c, m):
    grid = [[cells[r * 2 + k] for k in range(2)] for r in range(3)]
    g = [row[:] for row in grid]
    for row in range(2, -1, -1):
        if g[row][c] == 0:
            break
    g[row][c] = m
    flat = [x for r in g for x in r]
    return flat, flat.count(m)
"#;
        let t = run(
            src,
            "f",
            &[
                ("cells", InSpec::List(vec![InSpec::Int(0.0, 2.0); 6])),
                ("c", InSpec::Int(0.0, 1.0)),
                ("m", InSpec::Int(1.0, 2.0)),
            ],
        );
        let mut pts = vec![];
        for code in 0..729 {
            for c in 0..2 {
                for m in 1..3 {
                    let mut p: Vec<f64> = (0..6).map(|i| ((code / 3i32.pow(i)) % 3) as f64).collect();
                    p.push(c as f64);
                    p.push(m as f64);
                    pts.push(p);
                }
            }
        }
        check(&t, &pts, |v| {
            let mut g: Vec<Vec<f64>> = (0..3).map(|r| v[r * 2..r * 2 + 2].to_vec()).collect();
            let (c, m) = (v[6] as usize, v[7]);
            let mut row = 0;
            for r in (0..3).rev() {
                row = r;
                if g[r][c] == 0.0 {
                    break;
                }
            }
            g[row][c] = m;
            let mut flat: Vec<f64> = g.concat();
            let n = flat.iter().filter(|&&x| x == m).count() as f64;
            flat.push(n);
            flat
        });
    }

    #[test]
    fn unprovable_loop_is_rejected() {
        let src = "def f(x):\n    while x != 0:\n        x = x - 2\n    return x\n";
        let params = vec![("x".to_string(), InSpec::Int(0.0, 9.0))];
        let o = TraceOptions { max_loop_iters: 20, ..Default::default() };
        let r = trace(src, "f", "<t>", 1, 0, None, &params, &mut NoHost, o);
        assert!(r.is_err(), "odd x never terminates; must be rejected");
    }

    #[test]
    fn batcher_sorts() {
        for n in 1..12usize {
            for code in 0..(1u32 << n) {
                let mut v: Vec<u32> = (0..n).map(|i| (code >> i) & 1).collect();
                for (i, j) in batcher_pairs(n) {
                    if v[i] > v[j] {
                        v.swap(i, j);
                    }
                }
                assert!(v.windows(2).all(|w| w[0] <= w[1]), "n={n}");
            }
        }
    }
}
