//! Python bindings for neurify-core (module `neurify._core`).

use neurify_core as core;
use neurify_core::network::{Body, Dense};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use neurify_frontend as front;
use neurify_frontend::{InSpec, Key, OutSpec, Resolved, Value};
use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyString, PyTuple};
use std::rc::Rc;

pyo3::create_exception!(_core, CompileError, pyo3::exceptions::PyException);
pyo3::create_exception!(_core, TraceError, pyo3::exceptions::PyException);

fn to_py(e: core::CompileError) -> PyErr {
    CompileError::new_err(format!("[{:?}] {}", e.kind, e.msg))
}

fn batch_rows(values: usize, inputs: usize) -> PyResult<usize> {
    if inputs == 0 {
        if values != 0 {
            return Err(CompileError::new_err("a zero-input graph expects no input values"));
        }
        return Ok(1);
    }
    if values % inputs != 0 {
        return Err(CompileError::new_err(format!(
            "batch must contain complete rows of {inputs} inputs, got {values} values"
        )));
    }
    Ok(values / inputs)
}

#[pyclass(name = "Graph", module = "neurify._core")]
struct PyGraph {
    g: core::Graph,
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new() -> Self {
        PyGraph { g: core::Graph::new() }
    }

    fn input(&mut self, name: &str, kind: &str, lo: f64, hi: f64) -> PyResult<usize> {
        let k = core::Kind::parse(kind).map_err(to_py)?;
        self.g.input(name, k, lo, hi).map_err(to_py)
    }

    fn constant(&mut self, value: f64, kind: &str) -> PyResult<usize> {
        let k = core::Kind::parse(kind).map_err(to_py)?;
        self.g.constant(value, k).map_err(to_py)
    }

    fn binary(&mut self, op: &str, a: usize, b: usize) -> PyResult<usize> {
        let op = core::BinOp::parse(op).map_err(to_py)?;
        self.g.bin(op, a, b).map_err(to_py)
    }

    fn unary(&mut self, op: &str, a: usize) -> PyResult<usize> {
        let op = core::UnOp::parse(op).map_err(to_py)?;
        self.g.un(op, a).map_err(to_py)
    }

    fn compare(&mut self, op: &str, a: usize, b: usize) -> PyResult<usize> {
        let op = core::CmpOp::parse(op).map_err(to_py)?;
        self.g.cmp(op, a, b).map_err(to_py)
    }

    fn select(&mut self, c: usize, a: usize, b: usize) -> PyResult<usize> {
        self.g.select(c, a, b).map_err(to_py)
    }

    /// Tabulated function over a box of integer arguments (row-major grid).
    fn table(&mut self, args: Vec<usize>, lo: Vec<f64>, dims: Vec<usize>, grid: Vec<f64>, kind: &str) -> PyResult<usize> {
        let k = core::Kind::parse(kind).map_err(to_py)?;
        self.g.table(args, lo, dims, grid, k).map_err(to_py)
    }

    /// First node whose lowered value differs from the interpreter at x.
    #[pyo3(signature = (x, arch="mlp", method="depth", robust=true))]
    fn debug_lowering(&self, x: Vec<f64>, arch: &str, method: &str, robust: bool) -> PyResult<Option<(usize, String, f64, f64, (f64, f64))>> {
        let o = core::Options {
            arch: core::Arch::parse(arch).map_err(to_py)?,
            method: core::Method::parse(method).map_err(to_py)?,
            robust,
            ..Default::default()
        };
        core::debug_lowering(&self.g, &o, &x).map_err(to_py)
    }

    /// Op of a node (debugging).
    fn node_op(&self, a: usize) -> PyResult<String> {
        self.g.nodes.get(a)
            .map(|node| format!("{:?}", node.op))
            .ok_or_else(|| CompileError::new_err(format!("node {a} does not exist")))
    }

    /// Bounds of a node after exhaustive value analysis (when applicable).
    #[pyo3(signature = (a, exhaustive_points=65536))]
    fn refined_bounds(&self, a: usize, exhaustive_points: usize) -> PyResult<(f64, f64)> {
        if a >= self.g.nodes.len() {
            return Err(CompileError::new_err(format!("node {a} does not exist")));
        }
        let an = core::analysis::analyze(&self.g, exhaustive_points, 20_000_000);
        Ok((an.lo[a], an.hi[a]))
    }

    fn output(&mut self, a: usize) -> PyResult<()> {
        self.g.output(a).map_err(to_py)
    }

    /// Copy of the graph whose outputs are re-encoded ("raw", "rank", "onehot").
    #[pyo3(signature = (enc, lex = true, groups = vec![], weights = None))]
    fn encode_outputs(&self, enc: &str, lex: bool, groups: Vec<usize>, weights: Option<Vec<f64>>) -> PyResult<PyGraph> {
        let e = core::OutputEnc::parse(enc).map_err(to_py)?;
        Ok(PyGraph { g: self.g.encode_outputs_grouped(e, lex, &groups, weights.as_deref()).map_err(to_py)? })
    }

    /// (kind, lo, hi) of a node
    fn node_type(&self, a: usize) -> PyResult<(String, f64, f64)> {
        if a >= self.g.nodes.len() {
            return Err(CompileError::new_err(format!("node {a} does not exist")));
        }
        let t = self.g.ty(a);
        Ok((t.kind.name().to_string(), t.lo, t.hi))
    }

    fn const_value(&self, a: usize) -> Option<f64> {
        self.g.nodes.get(a).and_then(|_| self.g.const_of(a))
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.g.nodes.len()
    }

    #[getter]
    fn num_inputs(&self) -> usize {
        self.g.inputs.len()
    }

    #[getter]
    fn num_outputs(&self) -> usize {
        self.g.outputs.len()
    }

    /// [(name, kind, lo, hi)]
    fn input_specs(&self) -> Vec<(String, String, f64, f64)> {
        self.g.inputs.iter().map(|s| (s.name.clone(), s.ty.kind.name().to_string(), s.ty.lo, s.ty.hi)).collect()
    }

    /// Output kinds and sound bounds, refined by finite-domain analysis.
    #[pyo3(signature = (exhaustive_points=65536))]
    fn output_specs(&self, exhaustive_points: usize) -> Vec<(String, f64, f64)> {
        let an = core::analysis::analyze(&self.g, exhaustive_points, 20_000_000);
        self.g.outputs.iter().map(|&o| {
            (self.g.ty(o).kind.name().to_string(), an.lo[o], an.hi[o])
        }).collect()
    }

    /// Reference interpreter on a flat row-major batch. Returns (outputs,
    /// near_margin flags): a flag is set when some real-valued discontinuity
    /// is evaluated within `margin` of its threshold.
    #[pyo3(signature = (xs, margin=None))]
    fn eval_batch(&self, xs: Vec<f64>, margin: Option<f64>) -> PyResult<(Vec<f64>, Vec<bool>)> {
        let n = self.g.inputs.len();
        let rows = batch_rows(xs.len(), n)?;
        let mut out = Vec::with_capacity(rows * self.g.outputs.len());
        let mut near = Vec::with_capacity(rows);
        for r in 0..rows {
            let x = if self.g.inputs.is_empty() { &xs[0..0] } else { &xs[r * n..(r + 1) * n] };
            let (v, f) = self.g.eval_all(x, margin).map_err(to_py)?;
            out.extend(self.g.outputs.iter().map(|&o| v[o]));
            near.push(f);
        }
        Ok((out, near))
    }
}

#[pyclass(name = "Network", module = "neurify._core")]
#[derive(Clone)]
struct PyNetwork {
    net: core::Network,
}

fn dense_dict<'py>(py: Python<'py>, d: &Dense) -> PyResult<Bound<'py, PyDict>> {
    let x = PyDict::new(py);
    x.set_item("weight", d.w.clone())?;
    x.set_item("shape", (d.rows, d.cols))?;
    x.set_item("bias", d.b.clone())?;
    Ok(x)
}

#[pymethods]
impl PyNetwork {
    /// Forward pass on a flat row-major batch of raw inputs.
    fn forward(&self, xs: Vec<f64>) -> PyResult<Vec<f64>> {
        batch_rows(xs.len(), self.net.n_in)?;
        Ok(self.net.forward_batch(&xs))
    }

    #[getter]
    fn params(&self) -> usize {
        self.net.params()
    }
    #[getter]
    fn depth(&self) -> usize {
        self.net.depth()
    }
    #[getter]
    fn widths(&self) -> Vec<usize> {
        self.net.widths()
    }
    #[getter]
    fn d_model(&self) -> Option<usize> {
        self.net.d_model()
    }
    #[getter]
    fn arch(&self) -> &'static str {
        self.net.arch.name()
    }
    #[getter]
    fn heads(&self) -> Vec<usize> {
        self.net.heads()
    }
    #[getter]
    fn n_tokens(&self) -> usize {
        self.net.n_tokens()
    }
    #[getter]
    fn n_in(&self) -> usize {
        self.net.n_in
    }
    #[getter]
    fn n_out(&self) -> usize {
        self.net.n_out
    }

    fn copy(&self) -> PyNetwork {
        self.clone()
    }

    fn normalize_inputs(&mut self) {
        core::train::normalize_inputs(&mut self.net);
    }

    /// Scale units so their activations on the samples `xs` are O(1).
    fn normalize_activations(&mut self, xs: Vec<f64>) -> PyResult<bool> {
        batch_rows(xs.len(), self.net.n_in)?;
        Ok(core::train::normalize_activations(&mut self.net, &xs))
    }

    #[pyo3(signature = (sweeps=3))]
    fn balance(&mut self, sweeps: usize) {
        core::train::balance(&mut self.net, sweeps);
    }

    #[pyo3(signature = (extra, extra_dims=0, scale=1.0, seed=0))]
    fn widen(&mut self, extra: usize, extra_dims: usize, scale: f64, seed: u64) {
        core::train::widen(&mut self.net, extra, extra_dims, scale, seed);
    }

    #[pyo3(signature = (n, width=8, scale=1.0, seed=0))]
    fn deepen(&mut self, n: usize, width: usize, scale: f64, seed: u64) {
        core::train::deepen(&mut self.net, n, width, scale, seed);
    }

    /// All weights as nested Python data (flat row-major lists + shapes).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let n = &self.net;
        d.set_item("arch", n.arch.name())?;
        d.set_item("n_in", n.n_in)?;
        d.set_item("n_out", n.n_out)?;
        d.set_item("in_offset", n.in_offset.clone())?;
        d.set_item("in_scale", n.in_scale.clone())?;
        // features: (raw input index, value) -- value NaN = the raw input itself
        let feats: Vec<(usize, f64)> = n
            .features
            .iter()
            .map(|f| match *f {
                core::network::Feature::Raw(i) => (i, f64::NAN),
                core::network::Feature::Eq(i, v) => (i, v),
            })
            .collect();
        d.set_item("features", feats)?;
        d.set_item("input_bounds", n.input_bounds.clone())?;
        match &n.body {
            Body::Mlp { hidden, out } => {
                d.set_item("body", "mlp")?;
                let hs: PyResult<Vec<_>> = hidden.iter().map(|h| dense_dict(py, h)).collect();
                d.set_item("hidden", hs?)?;
                d.set_item("out", dense_dict(py, out)?)?;
            }
            Body::Residual { kind, embed, blocks, readout } => {
                d.set_item("body", "residual")?;
                d.set_item(
                    "block",
                    match kind {
                        core::network::BlockKind::Relu => "relu",
                        core::network::BlockKind::Gated => "gated",
                    },
                )?;
                d.set_item("embed", dense_dict(py, embed)?)?;
                let mut bs = vec![];
                for b in blocks {
                    let x = PyDict::new(py);
                    x.set_item("w_in", dense_dict(py, &b.w_in)?)?;
                    if let Some(v) = &b.w_val {
                        x.set_item("w_val", dense_dict(py, v)?)?;
                    }
                    x.set_item("w_out", dense_dict(py, &b.w_out)?)?;
                    bs.push(x);
                }
                d.set_item("blocks", bs)?;
                d.set_item("readout", dense_dict(py, readout)?)?;
            }
            Body::Transformer { kind, tok_embed, layers, ln_f, readout, anchor } => {
                d.set_item("body", "transformer")?;
                d.set_item(
                    "block",
                    match kind {
                        core::network::BlockKind::Relu => "relu",
                        core::network::BlockKind::Gated => "gated",
                    },
                )?;
                d.set_item("anchor", *anchor)?;
                let te: PyResult<Vec<_>> = tok_embed.iter().map(|e| dense_dict(py, e)).collect();
                d.set_item("tok_embed", te?)?;
                let ln_dict = |l: &core::network::LayerNorm| -> PyResult<Bound<'py, PyDict>> {
                    let x = PyDict::new(py);
                    x.set_item("gain", l.gain.clone())?;
                    x.set_item("bias", l.bias.clone())?;
                    x.set_item("eps", l.eps)?;
                    Ok(x)
                };
                let mut ls = vec![];
                for l in layers {
                    let x = PyDict::new(py);
                    if let Some(n) = &l.ln1 {
                        x.set_item("ln1", ln_dict(n)?)?;
                    }
                    if let Some(n) = &l.ln2 {
                        x.set_item("ln2", ln_dict(n)?)?;
                    }
                    if let Some(a) = &l.attn {
                        let ad = PyDict::new(py);
                        ad.set_item("mode", a.mode.name())?;
                        ad.set_item("heads", a.heads)?;
                        ad.set_item("d_k", a.d_k)?;
                        ad.set_item("w_q", dense_dict(py, &a.w_q)?)?;
                        ad.set_item("w_k", dense_dict(py, &a.w_k)?)?;
                        ad.set_item("w_v", dense_dict(py, &a.w_v)?)?;
                        ad.set_item("w_o", dense_dict(py, &a.w_o)?)?;
                        x.set_item("attn", ad)?;
                    }
                    if let Some(b) = &l.mlp {
                        let bd = PyDict::new(py);
                        bd.set_item("w_in", dense_dict(py, &b.w_in)?)?;
                        if let Some(v) = &b.w_val {
                            bd.set_item("w_val", dense_dict(py, v)?)?;
                        }
                        bd.set_item("w_out", dense_dict(py, &b.w_out)?)?;
                        x.set_item("mlp", bd)?;
                    }
                    ls.push(x);
                }
                d.set_item("layers", ls)?;
                if let Some(n) = ln_f {
                    d.set_item("ln_f", ln_dict(n)?)?;
                }
                d.set_item("readout", dense_dict(py, readout)?)?;
            }
        }
        Ok(d)
    }
}

#[pyfunction]
#[pyo3(signature = (graph, arch="mlp", method="depth", margin=1e-3, strict=false, max_table=4096,
                    exhaustive_points=65536, exhaustive_cells=20_000_000, robust=true,
                    attention="softmax", attn_tol=1e-6, mlp="relu", layernorm=false, ln_tol=1e-6, slope=0.0, repr="scalar", onehot_max=64))]
#[allow(clippy::too_many_arguments)]
fn compile<'py>(
    py: Python<'py>,
    graph: &PyGraph,
    arch: &str,
    method: &str,
    margin: f64,
    strict: bool,
    max_table: usize,
    exhaustive_points: usize,
    exhaustive_cells: usize,
    robust: bool,
    attention: &str,
    attn_tol: f64,
    mlp: &str,
    layernorm: bool,
    ln_tol: f64,
    slope: f64,
    repr: &str,
    onehot_max: usize,
) -> PyResult<(PyNetwork, Bound<'py, PyDict>)> {
    let onehot = match repr {
        "scalar" => false,
        "onehot" => true,
        _ => return Err(CompileError::new_err(format!("repr must be 'scalar' or 'onehot', got '{repr}'"))),
    };
    let tx_glu = match mlp {
        "relu" => false,
        "glu" | "reglu" => true,
        _ => return Err(CompileError::new_err(format!("mlp must be 'relu' or 'glu', got '{mlp}'"))),
    };
    let o = core::Options {
        arch: core::Arch::parse(arch).map_err(to_py)?,
        method: core::Method::parse(method).map_err(to_py)?,
        margin,
        strict,
        max_table,
        exhaustive_points,
        exhaustive_cells,
        robust,
        attention: core::AttnMode::parse(attention).map_err(to_py)?,
        attn_tol,
        tx_glu,
        layernorm,
        ln_tol,
        slope,
        onehot,
        onehot_max,
    };
    let (net, r) = core::compile(&graph.g, &o).map_err(to_py)?;
    let d = PyDict::new(py);
    d.set_item("arch", r.arch.name())?;
    d.set_item("method", r.method.name())?;
    d.set_item("depth", r.depth)?;
    d.set_item("widths", r.widths.clone())?;
    d.set_item("d_model", r.d_model)?;
    d.set_item("params", r.params)?;
    d.set_item("units", r.units)?;
    d.set_item("margin_sites", r.margin_sites)?;
    d.set_item("margin", r.margin)?;
    d.set_item("lookups", r.lookups)?;
    d.set_item("exhaustive_points", r.exhaustive_points)?;
    d.set_item("exact", r.exact())?;
    d.set_item("robust", robust)?;
    d.set_item("approx", r.approx.clone())?;
    d.set_item("heads", r.heads.clone())?;
    d.set_item("n_tokens", r.n_tokens)?;
    d.set_item("attention", attention)?;
    d.set_item("slope", slope)?;
    d.set_item("repr", repr)?;
    d.set_item("warnings", r.warnings.clone())?;
    Ok((PyNetwork { net }, d))
}

// ------------------------------------------------------------ frontend glue

fn py_key(o: &Bound<'_, PyAny>) -> Result<Key, String> {
    if o.is_none() {
        return Ok(Key::None);
    }
    if o.is_instance_of::<PyBool>() || o.is_instance_of::<PyInt>() {
        return o.extract::<i64>().map(Key::Int).map_err(|e| e.to_string());
    }
    if o.is_instance_of::<PyString>() {
        let s: String = o.extract::<String>().map_err(|e| e.to_string())?;
        return Ok(Key::Str(Rc::from(s.as_str())));
    }
    if o.is_instance_of::<PyTuple>() {
        let items: Vec<Bound<'_, PyAny>> = o.extract::<Vec<Bound<'_, PyAny>>>().map_err(|e| e.to_string())?;
        return Ok(Key::Tuple(items.iter().map(py_key).collect::<Result<_, _>>()?));
    }
    Err(format!("unsupported dict key {o}"))
}

fn py_to_value(o: &Bound<'_, PyAny>) -> Result<Value, String> {
    if o.is_none() {
        return Ok(Value::None);
    }
    if o.is_instance_of::<PyBool>() {
        return Ok(Value::Bool(o.extract::<bool>().map_err(|e| e.to_string())?));
    }
    if o.is_instance_of::<PyInt>() {
        return o.extract::<i64>().map(Value::Int).map_err(|_| format!("integer {o} does not fit in 64 bits"));
    }
    if o.is_instance_of::<PyFloat>() {
        return Ok(Value::Float(o.extract::<f64>().map_err(|e| e.to_string())?));
    }
    if o.is_instance_of::<PyString>() {
        let s: String = o.extract::<String>().map_err(|e| e.to_string())?;
        return Ok(Value::str(&s));
    }
    if o.is_instance_of::<PyList>() || o.is_instance_of::<PyTuple>() {
        let items: Vec<Bound<'_, PyAny>> = o.extract::<Vec<Bound<'_, PyAny>>>().map_err(|e| e.to_string())?;
        let vals = items.iter().map(py_to_value).collect::<Result<Vec<_>, _>>()?;
        return Ok(if o.is_instance_of::<PyList>() { Value::list(vals) } else { Value::tuple(vals) });
    }
    if let Ok(d) = o.extract::<Bound<'_, PyDict>>() {
        let mut out = front::value::Dict::default();
        for (k, v) in d.iter() {
            out.set(py_key(&k)?, py_to_value(&v)?);
        }
        return Ok(Value::dict(out));
    }
    Err(format!("cannot use a value of type {} at compile time", o.get_type()))
}

fn key_to_py<'py>(py: Python<'py>, k: &Key) -> PyResult<Bound<'py, PyAny>> {
    Ok(match k {
        Key::None => py.None().into_bound(py),
        Key::Int(i) => i.into_pyobject(py)?.into_any(),
        Key::Str(s) => PyString::new(py, s).into_any(),
        Key::Tuple(t) => {
            let items = t.iter().map(|x| key_to_py(py, x)).collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(py, items)?.into_any()
        }
    })
}

fn value_to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::None => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::Float(f) => f.into_pyobject(py)?.into_any(),
        Value::Str(s) => PyString::new(py, s).into_any(),
        Value::List(l) => {
            let items = l.borrow().iter().map(|x| value_to_py(py, x)).collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, items)?.into_any()
        }
        Value::Tuple(t) => {
            let items = t.iter().map(|x| value_to_py(py, x)).collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(py, items)?.into_any()
        }
        Value::Dict(d) => {
            let out = PyDict::new(py);
            for (k, x) in d.borrow().items.iter() {
                out.set_item(key_to_py(py, k)?, value_to_py(py, x)?)?;
            }
            out.into_any()
        }
        o => return Err(TraceError::new_err(format!("cannot pass a {} to a Python callable", o.type_name()))),
    })
}

fn py_spec(o: &Bound<'_, PyAny>) -> PyResult<InSpec> {
    let t: Bound<'_, PyTuple> = o.extract()?;
    let tag: String = t.get_item(0)?.extract()?;
    Ok(match tag.as_str() {
        "real" => InSpec::Real(t.get_item(1)?.extract()?, t.get_item(2)?.extract()?),
        "int" => InSpec::Int(t.get_item(1)?.extract()?, t.get_item(2)?.extract()?),
        "bool" => InSpec::Bool,
        "list" | "tuple" | "seq" => {
            let items: Vec<Bound<'_, PyAny>> = t.get_item(1)?.extract()?;
            let specs = items.iter().map(py_spec).collect::<PyResult<Vec<_>>>()?;
            match tag.as_str() {
                "list" => InSpec::List(specs),
                "tuple" => InSpec::Tuple(specs),
                _ => InSpec::Seq(specs),
            }
        }
        "dict" => {
            let items: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)> = t.get_item(1)?.extract()?;
            let mut v = vec![];
            for (k, s) in items {
                v.push((py_key(&k).map_err(TraceError::new_err)?, py_spec(&s)?));
            }
            InSpec::Dict(v)
        }
        "static" => InSpec::Static(py_to_value(&t.get_item(1)?).map_err(TraceError::new_err)?),
        _ => return Err(TraceError::new_err(format!("bad input spec tag '{tag}'"))),
    })
}

fn out_to_py<'py>(py: Python<'py>, o: &OutSpec) -> PyResult<Bound<'py, PyAny>> {
    let tup = |items: Vec<Bound<'py, PyAny>>| PyTuple::new(py, items).map(|t| t.into_any());
    Ok(match o {
        OutSpec::Leaf(k) => tup(vec![PyString::new(py, "leaf").into_any(), PyString::new(py, k).into_any()])?,
        OutSpec::None => tup(vec![PyString::new(py, "none").into_any()])?,
        OutSpec::Tuple(v) | OutSpec::List(v) => {
            let tag = if matches!(o, OutSpec::Tuple(_)) { "tuple" } else { "list" };
            let items = v.iter().map(|x| out_to_py(py, x)).collect::<PyResult<Vec<_>>>()?;
            tup(vec![PyString::new(py, tag).into_any(), PyList::new(py, items)?.into_any()])?
        }
        OutSpec::Dict(v) => {
            let mut items = vec![];
            for (k, x) in v {
                items.push(PyTuple::new(py, [key_to_py(py, k)?, out_to_py(py, x)?])?.into_any());
            }
            tup(vec![PyString::new(py, "dict").into_any(), PyList::new(py, items)?.into_any()])?
        }
    })
}

/// Host implemented by a Python object with resolve / getattr / call methods.
struct PyHost<'py> {
    obj: Bound<'py, PyAny>,
}

impl PyHost<'_> {
    fn decode(&self, r: Bound<'_, PyAny>) -> Result<Resolved, String> {
        let t: Bound<'_, PyTuple> = r.extract::<Bound<'_, PyTuple>>().map_err(|e| e.to_string())?;
        let tag: String = t.get_item(0).and_then(|x| x.extract::<String>()).map_err(|e| e.to_string())?;
        let item = |i: usize| t.get_item(i).map_err(|e| e.to_string());
        Ok(match tag.as_str() {
            "value" => Resolved::Value(py_to_value(&item(1)?)?),
            "builtin" => Resolved::Builtin(item(1)?.extract::<String>().map_err(|e| e.to_string())?),
            "object" => Resolved::Object(item(1)?.extract::<u64>().map_err(|e| e.to_string())?),
            "function" => Resolved::Function {
                source: item(1)?.extract::<String>().map_err(|e| e.to_string())?,
                name: item(2)?.extract::<String>().map_err(|e| e.to_string())?,
                file: item(3)?.extract::<String>().map_err(|e| e.to_string())?,
                first_line: item(4)?.extract::<usize>().map_err(|e| e.to_string())?,
                ctx: item(5)?.extract::<u64>().map_err(|e| e.to_string())?,
                handle: item(6)?.extract::<Option<u64>>().map_err(|e| e.to_string())?,
            },
            _ => Resolved::Missing,
        })
    }
}

impl front::Host for PyHost<'_> {
    fn resolve(&mut self, ctx: u64, name: &str) -> Result<Resolved, String> {
        let r = self.obj.call_method1("resolve", (ctx, name)).map_err(|e| e.to_string())?;
        self.decode(r)
    }
    fn getattr(&mut self, obj: u64, attr: &str) -> Result<Resolved, String> {
        let r = self.obj.call_method1("getattr", (obj, attr)).map_err(|e| e.to_string())?;
        self.decode(r)
    }
    fn call(&mut self, obj: u64, args: &[Value]) -> Result<Value, String> {
        let py = self.obj.py();
        let a = args.iter().map(|v| value_to_py(py, v)).collect::<PyResult<Vec<_>>>().map_err(|e| e.to_string())?;
        let a = PyList::new(py, a).map_err(|e| e.to_string())?;
        let r = self.obj.call_method1("call", (obj, a)).map_err(|e| e.to_string())?;
        py_to_value(&r)
    }
}

/// Trace a Python function into a Graph. Returns (graph, output_spec, warnings).
#[pyfunction]
#[pyo3(signature = (source, name, file, first_line, ctx, params, host, max_loop_iters=1000,
                    max_inline_depth=64, max_table=4096, exhaustive_points=65536))]
#[allow(clippy::too_many_arguments)]
fn trace<'py>(
    py: Python<'py>,
    source: &str,
    name: &str,
    file: &str,
    first_line: usize,
    ctx: u64,
    params: Vec<(String, Bound<'py, PyAny>)>,
    host: Bound<'py, PyAny>,
    max_loop_iters: usize,
    max_inline_depth: usize,
    max_table: usize,
    exhaustive_points: usize,
) -> PyResult<(PyGraph, Bound<'py, PyAny>, Vec<String>)> {
    let specs = params
        .iter()
        .map(|(n, s)| Ok((n.clone(), py_spec(s)?)))
        .collect::<PyResult<Vec<(String, InSpec)>>>()?;
    let mut h = PyHost { obj: host };
    let opts = front::TraceOptions { max_loop_iters, max_inline_depth, max_table, exhaustive_points };
    let t = front::trace(source, name, file, first_line, ctx, Some(ctx), &specs, &mut h, opts)
        .map_err(|e| TraceError::new_err(e.to_string()))?;
    Ok((PyGraph { g: t.graph }, out_to_py(py, &t.out)?, t.warnings))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyNetwork>()?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(trace, m)?)?;
    m.add("CompileError", m.py().get_type::<CompileError>())?;
    m.add("TraceError", m.py().get_type::<TraceError>())?;
    Ok(())
}
