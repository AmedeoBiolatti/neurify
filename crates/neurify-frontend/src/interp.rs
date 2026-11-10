//! Symbolic interpreter: Python AST -> neurify-core IR.
//!
//! The interpreter *partially evaluates* the program: everything computable
//! from concrete values is computed at compile time (loops over constant
//! ranges, helper calls on constants, dict/list bookkeeping). Only values
//! depending on inputs become IR nodes.
//!
//! Control flow on symbolic conditions:
//! * `if c:` runs both branches and merges every binding (and the contents
//!   of every reachable list/dict, in place) with `select(c, then, else)`.
//! * `return` / `break` / `continue` set path flags; once a flag is
//!   symbolic, the rest of the block runs only on the paths where it is not
//!   set (again by branching and merging).
//! * `for` over a concrete iterable is unrolled; `range(n)` with a bounded
//!   symbolic `n` is unrolled to its upper bound with guards; `while` is
//!   unrolled until its condition is provably false (interval or exhaustive
//!   analysis), else compilation fails: loops are exact or rejected.
//! * `raise` / failing `assert` exclude a path from the domain: the merged
//!   result takes the other side.

use crate::value::*;
use neurify_core::{analysis, BinOp, CmpOp, Graph, Kind, NodeId, UnOp};
use rustpython_parser::ast::{self, Ranged};
use rustpython_parser::Parse;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct TraceError {
    pub msg: String,
    pub loc: Option<(String, usize)>,
    /// inline-depth limit hit (triggers tabulation fallback)
    pub recursion: bool,
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.loc {
            Some((file, line)) => write!(f, "{file}:{line}: {}", self.msg),
            None => write!(f, "{}", self.msg),
        }
    }
}

impl std::error::Error for TraceError {}

impl From<neurify_core::CompileError> for TraceError {
    fn from(e: neurify_core::CompileError) -> Self {
        TraceError { msg: e.msg, loc: None, recursion: false }
    }
}

pub type R<T> = Result<T, TraceError>;

fn terr<T>(msg: impl Into<String>) -> R<T> {
    Err(TraceError { msg: msg.into(), loc: None, recursion: false })
}

#[derive(Clone, Debug)]
pub struct TraceOptions {
    /// max iterations of a `while` loop / symbolic `range` before giving up
    pub max_loop_iters: usize,
    /// max nesting of inlined calls (recursion guard)
    pub max_inline_depth: usize,
    /// max entries of a tabulated opaque call
    pub max_table: usize,
    /// enumeration budget for proving loop termination
    pub exhaustive_points: usize,
}

impl Default for TraceOptions {
    fn default() -> Self {
        TraceOptions { max_loop_iters: 1000, max_inline_depth: 64, max_table: 4096, exhaustive_points: 1 << 16 }
    }
}

#[derive(Clone)]
struct State {
    env: HashMap<Rc<str>, Value>,
    ret: Value,
    brk: Value,
    cont: Value,
    stop: Value,
    rv: Value,
}

impl State {
    fn new(env: HashMap<Rc<str>, Value>) -> State {
        State {
            env,
            ret: Value::Bool(false),
            brk: Value::Bool(false),
            cont: Value::Bool(false),
            stop: Value::Bool(false),
            rv: Value::Undef,
        }
    }
    fn clear_flags(&mut self) {
        self.ret = Value::Bool(false);
        self.brk = Value::Bool(false);
        self.cont = Value::Bool(false);
        self.stop = Value::Bool(false);
    }
}

fn is_true(v: &Value) -> bool {
    matches!(v, Value::Bool(true))
}

enum Cont {
    L(Rc<RefCell<Vec<Value>>>),
    D(Rc<RefCell<Dict>>),
}

enum Saved {
    L(Vec<Value>),
    D(Dict),
}

const BUILTINS: &[&str] = &[
    "abs", "min", "max", "sum", "len", "range", "enumerate", "zip", "list", "tuple", "dict", "int", "float",
    "bool", "round", "divmod", "pow", "sorted", "reversed", "any", "all", "print", "isinstance",
];

type Runner<'a, 'h> = &'a dyn Fn(&mut Interp<'h>, State) -> R<State>;

pub struct Interp<'h> {
    pub g: Graph,
    host: &'h mut dyn Host,
    pub opts: TraceOptions,
    roots: Vec<Vec<Value>>,
    srcs: Vec<Rc<Src>>,
    ctxs: Vec<Ctx>,
    cache: HashMap<(u64, String), Value>,
    depth: usize,
    active: HashMap<usize, usize>,
    n_seq: usize,
    pub warnings: Vec<String>,
}

impl<'h> Interp<'h> {
    pub fn new(host: &'h mut dyn Host, opts: TraceOptions) -> Interp<'h> {
        Interp {
            g: Graph::new(),
            host,
            opts,
            roots: vec![],
            srcs: vec![],
            ctxs: vec![],
            cache: HashMap::new(),
            depth: 0,
            active: HashMap::new(),
            n_seq: 0,
            warnings: vec![],
        }
    }

    // ================================================================ scalars

    fn from_node(&self, id: NodeId) -> Value {
        match self.g.const_of(id) {
            Some(c) => match self.g.ty(id).kind {
                Kind::Bool => Value::Bool(c != 0.0),
                Kind::Int => Value::Int(c as i64),
                Kind::Real => Value::Float(c),
            },
            None => Value::Sym(id),
        }
    }

    fn lift(&mut self, v: &Value) -> R<NodeId> {
        Ok(match v {
            Value::Bool(b) => self.g.constant(if *b { 1.0 } else { 0.0 }, Kind::Bool)?,
            Value::Int(i) => {
                if i.unsigned_abs() > (1u64 << 53) {
                    return terr(format!("integer constant {i} is too large to be represented exactly (> 2^53)"));
                }
                self.g.constant(*i as f64, Kind::Int)?
            }
            Value::Float(f) => self.g.constant(*f, Kind::Real)?,
            Value::Sym(id) => *id,
            Value::Undef => return terr("value is undefined on every path (each path raised or never assigned it)"),
            o => return terr(format!("expected a number, got {}", o.type_name())),
        })
    }

    fn kind_of(&self, v: &Value) -> Option<Kind> {
        match v {
            Value::Bool(_) => Some(Kind::Bool),
            Value::Int(_) => Some(Kind::Int),
            Value::Float(_) => Some(Kind::Real),
            Value::Sym(id) => Some(self.g.ty(*id).kind),
            _ => None,
        }
    }

    fn bounds_of(&self, v: &Value) -> Option<(f64, f64)> {
        match v {
            Value::Bool(b) => Some((*b as i64 as f64, *b as i64 as f64)),
            Value::Int(i) => Some((*i as f64, *i as f64)),
            Value::Float(f) => Some((*f, *f)),
            Value::Sym(id) => {
                let t = self.g.ty(*id);
                Some((t.lo, t.hi))
            }
            _ => None,
        }
    }

    fn b_not(&mut self, v: &Value) -> R<Value> {
        match v {
            Value::Bool(b) => Ok(Value::Bool(!b)),
            Value::Sym(id) => {
                let n = self.g.un(UnOp::Not, *id)?;
                Ok(self.from_node(n))
            }
            o => terr(format!("expected a bool, got {}", o.type_name())),
        }
    }

    fn b_and(&mut self, a: &Value, b: &Value) -> R<Value> {
        match (a, b) {
            (Value::Bool(false), _) | (_, Value::Bool(false)) => Ok(Value::Bool(false)),
            (Value::Bool(true), x) | (x, Value::Bool(true)) => Ok(x.clone()),
            _ => {
                let (x, y) = (self.lift(a)?, self.lift(b)?);
                let n = self.g.bin(BinOp::And, x, y)?;
                Ok(self.from_node(n))
            }
        }
    }

    fn b_or(&mut self, a: &Value, b: &Value) -> R<Value> {
        match (a, b) {
            (Value::Bool(true), _) | (_, Value::Bool(true)) => Ok(Value::Bool(true)),
            (Value::Bool(false), x) | (x, Value::Bool(false)) => Ok(x.clone()),
            _ => {
                let (x, y) = (self.lift(a)?, self.lift(b)?);
                let n = self.g.bin(BinOp::Or, x, y)?;
                Ok(self.from_node(n))
            }
        }
    }

    /// Python truthiness, as a Bool or a symbolic bool.
    fn truth(&mut self, v: &Value) -> R<Value> {
        Ok(match v {
            Value::Bool(b) => Value::Bool(*b),
            Value::Int(i) => Value::Bool(*i != 0),
            Value::Float(f) => Value::Bool(*f != 0.0),
            Value::None => Value::Bool(false),
            Value::Str(s) => Value::Bool(!s.is_empty()),
            Value::List(l) => Value::Bool(!l.borrow().is_empty()),
            Value::Tuple(t) => Value::Bool(!t.is_empty()),
            Value::Dict(d) => Value::Bool(!d.borrow().items.is_empty()),
            Value::Range(a, b, s) => Value::Bool(range_len(*a, *b, *s) > 0),
            Value::Sym(id) => {
                if self.g.ty(*id).kind == Kind::Bool {
                    v.clone()
                } else {
                    let z = self.g.constant(0.0, Kind::Int)?;
                    let n = self.g.cmp(CmpOp::Ne, *id, z)?;
                    self.from_node(n)
                }
            }
            Value::Undef => return terr("use of an undefined value"),
            Value::SymRange(..) => return terr("truth value of a symbolic range"),
            _ => Value::Bool(true),
        })
    }

    // ================================================================= merge

    fn merge(&mut self, c: NodeId, a: &Value, b: &Value) -> R<Value> {
        if a.same_object(b) {
            return Ok(a.clone());
        }
        match (a, b) {
            (Value::Undef, _) => return Ok(b.clone()),
            (_, Value::Undef) => return Ok(a.clone()),
            _ => {}
        }
        if a.is_scalar() && b.is_scalar() {
            let same = match (a, b) {
                (Value::Bool(x), Value::Bool(y)) => x == y,
                (Value::Int(x), Value::Int(y)) => x == y,
                (Value::Float(x), Value::Float(y)) => x == y,
                _ => false,
            };
            if same {
                return Ok(a.clone());
            }
            let (x, y) = (self.lift(a)?, self.lift(b)?);
            let n = self.g.select(c, x, y)?;
            return Ok(self.from_node(n));
        }
        let mismatch = |a: &Value, b: &Value| -> R<Value> {
            terr(format!(
                "a value has a different type or structure on the two sides of a data-dependent branch \
                 ({} vs {}); both paths must produce the same structure",
                a.type_name(),
                b.type_name()
            ))
        };
        match (a, b) {
            (Value::List(x), Value::List(y)) => {
                let (x, y) = (x.borrow().clone(), y.borrow().clone());
                if x.len() != y.len() {
                    return terr(format!(
                        "list length depends on a data-dependent condition ({} vs {} elements)",
                        x.len(),
                        y.len()
                    ));
                }
                let v: R<Vec<Value>> = x.iter().zip(&y).map(|(p, q)| self.merge(c, p, q)).collect();
                Ok(Value::list(v?))
            }
            (Value::Tuple(x), Value::Tuple(y)) => {
                if x.len() != y.len() {
                    return mismatch(a, b);
                }
                let v: R<Vec<Value>> = x.iter().zip(y.iter()).map(|(p, q)| self.merge(c, p, q)).collect();
                Ok(Value::tuple(v?))
            }
            (Value::Dict(x), Value::Dict(y)) => {
                let (x, y) = (x.borrow().clone(), y.borrow().clone());
                Ok(Value::dict(self.merge_dicts(c, &x, &y)?))
            }
            (Value::Str(x), Value::Str(y)) if x == y => Ok(a.clone()),
            (Value::Range(..), Value::Range(..)) if format!("{a:?}") == format!("{b:?}") => Ok(a.clone()),
            _ => mismatch(a, b),
        }
    }

    fn merge_dicts(&mut self, c: NodeId, x: &Dict, y: &Dict) -> R<Dict> {
        let kx: HashSet<&Key> = x.items.iter().map(|p| &p.0).collect();
        let ky: HashSet<&Key> = y.items.iter().map(|p| &p.0).collect();
        if kx != ky {
            let diff: Vec<String> = kx.symmetric_difference(&ky).map(|k| k.to_string()).collect();
            return terr(format!(
                "dict keys depend on a data-dependent condition (keys {} exist on one path only); \
                 keys must be known at compile time",
                diff.join(", ")
            ));
        }
        let mut out = Dict::default();
        for (k, v) in &x.items {
            let w = y.get(k).unwrap().clone();
            out.items.push((k.clone(), self.merge(c, v, &w)?));
        }
        Ok(out)
    }

    fn merge_state(&mut self, c: NodeId, a: State, b: State) -> R<State> {
        let mut env = HashMap::new();
        let keys: HashSet<Rc<str>> = a.env.keys().chain(b.env.keys()).cloned().collect();
        for k in keys {
            let va = a.env.get(&k).cloned().unwrap_or(Value::Undef);
            let vb = b.env.get(&k).cloned().unwrap_or(Value::Undef);
            env.insert(k, self.merge(c, &va, &vb)?);
        }
        let mut s = State::new(env);
        s.ret = self.merge(c, &a.ret, &b.ret)?;
        s.brk = self.merge(c, &a.brk, &b.brk)?;
        s.cont = self.merge(c, &a.cont, &b.cont)?;
        s.stop = self.merge(c, &a.stop, &b.stop)?;
        s.rv = self.merge(c, &a.rv, &b.rv)?;
        Ok(s)
    }

    fn collect(roots: &[Value]) -> Vec<Cont> {
        let mut seen = HashSet::new();
        let mut out = vec![];
        let mut stack: Vec<Value> = roots.to_vec();
        while let Some(v) = stack.pop() {
            match v {
                Value::List(l) => {
                    if seen.insert(Rc::as_ptr(&l) as usize) {
                        stack.extend(l.borrow().iter().cloned());
                        out.push(Cont::L(l));
                    }
                }
                Value::Dict(d) => {
                    if seen.insert(Rc::as_ptr(&d) as usize) {
                        stack.extend(d.borrow().items.iter().map(|p| p.1.clone()));
                        out.push(Cont::D(d));
                    }
                }
                Value::Tuple(t) => stack.extend(t.iter().cloned()),
                Value::Method(r, _) => stack.push(*r),
                _ => {}
            }
        }
        out
    }

    fn save(c: &Cont) -> Saved {
        match c {
            Cont::L(l) => Saved::L(l.borrow().clone()),
            Cont::D(d) => Saved::D(d.borrow().clone()),
        }
    }

    fn restore(c: &Cont, s: Saved) {
        match (c, s) {
            (Cont::L(l), Saved::L(v)) => *l.borrow_mut() = v,
            (Cont::D(d), Saved::D(v)) => *d.borrow_mut() = v,
            _ => unreachable!(),
        }
    }

    /// Run both sides of a data-dependent branch and merge, including the
    /// contents of every container reachable from the current frames.
    fn branch(&mut self, c: NodeId, st: State, run_t: Runner<'_, 'h>, run_f: Runner<'_, 'h>) -> R<State> {
        let mut roots: Vec<Value> = st.env.values().cloned().collect();
        roots.push(st.rv.clone());
        for r in &self.roots {
            roots.extend(r.iter().cloned());
        }
        let conts = Self::collect(&roots);
        let snap: Vec<Saved> = conts.iter().map(Self::save).collect();
        let st_t = run_t(self, st.clone())?;
        let after_t: Vec<Saved> = conts.iter().map(Self::save).collect();
        for (k, s) in conts.iter().zip(snap) {
            Self::restore(k, s);
        }
        let st_f = run_f(self, st)?;
        let after_f: Vec<Saved> = conts.iter().map(Self::save).collect();
        for ((k, t), f) in conts.iter().zip(after_t).zip(after_f) {
            let merged = match (t, f) {
                (Saved::L(x), Saved::L(y)) => {
                    if x.len() != y.len() {
                        return terr(format!(
                            "list length depends on a data-dependent condition ({} vs {} elements)",
                            x.len(),
                            y.len()
                        ));
                    }
                    let v: R<Vec<Value>> = x.iter().zip(&y).map(|(p, q)| self.merge(c, p, q)).collect();
                    Saved::L(v?)
                }
                (Saved::D(x), Saved::D(y)) => Saved::D(self.merge_dicts(c, &x, &y)?),
                _ => unreachable!(),
            };
            Self::restore(k, merged);
        }
        self.merge_state(c, st_t, st_f)
    }

    fn halt(&mut self, st: &State) -> R<Value> {
        let a = self.b_or(&st.ret, &st.brk)?;
        let b = self.b_or(&st.cont, &st.stop)?;
        self.b_or(&a, &b)
    }

    /// Run `run` only on paths that are not halted.
    fn guarded(&mut self, st: State, run: Runner<'_, 'h>) -> R<State> {
        match self.halt(&st)? {
            Value::Bool(true) => Ok(st),
            Value::Bool(false) => run(self, st),
            Value::Sym(h) => self.branch(
                h,
                st,
                &|_, s| Ok(s),
                &|me, mut s| {
                    s.clear_flags();
                    run(me, s)
                },
            ),
            _ => unreachable!(),
        }
    }

    // ============================================================ statements

    fn exec_block(&mut self, stmts: &[ast::Stmt], mut st: State) -> R<State> {
        for (i, s) in stmts.iter().enumerate() {
            match self.halt(&st)? {
                Value::Bool(true) => return Ok(st),
                Value::Bool(false) => st = self.exec_stmt_loc(s, st)?,
                Value::Sym(h) => {
                    let rest = &stmts[i..];
                    return self.branch(
                        h,
                        st,
                        &|_, s| Ok(s),
                        &|me, mut s| {
                            s.clear_flags();
                            me.exec_block(rest, s)
                        },
                    );
                }
                _ => unreachable!(),
            }
        }
        Ok(st)
    }

    fn exec_stmt_loc(&mut self, s: &ast::Stmt, st: State) -> R<State> {
        self.exec_stmt(s, st).map_err(|mut e| {
            if e.loc.is_none() {
                if let Some(src) = self.srcs.last() {
                    e.loc = Some((src.file.clone(), src.line(s.range().start().to_usize())));
                }
            }
            e
        })
    }

    fn current_ctx(&self) -> Ctx {
        self.ctxs.last().cloned().unwrap_or(Ctx::Host(0))
    }

    fn exec_stmt(&mut self, s: &ast::Stmt, mut st: State) -> R<State> {
        use ast::Stmt::*;
        match s {
            FunctionDef(d) => {
                if !d.decorator_list.is_empty() {
                    return terr("decorators on nested functions are not supported");
                }
                let captured = Rc::new(RefCell::new(st.env.clone()));
                let f = Value::Func(Rc::new(Func {
                    name: d.name.to_string(),
                    body: Rc::new(FuncBody::Def(d.clone())),
                    src: self.srcs.last().unwrap().clone(),
                    ctx: Ctx::Closure(captured.clone(), Box::new(self.current_ctx())),
                    handle: None,
                }));
                captured.borrow_mut().insert(Rc::from(d.name.as_str()), f.clone());
                st.env.insert(Rc::from(d.name.as_str()), f);
                Ok(st)
            }
            Return(r) => {
                let v = match &r.value {
                    Some(e) => self.eval(e, &mut st)?,
                    None => Value::None,
                };
                st.rv = v;
                st.ret = Value::Bool(true);
                Ok(st)
            }
            Assign(a) => {
                let v = self.eval(&a.value, &mut st)?;
                for t in &a.targets {
                    self.assign(t, v.clone(), &mut st)?;
                }
                Ok(st)
            }
            AnnAssign(a) => {
                if let Some(e) = &a.value {
                    let v = self.eval(e, &mut st)?;
                    self.assign(&a.target, v, &mut st)?;
                }
                Ok(st)
            }
            AugAssign(a) => {
                let rhs = self.eval(&a.value, &mut st)?;
                let cur = self.eval(&a.target, &mut st)?;
                if let (ast::Operator::Add, Value::List(l)) = (&a.op, &cur) {
                    let items = self.iterate(&rhs)?;
                    l.borrow_mut().extend(items);
                    return Ok(st);
                }
                let v = self.binop(a.op, &cur, &rhs)?;
                self.assign(&a.target, v, &mut st)?;
                Ok(st)
            }
            Expr(e) => {
                self.eval(&e.value, &mut st)?;
                Ok(st)
            }
            Pass(_) => Ok(st),
            Break(_) => {
                st.brk = Value::Bool(true);
                Ok(st)
            }
            Continue(_) => {
                st.cont = Value::Bool(true);
                Ok(st)
            }
            Raise(_) => {
                st.rv = Value::Undef;
                st.ret = Value::Bool(true);
                Ok(st)
            }
            Assert(a) => {
                let v = self.eval(&a.test, &mut st)?;
                match self.truth(&v)? {
                    Value::Bool(true) => Ok(st),
                    Value::Bool(false) => {
                        st.rv = Value::Undef;
                        st.ret = Value::Bool(true);
                        Ok(st)
                    }
                    c => {
                        // assumption: paths violating the assert are excluded
                        let nc = self.b_not(&c)?;
                        let Value::Sym(id) = nc else { unreachable!() };
                        self.branch(
                            id,
                            st,
                            &|_, mut s| {
                                s.rv = Value::Undef;
                                s.ret = Value::Bool(true);
                                Ok(s)
                            },
                            &|_, s| Ok(s),
                        )
                    }
                }
            }
            If(i) => {
                let v = self.eval(&i.test, &mut st)?;
                match self.truth(&v)? {
                    Value::Bool(true) => self.exec_block(&i.body, st),
                    Value::Bool(false) => self.exec_block(&i.orelse, st),
                    Value::Sym(c) => self.branch(
                        c,
                        st,
                        &|me, s| me.exec_block(&i.body, s),
                        &|me, s| me.exec_block(&i.orelse, s),
                    ),
                    _ => unreachable!(),
                }
            }
            For(f) => self.exec_for(f, st),
            While(w) => self.exec_while(w, st),
            Delete(d) => {
                for t in &d.targets {
                    match t {
                        ast::Expr::Name(n) => {
                            st.env.remove(n.id.as_str());
                        }
                        ast::Expr::Subscript(sub) => {
                            let c = self.eval(&sub.value, &mut st)?;
                            let k = self.eval(&sub.slice, &mut st)?;
                            match c {
                                Value::Dict(d) => {
                                    let key = self.key_of(&k)?;
                                    if d.borrow_mut().remove(&key).is_none() {
                                        return terr(format!("KeyError: {key}"));
                                    }
                                }
                                Value::List(l) => {
                                    let n = l.borrow().len();
                                    let i = self.index_of(&k, n)?;
                                    l.borrow_mut().remove(i);
                                }
                                o => return terr(format!("cannot delete items of {}", o.type_name())),
                            }
                        }
                        _ => return terr("unsupported del target"),
                    }
                }
                Ok(st)
            }
            o => terr(format!("unsupported statement: {}", stmt_name(o))),
        }
    }

    fn finish_loop(&mut self, orelse: &[ast::Stmt], mut st: State) -> R<State> {
        st.stop = Value::Bool(false);
        if !orelse.is_empty() {
            st = match st.brk.clone() {
                Value::Bool(true) => st,
                Value::Bool(false) => self.exec_block(orelse, st)?,
                Value::Sym(b) => self.branch(b, st, &|_, s| Ok(s), &|me, mut s| {
                    s.brk = Value::Bool(false);
                    me.exec_block(orelse, s)
                })?,
                _ => unreachable!(),
            };
        }
        st.brk = Value::Bool(false);
        st.stop = Value::Bool(false);
        Ok(st)
    }

    fn exec_for(&mut self, f: &ast::StmtFor, mut st: State) -> R<State> {
        let it = self.eval(&f.iter, &mut st)?;
        let (items, guard) = match &it {
            Value::SymRange(start, stop, step) => {
                let step = *step;
                let (sl, sh) = self.int_bounds(start, "range() start")?;
                let (el, eh) = self.int_bounds(stop, "range() stop")?;
                let k = if step > 0 {
                    ((eh - sl) as f64 / step as f64).ceil().max(0.0)
                } else {
                    ((sh - el) as f64 / (-step) as f64).ceil().max(0.0)
                } as usize;
                if k > self.opts.max_loop_iters {
                    return terr(format!(
                        "range() with a symbolic bound may run {k} iterations (> max_loop_iters={})",
                        self.opts.max_loop_iters
                    ));
                }
                let mut items = vec![];
                for j in 0..k {
                    items.push(self.binop(ast::Operator::Add, start, &Value::Int(j as i64 * step))?);
                }
                (items, Some(((**stop).clone(), step)))
            }
            _ => (self.iterate(&it)?, None),
        };
        for item in items {
            if is_true(&st.ret) || is_true(&st.brk) || is_true(&st.stop) {
                break;
            }
            if let Some((stop, step)) = &guard {
                let op = if *step > 0 { ast::CmpOp::Lt } else { ast::CmpOp::Gt };
                let c = self.compare(op, &item, stop)?;
                match c {
                    Value::Bool(false) => break,
                    Value::Bool(true) => {}
                    _ => {
                        let nc = self.b_not(&c)?;
                        st = self.guarded(st, &|_, mut s| {
                            s.stop = nc.clone();
                            Ok(s)
                        })?;
                    }
                }
            }
            st = self.guarded(st, &|me, mut s| {
                me.assign(&f.target, item.clone(), &mut s)?;
                Ok(s)
            })?;
            st = self.exec_block(&f.body, st)?;
            st.cont = Value::Bool(false);
        }
        self.finish_loop(&f.orelse, st)
    }

    fn exec_while(&mut self, w: &ast::StmtWhile, mut st: State) -> R<State> {
        let max = self.opts.max_loop_iters;
        for k in 0..=max {
            if is_true(&st.ret) || is_true(&st.brk) || is_true(&st.stop) {
                break;
            }
            let v = self.eval(&w.test, &mut st)?;
            let c = self.truth(&v)?;
            if matches!(c, Value::Bool(false)) {
                break;
            }
            if k == max {
                let h = self.halt(&st)?;
                let nh = self.b_not(&h)?;
                let running = self.b_and(&nh, &c)?;
                let proven = match running {
                    Value::Bool(false) => true,
                    Value::Bool(true) => false,
                    Value::Sym(id) => {
                        let an = analysis::analyze(&self.g, self.opts.exhaustive_points, 20_000_000);
                        an.hi[id] == 0.0
                    }
                    _ => false,
                };
                if proven {
                    break;
                }
                return terr(format!(
                    "while loop may run more than max_loop_iters={max} iterations for some inputs \
                     (could not prove termination); raise max_loop_iters or bound the inputs"
                ));
            }
            if let Value::Sym(_) = c {
                let nc = self.b_not(&c)?;
                st = self.guarded(st, &|_, mut s| {
                    s.stop = nc.clone();
                    Ok(s)
                })?;
            }
            st = self.exec_block(&w.body, st)?;
            st.cont = Value::Bool(false);
        }
        self.finish_loop(&w.orelse, st)
    }

    fn int_bounds(&self, v: &Value, what: &str) -> R<(i64, i64)> {
        match (self.kind_of(v), self.bounds_of(v)) {
            (Some(k), Some((lo, hi))) if k.integral() && lo.is_finite() && hi.is_finite() => Ok((lo as i64, hi as i64)),
            _ => terr(format!("{what} must be a bounded integer")),
        }
    }

    // ============================================================ assignment

    fn assign(&mut self, t: &ast::Expr, v: Value, st: &mut State) -> R<()> {
        match t {
            ast::Expr::Name(n) => {
                st.env.insert(Rc::from(n.id.as_str()), v);
                Ok(())
            }
            ast::Expr::Tuple(ast::ExprTuple { elts, .. }) | ast::Expr::List(ast::ExprList { elts, .. }) => {
                let vals = self.iterate(&v)?;
                let star = elts.iter().position(|e| matches!(e, ast::Expr::Starred(_)));
                match star {
                    None => {
                        if vals.len() != elts.len() {
                            return terr(format!("cannot unpack {} values into {} targets", vals.len(), elts.len()));
                        }
                        for (e, x) in elts.iter().zip(vals) {
                            self.assign(e, x, st)?;
                        }
                    }
                    Some(p) => {
                        let after = elts.len() - p - 1;
                        if vals.len() < elts.len() - 1 {
                            return terr("not enough values to unpack");
                        }
                        for (e, x) in elts[..p].iter().zip(&vals) {
                            self.assign(e, x.clone(), st)?;
                        }
                        let mid = vals[p..vals.len() - after].to_vec();
                        let ast::Expr::Starred(s) = &elts[p] else { unreachable!() };
                        self.assign(&s.value, Value::list(mid), st)?;
                        for (e, x) in elts[p + 1..].iter().zip(&vals[vals.len() - after..]) {
                            self.assign(e, x.clone(), st)?;
                        }
                    }
                }
                Ok(())
            }
            ast::Expr::Subscript(_) => {
                // a[i][j]... = v: walk the whole path so writes reach the real
                // containers even when an intermediate index is symbolic
                let mut keys = vec![];
                let mut cur = t;
                while let ast::Expr::Subscript(sub) = cur {
                    if matches!(&*sub.slice, ast::Expr::Slice(_)) {
                        return terr("slice assignment is not supported");
                    }
                    keys.push(&*sub.slice);
                    cur = &sub.value;
                }
                keys.reverse();
                let base = self.eval(cur, st)?;
                let mut kv = vec![];
                for k in keys {
                    kv.push(self.eval(k, st)?);
                }
                self.assign_path(&base, &kv, v)
            }
            ast::Expr::Attribute(_) => terr("assigning to attributes is not supported"),
            _ => terr("unsupported assignment target"),
        }
    }

    fn getitem_path(&mut self, c: &Value, keys: &[Value]) -> R<Value> {
        let mut cur = c.clone();
        for k in keys {
            cur = self.getitem(&cur, k)?;
        }
        Ok(cur)
    }

    fn assign_path(&mut self, c: &Value, keys: &[Value], v: Value) -> R<()> {
        if keys.len() == 1 {
            return self.setitem(c, &keys[0], v);
        }
        let (k, rest) = (&keys[0], &keys[1..]);
        if let (Value::List(l), Value::Sym(_)) = (c, k) {
            let n = l.borrow().len();
            for (i, cond) in self.index_conds(k, n)? {
                let child = l.borrow()[i].clone();
                match cond {
                    Value::Bool(true) => self.assign_path(&child, rest, v.clone())?,
                    Value::Sym(cid) => {
                        let old = self.getitem_path(&child, rest)?;
                        let nv = self.merge(cid, &v, &old)?;
                        self.assign_path(&child, rest, nv)?;
                    }
                    _ => unreachable!(),
                }
            }
            return Ok(());
        }
        let child = self.getitem(c, k)?;
        self.assign_path(&child, rest, v)
    }

    fn key_of(&self, v: &Value) -> R<Key> {
        Ok(match v {
            Value::None => Key::None,
            Value::Bool(b) => Key::Int(*b as i64),
            Value::Int(i) => Key::Int(*i),
            Value::Float(f) if f.fract() == 0.0 => Key::Int(*f as i64),
            Value::Str(s) => Key::Str(s.clone()),
            Value::Tuple(t) => Key::Tuple(t.iter().map(|x| self.key_of(x)).collect::<R<_>>()?),
            Value::Sym(_) => {
                return terr("dict keys must be known at compile time (got a value that depends on the inputs)")
            }
            o => return terr(format!("unhashable or unsupported dict key: {}", o.type_name())),
        })
    }

    fn index_of(&self, v: &Value, n: usize) -> R<usize> {
        let i = match v {
            Value::Int(i) => *i,
            Value::Bool(b) => *b as i64,
            o => return terr(format!("list indices must be integers, not {}", o.type_name())),
        };
        let j = if i < 0 { i + n as i64 } else { i };
        if j < 0 || j >= n as i64 {
            return terr(format!("IndexError: index {i} out of range for length {n}"));
        }
        Ok(j as usize)
    }

    /// conditions [idx == k] for every position k a symbolic index may take
    fn index_conds(&mut self, idx: &Value, n: usize) -> R<Vec<(usize, Value)>> {
        let mut out = vec![];
        for k in 0..n {
            let a = self.compare(ast::CmpOp::Eq, idx, &Value::Int(k as i64))?;
            let b = self.compare(ast::CmpOp::Eq, idx, &Value::Int(k as i64 - n as i64))?;
            let c = self.b_or(&a, &b)?;
            if !matches!(c, Value::Bool(false)) {
                out.push((k, c));
            }
        }
        Ok(out)
    }

    fn setitem(&mut self, c: &Value, k: &Value, v: Value) -> R<()> {
        match c {
            Value::List(l) => {
                let n = l.borrow().len();
                if let Value::Sym(_) = k {
                    for (i, cond) in self.index_conds(k, n)? {
                        let old = l.borrow()[i].clone();
                        let nv = match &cond {
                            Value::Bool(true) => v.clone(),
                            Value::Sym(cid) => self.merge(*cid, &v, &old)?,
                            _ => unreachable!(),
                        };
                        l.borrow_mut()[i] = nv;
                    }
                    return Ok(());
                }
                let i = self.index_of(k, n)?;
                l.borrow_mut()[i] = v;
                Ok(())
            }
            Value::Dict(d) => {
                let key = self.key_of(k)?;
                d.borrow_mut().set(key, v);
                Ok(())
            }
            o => terr(format!("'{}' object does not support item assignment", o.type_name())),
        }
    }

    /// sum_k c_k * v_k for one-hot conditions, recursively over structure.
    fn onehot_select(&mut self, pairs: &[(Value, Value)]) -> R<Value> {
        if pairs.is_empty() {
            return terr("IndexError: symbolic index is always out of range");
        }
        if pairs.len() == 1 {
            return Ok(pairs[0].1.clone());
        }
        if pairs.iter().all(|p| p.1.is_scalar()) {
            let mut acc = Value::Int(0);
            for (c, v) in pairs {
                let t = self.binop(ast::Operator::Mult, c, v)?;
                acc = self.binop(ast::Operator::Add, &acc, &t)?;
            }
            // keep bool-ness when all values are bools
            if pairs.iter().all(|p| self.kind_of(&p.1) == Some(Kind::Bool)) {
                let one = Value::Int(1);
                return self.compare(ast::CmpOp::Eq, &acc, &one);
            }
            return Ok(acc);
        }
        let first = &pairs[0].1;
        if pairs.iter().all(|p| p.1.same_object(first)) {
            return Ok(first.clone());
        }
        let seqs: Option<Vec<Vec<Value>>> = pairs
            .iter()
            .map(|p| match &p.1 {
                Value::List(l) => Some(l.borrow().clone()),
                Value::Tuple(t) => Some(t.to_vec()),
                _ => None,
            })
            .collect();
        if let Some(seqs) = seqs {
            let n = seqs[0].len();
            if seqs.iter().all(|s| s.len() == n) {
                let mut out = vec![];
                for j in 0..n {
                    let sub: Vec<(Value, Value)> = pairs.iter().zip(&seqs).map(|(p, s)| (p.0.clone(), s[j].clone())).collect();
                    out.push(self.onehot_select(&sub)?);
                }
                return Ok(match first {
                    Value::List(_) => Value::list(out),
                    _ => Value::tuple(out),
                });
            }
        }
        if let Value::Dict(d0) = first {
            let keys: Vec<Key> = d0.borrow().items.iter().map(|p| p.0.clone()).collect();
            let mut out = Dict::default();
            for k in keys {
                let mut sub = vec![];
                for (c, v) in pairs {
                    let Value::Dict(d) = v else { return terr("symbolic index over elements of different types") };
                    let Some(x) = d.borrow().get(&k).cloned() else {
                        return terr("symbolic index over dicts with different keys");
                    };
                    sub.push((c.clone(), x));
                }
                out.items.push((k, self.onehot_select(&sub)?));
            }
            return Ok(Value::dict(out));
        }
        terr("symbolic index over elements of different types or structure")
    }

    fn getitem(&mut self, c: &Value, k: &Value) -> R<Value> {
        let seq: Option<Vec<Value>> = match c {
            Value::List(l) => Some(l.borrow().clone()),
            Value::Tuple(t) => Some(t.to_vec()),
            Value::Range(a, b, s) => Some(range_items(*a, *b, *s)?.into_iter().map(Value::Int).collect()),
            _ => None,
        };
        if let Some(seq) = seq {
            if let Value::Sym(kid) = k {
                if !seq.is_empty() && seq.iter().all(|v| v.is_scalar()) {
                    // one IR op: becomes an attention read on transformer tokens,
                    // a masked sum elsewhere
                    let elems = seq.iter().map(|v| self.lift(v)).collect::<R<Vec<_>>>()?;
                    let n = self.g.gather(elems, *kid)?;
                    return Ok(self.from_node(n));
                }
                let conds = self.index_conds(k, seq.len())?;
                let pairs: Vec<(Value, Value)> = conds.into_iter().map(|(i, c)| (c, seq[i].clone())).collect();
                return self.onehot_select(&pairs);
            }
            let i = self.index_of(k, seq.len())?;
            return Ok(seq[i].clone());
        }
        match c {
            Value::Dict(d) => {
                let key = self.key_of(k)?;
                d.borrow().get(&key).cloned().ok_or_else(|| TraceError {
                    msg: format!("KeyError: {key}"),
                    loc: None,
                    recursion: false,
                })
            }
            Value::Str(s) => {
                let chars: Vec<char> = s.chars().collect();
                let i = self.index_of(k, chars.len())?;
                Ok(Value::str(&chars[i].to_string()))
            }
            o => terr(format!("'{}' object is not subscriptable", o.type_name())),
        }
    }

    fn slice(&mut self, c: &Value, s: &ast::ExprSlice, st: &mut State) -> R<Value> {
        let mut get = |me: &mut Self, e: &Option<Box<ast::Expr>>| -> R<Option<i64>> {
            match e {
                None => Ok(None),
                Some(e) => match me.eval(e, st)? {
                    Value::Int(i) => Ok(Some(i)),
                    Value::None => Ok(None),
                    _ => terr("slice bounds must be known at compile time"),
                },
            }
        };
        let (lo, hi, step) = (get(self, &s.lower)?, get(self, &s.upper)?, get(self, &s.step)?.unwrap_or(1));
        let items: Vec<Value> = match c {
            Value::List(l) => l.borrow().clone(),
            Value::Tuple(t) => t.to_vec(),
            Value::Str(x) => x.chars().map(|ch| Value::str(&ch.to_string())).collect(),
            Value::Range(a, b, st2) => range_items(*a, *b, *st2)?.into_iter().map(Value::Int).collect(),
            o => return terr(format!("cannot slice {}", o.type_name())),
        };
        let idx = slice_indices(items.len() as i64, lo, hi, step)?;
        let out: Vec<Value> = idx.into_iter().map(|i| items[i].clone()).collect();
        Ok(match c {
            Value::Tuple(_) => Value::tuple(out),
            Value::Str(_) => {
                let s: String = out.iter().map(|v| if let Value::Str(x) = v { x.to_string() } else { String::new() }).collect();
                Value::str(&s)
            }
            _ => Value::list(out),
        })
    }

    // ============================================================ iteration

    fn iterate(&mut self, v: &Value) -> R<Vec<Value>> {
        Ok(match v {
            Value::List(l) => l.borrow().clone(),
            Value::Tuple(t) => t.to_vec(),
            Value::Dict(d) => d.borrow().items.iter().map(|p| p.0.to_value()).collect(),
            Value::Range(a, b, s) => range_items(*a, *b, *s)?.into_iter().map(Value::Int).collect(),
            Value::Str(s) => s.chars().map(|c| Value::str(&c.to_string())).collect(),
            Value::Sym(_) => return terr("cannot iterate over a symbolic value"),
            Value::SymRange(..) => {
                return terr("a range with a symbolic bound can only be used directly in a for loop")
            }
            o => return terr(format!("'{}' object is not iterable", o.type_name())),
        })
    }

    // ============================================================ operators

    fn concrete_arith(&self, op: ast::Operator, a: &Value, b: &Value) -> R<Value> {
        use ast::Operator::*;
        let both_bool = matches!((a, b), (Value::Bool(_), Value::Bool(_)));
        let as_i = |v: &Value| match v {
            Value::Bool(x) => Some(*x as i64),
            Value::Int(i) => Some(*i),
            _ => None,
        };
        let as_f = |v: &Value| match v {
            Value::Bool(x) => *x as i64 as f64,
            Value::Int(i) => *i as f64,
            Value::Float(f) => *f,
            _ => f64::NAN,
        };
        let ovf = || terr::<Value>("integer overflow during compile-time evaluation");
        if let (Some(x), Some(y)) = (as_i(a), as_i(b)) {
            return match op {
                Add => x.checked_add(y).map(Value::Int).map_or_else(ovf, Ok),
                Sub => x.checked_sub(y).map(Value::Int).map_or_else(ovf, Ok),
                Mult => x.checked_mul(y).map(Value::Int).map_or_else(ovf, Ok),
                Div => {
                    if y == 0 {
                        terr("ZeroDivisionError: division by zero")
                    } else {
                        Ok(Value::Float(x as f64 / y as f64))
                    }
                }
                FloorDiv | Mod => {
                    if y == 0 {
                        return terr("ZeroDivisionError: integer division or modulo by zero");
                    }
                    let mut q = x / y;
                    if (x % y != 0) && ((x < 0) != (y < 0)) {
                        q -= 1;
                    }
                    Ok(Value::Int(if op == FloorDiv { q } else { x - q * y }))
                }
                Pow => {
                    if y >= 0 {
                        u32::try_from(y).ok().and_then(|e| x.checked_pow(e)).map(Value::Int).map_or_else(ovf, Ok)
                    } else {
                        Ok(Value::Float((x as f64).powf(y as f64)))
                    }
                }
                LShift => u32::try_from(y).ok().and_then(|e| x.checked_shl(e)).map(Value::Int).map_or_else(ovf, Ok),
                RShift => Ok(Value::Int(x >> y.clamp(0, 63))),
                BitAnd if both_bool => Ok(Value::Bool(x & y != 0)),
                BitOr if both_bool => Ok(Value::Bool(x | y != 0)),
                BitXor if both_bool => Ok(Value::Bool(x ^ y != 0)),
                BitAnd => Ok(Value::Int(x & y)),
                BitOr => Ok(Value::Int(x | y)),
                BitXor => Ok(Value::Int(x ^ y)),
                MatMult => terr("@ is not supported"),
            };
        }
        let (x, y) = (as_f(a), as_f(b));
        Ok(Value::Float(match op {
            Add => x + y,
            Sub => x - y,
            Mult => x * y,
            Div => {
                if y == 0.0 {
                    return terr("ZeroDivisionError: float division by zero");
                }
                x / y
            }
            FloorDiv | Mod => {
                if y == 0.0 {
                    return terr("ZeroDivisionError: float modulo");
                }
                let (q, r) = neurify_core::ir::py_divmod(x, y);
                if op == FloorDiv {
                    q
                } else {
                    r
                }
            }
            Pow => x.powf(y),
            _ => return terr("unsupported operand types for bitwise operation on floats"),
        }))
    }

    pub fn binop(&mut self, op: ast::Operator, a: &Value, b: &Value) -> R<Value> {
        use ast::Operator::*;
        match (op, a, b) {
            (Add, Value::Str(x), Value::Str(y)) => return Ok(Value::str(&format!("{x}{y}"))),
            (Add, Value::List(x), Value::List(y)) => {
                let mut v = x.borrow().clone();
                v.extend(y.borrow().iter().cloned());
                return Ok(Value::list(v));
            }
            (Add, Value::Tuple(x), Value::Tuple(y)) => {
                let mut v = x.to_vec();
                v.extend(y.iter().cloned());
                return Ok(Value::tuple(v));
            }
            (Mult, Value::List(_) | Value::Tuple(_), Value::Int(_)) => return self.repeat(a, b),
            (Mult, Value::Int(_), Value::List(_) | Value::Tuple(_)) => return self.repeat(b, a),
            _ => {}
        }
        if !(a.is_scalar() && b.is_scalar()) {
            return terr(format!("unsupported operand types for {:?}: {} and {}", op, a.type_name(), b.type_name()));
        }
        if !matches!(a, Value::Sym(_)) && !matches!(b, Value::Sym(_)) {
            return self.concrete_arith(op, a, b);
        }
        let ka = self.kind_of(a).unwrap();
        let kb = self.kind_of(b).unwrap();
        let g = |me: &mut Self, o: BinOp| -> R<Value> {
            let (x, y) = (me.lift(a)?, me.lift(b)?);
            let n = me.g.bin(o, x, y)?;
            Ok(me.from_node(n))
        };
        match op {
            Add => g(self, BinOp::Add),
            Sub => g(self, BinOp::Sub),
            Mult => g(self, BinOp::Mul),
            Div => g(self, BinOp::Div),
            FloorDiv => g(self, BinOp::FloorDiv),
            Mod => g(self, BinOp::Mod),
            Pow => {
                let e = match b {
                    Value::Int(i) => Some(*i),
                    Value::Bool(x) => Some(*x as i64),
                    Value::Float(f) if f.fract() == 0.0 => Some(*f as i64),
                    _ => None,
                };
                match e {
                    Some(n) if n >= 0 => {
                        let (mut acc, mut base, mut n) = (Value::Int(1), a.clone(), n);
                        while n > 0 {
                            if n & 1 == 1 {
                                acc = self.binop(Mult, &acc, &base)?;
                            }
                            n >>= 1;
                            if n > 0 {
                                base = self.binop(Mult, &base, &base)?;
                            }
                        }
                        if matches!(b, Value::Float(_)) {
                            return Ok(acc); // numerically identical
                        }
                        Ok(acc)
                    }
                    Some(n) => {
                        let p = self.binop(Pow, a, &Value::Int(-n))?;
                        self.binop(Div, &Value::Int(1), &p)
                    }
                    None => self.tabulate_op(op, a, b),
                }
            }
            LShift | RShift => match b {
                Value::Int(k) if *k >= 0 && *k < 53 => {
                    let p = Value::Int(1i64 << k);
                    self.binop(if op == LShift { Mult } else { FloorDiv }, a, &p)
                }
                _ => self.tabulate_op(op, a, b),
            },
            BitAnd | BitOr | BitXor if ka == Kind::Bool && kb == Kind::Bool => g(
                self,
                match op {
                    BitAnd => BinOp::And,
                    BitOr => BinOp::Or,
                    _ => BinOp::Xor,
                },
            ),
            BitAnd | BitOr | BitXor => self.tabulate_op(op, a, b),
            MatMult => terr("@ is not supported"),
        }
    }

    fn tabulate_op(&mut self, op: ast::Operator, a: &Value, b: &Value) -> R<Value> {
        self.tabulate(&[a.clone(), b.clone()], &mut |me, xs| me.concrete_arith(op, &xs[0], &xs[1]))
    }

    fn repeat(&mut self, seq: &Value, n: &Value) -> R<Value> {
        let Value::Int(n) = n else { unreachable!() };
        let items = self.iterate(seq)?;
        let mut out = vec![];
        for _ in 0..(*n).max(0) {
            out.extend(items.iter().cloned());
        }
        Ok(match seq {
            Value::Tuple(_) => Value::tuple(out),
            _ => Value::list(out),
        })
    }

    pub fn compare(&mut self, op: ast::CmpOp, a: &Value, b: &Value) -> R<Value> {
        use ast::CmpOp::*;
        match op {
            Is | IsNot => {
                let same = a.same_object(b)
                    || matches!((a, b), (Value::Bool(x), Value::Bool(y)) if x == y)
                    || matches!((a, b), (Value::Int(x), Value::Int(y)) if x == y);
                if (matches!(a, Value::Sym(_)) || matches!(b, Value::Sym(_))) && !same {
                    if !(matches!(a, Value::None) || matches!(b, Value::None)) {
                        return terr("'is' comparison on symbolic values is not supported");
                    }
                }
                return Ok(Value::Bool(same == (op == Is)));
            }
            In | NotIn => {
                let r = self.contains(b, a)?;
                return if op == In { Ok(r) } else { self.b_not(&r) };
            }
            _ => {}
        }
        if a.is_scalar() && b.is_scalar() {
            if !matches!(a, Value::Sym(_)) && !matches!(b, Value::Sym(_)) {
                let x = self.num(a);
                let y = self.num(b);
                return Ok(Value::Bool(match op {
                    Eq => x == y,
                    NotEq => x != y,
                    Lt => x < y,
                    LtE => x <= y,
                    Gt => x > y,
                    GtE => x >= y,
                    _ => unreachable!(),
                }));
            }
            let o = match op {
                Eq => CmpOp::Eq,
                NotEq => CmpOp::Ne,
                Lt => CmpOp::Lt,
                LtE => CmpOp::Le,
                Gt => CmpOp::Gt,
                GtE => CmpOp::Ge,
                _ => unreachable!(),
            };
            let (x, y) = (self.lift(a)?, self.lift(b)?);
            let n = self.g.cmp(o, x, y)?;
            return Ok(self.from_node(n));
        }
        if matches!(op, Eq | NotEq) {
            let e = self.struct_eq(a, b)?;
            return if op == Eq { Ok(e) } else { self.b_not(&e) };
        }
        match (a, b) {
            (Value::Str(x), Value::Str(y)) => Ok(Value::Bool(match op {
                Lt => x < y,
                LtE => x <= y,
                Gt => x > y,
                _ => x >= y,
            })),
            _ => terr(format!("ordering comparison between {} and {} is not supported", a.type_name(), b.type_name())),
        }
    }

    fn num(&self, v: &Value) -> f64 {
        match v {
            Value::Bool(b) => *b as i64 as f64,
            Value::Int(i) => *i as f64,
            Value::Float(f) => *f,
            _ => f64::NAN,
        }
    }

    fn struct_eq(&mut self, a: &Value, b: &Value) -> R<Value> {
        if a.is_scalar() && b.is_scalar() {
            return self.compare(ast::CmpOp::Eq, a, b);
        }
        let seq = |v: &Value| match v {
            Value::List(l) => Some((0, l.borrow().clone())),
            Value::Tuple(t) => Some((1, t.to_vec())),
            _ => None,
        };
        match (seq(a), seq(b)) {
            (Some((ta, x)), Some((tb, y))) => {
                if ta != tb || x.len() != y.len() {
                    return Ok(Value::Bool(false));
                }
                let mut acc = Value::Bool(true);
                for (p, q) in x.iter().zip(&y) {
                    let e = self.struct_eq(p, q)?;
                    acc = self.b_and(&acc, &e)?;
                }
                Ok(acc)
            }
            _ => match (a, b) {
                (Value::Dict(x), Value::Dict(y)) => {
                    let (x, y) = (x.borrow().clone(), y.borrow().clone());
                    if x.items.len() != y.items.len() {
                        return Ok(Value::Bool(false));
                    }
                    let mut acc = Value::Bool(true);
                    for (k, v) in &x.items {
                        let Some(w) = y.get(k) else { return Ok(Value::Bool(false)) };
                        let e = self.struct_eq(v, w)?;
                        acc = self.b_and(&acc, &e)?;
                    }
                    Ok(acc)
                }
                (Value::Str(x), Value::Str(y)) => Ok(Value::Bool(x == y)),
                _ => Ok(Value::Bool(a.same_object(b))),
            },
        }
    }

    fn contains(&mut self, c: &Value, x: &Value) -> R<Value> {
        match c {
            Value::Dict(d) => {
                let k = self.key_of(x)?;
                Ok(Value::Bool(d.borrow().get(&k).is_some()))
            }
            Value::Str(s) => match x {
                Value::Str(y) => Ok(Value::Bool(s.contains(&**y))),
                _ => terr("'in <string>' requires a string"),
            },
            Value::Range(a, b, 1) if matches!(x, Value::Sym(_)) => {
                let lo = self.compare(ast::CmpOp::GtE, x, &Value::Int(*a))?;
                let hi = self.compare(ast::CmpOp::Lt, x, &Value::Int(*b))?;
                let both = self.b_and(&lo, &hi)?;
                if self.kind_of(x).map_or(false, |k| k.integral()) {
                    return Ok(both);
                }
                let fl = self.binop(ast::Operator::FloorDiv, x, &Value::Int(1))?;
                let integral = self.compare(ast::CmpOp::Eq, &fl, x)?;
                self.b_and(&both, &integral)
            }
            _ => {
                let items = self.iterate(c)?;
                let mut acc = Value::Bool(false);
                for it in items {
                    let e = self.struct_eq(x, &it)?;
                    acc = self.b_or(&acc, &e)?;
                }
                Ok(acc)
            }
        }
    }

    fn unaryop(&mut self, op: ast::UnaryOp, v: &Value) -> R<Value> {
        use ast::UnaryOp::*;
        match op {
            Not => {
                let t = self.truth(v)?;
                self.b_not(&t)
            }
            UAdd => match v {
                Value::Bool(b) => Ok(Value::Int(*b as i64)),
                x if x.is_scalar() => Ok(x.clone()),
                o => terr(format!("bad operand type for unary +: {}", o.type_name())),
            },
            USub => match v {
                Value::Bool(b) => Ok(Value::Int(-(*b as i64))),
                Value::Int(i) => i.checked_neg().map(Value::Int).ok_or_else(|| TraceError {
                    msg: "integer overflow".into(),
                    loc: None,
                    recursion: false,
                }),
                Value::Float(f) => Ok(Value::Float(-f)),
                Value::Sym(id) => {
                    let n = self.g.un(UnOp::Neg, *id)?;
                    Ok(self.from_node(n))
                }
                o => terr(format!("bad operand type for unary -: {}", o.type_name())),
            },
            Invert => match v {
                Value::Float(_) => terr("bad operand type for unary ~: float"),
                x if x.is_scalar() => {
                    let n = self.unaryop(USub, x)?;
                    self.binop(ast::Operator::Sub, &n, &Value::Int(1))
                }
                o => terr(format!("bad operand type for unary ~: {}", o.type_name())),
            },
        }
    }

    // ============================================================ tabulation

    /// Tabulate `f` over the finite domain of the symbolic scalar arguments.
    fn tabulate(&mut self, args: &[Value], f: &mut dyn FnMut(&mut Self, &[Value]) -> R<Value>) -> R<Value> {
        let mut sym_pos = vec![];
        let (mut los, mut dims, mut ids) = (vec![], vec![], vec![]);
        for (i, a) in args.iter().enumerate() {
            if let Value::Sym(id) = a {
                let t = self.g.ty(*id);
                if !(t.kind.integral() && t.finite()) {
                    return terr(
                        "cannot evaluate this call/operation exactly: its symbolic arguments must be \
                         bounded integers (finite domain) so it can be tabulated",
                    );
                }
                sym_pos.push(i);
                los.push(t.lo);
                dims.push((t.hi - t.lo) as usize + 1);
                ids.push(*id);
            } else if !args[i].is_scalar() && args.iter().any(|x| matches!(x, Value::Sym(_))) {
                // containers with symbolic contents cannot be tabulated
                let has_sym = Self::collect(&[args[i].clone()]).iter().any(|c| match c {
                    Cont::L(l) => l.borrow().iter().any(|v| matches!(v, Value::Sym(_))),
                    Cont::D(d) => d.borrow().items.iter().any(|p| matches!(p.1, Value::Sym(_))),
                });
                if has_sym {
                    return terr("cannot tabulate a call whose container arguments hold symbolic values");
                }
            }
        }
        let total: usize = dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).unwrap_or(usize::MAX);
        if total > self.opts.max_table {
            return terr(format!(
                "cannot tabulate: the joint argument domain has {total} points (> max_table={})",
                self.opts.max_table
            ));
        }
        let mut grids: Vec<Vec<f64>> = vec![];
        let mut kinds: Vec<Kind> = vec![];
        let mut tuple_out: Option<usize> = None;
        let mut undefined = 0usize;
        let mut cur = args.to_vec();
        for j in 0..total {
            let mut rem = j;
            for p in (0..sym_pos.len()).rev() {
                cur[sym_pos[p]] = Value::Int(los[p] as i64 + (rem % dims[p]) as i64);
                rem /= dims[p];
            }
            // bools stay bools
            for (p, &i) in sym_pos.iter().enumerate() {
                if self.g.ty(ids[p]).kind == Kind::Bool {
                    cur[i] = Value::Bool(matches!(cur[i], Value::Int(1)));
                }
            }
            let out = f(self, &cur);
            let parts: Option<Vec<Value>> = match out {
                Ok(Value::Tuple(t)) => {
                    tuple_out.get_or_insert(t.len());
                    Some(t.to_vec())
                }
                Ok(v) if v.is_scalar() => Some(vec![v]),
                Ok(v) => return terr(format!("tabulated call returned a {} (expected numbers)", v.type_name())),
                Err(e) if e.recursion => return Err(e),
                Err(_) => None,
            };
            match parts {
                Some(ps) => {
                    if grids.is_empty() {
                        grids = vec![vec![f64::NAN; total]; ps.len()];
                        kinds = vec![Kind::Bool; ps.len()];
                    }
                    if ps.len() != grids.len() {
                        return terr("tabulated call returns tuples of varying length");
                    }
                    for (k, v) in ps.iter().enumerate() {
                        grids[k][j] = self.num(v);
                        kinds[k] = match (kinds[k], v) {
                            (_, Value::Float(_)) | (Kind::Real, _) => Kind::Real,
                            (_, Value::Int(_)) | (Kind::Int, _) => Kind::Int,
                            _ => Kind::Bool,
                        };
                    }
                }
                None => undefined += 1,
            }
        }
        if grids.is_empty() {
            return terr("tabulated call raised for every argument value");
        }
        if undefined > 0 {
            self.warnings.push(format!(
                "tabulated call raised for {undefined} of {total} argument values; those inputs are outside the domain"
            ));
        }
        let mut outs = vec![];
        for (k, mut grid) in grids.into_iter().enumerate() {
            let fill = grid.iter().copied().find(|v| !v.is_nan()).unwrap_or(0.0);
            for v in grid.iter_mut() {
                if v.is_nan() {
                    *v = fill;
                }
            }
            let n = self.g.table(ids.clone(), los.clone(), dims.clone(), grid, kinds[k])?;
            outs.push(self.from_node(n));
        }
        Ok(if tuple_out.is_some() { Value::tuple(outs) } else { outs.pop().unwrap() })
    }

    // ================================================================= calls

    fn resolved(&mut self, r: Resolved, what: &str) -> R<Value> {
        Ok(match r {
            Resolved::Value(v) => v,
            Resolved::Builtin(b) => Value::Builtin(Rc::from(b.as_str())),
            Resolved::Object(h) => Value::Object(h),
            Resolved::Function { source, name, file, first_line, ctx, handle } => {
                self.parse_function(&source, &name, &file, first_line, Ctx::Host(ctx), handle)?
            }
            Resolved::Missing => return terr(format!("NameError: name '{what}' is not defined")),
        })
    }

    pub fn parse_function(
        &mut self,
        source: &str,
        name: &str,
        file: &str,
        first_line: usize,
        ctx: Ctx,
        handle: Option<u64>,
    ) -> R<Value> {
        let suite = ast::Suite::parse(source, file).map_err(|e| TraceError {
            msg: format!("cannot parse source of '{name}': {e}"),
            loc: None,
            recursion: false,
        })?;
        let defs: Vec<&ast::StmtFunctionDef> = suite
            .iter()
            .filter_map(|s| if let ast::Stmt::FunctionDef(d) = s { Some(d) } else { None })
            .collect();
        let d = defs
            .iter()
            .find(|d| d.name.as_str() == name)
            .or(if defs.len() == 1 { defs.first() } else { None })
            .ok_or_else(|| TraceError { msg: format!("no function '{name}' in source"), loc: None, recursion: false })?;
        Ok(Value::Func(Rc::new(Func {
            name: name.to_string(),
            body: Rc::new(FuncBody::Def((*d).clone())),
            src: Rc::new(Src::new(file, first_line, source)),
            ctx,
            handle,
        })))
    }

    fn lookup(&mut self, name: &str, st: &State) -> R<Value> {
        if let Some(v) = st.env.get(name) {
            if let Value::Undef = v {
                return terr(format!("name '{name}' is not assigned on any path reaching this point"));
            }
            return Ok(v.clone());
        }
        let mut ctx = self.current_ctx();
        loop {
            match ctx {
                Ctx::Closure(m, outer) => {
                    if let Some(v) = m.borrow().get(name) {
                        return Ok(v.clone());
                    }
                    ctx = *outer;
                }
                Ctx::Host(id) => {
                    if let Some(v) = self.cache.get(&(id, name.to_string())) {
                        return Ok(v.clone());
                    }
                    let r = self.host.resolve(id, name).map_err(|e| TraceError { msg: e, loc: None, recursion: false })?;
                    let v = match r {
                        Resolved::Missing if BUILTINS.contains(&name) => Value::Builtin(Rc::from(name)),
                        r => self.resolved(r, name)?,
                    };
                    self.cache.insert((id, name.to_string()), v.clone());
                    return Ok(v);
                }
            }
        }
    }

    fn has_sym(v: &Value) -> bool {
        match v {
            Value::Sym(_) => true,
            Value::List(l) => l.borrow().iter().any(Self::has_sym),
            Value::Tuple(t) => t.iter().any(Self::has_sym),
            Value::Dict(d) => d.borrow().items.iter().any(|p| Self::has_sym(&p.1)),
            Value::SymRange(..) => true,
            _ => false,
        }
    }

    fn call(&mut self, f: &Value, args: Vec<Value>, kwargs: Vec<(String, Value)>, st: Option<&State>) -> R<Value> {
        let any_sym = args.iter().any(Self::has_sym) || kwargs.iter().any(|p| Self::has_sym(&p.1));
        match f {
            Value::Func(func) => {
                if !any_sym && kwargs.is_empty() {
                    if let Some(h) = func.handle {
                        if let Ok(v) = self.host.call(h, &args) {
                            return Ok(v);
                        }
                    }
                }
                if let Some(s) = st {
                    let mut r: Vec<Value> = s.env.values().cloned().collect();
                    r.push(s.rv.clone());
                    self.roots.push(r);
                }
                let key = Rc::as_ptr(func) as usize;
                *self.active.entry(key).or_insert(0) += 1;
                let res = self.inline(func, args.clone(), kwargs.clone());
                let left = {
                    let c = self.active.get_mut(&key).unwrap();
                    *c -= 1;
                    *c
                };
                if st.is_some() {
                    self.roots.pop();
                }
                match res {
                    Err(e) if e.recursion && left == 0 && kwargs.is_empty() && func.handle.is_some() => {
                        let h = func.handle.unwrap();
                        self.warnings.push(format!(
                            "recursion in '{}' does not unroll to a bounded depth; tabulated over its argument domain",
                            func.name
                        ));
                        self.tabulate(&args, &mut |me, xs| {
                            me.host.call(h, xs).map_err(|m| TraceError { msg: m, loc: None, recursion: false })
                        })
                    }
                    r => r,
                }
            }
            Value::Builtin(name) => self.call_builtin(name, args, kwargs),
            Value::Object(h) => {
                if !kwargs.is_empty() {
                    return terr("keyword arguments to opaque host callables are not supported");
                }
                let h = *h;
                if !any_sym {
                    return self.host.call(h, &args).map_err(|m| TraceError { msg: m, loc: None, recursion: false });
                }
                self.tabulate(&args, &mut |me, xs| {
                    me.host.call(h, xs).map_err(|m| TraceError { msg: m, loc: None, recursion: false })
                })
            }
            Value::Method(recv, name) => self.call_method(recv, name, args, kwargs),
            o => terr(format!("'{}' object is not callable", o.type_name())),
        }
    }

    fn inline(&mut self, func: &Rc<Func>, args: Vec<Value>, kwargs: Vec<(String, Value)>) -> R<Value> {
        if self.depth >= self.opts.max_inline_depth {
            return Err(TraceError {
                msg: format!(
                    "call depth exceeded max_inline_depth={} in '{}': recursion must terminate on \
                     compile-time-known values",
                    self.opts.max_inline_depth, func.name
                ),
                loc: None,
                recursion: true,
            });
        }
        self.depth += 1;
        self.ctxs.push(func.ctx.clone());
        self.srcs.push(func.src.clone());
        let res = self.inline_inner(func, args, kwargs);
        self.srcs.pop();
        self.ctxs.pop();
        self.depth -= 1;
        res
    }

    fn inline_inner(&mut self, func: &Rc<Func>, args: Vec<Value>, mut kwargs: Vec<(String, Value)>) -> R<Value> {
        let a = func.args();
        let mut env: HashMap<Rc<str>, Value> = HashMap::new();
        let mut dummy = State::new(HashMap::new());
        let positional: Vec<&ast::ArgWithDefault> = a.posonlyargs.iter().chain(a.args.iter()).collect();
        let mut args_it = args.into_iter();
        for p in &positional {
            let name = p.def.arg.as_str();
            let v = if let Some(v) = args_it.next() {
                v
            } else if let Some(i) = kwargs.iter().position(|k| k.0 == name) {
                kwargs.remove(i).1
            } else if let Some(d) = &p.default {
                self.eval(d, &mut dummy)?
            } else {
                return terr(format!("{}() missing required argument '{name}'", func.name));
            };
            env.insert(Rc::from(name), v);
        }
        let extra: Vec<Value> = args_it.collect();
        match &a.vararg {
            Some(v) => {
                env.insert(Rc::from(v.arg.as_str()), Value::tuple(extra));
            }
            None if !extra.is_empty() => {
                return terr(format!("{}() got {} unexpected positional arguments", func.name, extra.len()))
            }
            None => {}
        }
        for p in &a.kwonlyargs {
            let name = p.def.arg.as_str();
            let v = if let Some(i) = kwargs.iter().position(|k| k.0 == name) {
                kwargs.remove(i).1
            } else if let Some(d) = &p.default {
                self.eval(d, &mut dummy)?
            } else {
                return terr(format!("{}() missing keyword-only argument '{name}'", func.name));
            };
            env.insert(Rc::from(name), v);
        }
        match &a.kwarg {
            Some(k) => {
                let d = Dict { items: kwargs.into_iter().map(|(n, v)| (Key::Str(Rc::from(n.as_str())), v)).collect() };
                env.insert(Rc::from(k.arg.as_str()), Value::dict(d));
            }
            None if !kwargs.is_empty() => {
                return terr(format!("{}() got an unexpected keyword argument '{}'", func.name, kwargs[0].0))
            }
            None => {}
        }
        let mut st = State::new(env);
        match &*func.body {
            FuncBody::Lambda(l) => self.eval(&l.body, &mut st),
            FuncBody::Def(d) => {
                let st = self.exec_block(&d.body, st)?;
                match &st.ret {
                    Value::Bool(true) => Ok(st.rv),
                    Value::Bool(false) => Ok(Value::None),
                    Value::Sym(r) => {
                        let r = *r;
                        self.merge(r, &st.rv, &Value::None).map_err(|_| TraceError {
                            msg: format!("'{}' returns a value on some paths and falls off the end (None) on others", func.name),
                            loc: None,
                            recursion: false,
                        })
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    fn reduce_minmax(&mut self, items: Vec<Value>, max: bool) -> R<Value> {
        if items.is_empty() {
            return terr(format!("{}() arg is an empty sequence", if max { "max" } else { "min" }));
        }
        let mut layer = items;
        while layer.len() > 1 {
            let mut next = vec![];
            for pair in layer.chunks(2) {
                if pair.len() == 1 {
                    next.push(pair[0].clone());
                    continue;
                }
                next.push(self.minmax2(&pair[0], &pair[1], max)?);
            }
            layer = next;
        }
        Ok(layer.pop().unwrap())
    }

    fn minmax2(&mut self, a: &Value, b: &Value, max: bool) -> R<Value> {
        if !(a.is_scalar() && b.is_scalar()) {
            return terr("min/max of non-numbers is not supported");
        }
        if !matches!(a, Value::Sym(_)) && !matches!(b, Value::Sym(_)) {
            let (x, y) = (self.num(a), self.num(b));
            return Ok(if (max && y > x) || (!max && y < x) { b.clone() } else { a.clone() });
        }
        let (x, y) = (self.lift(a)?, self.lift(b)?);
        let n = self.g.bin(if max { BinOp::Max } else { BinOp::Min }, x, y)?;
        Ok(self.from_node(n))
    }

    fn sort_values(&mut self, mut items: Vec<Value>, reverse: bool) -> R<Vec<Value>> {
        if items.iter().all(|v| v.is_scalar() && !matches!(v, Value::Sym(_))) {
            let mut idx: Vec<usize> = (0..items.len()).collect();
            idx.sort_by(|&i, &j| self.num(&items[i]).partial_cmp(&self.num(&items[j])).unwrap());
            let mut out: Vec<Value> = idx.into_iter().map(|i| items[i].clone()).collect();
            if reverse {
                out.reverse();
            }
            return Ok(out);
        }
        if !items.iter().all(|v| v.is_scalar()) {
            return terr("sorting symbolic values requires a list of numbers");
        }
        // Batcher's merge-exchange sorting network (Knuth, Algorithm 5.2.2M)
        for (i, j) in batcher_pairs(items.len()) {
            let lo = self.minmax2(&items[i], &items[j], false)?;
            let hi = self.minmax2(&items[i], &items[j], true)?;
            items[i] = lo;
            items[j] = hi;
        }
        if reverse {
            items.reverse();
        }
        Ok(items)
    }

    fn call_builtin(&mut self, name: &str, args: Vec<Value>, kwargs: Vec<(String, Value)>) -> R<Value> {
        let kw = |k: &str| kwargs.iter().find(|p| p.0 == k).map(|p| p.1.clone());
        let arg = |i: usize| -> R<Value> {
            args.get(i).cloned().ok_or_else(|| TraceError {
                msg: format!("{name}() missing argument {}", i + 1),
                loc: None,
                recursion: false,
            })
        };
        let unary = |me: &mut Self, v: &Value, op: UnOp| -> R<Value> {
            let x = me.lift(v)?;
            let n = me.g.un(op, x)?;
            Ok(me.from_node(n))
        };
        match name {
            "abs" | "math.fabs" => {
                let v = arg(0)?;
                match v {
                    Value::Int(i) => Ok(Value::Int(i.abs())),
                    Value::Bool(b) => Ok(Value::Int(b as i64)),
                    Value::Float(f) => Ok(Value::Float(f.abs())),
                    _ => unary(self, &v, UnOp::Abs),
                }
            }
            "neurify.relu" => {
                let v = arg(0)?;
                if !matches!(v, Value::Sym(_)) {
                    let x = self.num(&v).max(0.0);
                    return Ok(if matches!(v, Value::Float(_)) { Value::Float(x) } else { Value::Int(x as i64) });
                }
                unary(self, &v, UnOp::Relu)
            }
            "min" | "max" => {
                if kw("key").is_some() {
                    return terr("min/max with key= is not supported");
                }
                let items = if args.len() == 1 { self.iterate(&args[0])? } else { args.clone() };
                if items.is_empty() {
                    if let Some(d) = kw("default") {
                        return Ok(d);
                    }
                }
                self.reduce_minmax(items, name == "max")
            }
            "sum" => {
                let items = self.iterate(&arg(0)?)?;
                let mut acc = args.get(1).cloned().or(kw("start")).unwrap_or(Value::Int(0));
                for it in items {
                    acc = self.binop(ast::Operator::Add, &acc, &it)?;
                }
                Ok(acc)
            }
            "len" => {
                let v = arg(0)?;
                Ok(Value::Int(match &v {
                    Value::Range(a, b, s) => range_len(*a, *b, *s),
                    _ => self.iterate(&v)?.len() as i64,
                }))
            }
            "range" => {
                let (start, stop, step) = match args.len() {
                    1 => (Value::Int(0), args[0].clone(), Value::Int(1)),
                    2 => (args[0].clone(), args[1].clone(), Value::Int(1)),
                    3 => (args[0].clone(), args[1].clone(), args[2].clone()),
                    _ => return terr("range() expects 1 to 3 arguments"),
                };
                let as_int = |v: &Value| match v {
                    Value::Int(i) => Some(*i),
                    Value::Bool(b) => Some(*b as i64),
                    _ => None,
                };
                let Some(stp) = as_int(&step) else { return terr("range() step must be a known integer") };
                if stp == 0 {
                    return terr("range() arg 3 must not be zero");
                }
                match (as_int(&start), as_int(&stop)) {
                    (Some(a), Some(b)) => Ok(Value::Range(a, b, stp)),
                    _ => Ok(Value::SymRange(Box::new(start), Box::new(stop), stp)),
                }
            }
            "enumerate" => {
                let items = self.iterate(&arg(0)?)?;
                let start = match args.get(1).cloned().or(kw("start")) {
                    Some(Value::Int(i)) => i,
                    None => 0,
                    _ => return terr("enumerate() start must be a known integer"),
                };
                Ok(Value::list(
                    items.into_iter().enumerate().map(|(i, v)| Value::tuple(vec![Value::Int(start + i as i64), v])).collect(),
                ))
            }
            "zip" => {
                let seqs: R<Vec<Vec<Value>>> = args.iter().map(|a| self.iterate(a)).collect();
                let seqs = seqs?;
                let n = seqs.iter().map(|s| s.len()).min().unwrap_or(0);
                Ok(Value::list((0..n).map(|i| Value::tuple(seqs.iter().map(|s| s[i].clone()).collect())).collect()))
            }
            "reversed" => {
                let mut v = self.iterate(&arg(0)?)?;
                v.reverse();
                Ok(Value::list(v))
            }
            "list" => Ok(Value::list(if args.is_empty() { vec![] } else { self.iterate(&args[0])? })),
            "tuple" => Ok(Value::tuple(if args.is_empty() { vec![] } else { self.iterate(&args[0])? })),
            "dict" => {
                let mut d = Dict::default();
                if let Some(a) = args.first() {
                    match a {
                        Value::Dict(x) => d = x.borrow().clone(),
                        other => {
                            for it in self.iterate(other)? {
                                let kv = self.iterate(&it)?;
                                if kv.len() != 2 {
                                    return terr("dict() sequence elements must be pairs");
                                }
                                let k = self.key_of(&kv[0])?;
                                d.set(k, kv[1].clone());
                            }
                        }
                    }
                }
                for (k, v) in kwargs {
                    d.set(Key::Str(Rc::from(k.as_str())), v);
                }
                Ok(Value::dict(d))
            }
            "int" | "math.trunc" => {
                let v = arg(0)?;
                match v {
                    Value::Int(_) => Ok(v),
                    Value::Bool(b) => Ok(Value::Int(b as i64)),
                    Value::Float(f) => Ok(Value::Int(f.trunc() as i64)),
                    Value::Str(s) => s.trim().parse::<i64>().map(Value::Int).map_err(|_| TraceError {
                        msg: format!("invalid literal for int(): '{s}'"),
                        loc: None,
                        recursion: false,
                    }),
                    _ => unary(self, &v, UnOp::Trunc),
                }
            }
            "math.floor" | "math.ceil" => {
                let v = arg(0)?;
                let op = if name == "math.floor" { UnOp::Floor } else { UnOp::Ceil };
                match v {
                    Value::Sym(_) => unary(self, &v, op),
                    _ => {
                        let x = self.num(&v);
                        Ok(Value::Int(if op == UnOp::Floor { x.floor() } else { x.ceil() } as i64))
                    }
                }
            }
            "float" => {
                let v = arg(0)?;
                match v {
                    Value::Sym(_) | Value::Float(_) => Ok(v),
                    Value::Bool(_) | Value::Int(_) => Ok(Value::Float(self.num(&v))),
                    Value::Str(s) => s.trim().parse::<f64>().map(Value::Float).map_err(|_| TraceError {
                        msg: format!("could not convert string to float: '{s}'"),
                        loc: None,
                        recursion: false,
                    }),
                    o => terr(format!("float() argument must be a number, not {}", o.type_name())),
                }
            }
            "bool" => self.truth(&args.first().cloned().unwrap_or(Value::Bool(false))),
            "round" => {
                let v = arg(0)?;
                if args.len() > 1 || kw("ndigits").is_some() {
                    return terr("round() with ndigits is not supported");
                }
                match v {
                    Value::Float(f) => {
                        let r = f.round();
                        let r = if (f - f.trunc()).abs() == 0.5 { 2.0 * (f / 2.0).round() } else { r };
                        Ok(Value::Int(r as i64))
                    }
                    Value::Sym(id) if self.g.ty(id).kind.integral() => Ok(v),
                    Value::Sym(_) => terr("round() of a real-valued symbolic value is not supported; use int(x + 0.5) or math.floor"),
                    other => Ok(other),
                }
            }
            "divmod" => {
                let (a, b) = (arg(0)?, arg(1)?);
                let q = self.binop(ast::Operator::FloorDiv, &a, &b)?;
                let r = self.binop(ast::Operator::Mod, &a, &b)?;
                Ok(Value::tuple(vec![q, r]))
            }
            "pow" => {
                if args.len() > 2 {
                    return terr("pow() with modulus is not supported");
                }
                self.binop(ast::Operator::Pow, &arg(0)?, &arg(1)?)
            }
            "sorted" => {
                if kw("key").is_some() {
                    return terr("sorted() with key= is not supported");
                }
                let rev = match kw("reverse") {
                    Some(v) => is_true(&self.truth(&v)?),
                    None => false,
                };
                let items = self.iterate(&arg(0)?)?;
                Ok(Value::list(self.sort_values(items, rev)?))
            }
            "any" | "all" => {
                let items = self.iterate(&arg(0)?)?;
                let mut acc = Value::Bool(name == "all");
                for it in items {
                    let t = self.truth(&it)?;
                    acc = if name == "all" { self.b_and(&acc, &t)? } else { self.b_or(&acc, &t)? };
                }
                Ok(acc)
            }
            "print" => Ok(Value::None),
            "isinstance" => {
                let v = arg(0)?;
                let types = match arg(1)? {
                    Value::Tuple(t) => t.to_vec(),
                    t => vec![t],
                };
                let kind = self.kind_of(&v);
                let mut r = false;
                for t in types {
                    let Value::Builtin(tn) = t else { return terr("isinstance() supports builtin types only") };
                    r |= match &*tn {
                        "int" => matches!(kind, Some(Kind::Int) | Some(Kind::Bool)),
                        "float" => kind == Some(Kind::Real),
                        "bool" => kind == Some(Kind::Bool),
                        "list" => matches!(v, Value::List(_)),
                        "tuple" => matches!(v, Value::Tuple(_)),
                        "dict" => matches!(v, Value::Dict(_)),
                        _ => false,
                    };
                }
                Ok(Value::Bool(r))
            }
            "neurify.where" => {
                let c = self.truth(&arg(0)?)?;
                let (a, b) = (arg(1)?, arg(2)?);
                match c {
                    Value::Bool(true) => Ok(a),
                    Value::Bool(false) => Ok(b),
                    Value::Sym(id) => self.merge(id, &a, &b),
                    _ => unreachable!(),
                }
            }
            _ => terr(format!("builtin '{name}' is not supported")),
        }
    }

    fn call_method(&mut self, recv: &Value, name: &str, args: Vec<Value>, kwargs: Vec<(String, Value)>) -> R<Value> {
        let arg = |i: usize| -> R<Value> {
            args.get(i).cloned().ok_or_else(|| TraceError {
                msg: format!("{name}() missing argument"),
                loc: None,
                recursion: false,
            })
        };
        match recv {
            Value::List(l) => match name {
                "append" => {
                    l.borrow_mut().push(arg(0)?);
                    Ok(Value::None)
                }
                "extend" => {
                    let items = self.iterate(&arg(0)?)?;
                    l.borrow_mut().extend(items);
                    Ok(Value::None)
                }
                "insert" => {
                    let n = l.borrow().len() as i64;
                    let Value::Int(i) = arg(0)? else { return terr("insert() index must be a known integer") };
                    let i = if i < 0 { (i + n).max(0) } else { i.min(n) } as usize;
                    l.borrow_mut().insert(i, arg(1)?);
                    Ok(Value::None)
                }
                "pop" => {
                    let n = l.borrow().len();
                    let i = self.index_of(&args.first().cloned().unwrap_or(Value::Int(-1)), n)?;
                    Ok(l.borrow_mut().remove(i))
                }
                "copy" => Ok(Value::list(l.borrow().clone())),
                "clear" => {
                    l.borrow_mut().clear();
                    Ok(Value::None)
                }
                "reverse" => {
                    l.borrow_mut().reverse();
                    Ok(Value::None)
                }
                "sort" => {
                    if kwargs.iter().any(|p| p.0 == "key") {
                        return terr("sort() with key= is not supported");
                    }
                    let rev = match kwargs.iter().find(|p| p.0 == "reverse") {
                        Some(p) => is_true(&self.truth(&p.1)?),
                        None => false,
                    };
                    let items = l.borrow().clone();
                    let sorted = self.sort_values(items, rev)?;
                    *l.borrow_mut() = sorted;
                    Ok(Value::None)
                }
                "count" => {
                    let x = arg(0)?;
                    let items = l.borrow().clone();
                    let mut acc = Value::Int(0);
                    for it in items.iter() {
                        let e = self.struct_eq(it, &x)?;
                        acc = self.binop(ast::Operator::Add, &acc, &e)?;
                    }
                    Ok(acc)
                }
                "index" => {
                    let x = arg(0)?;
                    let items = l.borrow().clone();
                    let mut hits = vec![];
                    for (i, it) in items.iter().enumerate() {
                        match self.struct_eq(it, &x)? {
                            Value::Bool(true) => hits.push(i),
                            Value::Bool(false) => {}
                            _ => return terr(format!("list.{name}() on symbolic values is not supported")),
                        }
                    }
                    {
                        hits.first().map(|&i| Value::Int(i as i64)).ok_or_else(|| TraceError {
                            msg: "ValueError: value not in list".into(),
                            loc: None,
                            recursion: false,
                        })
                    }
                }
                _ => terr(format!("list.{name}() is not supported")),
            },
            Value::Dict(d) => match name {
                "get" => {
                    let k = self.key_of(&arg(0)?)?;
                    Ok(d.borrow().get(&k).cloned().unwrap_or(args.get(1).cloned().unwrap_or(Value::None)))
                }
                "keys" => Ok(Value::list(d.borrow().items.iter().map(|p| p.0.to_value()).collect())),
                "values" => Ok(Value::list(d.borrow().items.iter().map(|p| p.1.clone()).collect())),
                "items" => Ok(Value::list(
                    d.borrow().items.iter().map(|p| Value::tuple(vec![p.0.to_value(), p.1.clone()])).collect(),
                )),
                "copy" => Ok(Value::dict(d.borrow().clone())),
                "clear" => {
                    d.borrow_mut().items.clear();
                    Ok(Value::None)
                }
                "update" => {
                    if let Some(o) = args.first() {
                        let other = self.call_builtin("dict", vec![o.clone()], vec![])?;
                        let Value::Dict(o) = other else { unreachable!() };
                        let items = o.borrow().items.clone();
                        for (k, v) in items {
                            d.borrow_mut().set(k, v);
                        }
                    }
                    for (k, v) in kwargs {
                        d.borrow_mut().set(Key::Str(Rc::from(k.as_str())), v);
                    }
                    Ok(Value::None)
                }
                "pop" => {
                    let k = self.key_of(&arg(0)?)?;
                    match d.borrow_mut().remove(&k) {
                        Some(v) => Ok(v),
                        None => args.get(1).cloned().ok_or_else(|| TraceError {
                            msg: format!("KeyError: {k}"),
                            loc: None,
                            recursion: false,
                        }),
                    }
                }
                "setdefault" => {
                    let k = self.key_of(&arg(0)?)?;
                    if let Some(v) = d.borrow().get(&k) {
                        return Ok(v.clone());
                    }
                    let v = args.get(1).cloned().unwrap_or(Value::None);
                    d.borrow_mut().set(k, v.clone());
                    Ok(v)
                }
                _ => terr(format!("dict.{name}() is not supported")),
            },
            o => terr(format!("'{}' has no method '{name}'", o.type_name())),
        }
    }

    // =========================================================== expressions

    fn eval_args(&mut self, args: &[ast::Expr], st: &mut State) -> R<Vec<Value>> {
        let mut out = vec![];
        for a in args {
            if let ast::Expr::Starred(s) = a {
                let v = self.eval(&s.value, st)?;
                out.extend(self.iterate(&v)?);
            } else {
                out.push(self.eval(a, st)?);
            }
        }
        Ok(out)
    }

    fn comprehension(
        &mut self,
        gens: &[ast::Comprehension],
        st: &mut State,
        emit: &mut dyn FnMut(&mut Self, &mut State) -> R<()>,
    ) -> R<()> {
        let Some((g, rest)) = gens.split_first() else { return emit(self, st) };
        let it = self.eval(&g.iter, st)?;
        for item in self.iterate(&it)? {
            let mut inner = st.clone();
            self.assign(&g.target, item, &mut inner)?;
            let mut keep = true;
            for cond in &g.ifs {
                let v = self.eval(cond, &mut inner)?;
                match self.truth(&v)? {
                    Value::Bool(b) => keep &= b,
                    _ => {
                        return terr(
                            "a data-dependent filter in a comprehension would make its length depend on \
                             the inputs; use a conditional expression in the element instead",
                        )
                    }
                }
            }
            if keep {
                self.comprehension(rest, &mut inner, emit)?;
            }
        }
        Ok(())
    }

    fn make_lambda(&mut self, l: &ast::ExprLambda, st: &State) -> Value {
        Value::Func(Rc::new(Func {
            name: "<lambda>".into(),
            body: Rc::new(FuncBody::Lambda(l.clone())),
            src: self.srcs.last().unwrap().clone(),
            ctx: Ctx::Closure(Rc::new(RefCell::new(st.env.clone())), Box::new(self.current_ctx())),
            handle: None,
        }))
    }

    fn eval(&mut self, e: &ast::Expr, st: &mut State) -> R<Value> {
        use ast::Expr::*;
        match e {
            Constant(c) => const_value(&c.value),
            Name(n) => self.lookup(n.id.as_str(), st),
            BinOp(b) => {
                let l = self.eval(&b.left, st)?;
                let r = self.eval(&b.right, st)?;
                self.binop(b.op, &l, &r)
            }
            UnaryOp(u) => {
                let v = self.eval(&u.operand, st)?;
                self.unaryop(u.op, &v)
            }
            BoolOp(b) => {
                let is_and = matches!(b.op, ast::BoolOp::And);
                let mut result = self.eval(&b.values[0], st)?;
                for v in &b.values[1..] {
                    let t = self.truth(&result)?;
                    match t {
                        Value::Bool(x) => {
                            if x != is_and {
                                return Ok(result); // short-circuit
                            }
                            result = self.eval(v, st)?;
                        }
                        Value::Sym(c) => {
                            let rhs = self.eval(v, st)?;
                            let both_bool =
                                self.kind_of(&result) == Some(Kind::Bool) && self.kind_of(&rhs) == Some(Kind::Bool);
                            result = if both_bool {
                                if is_and {
                                    self.b_and(&result, &rhs)?
                                } else {
                                    self.b_or(&result, &rhs)?
                                }
                            } else if is_and {
                                self.merge(c, &rhs, &result)?
                            } else {
                                self.merge(c, &result, &rhs)?
                            };
                        }
                        _ => unreachable!(),
                    }
                }
                Ok(result)
            }
            Compare(c) => {
                let mut left = self.eval(&c.left, st)?;
                let mut acc = Value::Bool(true);
                for (op, right) in c.ops.iter().zip(&c.comparators) {
                    let r = self.eval(right, st)?;
                    let v = self.compare(*op, &left, &r)?;
                    acc = self.b_and(&acc, &v)?;
                    if matches!(acc, Value::Bool(false)) {
                        return Ok(acc);
                    }
                    left = r;
                }
                Ok(acc)
            }
            IfExp(i) => {
                let v = self.eval(&i.test, st)?;
                match self.truth(&v)? {
                    Value::Bool(true) => self.eval(&i.body, st),
                    Value::Bool(false) => self.eval(&i.orelse, st),
                    Value::Sym(c) => {
                        let a = self.eval(&i.body, st)?;
                        let b = self.eval(&i.orelse, st)?;
                        self.merge(c, &a, &b)
                    }
                    _ => unreachable!(),
                }
            }
            Call(c) => {
                let f = self.eval(&c.func, st)?;
                let args = self.eval_args(&c.args, st)?;
                let mut kwargs = vec![];
                for k in &c.keywords {
                    let v = self.eval(&k.value, st)?;
                    match &k.arg {
                        Some(n) => kwargs.push((n.to_string(), v)),
                        None => {
                            let Value::Dict(d) = v else { return terr("** argument must be a dict") };
                            for (key, val) in d.borrow().items.iter() {
                                let Key::Str(s) = key else { return terr("** keys must be strings") };
                                kwargs.push((s.to_string(), val.clone()));
                            }
                        }
                    }
                }
                let snapshot = st.clone();
                self.call(&f, args, kwargs, Some(&snapshot))
            }
            Attribute(a) => {
                let v = self.eval(&a.value, st)?;
                let attr = a.attr.as_str();
                match &v {
                    Value::List(_) | Value::Dict(_) => Ok(Value::Method(Box::new(v.clone()), Rc::from(attr))),
                    Value::Object(h) => {
                        let r = self.host.getattr(*h, attr).map_err(|m| TraceError { msg: m, loc: None, recursion: false })?;
                        self.resolved(r, attr)
                    }
                    Value::Sym(_) | Value::Int(_) | Value::Float(_) | Value::Bool(_) if attr == "real" => Ok(v.clone()),
                    o => terr(format!("'{}' object has no attribute '{attr}'", o.type_name())),
                }
            }
            Subscript(s) => {
                let c = self.eval(&s.value, st)?;
                if let Slice(sl) = &*s.slice {
                    return self.slice(&c, sl, st);
                }
                let k = self.eval(&s.slice, st)?;
                self.getitem(&c, &k)
            }
            List(l) => Ok(Value::list(self.eval_args(&l.elts, st)?)),
            Tuple(t) => Ok(Value::tuple(self.eval_args(&t.elts, st)?)),
            Dict(d) => {
                let mut out = crate::value::Dict::default();
                for (k, v) in d.keys.iter().zip(&d.values) {
                    let val = self.eval(v, st)?;
                    match k {
                        Some(k) => {
                            let kv = self.eval(k, st)?;
                            let key = self.key_of(&kv)?;
                            out.set(key, val);
                        }
                        None => {
                            let Value::Dict(o) = val else { return terr("** in a dict display needs a dict") };
                            for (kk, vv) in o.borrow().items.iter() {
                                out.set(kk.clone(), vv.clone());
                            }
                        }
                    }
                }
                Ok(Value::dict(out))
            }
            ListComp(l) => {
                let mut out = vec![];
                self.comprehension(&l.generators, st, &mut |me, s| {
                    out.push(me.eval(&l.elt, s)?);
                    Ok(())
                })?;
                Ok(Value::list(out))
            }
            GeneratorExp(l) => {
                let mut out = vec![];
                self.comprehension(&l.generators, st, &mut |me, s| {
                    out.push(me.eval(&l.elt, s)?);
                    Ok(())
                })?;
                Ok(Value::list(out))
            }
            DictComp(l) => {
                let mut out = crate::value::Dict::default();
                self.comprehension(&l.generators, st, &mut |me, s| {
                    let k = me.eval(&l.key, s)?;
                    let key = me.key_of(&k)?;
                    let v = me.eval(&l.value, s)?;
                    out.set(key, v);
                    Ok(())
                })?;
                Ok(Value::dict(out))
            }
            Lambda(l) => Ok(self.make_lambda(l, st)),
            NamedExpr(n) => {
                let v = self.eval(&n.value, st)?;
                self.assign(&n.target, v.clone(), st)?;
                Ok(v)
            }
            Starred(_) => terr("starred expression in an unsupported position"),
            Slice(_) => terr("slice outside of a subscript"),
            o => terr(format!("unsupported expression: {}", expr_name(o))),
        }
    }

    // =================================================================== API

    fn build_input(&mut self, name: &str, spec: &InSpec) -> R<Value> {
        Ok(match spec {
            InSpec::Real(lo, hi) => Value::Sym(self.g.input(name, Kind::Real, *lo, *hi)?),
            InSpec::Int(lo, hi) => {
                let id = self.g.input(name, Kind::Int, *lo, *hi)?;
                self.from_node(id)
            }
            InSpec::Bool => Value::Sym(self.g.input(name, Kind::Bool, 0.0, 1.0)?),
            InSpec::List(items) => {
                let v: R<Vec<Value>> =
                    items.iter().enumerate().map(|(i, s)| self.build_input(&format!("{name}[{i}]"), s)).collect();
                Value::list(v?)
            }
            InSpec::Tuple(items) => {
                let v: R<Vec<Value>> =
                    items.iter().enumerate().map(|(i, s)| self.build_input(&format!("{name}[{i}]"), s)).collect();
                Value::tuple(v?)
            }
            InSpec::Dict(items) => {
                let mut d = crate::value::Dict::default();
                for (k, s) in items {
                    let v = self.build_input(&format!("{name}[{k}]"), s)?;
                    d.set(k.clone(), v);
                }
                Value::dict(d)
            }
            InSpec::Seq(items) => {
                let f = self.n_seq;
                self.n_seq += 1;
                let mut v = vec![];
                for (i, s) in items.iter().enumerate() {
                    let name = format!("{name}[{i}]");
                    let id = match s {
                        InSpec::Real(lo, hi) => self.g.input_token(&name, Kind::Real, *lo, *hi, f, i)?,
                        InSpec::Int(lo, hi) => self.g.input_token(&name, Kind::Int, *lo, *hi, f, i)?,
                        InSpec::Bool => self.g.input_token(&name, Kind::Bool, 0.0, 1.0, f, i)?,
                        _ => return terr("Seq elements must be scalars (Real / Int / Bool)"),
                    };
                    v.push(self.from_node(id));
                }
                Value::list(v)
            }
            InSpec::Static(v) => v.clone(),
        })
    }

    fn flatten(&mut self, v: &Value) -> R<OutSpec> {
        Ok(match v {
            x if x.is_scalar() => {
                let id = self.lift(x)?;
                self.g.output(id)?;
                OutSpec::Leaf(self.g.ty(id).kind.name())
            }
            Value::Tuple(t) => OutSpec::Tuple(t.iter().map(|x| self.flatten(x)).collect::<R<_>>()?),
            Value::List(l) => {
                let items = l.borrow().clone();
                OutSpec::List(items.iter().map(|x| self.flatten(x)).collect::<R<_>>()?)
            }
            Value::Dict(d) => {
                let items = d.borrow().items.clone();
                OutSpec::Dict(items.iter().map(|(k, x)| Ok((k.clone(), self.flatten(x)?))).collect::<R<_>>()?)
            }
            Value::None => OutSpec::None,
            Value::Undef => return terr("the function never returns a value (every path raises)"),
            o => return terr(format!("cannot return a {} from a compiled function", o.type_name())),
        })
    }
}

/// Result of tracing a Python function.
pub struct Traced {
    pub graph: Graph,
    pub out: OutSpec,
    pub warnings: Vec<String>,
}

/// Trace `name` defined in `source` on the given parameter domains.
#[allow(clippy::too_many_arguments)]
pub fn trace(
    source: &str,
    name: &str,
    file: &str,
    first_line: usize,
    ctx: u64,
    handle: Option<u64>,
    params: &[(String, InSpec)],
    host: &mut dyn Host,
    opts: TraceOptions,
) -> R<Traced> {
    let mut it = Interp::new(host, opts);
    // top-level defs of the traced source behave as module globals
    let suite = ast::Suite::parse(source, file).map_err(|e| TraceError {
        msg: format!("cannot parse source of '{name}': {e}"),
        loc: None,
        recursion: false,
    })?;
    let module: Rc<RefCell<HashMap<Rc<str>, Value>>> = Rc::new(RefCell::new(HashMap::new()));
    let mctx = Ctx::Closure(module.clone(), Box::new(Ctx::Host(ctx)));
    let src = Rc::new(Src::new(file, first_line, source));
    let mut main = None;
    for s in &suite {
        if let ast::Stmt::FunctionDef(d) = s {
            let f = Rc::new(Func {
                name: d.name.to_string(),
                body: Rc::new(FuncBody::Def(d.clone())),
                src: src.clone(),
                ctx: mctx.clone(),
                handle: if d.name.as_str() == name { handle } else { None },
            });
            if d.name.as_str() == name {
                main = Some(f.clone());
            }
            module.borrow_mut().insert(Rc::from(d.name.as_str()), Value::Func(f));
        }
    }
    let func = match main {
        Some(f) => f,
        None => {
            let Value::Func(f) = it.parse_function(source, name, file, first_line, Ctx::Host(ctx), handle)? else {
                unreachable!()
            };
            f
        }
    };
    let func = &func;
    let mut kwargs = vec![];
    for (p, spec) in params {
        kwargs.push((p.clone(), it.build_input(p, spec)?));
    }
    let rv = it.inline(func, vec![], kwargs)?;
    let out = it.flatten(&rv)?;
    Ok(Traced { graph: it.g, out, warnings: it.warnings })
}

// ================================================================= helpers

fn const_value(c: &ast::Constant) -> R<Value> {
    Ok(match c {
        ast::Constant::None => Value::None,
        ast::Constant::Bool(b) => Value::Bool(*b),
        ast::Constant::Str(s) => Value::str(s),
        ast::Constant::Int(i) => Value::Int(i.to_string().parse::<i64>().map_err(|_| TraceError {
            msg: format!("integer literal {i} is out of range"),
            loc: None,
            recursion: false,
        })?),
        ast::Constant::Float(f) => Value::Float(*f),
        ast::Constant::Tuple(t) => Value::tuple(t.iter().map(const_value).collect::<R<_>>()?),
        _ => return terr("unsupported constant (complex / bytes / ellipsis)"),
    })
}

fn range_len(a: i64, b: i64, s: i64) -> i64 {
    if s > 0 {
        ((b - a + s - 1) / s).max(0)
    } else {
        ((a - b - s - 1) / (-s)).max(0)
    }
}

fn range_items(a: i64, b: i64, s: i64) -> R<Vec<i64>> {
    let n = range_len(a, b, s);
    if n > 10_000_000 {
        return terr("range too large to unroll");
    }
    Ok((0..n).map(|k| a + k * s).collect())
}

fn slice_indices(n: i64, lo: Option<i64>, hi: Option<i64>, step: i64) -> R<Vec<usize>> {
    if step == 0 {
        return terr("slice step cannot be zero");
    }
    let norm = |v: i64, d: i64| if v < 0 { (v + n).max(if step > 0 { 0 } else { -1 }) } else { v.min(d) };
    let (start, stop) = if step > 0 {
        (lo.map_or(0, |v| norm(v, n)), hi.map_or(n, |v| norm(v, n)))
    } else {
        (lo.map_or(n - 1, |v| norm(v, n - 1)), hi.map_or(-1, |v| norm(v, n - 1)))
    };
    let mut out = vec![];
    let mut i = start;
    while (step > 0 && i < stop) || (step < 0 && i > stop) {
        if i >= 0 && i < n {
            out.push(i as usize);
        }
        i += step;
    }
    Ok(out)
}

/// Comparator pairs of Batcher's merge-exchange network for n keys.
pub fn batcher_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut out = vec![];
    if n < 2 {
        return out;
    }
    let t = usize::BITS - (n - 1).leading_zeros();
    let mut p = 1usize << (t - 1);
    while p > 0 {
        let (mut q, mut r, mut d) = (1usize << (t - 1), 0usize, p);
        loop {
            for i in 0..(n - d) {
                if i & p == r {
                    out.push((i, i + d));
                }
            }
            if q == p {
                break;
            }
            d = q - p;
            q >>= 1;
            r = p;
        }
        p >>= 1;
    }
    out
}

fn stmt_name(s: &ast::Stmt) -> &'static str {
    use ast::Stmt::*;
    match s {
        ClassDef(_) => "class definition",
        With(_) | AsyncWith(_) => "with",
        Match(_) => "match",
        Try(_) | TryStar(_) => "try",
        Global(_) => "global",
        Nonlocal(_) => "nonlocal",
        Import(_) | ImportFrom(_) => "import (import at module level instead)",
        AsyncFunctionDef(_) | AsyncFor(_) => "async",
        _ => "statement",
    }
}

fn expr_name(e: &ast::Expr) -> &'static str {
    use ast::Expr::*;
    match e {
        Set(_) | SetComp(_) => "set",
        Await(_) => "await",
        Yield(_) | YieldFrom(_) => "yield",
        JoinedStr(_) | FormattedValue(_) => "f-string",
        _ => "expression",
    }
}
