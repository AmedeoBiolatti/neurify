//! Typed dataflow IR.
//!
//! Every program value is a node with a [`Kind`] (Bool / Int / Real) and sound
//! interval bounds. The builder hash-conses nodes, folds constants and applies
//! cheap algebraic simplifications, so the frontend can emit naively.
//!
//! Semantics follow Python, made *total*: division, floor-division and modulo
//! by zero evaluate to 0 instead of raising. Code that guards against zero
//! (`if b != 0: ...`) therefore compiles to the same function Python computes.

use crate::error::*;
use std::collections::HashMap;

pub type NodeId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Bool,
    Int,
    Real,
}

impl Kind {
    /// Result kind of an arithmetic op (bools promote to int, like Python).
    pub fn arith(a: Kind, b: Kind) -> Kind {
        if a == Kind::Real || b == Kind::Real {
            Kind::Real
        } else {
            Kind::Int
        }
    }
    pub fn integral(self) -> bool {
        self != Kind::Real
    }
    pub fn name(self) -> &'static str {
        match self {
            Kind::Bool => "bool",
            Kind::Int => "int",
            Kind::Real => "real",
        }
    }
    pub fn parse(s: &str) -> Result<Kind> {
        match s {
            "bool" => Ok(Kind::Bool),
            "int" => Ok(Kind::Int),
            "real" | "float" => Ok(Kind::Real),
            _ => err(ErrorKind::Invalid, format!("unknown kind '{s}'")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ty {
    pub kind: Kind,
    pub lo: f64,
    pub hi: f64,
}

impl Ty {
    pub fn new(kind: Kind, lo: f64, hi: f64) -> Ty {
        let (mut lo, mut hi) = (lo, hi);
        if kind.integral() {
            lo = lo.ceil();
            hi = hi.floor();
        }
        if kind == Kind::Bool {
            lo = lo.max(0.0);
            hi = hi.min(1.0);
        }
        Ty { kind, lo, hi }
    }
    pub fn as_const(&self) -> Option<f64> {
        (self.lo == self.hi && self.lo.is_finite()).then_some(self.lo)
    }
    pub fn finite(&self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Max,
    Min,
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    Neg,
    Abs,
    Relu,
    Not,
    Floor,
    Ceil,
    Trunc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl BinOp {
    pub fn parse(s: &str) -> Result<BinOp> {
        use BinOp::*;
        Ok(match s {
            "add" => Add,
            "sub" => Sub,
            "mul" => Mul,
            "div" => Div,
            "floordiv" => FloorDiv,
            "mod" => Mod,
            "max" => Max,
            "min" => Min,
            "and" => And,
            "or" => Or,
            "xor" => Xor,
            _ => return err(ErrorKind::Invalid, format!("unknown binary op '{s}'")),
        })
    }
    fn commutative(self) -> bool {
        use BinOp::*;
        matches!(self, Add | Mul | Max | Min | And | Or | Xor)
    }
}

impl UnOp {
    pub fn parse(s: &str) -> Result<UnOp> {
        use UnOp::*;
        Ok(match s {
            "neg" => Neg,
            "abs" => Abs,
            "relu" => Relu,
            "not" => Not,
            "floor" => Floor,
            "ceil" => Ceil,
            "trunc" => Trunc,
            _ => return err(ErrorKind::Invalid, format!("unknown unary op '{s}'")),
        })
    }
}

impl CmpOp {
    pub fn parse(s: &str) -> Result<CmpOp> {
        use CmpOp::*;
        Ok(match s {
            "lt" => Lt,
            "le" => Le,
            "gt" => Gt,
            "ge" => Ge,
            "eq" => Eq,
            "ne" => Ne,
            _ => return err(ErrorKind::Invalid, format!("unknown comparison '{s}'")),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Input(usize),
    Const(f64),
    Bin(BinOp, NodeId, NodeId),
    Un(UnOp, NodeId),
    Cmp(CmpOp, NodeId, NodeId),
    Select(NodeId, NodeId, NodeId),
    /// Tabulated function of integer arguments (index into `Graph::tables`).
    Table(usize),
    /// Data-dependent element read (index into `Graph::gathers`).
    Gather(usize),
}

/// Dense table of a function over a box of integer arguments (row-major).
#[derive(Clone, Debug)]
pub struct Table {
    pub args: Vec<NodeId>,
    pub lo: Vec<f64>,
    pub dims: Vec<usize>,
    pub grid: Vec<f64>,
}

impl Table {
    pub fn strides(&self) -> Vec<usize> {
        let mut st = vec![1usize; self.dims.len()];
        for i in (0..self.dims.len().saturating_sub(1)).rev() {
            st[i] = st[i + 1] * self.dims[i + 1];
        }
        st
    }
    pub fn get(&self, xs: &[f64]) -> f64 {
        let st = self.strides();
        let mut idx = 0usize;
        for (i, &x) in xs.iter().enumerate() {
            let k = x - self.lo[i];
            if !(k >= 0.0 && (k as usize) < self.dims[i]) {
                return 0.0;
            }
            idx += k as usize * st[i];
        }
        self.grid[idx]
    }
}

impl Op {
    /// Operands (tables excluded: use [`Graph::args_of`]).
    pub fn args(&self) -> Vec<NodeId> {
        match *self {
            Op::Input(_) | Op::Const(_) | Op::Table(_) | Op::Gather(_) => vec![],
            Op::Bin(_, a, b) | Op::Cmp(_, a, b) => vec![a, b],
            Op::Un(_, a) => vec![a],
            Op::Select(c, a, b) => vec![c, a, b],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    Const(u64, Kind),
    Bin(BinOp, NodeId, NodeId),
    Un(UnOp, NodeId),
    Cmp(CmpOp, NodeId, NodeId),
    Select(NodeId, NodeId, NodeId),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub ty: Ty,
}

#[derive(Clone, Debug)]
pub struct InputSpec {
    pub name: String,
    pub ty: Ty,
    /// (feature, position) for inputs that live on a sequence token
    /// (transformer layout); None for inputs of the main token
    pub token: Option<(usize, usize)>,
}

/// elems[index] with a data-dependent index (Python semantics for -n..n-1).
#[derive(Clone, Debug)]
pub struct Gather {
    pub elems: Vec<NodeId>,
    pub index: NodeId,
}

impl Gather {
    pub fn pick(&self, idx: f64) -> Option<usize> {
        let n = self.elems.len() as f64;
        let i = if idx < 0.0 { idx + n } else { idx };
        (i >= 0.0 && i < n && i.fract() == 0.0).then_some(i as usize)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub inputs: Vec<InputSpec>,
    pub outputs: Vec<NodeId>,
    pub tables: Vec<Table>,
    pub gathers: Vec<Gather>,
    memo: HashMap<Key, NodeId>,
}

// ---------------------------------------------------------------- semantics

/// CPython's float divmod, made total (x / 0 -> 0).
pub fn py_divmod(vx: f64, wx: f64) -> (f64, f64) {
    if wx == 0.0 {
        return (0.0, 0.0);
    }
    let mut m = vx % wx;
    let mut div = (vx - m) / wx;
    if m != 0.0 {
        if (wx < 0.0) != (m < 0.0) {
            m += wx;
            div -= 1.0;
        }
    } else {
        m = 0.0f64.copysign(wx);
    }
    let fd = if div != 0.0 {
        let mut f = div.floor();
        if div - f > 0.5 {
            f += 1.0;
        }
        f
    } else {
        0.0f64.copysign(vx / wx)
    };
    (fd + 0.0, m + 0.0)
}

fn b2f(b: bool) -> f64 {
    if b {
        1.0
    } else {
        0.0
    }
}

pub fn eval_bin(op: BinOp, a: f64, b: f64) -> f64 {
    use BinOp::*;
    match op {
        Add => a + b,
        Sub => a - b,
        Mul => a * b,
        Div => {
            if b == 0.0 {
                0.0
            } else {
                a / b
            }
        }
        FloorDiv => py_divmod(a, b).0,
        Mod => py_divmod(a, b).1,
        Max => a.max(b),
        Min => a.min(b),
        And => b2f(a != 0.0 && b != 0.0),
        Or => b2f(a != 0.0 || b != 0.0),
        Xor => b2f((a != 0.0) != (b != 0.0)),
    }
}

pub fn eval_un(op: UnOp, a: f64) -> f64 {
    use UnOp::*;
    match op {
        Neg => -a,
        Abs => a.abs(),
        Relu => a.max(0.0),
        Not => b2f(a == 0.0),
        Floor => a.floor(),
        Ceil => a.ceil(),
        Trunc => a.trunc(),
    }
}

pub fn eval_cmp(op: CmpOp, a: f64, b: f64) -> f64 {
    use CmpOp::*;
    b2f(match op {
        Lt => a < b,
        Le => a <= b,
        Gt => a > b,
        Ge => a >= b,
        Eq => a == b,
        Ne => a != b,
    })
}

// ------------------------------------------------------------ interval math

pub fn mulz(x: f64, y: f64) -> f64 {
    if x == 0.0 || y == 0.0 {
        0.0
    } else {
        x * y
    }
}

pub fn hull(vals: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in vals {
        if v.is_nan() {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo, hi)
}

pub fn imul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    hull(&[mulz(a.0, b.0), mulz(a.0, b.1), mulz(a.1, b.0), mulz(a.1, b.1)])
}

fn union(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0.min(b.0), a.1.max(b.1))
}

fn idiv_nonzero(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    hull(&[a.0 / b.0, a.0 / b.1, a.1 / b.0, a.1 / b.1])
}

/// Bounds of a / b under total semantics (x / 0 = 0).
fn idiv(a: (f64, f64), b: (f64, f64), b_integral: bool) -> (f64, f64) {
    if b.0 > 0.0 || b.1 < 0.0 {
        return idiv_nonzero(a, b);
    }
    if b_integral {
        let mut r = (0.0, 0.0);
        if b.0 <= -1.0 {
            r = union(r, idiv_nonzero(a, (b.0, -1.0)));
        }
        if b.1 >= 1.0 {
            r = union(r, idiv_nonzero(a, (1.0, b.1)));
        }
        return r;
    }
    (f64::NEG_INFINITY, f64::INFINITY)
}

fn imod(a: (f64, f64), b: (f64, f64), integral: bool) -> (f64, f64) {
    let one = if integral { 1.0 } else { 0.0 };
    if b.0 == b.1 {
        let c = b.0;
        if c == 0.0 {
            return (0.0, 0.0);
        }
        if c > 0.0 {
            if a.0 >= 0.0 && a.1 < c {
                return a;
            }
            return (0.0, c - one);
        }
        if a.1 <= 0.0 && a.0 > c {
            return a;
        }
        return (c + one, 0.0);
    }
    let lo = if b.0 < 0.0 { b.0 + one } else { 0.0 };
    let mut hi = if b.1 > 0.0 { b.1 - one } else { 0.0 };
    if a.0 >= 0.0 && b.0 >= 0.0 {
        hi = hi.min(a.1);
    }
    (lo.min(0.0), hi.max(0.0))
}

// ------------------------------------------------------------------ builder

impl Graph {
    pub fn new() -> Graph {
        Graph::default()
    }

    pub fn args_of(&self, id: NodeId) -> Vec<NodeId> {
        match self.nodes[id].op {
            Op::Table(t) => self.tables[t].args.clone(),
            Op::Gather(k) => {
                let mut v = self.gathers[k].elems.clone();
                v.push(self.gathers[k].index);
                v
            }
            ref op => op.args(),
        }
    }

    fn is_not_of(&self, a: NodeId, b: NodeId) -> bool {
        matches!(self.nodes[a].op, Op::Un(UnOp::Not, x) if x == b)
            || matches!(self.nodes[b].op, Op::Un(UnOp::Not, x) if x == a)
    }

    /// Tabulated function: value = grid[(args - lo) in row-major order].
    pub fn table(&mut self, args: Vec<NodeId>, lo: Vec<f64>, dims: Vec<usize>, grid: Vec<f64>, kind: Kind) -> Result<NodeId> {
        for &a in &args {
            self.check(a)?;
            if !self.ty(a).kind.integral() {
                return err(ErrorKind::Invalid, "table arguments must be int/bool");
            }
        }
        let cells = dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        if args.len() != lo.len() || args.len() != dims.len() || cells != Some(grid.len()) {
            return err(ErrorKind::Invalid, "inconsistent table shape");
        }
        if lo.iter().any(|v| !v.is_finite() || v.fract() != 0.0) {
            return err(ErrorKind::Invalid, "table lower bounds must be finite integers");
        }
        if grid.iter().any(|v| !v.is_finite()) || grid.is_empty() {
            return err(ErrorKind::Invalid, "table entries must be finite");
        }
        let t = Table { args: args.clone(), lo, dims, grid };
        let consts: Option<Vec<f64>> = args.iter().map(|&a| self.const_of(a)).collect();
        if let Some(xs) = consts {
            return self.constant(t.get(&xs), kind);
        }
        let (lo, hi) = hull(&t.grid);
        self.tables.push(t);
        let ty = Ty::new(kind, lo, hi);
        if let Some(c) = ty.as_const() {
            return Ok(self.const_unchecked(c, kind));
        }
        self.nodes.push(Node { op: Op::Table(self.tables.len() - 1), ty });
        Ok(self.nodes.len() - 1)
    }

    pub fn ty(&self, id: NodeId) -> Ty {
        self.nodes[id].ty
    }

    pub fn const_of(&self, id: NodeId) -> Option<f64> {
        match self.nodes[id].op {
            Op::Const(v) => Some(v),
            _ => None,
        }
    }

    fn is_const(&self, id: NodeId, v: f64) -> bool {
        self.const_of(id) == Some(v)
    }

    fn check(&self, id: NodeId) -> Result<()> {
        if id >= self.nodes.len() {
            return err(ErrorKind::Invalid, format!("node {id} does not exist"));
        }
        Ok(())
    }

    pub fn input(&mut self, name: &str, kind: Kind, lo: f64, hi: f64) -> Result<NodeId> {
        let ty = Ty::new(kind, lo, hi);
        if !(ty.lo <= ty.hi) {
            return err(
                ErrorKind::Invalid,
                format!("input '{name}' has an empty domain [{lo}, {hi}] for kind {}", kind.name()),
            );
        }
        let idx = self.inputs.len();
        self.inputs.push(InputSpec { name: name.to_string(), ty, token: None });
        self.nodes.push(Node { op: Op::Input(idx), ty });
        Ok(self.nodes.len() - 1)
    }

    /// An input living on sequence token `pos`, as feature `feature`.
    pub fn input_token(&mut self, name: &str, kind: Kind, lo: f64, hi: f64, feature: usize, pos: usize) -> Result<NodeId> {
        let id = self.input(name, kind, lo, hi)?;
        self.inputs.last_mut().unwrap().token = Some((feature, pos));
        Ok(id)
    }

    /// elems[index] (index may be symbolic; negative indices wrap).
    pub fn gather(&mut self, elems: Vec<NodeId>, index: NodeId) -> Result<NodeId> {
        self.check(index)?;
        if elems.is_empty() {
            return err(ErrorKind::Invalid, "gather from an empty list");
        }
        for &e in &elems {
            self.check(e)?;
        }
        if !self.ty(index).kind.integral() {
            return err(ErrorKind::Invalid, "list indices must be integers");
        }
        let g = Gather { elems: elems.clone(), index };
        let t = self.ty(index);
        // positions the index can reach
        let mut reach = vec![];
        if t.finite() {
            let mut i = t.lo;
            while i <= t.hi && reach.len() <= elems.len() * 2 {
                if let Some(p) = g.pick(i) {
                    reach.push(p);
                }
                i += 1.0;
            }
        } else {
            reach = (0..elems.len()).collect();
        }
        reach.sort_unstable();
        reach.dedup();
        if reach.is_empty() {
            return err(ErrorKind::Invalid, "IndexError: index is always out of range");
        }
        if reach.len() == 1 {
            return Ok(elems[reach[0]]);
        }
        let tys: Vec<Ty> = reach.iter().map(|&p| self.ty(elems[p])).collect();
        let all_bool = tys.iter().all(|t| t.kind == Kind::Bool);
        let kind = if all_bool { Kind::Bool } else { tys.iter().fold(Kind::Int, |k, t| Kind::arith(k, t.kind)) };
        let lo = tys.iter().fold(f64::INFINITY, |m, t| m.min(t.lo));
        let hi = tys.iter().fold(f64::NEG_INFINITY, |m, t| m.max(t.hi));
        self.gathers.push(g);
        self.nodes.push(Node { op: Op::Gather(self.gathers.len() - 1), ty: Ty::new(kind, lo, hi) });
        Ok(self.nodes.len() - 1)
    }

    pub fn constant(&mut self, v: f64, kind: Kind) -> Result<NodeId> {
        if !v.is_finite() {
            return err(ErrorKind::Invalid, format!("non-finite constant {v}"));
        }
        let ok = match kind {
            Kind::Bool => v == 0.0 || v == 1.0,
            Kind::Int => v.fract() == 0.0,
            Kind::Real => true,
        };
        if !ok {
            return err(ErrorKind::Invalid, format!("constant {v} is not a valid {}", kind.name()));
        }
        Ok(self.const_unchecked(v, kind))
    }

    fn const_unchecked(&mut self, v: f64, kind: Kind) -> NodeId {
        let v = v + 0.0; // -0.0 -> 0.0
        let key = Key::Const(v.to_bits(), kind);
        if let Some(&id) = self.memo.get(&key) {
            return id;
        }
        self.nodes.push(Node { op: Op::Const(v), ty: Ty { kind, lo: v, hi: v } });
        let id = self.nodes.len() - 1;
        self.memo.insert(key, id);
        id
    }

    fn intern(&mut self, op: Op, ty: Ty) -> NodeId {
        if let Some(c) = ty.as_const() {
            return self.const_unchecked(c, ty.kind);
        }
        let key = match op {
            Op::Bin(o, a, b) => Key::Bin(o, a, b),
            Op::Table(_) | Op::Gather(_) => unreachable!(),
            Op::Un(o, a) => Key::Un(o, a),
            Op::Cmp(o, a, b) => Key::Cmp(o, a, b),
            Op::Select(c, a, b) => Key::Select(c, a, b),
            _ => unreachable!(),
        };
        if let Some(&id) = self.memo.get(&key) {
            return id;
        }
        self.nodes.push(Node { op, ty });
        let id = self.nodes.len() - 1;
        self.memo.insert(key, id);
        id
    }

    fn require_bool(&self, id: NodeId, what: &str) -> Result<()> {
        if self.ty(id).kind != Kind::Bool {
            return err(ErrorKind::Invalid, format!("{what} expects bool operands"));
        }
        Ok(())
    }

    pub fn bin(&mut self, op: BinOp, a: NodeId, b: NodeId) -> Result<NodeId> {
        use BinOp::*;
        self.check(a)?;
        self.check(b)?;
        if matches!(op, And | Or | Xor) {
            self.require_bool(a, "and/or/xor")?;
            self.require_bool(b, "and/or/xor")?;
        }
        let (ta, tb) = (self.ty(a), self.ty(b));
        let kind = match op {
            Div => Kind::Real,
            And | Or | Xor => Kind::Bool,
            Max | Min if ta.kind == Kind::Bool && tb.kind == Kind::Bool => Kind::Bool,
            _ => Kind::arith(ta.kind, tb.kind),
        };
        if let (Some(x), Some(y)) = (self.const_of(a), self.const_of(b)) {
            let v = eval_bin(op, x, y);
            return self.constant(v, kind);
        }
        let (a, b) = if op.commutative() && a > b { (b, a) } else { (a, b) };
        // algebraic simplifications
        match op {
            Add if self.is_const(a, 0.0) => return Ok(b),
            Add if self.is_const(b, 0.0) => return Ok(a),
            Sub if self.is_const(b, 0.0) => return Ok(a),
            Sub if a == b => return self.constant(0.0, kind),
            Mul if self.is_const(a, 1.0) => return Ok(b),
            Mul if self.is_const(b, 1.0) => return Ok(a),
            Mul if self.is_const(a, 0.0) || self.is_const(b, 0.0) => return self.constant(0.0, kind),
            Div | FloorDiv if self.is_const(b, 1.0) => return Ok(a),
            Max | Min | And | Or if a == b => return Ok(a),
            Xor if a == b => return self.constant(0.0, Kind::Bool),
            And if self.is_not_of(a, b) => return self.constant(0.0, Kind::Bool),
            Or | Xor if self.is_not_of(a, b) => return self.constant(1.0, Kind::Bool),
            And if self.is_const(a, 0.0) || self.is_const(b, 0.0) => return self.constant(0.0, Kind::Bool),
            And if self.is_const(a, 1.0) => return Ok(b),
            And if self.is_const(b, 1.0) => return Ok(a),
            Or if self.is_const(a, 1.0) || self.is_const(b, 1.0) => return self.constant(1.0, Kind::Bool),
            Or if self.is_const(a, 0.0) => return Ok(b),
            Or if self.is_const(b, 0.0) => return Ok(a),
            Xor if self.is_const(a, 0.0) => return Ok(b),
            Xor if self.is_const(b, 0.0) => return Ok(a),
            Xor if self.is_const(a, 1.0) => return self.un(UnOp::Not, b),
            Xor if self.is_const(b, 1.0) => return self.un(UnOp::Not, a),
            _ => {}
        }
        let ia = (ta.lo, ta.hi);
        let ib = (tb.lo, tb.hi);
        let (lo, hi) = match op {
            Add => (ta.lo + tb.lo, ta.hi + tb.hi),
            Sub => (ta.lo - tb.hi, ta.hi - tb.lo),
            Mul => {
                if a == b {
                    let (l, h) = imul(ia, ib);
                    (if ta.lo <= 0.0 && ta.hi >= 0.0 { 0.0 } else { l.max(0.0) }, h)
                } else {
                    imul(ia, ib)
                }
            }
            Div => idiv(ia, ib, tb.kind.integral()),
            FloorDiv => {
                let (l, h) = idiv(ia, ib, tb.kind.integral());
                (l.floor(), h.floor())
            }
            Mod => imod(ia, ib, kind.integral()),
            Max => (ta.lo.max(tb.lo), ta.hi.max(tb.hi)),
            Min => (ta.lo.min(tb.lo), ta.hi.min(tb.hi)),
            And | Or | Xor => (0.0, 1.0),
        };
        Ok(self.intern(Op::Bin(op, a, b), Ty::new(kind, lo, hi)))
    }

    pub fn un(&mut self, op: UnOp, a: NodeId) -> Result<NodeId> {
        use UnOp::*;
        self.check(a)?;
        if op == Not {
            self.require_bool(a, "not")?;
        }
        let ta = self.ty(a);
        let kind = match op {
            Not => Kind::Bool,
            Floor | Ceil | Trunc => Kind::Int,
            Abs | Neg if ta.kind == Kind::Bool => Kind::Int,
            _ => ta.kind,
        };
        if let Some(x) = self.const_of(a) {
            return self.constant(eval_un(op, x), kind);
        }
        if let Op::Un(inner, x) = self.nodes[a].op {
            if inner == op && matches!(op, Neg | Not) {
                return Ok(x);
            }
        }
        match op {
            Abs | Relu if ta.lo >= 0.0 => return Ok(a),
            Relu if ta.hi <= 0.0 => return self.constant(0.0, kind),
            Floor | Ceil | Trunc if ta.kind.integral() => return Ok(a),
            _ => {}
        }
        let (lo, hi) = match op {
            Neg => (-ta.hi, -ta.lo),
            Abs => {
                if ta.hi <= 0.0 {
                    (-ta.hi, -ta.lo)
                } else {
                    (0.0, (-ta.lo).max(ta.hi))
                }
            }
            Relu => (ta.lo.max(0.0), ta.hi.max(0.0)),
            Not => (1.0 - ta.hi, 1.0 - ta.lo),
            Floor => (ta.lo.floor(), ta.hi.floor()),
            Ceil => (ta.lo.ceil(), ta.hi.ceil()),
            Trunc => (ta.lo.trunc(), ta.hi.trunc()),
        };
        Ok(self.intern(Op::Un(op, a), Ty::new(kind, lo, hi)))
    }

    pub fn cmp(&mut self, op: CmpOp, a: NodeId, b: NodeId) -> Result<NodeId> {
        use CmpOp::*;
        self.check(a)?;
        self.check(b)?;
        // canonicalise: a > b == b < a, a >= b == b <= a
        let (op, a, b) = match op {
            Gt => (Lt, b, a),
            Ge => (Le, b, a),
            Eq | Ne if a > b => (op, b, a),
            _ => (op, a, b),
        };
        if let (Some(x), Some(y)) = (self.const_of(a), self.const_of(b)) {
            return self.constant(eval_cmp(op, x, y), Kind::Bool);
        }
        if a == b {
            let v = matches!(op, Le | Ge | Eq);
            return self.constant(b2f(v), Kind::Bool);
        }
        let (ta, tb) = (self.ty(a), self.ty(b));
        let known = match op {
            Lt if ta.hi < tb.lo => Some(true),
            Lt if ta.lo >= tb.hi => Some(false),
            Le if ta.hi <= tb.lo => Some(true),
            Le if ta.lo > tb.hi => Some(false),
            Eq | Ne if ta.hi < tb.lo || tb.hi < ta.lo => Some(op == Ne),
            _ => None,
        };
        if let Some(v) = known {
            return self.constant(b2f(v), Kind::Bool);
        }
        Ok(self.intern(Op::Cmp(op, a, b), Ty::new(Kind::Bool, 0.0, 1.0)))
    }

    pub fn select(&mut self, c: NodeId, a: NodeId, b: NodeId) -> Result<NodeId> {
        self.check(c)?;
        self.check(a)?;
        self.check(b)?;
        self.require_bool(c, "select condition")?;
        if let Some(v) = self.const_of(c) {
            return Ok(if v != 0.0 { a } else { b });
        }
        if a == b {
            return Ok(a);
        }
        let (ta, tb) = (self.ty(a), self.ty(b));
        if ta.kind == Kind::Bool && tb.kind == Kind::Bool {
            if a == c {
                return self.bin(BinOp::Or, c, b);
            }
            if b == c {
                return self.bin(BinOp::And, c, a);
            }
            match (self.const_of(a), self.const_of(b)) {
                (Some(x), Some(y)) if x == 1.0 && y == 0.0 => return Ok(c),
                (Some(x), Some(y)) if x == 0.0 && y == 1.0 => return self.un(UnOp::Not, c),
                (Some(x), _) if x == 0.0 => {
                    let nc = self.un(UnOp::Not, c)?;
                    return self.bin(BinOp::And, nc, b);
                }
                (Some(x), _) if x == 1.0 => return self.bin(BinOp::Or, c, b),
                (_, Some(y)) if y == 0.0 => return self.bin(BinOp::And, c, a),
                (_, Some(y)) if y == 1.0 => {
                    let nc = self.un(UnOp::Not, c)?;
                    return self.bin(BinOp::Or, nc, a);
                }
                _ => {}
            }
        }
        // select(c, a, select(c, x, y)) == select(c, a, y)   (and symmetric)
        let (a, b) = {
            let a2 = match self.nodes[a].op {
                Op::Select(c2, x, _) if c2 == c => x,
                _ => a,
            };
            let b2 = match self.nodes[b].op {
                Op::Select(c2, _, y) if c2 == c => y,
                _ => b,
            };
            (a2, b2)
        };
        if a == b {
            return Ok(a);
        }
        let (ta, tb) = (self.ty(a), self.ty(b));
        let kind = if ta.kind == Kind::Bool && tb.kind == Kind::Bool {
            Kind::Bool
        } else {
            Kind::arith(ta.kind, tb.kind)
        };
        let ty = Ty::new(kind, ta.lo.min(tb.lo), ta.hi.max(tb.hi));
        Ok(self.intern(Op::Select(c, a, b), ty))
    }

    pub fn output(&mut self, id: NodeId) -> Result<()> {
        self.check(id)?;
        self.outputs.push(id);
        Ok(())
    }

    // -------------------------------------------------------------- eval

    /// Evaluate every node. `margin`: if Some(eps), also report whether some
    /// real-valued discontinuity was evaluated within eps of its threshold
    /// (where margin-based gadgets are allowed to deviate).
    pub fn eval_all(&self, inputs: &[f64], margin: Option<f64>) -> Result<(Vec<f64>, bool)> {
        if inputs.len() != self.inputs.len() {
            return err(
                ErrorKind::Invalid,
                format!("expected {} inputs, got {}", self.inputs.len(), inputs.len()),
            );
        }
        let mut v = vec![0.0; self.nodes.len()];
        let mut near = false;
        for (i, n) in self.nodes.iter().enumerate() {
            v[i] = match n.op {
                Op::Input(k) => inputs[k],
                Op::Const(c) => c,
                Op::Bin(op, a, b) => {
                    if let Some(eps) = margin {
                        if matches!(op, BinOp::FloorDiv | BinOp::Mod) && n.ty.kind == Kind::Real && v[b] != 0.0 {
                            let q = v[a] / v[b];
                            near |= (q - q.round()).abs() < eps / v[b].abs().max(1e-300) + 1e-12;
                        }
                    }
                    eval_bin(op, v[a], v[b])
                }
                Op::Un(op, a) => {
                    if let Some(eps) = margin {
                        if matches!(op, UnOp::Floor | UnOp::Ceil | UnOp::Trunc)
                            && self.nodes[a].ty.kind == Kind::Real
                        {
                            near |= (v[a] - v[a].round()).abs() < eps;
                        }
                    }
                    eval_un(op, v[a])
                }
                Op::Cmp(op, a, b) => {
                    if let Some(eps) = margin {
                        let k = Kind::arith(self.nodes[a].ty.kind, self.nodes[b].ty.kind);
                        if k == Kind::Real {
                            near |= (v[a] - v[b]).abs() < eps;
                        }
                    }
                    eval_cmp(op, v[a], v[b])
                }
                Op::Select(c, a, b) => {
                    if v[c] != 0.0 {
                        v[a]
                    } else {
                        v[b]
                    }
                }
                Op::Table(t) => {
                    let t = &self.tables[t];
                    let xs: Vec<f64> = t.args.iter().map(|&a| v[a]).collect();
                    t.get(&xs)
                }
                Op::Gather(k) => {
                    let g = &self.gathers[k];
                    g.pick(v[g.index]).map_or(0.0, |p| v[g.elems[p]])
                }
            };
        }
        Ok((v, near))
    }

    /// Value of node `id` given the values of its arguments.
    pub fn eval_node(&self, id: NodeId, val: &dyn Fn(NodeId) -> f64) -> f64 {
        match self.nodes[id].op {
            Op::Input(_) => f64::NAN,
            Op::Const(c) => c,
            Op::Bin(op, a, b) => eval_bin(op, val(a), val(b)),
            Op::Un(op, a) => eval_un(op, val(a)),
            Op::Cmp(op, a, b) => eval_cmp(op, val(a), val(b)),
            Op::Select(c, a, b) => {
                if val(c) != 0.0 {
                    val(a)
                } else {
                    val(b)
                }
            }
            Op::Table(t) => {
                let t = &self.tables[t];
                let xs: Vec<f64> = t.args.iter().map(|&a| val(a)).collect();
                t.get(&xs)
            }
            Op::Gather(k) => {
                let g = &self.gathers[k];
                g.pick(val(g.index)).map_or(0.0, |p| val(g.elems[p]))
            }
        }
    }

    pub fn eval(&self, inputs: &[f64]) -> Result<Vec<f64>> {
        let (v, _) = self.eval_all(inputs, None)?;
        Ok(self.outputs.iter().map(|&o| v[o]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divmod_matches_python() {
        assert_eq!(py_divmod(7.0, 2.0), (3.0, 1.0));
        assert_eq!(py_divmod(-7.0, 2.0), (-4.0, 1.0));
        assert_eq!(py_divmod(7.0, -2.0), (-4.0, -1.0));
        assert_eq!(py_divmod(-7.0, -2.0), (3.0, -1.0));
        assert_eq!(py_divmod(5.0, 0.0), (0.0, 0.0));
    }

    #[test]
    fn folding_and_cse() {
        let mut g = Graph::new();
        let x = g.input("x", Kind::Int, 0.0, 10.0).unwrap();
        let c2 = g.constant(2.0, Kind::Int).unwrap();
        let a = g.bin(BinOp::Add, x, c2).unwrap();
        let b = g.bin(BinOp::Add, c2, x).unwrap();
        assert_eq!(a, b);
        assert_eq!(g.ty(a).lo, 2.0);
        // x + 2 > -1 is always true
        let m1 = g.constant(-1.0, Kind::Int).unwrap();
        let t = g.cmp(CmpOp::Gt, a, m1).unwrap();
        assert_eq!(g.const_of(t), Some(1.0));
    }
}

// ------------------------------------------------------- output encodings

/// How the program's outputs are presented by the network. All encodings
/// other than `Raw` treat the flat output vector as scores of one decision
/// and keep only their order, which gives a small, bounded output range
/// (trainability: decision margins stay O(1) relative to the outputs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputEnc {
    Raw,
    /// rank_i = #{j != i : s_i > s_j}, in 0..n-1 (ties share a rank)
    Rank,
    /// onehot_i = [s_i >= s_j for all j] (all maxima are 1 under ties)
    OneHot,
}

impl OutputEnc {
    pub fn parse(s: &str) -> Result<OutputEnc> {
        match s {
            "raw" => Ok(OutputEnc::Raw),
            "rank" => Ok(OutputEnc::Rank),
            "onehot" | "argmax" => Ok(OutputEnc::OneHot),
            _ => err(ErrorKind::Invalid, format!("unknown output encoding '{s}' (raw, rank, onehot)")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            OutputEnc::Raw => "raw",
            OutputEnc::Rank => "rank",
            OutputEnc::OneHot => "onehot",
        }
    }
}

impl Graph {
    /// A copy of the graph whose outputs are the encoded outputs. With
    /// `lex`, score comparisons are lexicographic over tiers where possible
    /// (see [`crate::lex`]): same values, much smaller internal ranges.
    pub fn encode_outputs(&self, enc: OutputEnc, lex: bool) -> Result<Graph> {
        self.encode_outputs_grouped(enc, lex, &[], None)
    }

    /// Like [`Graph::encode_outputs`], but the flat outputs are split into
    /// consecutive `groups` (sizes; empty = one group) and each group is
    /// encoded on its own (e.g. the move scores of several heuristics).
    /// With `weights` (one per group, groups of equal size n) the outputs
    /// are the n combinations `out_j = sum_g weights[g] * enc_g[j]`.
    pub fn encode_outputs_grouped(&self, enc: OutputEnc, lex: bool, groups: &[usize], weights: Option<&[f64]>) -> Result<Graph> {
        let mut g = self.clone();
        let all = std::mem::take(&mut g.outputs);
        let sizes: Vec<usize> = if groups.is_empty() { vec![all.len()] } else { groups.to_vec() };
        if sizes.iter().sum::<usize>() != all.len() {
            return err(
                ErrorKind::Invalid,
                format!("output groups {:?} do not add up to the {} outputs", sizes, all.len()),
            );
        }
        let mut encoded = vec![];
        let mut at = 0;
        for &k in &sizes {
            let s = &all[at..at + k];
            at += k;
            encoded.push(if enc == OutputEnc::Raw { s.to_vec() } else { encode_group(&mut g, s, enc, lex)? });
        }
        match weights {
            None => {
                for o in encoded.into_iter().flatten() {
                    g.output(o)?;
                }
            }
            Some(w) => {
                if w.len() != sizes.len() {
                    return err(ErrorKind::Invalid, format!("{} weights for {} output groups", w.len(), sizes.len()));
                }
                let n = sizes[0];
                if sizes.iter().any(|&k| k != n) {
                    return err(ErrorKind::Invalid, format!("weighted groups must have equal sizes, got {sizes:?}"));
                }
                for j in 0..n {
                    let mut acc: Option<NodeId> = None;
                    for (e, &wg) in encoded.iter().zip(w) {
                        if wg == 0.0 {
                            continue;
                        }
                        let kind = if wg.fract() == 0.0 { Kind::Int } else { Kind::Real };
                        let c = g.constant(wg, kind)?;
                        let t = g.bin(BinOp::Mul, c, e[j])?;
                        acc = Some(match acc {
                            None => t,
                            Some(a) => g.bin(BinOp::Add, a, t)?,
                        });
                    }
                    let o = match acc {
                        Some(a) => a,
                        None => g.constant(0.0, Kind::Int)?,
                    };
                    g.output(o)?;
                }
            }
        }
        Ok(g)
    }
}

/// rank / onehot encoding of one group of scores (new nodes in `g`).
fn encode_group(g: &mut Graph, s: &[NodeId], enc: OutputEnc, lex: bool) -> Result<Vec<NodeId>> {
    if s.len() < 2 {
        return err(ErrorKind::Invalid, format!("output='{}' needs at least 2 outputs (scores) per group", enc.name()));
    }
    let ls = if lex { Some(crate::lex::prepare(g, s)?) } else { None };
    let n = s.len();
    let mut gt = vec![vec![None; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let l = match &ls {
                    Some(ls) => crate::lex::greater(g, ls, i, j)?,
                    None => None,
                };
                gt[i][j] = Some(match l {
                    Some(x) => x,
                    None => g.cmp(CmpOp::Gt, s[i], s[j])?,
                });
            }
        }
    }
    let zero = g.constant(0.0, Kind::Int)?;
    let mut out = vec![];
    for i in 0..n {
        let mut acc = zero;
        for j in 0..n {
            if i == j {
                continue;
            }
            // rank counts the scores below s_i, onehot the scores above it
            let c = match enc {
                OutputEnc::Rank => gt[i][j].unwrap(),
                _ => gt[j][i].unwrap(),
            };
            acc = g.bin(BinOp::Add, acc, c)?;
        }
        out.push(match enc {
            OutputEnc::Rank => acc,
            _ => g.cmp(CmpOp::Eq, acc, zero)?,
        });
    }
    Ok(out)
}

