//! Runtime values of the symbolic interpreter.
//!
//! Values are either concrete Python data (evaluated at compile time:
//! partial evaluation) or symbolic scalars (`Sym`, a node of the IR graph).
//! Containers have fixed structure; dict keys are always concrete
//! (pytree-like, as in JAX).

use neurify_core::NodeId;
use rustpython_parser::ast;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    None,
    Int(i64),
    Str(Rc<str>),
    Tuple(Vec<Key>),
}

impl Key {
    pub fn to_value(&self) -> Value {
        match self {
            Key::None => Value::None,
            Key::Int(i) => Value::Int(*i),
            Key::Str(s) => Value::Str(s.clone()),
            Key::Tuple(t) => Value::Tuple(Rc::new(t.iter().map(|k| k.to_value()).collect())),
        }
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Key::None => write!(f, "None"),
            Key::Int(i) => write!(f, "{i}"),
            Key::Str(s) => write!(f, "'{s}'"),
            Key::Tuple(t) => {
                write!(f, "(")?;
                for (i, k) in t.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Insertion-ordered dict with concrete keys.
#[derive(Clone, Debug, Default)]
pub struct Dict {
    pub items: Vec<(Key, Value)>,
}

impl Dict {
    pub fn get(&self, k: &Key) -> Option<&Value> {
        self.items.iter().find(|(x, _)| x == k).map(|(_, v)| v)
    }
    pub fn set(&mut self, k: Key, v: Value) {
        match self.items.iter_mut().find(|(x, _)| *x == k) {
            Some(slot) => slot.1 = v,
            None => self.items.push((k, v)),
        }
    }
    pub fn remove(&mut self, k: &Key) -> Option<Value> {
        let i = self.items.iter().position(|(x, _)| x == k)?;
        Some(self.items.remove(i).1)
    }
}

/// Where unbound names are looked up.
#[derive(Clone, Debug)]
pub enum Ctx {
    /// a Python function known to the host (globals / closure cells)
    Host(u64),
    /// a function defined while interpreting: captured bindings + outer ctx
    Closure(Rc<RefCell<HashMap<Rc<str>, Value>>>, Box<Ctx>),
}

/// Source location info for error messages.
#[derive(Debug)]
pub struct Src {
    pub file: String,
    pub first_line: usize,
    pub line_starts: Vec<usize>,
}

impl Src {
    pub fn new(file: &str, first_line: usize, text: &str) -> Src {
        let mut line_starts = vec![0];
        for (i, b) in text.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i + 1);
            }
        }
        Src { file: file.to_string(), first_line, line_starts }
    }
    pub fn line(&self, offset: usize) -> usize {
        let l = match self.line_starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        self.first_line + l
    }
}

#[derive(Debug)]
pub enum FuncBody {
    Def(ast::StmtFunctionDef),
    Lambda(ast::ExprLambda),
}

#[derive(Debug)]
pub struct Func {
    pub name: String,
    pub body: Rc<FuncBody>,
    pub src: Rc<Src>,
    pub ctx: Ctx,
    /// host handle of the Python function object (for tabulation fallback)
    pub handle: Option<u64>,
}

impl Func {
    pub fn args(&self) -> &ast::Arguments {
        match &*self.body {
            FuncBody::Def(d) => &d.args,
            FuncBody::Lambda(l) => &l.args,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<str>),
    /// symbolic scalar (IR node)
    Sym(NodeId),
    List(Rc<RefCell<Vec<Value>>>),
    Tuple(Rc<Vec<Value>>),
    Dict(Rc<RefCell<Dict>>),
    Func(Rc<Func>),
    /// interpreter builtin, by name ("abs", "math.floor", "neurify.where", ...)
    Builtin(Rc<str>),
    /// opaque host object (module, callable, ...)
    Object(u64),
    /// bound method of a list / dict
    Method(Box<Value>, Rc<str>),
    Range(i64, i64, i64),
    /// range with a symbolic bound: iterations are guarded by `i < stop`
    SymRange(Box<Value>, Box<Value>, i64),
    /// "no value on this path" (unassigned name, or a path that raised)
    Undef,
}

impl Value {
    pub fn list(v: Vec<Value>) -> Value {
        Value::List(Rc::new(RefCell::new(v)))
    }
    pub fn tuple(v: Vec<Value>) -> Value {
        Value::Tuple(Rc::new(v))
    }
    pub fn dict(d: Dict) -> Value {
        Value::Dict(Rc::new(RefCell::new(d)))
    }
    pub fn str(s: &str) -> Value {
        Value::Str(Rc::from(s))
    }
    pub fn is_scalar(&self) -> bool {
        matches!(self, Value::Bool(_) | Value::Int(_) | Value::Float(_) | Value::Sym(_))
    }
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::None => "None",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::Sym(_) => "symbolic value",
            Value::List(_) => "list",
            Value::Tuple(_) => "tuple",
            Value::Dict(_) => "dict",
            Value::Func(_) => "function",
            Value::Builtin(_) => "builtin",
            Value::Object(_) => "object",
            Value::Method(..) => "method",
            Value::Range(..) | Value::SymRange(..) => "range",
            Value::Undef => "undefined",
        }
    }
    /// Same container object?
    pub fn same_object(&self, o: &Value) -> bool {
        match (self, o) {
            (Value::List(a), Value::List(b)) => Rc::ptr_eq(a, b),
            (Value::Dict(a), Value::Dict(b)) => Rc::ptr_eq(a, b),
            (Value::Tuple(a), Value::Tuple(b)) => Rc::ptr_eq(a, b),
            (Value::Func(a), Value::Func(b)) => Rc::ptr_eq(a, b),
            (Value::Object(a), Value::Object(b)) => a == b,
            (Value::Builtin(a), Value::Builtin(b)) => a == b,
            (Value::Sym(a), Value::Sym(b)) => a == b,
            (Value::None, Value::None) | (Value::Undef, Value::Undef) => true,
            _ => false,
        }
    }
}

/// Input domain specification for one function parameter.
#[derive(Clone, Debug)]
pub enum InSpec {
    Real(f64, f64),
    Int(f64, f64),
    Bool,
    List(Vec<InSpec>),
    Tuple(Vec<InSpec>),
    Dict(Vec<(Key, InSpec)>),
    /// sequence of scalars living on transformer tokens (a list otherwise)
    Seq(Vec<InSpec>),
    /// a concrete value (static argument, like jax.jit static_argnums)
    Static(Value),
}

/// Structure of the traced output (leaves are graph outputs, in order).
#[derive(Clone, Debug)]
pub enum OutSpec {
    /// kind name of the leaf ("bool" / "int" / "real")
    Leaf(&'static str),
    Tuple(Vec<OutSpec>),
    List(Vec<OutSpec>),
    Dict(Vec<(Key, OutSpec)>),
    None,
}

/// What the host returns when resolving a name or attribute.
pub enum Resolved {
    Value(Value),
    Function { source: String, name: String, file: String, first_line: usize, ctx: u64, handle: Option<u64> },
    Builtin(String),
    Object(u64),
    Missing,
}

/// The embedding environment (the Python process, or a test stub).
pub trait Host {
    fn resolve(&mut self, ctx: u64, name: &str) -> Result<Resolved, String>;
    fn getattr(&mut self, obj: u64, attr: &str) -> Result<Resolved, String>;
    /// Call an opaque object with concrete arguments.
    fn call(&mut self, obj: u64, args: &[Value]) -> Result<Value, String>;
}

/// A host that knows nothing (pure-Rust use and tests).
pub struct NoHost;

impl Host for NoHost {
    fn resolve(&mut self, _: u64, _: &str) -> Result<Resolved, String> {
        Ok(Resolved::Missing)
    }
    fn getattr(&mut self, _: u64, attr: &str) -> Result<Resolved, String> {
        Err(format!("no attribute '{attr}'"))
    }
    fn call(&mut self, _: u64, _: &[Value]) -> Result<Value, String> {
        Err("no host".into())
    }
}
