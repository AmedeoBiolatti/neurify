use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// The operation has no exact realisation in the requested architecture.
    NotExact,
    /// A gadget needs finite bounds on an operand (declare input domains).
    NeedsBounds,
    /// A lookup / staircase gadget would exceed `max_table` units.
    TableTooLarge,
    /// Malformed graph or options.
    Invalid,
}

#[derive(Debug, Clone)]
pub struct CompileError {
    pub kind: ErrorKind,
    pub msg: String,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.msg)
    }
}

impl std::error::Error for CompileError {}

pub type Result<T> = std::result::Result<T, CompileError>;

pub fn err<T>(kind: ErrorKind, msg: impl Into<String>) -> Result<T> {
    Err(CompileError { kind, msg: msg.into() })
}
