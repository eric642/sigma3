//! Errors raised by sigma. Currently only used by builder validation.

#[derive(Debug)]
pub enum SigmaError {
    /// Returned when a builder fails to produce a valid value.
    InvalidArgument(String),
}

impl std::fmt::Display for SigmaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "invalid args: {msg}"),
        }
    }
}

impl std::error::Error for SigmaError {}

impl From<derive_builder::UninitializedFieldError> for SigmaError {
    fn from(value: derive_builder::UninitializedFieldError) -> Self {
        Self::InvalidArgument(value.to_string())
    }
}

impl From<String> for SigmaError {
    fn from(value: String) -> Self {
        Self::InvalidArgument(value)
    }
}
