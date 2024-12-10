//BERT encoding
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::Deref;

#[derive(Debug, Clone, Copy)]
pub struct ZeroPoint {
    pub value: i32,
    pub scale: f32,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TokenJoins<F, O> {
    pub actor: usize,
    pub output: usize,
    pub _phantom: PhantomData<(F, O)>,
}

impl<F, O> TokenJoins<F, O> {
    pub fn new(actor: usize, output: usize) -> Self {
        TokenJoins { actor, output, _phantom: PhantomData }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub actor: usize,
    pub memory: usize,
}

