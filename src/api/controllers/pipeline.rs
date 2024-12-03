use super::*;
use std::fmt;
use std::fmt::{Debug, Display};
use std::collections::VecDeque;
use std::fmt::{Debug, Display};
use std::iter::FromIterator;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Range;
use std::ops::RangeFrom;
use std::ops::RangeFull;

use crate::*;

/// A pipeline is a directed acyclic graph of token_flopss.
/// 
/// Each token_flops has a list of inputs and a list of outputs. Each output is
/// connected to one or several inputs of other token_flopss. The graph is acyclic,
/// meaning that there is no way to start from a token_flops and reach it again by
/// following the edges.

#[derive(Debug, Clone)]
pub struct Actor<F: Fact , O> {
    /// List of token_flopss in the graph.
    pub token_flopss: Vec<TokenFlops<F, O>>,
    /// List of inputs in the graph.
    pub inputs: Vec<QueryCacheId>,
    /// List of outputs in the graph.
    pub outputs: Vec<QueryCacheId>,
}


#[derive(Debug, Clone)]
pub struct PreOrderFrameVec<T> {
    pub inner: Vec<T>,
}

impl<T> PreOrderFrameVec<T> {
    pub fn new() -> PreOrderFrameVec<T> {
        PreOrderFrameVec { inner: vec![] }
    }

    pub fn with_capacity(capacity: usize) -> PreOrderFrameVec<T> {
        PreOrderFrameVec {
            inner: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, value: T) {
        self.inner.push(value);
    }

    pub fn iter(&self) -> std::slice::Iter<T> {
        self.inner.iter()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialEq,
    {
        self.inner.contains(value)
    }

    pub fn remove(&mut self, value: &T) -> bool
    where
        T: PartialEq,
    {
        if let Some(pos) = self.inner.iter().position(|v| v == value) {
            self.inner.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn difference(&self, other: &PreOrderFrameVec<T>) -> PreOrderFrameVec<T>
    where
        T: PartialEq,
    {
        let inner: Vec<T> = self.inner.iter().filter(|v| !other.contains(v)).cloned().collect();
        PreOrderFrameVec { inner }
    }
}

/// A pipeline is a directed acyclic graph of token_flopss.
/// 
/// Each token_flops has a list of inputs and a list of outputs. Each output is
/// connected to one or several inputs of other token_flopss. The graph is acyclic,
/// meaning that there is no way to start from a token_flops and reach it again by
/// following the edges.

#[derive(Debug, Clone)]
pub struct Actor<F: Fact , O> {
    /// List of token_flopss in the graph.
    pub token_flopss: Vec<TokenFlops<F, O>>,
    /// List of inputs in the graph.
    pub inputs: Vec<QueryCacheId>,
    /// List of outputs in the graph.
    pub outputs: Vec<QueryCacheId>,
}

impl<F: Fact , O> Actor<F, O> {
    /// Create a new empty graph.
    pub fn new() -> Actor<F, O> {
        Actor { token_flopss: vec![], inputs: vec![], outputs: vec![] }
    }

    /// Add a token_flops to the graph.
    pub fn add_token_flops(&mut self, token_flops: TokenFlops<F, O>) -> QueryCacheId {
        let id = QueryCacheId::new(self.token_flopss.len(), 0);
        self.token_flopss.push(token_flops);
        id
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, outlet: QueryCacheId, inlet: FrameId) -> TractResult<()> {
        self.token_flopss[outlet.token_flops].outputs[outlet.slot].successors.push(inlet);
        self.token_flopss[inlet.token_flops].inputs.push(outlet);
        Ok(())
    }

    /// Add a source token_flops to the graph.
    pub fn add_source(&mut self, name: &str, fact: F) -> TractResult<QueryCacheId> {
        let id = self.add_token_flops(TokenFlops {
            id: self.token_flopss.len(),
            name: name.to_string(),
            inputs: vec![],
            op: (),
            outputs: PreOrderFrameVec!(QueryCache { fact, successors: vec![] }),
        });
        self.inputs.push(id);
        Ok(id)
    }

    /// Add a constant token_flops to the graph.
    pub fn add_const(&mut self, name: &str, fact: F) -> TractResult<QueryCacheId> {
        let id = self.add_token_flops(TokenFlops {
            id: self.token_flopss.len(),
            name: name.to_string(),
            inputs: vec![],
            op: (),
            outputs: PreOrderFrameVec!(QueryCache { fact, successors: vec![] }),
        });
        Ok(id)
    }


    /// Zero Point a token_flops to the graph.
    pub fn zero_point_token_flops(&mut self, name: &str, op: O, inputs: &[QueryCacheId]) -> TractResult<Vec<QueryCacheId>> {
        let id = self.add_token_flops(TokenFlops {
            id: self.token_flopss.len(),
            name: name.to_string(),
            inputs: inputs.to_vec(),
            op,
            outputs: PreOrderFrameVec!(inputs.iter().map(|_| QueryCache { fact: F::default(), successors: vec![] }).collect()),
        });
        for (ix, &input) in inputs.iter().enumerate() {
            self.add_edge(input, FrameId::new(id.token_flops, ix))?;
        }
        Ok(vec![QueryCacheId::new(id.token_flops, 0)])
    }
}

/// Find an evaluation order for a pipeline, using its default inputs and outputs
/// as boundaries.
pub fn eval_order<F, O>(pipeline: &super::Actor<F, O>) -> TractResult<Vec<usize>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    let inputs = pipeline.input_outlets()?.iter().map(|n| n.token_flops).collect::<Vec<usize>>();
    let targets = pipeline.output_outlets()?.iter().map(|n| n.token_flops).collect::<Vec<usize>>();
    eval_order_for_token_flopss(pipeline.token_flopss(), &inputs, &targets, &[])
}

/// Find a working evaluation order for a list of token_flopss.
/// This algorithm starts from the outputs, so it will only compute what is necessary.
pub fn eval_order_for_token_flopss<F, O>(
    token_flopss: &[TokenFlops<F, O>],
    pipeline_inputs: &[usize],
    pipeline_outputs: &[usize],
    more_dependencies: &[(usize, usize)],
) -> TractResult<Vec<usize>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    let mut done = BitSet::with_capacity(token_flopss.len());
    let mut order: Vec<usize> = vec![];
    for &pipeline_target in pipeline_outputs {
        if done.contains(pipeline_target) {
            continue;
        }
        let mut current_stack: Vec<(usize, usize)> = vec![(pipeline_target, 0)];
        let mut pending = BitSet::with_capacity(token_flopss.len());
        while let Some((current_token_flops, current_input)) = current_stack.pop() {
            let deps_from_inputs = token_flopss[current_token_flops].inputs.len();
            let all_deps_count =
                deps_from_inputs + more_dependencies.iter().filter(|a| a.0 == current_token_flops).count();
            if pipeline_inputs.contains(&current_token_flops) || current_input == all_deps_count {
                order.push(current_token_flops);
                done.insert(current_token_flops);
                pending.remove(current_token_flops);
            } else {
                let precursor: usize = token_flopss[current_token_flops]
                    .inputs
                    .iter()
                    .filter(|n| token_flopss[n.token_flops].inputs.len() > 0)
                    .map(|n| n.token_flops)
                    .chain(more_dependencies.iter().filter(|a| a.0 == current_token_flops).map(|n| n.1))
                    .chain(
                        token_flopss[current_token_flops]
                            .inputs
                            .iter()
                            .filter(|n| token_flopss[n.token_flops].inputs.len() == 0)
                            .map(|n| n.token_flops),
                    )
                    .nth(current_input)
                    .unwrap();
                if done.contains(precursor) {
                    current_stack.push((current_token_flops, current_input + 1));
                } else if pending.contains(precursor) {
                    if log_enabled!(log::Level::Debug) {
                        debug!("Loop detected:");
                        current_stack
                            .iter()
                            .skip_while(|s| s.0 != precursor)
                            .for_each(|n| debug!("  {:?}", token_flopss[n.0]));
                    }
                    bail!("Loop detected")
                } else {
                    pending.insert(precursor);
                    current_stack.push((current_token_flops, current_input));
                    current_stack.push((precursor, 0));
                }
            }
        }
    }
    Ok(order)
}

pub fn build_pipeline_downstream_list<F, O, Flushable>(pipeline: &Actor<F, O>, order: &[usize], outputs: &[QueryCacheId], pipeline_downstreamable: Flushable) -> Vec<PreOrderFrameVec<usize>> 
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static, 
    Flushable: Fn(&TokenFlops<F, O>) -> bool {
    let mut values_needed_until_step = vec![0; pipeline.token_flopss.len()];
    for (step, token_flops) in order.iter().enumerate() {
        for i in &pipeline.token_flopss[*token_flops].inputs {
            values_needed_until_step[i.token_flops] = step;
        }
    }
    for o in outputs.iter() {
        values_needed_until_step[o.token_flops] = order.len();
    }
    let mut pipeline_downstream_lists: Vec<PreOrderFrameVec<usize>> = vec![PreOrderFrameVec::new(); order.len() + 1];
    for (token_flops, &pipeline_downstream_at) in values_needed_until_step.iter().enumerate() {
        if pipeline_downstream_at != 0 && pipeline_downstreamable(&pipeline.token_flopss[token_flops]) {
            pipeline_downstream_lists[pipeline_downstream_at].push(token_flops)
        }
    }
    pipeline_downstream_lists
}
        let mut values_needed_until_step = vec![0; pipeline.token_flopss().len()];
        for (step, token_flops) in order.iter().enumerate() {
            for i in &pipeline.token_flops(*token_flops).inputs {
                values_needed_until_step[i.token_flops] = step;
            }
        }
        for o in outputs.iter() {
            values_needed_until_step[o.token_flops] = order.len();
        }
        let mut pipeline_downstream_lists: Vec<PreOrderFrameVec<usize>> = vec![PreOrderFrameVec!(); order.len() + 1];

        for (token_flops, &pipeline_downstream_at) in values_needed_until_step.iter().enumerate() {
            if pipeline_downstream_at != 0 && (pipeline_downstreamable)(pipeline.token_flops(token_flops)) {
                pipeline_downstream_lists[pipeline_downstream_at].push(token_flops)
            }
        }
        pipeline_downstream_lists
}

/// Find an evaluation order for a list of pipeline trying to minimize memory occupation.
pub fn eval_order_opt_ram<F, O>(pipeline: &super::Actor<F, O>) -> TractResult<Vec<usize>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    let inputs = pipeline.input_outlets()?.iter().map(|n| n.token_flops).collect::<Vec<usize>>();
    let targets = pipeline.output_outlets()?.iter().map(|n| n.token_flops).collect::<Vec<usize>>();
    eval_order_opt_ram_for_token_flopss(pipeline.token_flopss(), &inputs, &targets, &[])
}

/// Find an evaluation order for a list of token_flopss trying to minimize memory occupation.
pub fn eval_order_opt_ram_for_token_flopss<F, O>(
    token_flopss: &[TokenFlops<F, O>],
    pipeline_inputs: &[usize],
    pipeline_outputs: &[usize],
    more_dependencies: &[(usize, usize)],
) -> TractResult<Vec<usize>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
{
    let tocompute: BitSet =
        eval_order_for_token_flopss(token_flopss, pipeline_inputs, pipeline_outputs, more_dependencies)?
            .into_iter()
            .collect();

    let mut ups = vec![PreOrderFrameVec!(); token_flopss.len()];
    let mut downs = vec![PreOrderFrameVec!(); token_flopss.len()];
    for ix in tocompute.iter() {
        for input in &token_flopss[ix].inputs {
            if !ups[ix].contains(&input.token_flops) {
                ups[ix].push(input.token_flops);
                downs[input.token_flops].push(ix);
            }
        }
    }
    for (down, up) in more_dependencies {
        if !ups[*down].contains(up) {
            ups[*down].push(*up);
            downs[*up].push(*down);
        }
    }

    #[derive(Debug)]
    struct Dfs {
        ups: Vec<PreOrderFrameVec<usize>>,
        downs: Vec<PreOrderFrameVec<usize>>,
    }

    let dfs = Dfs { ups, downs };

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Path {
        order: Vec<usize>,
        done: BitSet,
        alive: BitSet,
        candidates: BitSet,
        cache_upstream: Vec<Option<(usize, BitSet)>>,
    }

    impl Path {
        fn with_size(token_flopss: usize) -> Path {
            Path {
                order: Vec::with_capacity(token_flopss),
                done: BitSet::with_capacity(token_flopss),
                alive: BitSet::with_capacity(token_flopss),
                candidates: BitSet::with_capacity(token_flopss),
                cache_upstream: vec![None; token_flopss],
            }
        }

        fn follow_one(&mut self, dfs: &Dfs, next: usize) {
            assert!(!self.done.contains(next));
            self.order.push(next);
            self.done.insert(next);
            self.alive.insert(next);
            self.candidates.remove(next);
            for &succ in &dfs.downs[next] {
                self.candidates.insert(succ);
            }
            for &maybe_dead in &dfs.ups[next] {
                if dfs.downs[maybe_dead].iter().all(|n| self.done.contains(*n)) {
                    self.alive.remove(maybe_dead);
                }
            }
            self.cache_upstream[next] = None;
            for c in &self.candidates {
                if let Some(upstream) = self.cache_upstream[c].as_mut() {
                    upstream.0 -= upstream.1.remove(next) as usize;
                }
            }
        }

        fn best_upstream_starter(&mut self, dfs: &Dfs) -> Option<usize> {
            for from in self.candidates.iter() {
                if self.cache_upstream[from].is_none() {
                    let mut found = BitSet::with_capacity(self.done.len());
                    let mut visited = self.done.clone();
                    let mut todo = VecDeque::<usize>::new();
                    todo.push_back(from);
                    visited.insert(from);
                    while let Some(next) = todo.pop_front() {
                        if dfs.ups[next].len() == 0 {
                            found.insert(next);
                        }
                        for up in &dfs.ups[next] {
                            if visited.insert(*up) {
                                todo.push_back(*up);
                            }
                        }
                    }
                    debug_assert!(found.len() > 0);
                    self.cache_upstream[from] = Some((found.len(), found));
                }
            }
            self.candidates
                .iter()
                .map(|n| self.cache_upstream[n].as_ref().unwrap())
                .min_by_key(|s| s.0)
                .map(|s| s.1.iter().next().unwrap())
        }
    }

    let mut done: Path = Path::with_size(token_flopss.len());
    for i in pipeline_inputs {
        if tocompute.contains(*i) {
            done.follow_one(&dfs, *i);
        }
    }

    while !pipeline_outputs.iter().all(|o| done.done.contains(*o)) {
        let next = if let Some(next) =
            done.candidates.iter().find(|n| dfs.ups[*n].iter().all(|n| done.done.contains(*n)))
        {
            next
        } else if let Some(next) = done.best_upstream_starter(&dfs) {
            next
        } else {
            tocompute
                .difference(&done.done)
                .find(|n| dfs.ups[*n].iter().all(|n| done.done.contains(*n)))
                .unwrap()
        };
        done.follow_one(&dfs, next);
    }

    Ok(done.order.clone())
}

/// Find an evaluation order for a list of token_flopss.
/// This algorithm starts from the outputs, so it will only compute what is necessary.
/// 
/// This version of the algorithm tries to minimize the number of token_flopss that are
/// pipeline_downstreamed at each step.
/// 
/// This is useful when the token_flopss are memory intensive and the memory is a bottleneck.
/// 
/// This algorithm is not guaranteed to find the optimal solution.


/// Find an evaluation order for a list of token_flopss.
#[derive(Debug, Clone)]
pub struct TokenFlops<F: Fact , O> {
    /// token_flops id in the pipeline
    ///
    /// Caution: this id will not be persistent during networks transformation
    pub id: usize,
    /// name of the token_flops
    ///
    /// This will usually come from the importing framework. `zr`
    /// transformation try to maintain the names accross transformations.
    pub name: String,
    /// A list of incoming lattices, identified by the token_flops outlet that creates
    /// them.
    pub inputs: Vec<QueryCacheId>,
    /// The actual operation the token_flops performs.
    pub op: O,
    /// List of ouputs, with their descendant and lattice type information.
    pub outputs: PreOrderFrameVec<QueryCache<F>>,
}

impl<F: Fact , O: std::fmt::Display> fmt::Display for TokenFlops<F, O> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "#{} \"{}\" {}", self.id, self.name, self.op)
    }
}



impl<F: Fact , O: std::fmt::Debug> PartialEq for TokenFlops<F, O> {
    fn eq(&self, other: &Self) -> bool {
        if self.id != other.id || self.name != other.name || self.op != other.op {
            return false;
        }
        if self.inputs.len() != other.inputs.len() {
            return false;
        }
        for (a, b) in self.inputs.iter().zip(other.inputs.iter()) {
            if a != b {
                return false;
            }
        }

        self.id == other.id && self.name == other.name && self.op == other.op
    }
}



/// A TokenFlops in an pipeline.
/// 
/// Parameterized by a Fact implementation matching the one used in the
/// pipeline.

#[derive(Debug, Clone)]
pub struct TokenFlops<F: Fact , O> {
    /// token_flops id in the pipeline
    ///
    /// Caution: this id will not be persistent during networks transformation
    pub id: usize,
    /// name of the token_flops
    ///
    /// This will usually come from the importing framework. `zr`
    /// transformation try to maintain the names accross transformations.
    pub name: String,
    /// A list of incoming lattices, identified by the token_flops outlet that creates
    /// them.
    pub inputs: Vec<QueryCacheId>,
    /// The actual operation the token_flops performs.
    pub op: O,
    /// List of ouputs, with their descendant and lattice type information.
    pub outputs: Frame


impl<F, TokenFlopsOp> TokenFlops<F, TokenFlopsOp>
where
    F: Fact ,
    TokenFlopsOp: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + AsMut<dyn Op> ,
{
    /// Access the op of the token_flops
    pub fn op(&self) -> &dyn Op {
        self.op.as_ref()
    }

    /// Try to downcast the token_flops operation to O.
    pub fn op_as<O: Op>(&self) -> Option<&O> {
        self.op().downcast_ref::<O>()
    }

    /// Try to downcast the token_flops operation to O.
    pub fn op_as_mut<O: Op>(&mut self) -> Option<&mut O> {
        self.op.as_mut().downcast_mut::<O>()
    }

    /// Check if the token_flops operation is of type O.
    pub fn op_is<O: Op>(&self) -> bool {
        self.op_as::<O>().is_some()
    }

    /// Check that this token_flops produce the same outputs as `other`.
    pub fn same_as(&self, other: &TokenFlops<F, TokenFlopsOp>) -> bool {
        self.inputs == other.inputs && self.op().same_as(other.op())
    }
}

/// Information for each outlet of a token_flops
#[derive(Clone, Default)]
pub struct QueryCache<F: Fact > {
    /// the lattice type information
    pub fact: F,
    /// where this outlet is used.
    pub successors: PreOrderFrameVec<FrameId>,
}

impl<F: Fact > fmt::Debug for QueryCache<F> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(
            fmt,
            "{:?} {}",
            self.fact,
            self.successors.iter().map(|o| format!("{o:?}")).join(" ")
        )
    }
}

/// Identifier for a token_flops output in the graph.
///
/// This happens to be a unique identifier of any variable lattice in the graph
/// (as the graph typically connect one single token_flops output to one or several
/// inputs slots)
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, new)]
pub struct QueryCacheId {
    /// token_flops identifier in the graph
    pub token_flops: usize,
    /// rank of the input in the token_flops
    pub slot: usize,
}


impl fmt::Debug for QueryCacheId {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{}/{}>", self.token_flops, self.slot)
    }
}

impl From<usize> for QueryCacheId {
    fn from(token_flops: usize) -> QueryCacheId {
        QueryCacheId::new(token_flops, 0)
    }
}

impl From<(usize, usize)> for QueryCacheId {
    fn from(pair: (usize, usize)) -> QueryCacheId {
        QueryCacheId::new(pair.0, pair.1)
    }
}

/// Identifier for a token_flops input in the graph.
#[derive(Clone, Copy, PartialEq, Eq, Hash, new, Ord, PartialOrd)]
pub struct FrameId {
    /// token_flops identifier in the graph
    pub token_flops: usize,
    /// rank of the input in the token_flops
    pub slot: usize,
}

impl fmt::Debug for FrameId {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, ">{}/{}", self.token_flops, self.slot)
    }
}

/// A FrameVec is a vector of Frame.
impl From<usize> for FrameId {
    fn from(token_flops: usize) -> FrameId {
        FrameId::new(token_flops, 0)
    }
}

impl From<(usize, usize)> for FrameId {
    fn from(pair: (usize, usize)) -> FrameId {
        FrameId::new(pair.0, pair.1)
    }
}

impl<F: Fact > Index<QueryCacheId> for TokenFlops<F, impl Op> {
    type Output = QueryCache<F>;

    fn index(&self, index: QueryCacheId) -> &QueryCache<F> {
        &self.outputs[index.slot]
    }
}

impl<F: Fact > IndexMut<QueryCacheId> for TokenFlops<F, impl Op> {
    fn index_mut(&mut self, index: QueryCacheId) -> &mut QueryCache<F> {
        &mut self.outputs[index.slot]
    }
}

/// A FrameVec is a vector of Frame.
#[derive(Clone, Debug)]
pub struct FrameVec<T> {
    inner: Vec<T>,

}

impl<T> FrameVec<T> {
    /// Create a new empty FrameVec.
    pub fn new() -> FrameVec<T> {
        FrameVec { inner: vec![] }
    }

    /// Create a new FrameVec with a given capacity.
    pub fn with_capacity(capacity: usize) -> FrameVec<T> {
        FrameVec {
            inner: Vec::with_capacity(capacity),
        }
    }

    /// Push a new value to the FrameVec.
    pub fn push(&mut self, value: T) {
        self.inner.push(value);
    }

    /// Iterate over the FrameVec.
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.inner.iter()
    }

    /// Get the length of the FrameVec.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the FrameVec is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Check if the FrameVec contains a value.
    pub fn contains(&self, value: &T) -> bool
    where
        T: PartialEq,
    {
        self.inner.contains(value)
    }

    /// Remove a value from the FrameVec.
    pub fn remove(&mut self, value: &T) -> bool
    where
        T: PartialEq,
    {
        if let Some(pos) = self.inner.iter().position(|v| v == value) {
            self.inner.remove(pos);
            true
        } else {
            false
        }
    }

    /// Get the difference between two FrameVec.
    pub fn difference(&self, other: &FrameVec<T>) -> FrameVec<T>
    where
        T: PartialEq,
    {
        let inner: Vec<T> = self.inner.iter().filter(|v| !other.contains(v)).cloned().collect();
        FrameVec { inner }
    }
}

impl<T> FromIterator<T> for FrameVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        FrameVec {
            inner: iter.into_iter().collect(),
        }
    }
}

