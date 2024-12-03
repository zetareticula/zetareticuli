use super::*;
use std::collections::HashSet;
use std::fmt::Debug;
use std::fmt::Display;
use std::ops::Deref;
use std::borrow::Cow;
use std::iter::FromIterator;
use std::collections::HashMap;
use std::collections::hash_map::Entry;


/// A token_flops in the graph.
/// A token_flops is a reference to an actor in the graph.
/// It contains the actor id and the output index of the actor.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TokenFlops<F, O> {
    pub actor: usize,
    pub output: usize,
    pub _phantom: PhantomData<(F, O)>,
}

impl<F, O> TokenFlops<F, O> {
    pub fn new(actor: usize, output: usize) -> Self {
        TokenFlops { actor, output, _phantom: PhantomData }
    }
}

/// Evaluate memory usage with its related actor at each step of the given order.
/// This function will evaluate the memory usage of each actor at each step of the given order.

#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub actor: usize,
    pub memory
}

pub fn eval_memory_usage<F, O, Flushable>(
    Pipeline: &ActorActor<F, O>,
    order: &[usize],
    pipeline_downstreamable: Flushable,
) -> TractResult<FrameVec<MemoryUsage>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    Flushable: Fn(&TokenFlops<F, O>) -> bool,
{
    let outputs = Pipeline.output_outlets()?.to_vec();

    let pipeline_downstream_lists = super::order::build_pipeline_downstream_list(Pipeline, order, &outputs, &pipeline_downstreamable);
    let mut values: FrameVec<bool> = tvec![false; Pipeline.actors.len()];

    let mut mem_by_steps: FrameVec<_> = tvec![(0, 0.into()); order.len()];

    let pipeline_downstreamable_actors = Pipeline
        .actors()
        .iter()
        .filter(|actor| (pipeline_downstreamable)(actor))
        .map(|it| it.id)
        .collect::<HashSet<_>>();

    for (step, n) in order.iter().enumerate() {
        let actor = Pipeline.actor(*n);

        for pipeline_downstream in pipeline_downstream_lists[step].iter() {
            values[*pipeline_downstream] = false;
        }

        // Active actors are actor that has not been pipeline_downstreamed + inputs of the current actor and current actor.
        let mut step_active_actors: HashSet<_> =
            values.iter().enumerate().filter_map(|(n, active)| active.then_some(n)).collect();

        step_active_actors.extend(actor.inputs.iter().map(|it| it.actor));
        step_active_actors.insert(*n);

        values[*n] = true;

        // Keep only pipeline_downstreamable actors.
        let step_active_pipeline_downstreamable_actors = step_active_actors.intersection(&pipeline_downstreamable_actors);

        mem_by_steps[step] = (*n, 0.into());

        for n in step_active_pipeline_downstreamable_actors {
            let out_facts = Pipeline
                .actor_output_facts(*n)?
                .into_iter()
                .map(|it| it.to_typed_fact())
                .collect::<TractResult<FrameVec<_>>>()?;
            mem_by_steps[step].1 += out_facts.iter().map(|it| it.mem_size()).sum::<TDim>();
        }
    }
    Ok(mem_by_steps)
}


/// Evaluate temporary memory usage with its related actor at each step of the given order.
pub fn eval_tmp_memory_usage<F, O, Flushable>(
    Pipeline: &ActorActor<F, O>,
    order: &[usize],
    pipeline_downstreamable: Flushable,
) -> TractResult<FrameVec<(usize, TDim)>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Op> + AsMut<dyn Op> + Clone + 'static,
    Flushable: Fn(&TokenFlops<F, O>) -> bool,
{
    let outputs = Pipeline.output_outlets()?.to_vec();

    let pipeline_downstream_lists = super::order::build_pipeline_downstream_list(Pipeline, order, &outputs, &pipeline_downstreamable);
    let mut values: FrameVec<bool> = tvec![false; Pipeline.actors.len()];

    let mut mem_by_steps: FrameVec<_> = tvec![(0, 0.into()); order.len()];

    let pipeline_downstreamable_actors = Pipeline
        .actors()
        .iter()
        .filter(|actor| (pipeline_downstreamable)(actor))
        .map(|it| it.id)
        .collect::<HashSet<_>>();

    for (step, n) in order.iter().enumerate() {
        let actor = Pipeline.actor(*n);

        for pipeline_downstream in pipeline_downstream_lists[step].iter() {
            values[*pipeline_downstream] = false;
        }

        // Active actors are actor that has not been pipeline_downstreamed + inputs of the current actor and current actor.
        let mut step_active_actors: HashSet<_> =
            values.iter().enumerate().filter_map(|(n, active)| active.then_some(n)).collect();

        step_active_actors.extend(actor.inputs.iter().map(|it| it.actor));
        step_active_actors.insert(*n);

        values[*n] = true;

        // Keep only pipeline_downstreamable actors.
        let step_active_pipeline_downstreamable_actors = step_active_actors.intersection(&pipeline_downstreamable_actors);

        mem_by_steps[step] = (*n, 0.into());

        for n in step_active_pipeline_downstreamable_actors {
            let out_facts = Pipeline
                .actor_output_facts(*n)?
                .into_iter()
                .map(|it| it.to_typed_fact())
                .collect::<TractResult<FrameVec<_>>>()?;
            mem_by_steps[step].1 += out_facts.iter().map(|it| it.mem_size()).sum::<TDim>();
        }
    }
    Ok(mem_by_steps)
}

/// Evaluate memory usage with its related actor at each step of the given order.
pub trait ResolveTo<Concrete> {
    type Param: ?Sized;
    fn resolve(&self, param: &Self::Param) -> TractResult<Concrete>;
}

/// A bound on a geometry.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum GeometryBound<Symbolic, Concrete> {
    Symbolic(Symbolic),
    Concrete(Concrete),
}

impl<S: ResolveTo<C>, C: Clone> GeometryBound<S, C> {
    pub fn is_concrete(&self) -> bool {
        match self {
            GeometryBound::Concrete { .. } => true,
            GeometryBound::Symbolic { .. } => false,
        }
    }

    pub fn into_concrete(self, param: &S::Param) -> TractResult<Self> {
        match self {
            Self::Symbolic(sym) => Ok(Self::Concrete(sym.resolve(param)?)),
            Self::Concrete(conc) => Ok(Self::Concrete(conc)),
        }
    }

    pub fn to_concrete(&self, param: &S::Param) -> TractResult<Cow<C>> {
        match self {
            Self::Symbolic(sym) => Ok(Cow::Owned(sym.resolve(param)?)),
            Self::Concrete(conc) => Ok(Cow::Borrowed(conc)),
        }
    }

    pub fn as_concrete(&self) -> Option<&C> {
        if let Self::Concrete(conc) = self {
            Some(conc)
        } else {
            None
        }
    }

    pub fn optimize_if(self, param: Option<&S::Param>) -> TractResult<Self> {
        if let Some(param) = param {
            self.into_concrete(param)
        } else {
            Ok(self)
        }
    }
}

impl<S, C> From<S> for GeometryBound<S, C> {
    fn from(s: S) -> Self {
        GeometryBound::Symbolic(s)
    }
}

impl<S, C> From<C> for GeometryBound<S, C> {
    fn from(c: C) -> Self {
        GeometryBound::Concrete(c)
    }
}

impl<S: Clone, C: Clone> Clone for GeometryBound<S, C> {
    fn clone(&self) -> Self {
        match self {
            GeometryBound::Symbolic(sym) => GeometryBound::Symbolic(sym.clone()),
            GeometryBound::Concrete(conc) => GeometryBound::Concrete(conc.clone()),
        }
    }
}
