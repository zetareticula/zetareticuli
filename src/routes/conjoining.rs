use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// A token_fljoins in the graph.
/// A token_fljoins is a reference to an actor in the graph.
/// It contains the actor id and the output index of the actor.

// #[derive(Debug, Clone, Hash, PartialEq, Eq)]
// pub struct TokenJoins<F, O> {
//     pub actor: usize,
//     pub output: usize,
//     pub _phantom: PhantomData<(F, O)>,
// }




// impl<F, O> TokenJoins<F, O> {
//     pub fn new(actor: usize, output: usize) -> Self {
//         TokenJoins { actor, output, _phantom: PhantomData }
//     }
// }

/// Evaluate memory usage with its related actor at each step of the given order.

#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub actor: usize,
    pub memory: usize,
}   

/// This function will evaluate the memory usage of each actor at each step of the given order.
pub fn eval_memory_usage<F, O, Flushable>(
    Pipeline: &Actor<F, O>,
    order: &[usize],
    pipeline_downstreamable: Flushable,
) -> TractResult<FrameVec<MemoryUsage>>
where
    F: Fact + Clone + 'static,
    O: Debug + Display + AsRef<dyn Join> + AsMut<dyn Join> + Clone + 'static,
    Flushable: Fn(&TokenJoins<F, O>) -> bool,
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
    }
}

#[derive(Debug, Clone)]
pub struct Actor<F, O> {
    pub actors: Vec<Actor<F, O>>,
    pub outputs: Vec<usize>,
}

/// An actor in the graph.
/// An actor is a node in the graph.
/// It contains the id of the actor, the inputs of the actor and the outputs of the actor.

#[derive(Debug, Clone)] 
pub struct Actor<F, O> {
    pub id: usize,
    pub inputs: Vec<TokenJoins<F, O>>,
    pub outputs: Vec<TokenJoins<F, O>>,
}

pub type Mem256f = v8;
pub type Mem256i = s8;
pub type v256 = v8;
pub type v256i = s8;
pub fn sum8(x: v256) -> f32 {
    x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7]
}

pub fn sum8i(x: v256i) -> i32 {
    x[0].wrapping_add(x[1])
        .wrapping_add(x[2])
        .wrapping_add(x[3])
        .wrapping_add(x[4])
        .wrapping_add(x[5])
        .wrapping_add(x[6])
        .wrapping_add(x[7])
}

pub fn log2i(x: v256i) -> v256 {
    [
        FastLog2(x[0] as u64),
        FastLog2(x[1] as u64),
        FastLog2(x[2] as u64),
        FastLog2(x[3] as u64),
        FastLog2(x[4] as u64),
        FastLog2(x[5] as u64),
        FastLog2(x[6] as u64),
        FastLog2(x[7] as u64),
    ]
    .into()
}


pub fn cast_i32_to_f32(x: v256i) -> v256 {
    [
        x[0] as f32,
        x[1] as f32,
        x[2] as f32,
        x[3] as f32,
        x[4] as f32,
        x[5] as f32,
        x[6] as f32,
        x[7] as f32,
    ]
    .into()
}
pub fn cast_f32_to_i32(x: v256) -> v256i {
    [
        x[0] as i32,
        x[1] as i32,
        x[2] as i32,
        x[3] as i32,
        x[4] as i32,
        x[5] as i32,
        x[6] as i32,
        x[7] as i32,
    ]
    .into()
}


/// Compute the cosine similarity between two vectors.
fn optimal_cosine_cycle_length() -> u32 {
    let mut best = 0;
    let mut best_err = std::f32::INFINITY;
    for i in 1..=256 {
        let err = (i as f32 * 2.0 * std::f32::consts::PI).cos().abs();
        if err < best_err {
            best = i;
            best_err = err;
        }
    }
    best
}

pub fn cosine_sim_avx_optimal(a: &[f32], b: &[f32]) -> f32 {
    let cycle_length = optimal_cosine_cycle_length();
    let mut sum = v256::splat(0.0);
    let mut sum_a = v256::splat(0.0);
    let mut sum_b = v256::splat(0.0);
    for i in 0..a.len() {
        let av = v256::load(&a[i]);
        let bv = v256::load(&b[i]);
        sum += av * bv;
        sum_a += av * av;
        sum_b += bv * bv;
    }
    sum = sum.horizontal_sum();
    sum_a = sum_a.horizontal_sum();
    sum_b = sum_b.horizontal_sum();
    sum / (sum_a.sqrt() * sum_b.sqrt())
}

pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
        sum_a += a[i] * a[i];
        sum_b += b[i] * b[i];
    }
    sum / (sum_a.sqrt() * sum_b.sqrt())
}


pub fn cosine_sim_avx(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = v256::splat(0.0);
    let mut sum_a = v256::splat(0.0);
    let mut sum_b = v256::splat(0.0);
    for i in 0..a.len() {
        let av = v256::load(&a[i]);
        let bv = v256::load(&b[i]);
        sum += av * bv;
        sum_a += av * av;
        sum_b += bv * bv;
    }
    sum = sum.horizontal_sum();
    sum_a = sum_a.horizontal_sum();
    sum_b = sum_b.horizontal_sum();
    sum / (sum_a.sqrt() * sum_b.sqrt())
}

pub fn cosine_sim_avx_i32(a: &[i32], b: &[i32]) -> f32 {
    let mut sum = v256::splat(0.0);
    let mut sum_a = v256::splat(0.0);
    let mut sum_b = v256::splat(0.0);
    for i in 0..a.len() {
        let av = cast_i32_to_f32(v256i::load(&a[i]));
        let bv = cast_i32_to_f32(v256i::load(&b[i]));
        sum += av * bv;
        sum_a += av * av;
        sum_b += bv * bv;
    }
    sum = sum.horizontal_sum();
    sum_a = sum_a.horizontal_sum();
    sum_b = sum_b.horizontal_sum();
    sum / (sum_a.sqrt() * sum_b.sqrt())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_sim() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_sim(&a, &b), 1.0);
    }

    #[test]
    fn test_cosine_sim_avx() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_sim_avx(&a, &b), 1.0);
    }

    #[test]
    fn test_cosine_sim_avx_i32() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3];
        assert_eq!(cosine_sim_avx_i32(&a, &b), 1.0);
    }

    #[test]
    fn test_cosine_sim_avx_optimal() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_sim_avx_optimal(&a, &b), 1.0);
    }
}