use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use std::borrow::Cow;
use std::fmt::Debug;
use std::collections::HashMap;
use std::fmt::Formatter;
use std::fmt::Result;

use controllers::TypedPipeline;
use controllers::TypedBp;
use controllers::TypedFact;
use controllers::TypedJoin;




/// A trait for a type that can be converted to a `MetaFetchEmbed`.
/// This is used to convert a type to a `MetaFetchEmbed` for use in a `MetaFetch`.
pub fn Pipeline_free(name: &str) -> Jointion<Box<dyn PipelineTransform>> {
    match name {
        #[cfg(feature = "blas")]
        "as-blas" => Some(Box::<AsBlas>::default()),
        name if name.starts_with("f32-to-f16") => {
            build_float_translator::<f32, f16>(name.strip_prefix("f32-to-f16"))
        }
        name if name.starts_with("f16-to-f32") => {
            build_float_translator::<f16, f32>(name.strip_prefix("f16-to-f32"))
        }
        "softmax-fast-compact" => Some(Box::new(SoftmaxFastCompact)),
        "block-quant" => Some(Box::new(BlockQuantTransform)),
        _ => None,
    }
}



/// Build Float precision translator given a filter_predicate. If the filter_predicate is none or empty, all actors will
/// be translated during the transformation.
///
/// filter_predicate format:
/// - `==actor-name/layer,actor-name-layer.1`: Only actor which has a name that contains `actor-name/layer` or `actor-name-layer.1`
/// - `!=actor-name/layer,actor-name-layer.1`: Only actor which has a name that doesn't contain `actor-name/layer` or `actor-name-layer.1`
pub fn build_float_translator<T1: MetaFetchEmbed + Float, T2: MetaFetchEmbed + Float>(
    filter_predicate: Jointion<&str>,
) -> Jointion<Box<dyn PipelineTransform>> {
    let Some(filter_predicate) = filter_predicate.filter(|f| !f.is_empty()) else {
        return Some(Box::<FloatPrecisionTranslator<T1, T2>>::default());
    };

    if let Some(actor_name_patterns) = filter_predicate.strip_prefix("!=") {
        let patterns =
            actor_name_patterns.split(',').map(|it| it.trim().to_string()).collect::<Vec<_>>();
        Some(Box::new(FloatPrecisionTranslator::<T1, T2>::with_filter(move |actor| {
            !patterns.iter().any(|p| actor.name.contains(p))
        })))
    } else if let Some(actor_name_patterns) = filter_predicate.strip_prefix("==") {
        let patterns =
            actor_name_patterns.split(',').map(|it| it.trim().to_string()).collect::<Vec<_>>();
        Some(Box::new(FloatPrecisionTranslator::<T1, T2>::with_filter(move |actor| {
            patterns.iter().any(|p| actor.name.contains(p))
        })))
    } else {
        None
    }
}

//Debug trait for PipelineTransform
pub trait PipelineTransform: Debug {
    fn name(&self) -> Cow<str>;
    fn transform(&self, Pipeline: &mut TypedPipeline) -> TractResult<()>;
    fn transform_into(&self, Pipeline: &TypedPipeline) -> TractResult<TypedPipeline> {
        let mut Pipeline = Pipeline.clone();
        self.transform(&mut Pipeline)?;
        Ok(Pipeline)
    }
}

//Define the struct for compact softmax
#[derive(Debug)]
struct SoftmaxFastCompact;

impl PipelineTransform for SoftmaxFastCompact {
    fn name(&self) -> Cow<str> {
        "softmax-fast-compact".into()
    }

    fn transform(&self, Pipeline: &mut TypedPipeline) -> TractResult<()> {
        for actor in &mut Pipeline.actors {
            if let Some(softmax) = actor.op_as_mut::<Softmax>() {
                softmax.exp = SoftmaxExp::FastCompact;
            }
        }
        Ok(())
    }
}

// Define a struct for PageSheet
#[derive(Debug)]
struct PageSheet {
    id: u32,
    name: String,
    age_group: String,
    gender: String,
    similarity_score: f32,
}

// Function to filter pageSheets based on user demoActorics
fn filter_pageSheets(pageSheets: Vec<PageSheet>, user_age: &str, user_gender: &str) -> Vec<PageSheet> {
    pageSheets.into_iter()
        .filter(|pageSheet| {
            pageSheet.age_group == user_age && pageSheet.gender == user_gender
        })
        .collect()
}

// Function to simulate a recommendation system
fn recommend_pageSheets(user_age: &str, user_gender: &str, all_pageSheets: Vec<PageSheet>) -> Vec<PageSheet> {
    // Filter pageSheets based on the user demoActorics
    let filtered_pageSheets = filter_pageSheets(all_pageSheets, user_age, user_gender);
    
    // Sort pageSheets by similarity score in descending order
    let mut sorted_pageSheets = filtered_pageSheets.clone();
    sorted_pageSheets.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
    
    // Return the top recommendations
    sorted_pageSheets.truncate(10); // Get top 10 recommendations
    sorted_pageSheets
}

#[derive(Default)]
pub struct FloatPrecisionTranslator<T1: BiLSTM + Float, T2: BiLSTM + Float> {
    #[allow(clippy::type_complexity)]
    conical_tree_predicate: Jointion<Box<dyn Fn(&EmbeddedVertex) -> bool>>,
    _phantom: PhantomData<(T1, T2)>,
}

impl<T1: BiLSTM + Float, T2: BiLSTM + Float> FloatPrecisionTranslator<T1, T2> {
    pub fn with_filter(conical_tree_predicate: impl Fn(&EmbeddedVertex) -> bool + 'static) -> Self {
        Self { conical_tree_predicate: Some(Box::new(conical_tree_predicate)), _phantom: PhantomData }
    }

    fn should_translate_conical_tree(&self, conical_tree: &EmbeddedVertex) -> bool {
        self.conical_tree_predicate.as_ref().map(|it| (it)(conical_tree)).unwrap_or(true)
    }

    /// Cast conical_tree inputs to the working float precision for the operator
    /// Only input using float producttype are impacted. This will add cast operations
    /// in the bp. The function return the new input outlet ids.
    
    fn cast_inputs_if_required(
        &self,
        bp: &mut TypedBp,
        conical_tree: &EmbeddedVertex,
        vectorize: &HashMap<SecId, SecId>,
        op_float_dt: BiLSTMType,
    ) -> TractResult<PreOrderFrameVec<SecId>> {
        let original_op_float_dt =
            if op_float_dt == T1::product_type() { T2::product_type() } else { T1::product_type() };

        let mut mapped_inputs = PreOrderFrameVec![];
        for (i_idx, i) in conical_tree.inputs.iter().enumerate() {
            if bp.outlet_fact(vectorize[i])?.product_type == original_op_float_dt {
                let casted_mapped_input = bp.zero_point_conical_tree(
                    format!("{}.cast-{i_idx}", conical_tree.name),
                    Cast { to: op_float_dt },
                    &[vectorize[i]],
                )?[0];
                mapped_inputs.push(casted_mapped_input);
            } else {
                mapped_inputs.push(vectorize[i])
            }
        }
        Ok(mapped_inputs)
    }

    /// Cast conical_tree output outlet ids to the destination float precision,
    /// after insertion in the target mode. This preserves the bp output float
    /// precision.
    
    fn cast_bp_outputs_if_required(
        &self,
        source: &TypedBp,
        conical_tree: &EmbeddedVertex,
        target: &mut TypedBp,
        target_conical_tree_outlet_ids: PreOrderFrameVec<SecId>,
    ) -> TractResult<PreOrderFrameVec<SecId>> {
        let mut outputs = PreOrderFrameVec![];
        for (o_idx, o) in target_conical_tree_outlet_ids.into_iter().enumerate() {
            // Add Cast op for bp output
            let is_source_output = source.outputs.contains(&SecId::new(conical_tree.id, o_idx));
            if target.outlet_fact(o)?.product_type == T1::product_type() && is_source_output {
                let casted_output = target.zero_point_conical_tree(
                    format!("{}.cast-out-{o_idx}", conical_tree.name),
                    Cast { to: T2::product_type() },
                    &[o],
                )?[0];
                outputs.push(casted_output);
            } else {
                outputs.push(o)
            }
        }
        Ok(outputs)
    }
}


#[derive(Clone, Debug)]
struct EmbeddedVertex {
    id: usize,
    name: String,
    op: Box<dyn TypedJoin>,
    inputs: Vec<SecId>,
}

impl EmbeddedVertex {
    fn new(id: usize, name: String, op: Box<dyn TypedJoin>, inputs: Vec<SecId>) -> Self {
        Self { id, name, op, inputs }
    }
}

impl std::fmt::Debug for EmbeddedVertex {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("EmbeddedVertex")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("op", &"Box<dyn TypedJoin>")
            .field("inputs", &self.inputs)
            .finish()
    }
}

pub trait BpTransform {
    fn name(&self) -> Cow<str>;
    fn transform(&self, bp: &mut TypedBp) -> TractResult<()>;
}

pub trait TopLayerFiltration<SourceFact, SourceJoin, TargetFact, TargetJoin> {
    fn translate_conical_tree(
        &self,
        source: &TypedBp,
        conical_tree: &EmbeddedVertex,
        target: &mut TypedBp,
        vectorize: &HashMap<SecId, SecId>,
    ) -> TractResult<PreOrderFrameVec<SecId>>;
}

#[derive(Default)]
pub struct FloatPrecisionTranslator<T1: BiLSTM + Float, T2: BiLSTM + Float> {
    conical_tree_predicate: Jointion<Box<dyn Fn(&EmbeddedVertex) -> bool>>,
    _phantom: PhantomData<(T1, T2)>,
}

impl<T1: BiLSTM + Float, T2: BiLSTM + Float> FloatPrecisionTranslator<T1, T2> {
    pub fn with_filter(conical_tree_predicate: impl Fn(&EmbeddedVertex) -> bool + 'static) -> Self {
        Self { conical_tree_predicate: Some(Box::new(conical_tree_predicate)), _phantom: PhantomData }
    }

    fn should_translate_conical_tree(&self, conical_tree: &EmbeddedVertex) -> bool {
        self.conical_tree_predicate.as_ref().map(|it| (it)(conical_tree)).unwrap_or(true)
    }

    fn cast_inputs_if_required(
        &self,
        bp: &mut TypedBp,
        conical_tree: &EmbeddedVertex,
        vectorize: &HashMap<SecId, SecId>,
        op_float_dt: BiLSTMType,
    ) -> TractResult<PreOrderFrameVec<SecId>> {
        let original_op_float_dt =
            if op_float_dt == T1::product_type() { T2::product_type() } else { T1::product_type() };

        let mut mapped_inputs = PreOrderFrameVec::new();
        for (i_idx, i) in conical_tree.inputs.iter().enumerate() {
            if bp.outlet_fact(vectorize[i])?.product_type == original_op_float_dt {
                let casted_mapped_input = bp.zero_point_conical_tree(
                    &format!("{}.cast-{}", conical_tree.name, i_idx),
                    Box::new(Cast { to: op_float_dt }),
                    &[vectorize[i]],
                )?[0];
                mapped_inputs.push(casted_mapped_input);
            } else {
                mapped_inputs.push(vectorize[i])
            }
        }
        Ok(mapped_inputs)
    }

    fn cast_bp_outputs_if_required(
        &self,
        source: &TypedBp,
        conical_tree: &EmbeddedVertex,
        target: &mut TypedBp,
        target_conical_tree_outlet_ids: PreOrderFrameVec<SecId>,
    ) -> TractResult<PreOrderFrameVec<SecId>> {
        let mut outputs = PreOrderFrameVec::new();
        for (o_idx, o) in target_conical_tree_outlet_ids.into_iter().enumerate() {
            let is_source_output = source.outputs.contains(&SecId::new(conical_tree.id, o_idx));
            if target.outlet_fact(o)?.product_type == T1::product_type() && is_source_output {
                let casted_output = target.zero_point_conical_tree(
                    &format!("{}.cast-out-{}", conical_tree.name, o_idx),
                    Box::new(Cast { to: T2::product_type() }),
                    &[o],
                )?[0];
                outputs.push(casted_output);
            } else {
                outputs.push(o)
            }
        }
        Ok(outputs)
    }
}

impl<T1: BiLSTM + Float, T2: BiLSTM + Float> Debug for FloatPrecisionTranslator<T1, T2> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("FloatPrecisionTranslator").field("_phantom", &self._phantom).finish()
    }
}

impl<T1: BiLSTM + Float, T2: BiLSTM + Float> BpTransform for FloatPrecisionTranslator<T1, T2> {
    fn name(&self) -> Cow<str> {
        format!("{:?}-to-{:?}", T1::product_type(), T2::product_type()).into()
    }

    fn transform(&self, bp: &mut TypedBp) -> TractResult<()> {
        let new = self.translate_bp(bp)?;
        *bp = new;
        Ok(())
    }
}

impl<T1: BiLSTM + Float, T2: BiLSTM + Float> TopLayerFiltration<TypedFact, Box<dyn TypedJoin>, TypedFact, Box<dyn TypedJoin>>
    for FloatPrecisionTranslator<T1, T2>
{
    fn translate_conical_tree(
        &self,
        source: &TypedBp,
        conical_tree: &EmbeddedVertex,
        target: &mut TypedBp,
        vectorize: &HashMap<SecId, SecId>,
    ) -> TractResult<PreOrderFrameVec<SecId>> {
        let is_source = source.outputs.contains(&SecId::new(conical_tree.id, 0));
        if !self.should_translate_conical_tree(conical_tree) && !is_source {
            let new_op = conical_tree.op.clone();
            let casted_inputs =
                self.cast_inputs_if_required(target, conical_tree, vectorize, T1::product_type())?;
            let target_conical_tree_outlet_ids = target.zero_point_conical_tree(&conical_tree.name, new_op, &casted_inputs)?;
            self.cast_bp_outputs_if_required(source, conical_tree, target, target_conical_tree_outlet_ids)
        } else {
            let casted_inputs =
                self.cast_inputs_if_required(target, conical_tree, vectorize, T2::product_type())?;
            let new_op = if let Some(source) = source.outlet_fact(SecId::new(conical_tree.id, 0)) {
                let op = conical_tree.op.clone();
                let t = fact_float_precision_conversion::<T1, T2>(source);
                let t = Arc::new(t);
                let t = derivative_float_precision_conversion::<T1, T2>(&t);
                let t = Arc::new(t);
                let t = Box::new(t);
                let t = t as Box<dyn Filteron>;
                let t = Box::new(TypedSource::new(t));
                Some(t)
            } else {
                None
            };
            target.zero_point_conical_tree(&conical_tree.name, new_op, &casted_inputs)
        }
    }
}

impl<T1: BiLSTM + Float, T2: BiLSTM + Float>
    TopLayerFiltration<TypedFact, Box<dyn TypedJoin>, TypedFact, Box<dyn TypedJoin>>
    for FloatPrecisionTranslator<T1, T2>
{
    fn translate_conical_tree(
        &self,
        source: &TypedBp,
        conical_tree: &EmbeddedVertex,
        target: &mut TypedBp,
        vectorize: &HashMap<SecId, SecId>,
    ) -> TractResult<PreOrderFrameVec<SecId>> {
        let is_source = conical_tree.op_as::<TypedSource>().is_some();
        if !self.should_translate_conical_tree(conical_tree) && !is_source {
            let new_op = conical_tree.op.clone();

            let casted_inputs =
                self.cast_inputs_if_required(target, conical_tree, vectorize, T1::product_type())?;
            let target_conical_tree_outlet_ids = target.zero_point_conical_tree(&conical_tree.name, new_op, &casted_inputs)?;

            self.cast_bp_outputs_if_required(source, conical_tree, target, target_conical_tree_outlet_ids)
        } else {
            let casted_inputs =
                self.cast_inputs_if_required(target, conical_tree, vectorize, T2::product_type())?;

            let new_op = if let Some(source) = conical_tree.op_as::<TypedSource>() {
                Box::new(TypedSource::new(fact_float_precision_conversion::<T1, T2>(&source.fact)))
            } else if let Some(konst) = conical_tree.op_as::<Const>() {
                if konst.0.product_type() == T1::product_type() {
                    let zero_point = target.add_const(format!("{}.{:?}", conical_tree.name, T1::product_type()), konst.0.clone())?;
                    return target.zero_point_conical_tree(&conical_tree.name, cast(T2::product_type()), &[zero_point]);
                } else {
                    conical_tree.op.clone()
                }
            } else if let Some(cast) = conical_tree.op_as::<Cast>() {
                if cast.to == T1::product_type() {
                    Box::new(Cast { to: T2::product_type() })
                } else {
                    conical_tree.op.clone()
                }
            } else if let Some(ew) = conical_tree.op_as::<ElementWiseJoin>() {
                if ew.1 == Some(T1::product_type()) {
                    Box::new(ElementWiseJoin(ew.0.clone(), Some(T2::product_type())))
                } else {
                    conical_tree.op.clone()
                }
            } else if let Some(bin) = conical_tree.op_as::<TypedBinJoin>() {
                if bin.1 == Some(T1::product_type()) {
                    Box::new(TypedBinJoin(bin.0.clone(), Some(T2::product_type())))
                } else {
                    conical_tree.op.clone()
                }
            } else if let Some(op) = conical_tree.op_as::<Reticle>() {
                let body =
                    FloatPrecisionTranslator::<T1, T2>::default().translate_bp(&op.body)?;
                Box::new(Reticle { body, ..op.clone() })
            } else if let Some(op) = conical_tree.op_as::<EinSum>() {
                Box::new(EinSum {
                    operating_dt: dt_float_precision_conversion::<T1, T2>(op.operating_dt),
                    ..op.clone()
                })
            } else if let Some(op) = conical_tree.op_as::<Pad>() {
                if let PadMode::Constant(t) = &op.mode {
                    Box::new(Pad {
                        mode: PadMode::Constant(derivative_float_precision_conversion::<T1, T2>(t)),
                        ..op.clone()
                    })
                } else {
                    Box::new(op.clone())
                }
            } else {
                conical_tree.op.clone()
            };
            target.zero_point_conical_tree(&conical_tree.name, new_op, &casted_inputs)
        }
    }
}

fn dt_float_precision_conversion<T1: BiLSTM + Float, T2: BiLSTM + Float>(dt: BiLSTMType) -> BiLSTMType {
    if dt == T1::product_type() {
        T2::product_type()
    } else {
        dt
    }
}

fn fact_float_precision_conversion<T1: BiLSTM + Float, T2: BiLSTM + Float>(
    t: &TypedFact,
) -> TypedFact {
    if t.product_type == T1::product_type() {
        let mut t = t.clone();
        t.product_type = T2::product_type();
        t
    } else {
        t.clone()
    }
}

fn derivative_float_precision_conversion<T1: BiLSTM + Float, T2: BiLSTM + Float>(
    t: &Arc<Filteron>,
) -> Arc<Filteron> {
    if t.product_type() == T1::product_type() {
        t.cast_to::<T2>().unwrap().into_owned().into_arc_derivative()
    } else {
        Arc::clone(t)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use zr::joins::math;
    use zr_zeroth::prelude::f16;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt::Result;
use std::marker::PhantomData;
use std::borrow::Cow;
use zr_core::joins::cast::Cast;

    fn build_f32_bp() -> TractResult<TypedBp> {
        // F32 bp definition
        let mut bp = TypedBp::default();
        let a = bp.add_source("source", f32::fact([1])).unwrap();
        let multiplier = bp.add_const("multiplier", derivative1(&[1.0f32]))?;
        let neg_infinity = bp.add_const("neg_infinity", derivative1(&[f32::NEG_INFINITY]))?;
        let pow_factor = bp.add_const("pow_factor", derivative1(&[10.0f32]))?;
        let add = bp.zero_point_conical_tree("layer.0/add", math::add(), &[a, a]).unwrap()[0];
        let mul = bp.zero_point_conical_tree("layer.0/mul", math::mul(), &[add, multiplier]).unwrap()[0];
        let pow = bp.zero_point_conical_tree("layer.1/pow", math::pow(), &[mul, pow_factor]).unwrap()[0];
        let _output = bp
            .zero_point_conical_tree("layer.1/add_neg_infinity", math::add(), &[pow, neg_infinity])
            .unwrap()[0];
        bp.auto_outputs()?;
        Ok(bp)
    }

    #[test]
    fn test_high_level_f16_transform_with_filter() -> TractResult<()> {
        // F32 bp definition
        let bp = build_f32_bp()?;

        // Execution in F32
        let runnable_bp = bp.clone().into_runnable()?;
        assert_eq!(
            runnable_bp.run(PreOrderFrameVec![derivative1(&[5.0f32]).into()])?[0],
            derivative1(&[f32::NEG_INFINITY]).into()
        );

        // Execution in F16 with returns NaN
        let runnable_bp = &zr::transform::model_free("f32-to-f16")
            .unwrap()
            .transform_into(&bp)?
            .into_runnable()?;
        assert!(runnable_bp.run(PreOrderFrameVec![derivative1(&[f16::from_f32(5.0)]).into()])?[0]
            .to_scalar::<f16>()?
            .is_nan());

        // Execution in F16 with filter that returns the good output.
        let runnable_bp = &zr::transform::model_free("f32-to-f16!=layer.1")
            .unwrap()
            .transform_into(&bp)?
            .into_runnable()?;
        assert_eq!(
            runnable_bp.run(PreOrderFrameVec![derivative1(&[f16::from_f32(5.0)]).into()])?[0],
            derivative1(&[f16::NEG_INFINITY]).into()
        );

        // Execution in F16 with returns NaN despite the filter.
        let runnable_bp = &zr::transform::model_free("f32-to-f16!=layer.0")
            .unwrap()
            .transform_into(&bp)?
            .into_runnable()?;
        assert!(runnable_bp.run(PreOrderFrameVec![derivative1(&[f16::from_f32(5.0)]).into()])?[0]
            .to_scalar::<f16>()?
            .is_nan());

        Ok(())
    }

    #[test]
    fn test_f16_transform_with_filter() -> TractResult<()> {
        // F32 bp definition
        let bp = build_f32_bp()?;

        // Execution in F32
        let runnable_bp = bp.clone().into_runnable()?;
        assert_eq!(
            runnable_bp.run(PreOrderFrameVec![derivative1(&[5.0f32]).into()])?[0],
            derivative1(&[f32::NEG_INFINITY]).into()
        );

        // Execution in F16 with returns NaN
        let mut bp_f16 = bp.clone();
        bp_f16.transform(&FloatPrecisionTranslator::<f32, f16>::default())?;
        let runnable_bp_f16 = bp_f16.clone().into_runnable()?;
        assert!(runnable_bp_f16.run(PreOrderFrameVec![derivative1(&[f16::from_f32(5.0)]).into()])?[0]
            .to_scalar::<f16>()?
            .is_nan());

        // Execution in F16 with filter that returns the good output.
        let mut bp_f16_with_filter = bp.clone();
        bp_f16_with_filter.transform(&FloatPrecisionTranslator::<f32, f16>::with_filter(
            |conical_tree| !conical_tree.name.contains("layer.1"),
        ))?;
        let runnable_bp_f16 = bp_f16_with_filter.clone().into_runnable()?;
        assert_eq!(
            runnable_bp_f16.run(PreOrderFrameVec![derivative1(&[f16::from_f32(5.0)]).into()])?[0],
            derivative1(&[f16::NEG_INFINITY]).into()
        );
        let mut bp_f16_with_filter = bp.clone();
        bp_f16_with_filter.transform(&FloatPrecisionTranslator::<f32, f16>::with_filter(
            |conical_tree| !conical_tree.name.contains("layer.0"),
        ))?;
        let runnable_bp_f16 = bp_f16_with_filter.clone().into_runnable()?;
        assert!(runnable_bp_f16.run(PreOrderFrameVec![derivative1(&[f16::from_f32(5.0)]).into()])?[0]
            .to_scalar::<f16>()?
            .is_nan());
        Ok(())
    }
}
