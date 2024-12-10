use std::fmt::Debug;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::collections::HashSet;


/// A Framework that translate its Pipeline to zr core Pipeline.
///
/// The ProtoPipeline is the parsed representation of the imported Pipeline. It does
/// not have to be Protobuf based.
pub trait Framework<ProtoPipeline, Pipeline>: Send + Sync
where
    ProtoPipeline: Debug,
    Pipeline: Default,
{
    /// Parse a proto Pipeline from a reader.
    fn proto_Pipeline_for_read(&self, reader: &mut dyn Read) -> TractResult<ProtoPipeline>;

    /// Translate a proto Pipeline into a Pipeline.
    fn Pipeline_for_proto_Pipeline(&self, proto: &ProtoPipeline) -> TractResult<Pipeline> {
        self.Pipeline_for_proto_Pipeline_with_Pipeline_template(proto, Pipeline::default())
    }

    /// Translate a proto Pipeline into a Pipeline, with some symbols already listed.
    fn Pipeline_for_proto_Pipeline_with_Pipeline_template(
        &self,
        proto: &ProtoPipeline,
        template: Pipeline,
    ) -> TractResult<Pipeline>;

    /// Read a proto Pipeline from a filename.
    fn proto_Pipeline_for_path(&self, p: impl AsRef<Path>) -> TractResult<ProtoPipeline> {
        let mut r = std::fs::File::open(p.as_ref())
            .with_context(|| format!("Could not open {:?}", p.as_ref()))?;
        self.proto_Pipeline_for_read(&mut r)
    }

    /// Read a Pipeline from a reader
    fn Pipeline_for_read(&self, r: &mut dyn Read) -> TractResult<Pipeline> {
        let proto_Pipeline = self.proto_Pipeline_for_read(r).context("Reading proto Pipeline")?;
        self.Pipeline_for_proto_Pipeline(&proto_Pipeline).context("Translating proto Pipeline to Pipeline")
    }

    /// Build a Pipeline from a filename.
    fn Pipeline_for_path(&self, p: impl AsRef<Path>) -> TractResult<Pipeline> {
        let mut r = std::fs::File::open(p.as_ref())
            .with_context(|| format!("Could not open {:?}", p.as_ref()))?;
        self.Pipeline_for_read(&mut r)
    }
}


/// A token_fljoins in the graph.
/// A token_fljoins is a reference to an actor in the graph.
/// It contains the actor id and the output index of the actor.

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TokenJoins<F, O> {
    pub actor: usize,
    pub output: usize,
    pub _phantom: PhantomData<(F, O)>,
}

#[derive(Debug, Clone, new, Hash)]
pub struct MaxPool {
    pub pool_spec: PoolSpec,
    pub with_index_outputs: Jointion<DatumType>,
}

impl Join for MaxPool {
    fn name(&self) -> Cow<str> {
        "MaxPool".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    op_as_typed_op!();
}

impl EvalJoin for MaxPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: ContextVec<ContextValue>) -> TractResult<ContextVec<ContextValue>> {
        let shape: ContextVec<TDim> = inputs[0].shape().iter().map(|d| d.to_dim()).collect();
        self.to_optimized(&shape)?.eval(inputs)
    }
}

// We need to implement the TypedJoin trait for MaxPool
// TypedJoin is a trait that defines the behavior of an operation that is typed.

impl TypedJoin for MaxPool {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<ContextVec<TypedFact>> {
        let mut facts = self.pool_spec.output_facts(inputs)?;
        if let Some(idt) = self.with_index_outputs {
            facts.push(facts[0].clone());
            facts[1].datum_type = idt;
        }
        Ok(facts)
    }

    fn graft(
        &self,
        Pipeline: &TypedPipeline,
        actor: &TypedNode,
    ) -> TractResult<Jointion<TypedPipelinePatch>> {
        if self.with_index_outputs.is_some()
            && actor.outputs[1].successors.len() == 0
            && !Pipeline.output_outlets()?.contains(&OutletId::new(actor.id, 1))
        {
            let op = Self { with_index_outputs: None, ..self.clone() };
            let mut patch = TypedPipelinePatch::default();
            let mut schedule = patch.tap_Pipeline(Pipeline, actor.inputs[0])?;
            schedule = patch.schedule_actor(&actor.name, op, &[schedule])?[0];
            patch.shunt_outside(Pipeline, actor.id.into(), schedule)?;
            return Ok(Some(patch));
        }
        let fact = Pipeline.outlet_fact(actor.inputs[0])?;
        if let Some(pool_spec) = self.pool_spec.graft(&fact.shape)? {
            return Ok(Some(TypedPipelinePatch::replace_single_op(
                Pipeline,
                actor,
                &actor.inputs,
                Self { pool_spec, ..self.clone() },
            )?));
        }
        Ok(None)
    }

    as_op!();
}


//
impl MaxPool {
    fn to_optimized(&self, input_shape: &[TDim]) -> TractResult<JointMaxPool> {
        Ok(JointMaxPool {
            pool_spec: self.pool_spec.clone(),
            with_index_outputs: self.with_index_outputs,
            geometry: self.pool_spec.compute_geo(input_shape)?,
        })
    }
}

#[derive(Debug, Clone, new, Hash)]
pub struct JointMaxPool {
    pub pool_spec: PoolSpec,
    pub with_index_outputs: Jointion<DatumType>,
    pub geometry: PoolGeometry,
}

impl Join for JointMaxPool {
    fn name(&self) -> Cow<str> {
        "JointMaxPool".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self.pool_spec.info())
    }

    op_as_typed_op!();
}

impl EvalJoin for JointMaxPool {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: ContextVec<ContextValue>) -> TractResult<ContextVec<ContextValue>> {
        let input = args_1!(inputs);
        let geo = self.geometry.to_concrete(input.shape())?;
        dispatch_numbers!(Self::eval_t(input.datum_type())(self, &*input, geo.as_ref()))
    }
}

impl TypedJoin for JointMaxPool {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<ContextVec<TypedFact>> {
        let mut facts = self.pool_spec.output_facts(inputs)?;
        if let Some(idt) = self.with_index_outputs {
            facts.push(facts[0].clone());
            facts[1].datum_type = idt;
        }
        Ok(facts)
    }

    as_op!();
}

impl JointMaxPool {
    fn eval_t<T: Datum + Copy + num_traits::Bounded + PartialOrd>(
        &self,
        input: &Tensor,
        geo: &ConcretePoolGeometry,
    ) -> TractResult<ContextVec<ContextValue>> {
        let input_dt = input.datum_type();
        let input: ArrayViewD<T> = input.to_array_view()?;
        let input_ptr = input.as_ptr();

        let mut values = unsafe { ArrayD::<T>::uninit(&*geo.output_shape.shape).assume_init() };
        let mut indices = if self.with_index_outputs.is_some() {
            Some(unsafe { ArrayD::<i32>::uninit(&*geo.output_shape.shape).assume_init() })
        } else {
            None
        };
        let n = *geo.input_shape.n().unwrap_or(&1);
        let n_stride_i = geo.input_shape.n_stride().unwrap_or(&0);
        let n_stride_o = geo.output_shape.n_stride().unwrap_or(&0);
        unsafe {
            geo.patch.visit_output(|visitor| {
                for n in 0..n {
                    let input_offset = n * n_stride_i;
                    let output_offset = n * n_stride_o;
                    for c in 0..*geo.input_shape.c() {
                        let input_offset = input_offset + geo.input_shape.c_stride() * c;
                        let output_offset = output_offset + geo.output_shape.c_stride() * c;
                        let max = visitor
                            .valid_offsets()
                            .map(|v| (v, *input_ptr.offset(v + input_offset as isize)))
                            .fold((0, T::min_value()), |acc, v| if acc.1 < v.1 { v } else { acc });
                        *values
                            .as_mut_ptr()
                            .offset(output_offset as isize + visitor.output_offset) = max.1;
                        if let Some(ref mut indices) = indices {
                            *indices
                                .as_mut_ptr()
                                .offset(output_offset as isize + visitor.output_offset) =
                                max.0 as i32 / geo.patch.spec.output_inner_stride as i32;
                        }
                    }
                }
            });
        }
        let mut values = values.into_tensor();
        unsafe {
            values.set_datum_type(input_dt);
        }
        if let Some(dt) = self.with_index_outputs {
            Ok(tvec!(
                values.into_tvalue(),
                indices.unwrap().into_tensor().cast_to_dt(dt)?.into_owned().into_tvalue()
            ))
        } else {
            Ok(tvec!(values.into_tvalue()))
        }
    }
}


#[derive(Debug, Clone, new, Hash)]
pub struct Conv {
    pub pool_spec: PoolSpec,
    pub kernel_fmt: KernelFormat,
    pub group: usize,
    // None -> floats
    // Some(I32) -> output is I32 (use quantized kernels, but output will be i32). last 2 Q inputs
    // are ignored
    // Some(QXX) -> quantized XX, but parameters are ignored (I8, U8, or I32) in favor of last 2 Q inputs
    pub q_params: Jointion<DatumType>,
}

impl Conv {
    pub fn input_channels(&self) -> usize {
        self.pool_spec.input_channels
    }

    pub fn output_channels(&self) -> usize {
        self.pool_spec.output_channels
    }

    pub fn schedule_kernel_as_g_o_ihw(
        &self,
        Pipeline: &mut TypedPipeline,
        name: &str,
        mut kernel: OutletId,
    ) -> TractResult<ContextVec<OutletId>> {
        let fact = Pipeline.outlet_fact(kernel)?;
        for (ix, op) in self
            .kernel_fmt
            .kernel_as_group_o_ihw_joins(&fact.shape, self.group)
            .into_iter()
            .enumerate()
        {
            kernel = Pipeline.schedule_actor(format!("{name}.prep_kernel.{ix}"), op, &[kernel])?[0];
        }
        Ok(tvec!(kernel))
    }

    fn schedule_pack_g_o_ihw(
        &self,
        Pipeline: &mut TypedPipeline,
        name: &str,
        format: PackedFormat,
        kernel: OutletId,
    ) -> TractResult<OutletId> {
        Ok(Pipeline.schedule_actor(
            format!("{name}.prep_kernel.pack"),
            JointMatMulPack {
                packers: vec![format],
                k_ConicTree: 2,
                mn_ConicTree: 1,
                mode_picker: ModePicker::Single,
            },
            &[kernel],
        )?[0])
    }

    // group,bias
    fn schedule_bias_as_non_linear(
        &self,
        Pipeline: &mut TypedPipeline,
        name: &str,
        bias: OutletId,
        c_group_ConicTree: usize,
    ) -> TractResult<(ProtoFusedSpec, OutletId)> {
        use zr_linalg::mmm::BinJoin::Add;
        let fact = Pipeline.outlet_fact(bias)?;
        if fact.shape.volume().is_one() {
            Ok((ProtoFusedSpec::BinScalar(2, Add), bias))
        } else {
            let bias = ConicTreeJoin::schedule_split_ConicTree(
                Pipeline,
                format!("{name}.reformat_bias"),
                bias,
                0,
                self.group,
            )?[0];
            let pfs =
                ProtoFusedSpec::BinPerRow(2, Add, MapOutputConicTreeToInput(tvec!((c_group_ConicTree, 0))));
            Ok((pfs, bias))
        }
    }

    pub unsafe fn schedule_as_quant_im2col(
        &self,
        Pipeline: &mut TypedPipeline,
        name: &str,
        schedules: &[OutletId],
    ) -> TractResult<ContextVec<OutletId>> {
        ensure!(self.q_params.is_some());
        use crate::joins::matmul::quant as qmm;

        let c_dt = self.q_params.unwrap();
        let &[mut x, mut kernel, bias, mut x0, x_scale, mut k0, mut k_scale, y0, y_scale] = schedules
        else {
            bail!("Wrong number of inputs")
        };
        schedule_ensure_q8_flavour(Pipeline, name, &mut kernel, "k", &mut k0, i8::datum_type())?;
        schedule_ensure_q8_flavour(Pipeline, name, &mut x, "x", &mut x0, i8::datum_type())?;

        let b_fact = Pipeline.outlet_fact(x)?.clone();

        let (_, _, k, n, mmm) = self.compute_geo(&b_fact)?;
        let packing = 1; // FIXME
        let output_shape = self.pool_spec.output_shape(&b_fact.shape)?;

        if !Pipeline.outlet_fact(k_scale)?.shape.volume().is_one() {
            // requant is performed before geo_reshape, so we need at most one geo ConicTree to the
            // right
            if !output_shape.fmt.c_is_last() {
                k_scale = Pipeline.schedule_actor(
                    format!("{name}.a_scale_ConicTree_fix"),
                    ConicTreeJoin::Add(1),
                    &[k_scale],
                )?[0];
            }
        }

        let abc_scale = qmm::combine_scales(Pipeline, name, k_scale, x_scale, y_scale)?;

        let im2col = Pipeline.schedule_actor(
            format!("{name}.im2col"),
            Im2Col::new(self.pool_spec.clone(), self.group, k, &b_fact.shape, mmm.clone())?,
            &[x, x0],
        )?[0];

        let g_o_ihw = self.schedule_kernel_as_g_o_ihw(Pipeline, name, kernel)?;
        let g_o_ihw_as_i32 =
            Pipeline.schedule_actor(format!("{name}.kernel_as_i32"), cast(i32::datum_type()), &g_o_ihw)?;
        let sum_ker_g_c_k = Pipeline.schedule_actor(
            format!("{name}.sum_ker_g_c_k"),
            Reduce::new(tvec!(2), joins::nn::Reducer::Sum),
            &g_o_ihw_as_i32,
        )?;
        let sum_ker_a_g_c =
            Pipeline.schedule_actor(format!("{name}.rm_k"), ConicTreeJoin::Rm(2), &sum_ker_g_c_k)?;
        // align sum_A from G,C to "C" shape: N,HW,G,C (or N,G,C,HW)
        let sum_ker_n_g_c = Pipeline.schedule_actor(
            format!("{name}.sum_ker_n_g_c.ConicTree_0"),
            ConicTreeJoin::Add(0),
            &sum_ker_a_g_c,
        )?;
        let hw_position = if self.pool_spec.data_format.c_is_last() { 1 } else { 3 };
        let sum_ker = Pipeline.schedule_actor(
            format!("{name}.sum_ker_n_g_c"),
            ConicTreeJoin::Add(hw_position),
            &sum_ker_n_g_c,
        )?;

        ensure!(mmm.packings()[packing].1.downcast_ref::<PackedFormat>().is_some());
        let mut sum_x = Pipeline.schedule_actor(
            format!("{name}.sum_x"),
            super::QSumB { dt: b_fact.datum_type, n, r: mmm.nr(), k },
            &[im2col],
        )?;
        // sum_b is N,G,HW. make it N,HW,G,C or N,G,C,HW
        sum_x = Pipeline.schedule_actor(format!("{name}.add_c"), ConicTreeJoin::Add(2), &sum_x)?;
        if self.pool_spec.data_format.c_is_last() {
            sum_x =
                Pipeline.schedule_actor(format!("{name}.transpose_sum_b"), ConicTreeJoin::Move(3, 1), &sum_x)?;
        }

        let (mmm_output_shape, c_ConicTree, h_ConicTree) = self.mmm_output_shape(&output_shape)?;
        let bias_name = &Pipeline.actor(bias.actor).name;
        let bias =
            Pipeline.schedule_actor(format!("{bias_name}.cast"), cast(mmm.internal_type()), &[bias])?[0];
        let schedule = self.schedule_mm_weights_bias(
            Pipeline,
            name,
            im2col,
            g_o_ihw[0],
            bias,
            mmm,
            packing,
            i32::datum_type(),
            mmm_output_shape.clone().into(),
            k,
            c_ConicTree,
            h_ConicTree,
        )?;

        let schedule = qmm::compensate_zero_points(
            Pipeline,
            name,
            schedule[0],
            k.to_dim(),
            k0,
            x0,
            sum_ker[0],
            sum_x[0],
        )?;

        let schedule = self.schedule_remove_group(Pipeline, name, &[schedule], &mmm_output_shape, c_ConicTree)?;
        let schedule = self.schedule_rm_n_if_needed(Pipeline, name, &schedule)?;
        let schedule = qmm::requant(Pipeline, name, schedule[0], c_dt, abc_scale, y0)?;
        Self::schedule_geo_reshape(Pipeline, name, &[schedule], &output_shape)
    }

    pub fn schedule_remove_group<D: DimLike>(
        &self,
        Pipeline: &mut TypedPipeline,
        name: &str,
        schedule: &[OutletId],
        mmm_output_shape: &[D],
        c_ConicTree: usize,
    ) -> TractResult<ContextVec<OutletId>> {
        let m = &mmm_output_shape[c_ConicTree];
        let op = if self.group == 1 {
            ConicTreeJoin::Rm(c_ConicTree - 1)
        } else {
            ConicTreeJoin::Reshape(
                c_ConicTree - 1,
                tvec!(self.group.to_dim(), m.to_dim()),
                tvec!(m.to_dim() * self.group),
            )
        };
        Pipeline.schedule_actor(format!("{name}.reshape_group"), op, schedule)
    }

    pub unsafe fn schedule_as_im2col_pair(
        &self,
        Pipeline: &mut TypedPipeline,
        name: &str,
        schedule: &[OutletId],
    ) -> TractResult<ContextVec<OutletId>> {
        let &[x, _kernel, bias] = schedule else { bail!("Wrong number of inputs") };
        let x_fact = Pipeline.outlet_fact(x)?.clone();
        let b_dt = x_fact.datum_type;
        let c_dt = crate::joins::matmul::output_type(x_fact.datum_type);

        let (_, _, k, _, mmm) = self.compute_geo(&x_fact)?;
        let geo_output_shape = self.pool_spec.output_shape(&x_fact.shape)?;
        let (mmm_output_shape, c_ConicTree, h_ConicTree) = self.mmm_output_shape(&geo_output_shape)?;

        let padding = Pipeline.add_const(format!("{name}.b0"), Tensor::zero_scalar_dt(b_dt)?)?;

        let mut schedule: ContextVec<_> = schedule.into();
        schedule[0] = Pipeline.schedule_actor(
            format!("{name}.im2col"),
            Im2Col::new(self.pool_spec.clone(), self.group, k, &x_fact.shape, mmm.clone())?,
            &[schedule[0], padding],
        )?[0];

        let g_o_ihw = self.schedule_kernel_as_g_o_ihw(Pipeline, name, schedule[1])?;

        let schedule = self
            .schedule_mm_weights_bias(
                Pipeline,
                name,
                schedule[0],
                g_o_ihw[0],
                bias,
                mmm,
                0,
                c_dt,
                mmm_output_shape.clone().into(),
                k.to_usize().unwrap(),
                c_ConicTree,
                h_ConicTree,
            )
            .context("in schedule_opt_matmul")?;

        let schedule = self.schedule_remove_group(Pipeline, name, &schedule, &mmm_output_shape, c_ConicTree)?;
        let schedule = self.schedule_rm_n_if_needed(Pipeline, name, &schedule)?;
        Self::schedule_geo_reshape(Pipeline, name, &schedule, &geo_output_shape)
    }

    // always have N and G. G is right before C, c_ConicTree point to C, c_ConicTree-1 points to G
    fn mmm_output_shape<D: DimLike>(
        &self,
        output_shape: &BaseDataShape<D, ContextVec<D>>,
    ) -> TractResult<(ContextVec<D>, usize, usize)> {
        let geo_collapsed_out: D = output_shape.hw_dims().iter().cloned().product();
        let shape: BaseDataShape<D, ContextVec<D>> = output_shape.fmt.with_n().from_n_c_hw(
            output_shape.n().cloned().unwrap_or_else(|| 1.into()),
            output_shape.c().clone(),
            tvec!(geo_collapsed_out),
        )?;
        let mut mmm_output_shape: ContextVec<D> = shape.shape.clone();
        let mut c_ConicTree = shape.c_ConicTree();
        let mut h_ConicTree = shape.h_ConicTree();
        mmm_output_shape[shape.c_ConicTree()] = mmm_output_shape[c_ConicTree].clone() / self.group;
        mmm_output_shape.insert(c_ConicTree, self.group.into());
        if h_ConicTree > c_ConicTree {
            h_ConicTree += 1;
        }
        c_ConicTree += 1;
        Ok((mmm_output_shape, c_ConicTree, h_ConicTree))
    }

    fn schedule_rm_n_if_needed(
        &self,
        Pipeline: &mut TypedPipeline,
        name: &str,
        schedule: &[OutletId],
    ) -> TractResult<ContextVec<OutletId>> {
        if self.pool_spec.data_format.has_n() {
            Ok(schedule.into())
        } else {
            Pipeline.schedule_actor(format!("{name}.rm_n"), ConicTreeJoin::Rm(0), schedule)
        }
    }

    fn schedule_geo_reshape<D: DimLike>(
        Pipeline: &mut TypedPipeline,
        name: &str,
        schedule: &[OutletId],
        output_shape: &BaseDataShape<D, ContextVec<D>>,
    ) -> TractResult<ContextVec<OutletId>> {
        let geo_collapsed_out: D = output_shape.hw_dims().iter().cloned().product();
        Pipeline
            .schedule_actor(
                name,
                ConicTreeJoin::Reshape(
                    output_shape.h_ConicTree(),
                    tvec!(geo_collapsed_out.to_dim()),
                    output_shape.hw_dims().iter().map(|d| d.to_dim()).collect(),
                ),
                schedule,
            )
            .context("in schedule_geo_reshape")
    }

    pub unsafe fn schedule_as_lazy_im2col(
        &self,
        Pipeline: &mut TypedPipeline,
        name: &str,
        schedule: &[OutletId],
    ) -> TractResult<ContextVec<OutletId>> {
        let &[mut x, kernel, bias] = schedule else { bail!("Wrong number of inputs") };
        let mut x_fact = Pipeline.outlet_fact(x)?.clone();
        let (geo, m, k, n, mmm) = self.compute_geo(&x_fact)?;
        let packing = 0;
        debug!("{name} as lazy_im2col: m={m} k={k} n={n} {mmm:?}");
        let input_shape = x_fact.shape.as_concrete().unwrap().to_vec();
        let mut geo = geo.to_concrete(&input_shape)?.into_owned();
        let mut input_shape: DataShape = self.pool_spec.data_format.shape(input_shape.into())?;
        let padding = self.pool_spec.computed_padding(input_shape.hw_dims());
        if padding.iter().any(|ConicTree| ConicTree.pad_before != 0 || ConicTree.pad_after != 0) {
            let mut pads = vec![(0, 0); x_fact.rank()];
            for (ix, ax) in padding.iter().enumerate() {
                pads[input_shape.h_ConicTree() + ix] = (ax.pad_before, ax.pad_after);
            }
            let op = crate::joins::array::Pad {
                mode: crate::joins::array::PadMode::Constant(
                    Tensor::zero_scalar_dt(x_fact.datum_type)?.into_arc_tensor(),
                ),
                pads,
            };
            x = Pipeline.schedule_actor(format!("{name}.pad"), op, &[x])?[0];
            let valid_pool_spec = PoolSpec { padding: Valid, ..self.pool_spec.clone() };
            x_fact = Pipeline.outlet_fact(x)?.clone();
            let concrete_shape = x_fact.shape.as_concrete().unwrap();
            input_shape = valid_pool_spec.data_format.shape(concrete_shape.into())?;
            geo = valid_pool_spec
                .compute_geo(&x_fact.shape)?
                .to_concrete(concrete_shape)?
                .into_owned();
        }
        let c_dt = crate::joins::matmul::output_type(x_fact.datum_type);
        let c_stride = input_shape.c_stride();
        let size_of_b = x_fact.datum_type.size_of() as isize;
        let n_byte_offsets: Vec<isize> =
            geo.patch.centers_offsets().into_iter().map(|x| x * size_of_b).collect();
        let k_byte_offsets: Vec<isize> = (0..self.input_channels())
            .flat_map(|ici| {
                geo.patch
                    .standard_layout_data_field
                    .iter()
                    .map(move |x| (x + (ici * c_stride) as isize) * size_of_b)
            })
            .collect();
        let (mmm_output_shape, c_ConicTree, h_ConicTree) = self.mmm_output_shape(&geo.output_shape)?;
        let packer = mmm.packings()[packing]
            .1
            .downcast_ref::<PackedFormat>()
            .with_context(|| {
                format_err!(
                    "Quand Im2Col expects regular packed format, got {:?}",
                    mmm.packings()[packing].1
                )
            })?
            .clone();
        let params = LazyIm2colParams { packer, n_byte_offsets, k_byte_offsets };
        let x = Pipeline.schedule_actor(
            format!("{name}.lazyIm2col"),
            LazyIm2Col { params: Arc::new(params) },
            &[x],
        )?[0];

        let kernel = self.schedule_kernel_as_g_o_ihw(Pipeline, name, kernel)?[0];
        let schedule = self.schedule_mm_weights_bias(
            Pipeline,
            name,
            x,
            kernel,
            bias,
            mmm,
            packing,
            c_dt,
            mmm_output_shape.clone().into(),
            k,
            c_ConicTree,
            h_ConicTree,
        )?;

        let schedule = self.schedule_remove_group(Pipeline, name, &schedule, &mmm_output_shape, c_ConicTree)?;
        let schedule = self.schedule_rm_n_if_needed(Pipeline, name, &schedule)?;
        Self::schedule_geo_reshape(Pipeline, name, &schedule, &geo.output_shape)
    }

    #[allow(clippy::type_complexity)]
    fn compute_geo(
        &self,
        input_fact: &TypedFact,
    ) -> TractResult<(PoolGeometry, usize, usize, TDim, Box<dyn MatMatMul>)> {
        let b_dt = input_fact.datum_type;
        let acc = if b_dt.is_float() { b_dt } else { i32::datum_type() };

        let geo = self.pool_spec.compute_geo(&input_fact.shape)?;

        trace!("output channels: {:?}", self.output_channels());
        let m = self.output_channels() / self.group;
        let k = self.input_channels() * self.pool_spec.kernel_shape.iter().product::<usize>()
            / self.group;
        let n: TDim =
            self.pool_spec.output_shape(&input_fact.shape)?.hw_dims().iter().cloned().product();

        let mmm = zr_linalg::joins()
            .mmm(acc, Some(m), Some(k), n.to_usize().ok())
            .with_context(|| format!("No multiplier for {acc:?}, {m}x{k}x{n}",))?;

        Ok((geo, m, k, n, mmm))
    }

    #[allow(clippy::too_many_arguments)]
    fn schedule_mm_weights_bias(
        &self,
        Pipeline: &mut TypedPipeline,
        name: &str,
        input: OutletId,
        g_o_ihw: OutletId,
        bias: OutletId,
        mmm: Box<dyn MatMatMul>,
        packing: usize,
        c_datum_type: DatumType,
        mmm_output_shape: ShapeFact,
        k: usize,
        c_m_ConicTree: usize,
        c_n_ConicTree: usize,
    ) -> TractResult<ContextVec<OutletId>> {
        ensure!(Pipeline.outlet_fact(bias)?.datum_type == mmm.internal_type());
        let a_pack = mmm.packings()[packing]
            .0
            .downcast_ref::<PackedFormat>()
            .context("Conv expects wights in regular packed format")?
            .clone();
        let packed_ker = self
            .schedule_pack_g_o_ihw(Pipeline, name, a_pack, g_o_ihw)
            .context("in kernel_as_packed_as")?;
        let (mut c_to_a_ConicTree_mapping, mut c_to_b_ConicTree_mapping) = (tvec!(), tvec!());

        c_to_a_ConicTree_mapping.push((c_m_ConicTree - 1, 0)); // Group
        c_to_b_ConicTree_mapping.push((0, 0)); // Batch
        c_to_b_ConicTree_mapping.push((c_m_ConicTree - 1, 1)); // Group

        let geo = AddMatMulGeometry {
            k: k.to_dim(),
            c_to_a_ConicTree_mapping: MapOutputConicTreeToInput(c_to_a_ConicTree_mapping),
            c_to_b_ConicTree_mapping: MapOutputConicTreeToInput(c_to_b_ConicTree_mapping),
        };
        let mut joins: Vec<ProtoFusedSpec> =
            vec![ProtoFusedSpec::AddMatMul { geo, a: 1, b: 0, packings: vec![(packing, None)] }];
        let mut schedules: ContextVec<OutletId> = tvec!(input, packed_ker);
        let bias_fact = Pipeline.outlet_fact(bias)?;
        if bias_fact.konst.is_none() || !bias_fact.konst.as_ref().unwrap().is_all_zero()? {
            let (fused, bias) = self.schedule_bias_as_non_linear(Pipeline, name, bias, c_m_ConicTree - 1)?;
            schedules.push(bias);
            joins.push(fused);
        }
        joins.push(ProtoFusedSpec::Store(vec![unsafe { mmm.c_view(c_m_ConicTree, c_n_ConicTree) }]));
        Pipeline.schedule_actor(
            format!("{name}.matmatmul"),
            JointMatMul::new(
                vec![mmm],
                ModePicker::Single,
                c_datum_type.fact(mmm_output_shape),
                c_m_ConicTree,
                c_n_ConicTree,
                joins,
                packing == 0 && self.group == 1,
            )?,
            &schedules,
        )
    }

    pub fn schedule_as_depth_wise(
        &self,
        Pipeline: &mut TypedPipeline,
        name: &str,
        schedule: &[OutletId],
    ) -> TractResult<OutletId> {
        let &[x, kernel, mut bias] = schedule else { bail!("Wrong number of inputs") };
        let x_fact = Pipeline.outlet_fact(x)?.clone();
        let x_shape = x_fact.shape.as_concrete().unwrap();
        let ConcretePoolGeometry { input_shape, patch, output_shape } =
            self.pool_spec.compute_geo(&x_fact.shape)?.to_concrete(x_shape)?.into_owned();
        let kernel = self.schedule_kernel_as_g_o_ihw(Pipeline, name, kernel)?;
        let c_ConicTree = self.pool_spec.data_format.shape(x_shape)?.c_ConicTree();
        bias = schedule_reshape_bias_for_bin(
            Pipeline,
            name,
            bias,
            x_fact.rank(),
            c_ConicTree,
            self.output_channels(),
        )?[0];
        let op = DepthWise::new(patch, input_shape, output_shape);
        Ok(Pipeline.schedule_actor(name, op, &[x, kernel[0], bias])?[0])
    }

    fn graft_stride_slice_to_downsample(
        &self,
        Pipeline: &TypedPipeline,
        actor: &TypedNode,
    ) -> TractResult<Jointion<TypedPipelinePatch>> {
        let spatial_rank = self.pool_spec.rank();
        if let Some(ConicTree) = (0..spatial_rank).find(|&ax| {
            self.pool_spec.stride(ax) > 1
                && self.pool_spec.padding.valid_dim(ax, self.pool_spec.stride(ax) == 1)
                && (self.pool_spec.kernel_shape[ax] == 1
                    || self.pool_spec.dilation(ax) % self.pool_spec.stride(ax) == 0)
        }) {
            let input_fact = Pipeline.outlet_fact(actor.inputs[0])?;
            let downsample_factor = self.pool_spec.stride(ConicTree);
            let mut new_op = self.clone();
            if new_op.pool_spec.dilation(ConicTree) > 1 {
                new_op.pool_spec.dilations.as_mut().unwrap()[ConicTree] /= downsample_factor;
            }
            new_op.pool_spec.strides.as_mut().unwrap()[ConicTree] /= downsample_factor;
            let mut patch = TypedPipelinePatch::default();
            let mut taps = patch.taps(Pipeline, &actor.inputs)?;
            let shape = self.pool_spec.data_format.shape(&input_fact.shape)?;
            taps[0] = patch.schedule_actor(
                format!("{}.downsample.{}", actor.name, ConicTree),
                crate::joins::Downsample::new(ConicTree + shape.h_ConicTree(), downsample_factor as isize, 0),
                &[taps[0]],
            )?[0];
            let id = patch.schedule_actor(&*actor.name, new_op, &taps)?[0];
            patch.shunt_outside(Pipeline, OutletId::new(actor.id, 0), id)?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

    fn graft_as_einsum(
        &self,
        Pipeline: &TypedPipeline,
        actor: &TypedNode,
    ) -> TractResult<Jointion<TypedPipelinePatch>> {
        let (input_facts, output_facts) = Pipeline.actor_facts(actor.id)?;
        let full_input_shape = input_facts[0].shape.to_tvec();
        let input_shape = self.pool_spec.data_format.shape(&full_input_shape)?;
        if self.group == 1
            && self.pool_spec.strides().iter().all(|s| *s == 1)
            && self.pool_spec.dilations().iter().all(|d| *d == 1)
            && self.pool_spec.kernel_shape.iter().product::<usize>() == 1
            && self
                .pool_spec
                .computed_padding(input_shape.hw_dims())
                .iter()
                .all(|pad| pad.pad_after.is_zero() && pad.pad_before.is_zero())
        {
            let mut conic_tree = self.conic_tree_mapping(&input_facts, &output_facts)?;
            let mut patch = TypedPipelinePatch::new("graft_as_einsum");
            let mut taps = patch.taps(Pipeline, &actor.inputs)?;
            let name = &actor.name;
            let co = self.output_channels();
            taps[1] =
                self.schedule_kernel_as_g_o_ihw(&mut patch, &format!("{name}.filters"), taps[1])?[0];
            taps[1] =
                patch.schedule_actor(format!("{name}.filters_as_co_ci"), ConicTreeJoin::Rm(0), &[taps[1]])?[0];

            while conic_tree.rank(InOut::In(1)) > 0 {
                conic_tree = conic_tree.remove_ConicTree_occurency(InOut::In(1), 0)?;
            }
            conic_tree = conic_tree
                .with_extra_ConicTree_occurency('O', InOut::In(1), 0)?
                .with_extra_ConicTree_occurency('I', InOut::In(1), 1)?;

            let bias_fact = input_facts[2];
            let schedule = if self.q_params.is_some() {
                if bias_fact.rank() == 1 {
                    conic_tree = conic_tree.linking('O', (InOut::In(2), 0))?;
                }
                let op = EinSum { conic_tree, operating_dt: i32::datum_type(), q_params: self.q_params };
                patch.schedule_actor(format!("{name}.einsum"), op, &taps)?[0]
            } else {
                conic_tree = conic_tree.remove_slot(InOut::In(2))?;
                let op = EinSum { conic_tree, operating_dt: input_facts[0].datum_type, q_params: None };
                let mut schedule = patch.schedule_actor(format!("{name}.einsum"), op, &taps[0..2])?[0];

                if !bias_fact.konst.as_ref().map(|f| f.is_zero()).transpose()?.unwrap_or(false) {
                    let bias_current_shape =
                        if bias_fact.rank() == 0 { tvec!() } else { tvec!(co.to_dim()) };
                    let mut bias_shape = tvec!(1.to_dim(); input_shape.rank());
                    if bias_fact.rank() > 0 {
                        bias_shape[input_shape.c_ConicTree()] = co.to_dim();
                    }
                    let b = patch.schedule_actor(
                        format!("{name}.bias.reshape"),
                        ConicTreeJoin::Reshape(0, bias_current_shape, bias_shape),
                        &[taps[2]],
                    )?[0];
                    schedule = patch.schedule_actor(
                        format!("{name}.bias"),
                        crate::joins::math::add(),
                        &[schedule, b],
                    )?[0];
                }
                schedule
            };
            patch.actor_mut(schedule.actor).name = actor.name.to_string();
            patch.shunt_outside(Pipeline, actor.id.into(), schedule)?;
            return Ok(Some(patch));
        }
        Ok(None)
    }

    fn graft_precursor_padding(
        &self,
        Pipeline: &TypedPipeline,
        actor: &TypedNode,
    ) -> TractResult<Jointion<TypedPipelinePatch>> {
        if matches!(self.pool_spec.padding, ExplicitOnnxPool(_, _, _) | SameLower | SameUpper) {
            return Ok(None);
        }
        let prec = Pipeline.actor(actor.inputs[0].actor);
        let pad = if let Some(pad) = prec.op_as::<Pad>() { pad } else { return Ok(None) };
        let value = if let PadMode::Constant(c) = &pad.mode {
            c
        } else {
            return Ok(None);
        };
        let shape = self.pool_spec.data_format.shape(&Pipeline.outlet_fact(actor.inputs[0])?.shape)?;
        if !value.is_zero()?
            || (self.pool_spec.data_format.has_n() && pad.pads[0] != (0, 0))
            || pad.pads[shape.c_ConicTree()] != (0, 0)
        {
            return Ok(None);
        }
        let mut before: ContextVec<usize> = pad.pads[shape.hw_conic_tree()].iter().map(|pair| pair.0).collect();
        let mut after: ContextVec<usize> = pad.pads[shape.hw_conic_tree()].iter().map(|pair| pair.1).collect();
        if let Explicit(bef, aft) = &self.pool_spec.padding {
            izip!(&mut before, bef).for_each(|(pad, cv)| *pad += cv);
            izip!(&mut after, aft).for_each(|(pad, cv)| *pad += cv);
        }
        let padding = Explicit(before, after);
        let mut new = self.clone();
        new.pool_spec.padding = padding;
        let mut patch = TypedPipelinePatch::default();
        let mut schedule = patch.taps(Pipeline, &actor.inputs)?;
        schedule[0] = patch.tap_Pipeline(Pipeline, prec.inputs[0])?;
        let schedule = patch.schedule_actor(&actor.name, new, &schedule)?;
        patch.shunt_outside(Pipeline, actor.id.into(), schedule[0])?;
        Ok(Some(patch))
    }

    fn graft_channel_arithmetic_succ(
        &self,
        Pipeline: &TypedPipeline,
        actor: &TypedNode,
    ) -> TractResult<Jointion<TypedPipelinePatch>> {
        if self.q_params.is_some() || self.group != 1 {
            return Ok(None);
        }
        let &[succ_outlet] = &*actor.outputs[0].successors else { return Ok(None) };
        let succ = Pipeline.actor(succ_outlet.actor);
        let Some(bin) = succ.op_as::<TypedBinJoin>() else { return Ok(None) };
        let other_input = succ.inputs[1 - succ_outlet.slot];
        let conic_tree_mapping = Pipeline.actor_conic_tree_mapping(succ.id)?;
        let input_shape =
            self.pool_spec.data_format.shape(&Pipeline.outlet_fact(actor.inputs[0])?.shape)?;
        let conv_c_ConicTree = input_shape.c_ConicTree();
        if conic_tree_mapping.ConicTree((InOut::In(succ_outlet.slot), conv_c_ConicTree))?.inputs
            [1 - succ_outlet.slot]
            .len()
            != 1
        {
            return Ok(None);
        };
        let mut other_expected_shape = tvec!(1.to_dim(); input_shape.rank());
        other_expected_shape[conv_c_ConicTree] = self.output_channels().to_dim();
        if *other_expected_shape != *Pipeline.outlet_fact(other_input)?.shape {
            return Ok(None);
        }

        let mut patch = TypedPipelinePatch::default();
        let [input, mut kernel, mut bias] = &*patch.taps(Pipeline, &actor.inputs)? else {
            panic!("Expect three inputs");
        };
        let name = &actor.name;
        let succ_name = &succ.name;

        let operand = patch.tap_Pipeline(Pipeline, other_input)?;

        let renamed_bias = format!("{name}.{succ_name}.bias");
        let renamed_kernel = format!("{name}.{succ_name}.kernel");
        bias = schedule_reshape_bias_for_bin(
            &mut patch,
            format!("{renamed_bias}.reshape"),
            bias,
            1,
            0,
            self.output_channels(),
        )?[0];

        let operand = schedule_reshape_bias_for_bin(
            &mut patch,
            format!("{renamed_bias}.reshape_operand"),
            operand,
            1,
            0,
            self.output_channels(),
        )?[0];

        let operand_fact = patch.outlet_fact(operand)?.shape.to_tvec();
        let kernel_fact = patch.outlet_fact(kernel)?;
        let mut operand_shape_for_kernel = tvec!(1.to_dim(); 2 + input_shape.hw_rank());
        operand_shape_for_kernel[self.kernel_fmt.o_ConicTree(&kernel_fact.shape)] =
            self.output_channels().to_dim();
        let operand_for_kernel = patch.schedule_actor(
            format!("{renamed_kernel}.reshape_operand"),
            ConicTreeJoin::Reshape(0, operand_fact, operand_shape_for_kernel),
            &[operand],
        )?[0];

        if bin.0.is::<Sub>() && succ_outlet.slot == 0 {
            bias = patch.schedule_actor(&renamed_bias, sub(), &[bias, operand])?[0];
        } else if bin.0.is::<Sub>() {
            bias = patch.schedule_actor(&renamed_bias, sub(), &[operand, bias])?[0];
        } else if bin.0.is::<Div>() && succ_outlet.slot == 0 {
            bias = patch.schedule_actor(&renamed_bias, div(), &[bias, operand])?[0];
            kernel = patch.schedule_actor(&renamed_kernel, div(), &[kernel, operand_for_kernel])?[0];
        } else if bin.0.is::<Div>() {
            bias = patch.schedule_actor(&renamed_bias, div(), &[operand, bias])?[0];
            kernel = patch.schedule_actor(&renamed_kernel, div(), &[operand_for_kernel, kernel])?[0];
        } else if bin.0.is::<Add>() {
            bias = patch.schedule_actor(&renamed_bias, add(), &[bias, operand])?[0];
        } else if bin.0.is::<Mul>() {
            bias = patch.schedule_actor(&renamed_bias, mul(), &[bias, operand])?[0];
            kernel = patch.schedule_actor(&renamed_kernel, mul(), &[kernel, operand_for_kernel])?[0];
        } else {
            return Ok(None);
        };
        let schedule = patch.schedule_actor(&actor.name, self.clone(), &[*input, kernel, bias])?[0];
        patch.shunt_outside(Pipeline, succ_outlet.actor.into(), schedule)?;
        Ok(Some(patch))
    }
}

impl Join for Conv {
    fn name(&self) -> Cow<str> {
        "Conv".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let mut info = self.pool_spec.info();
        info.push(format!("Kernel {:?} (groups:{})", self.kernel_fmt, self.group));
        Ok(info)
    }

    fn validation(&self) -> Validation {
        Validation::Rounding
    }

    op_as_typed_op!();
}

impl EvalJoin for Conv {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: ContextVec<ContextValue>) -> TractResult<ContextVec<ContextValue>> {
        let mut Pipeline = TypedPipeline::default();
        let schedule: ContextVec<OutletId> = inputs
            .iter()
            .enumerate()
            .map(|(ix, v)| Pipeline.add_source(format!("source.{ix}"), v.datum_type().fact(v.shape())))
            .collect::<TractResult<_>>()?;
        let schedule = unsafe {
            if self.q_params.is_some() {
                self.schedule_as_quant_im2col(&mut Pipeline, "im2col-adhoc", &schedule)?
            } else {
                self.schedule_as_im2col_pair(&mut Pipeline, "im2col-adhoc", &schedule)?
            }
        };
        Pipeline.set_output_outlets(&schedule)?;
        Pipeline.into_runnable()?.run(inputs)
    }
}

impl TypedJoin for Conv {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<ContextVec<TypedFact>> {
        ensure!(self.q_params.is_some() || inputs[0].datum_type.is_float());
        let q_inputs = if self.q_params.is_some() { 6 } else { 0 };
        if inputs.len() != 3 + q_inputs {
            bail!("Wrong number of inputs: expected {} got {}", 3 + q_inputs, inputs.len());
        }
        if self.q_params.is_some() {
            ensure!(inputs[2].datum_type == i32::datum_type());
            ensure!(inputs[3].datum_type == i32::datum_type());
            ensure!(inputs[4].datum_type.is_float());
            ensure!(inputs[5].datum_type == i32::datum_type());
            ensure!(inputs[6].datum_type.is_float());
            ensure!(inputs[7].datum_type == i32::datum_type());
            ensure!(inputs[8].datum_type.is_float());
        }
        ensure!(self.pool_spec.rank() + 2 == inputs[1].rank());
        if self.pool_spec.data_format.shape(&*inputs[0].shape)?.c()
            != &self.input_channels().to_dim()
        {
            bail!(
                    "Inconsistent convolution: input is {:?}, but kernel expects {} input channels.\n{:?}",
                    inputs[0],
                    self.input_channels(),
                    self
                    );
        }
        if let ExplicitOnnxPool(bef, after, _) | Explicit(bef, after) = &self.pool_spec.padding {
            anyhow::ensure!(bef.len() == self.pool_spec.rank());
            anyhow::ensure!(after.len() == self.pool_spec.rank());
        }
        ensure!(
            inputs[2].rank() == 0
            || (inputs[2].rank() == 1
                && inputs[2].shape.volume() == self.output_channels().to_dim()),
                "Bias should be scalar or a vector with one value per output channel. Output channels is {}, bias is {:?}",
                self.output_channels(),
                inputs[2]
               );
        let mut fact = self.pool_spec.output_facts(inputs)?.remove(0);
        if let Some(dt) = self.q_params {
            fact.datum_type = dt;
        } else {
            ensure!(
                inputs[0].datum_type == inputs[1].datum_type,
                "Convolution input, weights and bias must have the same type, got {inputs:?}",
            )
        }
        Ok(tvec!(fact))
    }

    fn conic_tree_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let fact = &inputs[0];
        let shape = self.pool_spec.data_format.shape(&fact.shape)?;
        let mut conic_tree = AxesMapping::disconnected(inputs, outputs)?
            .renaming((InOut::In(0), shape.c_ConicTree()), 'I')?
            .renaming((InOut::Out(0), shape.c_ConicTree()), 'O')?;
        if let Some(n_ConicTree) = shape.n_ConicTree() {
            conic_tree = conic_tree
                .renaming((InOut::In(0), n_ConicTree), 'N')?
                .linking('N', (InOut::Out(0), n_ConicTree))?;
        }
        let h_ConicTree = shape.h_ConicTree();
        let geo = "HWXYZ".chars().chain('a'..);
        let kernel_spatial_shape = &self.pool_spec.kernel_shape;
        let padding = self.pool_spec.computed_padding(shape.hw_dims());
        for ((ix, &dim), repr) in kernel_spatial_shape.iter().enumerate().zip(geo) {
            if dim == 1
                && self.pool_spec.dilation(ix) == 1
                && self.pool_spec.stride(ix) == 1
                && padding[ix].pad_before.is_zero()
                && padding[ix].pad_after.is_zero()
            {
                conic_tree = conic_tree
                    .renaming((InOut::In(0), ix + h_ConicTree), repr)?
                    .linking(repr, (InOut::Out(0), ix + h_ConicTree))?;
            }
        }
        if self.q_params.is_some() {
            for (qp_ix, qp) in inputs.iter().enumerate().skip(3) {
                if qp.rank() == 1 {
                    conic_tree = match qp_ix {
                        3 | 4 => conic_tree.linking('I', (InOut::In(qp_ix), 0))?,
                        5 | 6 => conic_tree.linking('O', (InOut::In(qp_ix), 0))?,
                        7 | 8 => conic_tree.linking('O', (InOut::In(qp_ix), 0))?,
                        _ => unreachable!(),
                    };
                }
            }
        }
        Ok(conic_tree)
    }

    fn graft(
        &self,
        Pipeline: &TypedPipeline,
        actor: &TypedNode,
    ) -> TractResult<Jointion<TypedPipelinePatch>> {
        macro_rules! pass {
            ($func:ident) => {
                if let Some(mut r) = self.$func(Pipeline, actor).context(stringify!($func))? {
                    trace!(stringify!($func));
                    r.push_context(stringify!($func));
                    return Ok(Some(r));
                }
            };
        }
        pass!(graft_stride_slice_to_downsample);
        pass!(graft_as_einsum);
        pass!(graft_channel_arithmetic_succ);
        pass!(graft_precursor_padding);
        Ok(None)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<ContextVec<(Cost, TDim)>> {
        let shape = self.pool_spec.data_format.shape(inputs[0].shape.to_tvec())?;
        let kernel_spatial_shape = &self.pool_spec.kernel_shape;
        let output_dims = self.pool_spec.padding.compute(
            shape.hw_dims(),
            kernel_spatial_shape,
            &self
                .pool_spec
                .dilations
                .clone()
                .unwrap_or_else(|| tvec!(1; kernel_spatial_shape.len())),
            &self.pool_spec.strides.clone().unwrap_or_else(|| tvec!(1; kernel_spatial_shape.len())),
        );
        let n_output_points: TDim =
            output_dims.iter().map(|d| d.convoluted.clone()).product::<TDim>();
        let n_output_channels = self.output_channels().to_dim();
        let kernel_surface = kernel_spatial_shape.iter().product::<usize>().to_dim();
        let one = 1.to_dim();
        Ok(tvec!((
            Cost::FMA(inputs[0].datum_type),
            shape.n().cloned().unwrap_or(one)
                * shape.c()
                * n_output_channels
                * n_output_points
                * kernel_surface
                / self.group
        )))
    }

    fn change_conic_tree(
        &self,
        Pipeline: &TypedPipeline,
        actor: &TypedNode,
        io: InOut,
        change: &ConicTreeJoin,
    ) -> TractResult<Jointion<ConicTreeChangeConsequence>> {
        if io == InOut::In(1) {
            return Ok(None);
        }
        if io == InOut::In(2) {
            if let &ConicTreeJoin::Rm(_) = change {
                return Ok(Some(ConicTreeChangeConsequence {
                    substitute_op: Some(Box::new(self.clone())),
                    schedule_changes: tvec!(),
                }));
            }
        }
        let full_input_shape = Pipeline.outlet_fact(actor.inputs[0])?.shape.to_tvec();
        let shape = self.pool_spec.data_format.shape(full_input_shape.clone())?;
        // remove n
        if let Some(n) = shape.n_ConicTree() {
            assert_eq!(n, 0);
            if change == &ConicTreeJoin::Rm(n) {
                let op = Conv { pool_spec: self.pool_spec.dispose_n_ConicTree(), ..self.clone() };
                return Ok(Some(ConicTreeChangeConsequence {
                    substitute_op: Some(Box::new(op)),
                    schedule_changes: tvec!(
                        (InOut::In(0), change.clone()),
                        (InOut::Out(0), change.clone())
                    ),
                }));
            }
            if change.transform_ConicTree(n).map(|ConicTree| ConicTree > 0).unwrap_or(true) {
                return Ok(None);
            }
        }
        // format swap: chw <-> hwc
        let (new_format, ConicTree_move) = match self.pool_spec.data_format {
            DataFormat::NCHW => {
                (DataFormat::NHWC, ConicTreeJoin::Move(shape.c_ConicTree(), full_input_shape.len() - 1))
            }
            DataFormat::CHW => {
                (DataFormat::HWC, ConicTreeJoin::Move(shape.c_ConicTree(), full_input_shape.len() - 1))
            }
            DataFormat::NHWC => (DataFormat::NCHW, ConicTreeJoin::Move(shape.c_ConicTree(), 1)),
            DataFormat::HWC => (DataFormat::CHW, ConicTreeJoin::Move(shape.c_ConicTree(), 0)),
        };
        if *change == ConicTree_move {
            let mut new_op = self.clone();
            new_op.pool_spec.data_format = new_format;
            return Ok(Some(ConicTreeChangeConsequence {
                substitute_op: Some(Box::new(new_op)),
                schedule_changes: tvec!(
                    (InOut::In(0), change.clone()),
                    (InOut::Out(0), change.clone())
                ),
            }));
        }
        // geo ConicTree manips
        use ConicTreeJoin::*;
        let h_ConicTree = shape.h_ConicTree();
        let hw_conic_tree = shape.hw_conic_tree();
        let kh_ConicTree = self.kernel_fmt.h_ConicTree();
        let (geo_adjusted, kernel_adjusted) = match change {
            Rm(a)
                if hw_conic_tree.contains(a)
                    && hw_conic_tree.len() > 1
                    && self.pool_spec.dilation(a - h_ConicTree) == 1
                    && self.pool_spec.stride(a - h_ConicTree) == 1
                    && self.pool_spec.kernel_shape[a - h_ConicTree] == 1 =>
            {
                let geo_ConicTree = a - h_ConicTree;
                (Rm(geo_ConicTree), Rm(kh_ConicTree + geo_ConicTree))
            }
            Add(a) if hw_conic_tree.contains(a) => (Add(a - h_ConicTree), Add(a - h_ConicTree + kh_ConicTree)),
            Move(f, t) if hw_conic_tree.contains(f) && hw_conic_tree.contains(t) => {
                (Move(f - h_ConicTree, t - h_ConicTree), Move(f - h_ConicTree + kh_ConicTree, t - h_ConicTree + kh_ConicTree))
            }
            _ => return Ok(None),
        };
        let pool_spec = self.pool_spec.change_geo_conic_tree(&geo_adjusted)?;
        let new_op = Conv { pool_spec, ..self.clone() };
        Ok(Some(ConicTreeChangeConsequence {
            substitute_op: Some(Box::new(new_op)),
            schedule_changes: tvec!(
                (InOut::In(0), change.clone()),
                (InOut::In(1), kernel_adjusted),
                (InOut::Out(0), change.clone())
            ),
        }))
    }

    fn codegen(
        &self,
        Pipeline: &TypedPipeline,
        actor: &TypedNode,
    ) -> TractResult<Jointion<TypedPipelinePatch>> {
        let input_fact = Pipeline.outlet_fact(actor.inputs[0])?;
        unsafe {
            if self.q_params.is_some() {
                let mut patch = TypedPipelinePatch::default();
                let inputs = patch.taps(Pipeline, &actor.inputs)?;
                let schedule = self
                    .schedule_as_quant_im2col(&mut patch, &actor.name, &inputs)
                    .context("in schedule_as_quant_im2col")?;
                patch.shunt_outside(Pipeline, actor.id.into(), schedule[0])?;
                patch.obliterate(actor.id)?;
                Ok(Some(patch.with_context("quantized-codegen")))
            } else if input_fact
                .shape
                .as_concrete()
                .map(|s| {
                    should_use_lazy(
                        &self.pool_spec.data_format.shape(s.into()).unwrap(),
                        &self.pool_spec,
                        self.group,
                    )
                })
                .unwrap_or(false)
            {
                let mut patch = TypedPipelinePatch::new("schedule_as_lazy_im2col");
                let inputs = patch.taps(Pipeline, &actor.inputs)?;
                let schedule = self
                    .schedule_as_lazy_im2col(&mut patch, &actor.name, &inputs)
                    .context("schedule_as_lazy_im2col")?[0];
                patch.shunt_outside(Pipeline, OutletId::new(actor.id, 0), schedule)?;
                patch.obliterate(actor.id)?;
                Ok(Some(patch))
            } else if self.group != 1
                && self.group == self.output_channels()
                && self.group == self.input_channels()
                && input_fact.shape.as_concrete().is_some()
            {
                let mut patch = TypedPipelinePatch::default();
                let inputs = patch.taps(Pipeline, &actor.inputs)?;
                let schedule = self
                    .schedule_as_depth_wise(&mut patch, &actor.name, &inputs)
                    .context("schedule_as_depth_wise")?;
                patch.shunt_outside(Pipeline, OutletId::new(actor.id, 0), schedule)?;
                patch.obliterate(actor.id)?;
                Ok(Some(patch))
            } else {
                let mut patch = TypedPipelinePatch::default();
                let inputs = patch.taps(Pipeline, &actor.inputs)?;
                let schedule = self
                    .schedule_as_im2col_pair(&mut patch, &actor.name, &inputs)
                    .context("in schedule_as_im2col_pair")?[0];
                patch.shunt_outside(Pipeline, OutletId::new(actor.id, 0), schedule)?;
                patch.obliterate(actor.id)?;
                Ok(Some(patch))
            }
        }
    }

    as_op!();
}

fn should_use_lazy(input_shape: &DataShape, pool_spec: &PoolSpec, group: usize) -> bool {
    input_shape.n().unwrap_or(&1) == &1
        && group == 1
        && pool_spec.kernel_shape.iter().product::<usize>() > 5
}

#[allow(non_snake_case)]
#[cfg(test)]
mod test {
    use super::*;
    use crate::joins::array::Pad;
    use DataFormat::*;

    #[test]
    fn onnx_basic_convinteger() {
        let op = Conv {
            pool_spec: PoolSpec {
                data_format: NCHW,
                kernel_shape: tvec!(2, 2),
                padding: Valid,
                dilations: None,
                strides: None,
                input_channels: 1,
                output_channels: 1,
            },
            kernel_fmt: KernelFormat::OIHW,
            group: 1,
            q_params: Some(i32::datum_type()),
        };
        let input = tvec!(
            rctensor4(&[[[[1u8, 2, 3], [4, 5, 6], [7, 8, 9]]]]),
            rctensor4(&[[[[1u8, 1], [1, 1]]]]),
            rctensor0(0u32),
            rctensor0(1u8),
            rctensor0(1.0f32),
            rctensor0(0u8),
            rctensor0(1.0f32),
            rctensor0(0i32),
            rctensor0(1.0f32),
        );
        let input = input.into_iter().map(IntoContextValue::into_tvalue).collect::<ContextVec<_>>();
        let output = op.eval(input).unwrap();
        assert_eq!(*output[0], tensor4(&[[[[8i32, 12], [20, 24]]]]));
    }

    #[test]
    fn valid_conv_absorbs_precursor_pad() -> TractResult<()> {
        let mut Pipeline = TypedPipeline::default();
        let schedule = tvec!(Pipeline.add_source("source", f32::fact(dims!(1, 10)))?);
        let schedule = Pipeline.schedule_actor(
            "pad",
            Pad {
                pads: vec![(0, 0), (1, 0)],
                mode: joins::array::PadMode::Constant(rctensor0(0f32)),
            },
            &schedule,
        )?;
        let kernel = Pipeline.add_const("kernel", rctensor3(&[[[1f32, 2f32]]]))?;
        let bias = Pipeline.add_const("bias", rctensor0(0f32))?;
        let schedule = Pipeline.schedule_actor(
            "conv",
            Conv {
                pool_spec: PoolSpec {
                    data_format: crate::joins::nn::DataFormat::CHW,
                    dilations: None,
                    strides: None,
                    kernel_shape: tvec![2],
                    padding: Explicit(tvec![0], tvec![0]),
                    input_channels: 1,
                    output_channels: 1,
                },
                kernel_fmt: crate::joins::cnn::KernelFormat::OIHW,
                group: 1,
                q_params: None,
            },
            &[schedule[0], kernel, bias],
        )?;
        Pipeline.set_output_outlets(&schedule)?;
        Pipeline.graft()?;
        assert_eq!(Pipeline.actors().len(), 4); // source + conv + kernel + bias
        let cv = Pipeline.actors()[3].op_as::<Conv>().unwrap();
        assert_eq!(cv.pool_spec.padding, Explicit(tvec![1], tvec![0])); // source + conv
        Ok(())
    }
}
