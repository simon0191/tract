use crate::internal::*;
use ndarray::*;

/// For every coordinate of `indices`, reads `data` at that same coordinate with
/// `axis` replaced by the index value found there. The output has `indices`'
/// shape. A negative index counts from the end of `axis`. `data` and `indices`
/// must have the same rank; off `axis`, `indices` may be smaller than `data`.
#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct GatherElements {
    pub axis: usize,
}

impl Op for GatherElements {
    fn name(&self) -> StaticName {
        "GatherElements".into()
    }

    op_as_typed_op!();
}

impl GatherElements {
    /// Gathers when `axis` is the last one and both operands are contiguous with
    /// identical leading dimensions. Every leading coordinate then addresses one
    /// whole row of both operands, so the op is a flat per-row lookup and none of
    /// the per-element dynamic-rank stride arithmetic of the generic path is
    /// needed. `Ok(None)` means the operands do not qualify and the caller must
    /// use the generic path.
    ///
    /// Index resolution is identical to the generic path, but an out-of-range
    /// index is reported as an error instead of panicking inside `ndarray`.
    fn eval_contiguous_last_axis<T: Datum>(
        &self,
        data: &ArrayViewD<T>,
        indices: &ArrayViewD<i64>,
    ) -> TractResult<Option<ArrayD<T>>> {
        let rank = data.ndim();
        if self.axis + 1 != rank
            || indices.ndim() != rank
            || data.shape()[..rank - 1] != indices.shape()[..rank - 1]
        {
            return Ok(None);
        }
        let (Some(data_slice), Some(index_slice)) = (data.as_slice(), indices.as_slice()) else {
            return Ok(None);
        };
        let row_len = data.shape()[rank - 1];
        let gathered_len = indices.shape()[rank - 1];
        let rows: usize = indices.shape()[..rank - 1].iter().product();
        let mut output = Vec::with_capacity(indices.len());
        for row in 0..rows {
            let data_row = &data_slice[row * row_len..][..row_len];
            for &index in &index_slice[row * gathered_len..][..gathered_len] {
                let resolved = if index < 0 { index + row_len as i64 } else { index };
                let value = usize::try_from(resolved)
                    .ok()
                    .and_then(|resolved| data_row.get(resolved))
                    .with_context(|| {
                        format!("Invalid GatherElements index {index} on axis of len {row_len}")
                    })?;
                output.push(value.clone());
            }
        }
        Ok(Some(ArrayD::from_shape_vec(indices.shape(), output)?))
    }

    unsafe fn eval_t<T: Datum>(
        &self,
        data: TValue,
        indices: &ArrayViewD<i64>,
    ) -> TractResult<TValue> {
        let data_plain = data.try_as_plain()?;
        let data_view = unsafe { data_plain.to_array_view_unchecked::<T>() };
        let output = match self.eval_contiguous_last_axis::<T>(&data_view, indices)? {
            Some(output) => output,
            None => ArrayD::<T>::from_shape_fn(indices.shape(), |mut coords| {
                let index = indices[&coords];
                coords[self.axis] =
                    if index < 0 { index + data_view.shape()[self.axis] as i64 } else { index }
                        as usize;
                data_view[coords].clone()
            }),
        };
        let mut tensor = output.into_tensor();
        unsafe { tensor.set_datum_type(data.datum_type()) };
        Ok(tensor.into_tvalue())
    }
}

impl TypedOp for GatherElements {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(&*inputs[1].shape)))
    }
}

impl EvalOp for GatherElements {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (data, indices) = args_2!(inputs);
        let indices = indices.cast_to::<i64>()?;
        let indices = indices.to_plain_array_view::<i64>()?;
        unsafe {
            Ok(tvec!(dispatch_datum_by_size!(Self::eval_t(data.datum_type())(
                self, data, &indices
            ))?))
        }
    }
}
