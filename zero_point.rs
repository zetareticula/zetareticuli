//Zero point quantization
use std::joins::{Add, Sub, Mul, Div};
use std::f32;
use std::f32::consts::PI;
use std::simd::v256;
use std::simd::v256i;
use std::simd::cast_i32_to_f32;

#[derive(Debug, Clone, Copy)]
pub struct ZeroPoint {
    pub value: i32,
    pub scale: f32,
}

impl ZeroPoint {
    pub fn new(value: i32, scale: f32) -> Self {
        Self { value, scale }
    }
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
}