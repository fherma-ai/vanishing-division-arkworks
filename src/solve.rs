//! Division by X^H - 1, folded rather than long-divided.
//!
//! X^H is 1 modulo the divisor, so every coefficient at or above H falls to
//! the one H places below it and contributes that much to the quotient. One
//! pass downwards, linear, and no polynomial division at all.
use ark_bn254::Fr;
use ark_ff::{BigInt, PrimeField};

use crate::fherma::{Inputs, Outputs, Point, Tensor};

/// The `i`-th row of a limb tensor as a field element.
///
/// `from_bigint` refuses a value at or above the modulus. The generator
/// reduces before it writes, so a refusal here means the case is not the one
/// this specification describes — better said than folded away.
fn fe(data: &[u64], i: usize, limbs: usize) -> Fr {
    let mut words = [0u64; 4];
    words[..limbs].copy_from_slice(&data[i * limbs..(i + 1) * limbs]);
    Fr::from_bigint(BigInt::new(words)).expect("a field element below the modulus")
}

fn elements(data: &[u64], limbs: usize) -> Vec<Fr> {
    (0..data.len() / limbs).map(|i| fe(data, i, limbs)).collect()
}

fn rows(values: &[Fr], limbs: usize) -> Tensor<u64> {
    let mut data = Vec::with_capacity(values.len() * limbs);
    for v in values {
        data.extend_from_slice(&v.into_bigint().0[..limbs]);
    }
    Tensor { shape: vec![values.len() as i64, limbs as i64], data }
}

fn row(value: Fr, limbs: usize) -> Tensor<u64> {
    Tensor { shape: vec![limbs as i64], data: value.into_bigint().0[..limbs].to_vec() }
}

#[derive(Default)]
pub struct State { limbs: usize, h: usize }

pub fn init(p: &Point) -> State { State { limbs: p.L as usize, h: p.H as usize } }

pub fn run(state: &mut State, inp: &Inputs) -> Outputs {
    let (limbs, h) = (state.limbs, state.h);
    let mut a = elements(&inp.a.data, limbs);
    let n = a.len();
    let mut quotient = vec![Fr::from(0u64); n];

    for i in (h..n).rev() {
        let c = a[i];
        quotient[i - h] += c;
        a[i - h] += c;
        a[i] = Fr::from(0u64);
    }

    Outputs { quotient: rows(&quotient, limbs), remainder: rows(&a[..h], limbs) }
}

pub fn free(state: State) { drop(state); }
