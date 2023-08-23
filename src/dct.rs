use std::sync::mpsc;
use std::ops::{SubAssign, AddAssign};
use std::f64::consts::{PI, TAU};
use ndarray::prelude::*;
use ndarray::par_azip;
use ndrustfft::{DctHandler, nddct2_par, nddct3_par};



// Weighted phase unwrapping. This does algorithm 2 from Ghiglia & Romero 1994,
// the Picard iterative method (slow).
pub fn unwrap_picard(
    wrapped: ArrayView2<f64>,
    weights: ArrayView2<f64>,
    iterations: usize,
    mut unwrapped: ArrayViewMut2<f64>,
    monitor: mpsc::Sender<usize>
) {
    let (h, w) = wrapped.dim();
    let mut handler_x = ndrustfft::DctHandler::<f64>::new(w);
    let mut handler_y = ndrustfft::DctHandler::<f64>::new(h);
    let mut temp = Array2::<f64>::zeros((h, w));
    let mut c = Array2::<f64>::zeros((h, w));

    diff(wrapped.view(), Axis(0), temp.view_mut(), true); // Delta y into temp
    weight_diffs(temp.view_mut(), weights.view(), Axis(0));

    c.assign(&temp);
    c.slice_mut(s![1.., ..]).sub_assign(&temp.slice(s![..-1, ..]));

    diff(wrapped.view(), Axis(1), temp.view_mut(), true); // Delta x into temp
    weight_diffs(temp.view_mut(), weights.view(), Axis(1));
    
    c.add_assign(&temp);
    c.slice_mut(s![.., 1..]).sub_assign(&temp.slice(s![.., ..-1]));

    unwrapped.fill(0.);

    let mut p = Array2::<f64>::zeros((h, w));

    for i in 0..iterations {
        p.assign(&c);

        diff(unwrapped.view(), Axis(0), temp.view_mut(), false);
        weight_diffs2(temp.view_mut(), weights.view(), Axis(0));

        p.add_assign(&temp);
        p.slice_mut(s![1.., ..]).sub_assign(&temp.slice(s![..-1, ..]));

        diff(unwrapped.view(), Axis(1), temp.view_mut(), false);
        weight_diffs2(temp.view_mut(), weights.view(), Axis(1));

        p.add_assign(&temp);
        p.slice_mut(s![.., 1..]).sub_assign(&temp.slice(s![.., ..-1]));
        
        solve_poisson(p.view_mut(), &mut handler_x, &mut handler_y, unwrapped.view_mut());

        monitor.send(i).unwrap();
    }
}

fn weight_diffs2(mut diffs: ArrayViewMut2<f64>, weights: ArrayView2<f64>, axis: Axis) {
    par_azip!((
        &w0 in weights.slice_axis(axis, (..-1).into()),
        &w1 in weights.slice_axis(axis, (1..).into()),
        d in diffs.slice_axis_mut(axis, (..-1).into())
    ) {
        *d *= 1.-w0.min(w1).powf(2.);
    });
}

fn weight_diffs(mut diffs: ArrayViewMut2<f64>, weights: ArrayView2<f64>, axis: Axis) {
    par_azip!((
        &w0 in weights.slice_axis(axis, (..-1).into()),
        &w1 in weights.slice_axis(axis, (1..).into()),
        d in diffs.slice_axis_mut(axis, (..-1).into())
    ) {
        *d *= w0.min(w1).powf(2.);
    });
}

// Unweighted phase unwrapping. This combines the 2 functions below by finding
// first the Laplacian of the wrapped phases, and then unwrapping from it.
pub fn unwrap(wrapped: ArrayView2<f64>, mut unwrapped: ArrayViewMut2<f64>) {
    let (h, w) = wrapped.dim();
    let mut handler_x = ndrustfft::DctHandler::<f64>::new(w);
    let mut handler_y = ndrustfft::DctHandler::<f64>::new(h);
    let mut p = Array2::<f64>::zeros((h, w));

    wrapped_laplacian(wrapped.view(), p.view_mut());
    solve_poisson(p.view_mut(), &mut handler_x, &mut handler_y, unwrapped.view_mut());
}

// This is a weird little function. It's like the Laplacian, but the discrete 1st
// order differences in x and y are wrapped to [-pi, pi] before the 2nd order
// differences are calculated.
fn wrapped_laplacian(input: ArrayView2<f64>, mut output: ArrayViewMut2<f64>) {
    let mut temp = Array2::<f64>::zeros(input.dim());
    
    diff(input.view(), Axis(0), temp.view_mut(), true); // Wrapped delta y into temp

    output.assign(&temp);
    output.slice_mut(s![1.., ..]).sub_assign(&temp.slice(s![..-1, ..]));

    diff(input.view(), Axis(1), temp.view_mut(), true); // Wrapped delta x into temp
    
    output.add_assign(&temp);
    output.slice_mut(s![.., 1..]).sub_assign(&temp.slice(s![.., ..-1]));
}

// This does algorithm 1 of Ghiglia & Romero 1994, unweighted phase unwrapping.
// P is the discrete wrapped Laplacian of the wrapped phases. Its contents will
// be destroyed by this function.
fn solve_poisson(
    mut p: ArrayViewMut2<f64>,
    handler_x: &mut DctHandler<f64>, handler_y: &mut DctHandler<f64>,
    mut out: ArrayViewMut2<f64>
) {
    let (h, w) = p.dim();

    // 2d dct of `p` into itself using `out` as temp storage
    nddct2_par(&p, &mut out, handler_x, 1);
    nddct2_par(&out, &mut p, handler_y, 0);
    
    let icos: Vec<f64> = (0..h).map(|i| (i as f64*PI/h as f64).cos()).collect();
    let jcos: Vec<f64> = (0..w).map(|j| (j as f64*PI/w as f64).cos()).collect();

    par_azip!((index (i, j), out in &mut out, &p in &p) {
        *out = 0.5*p/(icos[i]+jcos[j]-2.);
    });

    out[[0, 0]] = p[[0, 0]]; // after this, `out` is phi-hat i,j
    
    // 2d inverse dct of `out` into itself using `p` as temp storage
    nddct3_par(&out, &mut p, handler_x, 1);
    nddct3_par(&p, &mut out, handler_y, 0);
}

// Find the discrete 1st order differences along one of the axes of a 2D array.
// Optionally these differences may be wrapped to the range [-pi, pi].
fn diff(arr: ArrayView2<f64>, axis: Axis, out: ArrayViewMut2<f64>, wrap: bool) {
    let last_idx = out.len_of(axis)-1;
    let (mut valid, mut invalid) = out.split_at(axis, last_idx);

    valid.assign(&arr.slice_axis(axis, (1..).into()));
    valid.sub_assign(&arr.slice_axis(axis, (..-1).into()));
    
    if wrap { valid.mapv_inplace(|v| v-TAU*(v/TAU).round()); }

    invalid.fill(0.);
}
