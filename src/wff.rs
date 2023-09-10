use crate::util;
use ndarray::prelude::*;
use ndrustfft::{FftHandler, Complex, ndfft, ndifft};
use rayon::prelude::*;
use std::ops::AddAssign;
use flume;



pub fn filter(
    arr: ArrayView2<Complex<f64>>,
    window_size: usize,
    window_stride: usize,
    threshold: f64,
    mut out: ArrayViewMut2<Complex<f64>>,
    monitor: flume::Sender<usize>
) {
    let (h, w) = arr.dim();
    let m = window_size;

    let handler = FftHandler::<f64>::new(window_size);
    let threshold_sqr = (threshold*(m*m) as f64).powf(2.);
    let low_pass = util::fft2_freqs(m, m)
        .map_axis(Axis(2), |f| f[0].abs() < 0.25 && f[1].abs() < 0.25);
    
    let mut expanded = Array2::<Complex<f64>>::zeros((h+m, w+m));
    let mut expanded_out = Array2::<Complex<f64>>::zeros((h+m, w+m));
    
    pad(arr.view(), (m/2, m/2), true, expanded.view_mut());
    
    let (tx, rx) = flume::unbounded();

    let handle = std::thread::spawn(move || {
        (0..h).into_par_iter().step_by(window_stride).for_each(|i| {
            let mut handler = handler.clone();
            let mut afts = Array::zeros((m, w+m));
            let mut rwins = Array2::zeros((m, w+m));
            let mut windowed = Array2::zeros((m, m));
            let mut temp = Array2::zeros((m, m));
            
            ndfft(&expanded.slice(s![i..i+m, ..]), &mut afts, &mut handler, 0);

            for j in (0..w).step_by(window_stride) {
                ndfft(&afts.slice(s![.., j..j+m]), &mut windowed, &mut handler, 1);
                dlap(windowed.view_mut(), temp.view_mut());

                azip!((&pass in &low_pass, w in &mut windowed) {
                    if w.norm_sqr() < threshold_sqr || !pass {
                        *w = Complex::new(0., 0.);
                    }
                });

                dlap(windowed.view_mut(), temp.view_mut());
                ndifft(&windowed, &mut temp, &mut handler, 1);
                ndifft(&temp, &mut windowed, &mut handler, 0);

                rwins.slice_mut(s![.., j..j+m]).add_assign(&windowed);
            }

            tx.send((i, rwins)).unwrap();
        });
    });

    for (count, (i, rwins)) in rx.iter().enumerate() {
        expanded_out.slice_mut(s![i..i+m, ..]).add_assign(&rwins);
        monitor.send((count*100)/(h/window_stride)).unwrap();
    }
    
    // With stride > 1, different pixels will be affected by different numbers
    // of windows. For example with size = 7, stride = 3, an ideally infinite
    // 1D image would have pixels affected by ...3, 2, 2, 3, 2, 2, 3, 2, 2...
    // This calculates that pattern, `pat`, and divides the image to reduce
    // each pixel to the mean of the window values added to it.
    let s = window_stride as f64;
    let pat: Array1<f64> = (0..w.max(h)+m)
        .map(|i| i as f64)
        .map(|i| (i/s).floor()-((i-m as f64)/s).floor())
        .collect();

    azip!((index (i, j), v in &mut expanded_out) {
        *v /= pat[i]*pat[j];
    });
    
    // Wait for the producer thread to finish (unnecessary but good practice)
    handle.join().unwrap();
    
    assign_center(expanded_out.view(), out.view_mut());
}

fn dlap(mut arr: ArrayViewMut2<Complex<f64>>, mut temp: ArrayViewMut2<Complex<f64>>) {
    roll(arr.view(), Axis(0), false, temp.view_mut());
    arr -= &temp;
    roll(arr.view(), Axis(0), true, temp.view_mut());
    arr -= &temp;
    roll(arr.view(), Axis(1), false, temp.view_mut());
    arr -= &temp;
    roll(arr.view(), Axis(1), true, temp.view_mut());
    arr -= &temp;
}

fn roll<F: Clone>(arr: ArrayView2<F>, axis: Axis, rev: bool, mut out: ArrayViewMut2<F>) {
    let l = arr.len_of(axis);
    
    if rev {
        out.slice_axis_mut(axis, (..-1).into()).assign(&arr.slice_axis(axis, (1..).into()));
        out.index_axis_mut(axis, l-1).assign(&arr.index_axis(axis, 0));
    }
    else {
        out.slice_axis_mut(axis, (1..).into()).assign(&arr.slice_axis(axis, (..-1).into()));
        out.index_axis_mut(axis, 0).assign(&arr.index_axis(axis, l-1));
    }
}

// Copy the center of a bigger 2D array (arr) into a smaller one (out)
fn assign_center<F: Clone>(arr: ArrayView2<F>, out: ArrayViewMut2<F>) {
    let (h, w) = out.dim();
    let (m, n) = (arr.nrows()-h, arr.ncols()-w);
    let (x0, y0) = (n/2, m/2);
    let (x1, y1) = (x0+w, y0+h);
    
    arr.slice(s![y0..y1, x0..x1]).assign_to(out);
}

// Place a smaller 2D array (arr) into a bigger one (out) with the given offset,
// and optionally fill the space space around it by mirroring it outwards.
fn pad<F: Clone>(
    arr: ArrayView2<F>,
    offset: (usize, usize),
    fill_mirrored: bool,
    mut out: ArrayViewMut2<F>
) {
    let (h, w) = arr.dim();
    let (m, n) = (out.nrows()-h, out.ncols()-w);
    let (x0, y0) = offset;
    let (x1, y1) = (x0+w, y0+h);

    out.slice_mut(s![y0..y1, x0..x1]).assign(&arr);

    if fill_mirrored {
        out.slice_mut(s![..y0, ..x0]).assign(&arr.slice(s![..y0;-1, ..x0;-1]));
        out.slice_mut(s![..y0, x0..x1]).assign(&arr.slice(s![..y0;-1, ..]));
        out.slice_mut(s![..y0, x1..]).assign(&arr.slice(s![..y0;-1, x1-n..;-1]));
        out.slice_mut(s![y0..y1, x1..]).assign(&arr.slice(s![.., x1-n..;-1]));
        out.slice_mut(s![y1.., x1..]).assign(&arr.slice(s![y1-m..;-1, x1-n..;-1]));
        out.slice_mut(s![y1.., x0..x1]).assign(&arr.slice(s![y1-m..;-1, ..]));
        out.slice_mut(s![y1.., ..x0]).assign(&arr.slice(s![y1-m..;-1, ..x0;-1]));
        out.slice_mut(s![y0..y1, ..x0]).assign(&arr.slice(s![.., ..x0;-1]));
    }
}
