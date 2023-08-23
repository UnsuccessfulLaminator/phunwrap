use ndarray::prelude::*;



pub fn fft2_freqs(w: usize, h: usize) -> Array3<f64> {
    let mut x_freqs = Array::linspace(0., 1.-1./w as f64, w);
    let mut y_freqs = Array::linspace(0., 1.-1./h as f64, h);
    let mut out = Array3::<f64>::zeros((h, w, 2));

    x_freqs.mapv_inplace(|f| if f >= 0.5 { f-1. } else { f });
    y_freqs.mapv_inplace(|f| if f >= 0.5 { f-1. } else { f });

    for mut row in out.slice_mut(s![.., .., 0]).rows_mut() { row.assign(&x_freqs); }
    for mut col in out.slice_mut(s![.., .., 1]).columns_mut() { col.assign(&y_freqs); }

    out
}
