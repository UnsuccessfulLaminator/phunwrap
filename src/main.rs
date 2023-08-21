use ndarray::prelude::*;
use ndarray::par_azip;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use ndrustfft::{
    Complex, FftHandler, DctHandler, ndfft_par, ndifft_par, nddct2_par, nddct3_par
};
use std::fs::File;
use std::f64::consts::{PI, TAU};
use std::ops::{SubAssign, AddAssign};
use anyhow;



fn main() -> anyhow::Result<()> {
    let wphase = Array2::<f64>::read_npy(File::open("./wphase_f64.npy")?)?;
    let amp = Array2::<f64>::read_npy(File::open("./amp_f64.npy")?)?;
    let mean_amp = amp.mean().unwrap();

    println!("Loaded wphase array of shape {:?}", wphase.dim());
    
    let mut uphase = Array2::<f64>::zeros(wphase.dim());
    let weights = amp.mapv(|a| if a > mean_amp { 1. } else { 0. });
    
    println!("Solving with TIE...");
    tie_solve(wphase.view(), uphase.view_mut());
    println!("Done.");

    println!("Solving with DCT...");
    dct_solve(wphase.view(), uphase.view_mut());
    println!("Done.");
    
    println!("Solving with iterative DCT...");
    dct_solve_iterative(wphase.view(), weights.view(), uphase.view_mut());
    println!("Done.");

    let wff_in = wphase.mapv(|v| Complex { re: v, im: 0. });
    let mut wff_out = Array2::<Complex<f64>>::zeros(wphase.dim());
    
    let window = Array2::<Complex<f64>>::from_shape_fn((5, 5), |(i, j)| {
        Complex {
            re: (-(i as f64-2.).powf(2.)-(j as f64-2.).powf(2.)).exp(),
            im: 0.
        }
    });

    println!("WFF test...");
    wff_filter(wff_in.view(), window.view(), wff_out.view_mut());
    println!("Done.");

    /*let mut f = BufWriter::new(File::create("data")?);
 
    for ((i, j), &u) in uphase.indexed_iter() {
        write!(f, "{j} {i} {}\n", u)?;
    }*/

    uphase.write_npy(File::create("uphase.npy")?)?;
    wff_out.mapv(|v| v.re).write_npy(File::create("wff_re.npy")?)?;
    wff_out.mapv(|v| v.im).write_npy(File::create("wff_im.npy")?)?;
    
    Ok(())
}

fn wff_filter(
    arr: ArrayView2<Complex<f64>>,
    window: ArrayView2<Complex<f64>>,
    mut out: ArrayViewMut2<Complex<f64>>
) {
    let (h, w) = arr.dim();
    let (m, n) = window.dim();

    let mut handler_x = ndrustfft::FftHandler::<f64>::new(n);
    let mut handler_y = ndrustfft::FftHandler::<f64>::new(m);
    let mut windowed = Array2::<Complex<f64>>::zeros((m, n));
    let mut temp = Array2::<Complex<f64>>::zeros((m, n));
    let mut expanded = Array2::<Complex<f64>>::zeros((h+m, w+n));
    let mut expanded_out = Array2::<Complex<f64>>::zeros((h+m, w+n));
    
    let freqs = fft2_freqs(m, n);
    let threshold = 0.6*(m as f64).hypot(n as f64);

    let (rx0, ry0) = (n/2, m/2);
    let (rx1, ry1) = (rx0+w, ry0+h);

    expanded.slice_mut(s![ry0..ry1, rx0..rx1]).assign(&arr);
    expanded.slice_mut(s![..ry0, ..rx0]).assign(&arr.slice(s![..ry0;-1, ..rx0;-1]));
    expanded.slice_mut(s![..ry0, rx0..rx1]).assign(&arr.slice(s![..ry0;-1, ..]));
    expanded.slice_mut(s![..ry0, rx1..]).assign(&arr.slice(s![..ry0;-1, rx1-n..;-1]));
    expanded.slice_mut(s![ry0..ry1, rx1..]).assign(&arr.slice(s![.., rx1-n..;-1]));
    expanded.slice_mut(s![ry1.., rx1..]).assign(&arr.slice(s![ry1-m..;-1, rx1-n..;-1]));
    expanded.slice_mut(s![ry1.., rx0..rx1]).assign(&arr.slice(s![ry1-m..;-1, ..]));
    expanded.slice_mut(s![ry1.., ..rx0]).assign(&arr.slice(s![ry1-m..;-1, ..rx0;-1]));
    expanded.slice_mut(s![ry0..ry1, ..rx0]).assign(&arr.slice(s![.., ..rx0;-1]));

    for i in 0..h {
        for j in 0..w {
            // --- Forward WFT ---
            windowed.assign(&expanded.slice(s![i..i+m, j..j+n]));
            windowed *= &window;
            
            ndfft_par(&windowed, &mut temp, &mut handler_x, 1);
            ndfft_par(&temp, &mut windowed, &mut handler_y, 0);
            
            par_azip!((f in freqs.rows(), w in &mut windowed) {
                *w *= Complex::new(0., -TAU*(f[0]*j as f64+f[1]*i as f64)).exp();
            });
            // -------------------

            par_azip!((f in freqs.rows(), w in &mut windowed) {
                if w.norm() < threshold || f[0].abs() > 0.25 || f[1].abs() > 0.25 {
                    *w = Complex { re: 0., im: 0. };
                }
            });
            
            // --- Inverse WFT ---
            ndifft_par(&windowed, &mut temp, &mut handler_x, 1);
            ndifft_par(&temp, &mut windowed, &mut handler_y, 0);
            
            windowed *= &window;

            expanded_out.slice_mut(s![i..i+m, j..j+n]).add_assign(&windowed);
            // -------------------
        }
    }

    out.assign(&expanded_out.slice(s![ry0..ry1, rx0..rx1]));
}

// Slow Picard iterative method, algorithm 2 from Ghiglia & Romero 1994
fn dct_solve_iterative(
    wrapped: ArrayView2<f64>, weights: ArrayView2<f64>,
    mut unwrapped: ArrayViewMut2<f64>
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

    for _ in 0..20 {
        p.assign(&c);

        diff(unwrapped.view(), Axis(0), temp.view_mut(), false);
        weight_diffs2(temp.view_mut(), weights.view(), Axis(0));

        p.add_assign(&temp);
        p.slice_mut(s![1.., ..]).sub_assign(&temp.slice(s![..-1, ..]));

        diff(unwrapped.view(), Axis(1), temp.view_mut(), false);
        weight_diffs2(temp.view_mut(), weights.view(), Axis(1));

        p.add_assign(&temp);
        p.slice_mut(s![.., 1..]).sub_assign(&temp.slice(s![.., ..-1]));
        
        unwrap_p(p.view_mut(), &mut handler_x, &mut handler_y, unwrapped.view_mut());
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

fn dct_solve(wrapped: ArrayView2<f64>, mut unwrapped: ArrayViewMut2<f64>) {
    let (h, w) = wrapped.dim();
    let mut handler_x = ndrustfft::DctHandler::<f64>::new(w);
    let mut handler_y = ndrustfft::DctHandler::<f64>::new(h);
    let mut temp = Array2::<f64>::zeros((h, w));
    let mut p = Array2::<f64>::zeros((h, w));

    diff(wrapped.view(), Axis(0), temp.view_mut(), true); // Delta y into temp

    p.assign(&temp);
    p.slice_mut(s![1.., ..]).sub_assign(&temp.slice(s![..-1, ..]));

    diff(wrapped.view(), Axis(1), temp.view_mut(), true); // Delta x into temp
    
    p.add_assign(&temp);
    p.slice_mut(s![.., 1..]).sub_assign(&temp.slice(s![.., ..-1]));
    
    unwrap_p(p.view_mut(), &mut handler_x, &mut handler_y, unwrapped.view_mut());
}

// Will destroy p in the process
fn unwrap_p(
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

fn diff(arr: ArrayView2<f64>, axis: Axis, out: ArrayViewMut2<f64>, wrap: bool) {
    let last_idx = out.len_of(axis)-1;
    let (mut valid, mut invalid) = out.split_at(axis, last_idx);

    valid.assign(&arr.slice_axis(axis, (1..).into()));
    valid.sub_assign(&arr.slice_axis(axis, (..-1).into()));
    
    if wrap { valid.mapv_inplace(|v| v-(v/TAU).round()); }

    invalid.fill(0.);
}

// Apply a single step of the TIE-PUA algorithm, putting the result into `unwrapped`
fn tie_solve(wrapped: ArrayView2<f64>, mut unwrapped: ArrayViewMut2<f64>) {
    let (h, w) = wrapped.dim();
    let mut handler_x = FftHandler::<f64>::new(w);
    let mut handler_y = FftHandler::<f64>::new(h);
    let mut temp1 = Array2::<Complex<f64>>::zeros((h, w));
    let mut temp2 = Array2::<Complex<f64>>::zeros((h, w));

    let exp = wrapped.mapv(|w| Complex { re: 0., im: w }.exp());
    let freqs = fft2_freqs(w, h);
    let freq_sqr_mags = freqs.map_axis(Axis(2), |f| f[0]*f[0]+f[1]*f[1]);
    
    // 2D FFT of `exp` into `temp2`
    ndfft_par(&exp, &mut temp1, &mut handler_x, 1);
    ndfft_par(&temp1, &mut temp2, &mut handler_y, 0);
    
    par_azip!((v in &mut temp2, &f in &freq_sqr_mags) { *v *= f; });
    
    // 2D IFFT of `temp2` into itself
    ndifft_par(&temp2, &mut temp1, &mut handler_x, 1);
    ndifft_par(&temp1, &mut temp2, &mut handler_y, 0);
    
    par_azip!((v in &mut temp2, &e in &exp) {
        *v /= e;
        v.re = v.im;
        v.im = 0.;
    });
    
    // 2D FFT of `temp2` into itself
    ndifft_par(&temp2, &mut temp1, &mut handler_x, 1);
    ndifft_par(&temp1, &mut temp2, &mut handler_y, 0);

    par_azip!((v in &mut temp2, &f in &freq_sqr_mags) { *v /= f; });

    temp2[[0, 0]] = Complex::new(0., 0.);
    
    // 2D IFFT of `temp2` into itself
    ndifft_par(&temp2, &mut temp1, &mut handler_x, 1);
    ndifft_par(&temp1, &mut temp2, &mut handler_y, 0);

    unwrapped.assign(&temp2.view().split_complex().re);
}

fn fft2_freqs(w: usize, h: usize) -> Array3<f64> {
    let mut x_freqs = Array::linspace(0., 1.-1./w as f64, w);
    let mut y_freqs = Array::linspace(0., 1.-1./h as f64, h);
    let mut out = Array3::<f64>::zeros((h, w, 2));

    x_freqs.mapv_inplace(|f| if f >= 0.5 { f-1. } else { f });
    y_freqs.mapv_inplace(|f| if f >= 0.5 { f-1. } else { f });

    for mut row in out.slice_mut(s![.., .., 0]).rows_mut() { row.assign(&x_freqs); }
    for mut col in out.slice_mut(s![.., .., 1]).columns_mut() { col.assign(&y_freqs); }

    out
}
