use ndarray::prelude::*;
use ndarray::par_azip;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use ndrustfft::{
    Complex, FftHandler, DctHandler, ndfft_par, ndifft_par, nddct2_par, nddct3_par
};
use std::fs::File;
use std::f64::consts::{PI, TAU};
use std::ops::{SubAssign, AddAssign};
use std::thread;
use std::sync::{mpsc, Arc, Mutex};
use std::path::PathBuf;
use indicatif::{ProgressBar, ProgressStyle};
use clap::Parser;
use anyhow;



#[derive(Parser)]
struct Args {
    /// Numpy array file (.npy) containing a 2D array of wrapped phases
    wrapped: PathBuf,

    /// Destination file for the unwrapped phases
    unwrapped: PathBuf,
    
    #[arg(short, long, default_value_t = 1.7)]
    /// Standard deviation of the Gaussian window used for windowed Fourier filtering
    sigma: f64,

    #[arg(short, long, default_value_t = 0.15)]
    /// Threshold used for removing Fourier coefficients of small magnitude
    threshold: f64,

    #[arg(short, long, value_name = "FILE")]
    /// Output the image quality map in addition to the unwrapped phase
    quality: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE")]
    /// Output the filtered wrapped phase in addition to the unwrapped phase
    filtered: Option<PathBuf>
}



fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let wphase = Array2::<f64>::read_npy(File::open(&args.wrapped)?)?;

    println!("Loaded wrapped phase array of shape {:?}", wphase.dim());
    
    let sigma = args.sigma;
    let threshold = args.threshold;
    let window_size = (7.*sigma).round() as usize;
    let window = Array2::from_shape_fn((window_size, window_size), |(i, j)| {
        let x = j as f64-(window_size/2) as f64;
        let y = i as f64-(window_size/2) as f64;
        let gaussian = (-(x*x+y*y)/(2.*sigma*sigma)).exp()/(sigma*TAU.sqrt());

        Complex::new(gaussian, 0.)
    });

    let wff_in = wphase.mapv(|v| Complex::new(0., v).exp());
    let wff_out = Arc::new(Mutex::new(Array2::<Complex<f64>>::zeros(wphase.dim())));
    let wff_out_clone = wff_out.clone();

    let (tx, rx) = mpsc::channel();
    let template = "Filtering [{elapsed}] [{wide_bar:.cyan/blue}] {pos}/{len} pixels ({eta})";
    let bar = ProgressBar::new(wphase.len() as u64);
    let bar_style = ProgressStyle::with_template(template)
        .unwrap()
        .progress_chars("#>-");

    bar.set_style(bar_style);
    
    thread::spawn(move || {
        let mut wff_out = wff_out_clone.lock().unwrap();

        wff_filter(wff_in.view(), window.view(), threshold, wff_out.view_mut(), tx);
    });

    for idx in rx.iter() {
        bar.set_position(idx as u64);
    }

    bar.finish_with_message("Done");
    
    let wff_out = Arc::try_unwrap(wff_out).unwrap().into_inner().unwrap();
    let wphase_filtered = wff_out.mapv(|e| e.arg());
    let quality = wff_out.mapv(|e| e.norm());

    drop(wff_out);
    
    let mut uphase = Array2::<f64>::zeros(wphase.dim());
    let weights = Array2::<f64>::ones(wphase.dim());
    
    println!("Unwrapping");
    dct_solve_iterative(wphase_filtered.view(), weights.view(), uphase.view_mut());
    println!("Done");
    
    uphase.write_npy(File::create(&args.unwrapped)?)?;

    if let Some(path) = args.filtered.as_ref() {
        wphase_filtered.write_npy(File::create(path)?)?;
    }

    if let Some(path) = args.quality.as_ref() {
        quality.write_npy(File::create(path)?)?;
    }
    
    Ok(())
}

/*fn qgpu(wphase: ArrayView2<f64>, quality: ArrayView2<f64>, mut uphase: ArrayViewMut2<f64>) {
    let start = quality.indexed_iter()
        .max_by(|(_, q0), (_, q1)| q0.total_cmp(q1))
        .unwrap();
    
    let adjacent = vec![(start, quality[start])];

    while adjacent.len() > 0 {
        let best = adjacent.iter()
            .max_by(|(_, q0), (_, q1)| q0.total_cmp(q1))
            .unwrap()
            .0;

        x
    }
}*/

fn wff_filter(
    arr: ArrayView2<Complex<f64>>,
    window: ArrayView2<Complex<f64>>,
    threshold: f64,
    mut out: ArrayViewMut2<Complex<f64>>,
    idx_monitor: mpsc::Sender<usize>
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
    let threshold = threshold*(m as f64*n as f64).sqrt();

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
    
    use ndrustfft::{ndfft, ndifft};

    for i in 0..h {
        for j in 0..w {
            // --- Forward WFT ---
            windowed.assign(&expanded.slice(s![i..i+m, j..j+n]));
            windowed *= &window;
            
            ndfft(&windowed, &mut temp, &mut handler_x, 1);
            ndfft(&temp, &mut windowed, &mut handler_y, 0);
            // -------------------

            azip!((f in freqs.rows(), w in &mut windowed) {
                if w.norm() < threshold || f[0].abs() > 0.25 || f[1].abs() > 0.25 {
                    *w = Complex { re: 0., im: 0. };
                }
            });
            
            // --- Inverse WFT ---
            ndifft(&windowed, &mut temp, &mut handler_x, 1);
            ndifft(&temp, &mut windowed, &mut handler_y, 0);
            
            windowed *= &window; // Yes, multiply by the window again

            expanded_out.slice_mut(s![i..i+m, j..j+n]).add_assign(&windowed);
            // -------------------
        }

        idx_monitor.send(i*w).unwrap();
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
