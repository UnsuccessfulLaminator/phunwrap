mod util;
mod qgpu;
mod tie;
mod dct;

use crate::util::fft2_freqs;
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use ndrustfft::{Complex, ndfft, ndifft};
use rayon::prelude::*;
use image::io::Reader as ImageReader;
use std::fs::File;
use std::f64::consts::TAU;
use std::ops::AddAssign;
use std::thread;
use std::path::{Path, PathBuf};
use indicatif::{ProgressBar, ProgressStyle};
use clap::{Parser, ValueEnum};
use anyhow::{self, Context};
use flume;



#[derive(Clone, Copy, ValueEnum)]
enum UnwrapMethod {
    DCT, TIE, QGP
}

#[derive(Parser)]
#[command(author, about, long_about, verbatim_doc_comment)]
/// Tool for filtering & unwrapping of a wrapped phase image.
/// Author: Alc (2023)
/// --help will print a more detailed description than -h
///
/// This tool performs Windowed Fourier Filtering (threshold + lowpass) of a
/// wrapped phase image to produce a filtered image and a quality map. The WFF
/// is based on [1], but the implementation is very much my own and significantly
/// more optimised than the MATLAB code presented in that paper. In particular,
/// it is parallelised and uses a constant square window rather than a gaussian.
/// This allows it to run in seconds where the original algorithm took many minutes.
///
/// After this, the filtered image is unwrapped using one of a few possible methods:
///     dct - Discrete Cosine Transform, algorithm 2 from [2]
///     tie - Transport of Intensity Equation, single-pass algorithm from [3]
///     qgp - Quality Guided Path [4], using binary heap adjacency queue
/// For dct and qgp, the quality map is used to inform better unwrapping.
///
/// [1] Q Kemao, W Gao, and H Wang - "Windowed Fourier-filtered and quality-guided phase-unwrapping algorithm"
/// [2] DC Ghiglia and LA Romero - "Robust two-dimensional weighted and unweighted phase unwrapping that uses fast transforms and iterative methods"
/// [3] J Martinez-Carranza, K Falaggis, and T Kozacki - "Fast and accurate phase-unwrapping algorithm based on the transport of intensity equation"
/// [4] X Su and W Chen - "Reliability-guided phase unwrapping algorithm: a review"
struct Args {
    /// Numpy array file (.npy) containing a 2D array of wrapped phases
    wrapped: PathBuf,
    
    #[arg(long, visible_alias = "wsize", default_value_t = 12)]
    /// Window size in pixels used for windowed Fourier filtering
    window_size: usize,

    #[arg(long, visible_alias = "wstride", default_value_t = 1)]
    /// Shift in pixels from one window to the next
    window_stride: usize,

    #[arg(short, long, default_value_t = 0.35)]
    /// Threshold used for removing Fourier coefficients of small magnitude
    threshold: f64,

    #[arg(long, value_enum, value_name = "METHOD", default_value_t = UnwrapMethod::QGP)]
    /// Method to use for unwrapping the filtered phase
    unwrap_method: UnwrapMethod,

    #[arg(short, long, value_name = "FILE")]
    /// Output the unwrapped phase
    unwrapped: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE")]
    /// Output the image quality map
    quality: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE")]
    /// Output the filtered wrapped phase
    filtered: Option<PathBuf>
}



fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.unwrapped.is_none() && args.quality.is_none() && args.filtered.is_none() {
        eprintln!("No output files specified. Exiting.");
        return Ok(());
    }
    
    let wphase = load_phase_array(&args.wrapped)?;

    println!("Loaded wrapped phase array of shape {:?}", wphase.dim());
    
    let threshold = args.threshold;
    let wff_in = wphase.mapv(|v| Complex::new(0., v).exp());
    let mut wff_out = Array2::<Complex<f64>>::zeros(wphase.dim());

    let (tx, rx) = flume::unbounded();
    let template = "{prefix} ({elapsed}) [{wide_bar:.cyan/blue}] {pos}/{len} {msg} ({eta})";
    let bar = ProgressBar::new(wphase.len() as u64);
    let bar_clone = bar.clone();
    let bar_style = ProgressStyle::with_template(template)
        .unwrap()
        .progress_chars("#>-");
    
    bar.set_style(bar_style);
    bar.set_prefix("Filtering");
    bar.set_message("pixels");
    
    let handle = thread::spawn(move || {
        for inc in rx.iter() {
            bar_clone.inc(inc as u64);
        }
    });
    
    wff_filter(
        wff_in.view(), args.window_size, args.window_stride, threshold,
        wff_out.view_mut(), tx
    );
    
    handle.join().unwrap();
    bar.finish();

    println!("Filtered in {} seconds", bar.elapsed().as_secs_f32());
    
    let wphase_filtered = wff_out.mapv(|e| e.arg());
    let quality = wff_out.mapv(|e| e.norm());
    
    drop(wff_in);
    drop(wff_out);

    if let Some(path) = args.unwrapped.as_ref() {
        let mut uphase = Array2::<f64>::zeros(wphase.dim());
        
        let (tx, rx) = flume::unbounded();
        let bar_clone = bar.clone();
        let handle = thread::spawn(move || {
            for idx in rx.iter() {
                bar_clone.set_position(idx as u64);
            }
        });
        
        bar.reset();
        bar.set_prefix("Unwrapping");
        
        match args.unwrap_method {
            UnwrapMethod::DCT => {
                bar.set_length(20);
                bar.set_message("iterations");

                dct::unwrap_picard(
                    wphase_filtered.view(), quality.view(), 20, uphase.view_mut(), tx
                );
            },
            UnwrapMethod::TIE => {
                tie::unwrap(wphase_filtered.view(), uphase.view_mut());
            },
            UnwrapMethod::QGP => {
                qgpu::unwrap(wphase_filtered.view(), quality.view(), uphase.view_mut(), tx);
            }
        }

        handle.join().unwrap();
        bar.finish();
        
        println!("Unwrapped in {} seconds", bar.elapsed().as_secs_f32());

        uphase.write_npy(File::create(path)?)?;
    }

    if let Some(path) = args.filtered.as_ref() {
        wphase_filtered.write_npy(File::create(path)?)?;
    }

    if let Some(path) = args.quality.as_ref() {
        quality.write_npy(File::create(path)?)?;
    }
    
    Ok(())
}

fn wff_filter(
    arr: ArrayView2<Complex<f64>>,
    window_size: usize,
    window_stride: usize,
    threshold: f64,
    mut out: ArrayViewMut2<Complex<f64>>,
    idx_monitor: flume::Sender<usize>
) {
    let (h, w) = arr.dim();
    let m = window_size;

    let handler = ndrustfft::FftHandler::<f64>::new(window_size);
    let threshold_sqr = (threshold*(m*m) as f64).powf(2.);
    let low_pass = fft2_freqs(m, m)
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

                azip!((&pass in &low_pass, w in &mut windowed) {
                    if w.norm_sqr() < threshold_sqr || !pass {
                        *w = Complex::new(0., 0.);
                    }
                });

                ndifft(&windowed, &mut temp, &mut handler, 1);
                ndifft(&temp, &mut windowed, &mut handler, 0);

                rwins.slice_mut(s![.., j..j+m]).add_assign(&windowed);
            }

            tx.send((i, rwins)).unwrap();
        });
    });

    for (i, rwins) in rx.iter() {
        expanded_out.slice_mut(s![i..i+m, ..]).add_assign(&rwins);
        idx_monitor.send(w*window_stride).unwrap();
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

// Loads from a numpy .npy file or from any kind of image. If it loads from an
// image, the values are scaled to lie in the range [-pi, pi].
fn load_phase_array(path: &Path) -> anyhow::Result<Array2<f64>> {
    let name = path.display();

    Ok(if path.extension().is_some_and(|e| e == "npy") {
        let f = File::open(&path)
            .with_context(|| format!("Couldn't open {}", &name))?;

        Array2::<f64>::read_npy(f)
            .with_context(|| format!("Error reading npy data from {}", &name))?
    }
    else {
        let image = ImageReader::open(&path)
            .with_context(|| format!("Couldn't open {}", &name))?
            .decode()
            .with_context(|| format!("Error decoding image data from {}", &name))?
            .to_luma32f();
        
        let (w, h) = image.dimensions();
        let mut max = f64::MIN;

        let data = image.to_vec()
            .into_iter()
            .map(|v| v as f64)
            .inspect(|&v| if v > max { max = v; })
            .collect();
        
        let mut arr = Array2::from_shape_vec((h as usize, w as usize), data).unwrap();
        
        arr.mapv_inplace(|v| (v/max-0.5)*TAU);
        arr
    })
}
