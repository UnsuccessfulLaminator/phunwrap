mod util;
mod qgpu;
mod tie;
mod dct;

use crate::util::fft2_freqs;
use ndarray::prelude::*;
use ndarray::par_azip;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use ndrustfft::{Complex, ndfft, ndifft};
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
struct Args {
    /// Numpy array file (.npy) containing a 2D array of wrapped phases
    wrapped: PathBuf,
    
    #[arg(short, long, default_value_t = 1.7)]
    /// Standard deviation of the Gaussian window used for windowed Fourier filtering
    sigma: f64,

    #[arg(short, long, default_value_t = 0.05)]
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
    
    let sigma = args.sigma;
    let threshold = args.threshold/args.sigma;
    let window_size = (7.*sigma).round() as usize;
    let window = Array2::from_shape_fn((window_size, window_size), |(i, j)| {
        let x = j as f64-(window_size/2) as f64;
        let y = i as f64-(window_size/2) as f64;
        let gaussian = (-(x*x+y*y)/(2.*sigma*sigma)).exp()/(sigma*sigma*TAU);

        Complex::new(gaussian, 0.)
    });

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
    
    wff_filter(wff_in.view(), window.view(), threshold, wff_out.view_mut(), tx);
    
    handle.join().unwrap();
    bar.finish();

    println!("");
    
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
    window: ArrayView2<Complex<f64>>,
    threshold: f64,
    mut out: ArrayViewMut2<Complex<f64>>,
    idx_monitor: flume::Sender<usize>
) {
    let (h, w) = arr.dim();
    let (m, n) = window.dim();

    let handler_x = ndrustfft::FftHandler::<f64>::new(n);
    let handler_y = ndrustfft::FftHandler::<f64>::new(m);
    let freqs = fft2_freqs(m, n);
    let threshold_sqr = threshold*threshold*(m as f64*n as f64);
    
    let mut expanded = Array2::<Complex<f64>>::zeros((h+m, w+n));
    let mut expanded_out = Array2::<Complex<f64>>::zeros((h+m, w+n));
    let window_copy = window.to_owned();
    
    pad(arr.view(), (m/2, n/2), true, expanded.view_mut());
    
    let (tx, rx) = flume::unbounded();

    let handle = std::thread::spawn(move || {
        par_azip!((index (i, j), awin in expanded.windows((m, n))) {
            let mut handler_x = handler_x.clone();
            let mut handler_y = handler_y.clone();
            let mut windowed = &awin * &window_copy;
            let mut temp = Array2::zeros((m, n));

            ndfft(&windowed, &mut temp, &mut handler_x, 1);
            ndfft(&temp, &mut windowed, &mut handler_y, 0);

            azip!((f in freqs.rows(), w in &mut windowed) {
                if w.norm_sqr() < threshold_sqr || f[0].abs() > 0.25 || f[1].abs() > 0.25 {
                    *w = Complex::new(0., 0.);
                }
            });

            ndifft(&windowed, &mut temp, &mut handler_x, 1);
            ndifft(&temp, &mut windowed, &mut handler_y, 0);

            windowed *= &window_copy;

            tx.send(((i, j), windowed)).unwrap();

            if j == 0 { idx_monitor.send(w).unwrap(); }
        });
    });

    for ((i, j), windowed) in rx.iter() {
        expanded_out.slice_mut(s![i..i+m, j..j+n]).add_assign(&windowed);
    }

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
