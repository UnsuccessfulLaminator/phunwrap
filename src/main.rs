mod util;
mod wff;
mod unwrap;

use ndarray::prelude::*;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use ndrustfft::Complex;
use image::io::Reader as ImageReader;
use std::fs::File;
use std::f64::consts::TAU;
use std::thread;
use std::path::{Path, PathBuf};
use indicatif::{ProgressBar, ProgressStyle};
use clap::{Parser, ValueEnum};
use anyhow::{self, Context, bail};
use flume;



#[derive(Clone)]
struct Region {
    x: usize, y: usize, w: usize, h: usize
}

impl std::str::FromStr for Region {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let i = s.find('x').ok_or("No x found separating width and height".to_string())?;
        let p1 = s[i+1..].find('+').ok_or("No +offset found".to_string())?+i+1;
        let p2 = s[p1+1..].find('+').ok_or("Only one +offset found".to_string())?+p1+1;
        
        Ok(Self {
            w: s[..i].parse().map_err(|_| "Invalid width".to_string())?,
            h: s[i+1..p1].parse().map_err(|_| "Invalid height".to_string())?,
            x: s[p1+1..p2].parse().map_err(|_| "Invalid x offset".to_string())?,
            y: s[p2+1..].parse().map_err(|_| "Invalid y offset".to_string())?
        })
    }
}

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
    
    #[arg(short, long)]
    /// Optional rectangular region to crop the input to, imagemagick-style format.
    /// E.g. 640x480+100+10 will crop to a 640x480 region starting at (100, 10)
    region: Option<Region>,
    
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
        bail!("No output files specified");
    }
    
    let wphase = load_phase_array(&args.wrapped)?;

    println!("Loaded wrapped phase array of shape {:?}", wphase.dim());
    
    let region = args.region.unwrap_or(Region {
        x: 0, y: 0, w: wphase.ncols(), h: wphase.nrows()
    });

    let Region { x, y, w, h } = region;
    let wff_in = wphase.slice(s![y..y+h, x..x+w]).mapv(|v| Complex::new(0., v).exp());
    let mut wff_out = Array2::<Complex<f64>>::zeros(wff_in.dim());

    drop(wphase);
    
    wff_filter_with_progress(
        wff_in.view(), args.window_size, args.window_stride, args.threshold,
        wff_out.view_mut(),
        "Filtering ({elapsed}) [{wide_bar:.cyan/blue}] {pos}% ({eta})",
        "#>-"
    );
    
    let wphase_filtered = wff_out.mapv(|e| e.arg());
    let quality = wff_out.mapv(|e| e.norm());
    
    drop(wff_in);
    drop(wff_out);

    if let Some(path) = args.unwrapped.as_ref() {
        let mut uphase = Array2::<f64>::zeros((h, w));

        unwrap_with_progress(
            wphase_filtered.view(), quality.view(), args.unwrap_method,
            uphase.view_mut(),
            "Unwrapping ({elapsed}) [{wide_bar:.cyan/blue}] {pos}% ({eta})",
            "#>-"
        );

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

fn run_with_progress<F: FnOnce(flume::Sender<usize>)>(
    bar_template: &str,
    bar_progress_chars: &str,
    f: F
) {
    let (tx, rx) = flume::unbounded();
    let bar = ProgressBar::new(100);
    let bar_clone = bar.clone();
    let bar_style = ProgressStyle::with_template(bar_template)
        .unwrap()
        .progress_chars(bar_progress_chars);
    
    bar.set_style(bar_style);
    
    let handle = thread::spawn(move || {
        for percentage in rx.iter() {
            bar_clone.set_position(percentage as u64);
        }
    });
    
    f(tx);
    
    handle.join().unwrap();
    bar.finish();

    println!("Done in {} seconds", bar.elapsed().as_secs_f32());
}

fn unwrap_with_progress(
    phase: ArrayView2<f64>,
    quality: ArrayView2<f64>,
    method: UnwrapMethod,
    out: ArrayViewMut2<f64>,
    bar_template: &str,
    bar_progress_chars: &str
) {
    run_with_progress(bar_template, bar_progress_chars, move |tx| {
        match method {
            UnwrapMethod::DCT => unwrap::unwrap_dct(phase, quality, 20, out, tx),
            UnwrapMethod::TIE => unwrap::unwrap_tie(phase, out),
            UnwrapMethod::QGP => unwrap::unwrap_qgp(phase, quality, out, tx)
        }
    });
}

fn wff_filter_with_progress(
    arr: ArrayView2<Complex<f64>>,
    window_size: usize,
    window_stride: usize,
    threshold: f64,
    out: ArrayViewMut2<Complex<f64>>,
    bar_template: &str,
    bar_progress_chars: &str
) {
    run_with_progress(bar_template, bar_progress_chars, move |tx| {
        wff::filter(arr, window_size, window_stride, threshold, out, tx);
    });
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
