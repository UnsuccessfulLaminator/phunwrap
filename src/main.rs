mod util;
mod wff;
mod unwrap;

use util::Region;
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use ndrustfft::Complex;
use ndarray_linalg::LeastSquaresSvd;
use image::io::Reader as ImageReader;
use std::fs::File;
use std::f64::consts::TAU;
use std::thread;
use std::path::{Path, PathBuf};
use std::io::{BufWriter, Write};
use indicatif::{ProgressBar, ProgressStyle};
use clap::{Parser, ValueEnum};
use anyhow::{self, Context, bail};
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
/// it is parallelised and uses a cosine window rather than a Gaussian.
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
    /// Image or numpy array file containing a 2D array of wrapped phases
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

    #[arg(short, long, default_value_t = 0.7)]
    /// Threshold used for removing Fourier coefficients of small magnitude
    threshold: f64,

    #[arg(long, value_enum, value_name = "METHOD", default_value_t = UnwrapMethod::QGP)]
    /// Method to use for unwrapping the filtered phase
    unwrap_method: UnwrapMethod,

    #[arg(long, action)]
    /// Subtract a phase ramp, f(x,y) = (ax+by+c)/(dx+ey+1), from the unwrapped phase.
    /// Coefficients can be supplied with the --plane-coeffs option, otherwise they will
    /// be found by least-squares fitting to the data, and printed.
    subtract_plane: bool,

    #[arg(
        long, value_name = "COEFF", num_args = 5, allow_hyphen_values = true,
        requires = "subtract_plane"
    )]
    /// The 5 coefficients of f(x,y) = (ax+by+c)/(dx+ey+1). See --subtract-plane.
    plane_coeffs: Option<Vec<f64>>,
    
    #[arg(short, long, value_name = "FILE")]
    /// Output the unwrapped phase
    unwrapped: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE")]
    /// Output the image quality map
    quality: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE")]
    /// Output the filtered wrapped phase
    filtered: Option<PathBuf>,
    
    #[arg(short, long, value_name = "FILE")]
    /// Output data as a Comma-Separated Value file
    csv: Option<PathBuf>,

    #[arg(
        long, default_value = "xyufq", requires = "csv_output",
        value_parser = check_csv_format
    )]
    /// Specify the contents of the CSV file as a character sequence. Valid chars
    /// are x, y, u (unwrapped), f (filtered), and q (quality). For example, pass
    /// `--csv-format xyuq` and the CSV file will contain only those values and
    /// in that order.
    csv_format: String
}

fn check_csv_format(format: &str) -> Result<String, String> {
    if format.chars().all(|c| "xyufq".contains(c)) { Ok(format.to_string()) }
    else { Err("CSV format must contain only characters from xyufq".to_string()) }
}



fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    if args.unwrapped.is_none() && args.filtered.is_none()
    && args.quality.is_none() && args.csv.is_none() {
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
    
    let need_unwrap = args.unwrapped.is_some() || args.csv_format.contains('u');
    let uphase = need_unwrap.then(|| {
        let mut uphase = Array2::<f64>::zeros((h, w));

        unwrap_with_progress(
            wphase_filtered.view(), quality.view(), args.unwrap_method,
            uphase.view_mut(),
            "Unwrapping ({elapsed}) [{wide_bar:.cyan/blue}] {pos}% ({eta})",
            "#>-"
        );

        if args.subtract_plane {
            let coeffs = args.plane_coeffs.unwrap_or_else(|| {
                println!("Fit coefficients a,b,c,d,e:");

                plane_fit(uphase.view())
                    .into_iter()
                    .inspect(|c| println!("{}", c))
                    .collect()
            });

            azip!((index (i, j), u in &mut uphase) {
                let (x, y) = (j as f64, i as f64);

                *u -= (coeffs[0]*x+coeffs[1]*y+coeffs[2])/(coeffs[3]*x+coeffs[4]*y+1.);
            });
        }

        uphase
    });

    if let Some(path) = args.unwrapped.as_ref() {
        uphase.as_ref().unwrap().write_npy(File::create(path)?)?;
    }

    if let Some(path) = args.filtered.as_ref() {
        wphase_filtered.write_npy(File::create(path)?)?;
    }

    if let Some(path) = args.quality.as_ref() {
        quality.write_npy(File::create(path)?)?;
    }

    if let Some(path) = args.csv.as_ref() {
        let mut file = BufWriter::new(File::create(path)?);

        let titles: Vec<&str> = args.csv_format.chars().map(|c| match c {
            'x' => "x",
            'y' => "y",
            'f' => "filtered",
            'q' => "quality",
            'u' => "unwrapped",
            _ => panic!("Invalid CSV format char")
        }).collect();

        write!(file, "{}", titles.join(","))?;

        for ((i, j), &f) in wphase_filtered.indexed_iter() {
            if i > 0 && j > 0 { write!(file, "\n")?; }

            for (k, c) in args.csv_format.chars().enumerate() {
                if k > 0 { write!(file, ",")?; }

                write!(file, "{}", match c {
                    'x' => j as f64,
                    'y' => i as f64,
                    'f' => f,
                    'q' => quality[[i, j]],
                    'u' => uphase.as_ref().unwrap()[[i, j]],
                    _ => panic!("Invalid CSV format char")
                })?;
            }
        }
    }
    
    Ok(())
}

// Fit the equation z = (ax2+bx+c)/(dx2+ex+1) to the given array of points,
// which is a good approximation to the general equation for the phase image
// produced by a plane target. Returns [a, b, c, d, e].
fn plane_fit(arr: ArrayView2<f64>) -> Array1<f64> {
    let (h, w) = arr.dim();
    let mut matrix = Array2::<f64>::ones((w*h, 5)); // [[x, y, 1, -xz, -yz], ...]
    let vec = Array1::<f64>::from_iter(arr.iter().cloned());

    azip!((index (i, j), &z in &arr) {
        let (x, y) = (j as f64, i as f64);
        let point_idx = i*w+j;

        matrix[[point_idx, 0]] = x;
        matrix[[point_idx, 1]] = y;
        matrix[[point_idx, 3]] = -x*z;
        matrix[[point_idx, 4]] = -y*z;
    });

    matrix.least_squares(&vec)
        .expect("Could not find least squares fit for given points")
        .solution
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
