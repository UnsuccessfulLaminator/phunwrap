use crate::util::fft2_freqs;
use ndarray::prelude::*;
use ndarray::par_azip;
use ndrustfft::{Complex, FftHandler, ndfft_par, ndifft_par};



pub fn unwrap(wrapped: ArrayView2<f64>, mut unwrapped: ArrayViewMut2<f64>) {
    let (h, w) = wrapped.dim();
    let mut handler_x = FftHandler::<f64>::new(w);
    let mut handler_y = FftHandler::<f64>::new(h);
    let mut temp1 = Array2::<Complex<f64>>::zeros((h, w));
    let mut temp2 = Array2::<Complex<f64>>::zeros((h, w));

    let exp = wrapped.mapv(|w| Complex::new(0., w).exp());
    let freq_sqr_mags = fft2_freqs(w, h).map_axis(Axis(2), |f| f[0]*f[0]+f[1]*f[1]);
    
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
