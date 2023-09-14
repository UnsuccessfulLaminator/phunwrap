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



#[derive(Clone)]
pub struct Region {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize
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
