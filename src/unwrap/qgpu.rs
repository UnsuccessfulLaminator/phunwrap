use std::collections::BinaryHeap;
use std::f64::consts::TAU;
use ndarray::prelude::*;
use flume;



#[derive(PartialEq)]
struct Pixel {
    ij: (usize, usize),
    q: f64
}

impl Pixel {
    fn new(ij: (usize, usize), q: f64) -> Self {
        Self { ij, q }
    }
}

impl PartialOrd for Pixel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.q.partial_cmp(&other.q)
    }
}

impl Ord for Pixel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.q.total_cmp(&other.q)
    }
}

impl Eq for Pixel {}



#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum PixelFlag {
    #[default]
    Untouched,
    Adjacent,
    Processed
}



pub fn unwrap(
    wphase: ArrayView2<f64>,
    quality: ArrayView2<f64>,
    mut uphase: ArrayViewMut2<f64>,
    monitor: flume::Sender<usize>
) {
    let (h, w) = wphase.dim();
    let (start_ij, &start_q) = quality.indexed_iter()
        .max_by(|(_, q0), (_, q1)| q0.total_cmp(q1))
        .unwrap();
    
    let adjacency_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    let mut flags = Array2::<PixelFlag>::default((h, w));
    let mut adjacent = BinaryHeap::<Pixel>::from([Pixel::new(start_ij, start_q)]);
    let mut path = Array2::<u32>::zeros((h, w));
    let mut processed = 0;
    let mut ref_ij = start_ij;
    
    flags[start_ij] = PixelFlag::Adjacent;

    while let Some(best) = adjacent.pop() {
        for (di, dj) in &adjacency_offsets {
            let (i, j) = (best.ij.0 as isize+di, best.ij.1 as isize+dj);
            
            if i >= 0 && i < h as isize && j >= 0 && j < w as isize {
                let ij = (i as usize, j as usize);

                match flags[ij] {
                    PixelFlag::Adjacent => {},
                    PixelFlag::Processed => ref_ij = ij,
                    PixelFlag::Untouched => {
                        adjacent.push(Pixel::new(ij, quality[ij]));

                        flags[ij] = PixelFlag::Adjacent;
                    }
                }
            }
        }

        uphase[best.ij] = wphase[best.ij]+TAU*((uphase[ref_ij]-wphase[best.ij])/TAU).round();
        flags[best.ij] = PixelFlag::Processed;
        processed += 1;

        path[best.ij] = processed as u32;

        if processed%100 == 0 {
            monitor.send((processed*100)/wphase.len()).unwrap();
        }
    }
}
