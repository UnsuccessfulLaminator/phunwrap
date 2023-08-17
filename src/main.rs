use ndarray::prelude::*;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::f32::consts::TAU;
use std::ops::SubAssign;
use anyhow;



const IN_PATH: &str = "./wphase.npy";

struct Node {
    value: i32,
    parent: (u16, u16)
}

fn main() -> anyhow::Result<()> {
    let in_file = File::open(IN_PATH)?;
    let wphase = Array2::<f32>::read_npy(in_file)?;

    println!("Loaded wphase array of shape {:?}", wphase.dim());
    
    let (h, w) = wphase.dim();
    let weights = Array2::<i32>::ones((h+1, w+1));
    let (mut v, mut z) = jump_counts(wphase.view());

    let mut nodes = Array2::<Node>::from_shape_fn((h+1, w+1), |(i, j)| {
        Node { value: 0, parent: (i as u16, j as u16) }
    });

    loop {
        let mut done_anything = false;

        for (i, j) in ndarray::indices((h+1, w)) {
            let ij0 = (i as u16, j as u16);
            let ij1 = (i as u16, j as u16+1);
            let n0_value = nodes[[i, j]].value;
            let n1_value = nodes[[i, j+1]].value;
            let dv_forward = weights[[i, j]]*sgn(v[[i, j]]-1);
            let dv_backward = -weights[[i, j]]*sgn(v[[i, j]]);
            let delta_v_forward = n0_value+dv_forward-n1_value;
            let delta_v_backward = n1_value+dv_backward-n0_value;

            if delta_v_forward > 0 { // Then add the forward edge
                let is_loop = path_contains(nodes.view(), ij0, ij1);
                
                if !is_loop {
                    nodes[[i, j+1]].parent = ij0;
                    increment_tree(&mut nodes.view_mut(), ij1, delta_v_forward);
                }
                else { perform_eo(nodes.view_mut(), ij0, ij1); }

                done_anything = true;
            }
            else if delta_v_backward > 0 { // Then add the backward edge
                let is_loop = path_contains(nodes.view(), ij1, ij0);

                if !is_loop {
                    nodes[[i, j]].parent = ij1;
                    increment_tree(&mut nodes.view_mut(), ij0, delta_v_backward);
                }
                else { perform_eo(nodes.view_mut(), ij1, ij0); }

                done_anything = true;
            }
        }
        
        for (i, j) in ndarray::indices((h, w+1)) {
            let ij0 = (i as u16, j as u16);
            let ij1 = (i as u16+1, j as u16);
            let n0_value = nodes[[i, j]].value;
            let n1_value = nodes[[i+1, j]].value;
            let dv_forward = -weights[[i, j]]*sgn(z[[i, j]]);
            let dv_backward = weights[[i, j]]*sgn(z[[i, j]]-1);
            let delta_v_forward = n0_value+dv_forward-n1_value;
            let delta_v_backward = n1_value+dv_backward-n0_value;

            if delta_v_forward > 0 { // Then add the forward edge
                let is_loop = path_contains(nodes.view(), ij0, ij1);
                
                if !is_loop {
                    nodes[[i+1, j]].parent = ij0;
                    increment_tree(&mut nodes.view_mut(), ij1, delta_v_forward);
                }
                else { perform_eo(nodes.view_mut(), ij0, ij1); }

                done_anything = true;
            }
            else if delta_v_backward > 0 { // Then add the backward edge
                let is_loop = path_contains(nodes.view(), ij1, ij0);

                if !is_loop {
                    nodes[[i, j]].parent = ij1;
                    increment_tree(&mut nodes.view_mut(), ij0, delta_v_backward);
                }
                else { perform_eo(nodes.view_mut(), ij1, ij0); }

                done_anything = true;
            }
        }

        if !done_anything { break; }
    }

    Ok(())
}

fn perform_eo(
    mut arr: ArrayViewMut2::<Node>, ij0: (u16, u16), ij1: (u16, u16),
    mut v: ArrayViewMut2::<u32>, mut z: ArrayViewMut2::<u32>
) {
    print_path(arr.view(), ij0, ij1);

    let nonroots = arr.indexed_iter()
        .filter(|((i, j), n)| { n.parent != (*i as u16, *j as u16) })
        .count();

    println!("{nonroots} non-root nodes");

    remove_tree(arr.view_mut(), ij1);
}

fn remove_tree(mut arr: ArrayViewMut2::<Node>, ij: (u16, u16)) {
    let i = ij.0 as usize;
    let j = ij.1 as usize;

    arr[[i, j]].value = 0;
    arr[[i, j]].parent = ij;

    if i > 0 && arr[[i-1, j]].parent == ij {
        remove_tree(arr.view_mut(), (ij.0-1, ij.1));
    }
    
    if j > 0 && arr[[i, j-1]].parent == ij {
        remove_tree(arr.view_mut(), (ij.0, ij.1-1));
    }
    
    if i < arr.dim().0-1 && arr[[i+1, j]].parent == ij {
        remove_tree(arr.view_mut(), (ij.0+1, ij.1));
    }
    
    if j < arr.dim().1-1 && arr[[i, j+1]].parent == ij {
        remove_tree(arr.view_mut(), (ij.0, ij.1+1));
    }
}

fn print_path(arr: ArrayView2::<Node>, ij0: (u16, u16), ij1: (u16, u16)) {
    let mut ij = ij0;

    loop {
        println!("{:?}", ij);

        if ij == ij1 { break; }
        
        ij = arr[[ij.0 as usize, ij.1 as usize]].parent;
    }
}

// Built-in i32::signum returns 0 for 0 which we do not want
fn sgn(x: i32) -> i32 {
    if x >= 0 { 1 }
    else { -1 }
}

// Increment every node in the tree whose root is ij by some value
fn increment_tree(arr: &mut ArrayViewMut2::<Node>, ij: (u16, u16), value: i32) {
    let i = ij.0 as usize;
    let j = ij.1 as usize;

    arr[[i, j]].value += value;

    if i > 0 && arr[[i-1, j]].parent == ij {
        increment_tree(arr, (ij.0-1, ij.1), value);
    }
    
    if j > 0 && arr[[i, j-1]].parent == ij {
        increment_tree(arr, (ij.0, ij.1-1), value);
    }
    
    if i < arr.dim().0-1 && arr[[i+1, j]].parent == ij {
        increment_tree(arr, (ij.0+1, ij.1), value);
    }
    
    if j < arr.dim().1-1 && arr[[i, j+1]].parent == ij {
        increment_tree(arr, (ij.0, ij.1+1), value);
    }
}

// Check if the path from ij0 to its root contains ij1
fn path_contains(arr: ArrayView2::<Node>, ij0: (u16, u16), ij1: (u16, u16)) -> bool {
    let mut ij = ij0;

    loop {
        let parent = arr[[ij.0 as usize, ij.1 as usize]].parent;

        if parent == ij { break false; }
        else { ij = parent; }

        if ij == ij1 { break true; }
    }
}

fn jump_counts(arr: ArrayView2::<f32>) -> (Array2::<i32>, Array2::<i32>) {
    let (h, w) = arr.dim();
    let mut temp = Array2::zeros((h+1, w+1));
    
    temp.slice_mut(s![1..-1, ..-1]).assign(&arr.slice(s![1.., ..]));
    temp.slice_mut(s![1..-1, ..-1]).sub_assign(&arr.slice(s![..-1, ..]));

    let v = temp.mapv(|t| (t/TAU).round() as i32);
    
    temp.slice_mut(s![.., 0]).fill(0.);
    temp.slice_mut(s![.., -1]).fill(0.);
    temp.slice_mut(s![-1, ..]).fill(0.);
    temp.slice_mut(s![..-1, 1..-1]).assign(&arr.slice(s![.., 1..]));
    temp.slice_mut(s![..-1, 1..-1]).sub_assign(&arr.slice(s![.., ..-1]));

    let z = temp.mapv(|t| (t/TAU).round() as i32);

    (v, z)
}
