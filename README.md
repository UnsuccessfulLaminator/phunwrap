# Phunwrap

This is a command-line tool to perform efficient filtering and unwrapping of wrapped phase
images. It is written entirely in Rust, with the goal of being a single cohesive tool as an
alternative to the various scattered MATLAB implementations of algorithms from the scientific
literature.

## Basic usage

Run the tool with a wrapped phase input (supports common image formats & numpy .npy) and any
specified output (numpy .npy), and it will proceed with default settings.

wrapped.png
![input: wrapped.png](images/cropped_pyramid_w.png)

```
> phunwrap wrapped.png -u unwrapped.npy
Loaded wrapped phase array of shape (555, 750)
Filtering (1s) [#####################################################] 100% (0s)
Done in 1.0258435 seconds
Unwrapping (0s) [####################################################] 100% (0s)
Done in 0.170988 seconds
```

unwrapped.npy
![output: unwrapped.npy](images/cropped_pyramid_u_default.png)

It's difficult to tell whether the unwrap is good, because it ranges over such depth that we
can't see any detail. Let's try again, but this time with the option to remove an underlying
ramp.

```
> phunwrap wrapped.png -u unwrapped.npy --subtract-plane
...
```

unwrapped.npy
![output: unwrapped.npy](images/cropped_pyramid_u_subplane.png)

Nice, but there's clearly a lot of noise in the original image that has been carried through
to the output. Let's increase the size of the filtering window, and while we're at it we can
increase the window stride too, so the filtering will run much more quickly.

```
> phunwrap wrapped.png -u unwrapped.npy --subtract-plane --wsize 32 --wstride 4
```

unwrapped.npy
![output: unwrapped.npy](images/cropped_pyramid_u_bigwin.png)

Much better.

The tool has overview help text (-h) and detailed help text (--help) which should enable you
to use it without further reference. The overview is reproduced here:

```
Usage: phunwrap [OPTIONS] <WRAPPED>

Arguments:
  <WRAPPED>  Image or numpy array file containing a 2D array of wrapped phases

Options:
  -r, --region <REGION>
          Optional rectangular region to crop the input to, imagemagick-style format. E.g. 640x480+100+10 will crop to a 640x480 region starting at (100, 10)
      --window-size <WINDOW_SIZE>
          Window size in pixels used for windowed Fourier filtering [default: 12] [aliases: wsize]
      --window-stride <WINDOW_STRIDE>
          Shift in pixels from one window to the next [default: 1] [aliases: wstride]
  -t, --threshold <THRESHOLD>
          Threshold used for removing Fourier coefficients of small magnitude [default: 0.7]
      --unwrap-method <METHOD>
          Method to use for unwrapping the filtered phase [default: qgp] [possible values: dct, tie, qgp]
      --subtract-plane
          Subtract a phase ramp, f(x,y) = (ax+by+c)/(dx+ey+1), from the unwrapped phase. Coefficients can be supplied with the --plane-coeffs option, otherwise they will be found by least-squares fitting to the data, and printed
      --plane-coeffs <COEFF> <COEFF> <COEFF> <COEFF> <COEFF>
          The 5 coefficients of f(x,y) = (ax+by+c)/(dx+ey+1). See --subtract-plane
  -u, --unwrapped <FILE>
          Output the unwrapped phase
  -q, --quality <FILE>
          Output the image quality map
  -f, --filtered <FILE>
          Output the filtered wrapped phase
  -c, --csv <FILE>
          Output data as a Comma-Separated Value file
      --csv-format <CSV_FORMAT>
          Specify the contents of the CSV file as a character sequence. Valid chars are x, y, u (unwrapped), f (filtered), and q (quality). For example, pass `--csv-format xyuq` and the CSV file will contain only those values and in that order [default: xyufq]
  -h, --help
          Print help (see more with '--help')
```
