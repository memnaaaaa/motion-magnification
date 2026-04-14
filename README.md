# phase_based
PyTorch implementation of [Phase Based Motion Magnification](http://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf). It is based off of MATLAB source that can be found [here](http://people.csail.mit.edu/nwadhwa/phase-video/), the input videos can also be found at this location. The PyTorch implementation allows for easily parallization on a GPU and is even much faster than a numpy implementation even without a GPU.

The main notebook will contain a detailed hands-on overview of the Motion Magnification Algorithm. An image below will show an example of how the motion is amplified across all the video frames.


<br>


## Applying Motion Magnification

The following commandline arguments produce a motion magnification GIF: <br>
``` python motion_magnification.py -v videos/{name of file}.avi -a 25 -lo 0.2 -hi 0.25 -n luma3 -p half_octave -s 5.0 -b 4 -c 0.7 -gif True ``` 

### Arguments:
A list of the arguments is provided below. Please use the help option to find more info: 
``` python motion_magnification.py --help ```

- --video_path, -v         **&rarr;** Path to input video (**Required**)
- --phase_mag, -a          **&rarr;** Phase Magnification Factor (**Required**)
- --freq_lo, -lo           **&rarr;** Low Frequency cutoff for Temporal Filter (**Required**)
- --freq_hi, -hi           **&rarr;** High Frequency cutoff for Temporal Filter (**Required**)
- --colorspace, -n         **&rarr;** Colorspace for processing
- --pyramid_type, -p       **&rarr;** Complex Steerable Pyramid Type
- --sigma, -s              **&rarr;** Gaussian Kernel for Phase Filtering
- --attenuate, -a          **&rarr;** Attenuates Other frequencies outside of lo and hi
- --sample_frequency, -fs  **&rarr;** Overrides video sample frequency
- --reference_index, -r    **&rarr;** Index of DC reference frame
- --scale_factor, -c       **&rarr;** Factor to scale frames for processing
- --batch_size, -b         **&rarr;** CUDA batch size
- --save_directory, -d     **&rarr;** Directory for output files (default is input video directory)  
- --save_gif, -gif         **&rarr;** Saves results as a GIF
