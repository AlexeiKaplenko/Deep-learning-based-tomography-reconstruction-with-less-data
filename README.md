## Sparse sinogram in Radon Space
![Farmers Market Finder Demo](sinogram1.png)

## GAN inpainted sinogram
![Farmers Market Finder Demo](sinogram2.png)

## Original 3D reconstruction
![Farmers Market Finder Demo](original_compressed_300px.gif)

## Eight times less projections were used for reconstruction, the rest of the projections was inpainted GAN
![Farmers Market Finder Demo](eight_times_less_projections_compressed_300px.gif)


The framework consists of two networks:
1) GAN for sinogram inpainting. It was supervised trained using 3D reconstructions of TEM tomograms (ground truth labels). We took a smaller number of projections through reconstructed volume than the projection number that was used for original reconstruction. Using Radon transformation we can create sinograms from projections, each row in sinogram corresponds to one projection taken at a particular angle. In the case of sparse projections, rows with zero values will appear in the sinogram. Thus, the problem can be formulated as image inpainting where only each n-th row is present (sparse angles projections) or rows from x to y are not present (missing wedge problem). 
Weights of the trained GAN generator are frozen and the inpainted sinogram is sent as input to the second network.

2) Multi-layer perceptron to solve the inverse of the Radon transform directly. The network has been developed to reconstruct an image from a sinogram, then using Radon transform projections through the reconstructed image are taken and obtained in this way sinogram is compared to the input sinogram. Good quality reconstructions can be obtained during the minimization of the fitting errors in a repetitive way. The reconstruction is a self-training procedure based on the physics model, instead of on training data. The algorithm showed significant improvements in the reconstruction accuracy, especially for missing-wedge and sparse angle tomography. 
