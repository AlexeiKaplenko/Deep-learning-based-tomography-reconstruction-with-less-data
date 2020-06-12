# Deep-learning-based-tomography-reconstruction-with-less-data-

*Abstract  
Provide a brief summary of the invention and its key points of novelty.    

We would like to introduce a framework consisting of two neural networks for 3D Tomography reconstruction (TEM, uCT) allowing us to tackle reconstruction with a limited number of projections. According to our experiments, we are able to achieve the same reconstruction quality while using from 2 to 8 times a lower number of projections. Our approach is also able to tackle the missing wedge problem and reduce artifacts produced by classical reconstruction algorithms. The proposed network architecture is end-to-end trainable and can perform 3D reconstruction without the usage of any conventional reconstruction algorithms.
  
   
 
 
*Problem Description  
Describe the problems that motivated this invention. Indicate whether there has been a long-standing need for a solution to these problems. Describe any earlier attempts to solve these problems, whether successful or not.   

There are several limitations of TEM tomography method. An electron beam is destructive for the samples and only a certain amount of dose can be applied before the structural information is lost. This leads to a relatively low signal on single micrographs. Additionally, samples typically used in TEM tomography has to be very thin (~nm) to allow the beam to pass through. On tilt, the trajectory of the electron beam increases, rendering the high-tilt angles unusable. 
Reconstruction of data with missing angular range leads to missing wedge problem, basically missing spatial information, that is required for data reconstruction. It can be mitigated by specific reconstruction techniques, but this relies on sufficient projections available – bringing us to the limited number of projections for dose distribution again. 
Employing AI for filling out missing information could lead to a reduction in artifacts caused by missing information, without compromising data quality. Enabling to distribute the dose into the reduced number of projections, would increase the signal to noise ratio for each projection, thus significantly improving resulting resolution.
  
   
 
 
*Detailed Description of the Invention  
Provide a detailed description of the invention and the manner in which it operates to solve the problems described above. Describe any unexpected favorable results achieved by the invention.   

The framework consists of two networks:
1) GAN for sinogram inpainting. It was supervised trained using 3D reconstructions of TEM tomograms (ground truth labels). We took a smaller number of projections through reconstructed volume than the projection number that was used for original reconstruction. Using Radon transformation we can create sinograms from projections, each row in sinogram corresponds to one projection taken at a particular angle. In the case of sparse projections, rows with zero values will appear in the sinogram. Thus, the problem can be formulated as image inpainting where only each n-th row is present (sparse angles projections) or rows from x to y are not present (missing wedge problem). 
Weights of the trained GAN generator are frozen and the inpainted sinogram is sent as input to the second network.

2) Multi-layer perceptron to solve the inverse of the Radon transform directly. The network has been developed to reconstruct an image from a sinogram, then using Radon transform projections through the reconstructed image are taken and obtained in this way sinogram is compared to the input sinogram. Good quality reconstructions can be obtained during the minimization of the fitting errors in a repetitive way. The reconstruction is a self-training procedure based on the physics model, instead of on training data. The algorithm showed significant improvements in the reconstruction accuracy, especially for missing-wedge and sparse angle tomography. 
