# pca-compression
Trying out compression of images using PCA.

PCA is a technique from machine learning where high dimensional data are mapped into low dimensional space, 
preserving as much information as possible.

This can be used for image compression. E.g., if we have 128x128 greyscale image (256 color possibilities 
per pixel), we can think about each row of (128) pixels as being one 128 dimensional vector. Using PCA, we can 
find a new space with fewer dimensions, determined by top principal components (which are mutually 
orthogonal one to another). Once we have these, we transform each vector to this new, more compact space, 
effectively performing compression of the image. 

In order to see the image again, we need to decompress the new low-dimensional data. This is done by performing
reverse transformation on the new data. Depending on the number of principal components we chose (the degree of 
compression is inversely proportional to number of PCs) we may get worse quality then the original image.

## Example

Original image

![Original image](https://raw.githubusercontent.com/silverfield/pca-compression/master/others/small_lena.png "Original image")

Keeping 70% of PCs (1.42/1 compression)
![Keeping 70% of PCs](https://raw.githubusercontent.com/silverfield/pca-compression/master/results/comp_70perc.png "Keeping 70% of PCs")

Keeping 40% of PCs (2.5/1 compression)
![Keeping 40% of PCs](https://raw.githubusercontent.com/silverfield/pca-compression/master/results/comp_40perc.png "Keeping 40% of PCs")

Keeping 10% of PCs (10/1 compression)
![Keeping 10% of PCs](https://raw.githubusercontent.com/silverfield/pca-compression/master/results/comp_10perc.png "Keeping 10% of PCs")
