# Image_matching_evaluation

Tools for evaluation of matches between images

Contains two main functions:
- Compute_error_fundamental_matrix
- Compute_error_homography


Both functions have the same input and output, homography should be used when the images cover a totally planar scene while fundamental matrix should be used for non planar scenes.

## Inputs
- Folder
- Image 1
- Image 2
- File with ground truth correspondence between images- It can be a txt file with common points between images or a .h file containing the transformation matrix
- File with matches to be evaluated

## Output

A results file with mean and maximum error

## Examples:
```
python compute_error_fundamental_matrix.py E:\Prueba_fm\ img1.jpg img2.jpg matches_SIFT_SIFT_img1_img2.txt matches_gt_img1_img2.txt 10


python compute_error_homography.py E:\Prueba_homo\ img1.jpg img2.jpg matches_SIFT_SIFT_img1_img2.txt matches_gt_img1_img2.txt 10
```
