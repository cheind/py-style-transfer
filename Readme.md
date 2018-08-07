## Neural style transfer

**py-style-transfer** implements image style transfer as proposed by [1]. Given an artistic image and a content image, the method iteratively generates an image that is similar to the content but drawn in the desired artistic style. While the method is not real-time capable, it is the most flexible approach, not requiring any pre-training expect for a readily available pre-trained convolutional architecture such as VGG. While this implementation is based on [1] we also incorporate ideas from [2,3].

In this work we extend the approach to two more use-cases
 - **Seamless mode** generates tiles that can be stacked vertically/horizontally without visual seams.
 - **Tiled mode** allows generation of very large images that would otherwise not fit into memory. Like this [8192x8192 10Mb/Jpeg](https://drive.google.com/file/d/1modc1iGmTUx4LGbh-ZCTZsXxjujf-eHQ/view?usp=sharing) pure Picasso artistic style image.

See the interactive [StyleTransfer.ipynb](StyleTransfer.ipynb) notebook for usage and examples.

 ### References

 [1] Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).</br>
 [2] Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution." European Conference on Computer Vision. Springer, Cham, 2016.</br>
 [3] Gatys, Leon A., et al. "Controlling perceptual factors in neural style transfer." IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017.

 ### License

 MIT License