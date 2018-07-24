from style.transforms import ImagePyramid, to_pil_image

def run_multiscale(st, p, a, content_layer_id, style_layer_weights, sizes, x=None, **kwargs):

    pyr = ImagePyramid(sizes)
    imgs = []
    for scaler in pyr.iterate():
        if x is not None:
            x = scaler(x)

        pscaled = scaler(p)
        ascaled = a
        
        x, _ = st.run(pscaled, ascaled, content_layer_id, style_layer_weights, x=x, **kwargs)        
        x = to_pil_image(x)
        imgs.append(x)
    return imgs