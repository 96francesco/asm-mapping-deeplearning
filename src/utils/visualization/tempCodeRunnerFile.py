input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
                        input_img = (input_img * 255).astype(np.uint8)