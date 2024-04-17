import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import pandas as pd
from deepfillv2 import inpaint as dfv2inpaint

st.caption('Вебкамера может не работать в браузерах из-за отсутствия SSL сертификата.')

input_type = st.radio(
    "Выберите способ ввода изображения",
    ["Загрузить картинку", "Вебкамера"])

bg_image = False

if input_type == 'Вебкамера':
    bg_image = st.camera_input("Take a picture:", )
elif input_type == "Загрузить картинку":
    bg_image = st.file_uploader("Background image:", type=["png", "jpg"])

if bg_image:
    drawing_mode = st.selectbox(
        "Drawing tool:", ("freedraw", "line")
    )

    stroke_width = st.slider("Stroke width: ", 1, 25, 3)

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        height=400,
        width=600,
        stroke_color='white',
        background_color='#fff',
        background_image=Image.open(bg_image).resize((600, 400)),
        update_streamlit=True,
        drawing_mode=drawing_mode,
        point_display_radius=0,
        key="canvas",
    )

    dilate_mask = st.checkbox("Dilate mask", True)

    if st.button('Заполнить пустоты'):
        if not bg_image:
            st.warning("Пожалуйста, загрузите фоновое изображение.")
        else:
            def display_plot(header, data, ncols=2, nrows=2):
                fig, axes = plt.subplots(ncols=ncols, nrows=nrows)
                ax = axes.ravel()

                for idx, val in enumerate(data.keys()):
                    ax[idx].set_title(val)
                    ax[idx].imshow(data[val])
                    ax[idx].axis('off')

                fig.tight_layout()
                st.header(header)
                st.pyplot(fig)


            pil_image = Image.open(bg_image).convert('RGB')
            open_cv_image = np.array(pil_image)
            source_image = cv2.resize(open_cv_image, (600, 400))

            mask = canvas_result.image_data.copy()
            mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
            mask = cv2.resize(mask, (600, 400))

            inpaint_mask = mask

            if dilate_mask:
                inpaint_mask = cv2.dilate(mask, np.ones((5, 5)))

            dst_TELEA = cv2.inpaint(source_image, inpaint_mask, 3, cv2.INPAINT_TELEA)
            dst_NS = cv2.inpaint(source_image, inpaint_mask, 3, cv2.INPAINT_NS)

            dict_data = {
                'Source': source_image,
                'Mask': inpaint_mask,
                # https://www.olivier-augereau.com/docs/2004JGraphToolsTelea.pdf
                'OpenCV: TELEA': dst_TELEA,
                # https://www.researchgate.net/publication/3940597_Navier-Stokes_fluid_dynamics_and_image_and_video_inpainting
                'OpenCV: NS': dst_NS
            }

            display_plot('OpenCV: TELEA & NS', dict_data)

            dst_dfv2_places2 = dfv2inpaint(source_image, inpaint_mask, 'states_tf_places2')
            dst_dfv2_celebahq = dfv2inpaint(source_image, inpaint_mask, 'states_tf_celebahq')

            dict_data = {
                'Source': source_image,
                'Mask': inpaint_mask,
                'dfv2: places2 dataset': dst_dfv2_places2,
                'dfv2: celebahq dataset': dst_dfv2_celebahq
            }

            display_plot('DeepFillv2', dict_data)

            source_image = source_image.astype(np.float32)

            dst_TELEA = dst_TELEA.astype(np.float32)
            dst_NS = dst_NS.astype(np.float32)
            dst_dfv2_places2 = np.array(dst_dfv2_places2).astype(np.float32)
            dst_dfv2_celebahq = np.array(dst_dfv2_celebahq).astype(np.float32)

            telea_psnr = cv2.PSNR(source_image, dst_TELEA)
            ns_psnr = cv2.PSNR(source_image, dst_NS)
            dfv2_places2_psnr = cv2.PSNR(source_image, dst_dfv2_places2)
            dfv2_celebahq_psnr = cv2.PSNR(source_image, dst_dfv2_celebahq)

            telea_mse = mean_squared_error(source_image, dst_TELEA)
            ns_mse = mean_squared_error(source_image, dst_NS)
            dfv2_places2_mse = mean_squared_error(source_image, dst_dfv2_places2)
            dfv2_celebahq_mse = mean_squared_error(source_image, dst_dfv2_celebahq)

            telea_ssim = ssim(source_image, dst_TELEA, win_size=7,
                              data_range=source_image.max() - source_image.min(), multichannel=True, channel_axis=-1)
            ns_ssim = ssim(source_image, dst_NS, win_size=7,
                           data_range=source_image.max() - source_image.min(), multichannel=True, channel_axis=-1)
            dfv2_places2_ssim = ssim(source_image, dst_dfv2_places2, win_size=7,
                                   data_range=source_image.max() - source_image.min(),
                                   multichannel=True, channel_axis=-1)
            dfv2_celebahq_ssim = ssim(source_image, dst_dfv2_celebahq, win_size=7,
                                   data_range=source_image.max() - source_image.min(),
                                   multichannel=True, channel_axis=-1)

            st.header('Свод значений PSNR, MSE, SSIM')

            telea_psnr = round(telea_psnr, 2)
            ns_psnr = round(ns_psnr, 2)
            dfv2_places2_psnr = round(dfv2_places2_psnr, 2)
            dfv2_celebahq_psnr = round(dfv2_celebahq_psnr, 2)

            telea_mse = round(telea_mse, 2)
            ns_mse = round(ns_mse, 2)
            dfv2_places2_mse = round(dfv2_places2_mse, 2)
            dfv2_celebahq_mse = round(dfv2_celebahq_mse, 2)

            telea_ssim = round(telea_ssim, 2)
            ns_ssim = round(ns_ssim, 2)
            dfv2_places2_ssim = round(dfv2_places2_ssim, 2)
            dfv2_celebahq_ssim = round(dfv2_celebahq_ssim, 2)

            df = pd.DataFrame(
                {
                    "Algo": ["Telea", "NS", "Dfv2: places2", "Dfv2: celebahq"],
                    "PSNR": [telea_psnr, ns_psnr, dfv2_places2_psnr, dfv2_celebahq_psnr],
                    "MSE": [telea_mse, ns_mse, dfv2_places2_mse, dfv2_celebahq_mse],
                    "SSIM": [telea_ssim, ns_ssim, dfv2_places2_ssim, dfv2_celebahq_ssim],
                }
            )

            st.dataframe(df, hide_index=True, use_container_width=True)
