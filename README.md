# Image inpainting using OpenCV & DeepFillv2 (places2, celebahq) on Streamlit

The project was created to demonstrate tools for restoring voids in an image (not generating)

_Created for my friend._

## Example

By default, the app prompts you to upload your image or use your webcam.

![Uploading an image](/assets/Снимок%20экрана%202024-04-17%20в%2015.49.42.png)

Next you create a mask by drawing on the image.

![Снимок экрана 2024-04-17 в 15.52.02](/assets/Снимок%20экрана%202024-04-17%20в%2015.52.02.png)

Click the "Fill the blanks" button and get the result.

![Снимок экрана 2024-04-17 в 15.52.23](/assets/Снимок%20экрана%202024-04-17%20в%2015.52.23.png)
![Снимок экрана 2024-04-17 в 15.52.31](/assets/Снимок%20экрана%202024-04-17%20в%2015.52.31.png)

You can also see the values of the following metrics: PNSR, MSE, SSIM.

![Снимок экрана 2024-04-17 в 15.52.39](/assets/Снимок%20экрана%202024-04-17%20в%2015.52.39.png)

## Run app

Clone the repository

```console
git clone https://github.com/Mariama4/tea_image_inpainting.git
```

Install dependencies

```console
pip install -r requirements.txt
```

Start the application

```console
streamlit run main_page.py
```

## Links

1. https://streamlit.io/

1. https://www.olivier-augereau.com/docs/2004JGraphToolsTelea.pdf

1. https://www.researchgate.net/publication/3940597_Navier-Stokes_fluid_dynamics_and_image_and_video_inpainting

1. https://github.com/nipponjo/deepfillv2-pytorch
