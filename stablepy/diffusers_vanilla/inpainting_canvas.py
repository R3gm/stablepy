
# =====================================
# Inpainting canvas
# =====================================
canvas_html = """
<style>
.button {
  background-color: #4CAF50;
  border: none;
  color: white;
  padding: 15px 32px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  margin: 4px 2px;
  cursor: pointer;
}
</style>
<canvas1 width=%d height=%d>
</canvas1>
<canvas width=%d height=%d>
</canvas>

<button>Finish</button>
<script>
var canvas = document.querySelector('canvas')
var ctx = canvas.getContext('2d')

var canvas1 = document.querySelector('canvas1')
var ctx1 = canvas.getContext('2d')


ctx.strokeStyle = 'red';

var img = new Image();
img.src = "data:image/%s;charset=utf-8;base64,%s";
console.log(img)
img.onload = function() {
  ctx1.drawImage(img, 0, 0);
};
img.crossOrigin = 'Anonymous';

ctx.clearRect(0, 0, canvas.width, canvas.height);

ctx.lineWidth = %d
var button = document.querySelector('button')
var mouse = {x: 0, y: 0}

canvas.addEventListener('mousemove', function(e) {
  mouse.x = e.pageX - this.offsetLeft
  mouse.y = e.pageY - this.offsetTop
})
canvas.onmousedown = ()=>{
  ctx.beginPath()
  ctx.moveTo(mouse.x, mouse.y)
  canvas.addEventListener('mousemove', onPaint)
}
canvas.onmouseup = ()=>{
  canvas.removeEventListener('mousemove', onPaint)
}
var onPaint = ()=>{
  ctx.lineTo(mouse.x, mouse.y)
  ctx.stroke()
}

var data = new Promise(resolve=>{
  button.onclick = ()=>{
    resolve(canvas.toDataURL('image/png'))
  }
})
</script>
"""

import base64, os
# from google.colab.output import eval_js
from base64 import b64decode
import matplotlib.pyplot as plt
import numpy as np
from shutil import copyfile
import shutil
import matplotlib.pyplot as plt

# def draw(imgm, filename='drawing.png', w=400, h=200, line_width=1):
#   display(HTML(canvas_html % (w, h, w,h, filename.split('.')[-1], imgm, line_width)))
#   data = eval_js("data")
#   binary = b64decode(data.split(',')[1])
#   with open(filename, 'wb') as f:
#     f.write(binary)


### second option ###
import base64
import os
from IPython.display import display, HTML
import ipywidgets as widgets
from PIL import Image
import io
def draw(imgm, filename='drawing.png', w=400, h=200, line_width=1):
    # Create an HTML canvas widget
    canvas = widgets.HTML(f'<canvas width="{w}" height="{h}" id="canvas"></canvas>')
    
    # Create a button to trigger the save action
    save_button = widgets.Button(description="Save Image")
    output = widgets.Output()
    
    def on_button_click(b):
        # Capture the canvas content as an image
        with output:
            canvas_data = canvas.get_state()['model_data']
            binary_data = base64.b64decode(canvas_data.split(',')[1])
            with open(filename, 'wb') as f:
                f.write(binary_data)
    
    save_button.on_click(on_button_click)
    
    display(canvas)
    display(save_button)
    display(output)
    display(HTML(canvas_html))




# the control image of init_image and mask_image
def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

