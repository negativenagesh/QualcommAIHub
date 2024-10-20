source: https://aihub.qualcomm.com/get-started

# QualcommAIHub

Qualcomm® AI Hub simplifies deploying AI models for vision, audio, and speech applications to edge devices. You can optimize, validate, and deploy your own AI models on hosted Qualcomm platform devices within minutes

## Installation

The Qualcomm AI Hub library for optimization, profiling, and validation can be installed via PyPI. We recommend using Miniconda to manage your python versions and environments. To install, run the following command in your terminal. We recommend a Python version >= 3.8 and <= 3.10.

```bash
pip3 install qai-hub
```

Sign in to 
Qualcomm AI Hub
 with your Qualcomm® ID. After signing in, navigate to 
[your Qualcomm ID] → Settings → API Token
. This should provide an API token that you can use to configure your client.

```bash
qai-hub configure --api_token API_TOKEN
```

Once configured, you can check that your API token is installed correctly by fetching a list of available devices with the following command:

```bash
qai-hub list-devices
```

## Run your first PyTorch model on a hosted device

Once you have set up your Qualcomm AI Hub environment, the next step is to optimize, validate, and deploy a PyTorch model on a cloud-hosted device. This example requires some extra dependencies which can be installed using the following:

```bash
pip3 install "qai-hub[torch]"
```
Now, you can request an automated performance analysis of the MobileNet v2 network. This example compiles and profiles the model on a real hosted device:

Mobile:
```python
import qai_hub as hub
import torch
from torchvision.models import mobilenet_v2
import requests
import numpy as np
from PIL import Image

# Using pre-trained MobileNet
torch_model = mobilenet_v2(pretrained=True)
torch_model.eval()

# Step 1: Trace model
input_shape = (1, 3, 224, 224)
example_input = torch.rand(input_shape)
traced_torch_model = torch.jit.trace(torch_model, example_input)

# Step 2: Compile model
compile_job = hub.submit_compile_job(
    model=traced_torch_model,
    device=hub.Device("Samsung Galaxy S24 (Family)"),
    input_specs=dict(image=input_shape),
)

# Step 3: Profile on cloud-hosted device
target_model = compile_job.get_target_model()
profile_job = hub.submit_profile_job(
    model=target_model,
    device=hub.Device("Samsung Galaxy S24 (Family)"),
)

# Step 4: Run inference on cloud-hosted device
sample_image_url = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg"
)
response = requests.get(sample_image_url, stream=True)
response.raw.decode_content = True
image = Image.open(response.raw).resize((224, 224))
input_array = np.expand_dims(
    np.transpose(np.array(image, dtype=np.float32) / 255.0, (2, 0, 1)), axis=0
)

# Run inference using the on-device model on the input image
inference_job = hub.submit_inference_job(
    model=target_model,
    device=hub.Device("Samsung Galaxy S24 (Family)"),
    inputs=dict(image=[input_array]),
)
on_device_output = inference_job.download_output_data()

# Step 5: Post-processing the on-device output
output_name = list(on_device_output.keys())[0]
out = on_device_output[output_name][0]
on_device_probabilities = np.exp(out) / np.sum(np.exp(out), axis=1)

# Read the class labels for imagenet
sample_classes = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/imagenet_classes.txt"
response = requests.get(sample_classes, stream=True)
response.raw.decode_content = True
categories = [str(s.strip()) for s in response.raw]

# Print top five predictions for the on-device model
print("Top-5 On-Device predictions:")
top5_classes = np.argsort(on_device_probabilities[0], axis=0)[-5:]
for c in reversed(top5_classes):
    print(f"{c} {categories[c]:20s} {on_device_probabilities[0][c]:>6.1%}")

# Step 6: Download model
target_model = compile_job.get_target_model()
target_model.download("mobilenet_v2.tflite")
```
This will submit a compilation job, a profiling job, and an inference job printing the URLs 
for all jobs
. See the 
documentation
 for more details.

## ddrnet_slim23 -->

<img width="480" alt="{7751734B-DFB3-4107-8AFA-D35B0FEC55EE}" src="https://github.com/user-attachments/assets/366144b1-28d3-4dab-8e06-d952c47cec40">

