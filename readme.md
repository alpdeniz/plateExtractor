## plateExtractor

Extracts vehicle plates from a video stream via CNN using Keras.

### Usage

    pip3 install -r requirements.txt
    ./main.py {pathOrUrlToVideo}

### The model

You can generate any new model on your own by using train.py under scripts folder. Current model uses:
- convolution(32, kernel=5)
- convolution(64, kernel=3)
with maxpooling, dropouts and batch normalization.

**Note that this is trained and tested on Turkish vehicle plates.**

### Scaling

Current state seems to be optimal for 1920x1080 frames and it may not work well for other resolutions yet, though it should be easily doable.

### TODO
You are invited to create and fix issues.
- Make sure it works for all frame resolutions (plateExtractor:sobel_fill_dims and various hardcoded values such as plateExtractor:minContourArea)
- After reading plates, convert ambigious numbers to letters or vice-versa to fit the plate regex pattern (8 -> B, 0 -> D, I -> 1 ...)
- Refactor