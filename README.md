## Hair-Detection project
### Brief description
Pipeline that I implemented is very simple and consists of following steps:

1. Detect face

2. Detect landmarks

3. Align face using landmarks

4. Extract ROI for CNN Hair-detector model

5. Run CNN Hair-detector model on the extracted ROI


Steps 1 and 2 were implemented using opencv library with pre-trained and pretty basic face and landmark detectors.

Steps 3 and 4 are aimed to extract ROI as similar as possible to the training data, i.e. square that centered between the eyes. Also, this square should cover enough space for model to decide, if the hair is short or long.

Step 5 is just loading model and performing forward pass.

### Few notes about training
Training procedure was pretty basic: I used 70/30 train-test split while keeping classes balanced. 

Since 20k images is not a huge number, I decided to keep it simple and build a basic CNN with convolutions and separable convolutions. All other parts of architecture are also quite regular -- 3~128 channels, BatchNorms, ReLU, Linear layer at the end.

Since hair is usually quite noticeable (i.e. not some small detail/pattern), I also decided to work with grayscale images.

In order to avoid overfitting, I use RandomHorizontalFlip and 224px RandomCrops during the training phase along with color augmentations such as contrast and brightness.

### Assumptions
During my work on this project I assumed that anyone, who wants to run it, will use GPU with CUDA, so I omitted all of boilerplate code like "device=..." and "if torch.cuda.is_available():...".

### Further possible improvements
1. Better FaceDetector can drastically improve overall performance, so it might be beneficial to find a pre-trained one or even train new FD from scratch. LandmarkDetector looks fine for now, but it is also pretty basic. 

2. Train on bigger amount of data: we can use labeled publicly-available datasets with hair length or just label any data in semi-automatic mode using previously trained baseline model.

3. Add more complex augmentation (e.g. more aggressive random crop to support less centered faces; right now it is impossible given current extracted ROIs)

4. We can experiment with ROIs and their extraction as well, i.e. include more space around head or we can center ROIs in some other way. Also we can use better alignment near the corners of image (instead of constant padding that I saw in some train images)

5. And finally, given more data we can run wide range of experiments regarding training procedure, model architecture and so on.