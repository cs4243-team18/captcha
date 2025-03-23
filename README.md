# Poster

Link: https://www.canva.com/design/DAGh2Wd0fYk/mDj0Nnjm045wxwGvwsSGjQ/edit?utm_content=DAGh2Wd0fYk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

# Methodology to recognise CAPTCHAs

- We have split CAPTCHA recognition into three different phases in the pipeline:
  1. Preprocessing (denoising of image, binarisation)
  2. Segmentation (splitting each CAPTCHA into individual characters)
     2.1. NOTE: this method works best only if characters are spaced far apart; for overlapping characters, non-segmentation methods like using contours work better
  3. Recognition (recognition of each individual character)
     3.1. Our baseline model is using CNN, with aims to improve CNN performance by incorporating resnet and SVMs

# Collaborative guidelines

- Place all finalised methods for each phase in the pipeline inside `playground/helper_functions`, so that anyone can reuse these methods for subsequent phases in the pipeline.
  - For example, after we have found two suitable denoising methods for phase 1 of the pipeline, put them under `playground/helper_functions/prepreocessing`, so that they can be used in phase 2 of the pipeline
  - Likewise, after we have found suitable segmentation methods, put them under `playground/helper_functions/segmentation`, so that both phase 1 and phase 2 methods can be used for phase 3 of the pipeline
  - Since we are all using CNN the vanilla baseline, put helpers under `playground/helper_functions/recognition` as well, so that we can try out CNN models with different tweaks to the hyperparameters, or have additional layers.
- For other important helper functions which can be used at any phase of the pipeline, such as for visualisation or evaluation metric functions, put them under `playground/helper_functions/utils` and do `from helper_functions.utils import *` at the start of all our notebooks so we don't have to repeat ourselves
- For the main pipeline code which are our notebooks (any `.ipynb` files in `playground`), keep them as clean as possible so that it is more readable and maintainable, which makes it easier for other members to build new features on top of one another's work. For large changes, would be safer to work on a new branch and make a pull request for other members to review and resolve any file conflicts together instead
