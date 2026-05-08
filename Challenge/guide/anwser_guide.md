# Common Mistakes 

## imports
* Missing the "torch" import
* Misspelled "SkyDataset" as "SkaiDataset"

## data processing
* Transformer uses "randomresized" instead of a resize to 64x64
* Missing labels on the EDA, too many images being plotted
* Missing a `squeeze` on imshow

## model
* Using a "softmax" over the output function, for a binary problem this should be sigmoid
* Missing an "unsqueeze" to resize 
* Missing the 3rd conv block

## training
* Missing the optimizer step in training 
* Validation step still includes the gradient
* Only trains for 2 epochs

# Notebook 1
* Random input size, need to make sure the resize is doing it's job

# Notebook 2
* Imbalanced training data - 85-95% stars. 

# Notebook 3
* The data includes a ton of completely empty images with very large noise levels

# Notebook 4
* Extra class in the training data. Write a function that removes label N=3, or weight it 0. 
* Images are returned as numpy arrays and not torch tensors

# Notebook 5 
* This isn't a classification problem! It's a regression problem! - labels are 1x3 of radius, center x and center y. 