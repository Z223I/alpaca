# PCA Data Prep

## High Level Requirements

You are a highly skilled Python developer and architect.
You are a highly skilled statistician.
You are working in an atom - molecules architecture.

## Mid Level Requirements

Use the ORB class in orb.py.
Create a private method for PCA data standardization.
It is to conform to repo standards.
Check linting compliance.
Check for VS Code integration errors.
Create PyTest.
Test.

## Low Level Requirements

We will create another method later to call this new method.

### Step 1

Use self.pca_data.
Remove the symbol and timestamp fields.
Extract the volume
Extract the vector_angle

### Step 2

Perform the same statistical standardization across the remaining columns.  They are all prices.
Standardize the volume.
Standardize the vector_angle.
Put all the columns back together side by side.

### Step 3

Create local is_debugging variable and set it to True.
If debugging, print the resulting dataframe.



