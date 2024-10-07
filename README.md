# Project-2 - Team 2 to Tango

Niharika, Hazel, Azlan, Peter and Enrique

## One Stellar Classification

Machine learning can help us predict the type of a star based on features such as Visual Apparent Magnitude, Distance Between the Star and the Earth, Color Index, and Spectral Type the team explored a Star Dataset and implemented a ML model while building two pipelines: one to preprocess data and one to make predictions. Additionally, team 2 utilized custom formulas to create ‘new features’ and fine tuned the model for greater accuracy.

<img width="824" alt="Screenshot 2024-10-07 at 6 06 53 PM" src="https://github.com/user-attachments/assets/7fd8bc8b-5fbb-4b2f-8336-dec5cfc74cda">


Data Source: https://www.kaggle.com/datasets/vinesmsuic/star-categorization-giants-and-dwarfs/discussion/287630 

### Existing features from original dataset:
Vmag – Visual Apparent Magnitude of the Star 
<br> Plx – Distance Between the Star and the Earth 
<br> B-V color index – Hot star B-V close to 0 or negative, cool star has a B-V close to 2.0 
<br> SpType – Spectral type

### Data Pre-processing 

### Data Decisions
- We revisited the raw dataset and conducted our own data pre-processing.
- We found the "balanced" dataset unsatisfactory due to limited variables, missing potential variables, and the need for further cleaning.
- After additional research, we derived a few extra variables using existing data through calculated formulas.
- We defined our own classification method based on astrophysics research, establishing a custom target variable.

### Pre-processing Pipeline
1. **Dropna**
   - Parsing features for numerical values.
   - Adding new features and removing infinite values.
   - Removing nulls post star classification.
   
2. **Defining Target Class**
   - Using the Hertzsprung-Russell diagram, we established a custom target class based on spectral types.
   
3. **Adding New Values**
   - Integrated newly calculated features into the dataset.
  
4. **Building Preprocessing Pipeline**
    - code/preprocess_pipeline.py and code/preprocessing.ipynb

### New features added by the team:
Temperature
<br> Distance (parsecs)
<br> Distance (light years)
<br> Amag (absolute magnitude) 
<br> Luminosity (Sun=1)
<br> Radius (Sun=1)
<br> Plx (in arcsecs)

### Formulas used to calculate new features:
self.df["Plx"] = self.df["Plx"] / 1000
<br> ["Distance (parsecs)"] = 1/self.df["Plx"]
<br> ["Distance (light years)"] = self.df["Distance (parsecs)"] * 3.26156
<br> ["Amag"] = self.df["Vmag"] + 5 * (np.log10(self.df["Plx"]) + 1)
<br> ["Temperature (K)"] = 4600 * (1/(0.92*self.df["B-V"] + 1.7) + 1/(0.92*self.df["B-V"] + 0.62))
<br> ["Luminosity (Sun=1)"] = 10**(0.4 * (4.85-self.df["Amag"]))
<br> ["Radius (Sun=1)"] = np.sqrt(self.df["Luminosity (Sun=1)"]) * (5778 / self.df["Temperature (K)"])**2


### New Dataset!
insert image

### Coorelations
insert image

### Data Decisions
1. **BV and Temperature are highly negatively correlated at -0.93, we dropped BV**
   - we experimented with dropping either/both, but dropping BV gives the best results

2. **distance (parsecs) and distance(light years) are repetitive**
   - we chose distance (light years) because it is more commonly used

## Model Training and Comparison

### Models Used
- **Support Vector Classifier (SVC)**
- **Decision Tree**
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Random Forest**

### Rationale
- We focused on a **binary target** variable (0,1).
- Non-linear models were chosen to explore complex relationships within the data.

### Model Comparisons
We conducted the following comparisons to evaluate model performance:

1. **Comparison of Model Results:**
   - After training models on the pre-processed dataset, we compared their accuracy, precision, recall, and other key metrics.

2. **Processed Data Results:**
   - We assessed how each model performed between our preprocessed data and balanced data (cleaned data done by others).

3. **Comparison: Undersampling Data:**
   - We applied undersampling techniques to balance the dataset and compared how models responded to this adjustment.

