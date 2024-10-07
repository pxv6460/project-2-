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

## Data Pre-processing 

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
<img width="742" alt="Screenshot 2024-10-07 at 7 13 28 PM" src="https://github.com/user-attachments/assets/06f65809-e7ec-4d4e-a6ec-e6e4f9f3ee94">


### Correlations
<img width="743" alt="Screenshot 2024-10-07 at 7 14 11 PM" src="https://github.com/user-attachments/assets/0bed4562-36d2-4c4f-9554-980be5c37e23">


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

### ML Training Pipeline 

<img width="757" alt="Screenshot 2024-10-07 at 7 37 54 PM" src="https://github.com/user-attachments/assets/6144f48f-74b8-4243-9a12-d305ec4ab798">

## Model Performance

<img width="545" alt="Screenshot 2024-10-07 at 7 33 44 PM" src="https://github.com/user-attachments/assets/f4a42869-847a-41f6-864b-c0ac1d68bb28">

- **Random Forest (Undersampled)** performs best across all metrics, handling non-linear relationships and imbalanced data effectively.
- **KNN** and **Logistic Regression (Undersampled)** perform well but have lower recall and F1-scores, missing some classifications.
- **Logistic Regression (Regular)** performs weakest, especially in recall, making it less suitable for this task.

### MVPs

<img width="667" alt="Screenshot 2024-10-07 at 7 38 51 PM" src="https://github.com/user-attachments/assets/d25b029f-20a9-462a-afb0-84d75fa95026">

The most important features for stellar classification are **Radius** and **Temperature**, which strongly define a star's evolutionary stage. Other key features like **Magnitude**, **Luminosity**, and **Distance** provide additional predictive power but are less critical. The model's performance highlights the non-linear relationships between these features and the target variable, with **Random Forest** performing best.

## Comparisons 

- **Random Forest (Undersampled)**: Our version improved **accuracy** (0.8895 vs. 0.8784) and **F1-score** (0.8779 vs. 0.8813), highlighting better performance with undersampled data.
- **KNN**: Slight improvements in **accuracy** (0.8763 vs. 0.8682) and balanced **F1-scores** in our version (0.8626 vs. 0.8717).
- **Logistic Regression (Undersampled)**: Slight drop in **precision** (0.8493 vs. 0.8616), but improved **recall** (0.8395 vs. 0.8979).
- **Decision Tree (Undersampled)**: Improved **recall** (0.8435 vs. 0.8195), with a trade-off in **accuracy** (0.8399 vs. 0.8174).
- Overall, our processed data results in more balanced model performance, especially in non-linear models like **Random Forest**

## Conclusion

Stellar classification is a crucial step in understanding the evolutionary stage of a star, providing essential insights into its life cycle.

Our classification models were trained on both available data and newly engineered features. This process was streamlined through the development of data cleansing and model variation pipelines.

Among the models tested, the **Random Forest Classifier** run on undersampled data outperformed **Logistic Regression** and other models, suggesting non-linear relationships between the target variable and features.

The two most important features in classifying a star were found to be:
- **Radius**
- **Temperature**

These features contributed the most to the model's predictive performance.

## Citations

- [Star Categorization - Giants and Dwarfs Discussion](https://www.kaggle.com/datasets/vinesmsuic/star-categorization-giants-and-dwarfs/discussion/287630)
- [Stellar Parallax Glossary](https://itu.physics.uiowa.edu/glossary/stellar-parallax)
- [Measuring Distance to Nearby Stars](https://physicsfeed.com/post/how-do-we-measure-distance-nearby-star-earth/)
- [Calculations in Stellar Astrophysics](https://sites.uni.edu/morgans/astro/course/Notes/section3/math11.html)
- [Lecture Notes on Stellar Classification](https://sarahspolaor.faculty.wvu.edu/files/d/2cac9872-170f-4a59-893e-f69800e0d284/04_notes.pdf)
- [Absolute Magnitude of Stars](http://csep10.phys.utk.edu/OJTA2dev/ojta/c2c/ordinary_stars/magnitudes/absolute_tl.html)
- [Stefan-Boltzmann Law](https://www.teachastronomy.com/textbook/Properties-of-Stars/Stefan-Boltzmann-Law)

### Images
- **Webb images released by NASA (2022):** [NASA Webb Mission Multimedia](https://science.nasa.gov/mission/webb/multimedia/images/)
- **Spectral Classification Image:**
   The modern Morgan–Keenan spectral classification system, showing temperature ranges in Kelvin for each star class. Our Sun is classified as a G-class star with an effective temperature around 5800 K. First-generation stars are thought to consist primarily of O-type and B-type stars, some potentially over 1,000 times the mass of our Sun. ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:HR-diagram-18.png), Additions by E. Siegel)
- **Hubble Space Telescope Images:** [HubbleSite](https://hubblesite.org/home)

## Important Links

- [Presentation](https://docs.google.com/presentation/d/1wgXX1xskPhSk_I_Lj6hHCDrwOBuM2jeK_WAaUD8KIoQ/edit#slide=id.p1)
- [Pre-processing Code](https://github.com/pxv6460/project-2-/blob/main/code/preprocessing.ipynb)
- [Pre-processing Pipeline](https://github.com/pxv6460/project-2-/blob/main/code/preprocess_pipeline.py)
- [ML Model Code](https://github.com/pxv6460/project-2-/blob/main/code/main.ipynb)
- [ML Model Pipeline](https://github.com/pxv6460/project-2-/blob/main/code/pipeline.py)
