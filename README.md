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
- We went back to the raw dataset and did our own data pre-processing. 
- We saw the “balanced” dataset, and didn’t like it. We felt like the variables are limited, there are variables to be added,, and there is more cleaning to be done. 
- After researching, we used a few formulas to calculate a few “extra” variables, using existing variables. 
- We defined our own classification method based on astrophysics research, and defined our own Target variable

### Pre-processing pipeline 
1. Dropna
   a. parsing features for numerical values \n
   b. add new features \ remove infinite values \n 
   c. remove nulls after star classification \n
2. Defining target class \n
   a. Based on the Hertzprung-Russell diagram, we defined our own target class based on spectral_types \n
3. Adding new values 

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



 



