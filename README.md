# Intro Group Project
The proceeding project is meant to be tackled as a group for all new interns.
It will help you to become acquainted with each other an with many of the tools 
you will be using throughout your internship.

For this project, the process and workflow is as important as the code 
produced. Make sure to use:
  * test driven development, with full test coverage
  * correct git workflow
  * code quality checking with pylint

## Prerequisites:
  * Python 3.6 or greater (we'll use Anaconda <https://www.anaconda.com/distribution/#download-section>)
  * Install Python module dependencies: conda install --yes --file requirements.txt
  * Add the repo to your python path.  You can do this by running  the following command from the root directory of the repository.
```
export PYTHONPATH=$PYTHONPATH:`pwd`
```

## Problem definition
The physical system that will be modeled by your code is that of a projectile 
being launched at a fixed velocity, v, and an uncertain angle, θ.  The quantity 
of interest is the distance the projectile travels before hitting 
the ground, d.

![Projectile system](images/projectile.png)

v = 100 m/s

g = 9.8 m/s^2 

The system is to be modeled explicitly (directly coding the equation) and with 
by training a machine learning surrogate.  Both of these models will then be 
used for uncertainty propagation, i.e., finding the uncertainty in d as a 
result of the uncertainty in θ. 

## Code Specification
The code for this project can be broken into a number of components.  They do 
not need to be developed in any order, just ensure the final product is works.

#### ExplicitModel class
Create a class, called ExplicitModel that captures the physics of the above 
system exactly.  It should have a method called evaluate_distance, which 
takes in θ as an argument and returns d.  Use your best judgement on what else 
makes sense for the class to contain.

![ExplicitModel Diagram](http://www.plantuml.com/plantuml/png/SoWkIImgAStDuNAjACZ9JCuiySrFISsnKaWjIymfJIn9ZK_912h9IqxLACb8BKdKj598oybFBE5oICrB0Me30000)

#### Data generation script
Create a script that instances the ExplicitModel class, evaluates the distance 
at several values of θ, then outputs both the input (θ) and output (d) to an 
hdf5 file with the following structure:

**model_output.h5**
```
/                            Group
/explicit_model              Group
/explicit_model/theta        Dataset {100}
/explicit_model/distance     Dataset {100}
```

#### SurogateModel class
Create a SurrogateModel class which utilizes scikit-learn.  The class should 
have an evaluate_distance method which mirrors the ExplicitModel class, but 
utilizes a machine learning model rather than having the physics directly 
coded.  The model should also have a method, train_surrogate, which will train 
the machine learning model based on data that comes from an hdf5 with the above 
structure.  You may use any type of scikit-learn model that makes sense. Again, 
use your best judgement on what else makes sense for the class to contain.

![SurrogateModel Diagram](http://www.plantuml.com/plantuml/png/SoWkIImgAStDuGekBIhAJqyiIVNDJqdDiL98BKlCAGLourCoWMhoabCrIZ9IIn8rhHJISl8JIp3KbnGbPkR55yD4DJ9IqapZqp9pKlCISrCrkHnIyrA0rW00)

#### MonteCarlo class
Create a MonteCarlo class which performs Monte Carlo sampling of the distance 
distribution based on an input θ distribution.  The model which will be sampled 
(either ExplicitModel or SurrogateModel type) and the θ distribution should be 
inputs to the constructor.  It should contain a method called sample which 
returns an array of samples of the distance distribution.

![MonteCarlo Diagram](http://www.plantuml.com/plantuml/png/SoWkIImgAStDuVBDpoj9TKuioictKeXNYC_Coom1KXgv-IcfEJeA9Hcf9OdnAPd59KMPIQKbcVbvcYWwYXDp2t9IDV9AS-CXp69DAmKWhw1I8R6eAB6Ioo4rBmLe3G00)

#### Final Plotting Script
Create a script that creates two instances of the MonteCarlo class, one for an 
explicit model and the other for a surrogate model.  Use a normal distribution 
with a mean of 50 degrees and a standard deviation of 5 degrees for the θ 
distribution.  Plot two distance distributions as histograms by using the two 
MonteCarlo objects to generate samples.

#### Bonus:
Refactor the MonteCarlo class to perform the sampling in parallel using mpi4py.

## Background Reading:
  * pytest
    * Testing your code
    * <https://docs.pytest.org/en/latest/>
    * <https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest>
    * run pytest (including coverage report) by running the following command from the root directory of the repository.
```
pytest --cov=projectile_motion
```
  
  * pylint 
    * Checking your coding style/cleaniless/standards 
    * <https://docs.pylint.org/en/1.6.0/tutorial.html> 
    * example execution: 
```
pylint projectile_motion/__init__.py
```
  
  * scikit-learn
    * <https://scikit-learn.org/stable/documentation.html>
    * A good flowchart for picking your machine learning algorithm: <https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html>
  
  * Monte Carlo Simulation
  * 
   <div style="text-align:center"> <img src="images/monte_carlo.png" width="600"/> </div>

