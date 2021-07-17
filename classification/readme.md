Steps to get this up and running

* Create a new conda environment
	* e.g., `conda create --name classification --clone base`
* Activate the conda environment
	* e.g., `conda activate classification`
* Run the command
	*  `pip install -r requirements.txt`
* Start the application
	* `uvicorn main:app --reload`
* By default, the app will be accessible on localhost on port number 8000
	* http://localhost:8000/classify
* Model obtained from https://github.com/Kadakol/ML/blob/master/AlexNet.ipynb
* Try it out! 