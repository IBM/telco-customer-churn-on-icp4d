# Predict Customer Churn using Watson Machine Learning and Jupyter Notebooks on Cloud Pak for Data

In this Code Pattern, we use IBM Cloud Pak for Data to go through the whole data science pipeline to solve a business problem and predict customer churn using a Telco customer churn dataset. Cloud Pak for Data is an interactive, collaborative, cloud-based environment where data scientists, developers, and others interested in data science can use tools (e.g., RStudio, Jupyter Notebooks, Spark, etc.) to collaborate, share, and gather insight from their data as well as build and deploy machine learning and deep learning models.

When the reader has completed this Code Pattern, they will understand how to:

* Use [Jupyter Notebooks](https://jupyter.org/) to load, visualize, and analyze data
* Run Notebooks in [IBM Cloud Pak for Data](https://www.ibm.com/analytics/cloud-pak-for-data)
* Build, test and deploy a machine learning model using [Spark MLib](https://spark.apache.org/mllib/) on Cloud Pak for Data.
* Deploy a selected machine learning model to production using Cloud Pak for Data
* Create a front-end application to interface with the client and start consuming your deployed model.

![architecture diagram](doc/source/images/architecture.png)

## Flow

1. User loads the Jupyter notebook into the Cloud Pak for Data platform.
1. [Telco customer churn data set](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)  is loaded into the Jupyter Notebook, either directly from the github repo, or as Virtualized Data after following the [Data Virtualization Tutorial](https://developer.ibm.com/tutorials/virtualizing-db2-warehouse-data-with-data-virtualization) from the [IBM Cloud Pak for Data Learning Path](https://developer.ibm.com/series/cloud-pak-for-data-learning-path/).
1. Preprocess the data, build machine learning models and save to Watson Machine Learning on Cloud Pak for Data.
1. Deploy a selected machine learning model into production on the Cloud Pak for Data platform and obtain a scoring endpoint.
1. Use the model for credit prediction using a frontend application.

## Included components

* [IBM Cloud Pak for Data](https://www.ibm.com/products/cloud-pak-for-data)
* [Watson Machine Learning Add On for Cloud Pak for Data](https://www.ibm.com/cloud/machine-learning)

## Featured technologies

* [Jupyter Notebooks](https://jupyter.org/): An open-source web application that allows you to create and share documents that contain live code, equations, visualizations, and explanatory text.
* [Pandas](https://pandas.pydata.org/):  An open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
* [Seaborn](https://seaborn.pydata.org/): A Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
* [Spark MLib](https://spark.apache.org/mllib/): Apache Spark's scalable machine learning library.

## Prerequisites

* [IBM Cloud Pak for Data](https://www.ibm.com/analytics/cloud-pak-for-data)
* [Watson Machine Learning Add On for Cloud Pak for Data](https://www.ibm.com/support/producthub/icpdata/docs/content/SSQNUZ_current/wsj/analyze-data/ml-install-overview.html)

## Steps

1. [Create a new Project](#1-create-a-new-project)
1. [Create a Space for Machine Learning Deployments](#2-create-a-space-for-machine-learning-deployments)
1. [Upload the dataset](#3-upload-the-dataset) if you are not on the [Cloud Pak for Data Learning Path](https://developer.ibm.com/series/cloud-pak-for-data-learning-path/).
1. [Import notebook to Cloud Pak for Data](#4-import-notebook-to-cloud-pak-for-data)
1. [Run the notebook](#5-run-the-notebook)
1. [Deploying the model using the Cloud Pak for Data UI](#6-deploying-the-model-using-the-Cloud-Pak-for-Data-UI)
1. [Testing the model](#7-testing-the-model)
1. [Create a Python Flask app that uses the model](#8-create-a-python-flask-app-that-uses-the-model)

### 1. Create a new project

* Launch a browser and navigate to your Cloud Pak for Data deployment.

* Go the (☰) menu and click *Projects*:

![(☰) Menu -> Projects](doc/source/images/cpd-projects-menu.png)

* Click on *New project*. In the dialog that pops up, select the project type as `Analytics project` and click `Next`:

![Start a new project](doc/source/images/cpd-new-project.png)

* Click on the top tile for `Create an empty project`:

![Create an empty project](doc/source/images/cpd-create-empty-project.png)

* Give the project a unique name, an optional description and click `Create`:

![Pick a name](doc/source/images/cpd-new-project-name.png)

### 2. Create a Space for Machine Learning Deployments

Before we create a machine learning model, we will have to set up a deployment space where we can save and deploy the model.

Follow the steps in this section to create a new deployment space. If you already have a deployment space set up, you can skip this section and follow the steps to [upload the dataset](#3-upload-the-dataset).

* Navigate to the left-hand (☰) hamburger menu and choose `Analyze` -> `Analytics deployments`:

![(☰) Menu -> Analytics deployments](doc/source/images/ChooseAnalyticsDeployments.png)

* Click on `New deployment space +`:

![Add New deployment space](doc/source/images/addNewDeploymentSpace.png)

* Click on the top tile for 'Create an empty space':

![Create empty deployment space](doc/source/images/createEmptyDeploymentSpace.png)

* Give your deployment space a unique name, an optional description, then click `Create`.

![Create New deployment space](doc/source/images/createNewDeploymentSpace.png)

### 3. Upload the dataset

* If you are not on the [Cloud Pak for Data Learning Path](https://developer.ibm.com/series/cloud-pak-for-data-learning-path/), which uses Virtualized Data, upload the dataset into your project now, else skip down to [import notebook to Cloud Pak for Data](#4-import-notebook-to-cloud-pak-for-data).

* Clone this repository:

```bash
git clone https://github.com/IBM/telco-customer-churn-on-icp4d/
cd telco-customer-churn-on-icp4d
```

* In your project, on the `Assets` tab click the `01/00` icon and the `Load` tab, then either drag the `data/Telco-Customer-Churn.csv` file from the cloned repository to the window or navigate to it using `browse for files to upload`:

![Add data set](doc/source/images/cpd-add-data-set.png)

### 4. Import notebook to Cloud Pak for Data

* In your project, either click the `Add to project +` button, and choose `Notebook`, or, if the *Notebooks* section exists,  to the right of *Notebooks* click `New notebook +`:

![Add notebook](doc/source/images/wml-1-add-asset.png)

* On the next screen, select the *From URL* tab, give your notebook a *name* and an optional *description*, provide the following URL as the *Notebook URL*, and choose the `Python 3.6` environment as the *Runtime*:

```bash
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/notebooks/Telco-customer-churn-ICP4D.ipynb
```

![Add notebook name and URL](doc/source/images/wml-2-add-name-and-url.png)

* When the Jupyter notebook is loaded and the kernel is ready then we can start executing cells.

![Notebook loaded](doc/source/images/wml-3-notebook-loaded.png)

> **Important**: *Make sure that you stop the kernel of your notebook(s) when you are done, in order to conserve memory resources!*

![Stop kernel](doc/source/images/JupyterStopKernel.png)

> **Note**: The Jupyter notebook included in the project has been cleared of output. If you would like to see the notebook that has already been completed with output, you can refer [examples/Telco-customer-churn-ICP4D-with-output.ipynb](examples/Telco-customer-churn-ICP4D-with-output.ipynb).

### 5. Run the notebook

Spend some time looking through the sections of the notebook to get an overview. A notebook is composed of text (markdown or heading) cells and code cells. The markdown cells provide comments on what the code is designed to do.

You will run cells individually by highlighting each cell, then either click the `Run` button at the top of the notebook or hitting the keyboard short cut to run the cell (Shift + Enter but can vary based on platform). While the cell is running, an asterisk (`[*]`) will show up to the left of the cell. When that cell has finished executing a sequential number will show up (e.g. `[17]`).

**Please note that some of the comments in the notebook are directions for you to modify specific sections of the code. Perform any changes as indicated before running / executing the cell.**

#### Notebook sections

With the notebook open, you will notice:

- Section `1.0 Install required packages` will install some of the libraries we are going to use in the notebook (many libraries come pre-installed on Cloud Pak for Data). Note that we upgrade the installed version of Watson Machine Learning Python Client. Ensure the output of the first code cell is that the python packages were successfully installed.

![Install required packages](doc/source/images/wml-3.2-install-packages.png)

- Section `2.0 Load and Clean data` will load the data set we will use to build out the machine learning model. In order to import the data into the notebook, we are going to use the code generation capability of Watson Studio.
   - Highlight the code cell shown in the image below by clicking it. Ensure you place the cursor below the commented line.
   - Click the 01/00 "Find data" icon in the upper right of the notebook to find the data asset you need to import.
   - If you are following the [Cloud Pak for Data Learning Path](https://developer.ibm.com/series/cloud-pak-for-data-learning-path/), choose the *Files* tab, and pick the virtualized data set that has all three joined tables (i.e. `User<xyz>.BILLINGPRODUCTSCUSTOMERS`). Click `Insert to code` and choose `pandas DataFrame`.

![Add remote Pandas DataFrame](doc/source/images/wml-4-add-dataframe.png)

   - Otherwise, if you are using this notebook without virtualized data, you can use the [Telco-Customer-Churn.csv](data/Telco-Customer-Churn.csv) file version of the data set that has been included in this project and was uploaded to the Cloud Pak for Data project in [Step 3](#3-upload-the-dataset). Choose the *Files* tab. Select the *Telco-Customer-Churn.csv* file. Click `Insert to code` and choose `pandas DataFrame`.

![Add local Pandas DataFrame](doc/source/images/wml-3-add-local-dataframe.png)

   - The code to bring the data into the notebook environment and create a Pandas DataFrame will be added to the cell.
   - Run the cell and you will see the first five rows of the dataset.

![Generated code to handle Pandas DataFrame](doc/source/images/wml-5-generated-code-dataframe.png)

> **IMPORTANT**: Since we are using generated code to import the data, you will need to update the next cell to assign the `df` variable. Copy the variable that was generated in the previous cell (it will look like data_df_1, data_df_2, etc) and assign it to the `df` variable (for example df=df_data_1).

   - Continue to run the remaining cells in section 2 to explore and clean the data.

- Section `3.0 Create a model` cells will run through the steps to build a model pipeline.
   - We will split our data into training and test data, encode the categorial string values, create a model using the Random Forest Classifier algorithm, and evaluate the model against the test set.
   - Run all the cells in section 3 to build the model.

![Building the pipeline and model](doc/source/images/wml-6-build-pipeline-and-model.png)

- Section `4.0 Save the model` will save the model to your project.

- We will be saving and deploying the model to the Watson Machine Learning service within our Cloud Pak for Data platform. In the next code cell, be sure to update the `wml_credentials` variable.
  - The url should be the hostname of the Cloud Pak for Data instance.
  - The username and password should be the same credentials you used to log into Cloud Pak for Data.

- Update the `MODEL_NAME` variable and provide a unique and easily identifiable model name. Next, update the `DEPLOYMENT_SPACE_NAME` variable, providing the name of your deployment space which was created in [Step 2](#2-create-a-space-for-machine-learning-deployments) above.

![Provide model and deployment space name](doc/source/images/wml-provide-model-and-space-name.png)

![Update WML credentials](doc/source/images/wml-7-update-wml-credentials.png)

Continue to run the cells in the section to save the model to Cloud Pak for Data. We'll be able to test it out with the Cloud Pak for Data tools in just a few minutes!

> **Note**: You can use the following cell for cleaning up any previously created models and deployments.

![Clean up models and deployments](doc/source/images/cleanup-models-and-deployments.png)

### 6. Deploying the model using the Cloud Pak for Data UI

Now that we have created a model and saved it to our respository, we will want to deploy the model so it can be used by others. 

We will be creating an online deployment. This type of deployment will make an instance of the model available to make predictions in real time via an API. 

Although we use the Cloud Pak for Data UI to deploy the model here, the same can also be done programmatically.

- Navigate to the left-hand (☰) hamburger menu and choose `Analyze` -> `Analytics deployments`:

![Analytics Analyze deployments](doc/source/images/ChooseAnalyticsDeployments.png)

- Choose the deployment space you setup previously by clicking on the name of the space.

![Deployment space](doc/source/images/deployment-space.png)

- In your space overview, click the model name that you just built in the notebook:

![select model](doc/source/images/deployment-select-model.png)

- Click `Create deployment` on the top-right corner.

![Actions Deploy model](doc/source/images/ActionsDeployModel.png)

- On the 'Create a deployment' screen, choose `Online` for the Deployment Type, give the Deployment a name and an optional description and click `Create`:

![Online Deployment Create](doc/source/images/OnlineDeploymentCreate.png)

- The Deployment will show as *In progress* and then switch to *Deployed* when done.

![Status Deployed](doc/source/images/StatusDeployed.png)

### 7. Testing the model

Cloud Pak for Data offers tools to quickly test out Watson Machine Learning models. We begin with the built-in tooling.

- Click on the deployment. The Deployment *API reference* tab shows how to use the model using *cURL*, *Java*, *Javascript*, *Python*, and *Scala*. Click on the corresponding tab to get the code snippet in the language that you want to use:

![Deployment API reference](doc/source/images/api-reference-curl.png)

#### Test the saved model with built-in tooling

- To get to the built-in test tool, click on the Test tab. Click on the `Provide input data as JSON` icon and paste the following data under Body:

```json
{
   "input_data":[
      {
         "fields":[
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges"
         ],
         "values":[
            [
               "Female",
               0,
               "No",
               "No",
               1,
               "No",
               "No phone service",
               "DSL",
               "No",
               "No",
               "No",
               "No",
               "No",
               "No",
               "Month-to-month",
               "No",
               "Bank transfer (automatic)",
               25.25,
               25.25
            ]
         ]
      }
   ]
}
```

- Click the `Predict` button  and the model will be called with the input data. The results will display in the *Result* window. Scroll down to the bottom (Line #114) to see either a "Yes" or a "No" for Churn:

![Testing the deployed model](doc/source/images/TestingDeployedModel.png)

#### Test the deployed model with cURL

Now that the model is deployed, we can also test it from external applications. One way to invoke the model API is using the cURL command.

> NOTE: Windows users will need the *cURL* command. It's recommended to [download gitbash](https://gitforwindows.org/) for this, as you will also have other tools and you will be able to easily use the shell environment variables in the following steps. Also note that if you are not using gitbash, you may need to change *export* commands to *set* commands.

- In a terminal window (or command prompt in Windows), run the following command to get a token to access the API. Use your Cloud Pak for Data cluster `username` and `password`:

```bash
curl -k -X GET https://<cluster-url>/v1/preauth/validateAuth -u <username>:<password>
```

- A json string will be returned with a value for "accessToken" that will look *similar* to this:

```json
{"username":"snyk","role":"Admin","permissions":["access_catalog","administrator","manage_catalog","can_provision"],"sub":"snyk","iss":"KNOXSSO","aud":"DSX","uid":"1000331002","authenticator":"default","accessToken":"eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InNueWstYWRtaW4iLCJyb2xlIjoiQWRtaW4iLCJwZXJtaXNzaW9ucyI6WyJhZG1pbmlzdHJhdG9yIiwiY2FuX3Byb3Zpc2lvbiIsIm1hbmFnZV9jYXRhbG9nIiwibWFuYWdlX3F1YWxpdHkiLCJtYW5hZ2VfaW5mb3JtYXRpb25fYXNzZXRzIiwibWFuYWdlX2Rpc2NvdmVyeSIsIm1hbmFnZV9tZXRhZGF0YV9pbXBvcnQiLCJtYW5hZ2VfZ292ZXJuYW5jZV93b3JrZmxvdyIsIm1hbmFnZV9jYXRlZ29yaWVzIiwiYXV0aG9yX2dvdmVycmFuY2VfYXJ0aWZhY3RzIiwiYWNjZXNzX2NhdGFsb2ciLCJhY2Nlc3NfaW5mb3JtYXRpb25fYXNzZXRzIiwidmlld19xdWFsaXR5Iiwic2lnbl9pbl9vbmx5Il0sInN1YiI6InNueWstYWRtaW4iLCJpc3MiOiJLTk9YU1NPIiwiYXVkIjoiRFNYIiwidWlkIjoiMTAwMDMzMTAwMiIsImF1dGhlbnRpY2F0b3IiOiJkZWZhdWx0IiwiaWp0IjoxNTkyOTI3MjcxLCJleHAiOjE1OTI5NzA0MzV9.MExzML-45SAWhrAK6FQG5gKAYAseqdCpublw3-OpB5OsdKJ7isMqXonRpHE7N7afiwU0XNrylbWZYc8CXDP5oiTLF79zVX3LAWlgsf7_E2gwTQYGedTpmPOJgtk6YBSYIB7kHHMYSflfNSRzpF05JdRIacz7LNofsXAd94Xv9n1T-Rxio2TVQ4d91viN9kTZPTKGOluLYsRyMEtdN28yjn_cvjH_vg86IYUwVeQOSdI97GHLwmrGypT4WuiytXRoQiiNc-asFp4h1JwEYkU97ailr1unH8NAKZtwZ7-yy1BPDOLeaR5Sq6mYNIICyXHsnB_sAxRIL3lbBN87De4zAg","_messageCode_":"success","message":"success"}
```

- Use the export command to save the "accessToken" part of this response in the terminal window to a variable called `WML_AUTH_TOKEN`. 

```bash
export WML_AUTH_TOKEN=<value-of-access-token>
```

- Back on the model deployment page, gather the `URL` to invoke the model from the *API reference* by copying the `Endpoint`, and export it a variable called `URL`:

![Model Deployment Endpoint](doc/source/images/ModelDeploymentEndpoint.png)

```bash
export URL=https://blahblahblah.com
```

Now run this curl command from a terminal window to invoke the model with the same payload that was used previously:

```bash
curl -k -X POST --header 'Content-Type: application/json' --header 'Accept: application/json' --header "Authorization: Bearer  $WML_AUTH_TOKEN" -d '{"input_data": [{"fields": ["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"],"values": [["Female",0,"No","No",1,"No","No phone service","DSL","No","No","No","No","No","No","Month-to-month","No","Bank transfer (automatic)",25.25,25.25]]}]}' $URL
```

A json string similar to the one below will be returned with the response, including a "Yes" or "No" at the end indicating the prediction of whether the customer will churn or not.

```json
{"predictions":[{"fields":["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges","gender_IX","Partner_IX","Dependents_IX","PhoneService_IX","MultipleLines_IX","InternetService_IX","OnlineSecurity_IX","OnlineBackup_IX","DeviceProtection_IX","TechSupport_IX","StreamingTV_IX","StreamingMovies_IX","Contract_IX","PaperlessBilling_IX","PaymentMethod_IX","label","features","rawPrediction","probability","prediction","predictedLabel"],"values":[["Female",0,"No","No",1,"No","No phone service","DSL","No","No","No","No","No","No","Month-to-month","No","Bank transfer (automatic)",25.25,25.25,1.0,0.0,0.0,1.0,2.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,0.0,[18,[0,4,5,6,14,15,16,17],[1.0,1.0,2.0,1.0,1.0,2.0,25.25,25.25]],[10.806165651100262,9.193834348899738],[0.5403082825550131,0.45969171744498694],0.0,"No"]]}]}
```

### 8. Create a Python Flask app that uses the model

You can also access the online model deployment directly through the REST API. This allows you to use your model for inference in any of your apps. For this code pattern, we'll be using a Python Flask application to collect information, score it against the model, and show the results.

#### Install dependencies

> **NOTE**: This application only runs on Python 3.6 and above, so the instructions here are for Python 3.6+ only.

The general recommendation for Python development is to use a virtual environment ([`venv`](https://docs.python.org/3/tutorial/venv.html)). To install and initialize a virtual environment, use the `venv` module:

In a terminal, go to the `flaskapp` folder within the cloned repo directory.

```bash
git clone https://github.com/IBM/telco-customer-churn-on-icp4d/
cd telco-customer-churn-on-icp4d/flaskapp
```

Initialize a virtual environment with [`venv`](https://docs.python.org/3/tutorial/venv.html).

```bash
# Create the virtual environment using Python. 
# Note, it may be named python3 on your system.
python -m venv venv       # Python 3.X

# Source the virtual environment. Use one of the two commands depending on your OS.
source venv/bin/activate  # Mac or Linux
./venv/Scripts/activate   # Windows PowerShell
```

> **TIP** To terminate the virtual environment use the `deactivate` command.

Finally, install the Python requirements.

```bash
pip install -r requirements.txt
```

#### Update environment variables

It is best practice to store configurable information as environment variables, instead of hard-coding any important information. To reference our model and supply an API key, we will pass these values in via a file that is read; the key-value pairs in this file are stored as environment variables.

Copy the `env.sample` file to `.env`.

```bash
cp env.sample .env
```

Edit the .env file and fill in the `MODEL_URL` as well as the `AUTH_URL`, `AUTH_USERNAME`, and `AUTH_PASSWORD`.

* `MODEL_URL` is your web service URL for scoring which you got from the section above
* `AUTH_URL` is the preauth url of your CloudPak4Data and will look like this: https://<cluster_url>/v1/preauth/validateAuth
* `AUTH_USERNAME` is your username with which you login to the CloudPak4Data environment
* `AUTH_PASSWORD` is your password with which you login to the CloudPak4Data environment

> **NOTE**: Alternatively, you can fill in the AUTH_TOKEN instead of AUTH_URL, AUTH_USERNAME, and AUTH_PASSWORD. You will have generated this token in the section above. However, since tokens expire after a few hours and you would need to restart your app to update the token, this option is not suggested. Instead, if you use the username/password option, the app can generate a new token every time for you so it will always use a non-expired token.

```bash
# Copy this file to .env.
# Edit the .env file with the required settings before starting the app.

# 1. Required: Provide your web service URL for scoring.
# E.g., MODEL_URL=https://<cluster_url>/v4/deployments/<deployment_space_guid>/predictions
MODEL_URL=


# 2. Required: fill in EITHER section A OR B below:

# ### A: Authentication using username and password
#   Fill in the authntication url, your CloudPak4Data username, and CloudPak4Data password.
#   Example:
#     AUTH_URL=<cluster_url>/v1/preauth/validateAuth
#     AUTH_USERNAME=my_username
#     AUTH_PASSWORD=super_complex_password
AUTH_URL=
AUTH_USERNAME=
AUTH_PASSWORD=

# ### B: (advanced) Provide your bearer token.
#   Uncomment the "AUTH_TOKEN=" below and fill in your Bearer Token.
#   You can generate this token by followin the lab instuctions. This token should start with "Bearer ".
#   Note: that hese tokens will expire after a few hours so you'll need to generate a new one again later.
#   Example:
#       TOKEN=Bearer abCdwFghIjKLMnO1PqRsTuV2wWX3YzaBCDE4.fgH1r2... (and so on, tokens are long).
# AUTH_TOKEN=


# Optional: You can override the server's host and port here.
HOST=0.0.0.0
PORT=5000
```

#### Start the application

Start the flask server by running the following command:

```bash
python telcochurn.py
```

Use your browser to go to [http://0.0.0.0:5000](http://0.0.0.0:5000) and try it out.

> **TIP**: Use `ctrl`+`c` to stop the Flask server when you are done.

#### Sample output

Enter some sample values into the form:

![Input a bunch of data...](doc/source/images/input.png)

Click the `Submit` button and the churn percentage is returned:

![Get the churn percentage as a result](doc/source/images/score.png)

## Learn more
* **Artificial Intelligence Code Patterns**: Enjoyed this Code Pattern? Check out our other [AI Code Patterns](https://developer.ibm.com/technologies/artificial-intelligence/).
* **Data Analytics Code Patterns**: Enjoyed this Code Pattern? Check out our other [Data Analytics Code Patterns](https://developer.ibm.com/technologies/data-science/).
* **AI and Data Code Pattern Playlist**: Bookmark our [playlist](https://www.youtube.com/playlist?list=PLzUbsvIyrNfknNewObx5N7uGZ5FKH0Fde) with all of our Code Pattern videos.
* **With Watson**: Want to take your Watson app to the next level? Looking to utilize Watson Brand assets? [Join the With Watson program](https://www.ibm.com/watson/with-watson/) to leverage exclusive brand, marketing, and tech resources to amplify and accelerate your Watson embedded commercial solution.
* **IBM Watson Studio**: Master the art of data science with IBM's [Watson Studio](https://www.ibm.com/cloud/watson-studio).

## License
This code pattern is licensed under the Apache Software License, Version 2.  Separate third party code objects invoked within this code pattern are licensed by their respective providers pursuant to their own separate licenses. Contributions are subject to the [Developer Certificate of Origin, Version 1.1 (DCO)](https://developercertificate.org/) and the [Apache Software License, Version 2](https://www.apache.org/licenses/LICENSE-2.0.txt).

[Apache Software License (ASL) FAQ](https://www.apache.org/foundation/license-faq.html#WhatDoesItMEAN)
