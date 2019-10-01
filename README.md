# Predict Customer Churn using Watson Machine Learning and Jupyter Notebooks on Cloud Pak for Data

In this Code Pattern, we use IBM Cloud Pak for Data (ICP4D) to go through the whole data science pipeline to solve a business problem and predict customer churn using a Telco customer churn dataset. Cloud Pak for Data is an interactive, collaborative, cloud-based environment where data scientists, developers, and others interested in data science can use tools (e.g., RStudio, Jupyter Notebooks, Spark, etc.) to collaborate, share, and gather insight from their data as well as build and deploy machine learning and deep learning models.

When the reader has completed this Code Pattern, they will understand how to:

* Use [Jupyter Notebooks](https://jupyter.org/) to load, visualize, and analyze data
* Run Notebooks in [IBM Cloud Pak for Data](https://www.ibm.com/analytics/cloud-pak-for-data)
* Build, test and deploy a machine learning model using [Spark MLib](https://spark.apache.org/mllib/) on ICP4D.
* Deploy a selected machine learning model to production using Cloud Pak for Data
* Create a front-end application to interface with the client and start consuming your deployed model.

![](doc/source/images/architecture.png)

## Flow

1. Load the provided notebook into the Cloud Pak for Data platform.
1. [Telco customer churn data set](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)  is loaded into the Jupyter Notebook, either directly from the github repo, or as [Virtualized Data] after following the [Data Virtualization Tutorial]() from the [IBM Cloud Pak for Data Learning Path]().
1. Describe, analyze and visualize data in the notebook.
1. Preprocess the data, build machine learning models and test them.
1. Deploy a selected machine learning model into production.
1. Interact and consume your model using a frontend application.

## Included components

* [IBM Cloud Pak for Data]
* [Watson Machine Learning Add On for ICP4D]

## Featured technologies

* [Jupyter Notebooks](https://jupyter.org/): An open-source web application that allows you to create and share documents that contain live code, equations, visualizations, and explanatory text.
* [Pandas](https://pandas.pydata.org/):  An open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
* [Seaborn](https://seaborn.pydata.org/): A Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
* [Spark MLib](https://spark.apache.org/mllib/): Apache Spark's scalable machine learning library.

## Prerequisites

* [IBM Cloud Pak for Data](https://www.ibm.com/analytics/cloud-pak-for-data)
* [Watson Machine Learning Add On for ICP4D](https://www.ibm.com/support/producthub/icpdata/docs/content/SSQNUZ_current/com.ibm.icpdata.doc/zen/admin/add-ons.html#add-ons__ai)

# Steps

1. [Create a new Project](#1-create-a-new-project)
1. [Upload the dataset](#2-upload-the-dataset) if you are not on the [ICP4D Learning Path]().
1. [Import notebook to Cloud Pak for Data](#3-import-notebook-to-cloud-pak-for-data)
1. [Import dataset into the notebook](#4-import-dataset-into-the-notebook)
1. [Follow the steps in the notebook](#5-follow-the-steps-in-the-notebook)
1. [Create a project release](#6-create-a-project-release)
1. [Deploy the project](#7-deploy-the-project)
1. [Testing the model](#8-testing-the-model)
1. [Create a Python Flask app that uses the model](#9-create-a-python-flask-app-that-uses-the-model)

## 1. Create a new project

Launch a browser and navigate to your Cloud Pak for Data deployment

Go the (☰) menu and click *Projects*

![(☰) Menu -> Projects](doc/source/images/cpd-projects-menu.png)

Click on *New project*

![Start a new project](doc/source/images/cpd-new-project.png)

Create a new project, choose `Analytics project`, give it a unique name, and click *OK. Click `Create` on the next screen.

![Pick a name](doc/source/images/cpd-new-project-name.png)

## 2. Upload the dataset

If you are not on the [ICP4D Learning Path], which uses Virtualized Data, upload the dataset into your project now.

Clone this repository:

```bash
git clone https://github.com/IBM/telco-customer-churn-on-icp4d/
cd telco-customer-churn-on-icp4d
```

In your project, choose `Data sets` from the left-hand menu, and then click `+Add Data set`:

![Add data set](doc/source/images/cpd-add-data-set.png)

Either drag the `Telco-Customer-Churn.csv` file to the window or navigate to it using `Select from your local file system`:

![Select data set](doc/source/images/cpd-select-file-data.png)

## 3. Import notebook to Cloud Pak for Data

At the project overview click the *New Asset* button, and choose *Add notebook*.

![Add a new asset](doc/source/images/wml-1-add-asset.png)

On the next panel select the *From URL* tab, give your notebook a name, provide the following URL, and choose the Python 3.6 environment:

```bash
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/notebooks/Telco-customer-churn-ICP4D.ipynb
```

![Add notebook name and URL](doc/source/images/wml-2-add-name-and-url.png)

When the Jupyter notebook is loaded and the kernel is ready then we can start executing cells.

![Notebook loaded](doc/source/images/wml-3-notebook-loaded.png)

### 4. Import dataset into the notebook

Spend a minute looking through the sections of the notebook to get an overview.

In section `2.0 Load and Clean data`, highlight the cell labelled `# Place cursor ...` by clicking on it. Click on the *10/01* button to select a specific data set.

If you are following the [Learning Path](), Choose The *Remote* tab, and pick the virtualized data set that has all three joined tables (i.e. `User999.billing+products+customers`), and choose `Insert Pandas DataFrame`.

![Add remote Pandas DataFrame](doc/source/images/wml-4-add-dataframe.png)

If you are running this pattern alone, choose the *Local* tab, and choose the 'Telco-Customer-Churn.csv` file that you added, then choose `Insert Pandas DataFrame`:

![Add local Pandas DataFrame](doc/source/images/wml-3-add-local-dataframe.png)

By adding data a block of code will be added to the notebook. The code will automatically load that data set and create a Pandas DataFrame.


![Generated code to handle Pandas DataFrame](doc/source/images/wml-5-generated-code-dataframe.png)

> **IMPORTANT**: Don't forget to update the next cell `df = df1` (or whatever your dataframe number is) with the variable from the generated code, in the case of `Remote` Virtualized data. If you are using the `Local` data, use `df = df_data_1` (or whatever your dataframe number is).

### 5. Follow the steps in the notebook

You will run cells individually by highlighting each cell, then either click the `Run` button at the top of the notebook. While the cell is running, an asterisk (`[*]`) will show up to the left of the cell. When that cell has finished executing a sequential number will show up (i.e. `[17]`).

Continue to run the remaining cells in the section to clean the data.

#### Create the model

Section `3.0 Create a model` will split the data into training and test data, and create a model using the Random Forest Classifier algorithm.

![Building the pipeline and model](doc/source/images/wml-6-build-pipeline-and-model.png)

Continue to run the remaining cells in the section to build the model.

#### Save the model

Section `4.0 Save the model` will save the model to your project. Update the `MODEL_NAME` variable to something unique and easisly identifiable.

```python
MODEL_NAME = "user123 customer churn model"
```

Continue to run the remaining cells in the section to save the model to Cloud Pak for Data. We'll be able to test it out with the Cloud Pak for Data tools in just a few minutes!

## 6. Create a project release

Next, we'll create a project release and tag the model under version control. We'll use model management and deployment to make the released model available as a web service (REST API).

### Commit the project changes

On the project home click on the *Git* button on the top row and choose *Commit*

![Commit the project changes](doc/source/images/project-1-git-commit.png)

A list of the assets will appear that were created in this project. Provide a commit message to identify the changes being pushed.

![Provide a commit message](doc/source/images/project-2-git-commit-message.png)

Again, click the same *Git* button and this time choose *Push*.

![Push a new version](doc/source/images/project-3-git-push.png)

Provide a version tag under *Create version tag for release*. Add a tag, i.e. `v1` or `v2` and click *Push*.

![Provide a commit message](doc/source/images//project-4-git-push-version.png)

### Release a new version

Now that we have a committed and tagged version of the project, we can create a project release and deploy it as a web service.

To start creating a new project release, go the (☰) menu and click on the *Manage deployments* option.

![(☰) Menu -> Manage deployments](doc/source/images/project-5-manage-deployments.png)

Click on `+ Add Project Release` to start creating a new project release.

![There are currently no releases for this project](doc/source/images/project-6-new-project-release.png)

On the next panel ensure the *From IBM Cloud Pak for Data* tab is selected, and give your project release a name and route. Select the project and version from the drop down menus, and click on *Create*.

![Fill in the project release details](doc/source/images/project-7-new-project-release-details.png)

### Configure project release

It's now time to configure the project release. Here we will choose what assets will be deployed and how they will be deployed.

We start by deploying the model we built as a web service. Click on the model on the list of *Assets* and choose to add a *Web Service*.

![Deploy the model as a web service](doc/source/images/project-8-deployment-overview.png)

Give the web service a *Name*, select a *Model version*, and *Web Service environment*. Click the *Create* button.

![Deploy the model as a web service](doc/source/images/project-9-add-web-service.png)

Once created the model details will appear, take note of the *Endpoint* and *Deployment token* that have been generated.

![Endpoint and token for the deployed model](doc/source/images/project-10-model-endpoint.png)

> **NOTE**: The deployment is not yet active. We need to launch and enable it before it can be used.

## 7. Deploy the project

* You will be brought back to the project release page where you will see your model is *Disabled*. Click the *Launch* button to deploy your project.

![Deploy your project](doc/source/images/project-11-model-disabled.png)

Once the deployment is complete click on the action action menu (vertical 3 dots) of the model and select *Enable*.

![Endpoint and token for the deployed model](doc/source/images/project-12-model-enabled.png)

## 8. Testing the model

Cloud Pak for Data offers tools to quickly test out Watson Machine Learning models. We begin with the built-in tooling.

### Test the saved model with built-in tooling

Once the model is enabled we can test the API interface from Cloud Pak for Data. Click the enabled model deployment. From the *API* tab, default values are given and we can simply click the *Submit* button. The results are shown on the right.

![Testing the deployed model](doc/source/images/testing-1-api.png)

### Test the deployed model with cURL

Clicking the *Generate Code* button will pop open a window with some copy for you to copy. The code will use the cURL command line utility to test the REST APIs. Here's an example of the generated code that can be run from a terminal window with the `curl` command.

```bash
curl -k -X POST \
  https://9.10.111.122:31843/dmodel/v1/churn1/pyscript/churn/score \
  -H 'Authorization: Bearer yeJhbGaaaiJSUzI1NiIsInR5cCI6IkpXVCJ9...jJDMbgsGqy9C_AsK5n28HysmH2NeXzEN9A' \
  -H 'Cache-Control: no-cache' \
  -H 'Content-Type: application/json' \
  -d '{"args":{"input_json":[{"ID":4,"GENDER":"F","STATUS":"M","CHILDREN":2,"ESTINCOME":52004,"HOMEOWNER":"N","AGE":25,"TOTALDOLLARVALUETRADED":5030,"TOTALUNITSTRADED":23,"LARGESTSINGLETRANSACTION":1257,"SMALLESTSINGLETRANSACTION":125,"PERCENTCHANGECALCULATION":3,"DAYSSINCELASTLOGIN":2,"DAYSSINCELASTTRADE":19,"NETREALIZEDGAINS_YTD":0,"NETREALIZEDLOSSES_YTD":251}]}}'
```

## 9. Create a Python Flask app that uses the model

You can also access the web service directly through the REST API. This allows you to use your model for inference in any of your apps. For this workshop we'll be using a Python Flask application to collect information, score it against the model, and show the results.

### Install dependencies

The general recommendation for Python development is to use a virtual environment ([`venv`](https://docs.python.org/3/tutorial/venv.html)). To install and initialize a virtual environment, use the `venv` module on Python 3 (you install the virtualenv library for Python 2.7):

In a terminal go to the cloned repo directory.

```bash
git clone https://github.com/IBM/telco-customer-churn-on-icp4d/
cd telco-customer-churn-on-icp4d
```

Initialize a virtual environment with [`venv`](https://docs.python.org/3/tutorial/venv.html).

```bash
# Create the virtual environment using Python. Use one of the two commands depending on your Python version.
# Note, it may be named python3 on your system.
python -m venv venv       # Python 3.X
virtualenv venv           # Python 2.X

# Source the virtual environment. Use one of the two commands depending on your OS.
source venv/bin/activate  # Mac or Linux
./venv/Scripts/activate   # Windows PowerShell
```

> **TIP** To terminate the virtual environment use the `deactivate` command.

Finally, install the Python requirements.

```bash
cd flaskapp
pip install -r requirements.txt
```

### Update environment variables

It's best practice to store configurable information as environment variables, instead of hard-coding any important information. To reference our model and supply an API key, we'll pass these values in via a file that is read, the key-value pairs in this files are stored as environment variables.

Copy the `env.sample` file to `.env`.

```bash
cp env.sample .env
```

Edit `.env` to reference the `URL` and `TOKEN`.

* `URL` is your web service URL for scoring.
* `TOKEN` is your deployment access token.

```bash
# Required: Provide your web service URL for scoring.
# E.g., URL=https://9.10.222.3:31843/dmodel/v1/project/pyscript/tag/score
URL=

# Required: Provide your web service deployment access token.
#           This TOKEN should start with "Bearer ".
# E.g., TOKEN=Bearer abCdwFg.fgH1r2... (and so on, tokens are long).
TOKEN=
```

### Start the application

Start the flask server by running the following command:

```bash
python telcochurn.py
```

Use your browser to go to [http://0.0.0.0:5000](http://0.0.0.0:5000) and try it out.

> **TIP**: Use `ctrl`+`c` to stop the Flask server when you are done.

# Sample output

The user inputs various values

![Input a bunch of data...](doc/source/images/input.png)

The churn percentage is returned:

![Get the churn percentage as a result](doc/source/images/score.png)

# Learn more
* **Artificial Intelligence Code Patterns**: Enjoyed this Code Pattern? Check out our other [AI Code Patterns](https://developer.ibm.com/technologies/artificial-intelligence/).
* **Data Analytics Code Patterns**: Enjoyed this Code Pattern? Check out our other [Data Analytics Code Patterns](https://developer.ibm.com/technologies/data-science/)
* **AI and Data Code Pattern Playlist**: Bookmark our [playlist](https://www.youtube.com/playlist?list=PLzUbsvIyrNfknNewObx5N7uGZ5FKH0Fde) with all of our Code Pattern videos
* **With Watson**: Want to take your Watson app to the next level? Looking to utilize Watson Brand assets? [Join the With Watson program](https://www.ibm.com/watson/with-watson/) to leverage exclusive brand, marketing, and tech resources to amplify and accelerate your Watson embedded commercial solution.
* **IBM Watson Studio**: Master the art of data science with IBM's [Watson Studio](https://www.ibm.com/cloud/watson-studio)

# License
This code pattern is licensed under the Apache Software License, Version 2.  Separate third party code objects invoked within this code pattern are licensed by their respective providers pursuant to their own separate licenses. Contributions are subject to the [Developer Certificate of Origin, Version 1.1 (DCO)](https://developercertificate.org/) and the [Apache Software License, Version 2](https://www.apache.org/licenses/LICENSE-2.0.txt).

[Apache Software License (ASL) FAQ](https://www.apache.org/foundation/license-faq.html#WhatDoesItMEAN)
