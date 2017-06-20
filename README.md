# Merge and Acquisitions Prediction

## Rational

As
[Wikipedia says](https://en.wikipedia.org/wiki/Mergers_and_acquisitions):
"Mergers and acquisitions (M&A) are transactions in which the
ownership of companies, other business organizations or their
operating units are transferred or combined. As an aspect of strategic
management, M&A can allow enterprises to grow, shrink, and change the
nature of their business or competitive position."

Many traders and investors are interested in predicting M&As in order
to adjust their strategy according to an M&A upcoming event.

Having a rationally big dataset of company and M&A information we can
try to analyze it and build a predictive model using different machine
learning (ML) techniques. The model can be wrapped in a web
application available for traders and investors.

## Our Goals

* To collect a dataset of company and M&A information from Crunchbase
  (see the next section).
* Analyze the dataset and build a predictive model of M&A.
* Wrap the model into a web app that will display predictions of
  future M&A.

## Crunchbase

[Crunchbase](https://www.crunchbase.com/) is a database of the startup
ecosystem consisting of investors, incubators and start-ups, which
comprises around 500,000 data points profiling companies, people,
funds, fundings and events.

The good thing about Crunchbase is it provides a full-fledged API to
the start-ups data it has. Please
refer [this page](https://data.crunchbase.com/) to learn more about
the API.

A paid Crunchbase account can be provided by Crystalnix if it is
necessary.

## Project Stages

The project will consist of three major stages each of which depends
on the previous one:

1. Collecting the data.
2. Building the model.
3. Implementing a web app.

## Collecting the Data

The data is supposed to be collected by using Crunchbase API. Less
interesting but still viable option is to
use
[the full 2013 dataset](https://data.crunchbase.com/docs/getting-started#basic-access).

## Building the Model

It would be nice to avoid using a lot of computational power (such as
Amazon EMR) to produce the model. However we are not sure how large
the dataset is going to be. In case additional computational power is
necessary it is advised to contact to the project manager responsible
for the project.

## Implementing a Web App

The web app can be described by the following user stories:

* As a user I can register to the application using my email and
  password or by using a social account (Google, Facebook, Twitter,
  etc).
* As a user after login I can see the dashboard with company names and
  probabilities they will be acquired.

Example of the app dashboard:

| Comany Name               | Chance to Be Acquired, %   |
|---------------------------|----------------------------|
| Religare Health insurance | 69.97                      |
| Supremia Grup             | 9.16                       |
| Alpine Jan-San            | 55.58                      |
| Fitzroy Communications    | 9.99                       |
| Connextions, Inc.         | 40.04                      |


No design work for the first version of the app is considered.

## Technologies

The preferable technologies are:

* Python scientific stack for data analysis and ML.
* Django for the web app.
* Deploy on Amazon Web Services infrastructure.

The requirements are not touch and are subject to discussion.

## Assumptions

* It is possible to get data from Crunchbase via the REST API.
* The dataset is reasonably small in informative to be processed on a
  single machine.
* Collecting and processing data doesn't violate any Crunchbase
  license terms.

## Our Approach

In case of any issues they should be reported as soon as possible. We
prefer to make and spot mistakes quickly while it's cheap to do them
and adjust the direction.


## Application
For purpose of our work CLI application has been developed. For run application just use 'python acquisition_prediction.py'.

### Installation
For install application clone this repository in folder you need and run 'pip install -r requirements.txt'.

### Usage
acquisition_prediction.py [-h] [--extract | --fit] [--user USER]
                                 [--password PASSWORD] [--scheme SCHEME]

| Option                | Description                                                                                                                          |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| --extract             | Extract features from database and save them in files located by paths in settings.py. Also you should specify database credentials. |
| --fit                 | Build prediction model and print best params and scores.                                                                             |

Database credentials:

| Credential          | Description            |
|---------------------|------------------------|
| --user USER         | Database user          |
| --password PASSWORD | Database user password |
| --scheme SCHEME     | Database scheme        |