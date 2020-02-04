---
title: 'ASCENDS: Advanced data SCiENce toolkit for Non-Data Scientists'
tags:
  - machine learning
  - artificial intelligence
  - neural network
authors:
  - name: Sangkeun Lee
    orcid: 0000-0002-1317-5112
    affiliation: 1
  - name: Jian Peng
    orcid: 0000-0002-9763-4741
    affiliation: 1
  - name: Andrew Williams
    affiliation: 2
  - name: Dongwon Shin
    orcid: 0000-0002-5797-3423
    affiliation: 1
affiliations:
  - name: Oak Ridge National Laboratory
    index: 1
  - name: Cornell University
    index: 2
date: 23 Jul 2019
bibliography: paper.bib
---

# Acknowledgements

This manuscript has been authored by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non- exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).

The research was supported by the U. S. Department of Energy, Office of Energy Efficiency and Renewable Energy, Vehicle Technologies Office, Propulsion Materials Program.

# Summary

Recently, advances in machine learning and artificial intelligence have been playing more and more critical roles in a wide range of areas. For the last several years, industries have shown that how learning from data, identifying patterns and making decisions with minimal human intervention can be extremely useful to their business (e.g., image classification, recommending a product to a customer, finding friends in a social network, predicting customers actions, etc.). These success stories have been motivating scientists who study physics, chemistry, materials, medicine, and many other subjects, to explore a new pathway of utilizing machine learning techniques like regression and classification for their scientific activities. However, most existing machine learning tools, systems, and methodologies have been developed for programming experts but not for scientists (or any users) who have no or little knowledge of programming. 

ASCENDS is a toolkit that is developed to assist scientists (or any persons) who want to use their data for machine learning tasks, more specifically, correlation analysis, regression, and classification. ASCENDS does not require programming skills.  Instead, it provides a set of simple but powerful CLI (Command Line Interface) and GUI (Graphic User Interface) tools for non-data scientists to be able to intuitively perform advanced data analysis and machine learning techniques. ASCENDS has been implemented by wrapping around open-source software including Keras [@gulli2017deep], TensorFlow [@abadi2016tensorflow], and scikit-learn [@pedregosa2011scikit].

![Using Ascends via its web-based graphic user interface](./logo/web-ui.png)

![Using Ascends via its command-line interface](./logo/command-line-ui.png)

Users can perform three major tasks using ASCENDS as follows. First of all, users can easily perform correlation analysis [@ezekiel1930methods] using ASCENDS. ASCENDS can quantify the correlation between input variables ($X$) and an output variable ($y$) using various correlation coefficients including Pearson's correlation coefficient [@sedgwick2012pearson] and maximal information coefficient [@reshef2011detecting]. Second, users can train, test, save and utilize classification models [@ren2003learning] without any programming efforts. For instance, with ASCENDS, by executing a single command in a terminal, a user can train a model for predicting whether an email is a spam or not-spam. Last, similarly, users can train, test, save and use regression models [@darlington1990regression]. For instance, ASCENDS can be used to train a model for predicting the value of a house based on square footage, number of bedrooms, number of cars that can be parked in its garages, and amount of storage using the provided graphic user interface in a web browser.

Earlier versions of ASCENDS have been used for scientific research such as @shin2019modern, @shin2017petascale, and @wang2019machine. 

# References

