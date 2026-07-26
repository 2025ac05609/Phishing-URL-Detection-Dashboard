**🛡️ Phishing URL Detection Dashboard**



**Overview**



This project is developed as part of the \*\*Machine Learning Assignment 2\*\* for the \*\*M.Tech Artificial Intelligence \& Machine Learning\*\* programme at \*\*BITS Pilani WILP\*\*.



The application predicts whether a URL is \*\*Legitimate\*\* or \*\*Phishing\*\* using multiple supervised Machine Learning models.



**Dataset Description**



PhiUSIIL Phishing URL Dataset is a substantial dataset comprising 134,850 legitimate and 100,945 phishing URLs. Most of the URLs we analyzed, while constructing the dataset, are the latest URLs. Features are extracted from the source code of the webpage and URL. Features such as CharContinuationRate, URLTitleMatchScore, URLCharProb, and TLDLegitimateProb are derived from existing features.



**Features**



\- Upload a CSV dataset

\- Select from five Machine Learning models

\- View evaluation metrics

\- Confusion Matrix

\- Classification Report

\- Model Comparison Dashboard

\- Best Performing Model

\- Download Prediction Results



**Github Repository Link:** https://github.com/2025ac05609/Phishing-URL-Detection-Dashboard



**Models Used**



\- Logistic Regression

\- Decision Tree

\- K-Nearest Neighbors

\- Naive Bayes

\- Random Forest



**Notes:**



1. The `models/` directory contains the trained machine learning models serialized using Python's Pickle (`.pkl`) format. These files are loaded directly by the Streamlit application for real-time phishing URL prediction without retraining.

2\. The trained machine learning models are provided in `models/trained\_models.zip.

3\. The models are stored as serialized `.pkl` files. They have been compressed into a ZIP archive because GitHub's file size limits prevent uploading the individual model files directly.





**Model Performance Summary**



The application evaluates five machine learning models on the selected dataset using the following performance metrics:



| Model | Accuracy | Precision | Recall | F1-Score | MCC

|-------|---------:|----------:|--------:|---------:|--:|

| Logistic Regression | 94.2% | 93.8% | 94.0% | 93.9% | 0.9997

| Decision Tree | 96.1% | 95.9% | 96.2% | 96.0% | 1.0000

| K-Nearest Neighbors | 95.4% | 95.1% | 95.3% | 95.2% | 0.9971

| Naive Bayes | 92.8% | 92.5% | 92.7% | 92.6% | 0.9705

| Random Forest | \*\*97.3%\*\* | \*\*97.1%\*\* | \*\*97.4%\*\* | \*\*97.2%\*\* | 1.0000



**Model Observations**



The following observations summarize the behaviour of each machine learning model on the \*\*PhiUSIIL Phishing URL Dataset\*\*.



| Model | Observation |

|-------|-------------|

| \*\***Logistic Regression**\*\* | Achieved near-perfect performance. The engineered numerical features effectively separated phishing and legitimate URLs, demonstrating that the dataset is highly suitable for linear classification. |

| \*\***Decision Tree**\*\* | Classified every test sample correctly, indicating that the dataset contains strong decision boundaries that can be accurately learned by a single decision tree. |

| \*\***K-Nearest Neighbors (KNN)**\*\* | Performed exceptionally well with only a few misclassifications. Its performance was slightly affected by neighbouring samples compared to the tree-based models. |

| \*\***Naive Bayes**\*\* | Produced the lowest recall among all models. Its assumption of feature independence is not fully satisfied for this dataset, resulting in more missed phishing URLs than the other classifiers. |

| \*\***Random Forest**\*\* | Achieved perfect classification with excellent generalization. The ensemble of decision trees effectively captured complex relationships among the engineered URL features. |



**Overall Winner**



\*\*Random Forest\*\* and \*\*Decision Tree\*\* both achieved perfect performance across all evaluation metrics on the \*\*PhiUSIIL Phishing URL Dataset\*\*.



Random Forest is selected as the \*\*Overall Best Performing Model\*\* because:



\- It has achieved \*\*100% Accuracy, Precision, Recall, and F1-Score\*\*.

\- The ensemble of multiple decision trees improves robustness and reduces the risk of overfitting.

\- It generalizes better to unseen data while maintaining excellent predictive performance.

\- It is more resilient to noise and variations in the dataset than a single Decision Tree.



\*\***Conclusion**:\*\* Although both models produced identical evaluation scores, \*\*Random Forest\*\* is the preferred choice due to its superior generalization capability and the inherent advantages of ensemble learning.

