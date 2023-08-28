# AI-Reparability-scores-analysis
* The project provides a supervised and an unsupervised learning framework to evaluate the product repairability scores. An example of a smartphone dataset is used to demonstrate how the frameworks work.

## Supervised learning framwork ##
* The framework is shown in the below picture. The input is a teardown image, and the output is the repairability scores.
![alt text](https://github.com/haoyuliao/AI-Reparability-scores-analysis/blob/main/Figures/Sup.jpg?raw=true)

* Repairability scores evaluation by ResNet50 in a 3-class scale in the testing phase for (a) Samsung Galaxy S6 Edge and (b) Samsung Galaxy Note Fan Edition; Both are in the same cluster based on similarity assessment.
![alt text](https://github.com/haoyuliao/AI-Reparability-scores-analysis/blob/main/Figures/RGB.jpg?raw=true)

*Repairability scores evaluation by ResNet50 in a 3-class scale in the testing phase for Samsung Galaxy Note 20: (a) teardown image and (b) X-ray image.
![alt text](https://github.com/haoyuliao/AI-Reparability-scores-analysis/blob/main/Figures/X-ray.jpg?raw=true)

## Unsupervised learning framwork ##
* The unsupervised learning framework used ORB to extract features from teardown images before applying K-means to cluster the group. This framework is useful when the repairability scores are unknown.
![alt text](https://github.com/haoyuliao/AI-Reparability-scores-analysis/blob/main/Figures/Unsup.jpg?raw=true)

* The ORB keypoints matching results of Huawei Mate 10 Pro (left) and LG G6 (right) with 79 matching keypoints are in the same cluster.
![alt text](https://github.com/haoyuliao/AI-Reparability-scores-analysis/blob/main/Figures/Cluster.jpg?raw=true)
