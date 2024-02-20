# AI-Reparability-scores-analysis
* The project provides a supervised and unsupervised learning framework to evaluate the product repairability scores for academic research. An example of a smartphone dataset is used to demonstrate how the frameworks work. The models are built on pytorch and trained by GPU.

## Install Dependencies ##
* Python3
* matplotlib
* pandas
* opencv-python
* torch
* torchvision
* openpyxl
* pytorch

## Supervised learning framwork ##
* The framework is shown in the below picture. The input is a teardown image, and the output is the repairability scores.
![alt text](https://github.com/haoyuliao/AI-Reparability-scores-analysis/blob/main/Figures/Sup.jpg?raw=true)

* Repairability scores evaluation by ResNet50 in a 3-class scale in the testing phase for (left) Samsung Galaxy S6 Edge and (right) Samsung Galaxy Note Fan Edition; Both are in the same cluster based on similarity assessment.
![alt text](https://github.com/haoyuliao/AI-Reparability-scores-analysis/blob/main/Figures/RGB.jpg?raw=true)

* Repairability scores evaluation by ResNet50 in a 3-class scale in the testing phase for Samsung Galaxy Note 20: (left) teardown image and (right) X-ray image.
![alt text](https://github.com/haoyuliao/AI-Reparability-scores-analysis/blob/main/Figures/X-ray.jpg?raw=true)

## Unsupervised learning framwork ##
* The unsupervised learning framework used ORB to extract features from teardown images before applying K-means to cluster the group. This framework is useful when the repairability scores are unknown.
![alt text](https://github.com/haoyuliao/AI-Reparability-scores-analysis/blob/main/Figures/Unsup.jpg?raw=true)

* The ORB keypoints matching results of Huawei Mate 10 Pro (left) and LG G6 (right) with 79 matching keypoints are in the same cluster.
![alt text](https://github.com/haoyuliao/AI-Reparability-scores-analysis/blob/main/Figures/Cluster.jpg?raw=true)
<!---
## Trained models and dataset ##
Since the trained models and dataset are too big to upload the GitHub. Both are available by downloading the link:
[Trained models and dataset](https://uflorida-my.sharepoint.com/:f:/g/personal/haoyuliao_ufl_edu/EtKgjcOmIU1Hv-EwSLktHkQBo7D_Jlu3Da_ieYM9SREgjA?e=KLFdfC)
--->
## Citation ##
If you use the packages, please cite the paper by the following BibTex:
```
@article{liao2024automated,
  title={Automated Evaluation and Rating of Product Repairability Using Artificial Intelligence-Based Approaches},
  author={Liao, Hao-Yu and Esmaeilian, Behzad and Behdad, Sara},
  journal={Journal of Manufacturing Science and Engineering},
  volume={146},
  number={2},
  year={2024},
  publisher={American Society of Mechanical Engineers Digital Collection}
}
```
