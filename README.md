This code is used in publication 
T. T. Tran, V. Snasel and L. T. Nguyen, "Combining Social Relations and Interaction Data in Recommender System With Graph Convolution Collaborative Filtering," in IEEE Access, vol. 11, pp. 139759-139770, 2023, doi: 10.1109/ACCESS.2023.3340209

# Introduction - GCCF
* GCCF : Our final model, in which interation data and social friendship data are aggregated. Which is publication "Combining social relations and interaction data in Recommender System with Graph Convolution Collaborative Filtering" - IEEE Access.
* CombiGCN: Our model with only interaction between users and items take into account.
* BPRMF: Koren, Y., Bell, R. & Volinsky, C. Matrix Factorization Techniques for Recommender Systems. Computer. 42, 30-37 (2009), https://ieeexplore.ieee.org/document/5197422
* GCMC : Berg, R., Kipf, T. & Welling, M. Graph Convolutional Matrix Completion. (2017), https://arxiv.org/abs/1706.02263 
* LightGCN : He, X., Deng, K., Wang, X., Li, Y., Zhang, Y. & Wang, M. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (2020), https://arxiv.org/pdf/2002.02126
* NGCF : Wang, X., He, X., Wang, M., Feng, F. & Chua, T. Neural Graph Collaborative Filtering. CoRR. abs/1905.08108 (2019), http://arxiv.org/abs/1905.08108
* SEPT : Yu, J., Yin, H., Gao, M., Xia, X., Zhang, X. & Hung, N. Socially Aware Self-Supervised Tri-Training for Recommendation. CoRR. abs/2106.03569 (2021), https://arxiv.org/abs/2106.03569
* SocialLLGN : Liao, J., Zhou, W., Luo, F., Wen, J., Gao, M., Li, X. & Zeng, J. SocialLGN: Light graph convolution network for social  ecommendation. Information Sciences. 589 pp. 595-607 (2022), https://www.sciencedirect.com/science/article/pii/S0020025522000019
* SocialfnLGN : SocialLGN remove weighted matrices and non-linear activation function.

## Acknowledgement
* We thank our colleagues and reviewers for their comments.
* This is our improvement Tensorflow based on code of https://github.com/kuandeng/LightGCN
  
## Environment Requirement
The code has been tested running under Python 3.10.9 The required packages are as follows:
* tensorflow == 2.11.0
* numpy == 1.24.3
* scipy == 1.9.0
* sklearn == 1.2.0

## Examples to run a 3-layer CombiGCN

### Gowalla dataset
* Command
```
python GCCF.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 8192 --epoch 10000

```

### Librarything dataset
* Command
```
python GCCF.py --dataset librarything --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 8192 --epoch 10000

```

### Ciao dataset
* Command
```
python GCCF.py --dataset ciao --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 10000
```

### Epinions dataset
* Command
```
python GCCF.py --dataset epinions --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 10000
```



NOTE : the duration of training and testing depends on the running environment.
## Dataset
We provide four processed datasets: Gowalla, Librarything, Ciao and Epinions.
* `train.txt`
  * Train file.
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.

* `test.txt`
  * Test file (positive instances).
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.
  * Note that here we treat all unobserved interactions as the negative instances when reporting performance.
  
* `user_list.txt`
  * User file.
  * Each line is a triplet (org_id, remap_id) for one user, where org_id and remap_id represent the ID of the user in the original and our datasets, respectively.
  
* `item_list.txt`
  * Item file.
  * Each line is a triplet (org_id, remap_id) for one item, where org_id and remap_id represent the ID of the item in the original and our datasets, respectively.


=======
