# DDTCDR

This is our implementation for the paper:

**Pan Li and Alexander Tuzhilin. "DDTCDR: Deep Dual Transfer Cross Domain Recommendation." Proceedings of the 13th International Conference on Web Search and Data Mining. 2020.** [[Paper]](https://dl.acm.org/doi/abs/10.1145/3336191.3371793)

**Important:** Due to the confidential agreement with the company, we are not allowed to make the dataset publicly available. Nevertheless, we provide a sample of the dataset for you to get an understanding of the input strcuture. You are always welcome to use our codes for your own dataset.

**Please cite our WSDM'20 paper if you use our codes. Thanks!** 

Author: Pan Li (https://lpworld.github.io/)

## Environment Settings
We use PyTorch as the backend. 
- PyTorch version:  '1.2.0'

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the parse_args function). 

Run DDTCDR:
```
python train.py
```

## Acknowledgement
This implementation is inspired from [Neural Collaborative Filtering](https://github.com/hexiangnan/neural_collaborative_filtering). The authors would also like to thank Vladimir Bobrikov for providing the dataset for evaluation purposes.

Last Update: 2020/02/16
