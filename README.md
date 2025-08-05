# MoE-VQ
1.NoWag/models/datasets/generate_U.py  产生hessian的SVD分解后的U
2.NoWag/SVD/generate_WU.py 对原始的W融合U
3.1）NoWag/SVD/WU/generate_SVD.py 对WU进行SVD得到完整的USV，NoWag/SVD/WU/reconstruct.py 低秩分解后重建 （USV内存占用很大）
  2）NoWag/SVD/WU/straight_reconstruct.py 直接得到低秩分解并且重建后的W'
4.NoWag/SVD/WU/generate_residual.py 得到残差
5.然后可以把models/datasets模型中的original_weights直接换成残差，然后进行VQ
6.NoWag/models/datasets/add_residual.py 残差VQ结果与低秩重建结果相加 
