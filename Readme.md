#步骤说明

###计算label分布

    1.运行label_analysis.py，得到[[labelid，num]]的list
    2.更改label_distribution_echarts.js中dataset为list
    2.在echarts官网中运行label_distribution_echarts.js内容
    
###将原始数据分类，音频文件按类别存储

    运行datautils的label_process函数
 
###处理音频文件，将梅尔频谱图以npy格式存储
    
    运行proces_data
    
#细节说明

* mel文件特征：  96 * maxtime_length
   
   `(1, 96, 636, 1)`  BxMxTxC M是mel滤波器组数，T是音频最长长度

* 训练使用类别是原类别经过了重新编码的。因为类别读取是根据类别文件名list设置的，并且类别文件本身会有某些类别空缺