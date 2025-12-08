import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

scaler = StandardScaler()
def make_prediction(model, trace, new_data):
    """
    使用后验预测分布进行推断
    """
    # 预处理新数据 (必须与训练集使用相同的 Scaler!)
    new_sex = new_data['Sex'].map({'female': 1, 'male': 0}).values
    new_age = scaler.transform(new_data[['Age']].fillna(30)) # 简化处理
    new_pclass = scaler.transform(new_data[['Pclass']]) # 简化处理
    
    with model:
        # 替换模型中的数据容器
        pm.set_data({
            'X_sex': new_sex, 
            'X_pclass': new_pclass.flatten(), 
            'X_age': new_age.flatten()
        })
        
        # 生成后验预测样本
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)
    
    # ppc['posterior_predictive']['y_obs'] 的形状通常是 (chains, draws, n_samples)
    # 我们取平均值得到生存概率
    posterior_preds = ppc.posterior_predictive['y_obs'].mean(dim=["chain", "draw"])
    
    return posterior_preds

# 模拟一条测试数据 (3等舱, 22岁, 男性) -> 预期生存率极低
test_sample = pd.DataFrame({
    'Sex': ['male'],
    'Pclass': [3],
    'Age': [22]
})

predicted_prob = make_prediction(titanic_model, idata, test_sample)
print(f"该乘客的平均生存概率: {float(predicted_prob):.4f}")