import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 加载数据 (默认读取当前目录下的 train.csv)
def load_titanic_data(csv_path="train.csv"):
    data = pd.read_csv(csv_path)

    required_cols = ['Survived', 'Pclass', 'Sex', 'Age']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        missing_str = ', '.join(missing_cols)
        raise ValueError(f'CSV 缺少必要列: {missing_str}')

    data = data[required_cols].copy()
    return data

df = load_titanic_data()

# 2. 特征工程Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# 将 Sex 转换为 0/1 (Female=1)
df = df.dropna(subset=['Sex'])
df['IsFemale'] = df['Sex'].map({'female': 1, 'male': 0})

# 填充 Age 缺失值 (简单均值填充) 并 标准化
df['Age'] = df['Age'].fillna(df['Age'].mean())
scaler = StandardScaler()
df['Age_scaled'] = scaler.fit_transform(df[['Age']])

# Pclass 标准化 (使其均值为0，方差为1，方便应用通用先验)
df['Pclass_scaled'] = scaler.fit_transform(df[['Pclass']])

# 准备输入矩阵
X = df[['IsFemale', 'Pclass_scaled', 'Age_scaled']]
y = df['Survived'].values

print(f"数据准备完毕: {X.shape}")

with pm.Model() as titanic_model:
    # --- A. 定义先验 (Priors) ---
    # 依据上一轮的历史假设设定参数
    
    # Intercept: 均值 -0.5 (基准生存率偏低), 方差 2.0 -> sigma ≈ 1.41
    beta_0 = pm.Normal('Intercept', mu=-0.5, sigma=np.sqrt(2.0))
    
    # Sex (IsFemale): 均值 1.5 (女士优先), 方差 1.0 -> sigma = 1.0
    beta_sex = pm.Normal('IsFemale', mu=1.5, sigma=1.0)
    
    # Pclass: 均值 -0.5 (舱位等级数值越高，生存概率越低), 方差 1.0 -> sigma = 1.0
    beta_pclass = pm.Normal('Pclass', mu=-0.5, sigma=1.0)
    
    # Age: 均值 -0.2 (稍偏向年轻人), 方差 1.0 -> sigma = 1.0
    beta_age = pm.Normal('Age', mu=-0.2, sigma=1.0)
    
    # --- B. 线性组合 (Linear Combination) ---
    # shared数据容器以便后续预测替换
    X_sex = pm.MutableData('X_sex', X['IsFemale'].values)
    X_pclass = pm.MutableData('X_pclass', X['Pclass_scaled'].values)
    X_age = pm.MutableData('X_age', X['Age_scaled'].values)
    
    # 计算 Log-odds
    mu = beta_0 + beta_sex * X_sex + beta_pclass * X_pclass + beta_age * X_age
    
    # --- C. 似然函数 (Likelihood) ---
    # 使用 logit_p 参数，PyMC 会自动应用 Sigmoid 函数，数值上更稳定
    y_obs = pm.Bernoulli('y_obs', logit_p=mu, observed=y)

    # 查看模型结构
    pm.model_to_graphviz(titanic_model)
    
with titanic_model:
    # 采样设置：
    # draws: 正式采样次数
    # tune:用于调整采样器步长的预热次数 (会被丢弃)
    # chains: 并行链的数量，用于检测收敛性
    idata = pm.sample(draws=2000, tune=1000, chains=2, random_seed=42)

print("采样完成。")

# 绘制轨迹图
az.plot_trace(idata, var_names=['Intercept', 'IsFemale', 'Pclass', 'Age'])
plt.tight_layout()
plt.show()

# 打印详细统计摘要
# r_hat: 潜在尺度缩减因子。若 r_hat > 1.05，说明链未收敛，模型不可用。
summary = az.summary(idata)
print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']])

# 展示特定参数的后验分布，并显示 94% 最高密度区间 (HDI)
az.plot_posterior(idata, var_names=['IsFemale', 'Pclass', 'Age'], ref_val=0)
plt.show()