import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略一些不必要的警告
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(csv_path, is_train=True):
    """
    加载数据。如果是训练集，会检查 Survived 列。
    """
    data = pd.read_csv(csv_path)
    
    # 基础必要列
    required_cols = ['Pclass', 'Sex', 'Age', 'PassengerId']
    if is_train:
        required_cols.append('Survived')

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f'CSV {csv_path} 缺少必要列: {", ".join(missing_cols)}')

    return data[required_cols].copy()

def preprocess_data(df, artifacts=None):
    """
    数据预处理函数。
    artifacts: 包含训练集统计信息（均值、Scaler）的字典。
               如果是训练阶段，传入 None，函数会计算并返回这些信息。
               如果是测试阶段，传入训练阶段返回的 artifacts。
    """
    df = df.copy()
    
    # 1. 处理 Sex (将 female 映射为 1, male 映射为 0)
    # 注意：在真实预测任务中，测试集不应丢弃行，这里假设 Sex 无缺失或需手动填补
    if df['Sex'].isnull().any():
        print("警告: 数据集中存在 Sex 缺失值，建议填补而不是删除。本次演示将填充为 'male'")
        df['Sex'] = df['Sex'].fillna('male')
        
    df['IsFemale'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)

    # 2. 处理 Age
    if artifacts is None:
        # --- 训练阶段 ---
        age_mean = df['Age'].mean()
        scaler_age = StandardScaler()
        scaler_pclass = StandardScaler()
        
        # 填充缺失值
        df['Age'] = df['Age'].fillna(age_mean)
        
        # Fit 并 Transform
        df['Age_scaled'] = scaler_age.fit_transform(df[['Age']])
        df['Pclass_scaled'] = scaler_pclass.fit_transform(df[['Pclass']])
        
        # 保存统计信息供测试集使用
        new_artifacts = {
            'age_mean': age_mean,
            'scaler_age': scaler_age,
            'scaler_pclass': scaler_pclass
        }
        return df, new_artifacts
    else:
        # --- 测试阶段 ---
        # 使用训练集的均值填充
        df['Age'] = df['Age'].fillna(artifacts['age_mean'])
        
        # 使用训练集的 Scaler 进行 Transform (绝对不能重新 fit)
        df['Age_scaled'] = artifacts['scaler_age'].transform(df[['Age']])
        df['Pclass_scaled'] = artifacts['scaler_pclass'].transform(df[['Pclass']])
        
        return df

def main():
    # ==========================
    # 1. 训练阶段
    # ==========================
    print("正在加载训练数据...")
    df_train = load_data("train.csv", is_train=True)
    
    # 清洗数据并获取统计特征 (artifacts)
    # 注意：原代码直接 dropna Sex，这里为了简单沿用，但建议保留所有行
    df_train = df_train.dropna(subset=['Sex']) 
    df_train_processed, artifacts = preprocess_data(df_train, artifacts=None)

    # 准备训练输入
    X_train = df_train_processed[['IsFemale', 'Pclass_scaled', 'Age_scaled']]
    y_train = df_train_processed['Survived'].values

    print(f"训练数据准备完毕: {X_train.shape}")

    with pm.Model() as titanic_model:
        # --- A. 定义先验 (Priors) ---
        beta_0 = pm.Normal('Intercept', mu=-0.5, sigma=np.sqrt(2.0))
        beta_sex = pm.Normal('IsFemale', mu=1.5, sigma=1.0)
        beta_pclass = pm.Normal('Pclass', mu=-0.5, sigma=1.0)
        beta_age = pm.Normal('Age', mu=-0.2, sigma=1.0)
        
        # --- B. 输入容器 (Mutable Data) ---
        # 关键：使用 pm.Data 定义输入，以便后续替换为测试集数据
        X_sex = pm.Data('X_sex', X_train['IsFemale'].values)
        X_pclass = pm.Data('X_pclass', X_train['Pclass_scaled'].values)
        X_age = pm.Data('X_age', X_train['Age_scaled'].values)
        
        # --- C. 线性组合 & 似然 ---
        mu = beta_0 + beta_sex * X_sex + beta_pclass * X_pclass + beta_age * X_age
        pm.Deterministic('survival_prob', pm.math.sigmoid(mu))

        # 观测值
        y_obs = pm.Bernoulli('y_obs', logit_p=mu, observed=y_train)

        # --- D. 采样 ---
        print("开始 MCMC 采样...")
        idata = pm.sample(draws=2000, tune=1000, chains=2, random_seed=42)

    # 展示训练结果
    az.plot_trace(idata, var_names=['Intercept', 'IsFemale', 'Pclass', 'Age'])
    plt.tight_layout()
    plt.show()
    
    print("\n--- 训练集参数摘要 ---")
    print(az.summary(idata, var_names=['Intercept', 'IsFemale', 'Pclass', 'Age']))

    # ==========================
    # 2. 测试/预测阶段
    # ==========================
    print("\n正在加载测试数据 (test.csv)...")
    try:
        df_test = load_data("test.csv", is_train=False)
    except FileNotFoundError:
        print("未找到 test.csv，请确保文件存在。")
        return

    # 使用训练集的 artifacts 处理测试集
    df_test_processed = preprocess_data(df_test, artifacts=artifacts)
    
    # 准备测试输入
    X_test_sex = df_test_processed['IsFemale'].values
    X_test_pclass = df_test_processed['Pclass_scaled'].values
    X_test_age = df_test_processed['Age_scaled'].values

    print(f"测试数据准备完毕: {df_test_processed.shape}")

    with titanic_model:
        # 1. 更新数据容器的内容为测试集数据
        pm.set_data({
            'X_sex': X_test_sex,
            'X_pclass': X_test_pclass,
            'X_age': X_test_age
        })
        
        # 2. 进行后验预测 (Posterior Predictive Sampling)
        # 这会利用之前采样得到的参数分布，结合新的输入数据，生成 y 的预测分布
        print("正在进行后验预测...")
        post_pred = pm.sample_posterior_predictive(idata, var_names=['survival_prob'], random_seed=42)

    # ==========================
    # 3. 生成提交文件
    # ==========================
    # post_pred.posterior_predictive['y_obs'] 的形状通常是 (chains, draws, n_test_samples)
    # 我们需要对 chains 和 draws 取平均值，得到每个样本为 1 (Survived) 的概率
    
    # 获取预测矩阵 (PyMC 版本差异处理)
    if 'survival_prob' in post_pred.posterior_predictive:
        prob_samples = post_pred.posterior_predictive['survival_prob']
    else:
        prob_samples = post_pred['survival_prob']

    # 计算平均生存概率
    # mean(axis=(0, 1)) 表示对 链(chain) 和 抽样(draw) 维度求平均
    y_prob_mean = prob_samples.mean(dim=["chain", "draw"]).values
    
    # 阈值判定 (通常 > 0.5 判为生存)
    predictions = (y_prob_mean > 0.5).astype(int)

    # 创建提交 DataFrame
    submission = pd.DataFrame({
        'PassengerId': df_test_processed['PassengerId'],
        'Survived': predictions,
        'Survival_Prob': y_prob_mean # 额外保存概率以便分析
    })

    output_file = "submission.csv"
    submission.to_csv(output_file, index=False)
    
    print("\n--- 预测完成 ---")
    print(submission.head())
    print(f"\n结果已保存至: {output_file}")
    
    # 可视化前几个测试样本的预测概率分布（可选）
    # 比如看第一个乘客的生存概率分布
    plt.figure()
    passenger_samples = prob_samples.isel(survival_prob_dim_0=0).values.flatten()
    plt.hist(passenger_samples, bins=30, alpha=0.7)
    plt.title("Posterior Predicted Probability for 1st Test Passenger")
    plt.show()

if __name__ == "__main__":
    main()