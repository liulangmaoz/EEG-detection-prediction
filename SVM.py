"""
=========================================================================
                   科研级支持向量机（SVM）分类器训练脚本
=========================================================================

本脚本用于癫痫脑电信号的分类，采用严谨的科研标准进行数据处理、模型训练和评估。

主要功能：
1. 数据加载与预处理
2. 特征标准化与PCA降维（可选）
3. 网格搜索超参数优化
4. 5折交叉验证
5. 多指标模型评估
6. 多种可视化（混淆矩阵、ROC曲线、PCA可视化等）
7. 模型与结果保存

作者：AI科研助手
日期：2025
=========================================================================
"""

# 导入必要的库
import warnings

warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 机器学习库
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedKFold, cross_val_score)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_curve, auc, roc_auc_score, precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# 模型保存
import joblib
from datetime import datetime

# 设置随机种子保证可复现性
SEED = 42
np.random.seed(SEED)


class SVMResearchTrainer:
    """
    科研级SVM训练器类
    """

    def __init__(self, random_state=SEED):
        """
        初始化SVM训练器
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None
        self.best_params = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.results = {}

        # 创建结果保存目录
        self.results_dir = 'svm_results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def load_and_preprocess_data(self, excel_path):
        """
        加载和预处理数据

        参数:
            excel_path: Excel文件路径
        """
        print("=" * 70)
        print("步骤1: 数据加载与预处理")
        print("=" * 70)

        # 读取全部数据sheet
        df = pd.read_excel(excel_path, sheet_name='全部数据')
        print(f"\n✓ 数据加载成功")
        print(f"  - 数据集大小: {df.shape[0]} 样本 × {df.shape[1]} 特征")

        # 显示阶段分布
        print(f"\n  - 阶段分布:")
        print(df['阶段'].value_counts())

        # 筛选正常期和发作期数据
        # 只保留正常时期和发作期，舍去发作前期和发作后期
        df = df[df['阶段'].isin(['正常时期', '发作期'])]

        # 正常时期 -> 标签 0
        # 发作期 -> 标签 1
        df['label'] = df['阶段'].apply(lambda x: 0 if x == '正常时期' else 1)

        # 提取特征列（从第4列开始）
        feature_cols = df.columns[3:-1].tolist()  # 排除文件编号、类别、阶段和label列
        self.feature_names = feature_cols

        print(f"\n  - 特征数量: {len(feature_cols)}")
        print(f"  - 特征名称: {feature_cols}")

        # 提取特征和标签
        X = df[feature_cols].values
        y = df['label'].values

        print(f"\n  - 样本分布:")
        print(f"    - 发作间期(标签0): {np.sum(y == 0)} 样本")
        print(f"    - 发作期(标签1): {np.sum(y == 1)} 样本")

        # 处理缺失值
        if np.isnan(X).any():
            print(f"\n  - 检测到缺失值，使用均值填充")
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)

        # 保存处理后的数据集
        df_svm = pd.DataFrame(X, columns=feature_cols)
        df_svm['label'] = y
        df_svm.to_excel(os.path.join(self.results_dir, 'SVM.xlsx'), index=False)
        print(f"\n✓ 已保存处理后的数据至: {os.path.join(self.results_dir, 'SVM.xlsx')}")

        # 划分训练集和测试集（分层抽样）
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        print(f"\n  - 数据集划分:")
        print(f"    - 训练集: {self.X_train.shape[0]} 样本")
        print(f"    - 测试集: {self.X_test.shape[0]} 样本")

        # 特征标准化
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print(f"\n✓ 特征标准化完成")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def visualize_data_distribution(self):
        """
        可视化数据分布
        """
        print("\n" + "=" * 70)
        print("步骤2: 数据分布可视化")
        print("=" * 70)

        # 1. PCA降维可视化
        print("\n  - 进行PCA降维可视化...")
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(np.vstack([self.X_train, self.X_test]))
        y_all = np.hstack([self.y_train, self.y_test])

        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制散点图
        scatter = ax.scatter(X_pca[y_all == 0, 0], X_pca[y_all == 0, 1],
                             c='blue', label='发作间期 (0)', alpha=0.7, s=80, edgecolors='black', linewidth=1)
        scatter = ax.scatter(X_pca[y_all == 1, 0], X_pca[y_all == 1, 1],
                             c='red', label='发作期 (1)', alpha=0.7, s=80, edgecolors='black', linewidth=1)

        ax.set_xlabel(f'主成分1 (PC1, 解释方差: {pca.explained_variance_ratio_[0]:.3f})', fontsize=12,
                      fontweight='bold')
        ax.set_ylabel(f'主成分2 (PC2, 解释方差: {pca.explained_variance_ratio_[1]:.3f})', fontsize=12,
                      fontweight='bold')
        ax.set_title('脑电信号特征的PCA降维可视化', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'pca_visualization.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.results_dir, 'pca_visualization.pdf'), bbox_inches='tight')
        print(f"✓ PCA可视化已保存")

        # 2. 特征箱线图
        print("\n  - 绘制特征箱线图...")
        X_all = np.vstack([self.X_train, self.X_test])
        df_plot = pd.DataFrame(X_all, columns=self.feature_names)
        df_plot['label'] = y_all

        n_features = len(self.feature_names)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()

        for i, feature in enumerate(self.feature_names):
            ax = axes[i]
            sns.boxplot(x='label', y=feature, data=df_plot, ax=ax, palette=['#4472C4', '#C0504D'])
            ax.set_xlabel('类别', fontsize=10, fontweight='bold')
            ax.set_ylabel(feature, fontsize=10)
            ax.set_title(f'{feature} 分布', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        print(f"✓ 特征分布箱线图已保存")

        self.pca = pca

        return X_pca, y_all

    def train_model_with_grid_search(self):
        """
        使用网格搜索进行模型训练和超参数优化
        """
        print("\n" + "=" * 70)
        print("步骤3: SVM模型训练与网格搜索")
        print("=" * 70)

        # 定义参数网格
        param_grid = [
            # RBF核
            {'kernel': ['rbf'],
             'C': [0.01, 0.1, 1, 10, 100, 1000],
             'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]},
            # 线性核
            {'kernel': ['linear'],
             'C': [0.01, 0.1, 1, 10, 100, 1000]},
            # 多项式核
            {'kernel': ['poly'],
             'C': [0.01, 0.1, 1, 10, 100],
             'degree': [2, 3, 4],
             'gamma': ['scale', 'auto']}
        ]

        print(f"\n  - 定义参数网格:")
        print(f"    - 核函数: RBF, Linear, Poly")
        print(f"    - C值范围: 0.01 - 1000")
        print(f"    - gamma值范围: scale, auto, 0.001 - 10")

        # 初始化SVM分类器
        svm = SVC(random_state=self.random_state, probability=True, class_weight='balanced')

        # 分层5折交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # 网格搜索
        print(f"\n  - 开始网格搜索 (5折交叉验证)...")
        grid_search = GridSearchCV(estimator=svm,
                                   param_grid=param_grid,
                                   cv=cv,
                                   scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                                   refit='roc_auc',
                                   n_jobs=-1,
                                   verbose=1)

        grid_search.fit(self.X_train, self.y_train)

        # 保存最佳模型
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        print(f"\n✓ 网格搜索完成!")
        print(f"\n  - 最佳参数:")
        for key, value in self.best_params.items():
            print(f"    - {key}: {value}")

        print(f"\n  - 最佳交叉验证分数:")
        print(f"    - ROC-AUC: {grid_search.best_score_:.4f}")

        # 保存网格搜索结果
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv(os.path.join(self.results_dir, 'grid_search_results.csv'), index=False)
        print(f"\n✓ 网格搜索结果已保存")

        return self.model

    def evaluate_model(self):
        """
        评估模型性能
        """
        print("\n" + "=" * 70)
        print("步骤4: 模型评估")
        print("=" * 70)

        # 预测
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        # 计算各种指标
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=1)
        recall = recall_score(self.y_test, y_pred, zero_division=1)
        f1 = f1_score(self.y_test, y_pred, zero_division=1)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }

        print(f"\n  - 模型性能指标:")
        print(f"    - 准确率 (Accuracy):  {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"    - 精确率 (Precision): {precision:.4f} ({precision * 100:.2f}%)")
        print(f"    - 召回率 (Recall):    {recall:.4f} ({recall * 100:.2f}%)")
        print(f"    - F1值 (F1-Score):    {f1:.4f}")
        print(f"    - ROC-AUC:            {roc_auc:.4f}")

        # 详细分类报告
        print(f"\n  - 详细分类报告:")
        print(classification_report(self.y_test, y_pred, target_names=['发作间期(0)', '发作期(1)'], zero_division=1))

        # 保存评估结果
        with open(os.path.join(self.results_dir, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SVM模型评估结果\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"最佳参数: {self.best_params}\n\n")
            f.write("性能指标:\n")
            f.write(f"- 准确率 (Accuracy):  {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
            f.write(f"- 精确率 (Precision): {precision:.4f} ({precision * 100:.2f}%)\n")
            f.write(f"- 召回率 (Recall):    {recall:.4f} ({recall * 100:.2f}%)\n")
            f.write(f"- F1值 (F1-Score):    {f1:.4f}\n")
            f.write(f"- ROC-AUC:            {roc_auc:.4f}\n\n")
            f.write("详细分类报告:\n")
            f.write(classification_report(self.y_test, y_pred,
                                          target_names=['发作间期(0)', '发作期(1)'], zero_division=1))

        return y_pred, y_pred_proba

    def visualize_results(self, y_pred, y_pred_proba):
        """
        可视化结果
        """
        print("\n" + "=" * 70)
        print("步骤5: 结果可视化")
        print("=" * 70)

        # 1. 混淆矩阵
        print("\n  - 绘制混淆矩阵...")
        cm = confusion_matrix(self.y_test, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['预测: 发作间期(0)', '预测: 发作期(1)'],
                    yticklabels=['真实: 发作间期(0)', '真实: 发作期(1)'],
                    ax=ax, cbar_kws={'label': '样本数量'})

        ax.set_xlabel('预测标签', fontsize=12, fontweight='bold')
        ax.set_ylabel('真实标签', fontsize=12, fontweight='bold')
        ax.set_title('SVM模型混淆矩阵', fontsize=14, fontweight='bold', pad=20)

        # 添加百分比
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm[i, j] / np.sum(cm[i]) * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                        ha='center', va='center', fontsize=10, color='red', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.pdf'), bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存")

        # 2. ROC曲线
        print("\n  - 绘制ROC曲线...")
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='#C0504D', lw=3,
                label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='随机分类')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=12, fontweight='bold')
        ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=12, fontweight='bold')
        ax.set_title('SVM模型ROC曲线', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.results_dir, 'roc_curve.pdf'), bbox_inches='tight')
        print(f"✓ ROC曲线已保存")

        # 3. 精确率-召回率曲线
        print("\n  - 绘制精确率-召回率曲线...")
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recall_curve, precision_curve, color='#4472C4', lw=3,
                label=f'PR曲线 (AUC = {pr_auc:.4f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('召回率 (Recall)', fontsize=12, fontweight='bold')
        ax.set_ylabel('精确率 (Precision)', fontsize=12, fontweight='bold')
        ax.set_title('SVM模型精确率-召回率曲线', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
        print(f"✓ PR曲线已保存")

        # 4. 决策边界可视化 (使用PCA降维到2D)
        print("\n  - 绘制决策边界...")
        if self.pca is not None:
            # 使用训练好的PCA降维
            X_test_pca = self.pca.transform(self.X_test)

            # 创建网格
            h = 0.02
            x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
            y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            # 对网格进行预测（需要先通过PCA反向变换到原始空间再预测）
            grid_pca = np.c_[xx.ravel(), yy.ravel()]
            grid_original = self.pca.inverse_transform(grid_pca)
            Z = self.model.predict(grid_original)
            Z = Z.reshape(xx.shape)

            fig, ax = plt.subplots(figsize=(12, 10))

            # 绘制决策边界
            ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)
            ax.contour(xx, yy, Z, colors='k', linewidths=1.5, levels=[0.5])

            # 绘制测试集样本
            scatter = ax.scatter(X_test_pca[self.y_test == 0, 0], X_test_pca[self.y_test == 0, 1],
                                 c='blue', label='真实: 发作间期(0)', alpha=0.8, s=100, edgecolors='black',
                                 linewidth=1.5)
            scatter = ax.scatter(X_test_pca[self.y_test == 1, 0], X_test_pca[self.y_test == 1, 1],
                                 c='red', label='真实: 发作期(1)', alpha=0.8, s=100, edgecolors='black', linewidth=1.5)

            # 标记预测错误的样本
            error_idx = y_pred != self.y_test
            if np.sum(error_idx) > 0:
                ax.scatter(X_test_pca[error_idx, 0], X_test_pca[error_idx, 1],
                           s=200, facecolors='none', edgecolors='gold', linewidths=3, label='预测错误')

            ax.set_xlabel(f'主成分1 (PC1)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'主成分2 (PC2)', fontsize=12, fontweight='bold')
            ax.set_title('SVM模型决策边界可视化', fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'decision_boundary.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.results_dir, 'decision_boundary.pdf'), bbox_inches='tight')
            print(f"✓ 决策边界可视化已保存")

        # 5. 性能指标柱状图
        print("\n  - 绘制性能指标柱状图...")
        metrics = ['准确率', '精确率', '召回率', 'F1值', 'ROC-AUC']
        values = [self.results['accuracy'], self.results['precision'],
                  self.results['recall'], self.results['f1'], self.results['roc_auc']]
        colors = ['#4472C4', '#70AD47', '#FFC000', '#C0504D', '#7030A0']

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('评估指标', fontsize=12, fontweight='bold')
        ax.set_ylabel('分数', fontsize=12, fontweight='bold')
        ax.set_title('SVM模型性能指标', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')

        # 在柱子上添加数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        print(f"✓ 性能指标图已保存")

        return cm

    def save_model_and_results(self):
        """
        保存模型和结果
        """
        print("\n" + "=" * 70)
        print("步骤6: 保存模型与结果")
        print("=" * 70)

        # 保存模型
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'results': self.results
        }

        model_path = os.path.join(self.results_dir, 'svm_trained_model.pkl')
        joblib.dump(model_data, model_path)
        print(f"\n✓ 模型已保存至: {model_path}")

        # 保存结果摘要
        summary = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_parameters': self.best_params,
            'test_accuracy': self.results['accuracy'],
            'test_precision': self.results['precision'],
            'test_recall': self.results['recall'],
            'test_f1': self.results['f1'],
            'test_roc_auc': self.results['roc_auc']
        }

        summary_path = os.path.join(self.results_dir, 'training_summary.csv')
        pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"✓ 训练摘要已保存至: {summary_path}")

        # 生成README文件
        readme_content = """
=========================================================================
                    SVM模型训练结果说明文档
=========================================================================

本目录包含了癫痫脑电信号SVM分类模型的完整训练结果。

文件说明：
----------
1. SVM.xlsx
   - 预处理后的数据集，包含特征和标签
   - 标签0：发作间期（正常时期）
   - 标签1：发作期

2. svm_trained_model.pkl
   - 训练好的SVM模型文件
   - 包含：模型、标准化器、PCA对象、特征名称、最佳参数等
   - 使用joblib库加载

3. 可视化图表（.png和.pdf格式）
   - pca_visualization: 数据PCA降维可视化
   - feature_distributions: 特征分布箱线图
   - confusion_matrix: 混淆矩阵
   - roc_curve: ROC曲线
   - pr_curve: 精确率-召回率曲线
   - decision_boundary: 决策边界可视化
   - performance_metrics: 性能指标柱状图

4. 其他文件
   - grid_search_results.csv: 网格搜索详细结果
   - evaluation_results.txt: 模型评估结果
   - training_summary.csv: 训练摘要

模型使用方法：
--------------
import joblib

# 加载模型
model_data = joblib.load('svm_trained_model.pkl')
model = model_data['model']
scaler = model_data['scaler']

# 预测新数据
# X_new: 新数据 (n_samples × n_features)
X_new_scaled = scaler.transform(X_new)
y_pred = model.predict(X_new_scaled)
y_pred_proba = model.predict_proba(X_new_scaled)

=========================================================================
"""

        with open(os.path.join(self.results_dir, 'README.txt'), 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"\n✓ 所有结果已保存在 '{self.results_dir}' 目录中")


def main():
    """
    主函数 - 执行完整的SVM训练流程
    """
    print("\n" + "=" * 70)
    print("           科研级SVM分类器 - 癫痫脑电信号识别")
    print("=" * 70)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 初始化训练器
    trainer = SVMResearchTrainer()

    # 1. 加载和预处理数据
    trainer.load_and_preprocess_data('SVM输入变量.xlsx')

    # 2. 数据可视化
    trainer.visualize_data_distribution()

    # 3. 训练模型
    trainer.train_model_with_grid_search()

    # 4. 评估模型
    y_pred, y_pred_proba = trainer.evaluate_model()

    # 5. 结果可视化
    trainer.visualize_results(y_pred, y_pred_proba)

    # 6. 保存模型和结果
    trainer.save_model_and_results()

    print("\n" + "=" * 70)
    print("✓ 所有步骤已完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()