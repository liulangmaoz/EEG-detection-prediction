import os
import pandas as pd
import numpy as np

# ===================== 路径配置 =====================
# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 输入文件夹：癫痫数据和正常数据的结果目录
EPILEPSY_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "ictal_epilepsy")
NORMAL_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "interictal_normal")

# 输出目录：汇总结果保存位置
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "aggregated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输出文件名
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "eeg_data_aggregated.xlsx")
# =====================================================

def round_3(x):
    """保留三位小数"""
    if pd.isna(x):
        return x
    return round(float(x), 3)

def process_epilepsy_folder(folder_path, folder_name):
    """处理单个癫痫文件夹"""
    
    # ---------------------- 文件名 ----------------------
    file_energy = os.path.join(folder_path, "03发作各阶段能量占比.xlsx")
    file_power = os.path.join(folder_path, "04发作各阶段主频功率.xlsx")
    file_entropy = os.path.join(folder_path, "05发作各阶段功率谱熵.xlsx")
    feat_file = os.path.join(folder_path, "Seizure_Features_Summary.xlsx")
    
    # 读取数据
    df_energy = pd.read_excel(file_energy) if os.path.exists(file_energy) else pd.DataFrame()
    df_power = pd.read_excel(file_power) if os.path.exists(file_power) else pd.DataFrame()
    df_entropy = pd.read_excel(file_entropy) if os.path.exists(file_entropy) else pd.DataFrame()
    
    # ====================== 全局特征读取 ======================
    d2, pd2, le, ke = np.nan, np.nan, np.nan, np.nan
    if os.path.exists(feat_file):
        try:
            # 安全读取 + 指定引擎
            df_feat = pd.read_excel(
                feat_file,
                sheet_name="全局特征平均值汇总",
                engine="openpyxl"
            )
            
            # 清理特征名称：转字符串 + 去空格
            df_feat["特征名称"] = df_feat["特征名称"].astype(str).str.strip()
            
            for _, r in df_feat.iterrows():
                name = r["特征名称"]
                
                if "D2关联维数" in name:
                    d2 = r["全局平均值"]
                if "PD2点关联维" in name:
                    pd2 = r["全局平均值"]
                if "LE李雅普诺夫指数" in name:
                    le = r["全局平均值"]
                if "KE科尔莫戈罗夫熵" in name:
                    ke = r["全局平均值"]
                    
        except Exception as e:
            print(f"[{folder_name}] 读取特征文件失败：{str(e)}")
    
    all_rows = []
    
    # 只要有一个表有数据就继续
    if not df_energy.empty or not df_power.empty or not df_entropy.empty:
        # 阶段归类：去掉数字，比如 "发作前期1" -> "发作前期"
        if not df_energy.empty:
            df_energy["阶段归类"] = df_energy["阶段"].astype(str).str.replace(r"\d", "", regex=True)
        if not df_power.empty:
            df_power["阶段归类"] = df_power["阶段"].astype(str).str.replace(r"\d", "", regex=True)
        if not df_entropy.empty:
            df_entropy["阶段归类"] = df_entropy["阶段"].astype(str).str.replace(r"\d", "", regex=True)
        
        # 遍历三大阶段
        for phase_type in ["发作前期", "发作期", "发作后期"]:
            # 筛选当前阶段
            sub_energy = df_energy[df_energy["阶段归类"] == phase_type] if not df_energy.empty else pd.DataFrame()
            sub_power = df_power[df_power["阶段归类"] == phase_type] if not df_power.empty else pd.DataFrame()
            sub_entropy = df_entropy[df_entropy["阶段归类"] == phase_type] if not df_entropy.empty else pd.DataFrame()
            
            # 构建汇总行
            row = {
                "文件编号": folder_name,
                "阶段": phase_type,
                "类别": "癫痫"
            }
            
            # 1. 能量占比
            if not sub_energy.empty:
                row["DELTA能量_avg"] = round_3(sub_energy["DELTA能量"].mean())
                row["DELTA占比_avg"] = round_3(sub_energy["DELTA占比"].mean())
                row["THETA能量_avg"] = round_3(sub_energy["THETA能量"].mean())
                row["THETA占比_avg"] = round_3(sub_energy["THETA占比"].mean())
                row["ALPHA能量_avg"] = round_3(sub_energy["ALPHA能量"].mean())
                row["ALPHA占比_avg"] = round_3(sub_energy["ALPHA占比"].mean())
                row["BETA能量_avg"] = round_3(sub_energy["BETA能量"].mean())
                row["BETA占比_avg"] = round_3(sub_energy["BETA占比"].mean())
                row["GAMMA能量_avg"] = round_3(sub_energy["GAMMA能量"].mean())
                row["GAMMA占比_avg"] = round_3(sub_energy["GAMMA占比"].mean())
            
            # 2. 主频功率
            if not sub_power.empty:
                row["α主频_avg"] = round_3(sub_power["α主频"].mean())
                row["α功率_avg"] = round_3(sub_power["α功率"].mean())
                row["α总功率_avg"] = round_3(sub_power["α总功率"].mean())
                row["β主频_avg"] = round_3(sub_power["β主频"].mean())
                row["β功率_avg"] = round_3(sub_power["β功率"].mean())
                row["β总功率_avg"] = round_3(sub_power["β总功率"].mean())
                row["α+β主频_avg"] = round_3(sub_power["α+β主频"].mean())
                row["α+β功率_avg"] = round_3(sub_power["α+β功率"].mean())
                row["α+β总功率_avg"] = round_3(sub_power["α+β总功率"].mean())
            
            # 3. 功率谱熵
            if not sub_entropy.empty:
                row["总功率谱熵_avg"] = round_3(pd.to_numeric(sub_entropy["总功率谱熵"], errors='coerce').mean())
                row["归一化总熵_avg"] = round_3(pd.to_numeric(sub_entropy["归一化总熵"], errors='coerce').mean())
                row["α波熵_avg"] = round_3(pd.to_numeric(sub_entropy["α波熵"], errors='coerce').mean())
                row["归一化α熵_avg"] = round_3(pd.to_numeric(sub_entropy["归一化α熵"], errors='coerce').mean())
                row["β波熵_avg"] = round_3(pd.to_numeric(sub_entropy["β波熵"], errors='coerce').mean())
                row["归一化β熵_avg"] = round_3(pd.to_numeric(sub_entropy["归一化β熵"], errors='coerce').mean())
            
            # 4. 全局特征
            row["D2_avg"] = round_3(d2)
            row["PD2_avg"] = round_3(pd2)
            row["LE_avg"] = round_3(le)
            row["KE_avg"] = round_3(ke)
            
            all_rows.append(row)
    
    return all_rows

def process_normal_folder(folder_path, folder_name):
    """处理单个正常数据文件夹"""
    
    # ---------------------- 读取文件 ----------------------
    file03 = os.path.join(folder_path, "03无发作能量占比.xlsx")
    file04 = os.path.join(folder_path, "04无发作主频功率.xlsx")
    file05 = os.path.join(folder_path, "05无发作功率谱熵.xlsx")
    feat_file = os.path.join(folder_path, "Seizure_Features_Summary.xlsx")
    
    # 读取数据
    df03 = pd.read_excel(file03) if os.path.exists(file03) else pd.DataFrame()
    df04 = pd.read_excel(file04) if os.path.exists(file04) else pd.DataFrame()
    df05 = pd.read_excel(file05) if os.path.exists(file05) else pd.DataFrame()
    
    # ---------------------- 读取 D2 / PD2 / LE / KE ----------------------
    d2, pd2, le, ke = np.nan, np.nan, np.nan, np.nan
    
    if os.path.exists(feat_file):
        try:
            # 安全读取 + 指定引擎
            df_feat = pd.read_excel(
                feat_file,
                sheet_name="全局特征平均值汇总",
                engine="openpyxl"
            )
            
            # 清理特征名称：转字符串 + 去空格
            df_feat["特征名称"] = df_feat["特征名称"].astype(str).str.strip()
            
            # 匹配特征
            for _, r in df_feat.iterrows():
                name = r["特征名称"]
                
                if "D2关联维数" in name:
                    d2 = r["全局平均值"]
                if "PD2点关联维" in name:
                    pd2 = r["全局平均值"]
                if "LE李雅普诺夫指数" in name:
                    le = r["全局平均值"]
                if "KE科尔莫戈罗夫熵" in name:
                    ke = r["全局平均值"]
                    
        except Exception as e:
            print(f"[{folder_name}] 读取特征文件失败：{str(e)}")
    
    # ====================== 开始构建正常时期汇总行 ======================
    row = {
        "文件编号": folder_name,
        "阶段": "正常时期",
        "类别": "正常"
    }
    
    # 1. 能量占比（全部取平均）
    if not df03.empty:
        row["DELTA能量_avg"] = round_3(df03["DELTA能量"].mean())
        row["DELTA占比_avg"] = round_3(df03["DELTA占比"].mean())
        row["THETA能量_avg"] = round_3(df03["THETA能量"].mean())
        row["THETA占比_avg"] = round_3(df03["THETA占比"].mean())
        row["ALPHA能量_avg"] = round_3(df03["ALPHA能量"].mean())
        row["ALPHA占比_avg"] = round_3(df03["ALPHA占比"].mean())
        row["BETA能量_avg"] = round_3(df03["BETA能量"].mean())
        row["BETA占比_avg"] = round_3(df03["BETA占比"].mean())
        row["GAMMA能量_avg"] = round_3(df03["GAMMA能量"].mean())
        row["GAMMA占比_avg"] = round_3(df03["GAMMA占比"].mean())
    
    # 2. 主频功率（全部取平均）
    if not df04.empty:
        row["α主频_avg"] = round_3(df04["α主频"].mean())
        row["α功率_avg"] = round_3(df04["α功率"].mean())
        row["α总功率_avg"] = round_3(df04["α总功率"].mean())
        row["β主频_avg"] = round_3(df04["β主频"].mean())
        row["β功率_avg"] = round_3(df04["β功率"].mean())
        row["β总功率_avg"] = round_3(df04["β总功率"].mean())
        row["α+β主频_avg"] = round_3(df04["α+β主频"].mean())
        row["α+β功率_avg"] = round_3(df04["α+β功率"].mean())
        row["α+β总功率_avg"] = round_3(df04["α+β总功率"].mean())
    
    # 3. 功率谱熵（全部取平均，自动转数值）
    if not df05.empty:
        row["总功率谱熵_avg"] = round_3(pd.to_numeric(df05["总功率谱熵"], errors='coerce').mean())
        row["归一化总熵_avg"] = round_3(pd.to_numeric(df05["归一化总熵"], errors='coerce').mean())
        row["α波熵_avg"] = round_3(pd.to_numeric(df05["α波熵"], errors='coerce').mean())
        row["归一化α熵_avg"] = round_3(pd.to_numeric(df05["归一化α熵"], errors='coerce').mean())
        row["β波熵_avg"] = round_3(pd.to_numeric(df05["β波熵"], errors='coerce').mean())
        row["归一化β熵_avg"] = round_3(pd.to_numeric(df05["归一化β熵"], errors='coerce').mean())
    
    # 4. 全局特征
    row["D2_avg"] = round_3(d2)
    row["PD2_avg"] = round_3(pd2)
    row["LE_avg"] = round_3(le)
    row["KE_avg"] = round_3(ke)
    
    return [row]

def main():
    print("开始汇总脑电数据...")
    
    all_data = []
    
    # 处理癫痫数据
    print("\n处理癫痫数据...")
    if os.path.exists(EPILEPSY_RESULTS_DIR):
        for folder_name in os.listdir(EPILEPSY_RESULTS_DIR):
            folder_path = os.path.join(EPILEPSY_RESULTS_DIR, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if folder_name.isdigit():
                print(f"处理癫痫文件夹: {folder_name}")
                epilepsy_data = process_epilepsy_folder(folder_path, folder_name)
                all_data.extend(epilepsy_data)
    else:
        print(f"癫痫数据目录不存在: {EPILEPSY_RESULTS_DIR}")
    
    # 处理正常数据
    print("\n处理正常数据...")
    if os.path.exists(NORMAL_RESULTS_DIR):
        for folder_name in os.listdir(NORMAL_RESULTS_DIR):
            folder_path = os.path.join(NORMAL_RESULTS_DIR, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if folder_name.isdigit():
                print(f"处理正常文件夹: {folder_name}")
                normal_data = process_normal_folder(folder_path, folder_name)
                all_data.extend(normal_data)
    else:
        print(f"正常数据目录不存在: {NORMAL_RESULTS_DIR}")
    
    # 保存结果
    if all_data:
        print("\n汇总数据...")
        df_all = pd.DataFrame(all_data)
        
        # 创建Excel写入器
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
            # 保存所有数据
            df_all.to_excel(writer, sheet_name='所有数据', index=False)
            
            # 按类别保存
            for category in df_all['类别'].unique():
                df_category = df_all[df_all['类别'] == category]
                df_category.to_excel(writer, sheet_name=f'{category}数据', index=False)
            
            # 按阶段保存
            for phase in df_all['阶段'].unique():
                df_phase = df_all[df_all['阶段'] == phase]
                df_phase.to_excel(writer, sheet_name=f'{phase}', index=False)
        
        print(f"\n数据汇总完成！结果保存至：{OUTPUT_FILE}")
        print(f"总处理样本数：{len(df_all['文件编号'].unique())}")
        print(f"总数据行数：{len(df_all)}")
    else:
        print("\n未找到任何数据！")

if __name__ == "__main__":
    main()
