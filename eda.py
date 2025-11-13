import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

parquet_folder = "parquet"

print("\nLoading key datasets:")
demo = pd.read_parquet(os.path.join(parquet_folder, "DEMO_L.parquet"))
diq = pd.read_parquet(os.path.join(parquet_folder, "DIQ_L.parquet"))
bpq = pd.read_parquet(os.path.join(parquet_folder, "BPQ_L.parquet"))
bpxo = pd.read_parquet(os.path.join(parquet_folder, "BPXO_L.parquet"))
glu = pd.read_parquet(os.path.join(parquet_folder, "GLU_L.parquet"))
ghb = pd.read_parquet(os.path.join(parquet_folder, "GHB_L.parquet"))
bmx = pd.read_parquet(os.path.join(parquet_folder, "BMX_L.parquet"))
mcq = pd.read_parquet(os.path.join(parquet_folder, "MCQ_L.parquet"))

dr1tot = pd.read_parquet(os.path.join(parquet_folder, "DR1TOT_L.parquet"))
paq = pd.read_parquet(os.path.join(parquet_folder, "PAQ_L.parquet"))
smq = pd.read_parquet(os.path.join(parquet_folder, "SMQ_L.parquet"))
alq = pd.read_parquet(os.path.join(parquet_folder, "ALQ_L.parquet"))
tchol = pd.read_parquet(os.path.join(parquet_folder, "TCHOL_L.parquet"))
hdl = pd.read_parquet(os.path.join(parquet_folder, "HDL_L.parquet"))

print("Datasets loaded\n")

print("Merging datasets on SEQN")
df = demo.copy()

df = df.merge(diq[['SEQN', 'DIQ010', 'DID040', 'DIQ160', 'DIQ180']], 
              on='SEQN', how='left')

df = df.merge(bpq[['SEQN', 'BPQ020', 'BPQ030', 'BPQ150']], 
              on='SEQN', how='left')

df = df.merge(bpxo[['SEQN', 'BPXOSY1', 'BPXODI1', 'BPXOSY2', 'BPXODI2', 'BPXOSY3', 'BPXODI3']], 
              on='SEQN', how='left')

df = df.merge(glu[['SEQN', 'LBXGLU', 'LBDGLUSI']], 
              on='SEQN', how='left')

df = df.merge(ghb[['SEQN', 'LBXGH']], 
              on='SEQN', how='left')

df = df.merge(bmx[['SEQN', 'BMXBMI', 'BMXWAIST', 'BMXHT', 'BMXWT']], 
              on='SEQN', how='left')

df = df.merge(dr1tot[['SEQN', 'DR1TKCAL', 'DR1TCARB', 'DR1TTFAT', 'DR1TSFAT', 
                      'DR1TSODI', 'DR1TSUGR', 'DR1TFIBE', 'DR1TPROT']], 
              on='SEQN', how='left')

df = df.merge(paq[['SEQN', 'PAD680', 'PAD800']], 
              on='SEQN', how='left')

df = df.merge(smq[['SEQN', 'SMQ020', 'SMQ040']], 
              on='SEQN', how='left')

df = df.merge(alq[['SEQN', 'ALQ121', 'ALQ130']], 
              on='SEQN', how='left')

df = df.merge(tchol[['SEQN', 'LBXTC']], 
              on='SEQN', how='left')
df = df.merge(hdl[['SEQN', 'LBDHDD']], 
              on='SEQN', how='left')

print(f"Merged dataset: {len(df):,} rows × {len(df.columns):,} columns\n")

print("\nCleaning data")

missing_code_threshold = 1e-50

if 'RIDAGEYR' in df.columns:
    age_before = df['RIDAGEYR'].isnull().sum()
    df.loc[df['RIDAGEYR'] < 0.1, 'RIDAGEYR'] = np.nan
    age_after = df['RIDAGEYR'].isnull().sum()
    if age_after > age_before:
        print(f"  Cleaned {age_after - age_before} invalid age values")

if 'BMXBMI' in df.columns:
    bmi_before = df['BMXBMI'].isnull().sum()
    df.loc[(df['BMXBMI'] < 5) | (df['BMXBMI'] > 100), 'BMXBMI'] = np.nan
    bmi_after = df['BMXBMI'].isnull().sum()
    if bmi_after > bmi_before:
        print(f"  Cleaned {bmi_after - bmi_before} extreme BMI values (<10 or >60)")

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col not in ['SEQN', 'RIDAGEYR', 'BMXBMI']:
        small_vals = (df[col] > 0) & (df[col] < missing_code_threshold)
        if small_vals.sum() > 0:
            df.loc[small_vals, col] = np.nan

if 'SMQ020' in df.columns:
    df.loc[df['SMQ020'].isin([7, 9]), 'SMQ020'] = np.nan

if 'PAD680' in df.columns:
    pa_before = df['PAD680'].isnull().sum()
    df.loc[df['PAD680'] > 2000, 'PAD680'] = np.nan
    df.loc[df['PAD680'] < 0, 'PAD680'] = np.nan
    pa_after = df['PAD680'].isnull().sum()
    if pa_after > pa_before:
        print(f"  Cleaned {pa_after - pa_before} unrealistic physical activity values")

dietary_vars = ['DR1TKCAL', 'DR1TSODI', 'DR1TCARB', 'DR1TTFAT']
for var in dietary_vars:
    if var in df.columns:
        if var == 'DR1TKCAL':
            df.loc[(df[var] < 200) | (df[var] > 6000), var] = np.nan
        elif var == 'DR1TSODI':
            df.loc[(df[var] < 100) | (df[var] > 15000), var] = np.nan
        else:
            df.loc[df[var] < 0, var] = np.nan

bp_vars = ['BPXOSY1', 'BPXODI1', 'BPXOSY2', 'BPXODI2', 'BPXOSY3', 'BPXODI3']
for var in bp_vars:
    if var in df.columns:
        if 'SY' in var:
            df.loc[(df[var] < 50) | (df[var] > 250), var] = np.nan
        elif 'DI' in var:
            df.loc[(df[var] < 30) | (df[var] > 150), var] = np.nan

if 'LBXGLU' in df.columns:
    df.loc[(df['LBXGLU'] < 40) | (df['LBXGLU'] > 600), 'LBXGLU'] = np.nan

if 'LBXGH' in df.columns:
    df.loc[(df['LBXGH'] < 3) | (df['LBXGH'] > 20), 'LBXGH'] = np.nan

print("Data cleaning complete\n")

print("Data Volume:")

total_participants = len(df)
print(f"\nTotal participants: {total_participants:,}")

print(f"\nAge distribution:")
print(f"  Min: {df['RIDAGEYR'].min():.0f} years")
print(f"  Max: {df['RIDAGEYR'].max():.0f} years")
print(f"  Mean: {df['RIDAGEYR'].mean():.1f} years")
print(f"  Median: {df['RIDAGEYR'].median():.1f} years")

print(f"\nGender distribution:")
gender_counts = df['RIAGENDR'].value_counts()
print(f"  Male (1): {gender_counts.get(1, 0):,} ({gender_counts.get(1, 0)/total_participants*100:.1f}%)")
print(f"  Female (2): {gender_counts.get(2, 0):,} ({gender_counts.get(2, 0)/total_participants*100:.1f}%)")

print(f"\nRace/Ethnicity distribution (RIDRETH3):")
race_counts = df['RIDRETH3'].value_counts().sort_index()
race_labels = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other/Multi-racial"
}
for code, count in race_counts.items():
    label = race_labels.get(code, f"Code {code}")
    pct = count / total_participants * 100
    print(f"  {label}: {count:,} ({pct:.1f}%)")

print("\n--- Target Variables: ---")

diq_clean = df['DIQ010'].replace([7, 9], np.nan)
diabetes_yes = (diq_clean == 1).sum()
diabetes_no = (diq_clean == 2).sum()
diabetes_borderline = (diq_clean == 3).sum()
diabetes_total = diabetes_yes + diabetes_no + diabetes_borderline

print(f"\nDiabetes (self-reported, DIQ010):")
print(f"  Yes: {diabetes_yes:,} ({diabetes_yes/diabetes_total*100:.1f}% of those who answered)")
print(f"  No: {diabetes_no:,} ({diabetes_no/diabetes_total*100:.1f}% of those who answered)")
print(f"  Borderline: {diabetes_borderline:,} ({diabetes_borderline/diabetes_total*100:.1f}% of those who answered)")
print(f"  Missing/Refused: {df['DIQ010'].isnull().sum():,}")

bpq_clean = df['BPQ020'].replace([7, 9], np.nan)
hypertension_yes = (bpq_clean == 1).sum()
hypertension_no = (bpq_clean == 2).sum()
hypertension_total = hypertension_yes + hypertension_no

print(f"\nHypertension (self-reported, BPQ020):")
print(f"  Yes: {hypertension_yes:,} ({hypertension_yes/hypertension_total*100:.1f}% of those who answered)")
print(f"  No: {hypertension_no:,} ({hypertension_no/hypertension_total*100:.1f}% of those who answered)")
print(f"  Missing/Refused: {df['BPQ020'].isnull().sum():,}")

print(f"\n--- Lab Measurements ---")
print(f"Fasting Glucose (LBXGLU):")
glu_valid = df['LBXGLU'].notna().sum()
print(f"  Available: {glu_valid:,} participants")
if glu_valid > 0:
    print(f"  Mean: {df['LBXGLU'].mean():.1f} mg/dL")
    print(f"  Median: {df['LBXGLU'].median():.1f} mg/dL")
    diabetes_glu = (df['LBXGLU'] >= 126).sum()
    print(f"  ≥126 mg/dL (diabetic range): {diabetes_glu:,} ({diabetes_glu/glu_valid*100:.1f}%)")

print(f"\nHbA1c (LBXGH):")
ghb_valid = df['LBXGH'].notna().sum()
print(f"  Available: {ghb_valid:,} participants")
if ghb_valid > 0:
    print(f"  Mean: {df['LBXGH'].mean():.2f}%")
    print(f"  Median: {df['LBXGH'].median():.2f}%")
    diabetes_ghb = (df['LBXGH'] >= 6.5).sum()
    print(f"  ≥6.5% (diabetic range): {diabetes_ghb:,} ({diabetes_ghb/ghb_valid*100:.1f}%)")

print(f"\n--- Blood Pressure Measurements ---")
bp_systolic = df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1, skipna=True)
bp_diastolic = df[['BPXODI1', 'BPXODI2', 'BPXODI3']].mean(axis=1, skipna=True)

bp_valid = bp_systolic.notna().sum()
print(f"Blood Pressure (averaged across 3 measurements):")
print(f"  Available: {bp_valid:,} participants")
if bp_valid > 0:
    print(f"  Systolic - Mean: {bp_systolic.mean():.1f} mmHg, Median: {bp_systolic.median():.1f} mmHg")
    print(f"  Diastolic - Mean: {bp_diastolic.mean():.1f} mmHg, Median: {bp_diastolic.median():.1f} mmHg")
    hypertension_bp = ((bp_systolic >= 130) | (bp_diastolic >= 80)).sum()
    print(f"  ≥130/80 mmHg (hypertensive): {hypertension_bp:,} ({hypertension_bp/bp_valid*100:.1f}%)")

print(f"\n--- Physical Measurements ---")
print(f"BMI (BMXBMI):")
bmi_valid = df['BMXBMI'].notna().sum()
print(f"  Available: {bmi_valid:,} participants")
if bmi_valid > 0:
    print(f"  Mean: {df['BMXBMI'].mean():.1f} kg/m²")
    print(f"  Median: {df['BMXBMI'].median():.1f} kg/m²")
    obese = (df['BMXBMI'] >= 30).sum()
    print(f"  Obese (≥30): {obese:,} ({obese/bmi_valid*100:.1f}%)")
    overweight = (df['BMXBMI'] >= 25).sum()
    print(f"  Overweight (≥25): {overweight:,} ({overweight/bmi_valid*100:.1f}%)")

print(f"\nWaist Circumference (BMXWAIST):")
waist_valid = df['BMXWAIST'].notna().sum()
print(f"  Available: {waist_valid:,} participants")
if waist_valid > 0:
    print(f"  Mean: {df['BMXWAIST'].mean():.1f} cm")
    print(f"  Median: {df['BMXWAIST'].median():.1f} cm")

print(f"\n--- Lifestyle & Dietary Factors: ---")

print(f"\nDietary Intake (Day 1, DR1TOT):")
diet_valid = df['DR1TKCAL'].notna().sum()
print(f"  Available: {diet_valid:,} participants")
if diet_valid > 0:
    print(f"  Calories - Mean: {df['DR1TKCAL'].mean():.0f} kcal, Median: {df['DR1TKCAL'].median():.0f} kcal")
    print(f"  Carbohydrates - Mean: {df['DR1TCARB'].mean():.0f} g, Median: {df['DR1TCARB'].median():.0f} g")
    print(f"  Total Fat - Mean: {df['DR1TTFAT'].mean():.1f} g, Median: {df['DR1TTFAT'].median():.1f} g")
    print(f"  Saturated Fat - Mean: {df['DR1TSFAT'].mean():.1f} g, Median: {df['DR1TSFAT'].median():.1f} g")
    print(f"  Sodium - Mean: {df['DR1TSODI'].mean():.0f} mg, Median: {df['DR1TSODI'].median():.0f} mg")
    print(f"  Sugar - Mean: {df['DR1TSUGR'].mean():.0f} g, Median: {df['DR1TSUGR'].median():.0f} g")
    print(f"  Fiber - Mean: {df['DR1TFIBE'].mean():.1f} g, Median: {df['DR1TFIBE'].median():.1f} g")
    print(f"  Protein - Mean: {df['DR1TPROT'].mean():.0f} g, Median: {df['DR1TPROT'].median():.0f} g")

print(f"\nPhysical Activity (PAQ):")
pa_valid = df['PAD680'].notna().sum()
print(f"  Available: {pa_valid:,} participants")
if pa_valid > 0:
    print(f"  Total minutes/week (PAD680) - Mean: {df['PAD680'].mean():.0f} min, Median: {df['PAD680'].median():.0f} min")
    pa_vigorous_valid = df['PAD800'].notna().sum()
    if pa_vigorous_valid > 0:
        print(f"  Vigorous activity minutes/week (PAD800) - Mean: {df['PAD800'].mean():.0f} min, Median: {df['PAD800'].median():.0f} min")
        print(f"  Available: {pa_vigorous_valid:,} participants")

print(f"\nSmoking Status (SMQ):")
smq_valid = df['SMQ020'].notna().sum()
print(f"  Available: {smq_valid:,} participants")
if smq_valid > 0:
    ever_smoked = (df['SMQ020'].replace([7,9], np.nan) == 1).sum()
    never_smoked = (df['SMQ020'].replace([7,9], np.nan) == 2).sum()
    total_smq = ever_smoked + never_smoked
    if total_smq > 0:
        print(f"  Ever smoked: {ever_smoked:,} ({ever_smoked/total_smq*100:.1f}%)")
        print(f"  Never smoked: {never_smoked:,} ({never_smoked/total_smq*100:.1f}%)")
    current_smoker = ((df['SMQ040'].replace([7,9], np.nan) == 1) | 
                      (df['SMQ040'].replace([7,9], np.nan) == 2)).sum()
    smq040_valid = df['SMQ040'].notna().sum()
    if smq040_valid > 0:
        print(f"  Current smoker (every day or some days): {current_smoker:,} ({current_smoker/smq040_valid*100:.1f}% of those who answered)")

print(f"\nAlcohol Consumption (ALQ):")
alq_valid = df['ALQ121'].notna().sum()
print(f"  Available: {alq_valid:,} participants")
if alq_valid > 0:
    print(f"  Days per year drank alcohol (ALQ121) - Mean: {df['ALQ121'].mean():.0f} days, Median: {df['ALQ121'].median():.0f} days")
    alq130_valid = df['ALQ130'].notna().sum()
    if alq130_valid > 0:
        print(f"  Average drinks per day (ALQ130) - Mean: {df['ALQ130'].mean():.2f} drinks, Median: {df['ALQ130'].median():.2f} drinks")
        print(f"  Available: {alq130_valid:,} participants")

print(f"\n--- Lipid Panel ---")
print(f"Total Cholesterol (LBXTC):")
chol_valid = df['LBXTC'].notna().sum()
print(f"  Available: {chol_valid:,} participants")
if chol_valid > 0:
    print(f"  Mean: {df['LBXTC'].mean():.1f} mg/dL, Median: {df['LBXTC'].median():.1f} mg/dL")
    high_chol = (df['LBXTC'] >= 200).sum()
    print(f"  ≥200 mg/dL (high): {high_chol:,} ({high_chol/chol_valid*100:.1f}%)")

print(f"\nHDL Cholesterol (LBDHDD):")
hdl_valid = df['LBDHDD'].notna().sum()
print(f"  Available: {hdl_valid:,} participants")
if hdl_valid > 0:
    print(f"  Mean: {df['LBDHDD'].mean():.1f} mg/dL, Median: {df['LBDHDD'].median():.1f} mg/dL")
    low_hdl = (df['LBDHDD'] < 40).sum()
    print(f"  <40 mg/dL (low): {low_hdl:,} ({low_hdl/hdl_valid*100:.1f}%)")

key_vars = {
    'Demographics': ['RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'INDFMPIR'],
    'Diabetes': ['DIQ010', 'LBXGLU', 'LBXGH'],
    'Hypertension': ['BPQ020', 'BPXOSY1', 'BPXODI1'],
    'Physical': ['BMXBMI', 'BMXWAIST', 'BMXHT', 'BMXWT'],
    'Dietary': ['DR1TKCAL', 'DR1TCARB', 'DR1TSODI'],
    'Lifestyle': ['PAD680', 'SMQ020', 'ALQ121'],
    'Lipids': ['LBXTC', 'LBDHDD']
}

print("\nMissing data by category:")
for category, vars_list in key_vars.items():
    print(f"\n{category}:")
    for var in vars_list:
        if var in df.columns:
            missing = df[var].isnull().sum()
            pct = missing / total_participants * 100
            print(f"  {var}: {missing:,} ({pct:.1f}%)")

print("POTENTIAL INTERACTION SIGNALS:")

print("\n--- Diabetes Prevalence by Race/Ethnicity ---")
def calc_diabetes_stats(group):
    diq_clean = group['DIQ010'].replace([7,9], np.nan)
    diabetes_yes = (diq_clean == 1).sum()
    diabetes_no = (diq_clean == 2).sum()
    total = diabetes_yes + diabetes_no
    rate = (diabetes_yes / total * 100) if total > 0 else np.nan
    return pd.Series({
        'Total': len(group),
        'Diabetes_Yes': diabetes_yes,
        'Diabetes_No': diabetes_no,
        'Diabetes_Rate': rate
    })

diabetes_by_race = df[['DIQ010']].groupby(df['RIDRETH3']).apply(calc_diabetes_stats).round(2)

for code in diabetes_by_race.index:
    label = race_labels.get(code, f"Code {code}")
    rate = diabetes_by_race.loc[code, 'Diabetes_Rate']
    if not pd.isna(rate):
        print(f"  {label}: {rate:.1f}%")

print("\n--- Hypertension Prevalence by Race/Ethnicity ---")
def calc_hypertension_stats(group):
    bpq_clean = group['BPQ020'].replace([7,9], np.nan)
    hypertension_yes = (bpq_clean == 1).sum()
    hypertension_no = (bpq_clean == 2).sum()
    total = hypertension_yes + hypertension_no
    rate = (hypertension_yes / total * 100) if total > 0 else np.nan
    return pd.Series({
        'Total': len(group),
        'Hypertension_Yes': hypertension_yes,
        'Hypertension_No': hypertension_no,
        'Hypertension_Rate': rate
    })

hypertension_by_race = df[['BPQ020']].groupby(df['RIDRETH3']).apply(calc_hypertension_stats).round(2)

for code in hypertension_by_race.index:
    label = race_labels.get(code, f"Code {code}")
    rate = hypertension_by_race.loc[code, 'Hypertension_Rate']
    if not pd.isna(rate):
        print(f"  {label}: {rate:.1f}%")

print("\n--- BMI and Diabetes Relationship ---")
df_temp = df[df['BMXBMI'].notna() & df['DIQ010'].notna()].copy()
df_temp['DIQ010_clean'] = df_temp['DIQ010'].replace([7,9], np.nan)
df_temp['Has_Diabetes'] = (df_temp['DIQ010_clean'] == 1)

if len(df_temp) > 0:
    diabetes_bmi = df_temp[df_temp['Has_Diabetes'] == True]['BMXBMI'].mean()
    no_diabetes_bmi = df_temp[df_temp['DIQ010_clean'] == 2]['BMXBMI'].mean()
    print(f"  Mean BMI - With Diabetes: {diabetes_bmi:.1f} kg/m²")
    print(f"  Mean BMI - Without Diabetes: {no_diabetes_bmi:.1f} kg/m²")
    print(f"  Difference: {diabetes_bmi - no_diabetes_bmi:.1f} kg/m²")

print("\n--- Age and Diabetes Relationship ---")
df_temp = df[df['RIDAGEYR'].notna() & df['DIQ010'].notna()].copy()
df_temp['DIQ010_clean'] = df_temp['DIQ010'].replace([7,9], np.nan)
df_temp['Has_Diabetes'] = (df_temp['DIQ010_clean'] == 1)

if len(df_temp) > 0:
    diabetes_age = df_temp[df_temp['Has_Diabetes'] == True]['RIDAGEYR'].mean()
    no_diabetes_age = df_temp[df_temp['DIQ010_clean'] == 2]['RIDAGEYR'].mean()
    print(f"  Mean Age - With Diabetes: {diabetes_age:.1f} years")
    print(f"  Mean Age - Without Diabetes: {no_diabetes_age:.1f} years")

print("\n--- Income (PIR) and Health Outcomes ---")
if 'INDFMPIR' in df.columns:
    df_temp = df[df['INDFMPIR'].notna()].copy()
    df_temp['Income_Category'] = pd.cut(df_temp['INDFMPIR'], 
                                        bins=[0, 1.3, 3.5, 5, np.inf],
                                        labels=['Low', 'Medium', 'High', 'Very High'])
    
    def calc_diabetes_rate(x):
        diq_clean = x['DIQ010'].replace([7,9], np.nan)
        diabetes_yes = (diq_clean == 1).sum()
        diabetes_no = (diq_clean == 2).sum()
        total = diabetes_yes + diabetes_no
        if total > 0:
            return (diabetes_yes / total) * 100
        else:
            return np.nan

    diabetes_by_income = df_temp[['DIQ010']].groupby(df_temp['Income_Category'], observed=True).apply(calc_diabetes_rate)
    
    print("  Diabetes rate by income:")
    for cat, rate in diabetes_by_income.items():
        if not pd.isna(rate):
            print(f"    {cat}: {rate:.1f}%")

print("\n--- Physical Activity and Health Outcomes ---")
df_temp = df[df['PAD680'].notna() & df['DIQ010'].notna()].copy()
df_temp['DIQ010_clean'] = df_temp['DIQ010'].replace([7,9], np.nan)

if len(df_temp) > 0:
    df_temp['PA_Category'] = pd.cut(df_temp['PAD680'], 
                                   bins=[0, 150, 300, np.inf],
                                   labels=['Low (<150 min/week)', 'Moderate (150-300)', 'High (>300)'])
    
    diabetes_by_pa = df_temp.groupby('PA_Category', observed=True)['DIQ010_clean'].apply(
        lambda x: (x == 1).sum() / ((x == 1).sum() + (x == 2).sum()) * 100
        if ((x == 1).sum() + (x == 2).sum()) > 0 else np.nan
    )
    
    print("  Diabetes rate by physical activity level:")
    for cat, rate in diabetes_by_pa.items():
        if not pd.isna(rate):
            print(f"    {cat}: {rate:.1f}%")

print("\n--- Smoking and Health Outcomes ---")
df_temp = df[df['SMQ020'].notna() & df['DIQ010'].notna()].copy()
df_temp['DIQ010_clean'] = df_temp['DIQ010'].replace([7,9], np.nan)
df_temp['SMQ020_clean'] = df_temp['SMQ020'].replace([7,9], np.nan)
df_temp['Ever_Smoked'] = (df_temp['SMQ020_clean'] == 1)

if len(df_temp) > 0:
    smoker_df = df_temp[df_temp['Ever_Smoked'] == True]
    nonsmoker_df = df_temp[df_temp['Ever_Smoked'] == False]
    
    if len(smoker_df) > 0:
        smoker_diabetes = (smoker_df['DIQ010_clean'] == 1).sum() / \
                         ((smoker_df['DIQ010_clean'] == 1).sum() + 
                          (smoker_df['DIQ010_clean'] == 2).sum()) * 100
    else:
        smoker_diabetes = np.nan
    
    if len(nonsmoker_df) > 0:
        nonsmoker_diabetes = (nonsmoker_df['DIQ010_clean'] == 1).sum() / \
                            ((nonsmoker_df['DIQ010_clean'] == 1).sum() + 
                             (nonsmoker_df['DIQ010_clean'] == 2).sum()) * 100
    else:
        nonsmoker_diabetes = np.nan
    
    if not pd.isna(smoker_diabetes) and not pd.isna(nonsmoker_diabetes):
        print(f"  Diabetes rate - Ever smoked: {smoker_diabetes:.1f}%")
        print(f"  Diabetes rate - Never smoked: {nonsmoker_diabetes:.1f}%")

print("\n--- Dietary Factors and Health Outcomes ---")
df_temp = df[df['DR1TKCAL'].notna() & df['DIQ010'].notna()].copy()
df_temp['DIQ010_clean'] = df_temp['DIQ010'].replace([7,9], np.nan)

if len(df_temp) > 0:
    df_temp['Calorie_Category'] = pd.cut(df_temp['DR1TKCAL'], 
                                         bins=[0, 1500, 2000, 2500, np.inf],
                                         labels=['Low (<1500)', 'Moderate (1500-2000)', 'High (2000-2500)', 'Very High (>2500)'])
    
    diabetes_by_cal = df_temp.groupby('Calorie_Category', observed=True)['DIQ010_clean'].apply(
        lambda x: (x == 1).sum() / ((x == 1).sum() + (x == 2).sum()) * 100
        if ((x == 1).sum() + (x == 2).sum()) > 0 else np.nan
    )
    
    print("  Diabetes rate by daily calorie intake:")
    for cat, rate in diabetes_by_cal.items():
        if not pd.isna(rate):
            print(f"    {cat}: {rate:.1f}%")

    df_temp2 = df[df['DR1TSODI'].notna() & df['BPQ020'].notna()].copy()
    df_temp2['BPQ020_clean'] = df_temp2['BPQ020'].replace([7,9], np.nan)
    
    if len(df_temp2) > 0:
        df_temp2['High_Sodium'] = df_temp2['DR1TSODI'] >= 2300
        
        high_sodium_htn = (df_temp2[df_temp2['High_Sodium'] == True]['BPQ020_clean'] == 1).sum() / \
                         ((df_temp2[df_temp2['High_Sodium'] == True]['BPQ020_clean'] == 1).sum() + 
                          (df_temp2[df_temp2['High_Sodium'] == True]['BPQ020_clean'] == 2).sum()) * 100
        low_sodium_htn = (df_temp2[df_temp2['High_Sodium'] == False]['BPQ020_clean'] == 1).sum() / \
                        ((df_temp2[df_temp2['High_Sodium'] == False]['BPQ020_clean'] == 1).sum() + 
                         (df_temp2[df_temp2['High_Sodium'] == False]['BPQ020_clean'] == 2).sum()) * 100
        
        if not pd.isna(high_sodium_htn) and not pd.isna(low_sodium_htn):
            print(f"\n  Hypertension rate - High sodium (≥2300 mg/day): {high_sodium_htn:.1f}%")
            print(f"  Hypertension rate - Low sodium (<2300 mg/day): {low_sodium_htn:.1f}%")

print("EDA COMPLETE")

print("PREPARING DATA FOR VISUALIZATIONS")

df['Has_Diabetes'] = (df['DIQ010'].replace([7,9], np.nan) == 1).astype(float)
df['Has_Hypertension'] = (df['BPQ020'].replace([7,9], np.nan) == 1).astype(float)

race_labels = {
    1: "Mexican\nAmerican",
    2: "Other\nHispanic",
    3: "Non-Hispanic\nWhite",
    4: "Non-Hispanic\nBlack",
    6: "Non-Hispanic\nAsian",
    7: "Other/Multi-\nracial"
}
df['Race_Label'] = df['RIDRETH3'].map(race_labels)

print("Target variables created\n")

print("CREATING EXPLORATORY DATA VISUALIZATIONS")

output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

print("\nCreating Visualization 1: Disease Prevalence by Race/Ethnicity")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

diabetes_by_race = df.groupby('Race_Label')['Has_Diabetes'].agg(['mean', 'count']).reset_index()
diabetes_by_race = diabetes_by_race[diabetes_by_race['count'] > 100]  # Filter small groups
diabetes_by_race = diabetes_by_race.sort_values('mean', ascending=False)
diabetes_by_race['prevalence'] = diabetes_by_race['mean'] * 100

bars1 = ax1.bar(diabetes_by_race['Race_Label'], diabetes_by_race['prevalence'], 
                color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Diabetes Prevalence (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Race/Ethnicity', fontsize=12, fontweight='bold')
ax1.set_title('Diabetes Prevalence by Race/Ethnicity', fontsize=14, fontweight='bold', pad=20)
ax1.set_ylim(0, max(diabetes_by_race['prevalence']) * 1.15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for i, (bar, val) in enumerate(zip(bars1, diabetes_by_race['prevalence'])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

hypertension_by_race = df.groupby('Race_Label')['Has_Hypertension'].agg(['mean', 'count']).reset_index()
hypertension_by_race = hypertension_by_race[hypertension_by_race['count'] > 100]
hypertension_by_race = hypertension_by_race.sort_values('mean', ascending=False)
hypertension_by_race['prevalence'] = hypertension_by_race['mean'] * 100

bars2 = ax2.bar(hypertension_by_race['Race_Label'], hypertension_by_race['prevalence'],
                color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Hypertension Prevalence (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Race/Ethnicity', fontsize=12, fontweight='bold')
ax2.set_title('Hypertension Prevalence by Race/Ethnicity', fontsize=14, fontweight='bold', pad=20)
ax2.set_ylim(0, max(hypertension_by_race['prevalence']) * 1.15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for i, (bar, val) in enumerate(zip(bars2, hypertension_by_race['prevalence'])):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1.0,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_disease_prevalence_by_race.png'), dpi=300, bbox_inches='tight')
print(f" Saved: {output_dir}/1_disease_prevalence_by_race.png")
plt.close()

print("Creating Visualization 2: Correlation Heatmap")

corr_vars = ['RIDAGEYR', 'BMXBMI', 'BMXWAIST', 'LBXGLU', 'LBXGH', 
             'BPXOSY1', 'BPXODI1', 'DR1TKCAL', 'DR1TSODI', 'PAD680',
             'LBXTC', 'LBDHDD', 'Has_Diabetes', 'Has_Hypertension']

corr_df = df[corr_vars].corr()

mask = np.triu(np.ones_like(corr_df, dtype=bool))

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax, annot_kws={'size': 8})

ax.set_title('Correlation Matrix of Key Risk Factors and Health Outcomes', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '2_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/2_correlation_heatmap.png")
plt.close()

print("Creating Visualization 3: BMI Distribution by Diabetes Status")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

bmi_data = df[df['BMXBMI'].notna() & df['Has_Diabetes'].notna()].copy()
bmi_data['Diabetes_Status'] = bmi_data['Has_Diabetes'].map({0: 'No Diabetes', 1: 'Diabetes'})

for status in ['No Diabetes', 'Diabetes']:
    data = bmi_data[bmi_data['Diabetes_Status'] == status]['BMXBMI']
    ax1.hist(data, bins=30, alpha=0.6, label=status, density=True, edgecolor='black', linewidth=0.5)

ax1.set_xlabel('BMI (kg/m²)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
ax1.set_title('BMI Distribution by Diabetes Status', fontsize=14, fontweight='bold', pad=20)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3, linestyle='--')
ax1.axvline(x=25, color='orange', linestyle='--', linewidth=2, label='Overweight (25)')
ax1.axvline(x=30, color='red', linestyle='--', linewidth=2, label='Obese (30)')
ax1.legend(fontsize=10)

bmi_data_clean = bmi_data[bmi_data['Diabetes_Status'].isin(['No Diabetes', 'Diabetes'])]
sns.boxplot(data=bmi_data_clean, x='Diabetes_Status', y='BMXBMI', ax=ax2, 
            hue='Diabetes_Status', palette=['#3498db', '#e74c3c'], width=0.6, legend=False)
ax2.set_xlabel('Diabetes Status', fontsize=12, fontweight='bold')
ax2.set_ylabel('BMI (kg/m²)', fontsize=12, fontweight='bold')
ax2.set_title('BMI Distribution by Diabetes Status (Box Plot)', fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=25, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Overweight')
ax2.axhline(y=30, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Obese')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3_bmi_by_diabetes.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/3_bmi_by_diabetes.png")
plt.close()

print("Creating Visualization 4: Age Trends")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

df['Age_Group'] = pd.cut(df['RIDAGEYR'], bins=[0, 20, 40, 60, 80, 100], 
                        labels=['0-20', '21-40', '41-60', '61-80', '81+'])

diabetes_age = df.groupby('Age_Group', observed=True)['Has_Diabetes'].agg(['mean', 'count']).reset_index()
diabetes_age = diabetes_age[diabetes_age['count'] > 50]
diabetes_age['prevalence'] = diabetes_age['mean'] * 100
diabetes_age['ci_lower'] = diabetes_age['prevalence'] - 1.96 * np.sqrt(
    diabetes_age['prevalence'] * (100 - diabetes_age['prevalence']) / diabetes_age['count'])
diabetes_age['ci_upper'] = diabetes_age['prevalence'] + 1.96 * np.sqrt(
    diabetes_age['prevalence'] * (100 - diabetes_age['prevalence']) / diabetes_age['count'])

ax1.plot(diabetes_age['Age_Group'], diabetes_age['prevalence'], 
         marker='o', linewidth=2.5, markersize=10, color='#e74c3c', label='Diabetes')
ax1.fill_between(diabetes_age['Age_Group'], diabetes_age['ci_lower'], diabetes_age['ci_upper'],
                 alpha=0.2, color='#e74c3c')
ax1.set_xlabel('Age Group (years)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Prevalence (%)', fontsize=12, fontweight='bold')
ax1.set_title('Diabetes Prevalence by Age Group', fontsize=14, fontweight='bold', pad=20)
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(diabetes_age['prevalence']) * 1.2)

hypertension_age = df.groupby('Age_Group', observed=True)['Has_Hypertension'].agg(['mean', 'count']).reset_index()
hypertension_age = hypertension_age[hypertension_age['count'] > 50]
hypertension_age['prevalence'] = hypertension_age['mean'] * 100
hypertension_age['ci_lower'] = hypertension_age['prevalence'] - 1.96 * np.sqrt(
    hypertension_age['prevalence'] * (100 - hypertension_age['prevalence']) / hypertension_age['count'])
hypertension_age['ci_upper'] = hypertension_age['prevalence'] + 1.96 * np.sqrt(
    hypertension_age['prevalence'] * (100 - hypertension_age['prevalence']) / hypertension_age['count'])

ax2.plot(hypertension_age['Age_Group'], hypertension_age['prevalence'],
         marker='s', linewidth=2.5, markersize=10, color='#3498db', label='Hypertension')
ax2.fill_between(hypertension_age['Age_Group'], hypertension_age['ci_lower'], hypertension_age['ci_upper'],
                 alpha=0.2, color='#3498db')
ax2.set_xlabel('Age Group (years)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Prevalence (%)', fontsize=12, fontweight='bold')
ax2.set_title('Hypertension Prevalence by Age Group', fontsize=14, fontweight='bold', pad=20)
ax2.grid(alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(hypertension_age['prevalence']) * 1.2)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_age_trends.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/4_age_trends.png")
plt.close()

print("Creating Visualization 5: Missing Data Patterns")

missing_vars = ['DIQ010', 'BPQ020', 'BMXBMI', 'LBXGLU', 'LBXGH', 
                'BPXOSY1', 'DR1TKCAL', 'PAD680', 'SMQ020', 'LBXTC']
missing_data = []
for var in missing_vars:
    missing_count = df[var].isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_data.append({'Variable': var, 'Missing_Percentage': missing_pct})

missing_df = pd.DataFrame(missing_data).sort_values('Missing_Percentage', ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(missing_df['Variable'], missing_df['Missing_Percentage'],
                color='#95a5a6', alpha=0.7, edgecolor='black', linewidth=1)
ax.set_xlabel('Missing Data Percentage (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Variable', fontsize=12, fontweight='bold')
ax.set_title('Missing Data Patterns Across Key Variables', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')

for i, (bar, val) in enumerate(zip(bars, missing_df['Missing_Percentage'])):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', ha='left', va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '5_missing_data.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/5_missing_data.png")
plt.close()

print("Creating Visualization 6: Lifestyle Factors")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

smoking_data = df[df['SMQ020'].notna() & df['Has_Diabetes'].notna()].copy()
smoking_data['Smoking_Status'] = smoking_data['SMQ020'].replace([7,9], np.nan)
smoking_data['Ever_Smoked'] = (smoking_data['Smoking_Status'] == 1).map({True: 'Ever Smoked', False: 'Never Smoked'})
smoking_diabetes = smoking_data.groupby('Ever_Smoked')['Has_Diabetes'].mean() * 100

bars1 = axes[0, 0].bar(smoking_diabetes.index, smoking_diabetes.values,
                       color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0, 0].set_ylabel('Diabetes Prevalence (%)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Diabetes by Smoking Status', fontsize=12, fontweight='bold', pad=15)
axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars1, smoking_diabetes.values):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

pa_data = df[df['PAD680'].notna() & df['Has_Diabetes'].notna()].copy()
pa_data['PA_Category'] = pd.cut(pa_data['PAD680'], bins=[0, 150, 300, np.inf],
                                labels=['Low\n(<150)', 'Moderate\n(150-300)', 'High\n(>300)'])
pa_diabetes = pa_data.groupby('PA_Category', observed=True)['Has_Diabetes'].mean() * 100

bars2 = axes[0, 1].bar(pa_diabetes.index.astype(str), pa_diabetes.values,
                       color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0, 1].set_ylabel('Diabetes Prevalence (%)', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Physical Activity (min/week)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Diabetes by Physical Activity Level', fontsize=12, fontweight='bold', pad=15)
axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
for bar, val in zip(bars2, pa_diabetes.values):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

bmi_htn_data = df[df['BMXBMI'].notna() & df['Has_Hypertension'].notna()].copy()
bmi_htn_data['HTN_Status'] = bmi_htn_data['Has_Hypertension'].map({0: 'No HTN', 1: 'Hypertension'})
sns.boxplot(data=bmi_htn_data, x='HTN_Status', y='BMXBMI', ax=axes[1, 0],
            hue='HTN_Status', palette=['#3498db', '#e74c3c'], width=0.6, legend=False)
axes[1, 0].set_xlabel('Hypertension Status', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('BMI (kg/m²)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('BMI Distribution by Hypertension Status', fontsize=12, fontweight='bold', pad=15)
axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')

age_diabetes_data = df[df['RIDAGEYR'].notna() & df['Has_Diabetes'].notna()].copy()
age_diabetes_data['Diabetes_Status'] = age_diabetes_data['Has_Diabetes'].map({0: 'No Diabetes', 1: 'Diabetes'})
for status in ['No Diabetes', 'Diabetes']:
    data = age_diabetes_data[age_diabetes_data['Diabetes_Status'] == status]['RIDAGEYR']
    axes[1, 1].hist(data, bins=20, alpha=0.6, label=status, density=True, edgecolor='black', linewidth=0.5)
axes[1, 1].set_xlabel('Age (years)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Age Distribution by Diabetes Status', fontsize=12, fontweight='bold', pad=15)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '6_lifestyle_factors.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/6_lifestyle_factors.png")
plt.close()

print("Creating Visualization 7: BMI by Race and Diabetes Status")

bmi_race_data = df[df['BMXBMI'].notna() & df['Has_Diabetes'].notna() & df['Race_Label'].notna()].copy()
bmi_race_data['Diabetes_Status'] = bmi_race_data['Has_Diabetes'].map({0: 'No Diabetes', 1: 'Diabetes'})

fig, ax = plt.subplots(figsize=(16, 8))
sns.boxplot(data=bmi_race_data, x='Race_Label', y='BMXBMI', hue='Diabetes_Status',
            palette={'No Diabetes': '#3498db', 'Diabetes': '#e74c3c'}, ax=ax, width=0.7)
ax.set_xlabel('Race/Ethnicity', fontsize=12, fontweight='bold')
ax.set_ylabel('BMI (kg/m²)', fontsize=12, fontweight='bold')
ax.set_title('BMI Distribution by Race/Ethnicity and Diabetes Status', fontsize=14, fontweight='bold', pad=20)
ax.axhline(y=25, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Overweight (25)')
ax.axhline(y=30, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Obese (30)')
ax.legend(title='Diabetes Status', fontsize=10, title_fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '7_bmi_by_race_diabetes.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/7_bmi_by_race_diabetes.png")
plt.close()

print("Creating Visualization 8: Age-Stratified Disease Rates by Race")

age_strata = [(0, 40), (40, 60), (60, 100)]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, (age_min, age_max) in enumerate(age_strata):
    age_data = df[(df['RIDAGEYR'] >= age_min) & (df['RIDAGEYR'] < age_max) & 
                  df['Race_Label'].notna()].copy()

    diabetes_race_age = age_data.groupby('Race_Label')['Has_Diabetes'].agg(['mean', 'count']).reset_index()
    diabetes_race_age = diabetes_race_age[diabetes_race_age['count'] > 30]
    diabetes_race_age = diabetes_race_age.sort_values('mean', ascending=False)
    diabetes_race_age['prevalence'] = diabetes_race_age['mean'] * 100

    htn_race_age = age_data.groupby('Race_Label')['Has_Hypertension'].agg(['mean', 'count']).reset_index()
    htn_race_age = htn_race_age[htn_race_age['count'] > 30]
    htn_race_age = htn_race_age.sort_values('mean', ascending=False)
    htn_race_age['prevalence'] = htn_race_age['mean'] * 100
    
    if idx == 0:
        bars1 = axes[0].bar(diabetes_race_age['Race_Label'], diabetes_race_age['prevalence'],
                           color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('Diabetes Prevalence (%)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Race/Ethnicity', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Diabetes by Race/Ethnicity\nAge {age_min}-{age_max} years', 
                         fontsize=13, fontweight='bold', pad=15)
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        for bar, val in zip(bars1, diabetes_race_age['prevalence']):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        bars2 = axes[1].bar(htn_race_age['Race_Label'], htn_race_age['prevalence'],
                           color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('Hypertension Prevalence (%)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Race/Ethnicity', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Hypertension by Race/Ethnicity\nAge {age_min}-{age_max} years',
                         fontsize=13, fontweight='bold', pad=15)
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        for bar, val in zip(bars2, htn_race_age['prevalence']):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 1.0,
                        f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '8_age_stratified_by_race.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/8_age_stratified_by_race.png")
plt.close()

print("Creating Visualization 9: Diabetes Risk Factor Interactions by Race")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

smoking_race_data = df[df['SMQ020'].notna() & df['Has_Diabetes'].notna() & df['Race_Label'].notna()].copy()
smoking_race_data['Ever_Smoked'] = (smoking_race_data['SMQ020'].replace([7,9], np.nan) == 1)
smoking_race_diabetes = smoking_race_data.groupby(['Race_Label', 'Ever_Smoked'])['Has_Diabetes'].mean().unstack() * 100

smoking_race_diabetes.plot(kind='bar', ax=axes[0, 0], color=['#3498db', '#e74c3c'], 
                           alpha=0.7, edgecolor='black', linewidth=1)
axes[0, 0].set_ylabel('Diabetes Prevalence (%)', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Race/Ethnicity', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Diabetes by Race and Smoking Status', fontsize=12, fontweight='bold', pad=15)
axes[0, 0].legend(['Never Smoked', 'Ever Smoked'], fontsize=10)
axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0, 0].tick_params(axis='x', rotation=0)

pa_race_data = df[df['PAD680'].notna() & df['Has_Diabetes'].notna() & df['Race_Label'].notna()].copy()
pa_labels = ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']
if not pa_race_data.empty:
    pa_race_data['PA_Quartile'] = pd.qcut(
        pa_race_data['PAD680'],
        q=4,
        labels=pa_labels,
        duplicates='drop'
    )
    pa_race_diabetes = pa_race_data.groupby(['Race_Label', 'PA_Quartile'], observed=False)['Has_Diabetes'].mean().unstack() * 100
    pa_race_diabetes = pa_race_diabetes.reindex(columns=[label for label in pa_labels if label in pa_race_diabetes.columns])

    pa_colors = ['#95a5a6', '#27ae60', '#2980b9', '#8e44ad']
    pa_race_diabetes.plot(kind='bar', ax=axes[0, 1], color=pa_colors[:len(pa_race_diabetes.columns)],
                          alpha=0.7, edgecolor='black', linewidth=1)
    axes[0, 1].set_ylabel('Diabetes Prevalence (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Race/Ethnicity', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Diabetes by Race and Physical Activity Quartiles', fontsize=12, fontweight='bold', pad=15)
    axes[0, 1].legend(pa_race_diabetes.columns, fontsize=10, title='Physical Activity')
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0, 1].tick_params(axis='x', rotation=0)
else:
    axes[0, 1].axis('off')
    axes[0, 1].text(0.5, 0.5, 'Insufficient data for plot', ha='center', va='center', fontsize=12, fontweight='bold')

bmi_race_cat_data = df[df['BMXBMI'].notna() & df['Has_Diabetes'].notna() & df['Race_Label'].notna()].copy()
bmi_race_cat_data['BMI_Category'] = pd.cut(bmi_race_cat_data['BMXBMI'], 
                                          bins=[0, 25, 30, np.inf],
                                          labels=['Normal\n(<25)', 'Overweight\n(25-30)', 'Obese\n(≥30)'])
bmi_race_diabetes = bmi_race_cat_data.groupby(['Race_Label', 'BMI_Category'], observed=True)['Has_Diabetes'].mean().unstack() * 100

bmi_race_diabetes.plot(kind='bar', ax=axes[1, 0], color=['#3498db', '#f39c12', '#e74c3c'],
                      alpha=0.7, edgecolor='black', linewidth=1)
axes[1, 0].set_ylabel('Diabetes Prevalence (%)', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Race/Ethnicity', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Diabetes by Race and BMI Category', fontsize=12, fontweight='bold', pad=15)
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
axes[1, 0].tick_params(axis='x', rotation=0)

income_race_data = df[df['INDFMPIR'].notna() & df['Has_Diabetes'].notna() & df['Race_Label'].notna()].copy()
income_labels = ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']
if not income_race_data.empty:
    income_race_data['Income_Quartile'] = pd.qcut(
        income_race_data['INDFMPIR'],
        q=4,
        labels=income_labels,
        duplicates='drop'
    )
    income_race_diabetes = income_race_data.groupby(['Race_Label', 'Income_Quartile'], observed=False)['Has_Diabetes'].mean().unstack() * 100
    income_race_diabetes = income_race_diabetes.reindex(columns=[label for label in income_labels if label in income_race_diabetes.columns])

    income_colors = ['#9b59b6', '#e67e22', '#16a085', '#34495e']
    income_race_diabetes.plot(kind='bar', ax=axes[1, 1], color=income_colors[:len(income_race_diabetes.columns)],
                              alpha=0.7, edgecolor='black', linewidth=1)
    axes[1, 1].set_ylabel('Diabetes Prevalence (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Race/Ethnicity', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Diabetes by Race and Income Quartiles', fontsize=12, fontweight='bold', pad=15)
    axes[1, 1].legend(income_race_diabetes.columns, fontsize=10, title='Income (PIR)')
    axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1, 1].tick_params(axis='x', rotation=0)
else:
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, 0.5, 'Insufficient data for plot', ha='center', va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '9_risk_factors_by_race.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/9_risk_factors_by_race.png")
plt.close()

print("Creating Visualization 9B: Hypertension Risk Factor Interactions by Race")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

smoking_race_data_htn = df[df['SMQ020'].notna() & df['Has_Hypertension'].notna() & df['Race_Label'].notna()].copy()
smoking_race_data_htn['Smoking_Status'] = np.where(
    smoking_race_data_htn['SMQ020'].replace([7, 9], np.nan) == 1,
    'Ever Smoked',
    'Never Smoked'
)
smoking_race_htn = smoking_race_data_htn.groupby(['Race_Label', 'Smoking_Status'])['Has_Hypertension'].mean().unstack() * 100
smoking_race_htn = smoking_race_htn.reindex(columns=['Never Smoked', 'Ever Smoked'])

if not smoking_race_htn.empty:
    smoking_race_htn.plot(kind='bar', ax=axes[0, 0], color=['#5dade2', '#2874a6'],
                          alpha=0.75, edgecolor='black', linewidth=1)
    axes[0, 0].set_ylabel('Hypertension Prevalence (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Race/Ethnicity', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Hypertension by Race and Smoking Status', fontsize=12, fontweight='bold', pad=15)
    axes[0, 0].legend(fontsize=10, title='Smoking Status')
    axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0, 0].tick_params(axis='x', rotation=0)
else:
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, 0.5, 'Insufficient data for plot', ha='center', va='center', fontsize=12, fontweight='bold')

pa_race_data_htn = df[df['PAD680'].notna() & df['Has_Hypertension'].notna() & df['Race_Label'].notna()].copy()
pa_labels_htn = ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']
if not pa_race_data_htn.empty:
    pa_race_data_htn['PA_Quartile'] = pd.qcut(
        pa_race_data_htn['PAD680'],
        q=4,
        labels=pa_labels_htn,
        duplicates='drop'
    )
    pa_race_htn = pa_race_data_htn.groupby(['Race_Label', 'PA_Quartile'], observed=False)['Has_Hypertension'].mean().unstack() * 100
    pa_race_htn = pa_race_htn.reindex(columns=[label for label in pa_labels_htn if label in pa_race_htn.columns])

    pa_htn_colors = ['#95a5a6', '#1abc9c', '#2980b9', '#8e44ad']
    pa_race_htn.plot(kind='bar', ax=axes[0, 1], color=pa_htn_colors[:len(pa_race_htn.columns)],
                     alpha=0.75, edgecolor='black', linewidth=1)
    axes[0, 1].set_ylabel('Hypertension Prevalence (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Race/Ethnicity', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Hypertension by Race and Physical Activity Quartiles', fontsize=12, fontweight='bold', pad=15)
    axes[0, 1].legend(pa_race_htn.columns, fontsize=10, title='Physical Activity')
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0, 1].tick_params(axis='x', rotation=0)
else:
    axes[0, 1].axis('off')
    axes[0, 1].text(0.5, 0.5, 'Insufficient data for plot', ha='center', va='center', fontsize=12, fontweight='bold')

bmi_race_cat_data_htn = df[df['BMXBMI'].notna() & df['Has_Hypertension'].notna() & df['Race_Label'].notna()].copy()
bmi_race_cat_data_htn['BMI_Category'] = pd.cut(
    bmi_race_cat_data_htn['BMXBMI'],
    bins=[0, 25, 30, np.inf],
    labels=['Normal (<25)', 'Overweight (25-30)', 'Obese (≥30)']
)
bmi_race_htn = bmi_race_cat_data_htn.groupby(['Race_Label', 'BMI_Category'], observed=True)['Has_Hypertension'].mean().unstack() * 100

if not bmi_race_htn.empty:
    bmi_race_htn.plot(kind='bar', ax=axes[1, 0], color=['#3498db', '#f1c40f', '#e74c3c'],
                      alpha=0.75, edgecolor='black', linewidth=1)
    axes[1, 0].set_ylabel('Hypertension Prevalence (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Race/Ethnicity', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Hypertension by Race and BMI Category', fontsize=12, fontweight='bold', pad=15)
    axes[1, 0].legend(fontsize=9, title='BMI Category')
    axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1, 0].tick_params(axis='x', rotation=0)
else:
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'Insufficient data for plot', ha='center', va='center', fontsize=12, fontweight='bold')

income_race_data_htn = df[df['INDFMPIR'].notna() & df['Has_Hypertension'].notna() & df['Race_Label'].notna()].copy()
income_labels_htn = ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']
if not income_race_data_htn.empty:
    income_race_data_htn['Income_Quartile'] = pd.qcut(
        income_race_data_htn['INDFMPIR'],
        q=4,
        labels=income_labels_htn,
        duplicates='drop'
    )
    income_race_htn = income_race_data_htn.groupby(['Race_Label', 'Income_Quartile'], observed=False)['Has_Hypertension'].mean().unstack() * 100
    income_race_htn = income_race_htn.reindex(columns=[label for label in income_labels_htn if label in income_race_htn.columns])

    income_htn_colors = ['#9b59b6', '#e67e22', '#16a085', '#34495e']
    income_race_htn.plot(kind='bar', ax=axes[1, 1], color=income_htn_colors[:len(income_race_htn.columns)],
                         alpha=0.75, edgecolor='black', linewidth=1)
    axes[1, 1].set_ylabel('Hypertension Prevalence (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Race/Ethnicity', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Hypertension by Race and Income Quartiles', fontsize=12, fontweight='bold', pad=15)
    axes[1, 1].legend(income_race_htn.columns, fontsize=9, title='Income (PIR)')
    axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1, 1].tick_params(axis='x', rotation=0)
else:
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, 0.5, 'Insufficient data for plot', ha='center', va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '9b_hypertension_risk_factors_by_race.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/9b_hypertension_risk_factors_by_race.png")
plt.close()

print("Creating Visualization 10: Dietary Risk Patterns")

diet_plot_df = df[df['DR1TKCAL'].notna() & df['Has_Diabetes'].notna()].copy()
if not diet_plot_df.empty:
    diet_plot_df['Calorie_Bin'] = pd.qcut(diet_plot_df['DR1TKCAL'], q=10, duplicates='drop')
    calorie_prev = diet_plot_df.groupby('Calorie_Bin', observed=False)['Has_Diabetes'].mean().reset_index()
    calorie_prev['Midpoint'] = calorie_prev['Calorie_Bin'].apply(lambda interval: 0.5 * (interval.left + interval.right))
else:
    calorie_prev = pd.DataFrame({'Midpoint': [], 'Has_Diabetes': []})

sodium_plot_df = df[df['DR1TSODI'].notna() & df['Has_Hypertension'].notna()].copy()
if not sodium_plot_df.empty:
    sodium_plot_df['Sodium_Bin'] = pd.qcut(sodium_plot_df['DR1TSODI'], q=10, duplicates='drop')
    sodium_prev = sodium_plot_df.groupby('Sodium_Bin', observed=False)['Has_Hypertension'].mean().reset_index()
    sodium_prev['Midpoint'] = sodium_prev['Sodium_Bin'].apply(lambda interval: 0.5 * (interval.left + interval.right))
else:
    sodium_prev = pd.DataFrame({'Midpoint': [], 'Has_Hypertension': []})

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(calorie_prev['Midpoint'], calorie_prev['Has_Diabetes'] * 100,
             marker='o', linewidth=2, color='#3498db')
axes[0].fill_between(calorie_prev['Midpoint'], calorie_prev['Has_Diabetes'] * 100,
                     alpha=0.15, color='#3498db')
axes[0].set_ylabel('Diabetes Prevalence (%)', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Daily Calorie Intake (kcal)', fontsize=11, fontweight='bold')
axes[0].set_title('Diabetes Prevalence Across Calorie Intake Deciles', fontsize=12, fontweight='bold', pad=15)
axes[0].grid(alpha=0.3, linestyle='--')

axes[1].plot(sodium_prev['Midpoint'], sodium_prev['Has_Hypertension'] * 100,
             marker='s', linewidth=2, color='#e74c3c')
axes[1].fill_between(sodium_prev['Midpoint'], sodium_prev['Has_Hypertension'] * 100,
                     alpha=0.15, color='#e74c3c')
axes[1].set_ylabel('Hypertension Prevalence (%)', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Daily Sodium Intake (mg)', fontsize=11, fontweight='bold')
axes[1].set_title('Hypertension Prevalence Across Sodium Intake Deciles', fontsize=12, fontweight='bold', pad=15)
axes[1].grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '10_dietary_risk_patterns.png'), dpi=300, bbox_inches='tight')
print(f"  Saved: {output_dir}/10_dietary_risk_patterns.png")
plt.close()

print(f"\nAll visualizations saved to: {os.path.abspath(output_dir)}/")
print("EDA Complete")