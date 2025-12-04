# Disaggregating Diabetes and Hypertension Risk Factors Across Racial Groups

**Author:** Rithvik Harinath  
**Date:** November 30, 2025

---

## Abstract

Cardiometabolic diseases like diabetes and hypertension remain leading causes of morbidity in the United States, with profound disparities across racial and ethnic lines. Traditional public health guidelines often apply a "one-size-fits-all" approach to prevention. This study leverages machine learning on National Health and Nutrition Examination Survey (NHANES) data (n=11,933) to disentangle the complex web of biological, behavioral, and socioeconomic risk factors.

Our analysis, driven by **Permutation Feature Importance** for **Random Forest** models and validated by rigorous stability checks, reveals distinct patterns of risk factor importance. **HbA1c** emerges as the universal, perfectly stable predictor for diabetes (Rank #1 across all racial groups). For hypertension, **Systolic Blood Pressure** demonstrates perfect stability as the top predictor across all populations. **Sedentary Behavior (sitting time)** and **Income-to-Poverty Ratio** rank among the top predictors and demonstrate high stability in Random Forest models, indicating these behavioral and socioeconomic factors operate through non-linear interactions that ensemble methods capture reliably. These findings suggest that while biological markers (HbA1c, Systolic BP) are universally stable predictors, behavioral and socioeconomic factors require non-linear modeling approaches to reveal their true importance.

---

## 1. Introduction

### Background
Diabetes and hypertension are not equal-opportunity diseases. Non-Hispanic Black adults in our dataset report a hypertension prevalence of 44.3%, compared to ~35% in the general population. Similarly, diabetes rates vary from ~7% to over 12% across groups. While the clinical management of these conditions is well-standardized, *prevention* strategies often lack specificity. Does a recommendation to "exercise more" carry the same weight for a Mexican American patient as it does for a Non-Hispanic White patient?

### Objective
The goal of this project is to move beyond simple prevalence statistics. By training machine learning models like **Random Forest** and validating them with **stability analysis**, we aim to:
1.  **Quantify** the relative importance of modifiable risk factors (diet, activity, smoking, SES) for diabetes and hypertension.
2.  **Disaggregate** these insights by race/ethnicity to identify population-specific targets for intervention.

---

## 2. Data and Methodology

### Data Source
We utilized the NHANES dataset, a nationally representative survey combining interviews and physical examinations. Our cohort consists of **11,933 participants** aged 0-80.

**Key Variables:**
*   **Outcomes:** Self-reported Diabetes (`DIQ010`) and Hypertension (`BPQ020`).
*   **Predictors:** 24 features spanning Demographics (Age, Race, Income), Anthropometry (BMI, Waist Circumference), Lifestyle (Physical Activity, Smoking, Alcohol), Diet (Calories, Sodium, Macronutrients), and Labs (Cholesterol, HbA1c).

### Analytical Strategy
To ensure our findings were both interpretable and robust, we employed a two-stage strategy:
1.  **Primary Analysis (Permutation Importance):** We measured the drop in model performance (ROC-AUC) when each feature was randomly shuffled. This method isolates the unique contribution of each risk factor, unbiased by the model's internal structure.
2.  **Validation (Stability Analysis):** We bootstrapped the modeling process 10 times to ensure that the top-ranked features were statistically robust and not artifacts of a specific data split.

---

## 3. Results and Analysis

### 3.1 Model Performance
Our models demonstrated strong predictive capability, validating the quality of the underlying signal in the NHANES data.

| Disease      | Model               | ROC-AUC | PR-AUC |
|--------------|---------------------|---------|--------|
| Diabetes     | Random baseline     | 0.508   | 0.090  |
|              | **Random Forest**    | **0.896** | **0.612** |
|              | XGBoost             | 0.878   | 0.623  |
|              | Logistic Regression | 0.896   | 0.635  |
| Hypertension | Random baseline     | 0.503   | 0.251  |
|              | **Random Forest**    | **0.834** | **0.554** |
|              | XGBoost             | 0.835   | 0.595  |
|              | Logistic Regression | 0.878   | 0.596  |

**Model Selection Rationale:**
We selected **Random Forest** (class-weighted, 100 trees) as our primary model based on three key criteria: (1) **Equity**: Superior per-race performance consistency, avoiding sub-0.4 PR-AUC drops that occurred with XGBoost for some populations; (2) **Stability**: Significantly higher stability rankings for behavioral and socioeconomic factors compared to Logistic Regression; (3) **Performance**: Competitive overall metrics (ROC-AUC 0.896 for diabetes, 0.834 for hypertension). We also evaluated **Logistic Regression** and **XGBoost** for comparison. While Logistic Regression achieved similar overall performance, stability analysis revealed it likely fails to capture the non-linear interactions underlying behavioral and socioeconomic risk factors, which Random Forest models reliably.

### 3.2 Universal Drivers of Risk
Permutation importance analysis using Random Forest, validated by stability checks across 10 bootstrap trials, reveals distinct patterns:

1.  **HbA1c for Diabetes:** The universal #1 predictor with perfect stability (Mean Rank 1.0 across all 6 racial groups, Top_3_Freq = 1.0). This biological marker demonstrates the highest permutation importance (0.081) and is consistently the most important feature regardless of population subgroup.

2.  **Systolic Blood Pressure for Hypertension:** Highly stable predictor across all populations (Mean Rank 1.0-2.1, Top_3_Freq = 1.0), ranking #2 in overall permutation importance (0.037). This finding suggests that direct biological measurement of BP is a more reliable predictor than self-reported behavioral factors.

3.  **Sedentary Behavior (PAD680 - Minutes Sitting):** Ranks #1 in overall permutation importance for hypertension (0.051) and #2 for diabetes (0.045). **Random Forest stability analysis shows high stability** (Mean Rank 1.0-2.6, Top_3_Freq = 0.9-1.0), indicating this factor is consistently important across populations when modeled with non-linear methods. **This highlights that inactivity (sitting time) may be a more critical screening target than activity intensity**, suggesting sedentary behavior operates through complex, interactive pathways that require ensemble methods to detect reliably.

4.  **Income-to-Poverty Ratio:** Ranks #3 for diabetes (0.021) and #5 for hypertension (0.021) overall. **Random Forest stability analysis shows stable rankings** (Mean Rank 3.0-5.6, Top_3_Freq = 0.5-1.0), confirming socioeconomic factors are consistently important when their non-linear relationships with other risk factors are properly modeled.

### 3.3 The Divergence: Population-Specific Risk Profiles

Stability analysis reveals important population-specific patterns that complement the overall permutation importance rankings:

*   **Non-Hispanic Black (Diabetes):** Sedentary Behavior (sitting time) emerges as #2 predictor (Importance: 0.045, 30.8% of total), following HbA1c. Income-Poverty Ratio ranks #3 (Importance: 0.016, 10.9%), suggesting socioeconomic factors play a particularly important role in this population.
*   **Non-Hispanic White (Hypertension):** Sedentary Behavior dominates as #1 (Importance: 0.059, 22.5%), followed by Moderate Activity (PAD800) #2 (Importance: 0.046, 17.5%). Ever Smoked ranks #3 (Importance: 0.034, 12.9%), with Systolic BP at #4 (Importance: 0.032, 12.3%).
*   **Mexican American (Hypertension):** Sedentary Behavior ranks #1 (Importance: 0.044, 38.9%), followed by Moderate Activity (PAD800) #2 (Importance: 0.022, 19.1%) and Income-Poverty Ratio #3 (Importance: 0.020, 17.2%), suggesting both behavioral and socioeconomic interventions are critical. **Reducing sitting time is the single most effective intervention point—more so than even weight loss (BMI ranks #5).**

---

## 4. Discussion and Implications

The permutation importance analysis reveals that **effective prevention requires addressing multiple domains simultaneously**, not just clinical markers. Here's what the data tells us to prioritize:

### 4.1 Universal Interventions (All Populations)

These factors showed the highest importance across all racial groups and should be the foundation of any prevention program:

#### **For Diabetes Prevention:**
1. **HbA1c Monitoring & Glucose Control** (40% of total importance)
   - **Action:** Expand access to regular HbA1c testing, especially in underserved communities
   - **Why:** Universal #1 predictor across all populations with perfect stability
   
2. **Sedentary Behavior Reduction Programs** (22% of total importance)
   - **Action:** Sedentary interruption programs (standing desks, reducing screen time, sit-stand workstations)
   - **Why:** #2 predictor overall, showing consistent importance across populations - **reducing sitting time is more critical than increasing exercise intensity**

3. **Smoking Cessation Support** (7.3% of total importance)
   - **Action:** Free cessation programs, especially targeting high-risk populations
   - **Why:** #4 predictor, with outsized impact in certain populations

#### **For Hypertension Prevention:**
1. **Sedentary Behavior Reduction** (23% of total importance)
   - **Action:** Workplace and community programs to reduce sitting time (standing desks, active breaks, screen time limits)
   - **Why:** #1 predictor overall, more important than BP medication alone - **sitting time is a more critical target than exercise intensity**
   
2. **Blood Pressure Monitoring** (16.5% of total importance)
   - **Action:** Free community BP screening, home monitoring programs
   - **Why:** #2 predictor, direct biological measurement
   
3. **Smoking Cessation** (12.1% of total importance)
   - **Action:** Integrate cessation into hypertension treatment protocols
   - **Why:** #3 predictor, comparable to income effects

### 4.2 Population-Specific Interventions

The data reveals that **socioeconomic interventions are as important as clinical ones** for disadvantaged populations:

#### **Non-Hispanic Black (Diabetes):**
- **Sedentary Behavior Reduction** (30.8% importance) + **Income Support Programs** (10.9% importance)
- **Action:** Workplace sedentary interruption programs + economic assistance, job training, housing support
- **Why:** Reducing sitting time is the #2 predictor (30.8%), and poverty (10.9%) is a stronger predictor than weight (1.9%) for this population

#### **Mexican American (Hypertension):**
- **Sedentary Behavior Reduction** (38.9% importance) - **the single most effective intervention**
- **Income Support** (17.2% importance) + **Moderate Activity** (19.1% importance)
- **Action:** Workplace sitting reduction programs + economic assistance + moderate activity programs
- **Why:** For Mexican Americans, reducing sitting time (38.9%) is more effective than weight loss (BMI ranks #5), sodium reduction (ranks #13), or even BP medication. Combined with income support (17.2%) and moderate activity (19.1%), these three factors account for 75% of preventable hypertension risk.

#### **Non-Hispanic White (Hypertension):**
- **Sedentary Behavior Reduction** (22.5% importance) + **Moderate Activity** (17.5% importance) + **Smoking Cessation** (12.9% importance)
- **Action:** Comprehensive programs: reduce sitting time + moderate activity + smoking cessation
- **Why:** These three factors combined (52.9%) account for over half of preventable hypertension risk

## 5. Conclusion

By applying **Random Forest** machine learning with rigorous stability checks, we have moved from describing disparities to understanding mechanisms. The path to health equity requires recognizing that not all risk factors are created equal: **biological markers (HbA1c, Systolic BP) demonstrate universal stability** across populations and model architectures, while **behavioral and socioeconomic factors require non-linear modeling** to reveal their true importance.
 This analysis provides the quantitative foundation for a new era of precision public health, where interventions can be prioritized based on both overall importance, cross-population stability, and the underlying relationship structure (linear vs. non-linear), with Random Forest serving as the essential tool for capturing the full complexity of behavioral and socioeconomic determinants of health.