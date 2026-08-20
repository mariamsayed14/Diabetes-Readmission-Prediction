from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier


# =========================================================
# FASTAPI
# =========================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Diabetes Readmission Prediction",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# GLOBAL VARIABLES
# =========================================================

model = None
rf_tuned_model = None
xgb_model = None
knn_model = None
logreg_model = None
model_results = None
best_model_name = None

numerical_scaler = None
age_scaler = None
glu_scaler = None
a1c_scaler = None

diagnosis_encoder = None
categorical_encoder = None
id_encoder = None
medication_encoder = None

label_encoder = None

feature_columns = None


# =========================================================
# LOAD AND TRAIN MODEL
# =========================================================

def train_model():

    global model
    global rf_tuned_model
    global xgb_model
    global knn_model
    global logreg_model
    global model_results
    global best_model_name
    global numerical_scaler
    global age_scaler
    global glu_scaler
    global a1c_scaler
    global diagnosis_encoder
    global categorical_encoder
    global id_encoder
    global medication_encoder
    global label_encoder
    global feature_columns


    print("\n========================================")
    print("Loading dataset...")
    print("========================================")


    df = pd.read_csv(
        "diabetic_data.csv"
    )


    # =====================================================
    # BASIC CLEANING
    # =====================================================

    df = df.replace(
        "?",
        np.nan
    )


    df["gender"] = df["gender"].replace(
        "Unknown/Invalid",
        np.nan
    )


    # =====================================================
    # X / Y / GROUPS
    # =====================================================

    X = df.drop(columns=["readmitted", "examide", "citoglipton"])
    y = df["readmitted"]

    groups = df["patient_nbr"]


    # =====================================================
    # GROUP SHUFFLE SPLIT
    # Same approach as notebook
    # =====================================================

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )


    train_idx, test_idx = next(
        gss.split(
            X,
            y,
            groups=groups
        )
    )


    X_train = X.iloc[
        train_idx
    ].copy()

    X_test = X.iloc[
        test_idx
    ].copy()

    y_train = y.iloc[
        train_idx
    ].copy()

    y_test = y.iloc[
        test_idx
    ].copy()


    # =====================================================
    # REMOVE WEIGHT
    # =====================================================

    X_train = X_train.drop(
        columns=["weight"]
    )

    X_test = X_test.drop(
        columns=["weight"]
    )


    # =====================================================
    # MISSING VALUES
    # Based on notebook
    # =====================================================

    cols_to_fill = [
        "medical_specialty",
        "payer_code",
        "race"
    ]


    for col in cols_to_fill:

        X_train[col] = X_train[
            col
        ].fillna("unkown")

        X_test[col] = X_test[
            col
        ].fillna("unkown")


    X_train["max_glu_serum"] = (
        X_train["max_glu_serum"]
        .fillna("not taken")
    )

    X_test["max_glu_serum"] = (
        X_test["max_glu_serum"]
        .fillna("not taken")
    )


    X_train["A1Cresult"] = (
        X_train["A1Cresult"]
        .fillna("not taken")
    )

    X_test["A1Cresult"] = (
        X_test["A1Cresult"]
        .fillna("not taken")
    )


    # =====================================================
    # ID COLUMNS
    # =====================================================

    id_columns = [
        "admission_type_id",
        "admission_source_id",
        "discharge_disposition_id"
    ]


    for col in id_columns:

        mode_val = X_train[
            col
        ].mode()[0]

        X_train[col] = (
            X_train[col]
            .fillna(mode_val)
        )

        X_test[col] = (
            X_test[col]
            .fillna(mode_val)
        )


    # =====================================================
    # GENDER
    # =====================================================

    mode_gender = (
        X_train["gender"]
        .mode()[0]
    )


    X_train["gender"] = (
        X_train["gender"]
        .fillna(mode_gender)
    )

    X_test["gender"] = (
        X_test["gender"]
        .fillna(mode_gender)
    )


    # =====================================================
    # DIAGNOSIS
    # =====================================================

    for col in [
        "diag_1",
        "diag_2",
        "diag_3"
    ]:

        X_train[col] = (
            X_train[col]
            .fillna("Unknown")
        )

        X_test[col] = (
            X_test[col]
            .fillna("Unknown")
        )


    # =====================================================
    # DIAGNOSIS CATEGORIZATION
    # =====================================================

    def categorize_diagnosis(code):

        code = str(code)


        if code.startswith("V"):

            return "Supplemental"


        if code.startswith("E"):

            return "External"


        try:

            code_num = float(code)

        except:

            if code == "Unknown":

                return "Unknown"

            return "Other"


        if 250 <= code_num < 251:

            return "Diabetes"

        elif 390 <= code_num <= 459:

            return "Circulatory"

        elif 460 <= code_num <= 519:

            return "Respiratory"

        elif 520 <= code_num <= 579:

            return "Digestive"

        elif 800 <= code_num <= 999:

            return "Injury"

        elif 710 <= code_num <= 739:

            return "Musculoskeletal"

        elif 580 <= code_num <= 629:

            return "Genitourinary"

        elif 140 <= code_num <= 239:

            return "Neoplasms"

        else:

            return "Others"


    for col in [
        "diag_1",
        "diag_2",
        "diag_3"
    ]:

        X_train[
            f"{col}_category"
        ] = X_train[
            col
        ].apply(
            categorize_diagnosis
        )


        X_test[
            f"{col}_category"
        ] = X_test[
            col
        ].apply(
            categorize_diagnosis
        )


    # =====================================================
    # LOG TRANSFORMATION
    # =====================================================

    skewed_features = [

        "number_emergency",

        "number_outpatient",

        "number_inpatient",

        "num_medications",

        "num_procedures",

        "time_in_hospital"

    ]


    for col in skewed_features:

        X_train[col] = np.log1p(
            X_train[col]
        )

        X_test[col] = np.log1p(
            X_test[col]
        )


    # =====================================================
    # STANDARD SCALING
    # =====================================================

    numerical_columns = [

        "time_in_hospital",

        "num_lab_procedures",

        "num_procedures",

        "num_medications",

        "number_outpatient",

        "number_emergency",

        "number_inpatient",

        "number_diagnoses"

    ]


    numerical_scaler = (
        StandardScaler()
    )


    X_train[
        numerical_columns
    ] = numerical_scaler.fit_transform(
        X_train[
            numerical_columns
        ]
    )


    X_test[
        numerical_columns
    ] = numerical_scaler.transform(
        X_test[
            numerical_columns
        ]
    )


    # =====================================================
    # BINARY CATEGORICAL
    # =====================================================

    binary_mappings = {

        "gender": {
            "Female": 0,
            "Male": 1
        },

        "change": {
            "No": 0,
            "Ch": 1
        },

        "diabetesMed": {
            "No": 0,
            "Yes": 1
        }

    }


    for col, mapping in binary_mappings.items():

        X_train[col] = (
            X_train[col]
            .map(mapping)
        )

        X_test[col] = (
            X_test[col]
            .map(mapping)
        )


    # =====================================================
    # AGE
    # =====================================================

    age_mapping = {

        "[0-10)": 0,
        "[10-20)": 1,
        "[20-30)": 2,
        "[30-40)": 3,
        "[40-50)": 4,
        "[50-60)": 5,
        "[60-70)": 6,
        "[70-80)": 7,
        "[80-90)": 8,
        "[90-100)": 9

    }


    X_train["age"] = (
        X_train["age"]
        .map(age_mapping)
    )

    X_test["age"] = (
        X_test["age"]
        .map(age_mapping)
    )


    age_scaler = StandardScaler()


    X_train[
        ["age"]
    ] = age_scaler.fit_transform(
        X_train[
            ["age"]
        ]
    )


    X_test[
        ["age"]
    ] = age_scaler.transform(
        X_test[
            ["age"]
        ]
    )


    # =====================================================
    # GLUCOSE
    # =====================================================

    glu_mapping = {

        "not taken": 0,
        "Norm": 1,
        ">200": 2,
        ">300": 3

    }


    X_train[
        "max_glu_serum"
    ] = X_train[
        "max_glu_serum"
    ].map(glu_mapping)


    X_test[
        "max_glu_serum"
    ] = X_test[
        "max_glu_serum"
    ].map(glu_mapping)


    glu_scaler = StandardScaler()


    X_train[
        ["max_glu_serum"]
    ] = glu_scaler.fit_transform(
        X_train[
            ["max_glu_serum"]
        ]
    )


    X_test[
        ["max_glu_serum"]
    ] = glu_scaler.transform(
        X_test[
            ["max_glu_serum"]
        ]
    )


    # =====================================================
    # A1C
    # =====================================================

    a1c_mapping = {

        "not taken": 0,
        "Norm": 1,
        ">7": 2,
        ">8": 3

    }


    X_train[
        "A1Cresult"
    ] = X_train[
        "A1Cresult"
    ].map(a1c_mapping)


    X_test[
        "A1Cresult"
    ] = X_test[
        "A1Cresult"
    ].map(a1c_mapping)


    a1c_scaler = StandardScaler()


    X_train[
        ["A1Cresult"]
    ] = a1c_scaler.fit_transform(
        X_train[
            ["A1Cresult"]
        ]
    )


    X_test[
        ["A1Cresult"]
    ] = a1c_scaler.transform(
        X_test[
            ["A1Cresult"]
        ]
    )


    # =====================================================
    # ONE HOT - DIAGNOSIS
    # =====================================================

    diagnosis_columns = [

        "diag_1_category",
        "diag_2_category",
        "diag_3_category"

    ]


    diagnosis_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )


    X_train_diag = (
        diagnosis_encoder
        .fit_transform(
            X_train[
                diagnosis_columns
            ]
        )
    )


    X_test_diag = (
        diagnosis_encoder
        .transform(
            X_test[
                diagnosis_columns
            ]
        )
    )


    X_train_diag = pd.DataFrame(
        X_train_diag,
        index=X_train.index,
        columns=
        diagnosis_encoder
        .get_feature_names_out(
            diagnosis_columns
        )
    )


    X_test_diag = pd.DataFrame(
        X_test_diag,
        index=X_test.index,
        columns=
        diagnosis_encoder
        .get_feature_names_out(
            diagnosis_columns
        )
    )


    # =====================================================
    # ONE HOT - RACE / PAYER / SPECIALTY
    # =====================================================

    nominal_columns = [

        "race",
        "payer_code",
        "medical_specialty"

    ]


    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )


    X_train_cat = (
        categorical_encoder
        .fit_transform(
            X_train[
                nominal_columns
            ]
        )
    )


    X_test_cat = (
        categorical_encoder
        .transform(
            X_test[
                nominal_columns
            ]
        )
    )


    X_train_cat = pd.DataFrame(
        X_train_cat,
        index=X_train.index,
        columns=
        categorical_encoder
        .get_feature_names_out(
            nominal_columns
        )
    )


    X_test_cat = pd.DataFrame(
        X_test_cat,
        index=X_test.index,
        columns=
        categorical_encoder
        .get_feature_names_out(
            nominal_columns
        )
    )


    # =====================================================
    # ONE HOT - IDS
    # =====================================================

    id_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )


    X_train_id = (
        id_encoder
        .fit_transform(
            X_train[
                id_columns
            ]
        )
    )


    X_test_id = (
        id_encoder
        .transform(
            X_test[
                id_columns
            ]
        )
    )


    X_train_id = pd.DataFrame(
        X_train_id,
        index=X_train.index,
        columns=
        id_encoder
        .get_feature_names_out(
            id_columns
        )
    )


    X_test_id = pd.DataFrame(
        X_test_id,
        index=X_test.index,
        columns=
        id_encoder
        .get_feature_names_out(
            id_columns
        )
    )


    # =====================================================
    # ONE HOT - MEDICATIONS
    # =====================================================

    medication_columns = [

        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
        "acetohexamide",
        "glimepiride-pioglitazone"

    ]


    medication_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )


    X_train_med = (
        medication_encoder
        .fit_transform(
            X_train[
                medication_columns
            ]
        )
    )


    X_test_med = (
        medication_encoder
        .transform(
            X_test[
                medication_columns
            ]
        )
    )


    X_train_med = pd.DataFrame(
        X_train_med,
        index=X_train.index,
        columns=
        medication_encoder
        .get_feature_names_out(
            medication_columns
        )
    )


    X_test_med = pd.DataFrame(
        X_test_med,
        index=X_test.index,
        columns=
        medication_encoder
        .get_feature_names_out(
            medication_columns
        )
    )


    # =====================================================
    # DROP ORIGINAL COLUMNS
    # =====================================================

    columns_to_drop = (

        diagnosis_columns
        + nominal_columns
        + id_columns
        + medication_columns
        + [
            "diag_1",
            "diag_2",
            "diag_3"
        ]

    )


    X_train_base = X_train.drop(
        columns=columns_to_drop
    )


    X_test_base = X_test.drop(
        columns=columns_to_drop
    )


    # =====================================================
    # COMBINE
    # =====================================================

    X_train_encoded = pd.concat(
        [

            X_train_base,
            X_train_diag,
            X_train_cat,
            X_train_id,
            X_train_med

        ],
        axis=1
    )


    X_test_encoded = pd.concat(
        [

            X_test_base,
            X_test_diag,
            X_test_cat,
            X_test_id,
            X_test_med

        ],
        axis=1
    )


    X_train_encoded = (
        X_train_encoded
        .reset_index(drop=True)
    )

    X_test_encoded = (
        X_test_encoded
        .reset_index(drop=True)
    )


    y_train = (
        y_train
        .reset_index(drop=True)
    )

    y_test = (
        y_test
        .reset_index(drop=True)
    )


    # =====================================================
    # LABEL ENCODING
    # =====================================================

    label_encoder = LabelEncoder()


    y_train_encoded = (
        label_encoder
        .fit_transform(y_train)
    )


    y_test_encoded = (
        label_encoder
        .transform(y_test)
    )


    print(
        "Classes:",
        label_encoder.classes_
    )


    # =====================================================
    # SMOTE
    # =====================================================

    print("\nChecking X_train_encoded...")
    print(X_train_encoded.dtypes)

    non_numeric = X_train_encoded.select_dtypes(exclude=["number"])

    print("\nNon-numeric columns:")
    print(non_numeric.columns.tolist())

    for col in non_numeric.columns:
        print(f"\n{col}:")
        print(non_numeric[col].unique()[:20])

    smote = SMOTE(
        sampling_strategy={
            0: 13000,
            1: 30000
        },
        random_state=42
    )


    X_train_smote, y_train_smote = (
        smote.fit_resample(
            X_train_encoded,
            y_train_encoded
        )
    )


    # =====================================================
    # BINARY TARGET
    #
    # <30  -> 1 Readmitted
    # >30  -> 1 Readmitted
    # NO   -> 0 Not Readmitted
    # =====================================================

    y_train_binary = (
        pd.Series(
            y_train_smote
        )
        .map({
            0: 1,
            1: 1,
            2: 0
        })
    )


    # =====================================================
    # DROP ID COLUMNS BEFORE MODELING
    # Same as notebook
    # =====================================================

    X_train_encoded = X_train_encoded.drop(
        columns=["encounter_id", "patient_nbr"],
        errors="ignore"
    )

    X_test_encoded = X_test_encoded.drop(
        columns=["encounter_id", "patient_nbr"],
        errors="ignore"
    )

    # =====================================================
    # BINARY TARGET
    # NO -> 0 (Not Readmitted)
    # >30 and <30 -> 1 (Readmitted)
    # =====================================================

    y_train_binary = y_train.map({
        "NO": 0,
        ">30": 1,
        "<30": 1
    })

    y_test_binary = y_test.map({
        "NO": 0,
        ">30": 1,
        "<30": 1
    })

    # Remove NaN target rows exactly as notebook
    nan_mask_train = y_train_binary.isna()

    X_train_encoded_cleaned = X_train_encoded[~nan_mask_train]
    y_train_binary_cleaned = y_train_binary[~nan_mask_train]

    X_test_encoded_cleaned = X_test_encoded
    y_test_binary_cleaned = y_test_binary

    # =====================================================
    # LABEL ENCODING
    # =====================================================

    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(
        y_train_binary_cleaned
    )

    y_test_encoded = label_encoder.transform(
        y_test_binary_cleaned
    )

    print("Classes:", label_encoder.classes_)

    # =====================================================
    # RANDOM FOREST - GRIDSEARCH BEFORE SMOTETOMEK
    # =====================================================

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    rf_param_grid = {
        "n_estimators": [200],
        "max_depth": [None, 20],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "max_features": ["sqrt"]
    }

    rf_grid = GridSearchCV(
        estimator=rf,
        param_grid=rf_param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=2,
        verbose=1
    )

    print("Running Random Forest GridSearchCV...")
    rf_grid.fit(
        X_train_encoded_cleaned,
        y_train_encoded
    )

    y_pred_rf = rf_grid.predict(
        X_test_encoded_cleaned
    )

    rf_accuracy = accuracy_score(
        y_test_encoded,
        y_pred_rf
    )

    print(f"Random Forest GridSearch Test Accuracy: {rf_accuracy:.2%}")

    # =====================================================
    # SMOTETOMEK - SAME AS NOTEBOOK
    # =====================================================

    smote_tomek = SMOTETomek(
        random_state=42
    )

    X_train_encoded_imputed = (
        X_train_encoded_cleaned
        .fillna(X_train_encoded_cleaned.mean())
    )

    X_train_balanced, y_train_balanced = (
        smote_tomek.fit_resample(
            X_train_encoded_imputed,
            y_train_encoded
        )
    )

    # =====================================================
    # RANDOM FOREST + SMOTETOMEK
    # =====================================================

    rf_smote = RandomForestClassifier(
        random_state=42,
        n_jobs=2
    )

    rf_smote.fit(
        X_train_balanced,
        y_train_balanced
    )

    y_pred_rf_smote = rf_smote.predict(
        X_test_encoded
    )

    rf_smote_accuracy = accuracy_score(
        y_test_encoded,
        y_pred_rf_smote
    )

    print(
        f"Random Forest + SMOTETomek Accuracy: "
        f"{rf_smote_accuracy:.2%}"
    )

    # =====================================================
    # RANDOM FOREST - GRIDSEARCH AFTER SMOTETOMEK
    # This is the tuned RF used by the application.
    # =====================================================

    rf_tuned = RandomForestClassifier(
        random_state=42,
        n_jobs=2
    )

    rf_tuned_grid = {
        "n_estimators": [400],
        "max_depth": [None],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "max_features": ["sqrt"]
    }

    rf_tuned_search = GridSearchCV(
        rf_tuned,
        rf_tuned_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=2,
        verbose=1
    )

    print("Running tuned Random Forest GridSearchCV...")

    rf_tuned_search.fit(
        X_train_balanced,
        y_train_balanced
    )

    rf_tuned_model = (
        rf_tuned_search.best_estimator_
    )

    y_pred_rf_tuned = (
        rf_tuned_model.predict(
            X_test_encoded
        )
    )

    rf_tuned_accuracy = accuracy_score(
        y_test_encoded,
        y_pred_rf_tuned
    )

    print(
        f"Tuned Random Forest Test Accuracy: "
        f"{rf_tuned_accuracy:.2%}"
    )

    # =====================================================
    # XGBOOST - GRIDSEARCH
    # =====================================================

    xgb = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=2
    )

    xgb_param_grid = {
        "n_estimators": [200],
        "max_depth": [3],
        "learning_rate": [0.05],
        "subsample": [0.8]
    }

    xgb_grid = GridSearchCV(
        xgb,
        xgb_param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=2,
        verbose=1
    )

    print("Running XGBoost GridSearchCV...")

    xgb_grid.fit(
        X_train_balanced,
        y_train_balanced
    )

    xgb_model = xgb_grid.best_estimator_

    y_pred_xgb = xgb_model.predict(
        X_test_encoded
    )

    xgb_accuracy = accuracy_score(
        y_test_encoded,
        y_pred_xgb
    )

    print(
        f"XGBoost Test Accuracy: "
        f"{xgb_accuracy:.2%}"
    )

    # =====================================================
    # KNN - GRIDSEARCH
    # =====================================================

    # knn_pipeline = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("knn", KNeighborsClassifier(n_jobs=-1))
    # ])

    # knn_param_grid = {
    #     "knn__n_neighbors": [7],
    #     "knn__weights": ["distance"],
    #     "knn__p": [1]
    # }

    # knn_grid = GridSearchCV(
    #     knn_pipeline,
    #     knn_param_grid,
    #     scoring="accuracy",
    #     cv=3,
    #     n_jobs=1,
    #     verbose=1
    # )

    # print("Running KNN GridSearchCV...")

    # knn_grid.fit(
    #     X_train_balanced,
    #     y_train_balanced
    # )

    # knn_model = knn_grid.best_estimator_

    # y_pred_knn = knn_model.predict(
    #     X_test_encoded
    # )

    # knn_accuracy = accuracy_score(
    #     y_test_encoded,
    #     y_pred_knn
    # )

    # print(
    #     f"KNN Test Accuracy: "
    #     f"{knn_accuracy:.2%}"
    # )

    # =====================================================
    # LOGISTIC REGRESSION - GRIDSEARCH
    # =====================================================

    logreg_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "logreg",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ])

    logreg_param_grid = {
        "logreg__C": [0.1, 1],
        "logreg__solver": ["lbfgs"]
    }

    logreg_grid = GridSearchCV(
        logreg_pipeline,
        logreg_param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=2,
        verbose=1
    )

    print("Running Logistic Regression GridSearchCV...")

    logreg_grid.fit(
        X_train_balanced,
        y_train_balanced
    )

    logreg_model = (
        logreg_grid.best_estimator_
    )

    y_pred_logreg = logreg_model.predict(
        X_test_encoded
    )

    logreg_accuracy = accuracy_score(
        y_test_encoded,
        y_pred_logreg
    )

    print(
        f"Logistic Regression Test Accuracy: "
        f"{logreg_accuracy:.2%}"
    )

    # =====================================================
    # MODEL COMPARISON - SAME AS NOTEBOOK
    # =====================================================

    model_results = pd.DataFrame({
        "Model": [
            "Random Forest",
            "XGBoost",
            # "KNN",
            "Logistic Regression"
        ],
        "Test Accuracy": [
            rf_tuned_accuracy,
            xgb_accuracy,
            # knn_accuracy,
            logreg_accuracy
        ]
    })

    model_results = (
        model_results
        .sort_values(
            by="Test Accuracy",
            ascending=False
        )
        .reset_index(drop=True)
    )

    model_results["Test Accuracy"] = (
        model_results["Test Accuracy"]
        .round(4)
    )

    best_model_name = model_results.iloc[0]["Model"]

    model_lookup = {
        "Random Forest": rf_tuned_model,
        "XGBoost": xgb_model,
        # "KNN": knn_model,
        "Logistic Regression": logreg_model
    }

    model = model_lookup[best_model_name]

    feature_columns = (
        X_train_encoded.columns
        .tolist()
    )

    print("\n========================================")
    print("MODEL COMPARISON")
    print("========================================")
    print(model_results.to_string(index=False))
    print(f"\nBest Model: {best_model_name}")
    print(
        f"Best Test Accuracy: "
        f"{model_results.iloc[0]['Test Accuracy']:.2%}"
    )
    print("========================================\n")


# =========================================================
# INPUT MODEL
# =========================================================

class PatientData(BaseModel):

    age: str

    gender: str

    race: str

    time_in_hospital: int

    num_lab_procedures: int

    num_procedures: int

    num_medications: int

    number_outpatient: int

    number_emergency: int

    number_inpatient: int

    number_diagnoses: int

    max_glu_serum: str

    A1Cresult: str

    change: str

    diabetesMed: str
    insulin: str




# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return FileResponse(
        "index.html"
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {

        "status": "online",

        "model": best_model_name if best_model_name else "Training"

    ,

        "models": (
            model_results.to_dict(orient="records")
            if model_results is not None else []
        )

    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
def predict(data: PatientData):

    # -----------------------------------------
    # NOTE
    # -----------------------------------------
    #
    # Your notebook model uses 256 features.
    #
    # The UI currently sends the main patient
    # information only.
    #
    # Therefore, for the remaining features,
    # we create a representative default row.
    #
    # -----------------------------------------


    # Get one row from the original dataset

    df = pd.read_csv(
        "diabetic_data.csv"
    )


    # Use median/mode-like existing row
    # as a base for features not exposed
    # in the UI.

    patient = df.iloc[
        [0]
    ].copy()


    # =====================================================
    # OVERRIDE VALUES FROM UI
    # =====================================================

    patient["age"] = data.age

    patient["gender"] = data.gender

    patient["race"] = data.race

    patient[
        "time_in_hospital"
    ] = data.time_in_hospital

    patient[
        "num_lab_procedures"
    ] = data.num_lab_procedures

    patient[
        "num_procedures"
    ] = data.num_procedures

    patient[
        "num_medications"
    ] = data.num_medications

    patient[
        "number_outpatient"
    ] = data.number_outpatient

    patient[
        "number_emergency"
    ] = data.number_emergency

    patient[
        "number_inpatient"
    ] = data.number_inpatient

    patient[
        "number_diagnoses"
    ] = data.number_diagnoses

    patient[
        "max_glu_serum"
    ] = data.max_glu_serum

    patient[
        "A1Cresult"
    ] = data.A1Cresult

    patient[
        "change"
    ] = data.change

    patient[
        "diabetesMed"
    ] = data.diabetesMed


    # =====================================================
    # PREPROCESS SINGLE PATIENT
    # =====================================================

    patient = patient.replace(
        "?",
        np.nan
    )


    patient["gender"] = (
        patient["gender"]
        .replace(
            "Unknown/Invalid",
            np.nan
        )
    )


    patient = patient.drop(
        columns=["weight"]
    )


    # Missing values

    for col in [
        "medical_specialty",
        "payer_code",
        "race"
    ]:

        patient[col] = (
            patient[col]
            .fillna("unkown")
        )


    patient[
        "max_glu_serum"
    ] = patient[
        "max_glu_serum"
    ].fillna("not taken")


    patient[
        "A1Cresult"
    ] = patient[
        "A1Cresult"
    ].fillna("not taken")


    # IDs

    for col in [
        "admission_type_id",
        "admission_source_id",
        "discharge_disposition_id"
    ]:

        patient[col] = (
            patient[col]
            .fillna(1)
        )


    patient["gender"] = (
        patient["gender"]
        .fillna("Female")
    )


    for col in [
        "diag_1",
        "diag_2",
        "diag_3"
    ]:

        patient[col] = (
            patient[col]
            .fillna("Unknown")
        )


    # Diagnosis

    def categorize_diagnosis(code):

        code = str(code)

        if code.startswith("V"):
            return "Supplemental"

        if code.startswith("E"):
            return "External"

        try:

            code_num = float(code)

        except:

            if code == "Unknown":
                return "Unknown"

            return "Other"


        if 250 <= code_num < 251:
            return "Diabetes"

        elif 390 <= code_num <= 459:
            return "Circulatory"

        elif 460 <= code_num <= 519:
            return "Respiratory"

        elif 520 <= code_num <= 579:
            return "Digestive"

        elif 800 <= code_num <= 999:
            return "Injury"

        elif 710 <= code_num <= 739:
            return "Musculoskeletal"

        elif 580 <= code_num <= 629:
            return "Genitourinary"

        elif 140 <= code_num <= 239:
            return "Neoplasms"

        else:
            return "Others"


    for col in [
        "diag_1",
        "diag_2",
        "diag_3"
    ]:

        patient[
            f"{col}_category"
        ] = patient[
            col
        ].apply(
            categorize_diagnosis
        )


    # Log

    for col in [

        "number_emergency",
        "number_outpatient",
        "number_inpatient",
        "num_medications",
        "num_procedures",
        "time_in_hospital"

    ]:

        patient[col] = np.log1p(
            patient[col]
        )


    # Scale numerical

    numerical_columns = [

        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses"

    ]


    patient[
        numerical_columns
    ] = numerical_scaler.transform(
        patient[
            numerical_columns
        ]
    )


    # Binary

    patient["gender"] = (
        patient["gender"]
        .map({
            "Female": 0,
            "Male": 1
        })
    )


    patient["change"] = (
        patient["change"]
        .map({
            "No": 0,
            "Ch": 1
        })
    )


    patient["diabetesMed"] = (
        patient["diabetesMed"]
        .map({
            "No": 0,
            "Yes": 1
        })
    )


    # Age

    patient["age"] = (
        patient["age"]
        .map({
            "[0-10)": 0,
            "[10-20)": 1,
            "[20-30)": 2,
            "[30-40)": 3,
            "[40-50)": 4,
            "[50-60)": 5,
            "[60-70)": 6,
            "[70-80)": 7,
            "[80-90)": 8,
            "[90-100)": 9
        })
    )


    patient[
        ["age"]
    ] = age_scaler.transform(
        patient[
            ["age"]
        ]
    )


    # Glucose

    patient[
        "max_glu_serum"
    ] = patient[
        "max_glu_serum"
    ].map({
        "not taken": 0,
        "Norm": 1,
        ">200": 2,
        ">300": 3
    })


    patient[
        ["max_glu_serum"]
    ] = glu_scaler.transform(
        patient[
            ["max_glu_serum"]
        ]
    )


    # A1C

    patient[
        "A1Cresult"
    ] = patient[
        "A1Cresult"
    ].map({
        "not taken": 0,
        "Norm": 1,
        ">7": 2,
        ">8": 3
    })


    patient[
        ["A1Cresult"]
    ] = a1c_scaler.transform(
        patient[
            ["A1Cresult"]
        ]
    )


    # =====================================================
    # ONE HOT
    # =====================================================

    diagnosis_columns = [

        "diag_1_category",
        "diag_2_category",
        "diag_3_category"

    ]


    patient_diag = pd.DataFrame(

        diagnosis_encoder.transform(
            patient[
                diagnosis_columns
            ]
        ),

        columns=
        diagnosis_encoder
        .get_feature_names_out(
            diagnosis_columns
        ),

        index=patient.index

    )


    nominal_columns = [

        "race",
        "payer_code",
        "medical_specialty"

    ]


    patient_cat = pd.DataFrame(

        categorical_encoder.transform(
            patient[
                nominal_columns
            ]
        ),

        columns=
        categorical_encoder
        .get_feature_names_out(
            nominal_columns
        ),

        index=patient.index

    )


    id_columns = [

        "admission_type_id",
        "admission_source_id",
        "discharge_disposition_id"

    ]


    patient_id = pd.DataFrame(

        id_encoder.transform(
            patient[
                id_columns
            ]
        ),

        columns=
        id_encoder
        .get_feature_names_out(
            id_columns
        ),

        index=patient.index

    )


    medication_columns = [

        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
        "acetohexamide",
        "glimepiride-pioglitazone"

    ]


    patient_med = pd.DataFrame(

        medication_encoder.transform(
            patient[
                medication_columns
            ]
        ),

        columns=
        medication_encoder
        .get_feature_names_out(
            medication_columns
        ),

        index=patient.index

    )


    # =====================================================
    # DROP ORIGINAL
    # =====================================================

    columns_to_drop = (

        diagnosis_columns
        + nominal_columns
        + id_columns
        + medication_columns
        + [
            "diag_1",
            "diag_2",
            "diag_3"
        ]

    )


    patient_base = patient.drop(
        columns=columns_to_drop
    )


    # =====================================================
    # COMBINE
    # =====================================================

    patient_encoded = pd.concat(

        [

            patient_base,
            patient_diag,
            patient_cat,
            patient_id,
            patient_med

        ],

        axis=1

    )


    # Ensure exact feature order

    patient_encoded = (
        patient_encoded
        .reindex(
            columns=feature_columns,
            fill_value=0
        )
    )
    # =====================================================
    # REMOVE ANY REMAINING NaN VALUES
    # =====================================================

    patient_encoded = patient_encoded.fillna(0)


    # =====================================================
    # PREDICTION FOR ALL TUNED MODELS
    # =====================================================

    models = {}

    if rf_tuned_model is not None:
        models["Random Forest"] = rf_tuned_model

    if xgb_model is not None:
        models["XGBoost"] = xgb_model

    if logreg_model is not None:
        models["Logistic Regression"] = logreg_model

    all_predictions = {}

    accuracy_lookup = {
        row["Model"]: float(row["Test Accuracy"])
        for _, row in model_results.iterrows()
    }

    for model_name, current_model in models.items():

        prediction = int(
            current_model.predict(
                patient_encoded
            )[0]
        )

        probabilities = (
            current_model
            .predict_proba(
                patient_encoded
            )[0]
        )

        class_probabilities = {
            int(cls): float(prob)
            for cls, prob in zip(
                current_model.classes_,
                probabilities
            )
        }

        readmitted_percentage = round(
            class_probabilities.get(1, 0.0) * 100,
            2
        )

        not_readmitted_percentage = round(
            class_probabilities.get(0, 0.0) * 100,
            2
        )

        prediction_text = (
            "Readmitted"
            if prediction == 1
            else "Not Readmitted"
        )

        if readmitted_percentage >= 70:
            risk = "High Risk"
        elif readmitted_percentage >= 40:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"

        all_predictions[model_name] = {
            "prediction": prediction_text,
            "readmitted_percentage": readmitted_percentage,
            "not_readmitted_percentage": not_readmitted_percentage,
            "risk": risk,
            "accuracy": round(
                accuracy_lookup[model_name] * 100,
                2
            )
        }

    best_prediction = all_predictions[best_model_name]

    return {
        "best_model": best_model_name,
        "best_accuracy": best_prediction["accuracy"],
        "prediction": best_prediction["prediction"],
        "probability": best_prediction["readmitted_percentage"],
        "risk": best_prediction["risk"],
        "models": all_predictions
    }


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup_event():

    train_model()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)