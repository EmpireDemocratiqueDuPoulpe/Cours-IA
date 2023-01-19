from timeit import default_timer as timer
from datetime import timedelta
import colorama
from colorama import Style, Fore
import pandas
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import mlsp


def main():
    start_time = timer()

    # Read CSV
    data = pandas.read_csv("./data/arrhythmia.csv")

    # First look
    mlsp.misc.print_title("First look")
    mlsp.df.first_look(data)

    # Missing values
    mlsp.misc.print_title("Missing values")
    mlsp.df.missing_values(data, keep_zeros=False)

    # Split data to train/test
    mlsp.misc.print_title("Splitting dataset")
    x_train, x_test = mlsp.df.split_train_test(data, test_size=0.33)
    print(f"Train data: {Fore.LIGHTGREEN_EX}{x_train.shape}")
    print(f"Test data: {Fore.LIGHTGREEN_EX}{x_test.shape}")

    # Transform numeric and categorical values
    mlsp.misc.print_title("Transform numeric and categorical values")
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant")),
        ("encoder", OrdinalEncoder())
    ])

    numeric_features = ["age", "height", "weight", "QRSduration", "PRinterval", "Q-Tinterval", "Tinterval", "Pinterval",
                        "QRS", "T", "P", "QRST", "J", "heartrate", "chDI_Qwave", "chDI_Rwave", "chDI_Swave",
                        "chDI_RPwave", "chDI_SPwave", "chDI_intrinsicReflecttions", "chDII_Qwave", "chDII_Rwave",
                        "chDII_Swave", "chDII_RPwave", "chDII_SPwave", "chDII_intrinsicReflecttions", "chDIII_Qwave",
                        "chDIII_Rwave", "chDIII_Swave", "chDIII_RPwave", "chDIII_SPwave",
                        "chDIII_intrinsicReflecttions", "chAVR_Qwave", "chAVR_Rwave", "chAVR_Swave", "chAVR_RPwave",
                        "chAVR_SPwave", "chAVR_intrinsicReflecttions", "chAVL_Qwave", "chAVL_Rwave", "chAVL_Swave",
                        "chAVL_RPwave", "chAVL_SPwave", "chAVL_intrinsicReflecttions", "chAVF_Qwave", "chAVF_Rwave",
                        "chAVF_Swave", "chAVF_RPwave", "chAVF_SPwave", "chAVF_intrinsicReflecttions", "chV1_Qwave",
                        "chV1_Rwave", "chV1_Swave", "chV1_RPwave", "chV1_SPwave", "chV1_intrinsicReflecttions",
                        "chV2_Qwave", "chV2_Rwave", "chV2_Swave", "chV2_RPwave", "chV2_SPwave",
                        "chV2_intrinsicReflecttions", "chV3_Qwave", "chV3_Rwave", "chV3_Swave", "chV3_RPwave",
                        "chV3_SPwave", "chV3_intrinsicReflecttions", "chV4_Qwave", "chV4_Rwave", "chV4_Swave",
                        "chV4_RPwave", "chV4_SPwave", "chV4_intrinsicReflecttions", "chV5_Qwave", "chV5_Rwave",
                        "chV5_Swave", "chV5_RPwave", "chV5_SPwave", "chV5_intrinsicReflecttions", "chV6_Qwave",
                        "chV6_Rwave", "chV6_Swave", "chV6_RPwave", "chV6_SPwave", "chV6_intrinsicReflecttions",
                        "chDI_JJwaveAmp", "chDI_QwaveAmp", "chDI_RwaveAmp", "chDI_SwaveAmp", "chDI_RPwaveAmp",
                        "chDI_SPwaveAmp", "chDI_PwaveAmp", "chDI_TwaveAmp", "chDI_QRSA", "chDI_QRSTA",
                        "chDII_JJwaveAmp", "chDII_QwaveAmp", "chDII_RwaveAmp", "chDII_SwaveAmp", "chDII_RPwaveAmp",
                        "chDII_SPwaveAmp", "chDII_PwaveAmp", "chDII_TwaveAmp", "chDII_QRSA", "chDII_QRSTA",
                        "chDIII_JJwaveAmp", "chDIII_QwaveAmp", "chDIII_RwaveAmp", "chDIII_SwaveAmp", "chDIII_RPwaveAmp",
                        "chDIII_SPwaveAmp", "chDIII_PwaveAmp", "chDIII_TwaveAmp", "chDIII_QRSA", "chDIII_QRSTA",
                        "chAVR_JJwaveAmp", "chAVR_QwaveAmp", "chAVR_RwaveAmp", "chAVR_SwaveAmp", "chAVR_RPwaveAmp",
                        "chAVR_SPwaveAmp", "chAVR_PwaveAmp", "chAVR_TwaveAmp", "chAVR_QRSA", "chAVR_QRSTA",
                        "chAVL_JJwaveAmp", "chAVL_QwaveAmp", "chAVL_RwaveAmp", "chAVL_SwaveAmp", "chAVL_RPwaveAmp",
                        "chAVL_SPwaveAmp", "chAVL_PwaveAmp", "chAVL_TwaveAmp", "chAVL_QRSA", "chAVL_QRSTA",
                        "chAVF_JJwaveAmp", "chAVF_QwaveAmp", "chAVF_RwaveAmp", "chAVF_SwaveAmp", "chAVF_RPwaveAmp",
                        "chAVF_SPwaveAmp", "chAVF_PwaveAmp", "chAVF_TwaveAmp", "chAVF_QRSA", "chAVF_QRSTA",
                        "chV1_JJwaveAmp", "chV1_QwaveAmp", "chV1_RwaveAmp", "chV1_SwaveAmp", "chV1_RPwaveAmp",
                        "chV1_SPwaveAmp", "chV1_PwaveAmp", "chV1_TwaveAmp", "chV1_QRSA", "chV1_QRSTA", "chV2_JJwaveAmp",
                        "chV2_QwaveAmp", "chV2_RwaveAmp", "chV2_SwaveAmp", "chV2_RPwaveAmp", "chV2_SPwaveAmp",
                        "chV2_PwaveAmp", "chV2_TwaveAmp", "chV2_QRSA", "chV2_QRSTA", "chV3_JJwaveAmp", "chV3_QwaveAmp",
                        "chV3_RwaveAmp", "chV3_SwaveAmp", "chV3_RPwaveAmp", "chV3_SPwaveAmp", "chV3_PwaveAmp",
                        "chV3_TwaveAmp", "chV3_QRSA", "chV3_QRSTA", "chV4_JJwaveAmp", "chV4_QwaveAmp", "chV4_RwaveAmp",
                        "chV4_SwaveAmp", "chV4_RPwaveAmp", "chV4_SPwaveAmp", "chV4_PwaveAmp", "chV4_TwaveAmp",
                        "chV4_QRSA", "chV4_QRSTA", "chV5_JJwaveAmp", "chV5_QwaveAmp", "chV5_RwaveAmp", "chV5_SwaveAmp",
                        "chV5_RPwaveAmp", "chV5_SPwaveAmp", "chV5_PwaveAmp", "chV5_TwaveAmp", "chV5_QRSA", "chV5_QRSTA",
                        "chV6_JJwaveAmp", "chV6_QwaveAmp", "chV6_RwaveAmp", "chV6_SwaveAmp", "chV6_RPwaveAmp",
                        "chV6_SPwaveAmp", "chV6_PwaveAmp", "chV6_TwaveAmp", "chV6_QRSA", "chV6_QRSTA", "class"]
    categorical_features = ["sex", "chDI_RRwaveExists", "chDI_DD_RRwaveExists", "chDI_RPwaveExists",
                            "chDI_DD_RPwaveExists", "chDI_RTwaveExists", "chDI_DD_RTwaveExists", "chDII_RRwaveExists",
                            "chDII_DD_RRwaveExists", "chDII_RPwaveExists", "chDII_DD_RPwaveExists",
                            "chDII_RTwaveExists", "chDII_DD_RTwaveExists", "chDIII_RRwaveExists",
                            "chDIII_DD_RRwaveExists", "chDIII_RPwaveExists", "chDIII_DD_RPwaveExists",
                            "chDIII_RTwaveExists", "chDIII_DD_RTwaveExists", "chAVR_RRwaveExists",
                            "chAVR_DD_RRwaveExists", "chAVR_RPwaveExists", "chAVR_DD_RPwaveExists",
                            "chAVR_RTwaveExists", "chAVR_DD_RTwaveExists", "chAVL_RRwaveExists",
                            "chAVL_DD_RRwaveExists", "chAVL_RPwaveExists", "chAVL_DD_RPwaveExists",
                            "chAVL_RTwaveExists", "chAVL_DD_RTwaveExists", "chAVF_RRwaveExists",
                            "chAVF_DD_RRwaveExists", "chAVF_RPwaveExists", "chAVF_DD_RPwaveExists",
                            "chAVF_RTwaveExists", "chAVF_DD_RTwaveExists", "chV1_RRwaveExists", "chV1_DD_RRwaveExists",
                            "chV1_RPwaveExists", "chV1_DD_RPwaveExists", "chV1_RTwaveExists", "chV1_DD_RTwaveExists",
                            "chV2_RRwaveExists", "chV2_DD_RRwaveExists", "chV2_RPwaveExists", "chV2_DD_RPwaveExists",
                            "chV2_RTwaveExists", "chV2_DD_RTwaveExists", "chV3_RRwaveExists", "chV3_DD_RRwaveExists",
                            "chV3_RPwaveExists", "chV3_DD_RPwaveExists", "chV3_RTwaveExists", "chV3_DD_RTwaveExists",
                            "chV4_RRwaveExists", "chV4_DD_RRwaveExists", "chV4_RPwaveExists", "chV4_DD_RPwaveExists",
                            "chV4_RTwaveExists", "chV4_DD_RTwaveExists", "chV5_RRwaveExists", "chV5_DD_RRwaveExists",
                            "chV5_RPwaveExists", "chV5_DD_RPwaveExists", "chV5_RTwaveExists", "chV5_DD_RTwaveExists",
                            "chV6_RRwaveExists", "chV6_DD_RRwaveExists", "chV6_RPwaveExists", "chV6_DD_RPwaveExists",
                            "chV6_RTwaveExists", "chV6_DD_RTwaveExists"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ]
    )
    preprocessor.fit(x_train)

    exit(0)

    # Model
    mlsp.misc.print_title("Model")
    classifier = KMeans(random_state=42)
    visualizer = KElbowVisualizer(classifier, k=(2, 100))

    visualizer.fit(data_x)
    visualizer.show()

    print(f"{Fore.YELLOW}Using a KMean model with n_clusters={visualizer.elbow_value_}...")
    classifier = KMeans(n_clusters=visualizer.elbow_value_, random_state=42)
    model = LogisticRegression(solver="lbfgs", multi_class="ovr", max_iter=5000, random_state=42)

    processor = Pipeline(steps=[
        ("classifier", classifier),
        ("model", model)
    ])

    model, scores = mlsp.models.common.process_model(
        processor,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        verbose=True
    )

    # Program end
    end_time = timer()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"\n{Fore.GREEN}Successful processing of digits dataset in {elapsed_time}.")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()
