plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.hexagonmesh.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.hexagonmesh.app"
        minSdk = 28
        targetSdk = 35
        versionCode = 9
        versionName = "0.9.0"
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    buildTypes {
        release { isMinifyEnabled = false }
    }
    packaging {
        // Unpack native libraries to real files: Qualcomm's DSP loader cannot read inside the APK.
        jniLibs { useLegacyPackaging = true }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    // ONNX Runtime with Qualcomm's QNN execution provider (Hexagon NPU).
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.29.0")
}
