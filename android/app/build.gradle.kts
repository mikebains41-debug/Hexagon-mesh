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
        versionCode = 15
        versionName = "0.15.0"
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    // Permanent signing key, supplied by the build server from encrypted GitHub secrets.
    // Same key on every build = updates install over the old app, settings kept.
    val keystoreFile = System.getenv("HM_KEYSTORE_FILE")
    signingConfigs {
        create("hexagon") {
            if (keystoreFile != null) {
                storeFile = file(keystoreFile)
                storeType = "pkcs12"
                storePassword = System.getenv("HM_KEYSTORE_PASSWORD")
                keyAlias = System.getenv("HM_KEY_ALIAS")
                keyPassword = System.getenv("HM_KEYSTORE_PASSWORD")
            }
        }
    }

    buildTypes {
        debug {
            if (keystoreFile != null) signingConfig = signingConfigs.getByName("hexagon")
        }
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
