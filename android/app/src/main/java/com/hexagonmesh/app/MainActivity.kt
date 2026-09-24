package com.hexagonmesh.app

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.system.Os
import android.view.WindowInsets
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

class MainActivity : Activity() {

    private lateinit var status: TextView
    private lateinit var npuLog: TextView
    private lateinit var npuButton: Button
    private lateinit var embedButton: Button
    private val handler = Handler(Looper.getMainLooper())
    private val tick = object : Runnable {
        override fun run() {
            status.text = readStatus()
            handler.postDelayed(this, 2000)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Let the Hexagon DSP loader find the QNN libraries shipped inside this app.
        val nativeDir = applicationInfo.nativeLibraryDir
        Os.setenv("ADSP_LIBRARY_PATH",
            "$nativeDir;/odm/lib/rfsa/adsp;/vendor/lib/rfsa/adsp/;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp",
            true)

        val pad = (16 * resources.displayMetrics.density).toInt()
        val title = TextView(this).apply { text = "Hexagon Mesh"; textSize = 28f }
        val subtitle = TextView(this).apply { text = "Node status, updates every 2 seconds"; textSize = 14f }
        status = TextView(this).apply { textSize = 17f; setPadding(0, pad, 0, pad) }
        npuButton = Button(this).apply {
            text = "Run NPU test"
            setOnClickListener { startNpuTest() }
        }
        embedButton = Button(this).apply {
            text = "Run embedding benchmark (real AI)"
            setOnClickListener { startEmbedBench() }
        }
        npuLog = TextView(this).apply { textSize = 15f; setPadding(0, pad, 0, pad) }
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
            addView(title); addView(subtitle); addView(status); addView(npuButton); addView(embedButton); addView(npuLog)
        }
        val scroll = ScrollView(this).apply { addView(column) }
        // Keep content clear of the status bar and navigation bar (edge-to-edge on Android 15+).
        scroll.setOnApplyWindowInsetsListener { _, insets ->
            val top: Int
            val bottom: Int
            if (Build.VERSION.SDK_INT >= 30) {
                val bars = insets.getInsets(WindowInsets.Type.systemBars())
                top = bars.top; bottom = bars.bottom
            } else {
                top = insets.systemWindowInsetTop
                bottom = insets.systemWindowInsetBottom
            }
            column.setPadding(pad, top + pad, pad, bottom + pad)
            insets
        }
        setContentView(scroll)
    }

    private fun startNpuTest() {
        runInBackground("Running NPU test (about 10-30 seconds)...") { log -> NpuTest.run(this, log) }
    }

    private fun startEmbedBench() {
        runInBackground("Running embedding benchmark (about 4 minutes, keep the app open, unplugged for power numbers)...") { log ->
            EmbedBench.run(this, log)
        }
    }

    private fun runInBackground(intro: String, task: ((String) -> Unit) -> Unit) {
        npuButton.isEnabled = false
        embedButton.isEnabled = false
        npuLog.text = intro + "\n"
        Thread {
            try {
                task { line -> runOnUiThread { npuLog.append(line + "\n") } }
            } catch (e: Throwable) {
                runOnUiThread { npuLog.append("Error: ${e.message}\n") }
            }
            runOnUiThread {
                npuButton.isEnabled = true
                embedButton.isEnabled = true
                npuLog.append("Done.\n")
            }
        }.start()
    }

    override fun onResume() {
        super.onResume()
        handler.post(tick)
    }

    override fun onPause() {
        handler.removeCallbacks(tick)
        super.onPause()
    }

    private fun readStatus(): String {
        val b = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = b?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = b?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val pct = if (level >= 0 && scale > 0) level * 100 / scale else -1
        val pluggedCode = b?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0
        val plugged = when (pluggedCode) {
            0 -> "not charging"
            BatteryManager.BATTERY_PLUGGED_AC -> "AC charger"
            BatteryManager.BATTERY_PLUGGED_USB -> "USB"
            BatteryManager.BATTERY_PLUGGED_WIRELESS -> "wireless"
            else -> "charging"
        }
        val tempC = (b?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10.0

        val bm = getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val currentMa = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) / 1000

        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        val thermal = if (Build.VERSION.SDK_INT >= 29) pm.currentThermalStatus else -1
        val headroom = if (Build.VERSION.SDK_INT >= 30) pm.getThermalHeadroom(10) else Float.NaN

        val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val caps = cm.getNetworkCapabilities(cm.activeNetwork)
        val wifi = caps?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true
        val unmetered = caps?.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED) == true

        val chip = if (Build.VERSION.SDK_INT >= 31) {
            Build.SOC_MANUFACTURER + " " + Build.SOC_MODEL
        } else {
            Build.HARDWARE
        }
        val ready = pluggedCode != 0 && pct >= 80 && tempC < 38.0 && wifi

        return buildString {
            appendLine("Phone: ${Build.MANUFACTURER} ${Build.MODEL}")
            appendLine("Chip: $chip")
            appendLine("Android: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
            appendLine()
            appendLine("Battery: $pct%  ($plugged)")
            appendLine("Temperature: ${"%.1f".format(tempC)} C")
            appendLine("Current: $currentMa mA")
            appendLine("Thermal status: ${thermalName(thermal)}")
            appendLine("Thermal headroom: ${if (headroom.isNaN()) "n/a" else "%.2f".format(headroom)}")
            appendLine("Wi-Fi: ${if (wifi) "connected" else "no"}${if (unmetered) " (unmetered)" else ""}")
            appendLine()
            append(if (ready) "READY for work" else "WAIT (needs charger, 80%+ battery, under 38C, Wi-Fi)")
        }
    }

    private fun thermalName(code: Int): String = when (code) {
        0 -> "none"
        1 -> "light"
        2 -> "moderate"
        3 -> "severe"
        4 -> "critical"
        5 -> "emergency"
        6 -> "shutdown"
        else -> "n/a"
    }
}
