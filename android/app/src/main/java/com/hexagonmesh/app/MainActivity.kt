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
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/** M0: shows everything the node needs to know about this phone, refreshed every 2 seconds. */
class MainActivity : Activity() {

    private lateinit var status: TextView
    private val handler = Handler(Looper.getMainLooper())
    private val tick = object : Runnable {
        override fun run() {
            status.text = readStatus()
            handler.postDelayed(this, 2000)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (16 * resources.displayMetrics.density).toInt()
        val title = TextView(this).apply { text = "Hexagon Mesh"; textSize = 28f }
        val subtitle = TextView(this).apply { text = "Node status, updates every 2 seconds"; textSize = 14f }
        status = TextView(this).apply { textSize = 18f; setPadding(0, pad, 0, 0) }
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad * 3, pad, pad)
            addView(title)
            addView(subtitle)
            addView(status)
        }
        setContentView(ScrollView(this).apply { addView(column) })
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
