package com.hexagonmesh.app

import android.app.ActivityManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.SharedPreferences
import android.content.pm.ServiceInfo
import android.graphics.drawable.Icon
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import org.json.JSONArray
import java.util.UUID

/**
 * M3: the phone as a Hexagon Mesh node. While charging (80%+, under 38 C, on Wi-Fi) it checks in with
 * the coordinator, pulls tickets, embeds the texts on the NPU, submits the vectors, and repeats.
 */
class NodeService : Service() {
    companion object {
        const val ACTION_STOP = "com.hexagonmesh.app.STOP"
        const val DEFAULT_URL = "http://127.0.0.1:8080"
        private const val CHANNEL = "node"
        private const val NOTIF_ID = 1

        fun prefs(ctx: Context): SharedPreferences = ctx.getSharedPreferences("node", Context.MODE_PRIVATE)

        fun nodeId(ctx: Context): String {
            val p = prefs(ctx)
            val existing = p.getString("node_id", null)
            if (existing != null) return existing
            val id = UUID.randomUUID().toString().replace("-", "")
            p.edit().putString("node_id", id).apply()
            return id
        }
    }

    @Volatile private var stopping = false
    @Volatile private var myGeneration = -1
    private var worker: Thread? = null
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopping = true
            worker?.interrupt()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }
        startForegroundCompat("Starting...")
        if (worker == null) {
            stopping = false
            worker = Thread { loop() }.also { it.start() }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        stopping = true
        worker?.interrupt()
        releaseWake()
        NodeState.running = false
        NodeState.status = "stopped"
        super.onDestroy()
    }

    /** True while this worker is the newest one; an older worker (e.g. from a quick stop/start) stays silent. */
    private fun isCurrent() = myGeneration == NodeState.generation

    private fun loop() {
        myGeneration = synchronized(NodeState) { ++NodeState.generation }
        NodeState.running = true
        NodeState.backend = ""
        val p = prefs(this)
        val coord = Coordinator(p.getString("url", DEFAULT_URL) ?: DEFAULT_URL, p.getString("key", "") ?: "")
        val wallet = p.getString("wallet", "") ?: ""
        val ignoreRules = p.getBoolean("ignore_rules", false)
        val id = nodeId(this)
        var embedder: NpuEmbedder? = null
        var idle = 0
        var lastCheckin = 0L
        try {
            status("Loading AI model...")
            val emb = NpuEmbedder(this)
            embedder = emb
            if (isCurrent()) NodeState.backend = emb.backend
            while (!stopping && isCurrent()) {
                val wait = if (ignoreRules) null else waitReason()
                if (wait != null) {
                    releaseWake()
                    status("WAIT: $wait")
                    pause(60_000)
                    continue
                }
                acquireWake()
                try {
                    if (System.currentTimeMillis() - lastCheckin > 600_000) {
                        val r = coord.checkin(id, wallet, ramGb(), listOf(NpuEmbedder.MODEL_NAME))
                        if (r.code != 200) throw RuntimeException("check-in refused (HTTP ${r.code})")
                        lastCheckin = System.currentTimeMillis()
                    }
                    val r = coord.next(id)
                    if (r.code == 403) {
                        status("Suspended by the coordinator")
                        break
                    }
                    val ticket = if (r.code == 200) r.body.optJSONObject("ticket") else null
                    if (ticket == null) {
                        val bal = coord.balance(id)
                        if (bal.code == 200) NodeState.credits = bal.body.optDouble("credits", NodeState.credits)
                        idle++
                        val ms = minOf(5_000L shl minOf(idle - 1, 5), 120_000L)
                        status("READY on ${emb.backend}. No work right now, next check in ${ms / 1000}s")
                        pause(ms)
                        continue
                    }
                    idle = 0
                    val arr = ticket.getJSONObject("payload").getJSONArray("texts")
                    val texts = List(arr.length()) { arr.getString(it) }
                    status("Working: ${texts.size} texts on ${emb.backend}")
                    val t0 = System.nanoTime()
                    val vectors = emb.embed(texts)
                    val ms = (System.nanoTime() - t0) / 1_000_000
                    val result = JSONArray()
                    for (v in vectors) {
                        val row = JSONArray()
                        for (x in v) row.put(x.toDouble())
                        result.put(row)
                    }
                    val s = coord.submit(id, ticket.getLong("assignment_id"), result)
                    NodeState.tickets++
                    NodeState.texts += texts.size
                    NodeState.lastTicketMs = ms
                    val bal = coord.balance(id)
                    if (bal.code == 200) NodeState.credits = bal.body.optDouble("credits", NodeState.credits)
                    NodeState.lastError = ""
                    status("Ticket done in $ms ms (${s.body.optString("status", "sent")})")
                } catch (e: Exception) {
                    if (stopping) break
                    NodeState.lastError = e.message ?: e.javaClass.simpleName
                    status("Can't reach the coordinator, retrying in 30s")
                    pause(30_000)
                }
            }
        } catch (e: Exception) {
            if (isCurrent()) {
                NodeState.lastError = e.message ?: e.javaClass.simpleName
                status("Stopped: ${NodeState.lastError.take(80)}")
            }
        } finally {
            embedder?.close()
            releaseWake()
            if (isCurrent()) NodeState.running = false
        }
    }

    private fun waitReason(): String? {
        val b = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) ?: return "no battery info"
        val level = b.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = b.getIntExtra(BatteryManager.EXTRA_SCALE, 100)
        val pct = if (level >= 0 && scale > 0) level * 100 / scale else 0
        val plugged = b.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0
        val temp = b.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) / 10.0
        val cm = getSystemService(ConnectivityManager::class.java)
        val caps = cm.getNetworkCapabilities(cm.activeNetwork)
        val wifi = caps?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true
        return when {
            !plugged -> "not charging"
            pct < 80 -> "battery $pct%, waiting for 80%"
            temp >= 38.0 -> "cooling down (${"%.1f".format(temp)} C)"
            !wifi -> "not on Wi-Fi"
            else -> null
        }
    }

    private fun status(text: String) {
        if (!isCurrent()) return
        NodeState.status = text
        getSystemService(NotificationManager::class.java).notify(NOTIF_ID, notification(text))
    }

    private fun pause(ms: Long) {
        try {
            Thread.sleep(ms)
        } catch (e: InterruptedException) {
            stopping = true
        }
    }

    private fun acquireWake() {
        if (wakeLock?.isHeld == true) return
        val pm = getSystemService(PowerManager::class.java)
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "hexagonmesh:node").apply {
            setReferenceCounted(false)
            acquire(30 * 60 * 1000L)
        }
    }

    private fun releaseWake() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    private fun ramGb(): Double {
        val am = getSystemService(ActivityManager::class.java)
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        return Math.round(mi.totalMem / 1e8) / 10.0
    }

    private fun notification(text: String): Notification {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Hexagon Mesh node", NotificationManager.IMPORTANCE_LOW))
        }
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val stop = PendingIntent.getService(this, 1,
            Intent(this, NodeService::class.java).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE)
        val noIcon: Icon? = null
        return Notification.Builder(this, CHANNEL)
            .setContentTitle("Hexagon Mesh node")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentIntent(open)
            .setOngoing(true)
            .addAction(Notification.Action.Builder(noIcon, "Stop", stop).build())
            .build()
    }

    private fun startForegroundCompat(text: String) {
        val n = notification(text)
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTIF_ID, n)
        }
    }
}
