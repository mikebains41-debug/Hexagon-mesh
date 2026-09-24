package com.hexagonmesh.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import java.io.File
import java.nio.LongBuffer
import java.util.Random
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * M2: a real embedding model (all-MiniLM-L6-v2) on realistic documents,
 * CPU vs Hexagon NPU, with accuracy, op placement proof, and a sustained power run.
 */
object EmbedBench {
    private const val BATCH = 16
    private const val SEQ = 256
    private const val MODEL = "minilm_16x256.onnx"
    private const val NOMINAL_V = 3.85
    private const val SUSTAIN_S = 45
    private const val IDLE_S = 10

    private class Timing(val tokPerSec: Double, val msPerBatch: Double, val emb: Array<FloatArray>)

    fun run(ctx: Context, log: (String) -> Unit) {
        val env = OrtEnvironment.getEnvironment()
        val tok = WordPiece(ctx.assets.open("vocab.txt").bufferedReader().readLines())
        log("Preparing model (first run copies about 90 MB, a few seconds)...")
        val model = assetToFile(ctx, MODEL)
        val batches = (0 until 4).map { tok.encodeBatch(Docs.make(BATCH, it.toLong()), SEQ) }
        log("Model: all-MiniLM-L6-v2 (22M parameters), $BATCH documents x $SEQ tokens per batch")
        log("Real tokens per batch: ${batches[0].realTokens}")

        // 1. CPU (ONNX Runtime, same model) for a fair comparison
        val cpu = try {
            OrtSession.SessionOptions().use { o ->
                env.createSession(model, o).use { s -> measure(env, s, batches, 2) }
            }
        } catch (e: Exception) {
            log("CPU run failed: ${NpuTest.reason(e)}"); return
        }
        log("CPU: ${f0(cpu.tokPerSec)} tokens/sec (${f0(cpu.msPerBatch)} ms per batch)")

        // 2. NPU, strict first (proves full offload), then with fallback if needed
        val profile = File(ctx.filesDir, "embed_profile").absolutePath
        var strict = true
        var npuOpts = npuOptions(profile, true)
        var session = try {
            env.createSession(model, npuOpts)
        } catch (e: Exception) {
            npuOpts.close()
            log("Strict NPU mode refused: ${NpuTest.reason(e).take(300)}")
            log("Retrying with CPU fallback allowed, to see how much runs on the NPU...")
            strict = false
            npuOpts = npuOptions(profile, false)
            try {
                env.createSession(model, npuOpts)
            } catch (e2: Exception) {
                npuOpts.close()
                log("NPU run failed: ${NpuTest.reason(e2)}"); return
            }
        }
        try {
            val npu = measure(env, session, batches, 4)
            log("NPU: ${f0(npu.tokPerSec)} tokens/sec (${f0(npu.msPerBatch)} ms per batch)")
            log("Speed: NPU is ${f1(npu.tokPerSec / cpu.tokPerSec)}x the CPU on the same model")
            log(accuracy(cpu.emb, npu.emb))
            log(NpuTest.providerReport(session.endProfiling()) +
                if (strict) "" else "\n(fallback mode: some work may be on the CPU)")
        } catch (e: Exception) {
            log("NPU benchmark failed: ${NpuTest.reason(e)}")
        } finally {
            session.close(); npuOpts.close()
        }

        // 3. Sustained NPU run with power measurement (profiling off)
        val opts = npuOptions(null, strict)
        try {
            env.createSession(model, opts).use { s -> sustained(ctx, env, s, batches, log) }
        } catch (e: Exception) {
            log("Sustained run failed: ${NpuTest.reason(e)}")
        } finally {
            opts.close()
        }
    }

    /** Copies the model out of the APK once, so ONNX Runtime can load it without using app memory. */
    private fun assetToFile(ctx: Context, name: String): String {
        val f = File(ctx.filesDir, "m2_$name")
        if (!f.exists() || f.length() == 0L) {
            val tmp = File(ctx.filesDir, "m2_$name.tmp")
            ctx.assets.open(name).use { input -> tmp.outputStream().use { out -> input.copyTo(out) } }
            tmp.renameTo(f)
        }
        return f.absolutePath
    }

    private fun npuOptions(profilePath: String?, strict: Boolean): OrtSession.SessionOptions {
        val o = OrtSession.SessionOptions()
        if (strict) o.addConfigEntry("session.disable_cpu_ep_fallback", "1")
        if (profilePath != null) o.enableProfiling(profilePath)
        NpuTest.addQnn(o, mapOf(
            "backend_path" to "libQnnHtp.so",
            "htp_performance_mode" to "burst",
            "enable_htp_fp16_precision" to "1",
        ))
        return o
    }

    private fun feeds(env: OrtEnvironment, s: OrtSession, b: WordPiece.Batch): Map<String, OnnxTensor> {
        val shape = longArrayOf(BATCH.toLong(), SEQ.toLong())
        val m = HashMap<String, OnnxTensor>()
        for (name in s.inputNames) {
            val data = when (name) {
                "input_ids" -> b.ids
                "attention_mask" -> b.mask
                else -> LongArray(BATCH * SEQ)
            }
            m[name] = OnnxTensor.createTensor(env, LongBuffer.wrap(data), shape)
        }
        return m
    }

    private fun measure(env: OrtEnvironment, s: OrtSession, batches: List<WordPiece.Batch>, reps: Int): Timing {
        val allFeeds = batches.map { feeds(env, s, it) }
        try {
            s.run(allFeeds[0]).close()                      // warm-up
            var emb: Array<FloatArray> = emptyArray()
            var tokens = 0L
            val t0 = System.nanoTime()
            for (r in 0 until reps) {
                allFeeds.forEachIndexed { i, f ->
                    s.run(f).use { res ->
                        if (r == 0 && i == 0) emb = pooled(res.get(0) as OnnxTensor, batches[0].mask)
                    }
                    tokens += batches[i].realTokens
                }
            }
            val secs = (System.nanoTime() - t0) / 1e9
            return Timing(tokens / secs, secs * 1000 / (reps * batches.size), emb)
        } finally {
            allFeeds.forEach { f -> f.values.forEach { it.close() } }
        }
    }

    private fun sustained(ctx: Context, env: OrtEnvironment, s: OrtSession,
                          batches: List<WordPiece.Batch>, log: (String) -> Unit) {
        val bm = ctx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        if (plugged(ctx)) {
            log("Sustained run skipped: unplug the charger to measure power")
            return
        }
        log("Measuring idle power for ${IDLE_S}s, leave the phone alone...")
        val idle = sample(bm, IDLE_S * 1000L) {}
        val tempStart = tempC(ctx)
        log("Sustained NPU run for ${SUSTAIN_S}s...")
        val allFeeds = batches.map { feeds(env, s, it) }
        var tokens = 0L
        val active = ArrayList<Double>()
        try {
            val t0 = System.nanoTime()
            var nextSample = System.currentTimeMillis()
            var i = 0
            while (System.nanoTime() - t0 < SUSTAIN_S * 1_000_000_000L) {
                s.run(allFeeds[i % allFeeds.size]).close()
                tokens += batches[i % batches.size].realTokens
                i++
                if (System.currentTimeMillis() >= nextSample) {
                    active.add(abs(bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)) / 1000.0)
                    nextSample += 1000
                }
            }
            val secs = (System.nanoTime() - t0) / 1e9
            val tps = tokens / secs
            val idleMa = idle.average()
            val activeMa = active.average()
            val netW = (activeMa - idleMa).coerceAtLeast(0.0) / 1000 * NOMINAL_V
            val allInW = activeMa / 1000 * NOMINAL_V
            log("Sustained: ${f0(tps)} tokens/sec over ${f0(secs)}s, temperature ${f1(tempStart)} -> ${f1(tempC(ctx))} C")
            log("Power: ${f0(activeMa)} mA working vs ${f0(idleMa)} mA idle -> about ${f2(netW)} W for the AI work")
            if (tps > 0) {
                log("Energy: ~${f3(netW / tps * 1000)} mJ per token (AI work only), ~${f3(allInW / tps * 1000)} mJ per token all-in")
                log("If sustained for a 6-hour night: ~${f0(tps * 6 * 3600 / 1e6)} million tokens")
            }
        } finally {
            allFeeds.forEach { f -> f.values.forEach { it.close() } }
        }
    }

    private fun sample(bm: BatteryManager, ms: Long, between: () -> Unit): List<Double> {
        val out = ArrayList<Double>()
        val end = System.currentTimeMillis() + ms
        while (System.currentTimeMillis() < end) {
            out.add(abs(bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)) / 1000.0)
            between()
            Thread.sleep(500)
        }
        return out
    }

    private fun batteryIntent(ctx: Context): Intent? =
        ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))

    private fun plugged(ctx: Context) = (batteryIntent(ctx)?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0

    private fun tempC(ctx: Context) = (batteryIntent(ctx)?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10.0

    /** Mean-pools token vectors over real tokens and normalizes, like sentence-transformers. */
    private fun pooled(t: OnnxTensor, mask: LongArray): Array<FloatArray> {
        val shape = (t.info as TensorInfo).shape
        val fb = t.floatBuffer
        val flat = FloatArray(fb.remaining()).also { fb.get(it) }
        return if (shape.size == 3) {
            val (b, sq, h) = Triple(shape[0].toInt(), shape[1].toInt(), shape[2].toInt())
            Array(b) { row ->
                val v = FloatArray(h)
                var n = 0
                for (p in 0 until sq) {
                    if (mask[row * sq + p] == 0L) continue
                    n++
                    val base = (row * sq + p) * h
                    for (k in 0 until h) v[k] += flat[base + k]
                }
                normalize(v.also { arr -> if (n > 0) for (k in arr.indices) arr[k] /= n })
            }
        } else {
            val (b, h) = Pair(shape[0].toInt(), shape[1].toInt())
            Array(b) { row -> normalize(flat.copyOfRange(row * h, row * h + h)) }
        }
    }

    private fun normalize(v: FloatArray): FloatArray {
        var n = 0.0
        for (x in v) n += x * x
        val s = sqrt(n).toFloat()
        if (s > 0f) for (k in v.indices) v[k] /= s
        return v
    }

    private fun accuracy(a: Array<FloatArray>, b: Array<FloatArray>): String {
        if (a.isEmpty() || a.size != b.size) return "Accuracy: could not compare"
        val sims = a.indices.map { i -> a[i].indices.sumOf { k -> (a[i][k] * b[i][k]).toDouble() } }
        return "Accuracy vs CPU: average cosine ${"%.5f".format(sims.average())}, worst ${"%.5f".format(sims.min())} (1.0 = identical)"
    }

    private fun f0(x: Double) = "%.0f".format(x)
    private fun f1(x: Double) = "%.1f".format(x)
    private fun f2(x: Double) = "%.2f".format(x)
    private fun f3(x: Double) = "%.3f".format(x)
}

/** Realistic-length test documents (same generator as the Termux benchmark). */
object Docs {
    private val WORDS = ("the a an and or but of to in on for with from by at as is are was were be been " +
        "network phone data model energy system report customer market service order price value " +
        "growth team product design process result policy research study analysis support update " +
        "security privacy device battery signal cloud server compute memory storage speed quality " +
        "city region company project meeting plan budget schedule contract partner supply demand " +
        "review summary document translation transcript article email ticket request response " +
        "people family school health travel weather music video image story history science " +
        "quickly carefully often usually recently clearly strongly slowly directly easily " +
        "important new large small local global public private final early recent common " +
        "increase reduce improve provide include require create develop measure deliver share").split(" ")

    fun make(n: Int, seed: Long): List<String> {
        val rng = Random(seed)
        return List(n) {
            val target = 180 + rng.nextInt(141)
            val words = ArrayList<String>()
            while (words.size < target) {
                val len = 8 + rng.nextInt(13)
                val s = List(len) { WORDS[rng.nextInt(WORDS.size)] }.toMutableList()
                s[0] = s[0].replaceFirstChar { c -> c.uppercase() }
                s[len - 1] = s[len - 1] + "."
                words.addAll(s)
            }
            words.joinToString(" ")
        }
    }
}
