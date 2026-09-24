package com.hexagonmesh.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import org.json.JSONObject
import java.io.File
import java.nio.LongBuffer
import java.util.Random
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * M2.9: all-MiniLM-L6-v2 embeddings on realistic documents. A fp32 CPU run is the reference;
 * each accelerator variant is scored for speed, accuracy (PASS at cosine 0.99) and where its
 * operations ran. The best passing NPU variant then gets a sustained power run.
 */
object EmbedBench {
    private const val BATCH = 16
    private const val SEQ = 256
    private const val LN_MODEL = "minilm_ln_16x256.onnx"
    private const val Q8_MODEL = "minilm_a16w8_16x256.onnx"
    private const val HTP = "libQnnHtp.so"
    private const val GPU = "libQnnGpu.so"
    private const val PASS = 0.99
    private const val NOMINAL_V = 3.85
    private const val SUSTAIN_S = 45
    private const val IDLE_S = 10

    private class Variant(val label: String, val file: String, val backend: String, val fp16: Boolean)

    private val VARIANTS = listOf(
        Variant("NPU (quantized attention + fp16 residual stream)", Q8_MODEL, HTP, true),
        Variant("GPU fp32 (Adreno), for comparison", LN_MODEL, GPU, false),
    )

    private class Timing(val tokPerSec: Double, val emb: Array<FloatArray>)
    private class Result(val v: Variant, val tokPerSec: Double, val avg: Double, val worst: Double, val strict: Boolean)

    fun run(ctx: Context, log: (String) -> Unit) {
        val env = OrtEnvironment.getEnvironment()
        val tok = WordPiece(ctx.assets.open("vocab.txt").bufferedReader().readLines())
        val batches = (0 until 4).map { tok.encodeBatch(Docs.make(BATCH, it.toLong()), SEQ) }
        log("Model: all-MiniLM-L6-v2 (22M parameters), $BATCH documents x $SEQ tokens per batch")
        buildReport(ctx)?.let(log)
        log("Preparing models (first run copies about 120 MB)...")
        val paths = listOf(LN_MODEL, Q8_MODEL).associateWith { assetToFile(ctx, it) }

        val cpu = try {
            OrtSession.SessionOptions().use { o ->
                env.createSession(paths.getValue(LN_MODEL), o).use { s -> measure(env, s, batches, 2) }
            }
        } catch (e: Exception) {
            log("CPU reference failed: ${NpuTest.reason(e)}"); return
        }
        log("CPU fp32 reference: ${f0(cpu.tokPerSec)} tokens/sec")

        val results = ArrayList<Result>()
        for (v in VARIANTS) {
            log("\n[${v.label}]")
            runVariant(ctx, env, paths.getValue(v.file), v, batches, cpu, log)?.let { results.add(it) }
        }

        val npu = results.filter { it.v.backend == HTP }
        val best = npu.filter { it.avg >= PASS }.maxByOrNull { it.tokPerSec } ?: npu.maxByOrNull { it.avg }
        log("\nSUMMARY")
        for (r in results) {
            log("${if (r.avg >= PASS) "PASS" else "FAIL"}  ${r.v.label}: ${f0(r.tokPerSec)} tok/s, accuracy ${f4(r.avg)}")
        }
        if (best == null) { log("No NPU variant ran."); return }
        log("Best NPU variant: ${best.v.label}${if (best.avg >= PASS) "" else " (still below 0.99)"}")

        val opts = options(best.v, null, best.strict)
        try {
            env.createSession(paths.getValue(best.v.file), opts).use { s -> sustained(ctx, env, s, batches, log) }
        } catch (e: Exception) {
            log("Sustained run failed: ${NpuTest.reason(e)}")
        } finally {
            opts.close()
        }
    }

    private fun runVariant(ctx: Context, env: OrtEnvironment, path: String, v: Variant,
                           batches: List<WordPiece.Batch>, cpu: Timing, log: (String) -> Unit): Result? {
        val profile = File(ctx.filesDir, "embed_profile").absolutePath
        var strict = true
        var opts = options(v, profile, true)
        val session = try {
            env.createSession(path, opts)
        } catch (e: Exception) {
            opts.close()
            log("Strict mode refused: ${NpuTest.reason(e).take(200)}")
            strict = false
            opts = options(v, profile, false)
            try {
                env.createSession(path, opts)
            } catch (e2: Exception) {
                opts.close()
                log("Could not run: ${NpuTest.reason(e2).take(300)}")
                return null
            }
        }
        try {
            val t = measure(env, session, batches, 4)
            val (avg, worst) = cosines(cpu.emb, t.emb)
            log("Speed: ${f0(t.tokPerSec)} tokens/sec (${f1(t.tokPerSec / cpu.tokPerSec)}x the CPU)")
            log("Accuracy vs CPU: average ${f4(avg)}, worst ${f4(worst)}  ${if (avg >= PASS) "PASS" else "FAIL"}")
            log(NpuTest.providerReport(session.endProfiling()) + if (strict) "" else "\n(CPU fallback was allowed)")
            return Result(v, t.tokPerSec, avg, worst, strict)
        } catch (e: Exception) {
            log("Failed while running: ${NpuTest.reason(e)}")
            return null
        } finally {
            session.close(); opts.close()
        }
    }

    private fun options(v: Variant, profilePath: String?, strict: Boolean): OrtSession.SessionOptions {
        val o = OrtSession.SessionOptions()
        if (strict) o.addConfigEntry("session.disable_cpu_ep_fallback", "1")
        if (profilePath != null) o.enableProfiling(profilePath)
        val qnn = HashMap<String, String>()
        qnn["backend_path"] = v.backend
        if (v.backend == HTP) {
            qnn["htp_performance_mode"] = "burst"
            if (v.fp16) qnn["enable_htp_fp16_precision"] = "1"
        }
        NpuTest.addQnn(o, qnn)
        return o
    }

    /** Checks the build server ran on this model (overflow diagnosis and CPU-side accuracy). */
    private fun buildReport(ctx: Context): String? = try {
        val j = JSONObject(ctx.assets.open("embed_variants.json").bufferedReader().readText())
        val sb = StringBuilder("Build-server checks:")
        if (j.has("max_layernorm_square")) {
            val mx = j.getDouble("max_layernorm_square")
            sb.append("\n  Largest single squared value inside LayerNorm: ${"%,.0f".format(mx)}")
            sb.append("\n  LayerNorm sums 384 of these, which can pass fp16's 65,504 limit")
        }
        j.optJSONObject("fused")?.let { f ->
            sb.append("\n  Fused: ${f.optInt("LayerNormalization")} LayerNorm, ${f.optInt("Gelu")} Gelu (loose Pow left: ${f.optInt("Pow")})")
        }
        if (j.has("prescale")) {
            val sc = j.getDouble("prescale")
            sb.append("\n  LayerNorm inputs pre-scaled by ${"%.4f".format(sc)}: squares ${"%.0f".format(1 / (sc * sc))}x smaller")
        }
        if (j.has("attention_scale_folded")) {
            sb.append("\n  Attention scaling folded into Q in ${j.getInt("attention_scale_folded")} layers")
            sb.append("\n  Ops feeding softmax: ${j.optString("softmax_chain_before")} -> ${j.optString("softmax_chain_after")}")
        }
        j.optJSONArray("ln_vs_original")?.let { a ->
            sb.append("\n  Rewritten fp32 model vs original (CPU): ${f4(a.getDouble(0))}")
        }
        j.optJSONArray("recipes")?.let { rs ->
            sb.append("\n  Quantization recipes scored on the build server (CPU):")
            for (i in 0 until rs.length()) {
                val r = rs.getJSONObject(i)
                val err = r.optString("error", "")
                sb.append("\n    ${r.optString("recipe")}: " +
                    if (err.isNotEmpty()) "failed" else "${f4(r.optDouble("avg"))} (worst doc ${f4(r.optDouble("worst"))})")
            }
            sb.append("\n  Shipped recipe: ${j.optString("quant_recipe")}")
        }
        sb.toString()
    } catch (e: Exception) {
        null
    }

    /** Copies a model out of the APK once, so ONNX Runtime can load it without using app memory. */
    internal fun assetToFile(ctx: Context, name: String): String {
        val f = File(ctx.filesDir, "m29_$name")
        if (!f.exists() || f.length() == 0L) {
            val tmp = File(ctx.filesDir, "m29_$name.tmp")
            ctx.assets.open(name).use { input -> tmp.outputStream().use { out -> input.copyTo(out) } }
            tmp.renameTo(f)
        }
        return f.absolutePath
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
            return Timing(tokens / secs, emb)
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
        log("\nMeasuring idle power for ${IDLE_S}s, leave the phone alone...")
        val idle = sample(bm, IDLE_S * 1000L)
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

    private fun sample(bm: BatteryManager, ms: Long): List<Double> {
        val out = ArrayList<Double>()
        val end = System.currentTimeMillis() + ms
        while (System.currentTimeMillis() < end) {
            out.add(abs(bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)) / 1000.0)
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
            val b = shape[0].toInt()
            val sq = shape[1].toInt()
            val h = shape[2].toInt()
            Array(b) { row ->
                val v = FloatArray(h)
                var n = 0
                for (p in 0 until sq) {
                    if (mask[row * sq + p] == 0L) continue
                    n++
                    val base = (row * sq + p) * h
                    for (k in 0 until h) v[k] += flat[base + k]
                }
                if (n > 0) for (k in 0 until h) v[k] /= n
                normalize(v)
            }
        } else {
            val b = shape[0].toInt()
            val h = shape[1].toInt()
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

    private fun cosines(a: Array<FloatArray>, b: Array<FloatArray>): Pair<Double, Double> {
        if (a.isEmpty() || a.size != b.size) return Pair(0.0, 0.0)
        val sims = a.indices.map { i -> a[i].indices.sumOf { k -> (a[i][k] * b[i][k]).toDouble() } }
        return Pair(sims.average(), sims.min())
    }

    private fun f0(x: Double) = "%.0f".format(x)
    private fun f1(x: Double) = "%.1f".format(x)
    private fun f2(x: Double) = "%.2f".format(x)
    private fun f3(x: Double) = "%.3f".format(x)
    private fun f4(x: Double) = "%.4f".format(x)
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
