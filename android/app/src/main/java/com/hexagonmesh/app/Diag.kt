package com.hexagonmesh.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import org.json.JSONObject
import java.nio.LongBuffer
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * M2.4 diagnostic: runs one batch on the CPU (fp32) and on each NPU variant, then compares every
 * exposed internal tensor (each LayerNorm output, first attention probs, first Gelu) to find the
 * first point where the NPU's numbers diverge from the CPU's.
 */
object Diag {
    private const val BATCH = 16
    private const val SEQ = 256
    private const val LN_MODEL = "minilm_ln_16x256.onnx"
    private const val Q8_MODEL = "minilm_a16w8_16x256.onnx"
    private const val HTP = "libQnnHtp.so"

    private class Run(val session: OrtSession, val result: OrtSession.Result, val feeds: Map<String, OnnxTensor>) {
        fun close() { result.close(); feeds.values.forEach { it.close() }; session.close() }
    }

    fun run(ctx: Context, log: (String) -> Unit) {
        val env = OrtEnvironment.getEnvironment()
        val tok = WordPiece(ctx.assets.open("vocab.txt").bufferedReader().readLines())
        val batch = tok.encodeBatch(Docs.make(BATCH, 0L), SEQ)
        val labels = LinkedHashMap<String, String>()
        try {
            val j = JSONObject(ctx.assets.open("embed_variants.json").bufferedReader().readText())
            val arr = j.getJSONArray("diag_outputs")
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                labels[o.getString("name")] = o.getString("label")
            }
        } catch (e: Exception) {
            log("This build has no internal-tensor list; rebuild with M2.4."); return
        }
        log("Comparing ${labels.size} internal values, CPU fp32 vs NPU, on one batch of $BATCH documents.")
        log("cos = similarity (1.0 = identical). max|cpu| = largest value. diff = largest difference.")

        val lnPath = EmbedBench.assetToFile(ctx, LN_MODEL)
        val qPath = EmbedBench.assetToFile(ctx, Q8_MODEL)
        val cpu = try {
            open(env, lnPath, null, batch)
        } catch (e: Exception) {
            log("CPU run failed: ${NpuTest.reason(e)}"); return
        }
        try {
            val variants = listOf(
                Triple("NPU fp16", lnPath, true),
                Triple("NPU quantized (16-bit act, 8-bit weights)", qPath, false),
            )
            for ((label, path, fp16) in variants) {
                log("\n[$label]")
                val npu = try {
                    open(env, path, qnnOptions(fp16, true), batch)
                } catch (e: Exception) {
                    log("strict NPU refused (${NpuTest.reason(e).take(80)}), allowing fallback")
                    try {
                        open(env, path, qnnOptions(fp16, false), batch)
                    } catch (e2: Exception) {
                        log("could not run: ${NpuTest.reason(e2).take(200)}"); continue
                    }
                }
                try {
                    for ((name, lab) in labels) {
                        val a = tensor(cpu.result, name)
                        val b = tensor(npu.result, name)
                        if (a == null || b == null) { log("  $lab: not available"); continue }
                        log("  " + compare(lab, a, b, batch.mask))
                    }
                    val finalName = cpu.session.outputNames.firstOrNull { it !in labels }
                    val a0 = if (finalName != null) tensor(cpu.result, finalName) else null
                    val b0 = if (finalName != null) tensor(npu.result, finalName) else null
                    if (a0 != null && b0 != null) log("  " + compare("final output", a0, b0, batch.mask))
                } finally {
                    npu.close()
                }
            }
        } finally {
            cpu.close()
        }
    }

    private fun qnnOptions(fp16: Boolean, strict: Boolean): OrtSession.SessionOptions {
        val o = OrtSession.SessionOptions()
        if (strict) o.addConfigEntry("session.disable_cpu_ep_fallback", "1")
        val qnn = hashMapOf("backend_path" to HTP, "htp_performance_mode" to "burst")
        if (fp16) qnn["enable_htp_fp16_precision"] = "1"
        NpuTest.addQnn(o, qnn)
        return o
    }

    private fun open(env: OrtEnvironment, path: String, opts: OrtSession.SessionOptions?, b: WordPiece.Batch): Run {
        val session = if (opts == null) env.createSession(path) else env.createSession(path, opts)
        val shape = longArrayOf(BATCH.toLong(), SEQ.toLong())
        val feeds = HashMap<String, OnnxTensor>()
        for (name in session.inputNames) {
            val data = when (name) {
                "input_ids" -> b.ids
                "attention_mask" -> b.mask
                else -> LongArray(BATCH * SEQ)
            }
            feeds[name] = OnnxTensor.createTensor(env, LongBuffer.wrap(data), shape)
        }
        val result = session.run(feeds)
        opts?.close()
        return Run(session, result, feeds)
    }

    private fun tensor(r: OrtSession.Result, name: String): OnnxTensor? =
        try { r.get(name).orElse(null) as? OnnxTensor } catch (e: Exception) { null }

    /** Cosine over the entries that belong to real (non-padding) tokens, plus magnitude and max difference. */
    private fun compare(label: String, a: OnnxTensor, b: OnnxTensor, mask: LongArray): String {
        val shape = (a.info as TensorInfo).shape
        val fa = a.floatBuffer
        val fb = b.floatBuffer
        val n = fa.remaining()
        if (fb.remaining() != n) return "$label: size mismatch (${fb.remaining()} vs $n)"
        val rowLen: Int
        val rowToken: (Int) -> Int
        if (shape.size == 3) {
            rowLen = shape[2].toInt()
            rowToken = { row -> row }                                   // [batch, seq, hidden] -> token index
        } else if (shape.size == 4) {
            rowLen = shape[3].toInt()
            val heads = shape[1].toInt(); val sq = shape[2].toInt()
            rowToken = { row -> (row / (heads * sq)) * sq + row % sq }  // [batch, heads, q, k] -> query token
        } else {
            rowLen = n; rowToken = { 0 }
        }
        var dot = 0.0; var na = 0.0; var nb = 0.0; var maxAbs = 0.0; var maxDiff = 0.0; var bad = 0
        val rows = n / rowLen
        for (row in 0 until rows) {
            if (rowLen != n && mask[rowToken(row)] == 0L) continue
            val base = row * rowLen
            for (k in 0 until rowLen) {
                val x = fa.get(base + k); val y = fb.get(base + k)
                if (y.isNaN() || y.isInfinite()) { bad++; continue }
                dot += x * y; na += x * x; nb += y * y
                maxAbs = maxOf(maxAbs, abs(x).toDouble()); maxDiff = maxOf(maxDiff, abs(x - y).toDouble())
            }
        }
        val cos = if (na > 0 && nb > 0) dot / (sqrt(na) * sqrt(nb)) else 0.0
        return "$label: cos ${"%.4f".format(cos)}, max|cpu| ${"%.1f".format(maxAbs)}, diff ${"%.3f".format(maxDiff)}" +
            if (bad > 0) ", NaN/Inf on NPU: $bad" else ""
    }
}
