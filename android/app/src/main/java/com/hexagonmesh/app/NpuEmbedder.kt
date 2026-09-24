package com.hexagonmesh.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import java.nio.LongBuffer
import kotlin.math.sqrt

/** all-MiniLM-L6-v2 on the Hexagon NPU, using the verified mixed-precision model (0.999 vs CPU). */
class NpuEmbedder(ctx: Context) : AutoCloseable {
    companion object {
        const val MODEL_NAME = "minilm-l6"
        private const val BATCH = 16
        private const val SEQ = 256
        private const val MODEL_FILE = "minilm_a16w8_16x256.onnx"
    }

    private val env = OrtEnvironment.getEnvironment()
    private val tok = WordPiece(ctx.assets.open("vocab.txt").bufferedReader().readLines())
    private val opts = OrtSession.SessionOptions()
    private val session: OrtSession
    val backend: String

    init {
        val path = EmbedBench.assetToFile(ctx, MODEL_FILE)
        var s: OrtSession?
        var b: String
        try {
            NpuTest.addQnn(opts, hashMapOf(
                "backend_path" to "libQnnHtp.so",
                "htp_performance_mode" to "burst",
                "enable_htp_fp16_precision" to "1"))
            s = env.createSession(path, opts)
            b = "NPU"
        } catch (e: Exception) {
            s = env.createSession(path, OrtSession.SessionOptions())
            b = "CPU (NPU unavailable)"
        }
        session = s!!
        backend = b
    }

    /** One normalized 384-number embedding per text, in order. */
    fun embed(texts: List<String>): List<FloatArray> {
        val out = ArrayList<FloatArray>(texts.size)
        var i = 0
        val shape = longArrayOf(BATCH.toLong(), SEQ.toLong())
        while (i < texts.size) {
            val chunk = texts.subList(i, minOf(i + BATCH, texts.size))
            val padded = chunk + List(BATCH - chunk.size) { "" }
            val b = tok.encodeBatch(padded, SEQ)
            val feeds = HashMap<String, OnnxTensor>()
            try {
                for (name in session.inputNames) {
                    val data = when (name) {
                        "input_ids" -> b.ids
                        "attention_mask" -> b.mask
                        else -> LongArray(BATCH * SEQ)
                    }
                    feeds[name] = OnnxTensor.createTensor(env, LongBuffer.wrap(data), shape)
                }
                session.run(feeds).use { res ->
                    val pooled = pool(res.get(0) as OnnxTensor, b.mask)
                    for (k in chunk.indices) out.add(pooled[k])
                }
            } finally {
                feeds.values.forEach { it.close() }
            }
            i += BATCH
        }
        return out
    }

    private fun pool(t: OnnxTensor, mask: LongArray): Array<FloatArray> {
        val shape = (t.info as TensorInfo).shape
        val fb = t.floatBuffer
        val flat = FloatArray(fb.remaining()).also { fb.get(it) }
        val rows = shape[0].toInt()
        val sq = shape[1].toInt()
        val h = shape[2].toInt()
        return Array(rows) { row ->
            val v = FloatArray(h)
            var n = 0
            for (p in 0 until sq) {
                if (mask[row * sq + p] == 0L) continue
                n++
                val base = (row * sq + p) * h
                for (k in 0 until h) v[k] += flat[base + k]
            }
            if (n > 0) for (k in 0 until h) v[k] /= n
            var norm = 0.0
            for (x in v) norm += x * x
            val len = sqrt(norm).toFloat()
            if (len > 0f) for (k in 0 until h) v[k] /= len
            v
        }
    }

    override fun close() {
        session.close()
        opts.close()
    }
}
