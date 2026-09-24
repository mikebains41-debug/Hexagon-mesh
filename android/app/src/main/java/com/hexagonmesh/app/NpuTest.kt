package com.hexagonmesh.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.lang.reflect.InvocationTargetException
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * M1: NPU smoke test. Runs the same small network on the CPU and on the Hexagon NPU
 * (QNN HTP backend) with CPU fallback disabled, then proves where each operation ran
 * from ONNX Runtime's profiling trace.
 */
object NpuTest {
    private const val BATCH = 64
    private const val DIM = 768
    private const val LAYERS = 6
    private const val RUNS = 30
    private const val FLOPS = 2.0 * BATCH * DIM * DIM * LAYERS

    fun run(ctx: Context, log: (String) -> Unit) {
        log("Test model: $LAYERS layers, ${BATCH}x$DIM, ${f0(FLOPS / 1e6)} MFLOP per run")
        log(qnnLibsReport(ctx))

        val env = OrtEnvironment.getEnvironment()
        val model = ctx.assets.open("npu_test.onnx").use { it.readBytes() }
        val input = FloatArray(BATCH * DIM) { i -> ((i.toLong() * 7919 % 1000) / 1000f) - 0.5f }

        // 1. CPU baseline
        val cpu = try {
            OrtSession.SessionOptions().use { opts ->
                env.createSession(model, opts).use { s -> timeRuns(env, s, input) }
            }
        } catch (e: Exception) {
            log("CPU run failed: ${reason(e)}")
            return
        }
        log("CPU: ${f2(cpu.first)} ms/run  (${f1(FLOPS / cpu.first / 1e6)} GFLOPS)")

        // 2. NPU (QNN HTP), CPU fallback disabled so success proves it is on the NPU
        val profilePrefix = File(ctx.filesDir, "npu_profile").absolutePath
        try {
            OrtSession.SessionOptions().use { opts ->
                opts.addConfigEntry("session.disable_cpu_ep_fallback", "1")
                opts.enableProfiling(profilePrefix)
                addQnn(opts, mapOf(
                    "backend_path" to "libQnnHtp.so",
                    "htp_performance_mode" to "burst",
                    "enable_htp_fp16_precision" to "1",
                ))
                val t0 = System.nanoTime()
                env.createSession(model, opts).use { s ->
                    log("NPU session created in ${f0((System.nanoTime() - t0) / 1e6)} ms (graph compiled for Hexagon)")
                    val npu = timeRuns(env, s, input)
                    log("NPU: ${f2(npu.first)} ms/run  (${f1(FLOPS / npu.first / 1e6)} GFLOPS)")
                    log("Speed: NPU is ${f1(cpu.first / npu.first)}x the CPU")
                    log(accuracyReport(cpu.second, npu.second))
                    log(providerReport(s.endProfiling()))
                }
            }
        } catch (e: Exception) {
            log("NPU run failed: ${reason(e)}")
        }
    }

    private fun timeRuns(env: OrtEnvironment, s: OrtSession, input: FloatArray): Pair<Double, FloatArray> {
        val shape = longArrayOf(BATCH.toLong(), DIM.toLong())
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape).use { t ->
            val feeds = mapOf("input" to t)
            repeat(3) { s.run(feeds).close() }                 // warm-up
            var out = FloatArray(0)
            val t0 = System.nanoTime()
            for (i in 0 until RUNS) {
                s.run(feeds).use { r ->
                    if (i == RUNS - 1) {
                        val fb = (r.get(0) as OnnxTensor).floatBuffer
                        out = FloatArray(fb.remaining()).also { fb.get(it) }
                    }
                }
            }
            Pair((System.nanoTime() - t0) / 1e6 / RUNS, out)
        }
    }

    /** addQnn is looked up at runtime so a missing method gives a clear message, not a crash. */
    private fun addQnn(opts: OrtSession.SessionOptions, options: Map<String, String>) {
        val m = opts.javaClass.methods.firstOrNull { it.name == "addQnn" && it.parameterTypes.size == 1 }
            ?: throw IllegalStateException("this ONNX Runtime has no addQnn(); available: " +
                opts.javaClass.methods.map { it.name }.filter { it.startsWith("add") }.distinct())
        m.invoke(opts, options)
    }

    private fun qnnLibsReport(ctx: Context): String {
        val dir = File(ctx.applicationInfo.nativeLibraryDir)
        val qnn = dir.list()?.filter { it.contains("Qnn") }?.sorted() ?: emptyList()
        val skels = qnn.filter { it.contains("Skel") }
        return "QNN libraries on disk: ${qnn.size}" +
            (if (skels.isNotEmpty()) " (DSP skels: ${skels.joinToString { it.removePrefix("libQnnHtp").removeSuffix("Skel.so") }})" else " (no DSP skel found)")
    }

    private fun accuracyReport(a: FloatArray, b: FloatArray): String {
        if (a.size != b.size || a.isEmpty()) return "Accuracy: output size mismatch"
        var maxDiff = 0.0; var dot = 0.0; var na = 0.0; var nb = 0.0
        for (i in a.indices) {
            maxDiff = maxOf(maxDiff, abs(a[i] - b[i]).toDouble())
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        val cos = dot / (sqrt(na) * sqrt(nb))
        return "Accuracy vs CPU: cosine ${"%.5f".format(cos)}, max difference ${"%.5f".format(maxDiff)} (NPU uses fp16)"
    }

    /** Counts which execution provider ran each operation, from the profiling JSON. */
    private fun providerReport(path: String): String {
        val text = try { File(path).readText() } catch (e: Exception) { return "Profile unreadable: ${e.message}" }
        val counts = Regex("\"provider\"\\s*:\\s*\"(\\w+)\"").findAll(text)
            .groupingBy { it.groupValues[1] }.eachCount()
        if (counts.isEmpty()) return "Profile: no per-operation records found"
        val onNpu = counts.keys.all { it == "QNNExecutionProvider" }
        return "Where ops ran: " + counts.entries.joinToString { "${it.key} x${it.value}" } +
            if (onNpu) "\nPROOF: every operation ran on the Hexagon NPU" else "\nWARNING: some operations did not run on the NPU"
    }

    private fun reason(e: Throwable): String {
        val root = if (e is InvocationTargetException && e.cause != null) e.cause!! else e
        return (root.message ?: root.javaClass.simpleName).take(600)
    }

    private fun f0(x: Double) = "%.0f".format(x)
    private fun f1(x: Double) = "%.1f".format(x)
    private fun f2(x: Double) = "%.2f".format(x)
}
