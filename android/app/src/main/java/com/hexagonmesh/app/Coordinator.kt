package com.hexagonmesh.app

import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/** Minimal client for the Hexagon Mesh coordinator API. */
class Coordinator(baseUrl: String, private val apiKey: String) {
    private val base = baseUrl.trim().trimEnd('/')

    class Response(val code: Int, val body: JSONObject)

    private fun call(method: String, path: String, body: JSONObject?): Response {
        val conn = URL(base + path).openConnection() as HttpURLConnection
        try {
            conn.requestMethod = method
            conn.connectTimeout = 10_000
            conn.readTimeout = 60_000
            conn.setRequestProperty("Content-Type", "application/json")
            if (apiKey.isNotBlank()) conn.setRequestProperty("X-API-Key", apiKey)
            if (body != null) {
                conn.doOutput = true
                conn.outputStream.use { it.write(body.toString().toByteArray()) }
            }
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val text = stream?.bufferedReader()?.use { it.readText() } ?: ""
            val json = try {
                if (text.isBlank()) JSONObject() else JSONObject(text)
            } catch (e: Exception) {
                JSONObject()
            }
            return Response(code, json)
        } finally {
            conn.disconnect()
        }
    }

    fun checkin(nodeId: String, wallet: String, ramGb: Double, models: List<String>): Response =
        call("POST", "/nodes/checkin", JSONObject()
            .put("node_id", nodeId).put("wallet", wallet).put("ram_gb", ramGb).put("models", JSONArray(models)))

    fun next(nodeId: String): Response =
        call("POST", "/work/next", JSONObject().put("node_id", nodeId))

    fun submit(nodeId: String, assignmentId: Long, result: JSONArray): Response =
        call("POST", "/work/submit", JSONObject()
            .put("node_id", nodeId).put("assignment_id", assignmentId).put("result", result))

    fun balance(nodeId: String): Response = call("GET", "/nodes/$nodeId/balance", null)
}
