package com.hexagonmesh.app

/** Live node status, written by NodeService and shown by MainActivity. */
object NodeState {
    @Volatile var generation = 0
    @Volatile var running = false
    @Volatile var status = "stopped"
    @Volatile var backend = ""
    @Volatile var tickets = 0
    @Volatile var texts = 0L
    @Volatile var credits = 0.0
    @Volatile var lastTicketMs = 0L
    @Volatile var lastError = ""
}
