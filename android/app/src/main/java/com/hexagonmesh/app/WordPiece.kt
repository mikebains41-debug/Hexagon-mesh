package com.hexagonmesh.app

import java.text.Normalizer

/** BERT WordPiece tokenizer (lowercase, accent stripping), as used by all-MiniLM-L6-v2. */
class WordPiece(vocabLines: List<String>) {
    private val vocab = HashMap<String, Int>(vocabLines.size * 2)
    init { vocabLines.forEachIndexed { i, w -> vocab[w.trim()] = i } }

    private val unk = vocab["[UNK]"] ?: 100
    private val cls = vocab["[CLS]"] ?: 101
    private val sep = vocab["[SEP]"] ?: 102
    private val pad = vocab["[PAD]"] ?: 0

    class Batch(val ids: LongArray, val mask: LongArray, val realTokens: Int)

    fun encodeBatch(texts: List<String>, seq: Int): Batch {
        val ids = LongArray(texts.size * seq) { pad.toLong() }
        val mask = LongArray(texts.size * seq)
        var real = 0
        texts.forEachIndexed { row, text ->
            val toks = ArrayList<Int>(seq)
            toks.add(cls)
            tokenize(text).take(seq - 2).forEach { toks.add(it) }
            toks.add(sep)
            toks.forEachIndexed { col, id ->
                ids[row * seq + col] = id.toLong()
                mask[row * seq + col] = 1L
            }
            real += toks.size
        }
        return Batch(ids, mask, real)
    }

    fun tokenize(text: String): List<Int> {
        val out = ArrayList<Int>()
        for (word in basicSplit(text)) {
            if (word.length > 100) { out.add(unk); continue }
            val pieces = ArrayList<Int>()
            var start = 0
            var bad = false
            while (start < word.length) {
                var end = word.length
                var found = -1
                while (start < end) {
                    val sub = (if (start > 0) "##" else "") + word.substring(start, end)
                    val id = vocab[sub]
                    if (id != null) { found = id; break }
                    end--
                }
                if (found < 0) { bad = true; break }
                pieces.add(found)
                start = end
            }
            if (bad) out.add(unk) else out.addAll(pieces)
        }
        return out
    }

    private fun basicSplit(text: String): List<String> {
        val norm = Normalizer.normalize(text.lowercase(), Normalizer.Form.NFD)
        val words = ArrayList<String>()
        val sb = StringBuilder()
        fun flush() {
            if (sb.isNotEmpty()) { words.add(sb.toString()); sb.setLength(0) }
        }
        for (ch in norm) {
            when {
                Character.getType(ch) == Character.NON_SPACING_MARK.toInt() -> {}
                ch.isWhitespace() -> flush()
                ch.code == 0 || ch.code == 0xFFFD || Character.isISOControl(ch) -> {}
                isPunctuation(ch) || isCjk(ch) -> { flush(); words.add(ch.toString()) }
                else -> sb.append(ch)
            }
        }
        flush()
        return words
    }

    private fun isPunctuation(ch: Char): Boolean {
        val c = ch.code
        if (c in 33..47 || c in 58..64 || c in 91..96 || c in 123..126) return true
        return when (Character.getType(ch)) {
            Character.CONNECTOR_PUNCTUATION.toInt(), Character.DASH_PUNCTUATION.toInt(),
            Character.START_PUNCTUATION.toInt(), Character.END_PUNCTUATION.toInt(),
            Character.INITIAL_QUOTE_PUNCTUATION.toInt(), Character.FINAL_QUOTE_PUNCTUATION.toInt(),
            Character.OTHER_PUNCTUATION.toInt() -> true
            else -> false
        }
    }

    private fun isCjk(ch: Char): Boolean {
        val c = ch.code
        return c in 0x4E00..0x9FFF || c in 0x3400..0x4DBF || c in 0xF900..0xFAFF || c in 0x2F800..0x2FA1F
    }
}
