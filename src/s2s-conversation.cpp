// s2s-conversation.cpp: the voice loop of one conversation
//
// One process, any number of conversations. Every model loads once and is
// shared: the VAD, the turn classifier and the recognizer run one call at a
// time, the echo canceller and the voice batch the connections together, and
// every per stream state lives in the session or the connection.
//
// Each connection runs four threads. The reader owns the incoming frames and
// the listening half: it decodes them, feeds the session, and answers the
// events the state machine raises. The recognizer transcribes every
// committed turn and revision as soon as it is committed, and hands it to
// the responder, which asks the endpoint and speaks the answer unit by unit.
// Splitting them is what makes the barge-in work, since the reader keeps
// consuming audio while an answer runs, and what keeps the words of the user
// live whatever the endpoint does: a request stuck on the network holds the
// answer, never the next transcript. The writer sends every outgoing frame,
// so a client that reads slowly never holds the synthesis worker the
// connections share.
//
// Modes:
//   conversation  recognize, ask the LLM, speak the answer
//   agentic       the same, with the rounds of tool calls the model asks for
//   loopback      recognize and speak the transcript back, no endpoint in the
//                 path, which is how the microphone, the turn detection, the
//                 recognizer and the voice get tested on their own

#include "s2s-conversation.h"

#include "http-client.h"
#include "jarvis-fx.h"
#include "llm-agent.h"
#include "s2s-error.h"
#include "s2s-session.h"
#include "sentence-split.h"
#include "timer.h"
#include "yyjson.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

std::string url_host(const std::string & url) {
    const size_t scheme = url.find("://");
    const size_t start  = scheme == std::string::npos ? 0 : scheme + 3;
    const size_t end    = url.find('/', start);
    return url.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

bool host_allowed(const std::vector<std::string> & hosts, const std::string & url) {
    if (hosts.empty()) {
        return true;
    }
    const std::string host = url_host(url);
    for (const std::string & entry : hosts) {
        if (host == url_host(entry)) {
            return true;
        }
    }
    return false;
}

#define VOICE_TOOL "set_voice"  // the name the model calls the built-in voice tool by
#define SLEEP_TOOL "sleep"      // and the built-in sleep tool

// Every effect and what it does, in the words the model reads.
struct VoiceEffect {
    const char * name;
    const char * about;
};

static const VoiceEffect EFFECTS[] = {
    { "off",    "the voice as it is"   },
    { "jarvis", "an echo and a chorus" },
};

const std::vector<std::string> & conn_effects() {
    static const std::vector<std::string> names = []() {
        std::vector<std::string> out;
        for (const VoiceEffect & effect : EFFECTS) {
            out.push_back(effect.name);
        }
        return out;
    }();
    return names;
}

static bool conn_effect_known(const std::string & effect) {
    const std::vector<std::string> & names = conn_effects();
    return std::find(names.begin(), names.end(), effect) != names.end();
}

const std::vector<std::string> & conn_wake_modes() {
    static const std::vector<std::string> modes = { "off", "alone", "anywhere" };
    return modes;
}

static bool conn_wake_known(const std::string & mode) {
    const std::vector<std::string> & modes = conn_wake_modes();
    return std::find(modes.begin(), modes.end(), mode) != modes.end();
}

// The words of a text as the wake gate compares them: lowercase, and every
// run of anything but letters and digits one separator. A byte past ASCII
// counts as a letter, so an accented word stays whole.
static std::vector<std::string> wake_words(const std::string & text) {
    std::vector<std::string> words;
    std::string              word;
    for (const unsigned char c : text) {
        if (c >= 0x80 || isalnum(c)) {
            word += (char) tolower(c);
        } else if (!word.empty()) {
            words.push_back(word);
            word.clear();
        }
    }
    if (!word.empty()) {
        words.push_back(word);
    }
    return words;
}

// Whether the words of one of the phrases make the whole text, alone, or
// stand in it in a row anywhere.
static bool wake_heard(const std::string & text, const std::vector<std::string> & phrases, bool alone) {
    const std::vector<std::string> words = wake_words(text);
    for (const std::string & phrase : phrases) {
        const std::vector<std::string> wanted = wake_words(phrase);
        if (wanted.empty()) {
            continue;
        }
        if (alone ? words == wanted :
                    std::search(words.begin(), words.end(), wanted.begin(), wanted.end()) != words.end()) {
            return true;
        }
    }
    return false;
}

llm_tool conn_sleep_tool() {
    const std::string about =
        "Stops answering until the user says a wake phrase again, for the rest of the conversation when none is "
        "set. What you write with the call is still spoken. "
        "Call it when the user thanks you, says goodbye or expects nothing more, or when what you hear makes no "
        "sense or is not meant for you.";

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "type", "function");
    yyjson_mut_val * function = yyjson_mut_obj_add_obj(doc, root, "function");
    yyjson_mut_obj_add_str(doc, function, "name", SLEEP_TOOL);
    yyjson_mut_obj_add_strn(doc, function, "description", about.c_str(), about.size());
    yyjson_mut_val * parameters = yyjson_mut_obj_add_obj(doc, function, "parameters");
    yyjson_mut_obj_add_str(doc, parameters, "type", "object");
    yyjson_mut_obj_add_obj(doc, parameters, "properties");

    char *   json = yyjson_mut_write(doc, 0, nullptr);
    llm_tool tool = { SLEEP_TOOL, json ? json : "{}" };
    free(json);
    yyjson_mut_doc_free(doc);
    return tool;
}

llm_tool conn_voice_tool(const tts_bridge * tts) {
    std::string effects;
    for (const VoiceEffect & e : EFFECTS) {
        effects += std::string(effects.empty() ? "" : ", ") + e.name + " for " + e.about;
    }
    const std::string about =
        "Changes the voice you speak with, for the rest of the conversation: what you write after the call is "
        "spoken with it. The suffix of a voice is the language it was recorded in: an original accent voice "
        "speaks with the accent of that language, a timbre only voice keeps its timbre alone. Try to pick a voice "
        "in the language of the user. effect runs over any voice: " +
        effects + "; left out, it does not change.";

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "type", "function");
    yyjson_mut_val * function = yyjson_mut_obj_add_obj(doc, root, "function");
    yyjson_mut_obj_add_str(doc, function, "name", VOICE_TOOL);
    yyjson_mut_obj_add_strn(doc, function, "description", about.c_str(), about.size());
    yyjson_mut_val * parameters = yyjson_mut_obj_add_obj(doc, function, "parameters");
    yyjson_mut_obj_add_str(doc, parameters, "type", "object");
    yyjson_mut_val * properties = yyjson_mut_obj_add_obj(doc, parameters, "properties");
    yyjson_mut_val * name       = yyjson_mut_obj_add_obj(doc, properties, "voice");
    yyjson_mut_obj_add_str(doc, name, "type", "string");
    yyjson_mut_val * labels = yyjson_mut_obj_add_arr(doc, name, "enum");
    for (const std::string & label : tts_bridge_voices(tts)) {
        yyjson_mut_arr_add_strn(doc, labels, label.c_str(), label.size());
    }
    yyjson_mut_val * fx = yyjson_mut_obj_add_obj(doc, properties, "effect");
    yyjson_mut_obj_add_str(doc, fx, "type", "string");
    yyjson_mut_val * names = yyjson_mut_obj_add_arr(doc, fx, "enum");
    for (const VoiceEffect & e : EFFECTS) {
        yyjson_mut_arr_add_str(doc, names, e.name);
    }
    yyjson_mut_val * required = yyjson_mut_obj_add_arr(doc, parameters, "required");
    yyjson_mut_arr_add_str(doc, required, "voice");

    char *   json = yyjson_mut_write(doc, 0, nullptr);
    llm_tool tool = { VOICE_TOOL, json ? json : "{}" };
    free(json);
    yyjson_mut_doc_free(doc);
    return tool;
}

// A committed turn, with the identity the session gave it.
struct TurnAudio {
    std::vector<float> pcm;
    int                turn_id  = 0;
    int                revision = 0;
};

// A recognized turn, waiting for its answer.
struct AnswerJob {
    std::string transcript;
    int         turn_id  = 0;
    int         revision = 0;
};

struct Connection {
    // What every conversation of the process shares, and how a frame reaches
    // the client: the transport belongs to whoever opened the conversation.
    const ConversationSetup * setup     = nullptr;
    conn_send_fn              send      = nullptr;
    void *                    send_user = nullptr;

    // Numbers the connection in the log: its four threads write as Reader-N,
    // Recognizer-N, Responder-N and Writer-N, so one grep follows one client.
    int id = 0;

    // The turn state machine. The reader feeds it and the responder releases
    // its turns, each call under session_mutex. Its events run under that
    // lock and take the turn, queue and output locks after it; a session
    // update takes it under client_mutex. Never the reverse.
    std::mutex         session_mutex;
    s2s_session *      session = nullptr;
    s2s_session_params params;

    // server and both run a canceller here: client and off leave the
    // microphone as the client captured it. The default method is the
    // component's, which picks its microphone constraints before the server
    // is even reached.
    std::string echo = "client";

    // Written by the reader, copied whole by the responder when it answers a
    // turn: a change lands on the next answer, never under a running one.
    std::mutex     client_mutex;
    ClientSettings client;

    // The wake phrase gate, under client_mutex: whether the assistant
    // answers, and the turn it last let through, whose revisions pass as it
    // did. It answers with the gate off, and a change of mode wakes it for
    // off and puts it to sleep for the others.
    bool awake         = true;
    int  admitted_turn = -1;

    // Frames to the client. Every thread queues, one writer sends: a client
    // that reads slowly stalls its own writer, never the synthesis worker
    // every connection shares.
    std::mutex              out_mutex;
    std::condition_variable out_cv;
    std::queue<SendData> out;
    bool                    out_stop = false;

    // The endpoint client, the responder's alone. It lives across turns so its
    // connection stays open: a turn pays the TCP and TLS handshakes only when
    // the endpoint host changes.
    llm_client * llm = nullptr;

    // The tool servers of the agentic mode, the responder's alone. The MCP
    // sessions in it stay open across turns and follow the list the session
    // named, kept here to tell a change.
    llm_agent *                    agent = nullptr;
    std::vector<mcp_server_params> agent_servers;

    // Committed turns waiting for the recognizer, and recognized ones waiting
    // for the responder.
    std::mutex              queue_mutex;
    std::condition_variable queue_cv;
    std::queue<TurnAudio>   queue;
    std::condition_variable answers_cv;
    std::queue<AnswerJob>   answers;
    std::atomic<bool>       stop{ false };

    // Raised against the answer in flight; the responder lowers it as it
    // starts the next one.
    std::atomic<bool> cancel{ false };

    // The first sample of an answer waits for its turn to be final. The
    // session decides on the reader thread: the latest turn declared final,
    // and the latest turn and revision that opened. An answer to anything
    // older than what opened last is void.
    std::mutex              turn_mutex;
    std::condition_variable turn_cv;
    int                     final_turn    = 0;
    int                     open_turn     = 0;
    int                     open_revision = 0;

    // The answer in flight: its turn and revision, written by the responder
    // under turn_mutex and read there by the reader, and whether its first
    // unit has gone past the wait, the responder's alone.
    int  answer_turn     = 0;
    int  answer_revision = 0;
    bool answer_released = false;

    // Whether the assistant holds the floor: set by the responder, handed to
    // the session by the reader with every push.
    std::atomic<bool> speaking{ false };

    // The Jarvis effect over the voice, one stream per answer. Only the
    // synthesis callbacks of the answer in flight touch it, and the responder
    // between two answers, never both at once.
    jarvis_fx *        fx = nullptr;
    std::vector<float> fx_pcm;

    // The protocol carries 24 kHz, the models want 16 kHz: the microphone and
    // the reference go through the same Hann-windowed sinc, the one of
    // torchaudio, so they stay aligned sample for sample.
    AudioResampleStream mic_resample;
    AudioResampleStream ref_resample;

    // Server side echo cancellation, alive while the client asks for it. The
    // canceller works in hops, so what does not fill one waits here.
    lv_state *         aec = nullptr;
    std::vector<float> aec_mic;
    std::vector<float> aec_ref;

    // Energy in and out of the canceller over one run of playback, and the
    // compute of every hop since the last report, for the log lines that
    // close the run. The output is one hop late, so the energy of the
    // microphone hop waits one call for the output it produced.
    double aec_in         = 0.0;
    double aec_out        = 0.0;
    size_t aec_played     = 0;
    int    aec_quiet      = 0;
    size_t aec_hops       = 0;
    double aec_ms         = 0.0;
    double aec_peak       = 0.0;
    double aec_pending_in = 0.0;
    bool   aec_pending    = false;
    bool   aec_failing    = false;

    // The threads of the talking half, and what the reader keeps between two
    // frames: whether the microphone was announced, and its buffers.
    std::thread        writer;
    std::thread        recognizer;
    std::thread        responder;
    bool               receiving = false;
    std::vector<float> resampled;
    std::vector<float> reference;
    std::vector<float> silence;
};

// Frames without playback that close a run of it, 0.5 s of client frames: a
// shorter gap is a pause between two sentences of the same answer.
#define AEC_QUIET_FRAMES 25

static void conn_send(Connection * conn, std::string frame) {
    std::lock_guard<std::mutex> lock(conn->out_mutex);
    if (!conn->out_stop) {
        conn->out.push(SendData{std::move(frame), false});
        conn->out_cv.notify_one();
    }
}

// send binary
static void conn_send(Connection * conn, const float* frame, int samples) {
    std::lock_guard<std::mutex> lock(conn->out_mutex);
    if (!conn->out_stop) {
        auto pcm16 = rt_float_to_pcm16(frame, samples);
        conn->out.push(SendData{std::move(pcm16), true});
        conn->out_cv.notify_one();
    }
}

// A failure goes to the client and to the log alike: the client sees what
// went wrong, and the log keeps it. The messages carry the tag of the module
// that failed.
static void conn_error(Connection * conn, const char * message) {
    s2s_log(S2S_LOG_WARN, "%s", message);
    conn_send(conn, rt_event_error(message));
}

static std::string http_route(const std::string & base_url, const std::string & suffix) {
    std::string host;
    std::string path;
    if (!http_split_url(base_url, host, path)) {
        return suffix;
    }
    while (!path.empty() && path.back() == '/') {
        path.pop_back();
    }
    if (path.empty() || path == "/") {
        return "/v1" + suffix;
    }
    if (path.size() >= 3 && path.compare(path.size() - 3, 3, "/v1") == 0) {
        return path + suffix;
    }
    return path + "/v1" + suffix;
}

static void wav_le16(std::string & out, size_t pos, uint16_t value) {
    out[pos + 0] = (char) (value & 0xff);
    out[pos + 1] = (char) ((value >> 8) & 0xff);
}

static void wav_le32(std::string & out, size_t pos, uint32_t value) {
    out[pos + 0] = (char) (value & 0xff);
    out[pos + 1] = (char) ((value >> 8) & 0xff);
    out[pos + 2] = (char) ((value >> 16) & 0xff);
    out[pos + 3] = (char) ((value >> 24) & 0xff);
}

static std::string wav_encode_pcm16(const std::vector<float> & samples, int sample_rate) {
    (void) sample_rate;
    const uint32_t n_samples = (uint32_t) samples.size();
    const uint32_t data_size = n_samples * 2u;
    std::string out(44 + data_size, '\0');
    memcpy(out.data(), "RIFF", 4);
    wav_le32(out, 4, 36u + data_size);
    memcpy(out.data() + 8, "WAVE", 4);
    memcpy(out.data() + 12, "fmt ", 4);
    wav_le32(out, 16, 16u);
    wav_le16(out, 20, 1u);
    wav_le16(out, 22, 1u);
    wav_le32(out, 24, (uint32_t) sample_rate);
    wav_le32(out, 28, (uint32_t) sample_rate * 2u);
    wav_le16(out, 32, 2u);
    wav_le16(out, 34, 16u);
    memcpy(out.data() + 36, "data", 4);
    wav_le32(out, 40, data_size);

    for (uint32_t i = 0; i < n_samples; ++i) {
        float v = samples[i];
        if (v < -1.0f) {
            v = -1.0f;
        } else if (v > 1.0f) {
            v = 1.0f;
        }
        const int16_t s = (int16_t) lrintf(v * 32767.0f);
        const uint16_t u = (uint16_t) s;
        out[44 + i * 2 + 0] = (char) (u & 0xff);
        out[44 + i * 2 + 1] = (char) ((u >> 8) & 0xff);
    }
    return out;
}

static uint16_t wav_u16le(const std::string & data, size_t pos) {
    return (uint16_t) ((unsigned char) data[pos + 0] | ((unsigned char) data[pos + 1] << 8));
}

static uint32_t wav_u32le(const std::string & data, size_t pos) {
    return (uint32_t) ((unsigned char) data[pos + 0] |
                      ((unsigned char) data[pos + 1] << 8) |
                      ((unsigned char) data[pos + 2] << 16) |
                      ((unsigned char) data[pos + 3] << 24));
}

static std::vector<float> wav_decode_pcm16(const std::string & bytes) {
    std::vector<float> out;
    if (bytes.size() < 44 || memcmp(bytes.data(), "RIFF", 4) != 0 || memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        return out;
    }
    size_t pos = 12;
    uint16_t channels = 1;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 16;
    size_t data_offset = 0;
    size_t data_size = 0;
    while (pos + 8 <= bytes.size()) {
        const std::string tag(bytes.data() + pos, bytes.data() + pos + 4);
        const uint32_t size = wav_u32le(bytes, pos + 4);
        pos += 8;
        if (tag == "fmt ") {
            if (pos + 16 <= bytes.size()) {
                const uint16_t format_tag = wav_u16le(bytes, pos + 0);
                channels = wav_u16le(bytes, pos + 2);
                sample_rate = wav_u32le(bytes, pos + 4);
                bits_per_sample = wav_u16le(bytes, pos + 14);
                if (format_tag != 1) {
                    return out;
                }
            }
        } else if (tag == "data") {
            data_offset = pos;
            data_size = size;
            break;
        }
        pos += size + (size & 1u);
    }
    if (data_offset == 0 || data_size == 0 || bits_per_sample != 16) {
        return out;
    }
    const size_t n_frames = data_size / (size_t) (channels * 2);
    out.reserve(n_frames);
    for (size_t i = 0; i < n_frames; ++i) {
        int32_t sum = 0;
        for (uint16_t ch = 0; ch < channels; ++ch) {
            const size_t off = data_offset + (i * channels + ch) * 2;
            if (off + 1 >= bytes.size()) {
                return out;
            }
            const int16_t sample = (int16_t) ((uint16_t) ((unsigned char) bytes[off + 0]) |
                                              ((uint16_t) (unsigned char) bytes[off + 1] << 8));
            sum += sample;
        }
        const float v = (float) sum / (float) (channels * 32768);
        out.push_back(v);
    }
    (void) sample_rate;
    return out;
}

static std::vector<float> wav_decode_any(const std::string & bytes) {
    if (bytes.size() >= 12 && memcmp(bytes.data(), "RIFF", 4) == 0) {
        return wav_decode_pcm16(bytes);
    }
    if (bytes.size() % 2 != 0) {
        return {};
    }
    std::vector<float> out(bytes.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        const int16_t sample = (int16_t) ((uint16_t) ((unsigned char) bytes[i * 2 + 0]) |
                                          ((uint16_t) (unsigned char) bytes[i * 2 + 1] << 8));
        out[i] = (float) sample / 32768.0f;
    }
    return out;
}

static std::string http_error_body(const std::string & body) {
    yyjson_doc * doc = yyjson_read(body.c_str(), body.size(), 0);
    if (!doc) {
        return body;
    }
    yyjson_val * error = yyjson_obj_get(yyjson_doc_get_root(doc), "error");
    yyjson_val * message = nullptr;
    if (error && yyjson_is_obj(error)) {
        message = yyjson_obj_get(error, "message");
    }
    if (!message) {
        message = yyjson_obj_get(yyjson_doc_get_root(doc), "text");
    }
    std::string out;
    if (message && yyjson_is_str(message)) {
        out.assign(yyjson_get_str(message), yyjson_get_len(message));
    }
    yyjson_doc_free(doc);
    return out.empty() ? body : out;
}

static bool conn_remote_asr(Connection * conn, const std::vector<float> & pcm, std::string & transcript) {
    if (conn->setup->asr_url.empty()) {
        return false;
    }
    std::string host;
    std::string path;
    if (!http_split_url(conn->setup->asr_url, host, path)) {
        conn_error(conn, "[ASR] Remote ASR URL has no scheme");
        return false;
    }
    std::unique_ptr<httplib::Client> http = http_open(host, "[ASR]");
    if (!http) {
        return false;
    }
    const std::string endpoint = http_route(conn->setup->asr_url, "/audio/transcriptions");
    httplib::UploadFormDataItems items;
    httplib::UploadFormData file;
    file.name = "file";
    file.filename = "audio.wav";
    file.content_type = "audio/wav";
    file.content = wav_encode_pcm16(pcm, S2S_MODEL_RATE);
    items.push_back(file);
    if (!conn->setup->asr_model.empty()) {
        httplib::UploadFormData model;
        model.name = "model";
        model.content = conn->setup->asr_model;
        items.push_back(model);
    }
    if (!conn->client.tts.language.empty() && conn->client.tts.language != "auto") {
        httplib::UploadFormData language;
        language.name = "language";
        language.content = conn->client.tts.language;
        items.push_back(language);
    }

    httplib::Headers headers = httplib::Headers();
    if (!conn->setup->asr_api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + conn->setup->asr_api_key);
        headers.emplace("api-key", conn->setup->asr_api_key);
    }
    const httplib::Result result = http->Post(endpoint, headers, items);
    if (!result) {
        const std::string err = std::string("[ASR] ") + http_error(result);
        conn_error(conn, err.c_str());
        return false;
    }
    if (result->status != 200) {
        const std::string err = "[ASR] " + std::to_string(result->status) + " " + http_error_body(result->body);
        conn_error(conn, err.c_str());
        return false;
    }

    yyjson_doc * doc = yyjson_read(result->body.c_str(), result->body.size(), 0);
    if (doc) {
        yyjson_val * text = yyjson_obj_get(yyjson_doc_get_root(doc), "text");
        if (text && yyjson_is_str(text)) {
            transcript.assign(yyjson_get_str(text), yyjson_get_len(text));
        }
        yyjson_doc_free(doc);
    }
    if (transcript.empty()) {
        transcript = result->body;
    }
    while (!transcript.empty() && (transcript.front() == '\n' || transcript.front() == '\r' || transcript.front() == ' ')) {
        transcript.erase(transcript.begin());
    }
    while (!transcript.empty() && (transcript.back() == '\n' || transcript.back() == '\r' || transcript.back() == ' ')) {
        transcript.pop_back();
    }
    return !transcript.empty();
}

static bool conn_remote_tts(Connection * conn, const SentenceUnit & unit, const tts_request & tts,
                            std::vector<float> & pcm) {
    if (conn->setup->tts_url.empty()) {
        return false;
    }
    std::string host;
    std::string path;
    if (!http_split_url(conn->setup->tts_url, host, path)) {
        conn_error(conn, "[TTS] Remote TTS URL has no scheme");
        return false;
    }
    std::unique_ptr<httplib::Client> http = http_open(host, "[TTS]");
    if (!http) {
        return false;
    }

    std::string body;
    {
        yyjson_mut_doc * doc = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val * root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add_str(doc, root, "input", unit.text.c_str());
        if (!tts.voice.empty()) {
            yyjson_mut_obj_add_str(doc, root, "voice", tts.voice.c_str());
        }
        if (!tts.language.empty() && tts.language != "auto") {
            yyjson_mut_obj_add_str(doc, root, "language", tts.language.c_str());
        }
        if (conn->setup->tts_model.empty()) {
            yyjson_mut_obj_add_str(doc, root, "model", conn->setup->tts_model.c_str());
        }
        yyjson_mut_obj_add_real(doc, root, "speed", 1.0);
        char * json = yyjson_mut_write(doc, 0, nullptr);
        body = json ? json : "{}";
        free(json);
        yyjson_mut_doc_free(doc);
    }

    httplib::Headers headers;
    if (!conn->setup->tts_api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + conn->setup->tts_api_key);
        headers.emplace("api-key", conn->setup->tts_api_key);
    }
    const std::string endpoint = http_route(conn->setup->tts_url, "/audio/speech");

    std::string streamed;
    auto receiver = [&](const char * data, size_t len) -> bool {
        if (len == 0) {
            return true;
        }
        streamed.append(data, len);
        return true;
    };

    const httplib::Result result = http->Post(endpoint, headers, body, "application/json", receiver);
    if (!result) {
        const std::string err = std::string("[TTS] ") + http_error(result);
        conn_error(conn, err.c_str());
        return false;
    }
    if (result->status != 200) {
        const std::string err = "[TTS] " + std::to_string(result->status) + " " + http_error_body(result->body);
        conn_error(conn, err.c_str());
        return false;
    }

    pcm = wav_decode_any(streamed);
    if (pcm.empty()) {
        conn_error(conn, "[TTS] remote endpoint returned no decodable audio");
        return false;
    }
    return true;
}

static bool conn_remote_tts_stream(Connection * conn, const SentenceUnit & unit, const tts_request & tts,
                                  tts_chunk_cb cb, void * user, const std::atomic<bool> * cancel) {
    if (conn->setup->tts_url.empty()) {
        return false;
    }
    std::string host;
    std::string path;
    if (!http_split_url(conn->setup->tts_url, host, path)) {
        conn_error(conn, "[TTS] Remote TTS URL has no scheme");
        return false;
    }
    std::unique_ptr<httplib::Client> http = http_open(host, "[TTS]");
    if (!http) {
        return false;
    }

    std::string body;
    {
        yyjson_mut_doc * doc = yyjson_mut_doc_new(nullptr);
        yyjson_mut_val * root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add_str(doc, root, "input", unit.text.c_str());
        if (!tts.voice.empty()) {
            yyjson_mut_obj_add_str(doc, root, "voice", tts.voice.c_str());
        }
        if (!tts.language.empty() && tts.language != "auto") {
            yyjson_mut_obj_add_str(doc, root, "language", tts.language.c_str());
        }
        if (!conn->setup->tts_model.empty()) {
            yyjson_mut_obj_add_str(doc, root, "model", conn->setup->tts_model.c_str());
        }
        yyjson_mut_obj_add_str(doc, root, "response_format", "wav");
        yyjson_mut_obj_add_real(doc, root, "speed", 1.0);
        char * json = yyjson_mut_write(doc, 0, nullptr);
        body = json ? json : "{}";
        free(json);
        yyjson_mut_doc_free(doc);
    }

    httplib::Headers headers;
    if (!conn->setup->tts_api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + conn->setup->tts_api_key);
        headers.emplace("api-key", conn->setup->tts_api_key);
    }
    const std::string endpoint = http_route(conn->setup->tts_url, "/audio/speech");

    std::string pending;
    bool        first = true;
    auto receiver = [&](const char * data, size_t len) -> bool {
        if (!data || len == 0) {
            return !(cancel && cancel->load());
        }
        if (cancel && cancel->load()) {
            return false;
        }
        pending.append(data, len);

        std::vector<float> pcm = wav_decode_any(pending);
        if (!pcm.empty()) {
            if (first) {
                conn_send(conn, rt_event_spoken(unit.text, unit.end));
                first = false;
            }
            if (!cb || !cb(pcm.data(), pcm.size(), user)) {
                return false;
            }
            pending.clear();
        }
        return true;
    };

    const httplib::Result result = http->Post(endpoint, headers, body, "application/json", receiver);
    if (!result) {
        const std::string err = std::string("[TTS] ") + http_error(result);
        conn_error(conn, err.c_str());
        return false;
    }
    if (result->status != 200) {
        const std::string err = "[TTS] " + std::to_string(result->status) + " " + http_error_body(result->body);
        conn_error(conn, err.c_str());
        return false;
    }

    if (!pending.empty()) {
        std::vector<float> leftover = wav_decode_any(pending);
        if (!leftover.empty()) {
            if (first) {
                conn_send(conn, rt_event_spoken(unit.text, unit.end));
                first = false;
            }
            if (!cb || !cb(leftover.data(), leftover.size(), user)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<std::string> conn_tts_voices(const ConversationSetup* setup) {
    std::vector<std::string> ret;
    if (setup->tts_url.empty()) {
        return ret;
    }
    std::string host;
    std::string path;
    if (!http_split_url(setup->tts_url, host, path)) {
        s2s_log(S2S_LOG_ERROR, "[TTS] Remote TTS URL has no scheme");
        return ret;
    }
    std::unique_ptr<httplib::Client> http = http_open(host, "[TTS]");
    if (!http) {
        return ret;
    }
    httplib::Headers headers;
    if (!setup->tts_api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + setup->tts_api_key);
        headers.emplace("api-key", setup->tts_api_key);
    }
    const std::string endpoint = http_route(setup->tts_url, "/audio/voices?model=" + setup->tts_model);
    const httplib::Result result = http->Get(endpoint, headers);
    if (!result) {
        s2s_log(S2S_LOG_ERROR, "[TTS] Failed to fetch voices: %s", http_error(result).c_str());
        return ret;
    }
    if (result->status != 200) {
        s2s_log(S2S_LOG_ERROR, "[TTS] Failed to fetch voices: %d %s", result->status, http_error_body(result->body).c_str());
        return ret;
    }
    // Parse the JSON response to extract the list of voices
    yyjson_doc * doc = yyjson_read(result->body.c_str(), result->body.size(), 0);
    if (doc) {
        yyjson_val * voices = yyjson_obj_get(yyjson_doc_get_root(doc), "voices");
        if (voices && yyjson_is_arr(voices)) {
            yyjson_val * val;
            yyjson_arr_iter iter;
            yyjson_arr_iter_init(voices, &iter);
            while ((val = yyjson_arr_iter_next(&iter))) {
                if (yyjson_is_str(val)) {
                    ret.push_back(std::string(yyjson_get_str(val), yyjson_get_len(val)));
                }
            }
        }
        yyjson_doc_free(doc);
    }
    return ret;
}

// Runs the canceller over every complete hop and hands back in pcm what is
// clean so far. The canceller is one hop late and its hops of 256 samples do
// not line up with the client frames, so the cleaned stream trails the raw
// one by one to two hops, about 16 to 32 ms.
static void conn_cancel_echo(Connection * conn, std::vector<float> & pcm, const std::vector<float> & ref, bool played) {
    conn->aec_mic.insert(conn->aec_mic.end(), pcm.begin(), pcm.end());
    conn->aec_ref.insert(conn->aec_ref.end(), ref.begin(), ref.end());

    const size_t hop  = (size_t) lv_hop(conn->setup->models.aec);
    size_t       done = 0;
    pcm.clear();
    for (; done + hop <= conn->aec_mic.size(); done += hop) {
        const size_t base = pcm.size();
        pcm.resize(base + hop);
        Timer      t_hop;
        const bool failed =
            lv_process(conn->aec, conn->aec_mic.data() + done, conn->aec_ref.data() + done, pcm.data() + base) != 0;
        if (failed && !conn->aec_failing) {
            s2s_log(S2S_LOG_ERROR, "[AEC] %s", lv_last_error());
            conn_send(conn, rt_event_error(lv_last_error()));
        }
        conn->aec_failing = failed;

        const double ms = t_hop.ms();
        conn->aec_ms += ms;
        conn->aec_peak = ms > conn->aec_peak ? ms : conn->aec_peak;
        conn->aec_hops++;

        // pcm holds the output of the previous hop: it is weighed against the
        // microphone energy kept from the previous call, when that hop played.
        if (conn->aec_pending) {
            conn->aec_in += conn->aec_pending_in;
            for (size_t i = 0; i < hop; i++) {
                conn->aec_out += (double) pcm[base + i] * pcm[base + i];
            }
            conn->aec_played += hop;
        }
        conn->aec_pending    = false;
        conn->aec_pending_in = 0.0;
        for (size_t i = 0; i < hop; i++) {
            conn->aec_pending = conn->aec_pending || conn->aec_ref[done + i] != 0.0f;
            conn->aec_pending_in += (double) conn->aec_mic[done + i] * conn->aec_mic[done + i];
        }
    }
    conn->aec_mic.erase(conn->aec_mic.begin(), conn->aec_mic.begin() + (ptrdiff_t) done);
    conn->aec_ref.erase(conn->aec_ref.begin(), conn->aec_ref.begin() + (ptrdiff_t) done);

    conn->aec_quiet = played ? 0 : conn->aec_quiet + 1;
    if (conn->aec_played > 0 && conn->aec_quiet == AEC_QUIET_FRAMES) {
        s2s_log(S2S_LOG_INFO, "[AEC] Microphone %.1f dB above the cleaned signal over %.2fs of playback",
                10.0 * log10((conn->aec_in + 1e-12) / (conn->aec_out + 1e-12)),
                (double) conn->aec_played / S2S_MODEL_RATE);
        const double hop_ms = (double) hop * 1000.0 / S2S_MODEL_RATE;
        const double mean   = conn->aec_ms / (double) conn->aec_hops;
        s2s_log(S2S_LOG_INFO, "[Perf] AEC %zu hops, %.2f ms per hop, peak %.2f ms, %.1fx real time", conn->aec_hops,
                mean, conn->aec_peak, hop_ms / mean);
        conn->aec_in     = 0.0;
        conn->aec_out    = 0.0;
        conn->aec_played = 0;
        conn->aec_hops   = 0;
        conn->aec_ms     = 0.0;
        conn->aec_peak   = 0.0;
    }
}

// Wakes whatever waits on the turn: a final turn, a new one, a cancel or the
// end of the connection.
static void conn_turn_notify(Connection * conn) {
    std::lock_guard<std::mutex> lock(conn->turn_mutex);
    conn->turn_cv.notify_all();
}

// Ends the talking half: the answer in flight stops and the queued turns are
// dropped, nobody is left to hear them.
static void conn_stop(Connection * conn) {
    conn->cancel.store(true);
    {
        std::lock_guard<std::mutex> lock(conn->queue_mutex);
        conn->stop = true;
        conn->queue_cv.notify_one();
        conn->answers_cv.notify_one();
    }
    conn_turn_notify(conn);
}

// Sends the queued frames in order. A failed send means the client is gone,
// or stopped reading for longer than the write timeout: the answer in flight
// stops, nothing more is queued for it, and the connection is over. The
// reader closes it: it is the one thread that reads the socket, closing
// handshake included. It sees the stop with the next frame the client sends,
// or, from a client gone silent, when its read times out.
static void conn_writer(Connection * conn) {
    s2s_log_thread(("Writer-" + std::to_string(conn->id)).c_str());
    for (;;) {
        SendData frame;
        {
            std::unique_lock<std::mutex> lock(conn->out_mutex);
            conn->out_cv.wait(lock, [conn]() { return conn->out_stop || !conn->out.empty(); });
            if (conn->out_stop) {
                return;
            }
            frame = std::move(conn->out.front());
            conn->out.pop();
        }
        if (!conn->send(frame, conn->send_user)) {
            s2s_log(S2S_LOG_WARN, "[Server] Send failed, the client is gone or too slow");
            {
                std::lock_guard<std::mutex> lock(conn->out_mutex);
                conn->out_stop = true;
                std::queue<SendData>().swap(conn->out);
            }
            conn_stop(conn);
            return;
        }
    }
}

// Whether something opened after this turn and revision: a later turn, or
// the same turn resumed. The user spoke again, so an answer to it that
// nobody heard yet has lost its point. Called with turn_mutex held.
static bool conn_outdated(const Connection * conn, int turn_id, int revision) {
    return conn->open_turn > turn_id || (conn->open_turn == turn_id && conn->open_revision > revision);
}

// Whether the same turn went on past this revision: its audio comes again,
// whole, in a later commit.
static bool conn_revised(Connection * conn, int turn_id, int revision) {
    std::lock_guard<std::mutex> lock(conn->turn_mutex);
    return conn->open_turn == turn_id && conn->open_revision > revision;
}

// Tells the session that the answer to this turn and revision is ready to be
// heard, or over: a turn stays resumable until the user can hear something.
// Called with no other lock held, like every session call.
static void conn_release(Connection * conn, int turn_id, int revision) {
    std::lock_guard<std::mutex> lock(conn->session_mutex);
    s2s_session_release(conn->session, turn_id, revision);
}

// Holds the first unit of the answer until its turn is final. Returns false
// when the floor went back meanwhile, or when something opened after it.
static bool conn_wait_final(Connection * conn) {
    std::unique_lock<std::mutex> lock(conn->turn_mutex);
    conn->turn_cv.wait(lock, [conn]() {
        return conn->stop || conn->cancel.load() || conn->final_turn >= conn->answer_turn ||
               conn_outdated(conn, conn->answer_turn, conn->answer_revision);
    });
    if (!conn->cancel.load() && conn_outdated(conn, conn->answer_turn, conn->answer_revision)) {
        s2s_log(S2S_LOG_INFO, "[Turn] Answer to turn %d rev %d voided by turn %d rev %d before its first unit",
                conn->answer_turn, conn->answer_revision, conn->open_turn, conn->open_revision);
        conn->cancel.store(true);
    }
    return !conn->stop && !conn->cancel.load();
}

// Speaks one unit and streams it to the client. The transcript of the unit
// goes out right before its first audio chunk, so the client only ever sees
// text that has sound behind it, and places it in the audio stream. Returns
// false when the floor was taken back.
static bool conn_speak(Connection * conn, const SentenceUnit & unit, const tts_request & tts) {
    if (!conn->answer_released) {
        conn_release(conn, conn->answer_turn, conn->answer_revision);
        if (!conn_wait_final(conn)) {
            return false;
        }
        conn->answer_released = true;
    }

    struct SpeakTap {
        Connection *         conn    = nullptr;
        const SentenceUnit * unit    = nullptr;
        bool                 jarvis  = false;
        size_t               samples = 0;
    } speak_tap;

    speak_tap.conn   = conn;
    speak_tap.unit   = &unit;
    speak_tap.jarvis = tts.effect == "jarvis";
    s2s_log(S2S_LOG_INFO, "[TTS] Speaking unit: %s", unit.text.c_str());
    bool spoke = false;
    if (!conn->setup->tts_url.empty()) {
        spoke = conn_remote_tts_stream(
            conn, unit, tts,
            [](const float * pcm, size_t n_samples, void * user) {
                SpeakTap * self = (SpeakTap *) user;
                if (self->samples == 0 && n_samples > 0) {
                    // rt_event_spoken is emitted by conn_remote_tts_stream when the first chunk arrives.
                }
                self->samples += n_samples;
                if (self->jarvis) {
                    std::vector<float> & out = self->conn->fx_pcm;
                    out.assign(pcm, pcm + n_samples);
                    jarvis_fx_process(self->conn->fx, out.data(), n_samples);
                    pcm = out.data();
                }
                conn_send(self->conn, pcm, (int) n_samples);
                return !self->conn->cancel.load();
            },
            &speak_tap, &conn->cancel);
    } else {
        spoke = tts_bridge_speak(
            conn->setup->models.tts, unit.text, tts,
            [](const float * pcm, size_t n_samples, void * user) {
                SpeakTap * self = (SpeakTap *) user;
                if (self->samples == 0 && n_samples > 0) {
                    conn_send(self->conn, rt_event_spoken(self->unit->text, self->unit->end));
                }
                self->samples += n_samples;
                if (self->jarvis) {
                    std::vector<float> & out = self->conn->fx_pcm;
                    out.assign(pcm, pcm + n_samples);
                    jarvis_fx_process(self->conn->fx, out.data(), n_samples);
                    pcm = out.data();
                }
                conn_send(self->conn, pcm, n_samples);
                return !self->conn->cancel.load();
            },
            &speak_tap, &conn->cancel);
    }

    if (!spoke && !conn->cancel.load()) {
        if (!conn->setup->tts_url.empty()) {
            conn_send(conn, rt_event_error("[TTS] remote synthesis failed"));
        } else {
            s2s_log(S2S_LOG_WARN, "[TTS] %s", tts_bridge_last_error());
            conn_send(conn, rt_event_error(tts_bridge_last_error()));
        }
    }

    return spoke;
}

// What set_voice changes: the voice of the answer in flight, which its next
// unit speaks with, and the one of the connection, which every later turn
// takes.
struct VoiceSwitch {
    Connection *  conn = nullptr;
    tts_request * tts  = nullptr;
};

// The session belongs to the client, so the client hears of the change: the
// next session.update it sends carries the new voice instead of the old one.
// A call without effect leaves the effect as it is.
static bool conn_set_voice(const std::string & arguments, void * user, std::string & result) {
    VoiceSwitch * self = (VoiceSwitch *) user;

    std::string  voice;
    std::string  effect;
    yyjson_doc * doc = yyjson_read(arguments.c_str(), arguments.size(), 0);
    if (doc) {
        voice  = rt_json_str(yyjson_doc_get_root(doc), "voice");
        effect = rt_json_str(yyjson_doc_get_root(doc), "effect");
        yyjson_doc_free(doc);
    }
    const std::vector<std::string> & voices = tts_bridge_voices(self->conn->setup->models.tts);
    if (std::find(voices.begin(), voices.end(), voice) == voices.end()) {
        s2s_set_error("[Agent] " VOICE_TOOL " knows no voice named \"%s\"", voice.c_str());
        return false;
    }

    if (!effect.empty() && !conn_effect_known(effect)) {
        s2s_set_error("[Agent] " VOICE_TOOL " knows no effect named \"%s\"", effect.c_str());
        return false;
    }

    self->tts->voice = voice;
    if (!effect.empty()) {
        self->tts->effect = effect;
    }
    {
        std::lock_guard<std::mutex> lock(self->conn->client_mutex);
        self->conn->client.tts.voice  = voice;
        self->conn->client.tts.effect = self->tts->effect;
    }
    conn_send(self->conn, rt_event_voice(voice, self->tts->effect));
    s2s_log(S2S_LOG_INFO, "[Agent] Voice set to %s, effect %s", voice.c_str(), self->tts->effect.c_str());

    result = "Voice set to " + voice + ", effect " + self->tts->effect;
    return true;
}

// What sleep changes: from the turn after this one, the connection answers
// none until a wake phrase is heard again, and none at all for the rest of
// the session while the gate is off, with no phrase to wake it.
static bool conn_sleep(const std::string &, void * user, std::string & result) {
    Connection * conn = (Connection *) user;
    bool         off  = false;
    {
        std::lock_guard<std::mutex> lock(conn->client_mutex);
        conn->awake = false;
        off         = conn->client.wake.mode == "off";
    }
    s2s_log(S2S_LOG_INFO, "[Wake] The model put the assistant to sleep%s", off ? " for the rest of the session" : "");
    result = off ? "Asleep for the rest of the conversation" : "Asleep until the user says a wake phrase";
    return true;
}

static bool conn_time(const std::string &, void * user, std::string & result) {
    time_t now = time(nullptr);
    result = ctime(&now);
    return true;
}

llm_tool conn_time_tool() {
    static const char* TIME_TOOL = "time";       // and built-in time tool
    static const std::string about = u8"获取当前时间, 当查询时间或日期时使用";

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "type", "function");
    yyjson_mut_val * function = yyjson_mut_obj_add_obj(doc, root, "function");
    yyjson_mut_obj_add_str(doc, function, "name", TIME_TOOL);
    yyjson_mut_obj_add_strn(doc, function, "description", about.c_str(), about.size());
    yyjson_mut_val * parameters = yyjson_mut_obj_add_obj(doc, function, "parameters");
    yyjson_mut_obj_add_str(doc, parameters, "type", "object");
    yyjson_mut_obj_add_obj(doc, parameters, "properties");

    char *   json = yyjson_mut_write(doc, 0, nullptr);
    llm_tool tool = { TIME_TOOL, json ? json : "{}" };
    free(json);
    yyjson_mut_doc_free(doc);
    return tool;
}

std::vector<llm_agent_builtin> conn_builtin_tools(Connection * conn, const ConversationSetup* setup) {
    std::vector<llm_agent_builtin> builtins;
    if (conn) {
        VoiceSwitch voice_switch = { conn, &conn->client.tts };
        builtins.push_back({ conn_voice_tool(conn->setup->models.tts), conn_set_voice, &voice_switch });
        builtins.push_back({ conn_sleep_tool(),                        conn_sleep,     conn          });
        builtins.push_back({ conn_time_tool(),                         conn_time,      conn          });
    } else if(setup) {
        builtins.push_back({ conn_voice_tool(setup->models.tts) });
        builtins.push_back({ conn_sleep_tool() });
        builtins.push_back({ conn_time_tool() });
    }
    return builtins;
}

// Whether a turn reaches the model, in the conversation and agentic modes.
// Asleep, only the turn that wakes the assistant gets through, and none with
// the gate off; a turn that got through keeps doing so through its
// revisions, so a wake phrase followed by a question in the same breath is
// answered as a whole. With the gate on, a turn that holds a sleep phrase
// gets through and puts the assistant to sleep for the turns after it.
static bool conn_admit(Connection * conn, int turn_id, const std::string & transcript) {
    std::lock_guard<std::mutex> lock(conn->client_mutex);
    const WakeSettings &        wake = conn->client.wake;
    if (conn->client.mode == "loopback") {
        return true;
    }
    if (turn_id != conn->admitted_turn) {
        if (!conn->awake) {
            if (wake.mode == "off" || !wake_heard(transcript, wake.phrases, wake.mode == "alone")) {
                return false;
            }
            conn->awake = true;
            s2s_log(S2S_LOG_INFO, "[Wake] Turn %d woke the assistant", turn_id);
        }
        conn->admitted_turn = turn_id;
    }
    if (wake.mode != "off" && conn->awake && wake_heard(transcript, wake.sleep_phrases, false)) {
        conn->awake = false;
        s2s_log(S2S_LOG_INFO, "[Wake] Turn %d puts the assistant to sleep after its answer", turn_id);
    }
    return true;
}

// The endpoint client of the connection, created on its first turn and
// pointed at the settings of every turn after.
static llm_client * conn_llm(Connection * conn, const llm_client_params & params) {
    if (!conn->llm) {
        conn->llm = llm_client_new(params);
        return conn->llm;
    }
    return llm_client_set_params(conn->llm, params) ? conn->llm : nullptr;
}

// Appends a message, joining two user messages in a row into one: a turn
// whose answer was never heard is followed by the next one, and some chat
// templates refuse two user messages in a row.
static void llm_push(std::vector<llm_message> & messages, const llm_message & message) {
    if (message.role == "user" && !messages.empty() && messages.back().role == "user") {
        messages.back().content += " " + message.content;
    } else {
        messages.push_back(message);
    }
}

// Turn audio in, transcript out, and on to the responder. A turn this ends
// without a transcript is released here, like one whose answer is over.
static void conn_recognize(Connection * conn, const TurnAudio & turn) {
    const std::vector<float> & pcm = turn.pcm;

    // A later revision of the same turn carries this audio and more: only the
    // last one is worth recognizing.
    if (conn_revised(conn, turn.turn_id, turn.revision)) {
        s2s_log(S2S_LOG_INFO, "[Turn] Turn %d rev %d resumed before its recognition", turn.turn_id, turn.revision);
        conn_release(conn, turn.turn_id, turn.revision);
        return;
    }

    Timer t_asr;

    std::string transcript;
    if (!conn->setup->asr_url.empty()) {
        if (!conn_remote_asr(conn, pcm, transcript)) {
            conn_release(conn, turn.turn_id, turn.revision);
            return;
        }
    } else {
        qa_transcribe_params asr_params = qa_transcribe_default_params();
        char *               text       = nullptr;
        qa_status            status;
        {
            std::lock_guard<std::mutex> asr_lock(conn->setup->models.asr_mutex);
            status = qa_transcribe(conn->setup->models.asr, pcm.data(), pcm.size(), S2S_MODEL_RATE, &asr_params, &text);
        }
        if (status != QA_STATUS_OK) {
            conn_error(conn, qa_last_error());
            conn_release(conn, turn.turn_id, turn.revision);
            return;
        }
        transcript = text;
        qa_free_text(text);
    }

    s2s_log(S2S_LOG_INFO, "[Perf] Recognize %.1f ms (%.2fs of audio)", t_asr.ms(),
            (double) pcm.size() / S2S_MODEL_RATE);

    if (transcript.empty()) {
        s2s_log(S2S_LOG_INFO, "[Turn] Empty transcript, nothing to answer");
        conn_release(conn, turn.turn_id, turn.revision);
        return;
    }

    // Asleep, a turn without a wake phrase goes nowhere: not to the model,
    // not to the client.
    if (!conn_admit(conn, turn.turn_id, transcript)) {
        s2s_log(S2S_LOG_INFO, "[Wake] Turn %d rev %d unheard while asleep", turn.turn_id, turn.revision);
        conn_release(conn, turn.turn_id, turn.revision);
        return;
    }

    s2s_log(S2S_LOG_INFO, "[Turn] Heard %zu characters in %.2fs of audio (turn %d rev %d)", transcript.size(),
            (double) pcm.size() / S2S_MODEL_RATE, turn.turn_id, turn.revision);
    conn_send(conn, rt_event_transcript("turn_" + std::to_string(turn.turn_id), transcript));

    AnswerJob job;
    job.transcript = transcript;
    job.turn_id    = turn.turn_id;
    job.revision   = turn.revision;

    std::lock_guard<std::mutex> lock(conn->queue_mutex);
    conn->answers.push(std::move(job));
    conn->answers_cv.notify_one();
}

// Transcript in, answer spoken out.
static void conn_answer(Connection * conn, const AnswerJob & job) {
    const std::string & transcript = job.transcript;
    const std::string   item       = "turn_" + std::to_string(job.turn_id);

    // Whatever way the answer ends here, the responder is done with it.
    struct TurnRelease {
        Connection * conn;
        int          turn_id;
        int          revision;

        ~TurnRelease() { conn_release(conn, turn_id, revision); }
    } release = { conn, job.turn_id, job.revision };

    if (conn->stop) {
        return;
    }

    // A cancel raised from here on belongs to this answer, and its effect
    // starts from silence: nothing of a cut answer echoes into it.
    conn->cancel.store(false);
    jarvis_fx_reset(conn->fx);

    // The settings and the list as they stand now: a change that arrives
    // later belongs to the next turn, not to this one.
    ClientSettings client;
    {
        std::lock_guard<std::mutex> lock(conn->client_mutex);
        client = conn->client;
    }

    // The answer in flight, published with the floor under the turn lock: from
    // here on a turn or a revision that opens voids it at once. If the user
    // spoke again during the recognition, it is void before it starts, and
    // still opens and closes like any other, so the client files the turn in
    // its conversation all the same.
    {
        std::lock_guard<std::mutex> lock(conn->turn_mutex);
        conn->answer_turn     = job.turn_id;
        conn->answer_revision = job.revision;
        conn->answer_released = false;
        conn->speaking.store(true);
        if (conn_outdated(conn, job.turn_id, job.revision)) {
            conn->cancel.store(true);
        }
    }
    conn_send(conn, rt_event("response.created"));

    std::string answer;

    if (conn->cancel.load()) {
        s2s_log(S2S_LOG_INFO, "[Turn] Turn %d superseded or cancelled before its answer", job.turn_id);
    } else if (client.mode == "loopback") {
        // No endpoint in the path: the recognized text is the answer.
        answer = transcript;

        // A couple of characters is not speech, it is the tail of a noise the
        // recognizer had to name: spoken back, it sends the talker off its
        // distribution. Characters, not bytes: an accented letter is one.
        int n_chars = 0;
        for (const char c : answer) {
            n_chars += ((unsigned char) c & 0xC0) != 0x80;
        }
        if (n_chars < client.tts.guards.min_chars) {
            s2s_log(S2S_LOG_INFO, "[TTS] Skipped %d characters, fewer than %d", n_chars, client.tts.guards.min_chars);
        } else {
            SentenceUnit unit;
            unit.text = answer;
            unit.end  = sentence_utf16_len(answer);
            conn_speak(conn, unit, client.tts);
        }
    } else {
        std::vector<llm_message> messages;
        if (!client.system_prompt.empty()) {
            messages.push_back({ "system", client.system_prompt });
        }
        // An earlier revision of this turn may sit in the list, pushed when
        // its answer closed unheard: the transcript below replaces it.
        for (const rt_message & message : client.history) {
            if (message.role != "user" || message.item != item) {
                llm_push(messages, { message.role, message.content });
            }
        }
        llm_push(messages, { "user", transcript });

        if (client.llm.base_url.empty()) {
            conn_error(conn, "[Realtime] No endpoint: name one in the session, or start the server with --llm-url");
        } else if (llm_client * llm = conn_llm(conn, client.llm); !llm) {
            conn_error(conn, llm_client_last_error());
        } else {
            struct StreamTap {
                Connection *        conn = nullptr;
                const tts_request * tts  = nullptr;
                SentenceSplitter    splitter;
                Timer               timer;
                double              first_unit_ms = -1.0;
            } stream_tap;

            stream_tap.conn = conn;
            stream_tap.tts  = &client.tts;

            // What the model writes, as it writes it. The transcript event
            // that follows says what is really spoken, one unit later.
            llm_delta_cb on_delta = [](const char * delta, void * user) {
                StreamTap * self = (StreamTap *) user;

                conn_send(self->conn, rt_event_text("response.output_text.delta", "delta", delta));

                for (const SentenceUnit & unit : sentence_split_push(&self->splitter, delta)) {
                    if (self->first_unit_ms < 0.0) {
                        self->first_unit_ms = self->timer.ms();
                    }
                    if (!conn_speak(self->conn, unit, *self->tts)) {
                        return false;
                    }
                }
                return !self->conn->cancel.load();
            };

            // The calls travel between the deltas, at their place in the
            // text.
            llm_agent_call_cb on_call = [](const std::string & name, bool done, bool ok, double ms, size_t bytes,
                                           void * user) {
                StreamTap * self = (StreamTap *) user;
                conn_send(self->conn, done ? rt_event_tool_done(name, ok, ms, bytes) : rt_event_tool_started(name));
            };

            // The MCP sessions of the agent live with the connection and
            // follow its list of servers: a changed list, or a changed tool
            // timeout, gets fresh ones.
            if (client.mode == "agentic") {
                std::vector<mcp_server_params> servers = client.mcp;
                for (mcp_server_params & server : servers) {
                    server.timeout_sec = client.llm.tool_timeout_sec;
                }
                if (!conn->agent || conn->agent_servers != servers) {
                    llm_agent_free(conn->agent);
                    conn->agent         = llm_agent_new(servers);
                    conn->agent_servers = servers;
                }
            }

            // The built-in tools act on this answer and this connection.
            auto builtins = conn_builtin_tools(conn, conn->setup);

            Timer      t_llm;
            const bool streamed =
                client.mode == "agentic" ?
                    llm_agent_run(conn->agent, builtins, llm, client.llm, client.tools, client.max_rounds, messages,
                                  on_delta, on_call, &stream_tap, &conn->cancel, answer) :
                    llm_client_stream(llm, messages, on_delta, &stream_tap, &conn->cancel, answer);

            // The tail is a unit like the others: a one sentence answer has
            // no other, and its time is the time to the first unit.
            const SentenceUnit tail = sentence_split_flush(&stream_tap.splitter);
            if (streamed && !tail.text.empty()) {
                if (stream_tap.first_unit_ms < 0.0) {
                    stream_tap.first_unit_ms = stream_tap.timer.ms();
                }
                conn_speak(conn, tail, client.tts);
            }

            if (stream_tap.first_unit_ms < 0.0) {
                s2s_log(S2S_LOG_INFO, "[Perf] Respond %.1f ms (no unit, %zu characters)", t_llm.ms(), answer.size());
            } else {
                s2s_log(S2S_LOG_INFO, "[Perf] Respond %.1f ms (first unit %.1f ms, %zu characters)", t_llm.ms(),
                        stream_tap.first_unit_ms, answer.size());
            }

            if (!streamed && !conn->cancel.load()) {
                conn_error(conn, llm_client_last_error());
            } else if (streamed && answer.empty()) {
                // A model that ends on its first token: the client hears
                // nothing and would not know why, and the turn stays
                // unanswered so the next request does not carry a silence
                // for the model to imitate.
                conn_error(conn, "[LLM] The model answered nothing");
            }
        }
    }

    // The echo of the last syllable, heard when the answer ends on its own.
    if (client.tts.effect == "jarvis" && conn->answer_released && !conn->cancel.load()) {
        std::vector<float> tail(jarvis_fx_tail(conn->fx), 0.0f);
        jarvis_fx_process(conn->fx, tail.data(), tail.size());
        conn_send(conn, rt_event_audio(tail.data(), tail.size()));
    }

    s2s_log(S2S_LOG_INFO, "[Turn] Answered %zu characters%s", answer.size(), conn->cancel.load() ? ", cut short" : "");

    // Every response.created ends here, with exactly one of the two.
    conn->speaking.store(false);
    conn_send(conn, rt_event(conn->cancel.load() ? "response.cancelled" : "response.done"));
}

static void conn_recognizer(Connection * conn) {
    s2s_log_thread(("Recognizer-" + std::to_string(conn->id)).c_str());
    for (;;) {
        TurnAudio turn;
        {
            std::unique_lock<std::mutex> lock(conn->queue_mutex);
            conn->queue_cv.wait(lock, [conn]() { return conn->stop || !conn->queue.empty(); });
            if (conn->stop) {
                return;
            }
            turn = std::move(conn->queue.front());
            conn->queue.pop();
        }
        conn_recognize(conn, turn);
    }
}

static void conn_responder(Connection * conn) {
    s2s_log_thread(("Responder-" + std::to_string(conn->id)).c_str());
    for (;;) {
        AnswerJob job;
        {
            std::unique_lock<std::mutex> lock(conn->queue_mutex);
            conn->answers_cv.wait(lock, [conn]() { return conn->stop || !conn->answers.empty(); });
            if (conn->stop) {
                return;
            }
            job = std::move(conn->answers.front());
            conn->answers.pop();
        }
        conn_answer(conn, job);
    }
}

// The latest turn and revision that opened. An answer in flight to anything
// older is void from this moment: its synthesis and its endpoint request
// stop now instead of at their first unit, and the responder is free for
// what just opened.
static void conn_opened(Connection * conn, int turn_id, int revision) {
    std::lock_guard<std::mutex> lock(conn->turn_mutex);
    if (turn_id > conn->open_turn || (turn_id == conn->open_turn && revision > conn->open_revision)) {
        conn->open_turn     = turn_id;
        conn->open_revision = revision;
    }
    if (conn->speaking.load() && !conn->cancel.load() &&
        conn_outdated(conn, conn->answer_turn, conn->answer_revision)) {
        s2s_log(S2S_LOG_INFO,
                "[Turn] Answer to turn %d rev %d voided by turn %d rev %d, its synthesis and endpoint request stop",
                conn->answer_turn, conn->answer_revision, turn_id, revision);
        conn->cancel.store(true);
    }
    conn->turn_cv.notify_all();
}

// Events the state machine raises, on the reader thread.
static void conn_on_session_event(const s2s_session_report * report, void * user) {
    Connection * conn = (Connection *) user;

    switch (report->event) {
        case S2S_EVENT_SPEECH_STARTED:
            {
                s2s_log(S2S_LOG_INFO, "[Session] Speech started at %.2fs (turn %d rev %d)", report->time_sec,
                        report->turn_id, report->revision);
                conn_send(conn, rt_event("input_audio_buffer.speech_started"));
                conn_opened(conn, report->turn_id, report->revision);
                break;
            }

        case S2S_EVENT_TURN_RESUMED:
            {
                // The client sees the user talking again, like any start; the
                // answer to the previous revision is void.
                s2s_log(S2S_LOG_INFO, "[Session] Turn resumed at %.2fs (turn %d rev %d), its audio goes on",
                        report->time_sec, report->turn_id, report->revision);
                conn_send(conn, rt_event("input_audio_buffer.speech_started"));
                conn_opened(conn, report->turn_id, report->revision);
                break;
            }

        case S2S_EVENT_TURN_FINAL:
            {
                s2s_log(S2S_LOG_INFO, "[Session] Turn final at %.2fs (turn %d rev %d), its answer may be heard",
                        report->time_sec, report->turn_id, report->revision);
                std::lock_guard<std::mutex> lock(conn->turn_mutex);
                conn->final_turn = report->turn_id > conn->final_turn ? report->turn_id : conn->final_turn;
                conn->turn_cv.notify_all();
                break;
            }

        case S2S_EVENT_SPEECH_STOPPED:
            s2s_log(S2S_LOG_INFO, "[Session] Speech stopped at %.2fs (turn %d rev %d)", report->time_sec,
                    report->turn_id, report->revision);
            break;

        case S2S_EVENT_TURN_REOPENED:
            s2s_log(S2S_LOG_INFO, "[Session] Turn reopened at %.2fs (turn %d rev %d), completion %.3f",
                    report->time_sec, report->turn_id, report->revision, (double) report->turn_score);
            break;

        case S2S_EVENT_BARGE_IN:
            // The floor goes back to the user: the synthesis and the endpoint
            // request stop, the client flushes its playback on the
            // speech_started that follows, and the responder closes the
            // response.
            s2s_log(S2S_LOG_INFO, "[Session] Barge-in at %.2fs, the user took the floor", report->time_sec);
            conn->cancel.store(true);
            conn_turn_notify(conn);
            break;

        case S2S_EVENT_TURN_COMMITTED:
            {
                s2s_log(S2S_LOG_INFO,
                        "[Session] Turn committed at %.2fs (turn %d rev %d), %.2fs of audio kept of %.2fs, completion "
                        "%.3f, "
                        "grace %.2fs",
                        report->time_sec, report->turn_id, report->revision,
                        (double) report->n_samples / S2S_MODEL_RATE, (double) report->n_held / S2S_MODEL_RATE,
                        (double) report->turn_score, report->grace_sec);

                TurnAudio turn;
                turn.pcm.assign(report->pcm, report->pcm + report->n_samples);
                turn.turn_id  = report->turn_id;
                turn.revision = report->revision;

                std::lock_guard<std::mutex> lock(conn->queue_mutex);
                conn->queue.push(std::move(turn));
                conn->queue_cv.notify_one();
                break;
            }
    }
}

// Every session.update describes the whole session: a field it leaves out
// takes the server default, so clearing a field on the page brings the
// default back. The conversation has its own event and stays, and the echo
// canceller, a state with a learned path, only changes when a method is named.
static void conn_apply_patch(Connection * conn, const rt_session_patch & patch) {
    std::lock_guard<std::mutex> lock(conn->client_mutex);

    std::vector<rt_message> history = std::move(conn->client.history);
    const std::string       waking  = conn->client.wake.mode;
    conn->client                    = conn->setup->defaults;
    conn->client.history            = std::move(history);

    const s2s_session_params listening = conn->params;
    conn->params                       = s2s_session_params();

    if (!patch.mode.empty()) {
        conn->client.mode = patch.mode;
    }
    conn->client.tools = patch.tools;
    if (conn->setup->mcp_fixed) {
        if (!patch.mcp.empty()) {
            conn_error(conn, "[Realtime] The MCP servers are set by the server");
        }
    } else {
        conn->client.mcp.clear();
        for (const mcp_server_params & server : patch.mcp) {
            if (host_allowed(conn->setup->llm_hosts, server.url)) {
                conn->client.mcp.push_back(server);
            } else {
                conn_error(conn, ("[Realtime] MCP host " + url_host(server.url) + " is not allowed").c_str());
            }
        }
    }
    if (patch.max_rounds > 0) {
        conn->client.max_rounds = patch.max_rounds;
    }
    if (conn_wake_known(patch.wake_mode)) {
        conn->client.wake.mode = patch.wake_mode;
    } else if (!patch.wake_mode.empty()) {
        conn_error(conn, ("[Realtime] Unknown wake mode " + patch.wake_mode + ", every turn is answered").c_str());
    }
    if (!patch.wake_phrases.empty()) {
        conn->client.wake.phrases = patch.wake_phrases;
    }
    if (!patch.sleep_phrases.empty()) {
        conn->client.wake.sleep_phrases = patch.sleep_phrases;
    }
    if (conn->client.wake.mode != waking) {
        conn->awake = conn->client.wake.mode == "off";
        s2s_log(S2S_LOG_INFO, "[Wake] Mode %s%s", conn->client.wake.mode.c_str(),
                conn->client.wake.mode == "off" ? "" : ", asleep");
    }
    if (patch.tool_timeout_sec > 0) {
        conn->client.llm.tool_timeout_sec = patch.tool_timeout_sec;
    }
    if (!patch.echo.empty() && patch.echo != conn->echo) {
        // A fresh canceller learns the echo path of the new setup from
        // nothing.
        conn->echo = patch.echo;
        if (conn->aec) {
            s2s_log(S2S_LOG_INFO, "[AEC] Canceller off");
        }
        lv_state_free(conn->aec);
        conn->aec = conn->echo == "server" || conn->echo == "both" ? lv_state_new(conn->setup->models.aec) : nullptr;
        if (conn->aec) {
            s2s_log(S2S_LOG_INFO, "[AEC] Canceller on, fresh echo path");
        }
        // The microphone and the reference restart their resampling together:
        // two streams of equal length in give two of equal length out, hop
        // for hop.
        conn->aec_mic.clear();
        conn->aec_ref.clear();
        audio_resample_stream_init(&conn->mic_resample, S2S_INPUT_RATE, S2S_MODEL_RATE);
        audio_resample_stream_init(&conn->ref_resample, S2S_INPUT_RATE, S2S_MODEL_RATE);
        conn->aec_in         = 0.0;
        conn->aec_out        = 0.0;
        conn->aec_played     = 0;
        conn->aec_quiet      = 0;
        conn->aec_hops       = 0;
        conn->aec_ms         = 0.0;
        conn->aec_peak       = 0.0;
        conn->aec_pending_in = 0.0;
        conn->aec_pending    = false;
        conn->aec_failing    = false;
    }
    if (conn->setup->llm_fixed) {
        if (!patch.llm_url.empty() || !patch.llm_model.empty() || !patch.llm_key.empty()) {
            conn_error(conn, "[Realtime] The endpoint is set by the server");
        }
    } else {
        // A key travels with its URL: the URL of a patch takes the key of the
        // same patch, none when it carries none, so a key never reaches a
        // host it was not given for.
        if (!patch.llm_url.empty()) {
            if (host_allowed(conn->setup->llm_hosts, patch.llm_url)) {
                conn->client.llm.base_url = patch.llm_url;
                conn->client.llm.api_key  = patch.llm_key;
            } else {
                conn_error(conn, ("[Realtime] Endpoint host " + url_host(patch.llm_url) + " is not allowed").c_str());
            }
        }
        if (!patch.llm_model.empty()) {
            conn->client.llm.model = patch.llm_model;
        }
    }
    if (!patch.system_prompt.empty()) {
        conn->client.system_prompt = patch.system_prompt;
    }
    if (patch.temperature >= 0.0f) {
        conn->client.llm.sampling.temperature = patch.temperature;
    }
    if (patch.top_p >= 0.0f) {
        conn->client.llm.sampling.top_p = patch.top_p;
    }
    if (patch.top_k >= 0) {
        conn->client.llm.sampling.top_k = patch.top_k;
    }
    if (patch.min_p >= 0.0f) {
        conn->client.llm.sampling.min_p = patch.min_p;
    }
    if (patch.max_tokens >= 0) {
        conn->client.llm.sampling.max_tokens = patch.max_tokens;
    }
    if (patch.presence_penalty > -3.0f) {
        conn->client.llm.sampling.presence_penalty = patch.presence_penalty;
    }
    if (patch.frequency_penalty > -3.0f) {
        conn->client.llm.sampling.frequency_penalty = patch.frequency_penalty;
    }
    if (patch.seed >= 0) {
        conn->client.llm.sampling.seed = (int) patch.seed;
    }
    if (!patch.reasoning_effort.empty()) {
        conn->client.llm.sampling.reasoning_effort = patch.reasoning_effort;
    }
    if (!patch.tts_voice.empty()) {
        conn->client.tts.voice = patch.tts_voice;
    }
    if (conn_effect_known(patch.tts_effect)) {
        conn->client.tts.effect = patch.tts_effect;
    } else if (!patch.tts_effect.empty()) {
        conn_error(conn, ("[Realtime] Unknown effect " + patch.tts_effect + ", the voice keeps none").c_str());
    }
    if (!patch.tts_language.empty()) {
        conn->client.tts.language = patch.tts_language;
    }
    if (!patch.tts_browser_languages.empty()) {
        conn->client.tts.browser_languages = patch.tts_browser_languages;
    }
    if (patch.tts_min_chars >= 0) {
        conn->client.tts.guards.min_chars = patch.tts_min_chars;
    }
    if (patch.tts_chars_per_second > 0.0f) {
        conn->client.tts.guards.chars_per_second = patch.tts_chars_per_second;
    }
    if (patch.tts_margin_seconds >= 0.0f) {
        conn->client.tts.guards.margin_seconds = patch.tts_margin_seconds;
    }

    if (patch.llm_timeout_sec > 0) {
        conn->client.llm.timeout_sec = patch.llm_timeout_sec;
    }

    if (patch.tts_temperature >= 0.0f) {
        conn->client.tts.sampling.temperature = patch.tts_temperature;
    }
    if (patch.tts_top_k >= 0) {
        conn->client.tts.sampling.top_k = patch.tts_top_k;
    }
    if (patch.tts_top_p >= 0.0f) {
        conn->client.tts.sampling.top_p = patch.tts_top_p;
    }
    if (patch.tts_repetition_penalty >= 0.0f) {
        conn->client.tts.sampling.repetition_penalty = patch.tts_repetition_penalty;
    }
    if (patch.tts_subtalker_temperature >= 0.0f) {
        conn->client.tts.sampling.subtalker_temperature = patch.tts_subtalker_temperature;
    }
    if (patch.tts_subtalker_top_k >= 0) {
        conn->client.tts.sampling.subtalker_top_k = patch.tts_subtalker_top_k;
    }
    if (patch.tts_subtalker_top_p >= 0.0f) {
        conn->client.tts.sampling.subtalker_top_p = patch.tts_subtalker_top_p;
    }
    if (patch.tts_max_new_tokens >= 0) {
        conn->client.tts.sampling.max_new_tokens = patch.tts_max_new_tokens;
    }
    if (patch.tts_seed >= 0) {
        conn->client.tts.sampling.seed = patch.tts_seed;
    }

    if (patch.vad_neg_threshold >= 0.0f) {
        conn->params.vad_neg_threshold = patch.vad_neg_threshold;
    }
    if (patch.vad_threshold >= 0.0f) {
        conn->params.vad_threshold = patch.vad_threshold;
    }
    if (patch.min_speech_ms >= 0) {
        conn->params.min_speech_ms = patch.min_speech_ms;
    }
    if (patch.barge_in_ms >= 0) {
        conn->params.barge_in_ms = patch.barge_in_ms;
    }
    if (patch.min_silence_ms >= 0) {
        conn->params.min_silence_ms = patch.min_silence_ms;
    }
    if (patch.min_speech_continuation_ms >= 0) {
        conn->params.min_speech_continuation_ms = patch.min_speech_continuation_ms;
    }
    if (patch.speech_pad_ms >= 0) {
        conn->params.speech_pad_ms = patch.speech_pad_ms;
    }
    if (patch.turn_threshold >= 0.0f) {
        conn->params.turn_threshold = patch.turn_threshold;
    }
    if (patch.incomplete_delay_ms >= 0) {
        conn->params.incomplete_delay_ms = patch.incomplete_delay_ms;
    }
    if (patch.turn_max_wait_ms >= 0) {
        conn->params.turn_max_wait_ms = patch.turn_max_wait_ms;
    }
    if (patch.reopen_grace_ms >= 0) {
        conn->params.reopen_grace_ms = patch.reopen_grace_ms;
    }

    // Only a change of the listening thresholds reaches the session, which
    // keeps the turn it is in the middle of hearing.
    if (memcmp(&listening, &conn->params, sizeof(listening)) != 0) {
        std::lock_guard<std::mutex> session_lock(conn->session_mutex);
        s2s_session_set_params(conn->session, conn->params);
    }
}

Connection * conn_open(const ConversationSetup * setup, int id, conn_send_fn send, void * send_user) {
    Connection * conn = new Connection();
    conn->setup       = setup;
    conn->send        = send;
    conn->send_user   = send_user;
    conn->id          = id;
    audio_resample_stream_init(&conn->mic_resample, S2S_INPUT_RATE, S2S_MODEL_RATE);
    audio_resample_stream_init(&conn->ref_resample, S2S_INPUT_RATE, S2S_MODEL_RATE);
    conn->client  = setup->defaults;
    conn->fx      = jarvis_fx_new(S2S_INPUT_RATE);
    conn->session = s2s_session_new(setup->models.vad, setup->models.turn, conn->params, conn_on_session_event, conn);
    if (!conn->session) {
        s2s_log(S2S_LOG_WARN, "%s", sv_last_error());
        send({rt_event_error(sv_last_error()), false}, send_user);
        jarvis_fx_free(conn->fx);
        delete conn;
        return nullptr;
    }

    conn->writer     = std::thread(conn_writer, conn);
    conn->recognizer = std::thread(conn_recognizer, conn);
    conn->responder  = std::thread(conn_responder, conn);
    return conn;
}

void conn_append_audio(Connection * conn, const float* data, int samples, const float* reference) {
    if (!conn->receiving) {
        conn->receiving = true;
        s2s_log(S2S_LOG_INFO, "[Realtime] Microphone streaming, %d samples per frame", samples);
    }
    audio_resample_stream_push(&conn->mic_resample, data, samples, conn->resampled);
    if (conn->aec) {
        // A frame without reference is a frame where nothing played.
        bool played = reference != nullptr;
        if (!played) {
            conn->silence.assign(samples, 0.0f);
            reference = conn->silence.data();
        }
        audio_resample_stream_push(&conn->ref_resample, reference, samples, conn->reference);
        conn_cancel_echo(conn, conn->resampled, conn->reference, played);
    }
    {
        std::lock_guard<std::mutex> lock(conn->session_mutex);
        s2s_session_set_speaking(conn->session, conn->speaking.load());
        s2s_session_push(conn->session, conn->resampled.data(), conn->resampled.size());
    }
}

void conn_frame(Connection * conn, const std::string & frame) {
    const rt_client_message message = rt_parse(frame);
    switch (message.type) {
        case RT_CLIENT_SESSION_UPDATE:
            conn_apply_patch(conn, message.patch);
            if (!message.patch.invalid.empty()) {
                conn_error(conn, ("[Realtime] Session update: " + message.patch.invalid +
                                  " is not a number in its range, it keeps the server default")
                                     .c_str());
            }
            // The log is streamed to every page on /logs: it says
            // whether an endpoint is set, never which one.
            s2s_log(S2S_LOG_INFO, "[Realtime] Session update: mode %s, echo %s, endpoint %s, voice %s, language %s",
                    conn->client.mode.c_str(), conn->echo.c_str(), conn->client.llm.base_url.empty() ? "none" : "set",
                    conn->client.tts.voice.empty() ? "default" : conn->client.tts.voice.c_str(),
                    conn->client.tts.language.empty() ? "default" : conn->client.tts.language.c_str());
            conn_send(conn, rt_event("session.updated"));
            break;

        case RT_CLIENT_AUDIO_APPEND:
            conn_append_audio(conn, message.audio.data(), message.audio.size(),
                message.reference.size() == message.audio.size() ? message.reference.data() : nullptr);
            break;

        case RT_CLIENT_RESPONSE_CANCEL:
            s2s_log(S2S_LOG_INFO, "[Realtime] Response cancel");
            conn->cancel.store(true);
            conn_turn_notify(conn);
            break;

        case RT_CLIENT_HISTORY:
            {
                std::lock_guard<std::mutex> lock(conn->client_mutex);
                conn->client.history = message.messages;
                s2s_log(S2S_LOG_INFO, "[Realtime] History: %zu messages", conn->client.history.size());
            }
            break;

        case RT_CLIENT_UNKNOWN:
            s2s_log(S2S_LOG_WARN, "[Realtime] Unsupported event");
            conn_send(conn, rt_event_error("unsupported event"));
            break;
    }
}

bool conn_stopped(const Connection * conn) {
    return conn->stop;
}

void conn_close(Connection * conn) {
    conn_stop(conn);
    conn->recognizer.join();
    conn->responder.join();
    llm_client_free(conn->llm);
    llm_agent_free(conn->agent);
    {
        std::lock_guard<std::mutex> lock(conn->out_mutex);
        conn->out_stop = true;
        conn->out_cv.notify_one();
    }
    conn->writer.join();
    s2s_session_free(conn->session);
    lv_state_free(conn->aec);
    jarvis_fx_free(conn->fx);
    s2s_log(S2S_LOG_INFO, "[Server] Connection closed");
    delete conn;
}
