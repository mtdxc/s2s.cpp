// s2s-models.cpp: the models of the voice loop, found and loaded once

#include "s2s-models.h"

#include "model-find.h"
#include "s2s-error.h"

bool models_load(const std::string & models_dir,
                 const std::string & voices_dir,
                 const std::string & device,
                 const tts_engine &  engine,
                 ModelFiles &        files,
                 ConversationSetup & setup) {
    files.vad  = find_model(models_dir, "silero-vad", "");
    files.turn = find_model(models_dir, "smart-turn", "");
    files.aec  = find_model(models_dir, "localvqe", "");
    if (setup.asr_url.empty()) {
        files.asr = find_model(models_dir, "qwenasr-", "");
    }
    if (setup.tts_url.empty()) {
        files.talker = find_model(models_dir, "qwen-talker", "-base-");
        files.codec  = find_model(models_dir, "qwen-tokenizer", "");
    }

    if (files.vad.empty() || files.turn.empty() || files.aec.empty() ||
        (!setup.asr_url.empty() && !setup.tts_url.empty() ? false : false) ||
        (!setup.asr_url.empty() ? false : files.asr.empty()) ||
        (!setup.tts_url.empty() ? false : (files.talker.empty() || files.codec.empty()))) {
        s2s_log(S2S_LOG_ERROR, "[Load] FATAL: missing models in %s, run ./models.sh", models_dir.c_str());
        return false;
    }

    s2s_log(S2S_LOG_INFO, "[Load] VAD %s", files.vad.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] Turn %s", files.turn.c_str());
    s2s_log(S2S_LOG_INFO, "[Load] AEC %s", files.aec.c_str());
    if (setup.asr_url.empty()) {
        s2s_log(S2S_LOG_INFO, "[Load] ASR %s", files.asr.c_str());
    } else {
        s2s_log(S2S_LOG_INFO, "[Load] ASR remote %s", setup.asr_url.c_str());
    }
    if (setup.tts_url.empty()) {
        s2s_log(S2S_LOG_INFO, "[Load] TTS %s + %s, voices from %s", files.talker.c_str(), files.codec.c_str(),
                voices_dir.c_str());
    } else {
        s2s_log(S2S_LOG_INFO, "[Load] TTS remote %s", setup.tts_url.c_str());
    }

    ServerModels & models = setup.models;

    models.vad = sv_init(files.vad.c_str());
    if (!models.vad) {
        s2s_log(S2S_LOG_ERROR, "[Load] FATAL: %s", sv_last_error());
        return false;
    }

    models.turn = st_init(files.turn.c_str());
    if (!models.turn) {
        s2s_log(S2S_LOG_ERROR, "[Load] FATAL: %s", st_last_error());
        return false;
    }

    if (setup.asr_url.empty()) {
        qa_init_params asr_init = qa_init_default_params();
        asr_init.model_path     = files.asr.c_str();
        asr_init.device         = device.empty() ? nullptr : device.c_str();

        models.asr = qa_init(&asr_init);
        if (!models.asr) {
            s2s_log(S2S_LOG_ERROR, "[Load] FATAL: %s", qa_last_error());
            return false;
        }
    }

    if (setup.tts_url.empty()) {
        tts_bridge_params tts_init;
        tts_init.talker_path             = files.talker;
        tts_init.codec_path              = files.codec;
        tts_init.voices_dir              = voices_dir;
        tts_init.device                  = device;
        tts_init.sampling.max_new_tokens = S2S_TTS_MAX_NEW_TOKENS;
        tts_init.engine                  = engine;

        models.tts = tts_bridge_load(tts_init);
        if (!models.tts) {
            s2s_log(S2S_LOG_ERROR, "[Load] FATAL: %s", tts_bridge_last_error());
            return false;
        }
        setup.defaults.tts = tts_bridge_defaults_request(models.tts);
    } else {
        auto voices = conn_tts_voices(&setup);
        models.tts = tts_bridge_load2(voices);
    }

    models.aec = lv_init(files.aec.c_str(), device.empty() ? nullptr : device.c_str());
    if (!models.aec) {
        s2s_log(S2S_LOG_ERROR, "[Load] FATAL: %s", lv_last_error());
        return false;
    }

    if (setup.tts_url.empty()) {
        return true;
    }
    setup.defaults.tts = {};
    setup.defaults.tts.effect = "off";
    return true;
}

void models_free(ServerModels & models) {
    lv_free(models.aec);
    tts_bridge_free(models.tts);
    qa_free(models.asr);
    st_free(models.turn);
    sv_free(models.vad);
    models.aec  = nullptr;
    models.tts  = nullptr;
    models.asr  = nullptr;
    models.turn = nullptr;
    models.vad  = nullptr;
}
