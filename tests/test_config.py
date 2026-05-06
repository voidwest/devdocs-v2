from app.config import Settings, get_settings


def test_settings_defaults():
    settings = Settings()
    assert settings.llm_model == "phi3:mini"
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64
    assert settings.top_k == 5
    assert settings.temperature == 0.1
    assert settings.max_prompt_chars == 12000
    assert settings.request_timeout == 300


def test_settings_data_dir():
    settings = Settings()
    assert settings.data_dir == Settings().data_dir


def test_get_settings_is_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_settings_override():
    settings = Settings(llm_model="test-model", top_k=10)
    assert settings.llm_model == "test-model"
    assert settings.top_k == 10
