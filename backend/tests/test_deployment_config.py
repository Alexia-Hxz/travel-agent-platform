from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_compose_defines_backend_and_frontend_services() -> None:
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "backend:" in compose
    assert "frontend:" in compose
    assert "8001:8000" in compose
    assert "5173:80" in compose


def test_compose_restarts_services_after_docker_restart() -> None:
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert compose.count("restart: unless-stopped") == 2


def test_backend_dockerfile_has_a_health_check() -> None:
    dockerfile = (PROJECT_ROOT / "deploy" / "Dockerfile.backend").read_text(encoding="utf-8")

    assert "HEALTHCHECK" in dockerfile
    assert "/api/v1/health" in dockerfile


def test_backend_dockerfile_installs_paddle_runtime_library() -> None:
    dockerfile = (PROJECT_ROOT / "deploy" / "Dockerfile.backend").read_text(encoding="utf-8")
    assert "libgomp1" in dockerfile


def test_backend_dockerfile_installs_opencv_runtime_library() -> None:
    dockerfile = (PROJECT_ROOT / "deploy" / "Dockerfile.backend").read_text(encoding="utf-8")

    # PaddleOCR imports OpenCV, which requires libGL.so.1 in Debian slim images.
    assert "libgl1" in dockerfile


def test_backend_dockerfile_uses_paddle_compatible_debian_base() -> None:
    dockerfile = (PROJECT_ROOT / "deploy" / "Dockerfile.backend").read_text(encoding="utf-8")
    # PaddlePaddle 2.6.2 is not compatible with the newer Debian trixie base
    # currently selected by the floating python:3.11-slim tag.
    assert "python:3.11-slim-bookworm" in dockerfile


def test_backend_dockerfile_avoids_broken_charset_native_extensions() -> None:
    dockerfile = (PROJECT_ROOT / "deploy" / "Dockerfile.backend").read_text(encoding="utf-8")
    # The native charset-normalizer modules can fail to import in slim images;
    # its pure-Python fallback is compatible with PaddleOCR and requests.
    assert "charset_normalizer -name '*.so' -delete" in dockerfile


def test_environment_template_documents_optional_agent_keys_without_values() -> None:
    template = (PROJECT_ROOT / "backend" / ".env.example").read_text(encoding="utf-8")

    assert "DASHSCOPE_API_KEY=" in template
    assert "PEXELS_API_KEY=" in template
    assert "sk-" not in template
    # The metered search provider was removed, so its key must not still be advertised:
    # an operator who fills it in would be configuring a service nothing calls.
    assert "TAVILY_API_KEY" not in template


def test_readme_documents_compose_startup_and_browser_addresses() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "docker compose up --build" in readme
    assert "http://127.0.0.1:5173" in readme
    assert "http://127.0.0.1:8001/api/v1/health" in readme


def test_nginx_allows_configured_file_upload_limit() -> None:
    nginx = (PROJECT_ROOT / "deploy" / "nginx.conf").read_text(encoding="utf-8")

    # The UI permits 10 MiB files; the extra margin covers multipart headers.
    assert "client_max_body_size 12m;" in nginx


def test_nginx_waits_for_slow_multimodal_requests() -> None:
    nginx = (PROJECT_ROOT / "deploy" / "nginx.conf").read_text(encoding="utf-8")

    # Planning plus multimodal analysis can take minutes; a short read timeout turns
    # a slow but healthy request into a 504 the user reads as "backend not running".
    assert "proxy_read_timeout 600s;" in nginx
    assert "proxy_send_timeout 600s;" in nginx
