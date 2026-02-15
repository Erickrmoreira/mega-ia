from typing import Callable

from fastapi import Header, HTTPException, Request


def client_ip(request: Request) -> str:
    """
    Resolve origem de cliente para autenticação contextual e rate limit.

    Contrato de segurança:
    - Só considera `X-Forwarded-For` quando o proxy de entrada está na lista confiável.
    - Em qualquer outro cenário, usa o IP do socket para evitar spoof de origem.
    """
    settings = getattr(request.app.state, "settings", None)
    trust_proxy = bool(getattr(settings, "trust_proxy_headers", False))
    trusted_proxies = set(getattr(settings, "trusted_proxy_ips", []))
    client_host = (
        request.client.host if request.client and request.client.host else "unknown"
    )
    proxy_is_trusted = (not trusted_proxies) or (client_host in trusted_proxies)
    if trust_proxy and proxy_is_trusted:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
    return client_host


def require_api_key_factory(settings) -> Callable:
    """
    Cria dependência de autenticação para rotas operacionais da API.

    Contrato: quando `require_api_key` estiver desabilitado, a validação é bypassada;
    caso contrário, rejeita credencial ausente ou inválida com 401.
    """

    def require_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key")
    ) -> None:
        if not settings.require_api_key:
            return
        if x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Nao autorizado.")

    return require_api_key


def require_train_api_key_factory(settings) -> Callable:
    """
    Cria dependência de autenticação exclusiva para endpoints de treino.

    Contrato: exige chave específica de treino e falha com 500 quando a configuração
    de chave obrigatória estiver ausente.
    """

    def require_train_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key")
    ) -> None:
        if not settings.require_api_key:
            return
        if not settings.train_api_key:
            raise HTTPException(
                status_code=500, detail="Train API key nao configurada."
            )
        if x_api_key != settings.train_api_key:
            raise HTTPException(status_code=401, detail="Nao autorizado para treino.")

    return require_train_api_key


def require_model_info_api_key_factory(settings) -> Callable:
    """
    Cria dependência de autenticação para leitura de metadados de modelo.

    Contrato: separa escopo de leitura de modelo das chaves de geração e treino.
    """

    def require_model_info_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key")
    ) -> None:
        if not settings.require_api_key:
            return
        if not settings.model_info_api_key:
            raise HTTPException(
                status_code=500, detail="Model info API key nao configurada."
            )
        if x_api_key != settings.model_info_api_key:
            raise HTTPException(
                status_code=401, detail="Nao autorizado para leitura de modelo."
            )

    return require_model_info_api_key


def require_metrics_api_key_factory(settings) -> Callable:
    """
    Cria dependência de autenticação para endpoints de observabilidade.

    Contrato: protege métricas operacionais com chave dedicada para reduzir exposição.
    """

    def require_metrics_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key")
    ) -> None:
        if not settings.require_api_key:
            return
        if not settings.metrics_api_key:
            raise HTTPException(
                status_code=500, detail="Metrics API key nao configurada."
            )
        if x_api_key != settings.metrics_api_key:
            raise HTTPException(status_code=401, detail="Nao autorizado para metricas.")

    return require_metrics_api_key


def validate_generation_cost(settings, n_games: int, candidates: int) -> None:
    """
    Aplica limites de custo para proteger capacidade computacional da API.

    Contrato: rejeita requisições que excedem teto de candidatos ou custo combinado.
    """
    if candidates > settings.max_candidates:
        raise HTTPException(
            status_code=422,
            detail=f"candidates excede limite ({settings.max_candidates}).",
        )
    if n_games * candidates > settings.max_request_cost:
        raise HTTPException(
            status_code=422, detail="Custo da requisicao excede limite de seguranca."
        )
