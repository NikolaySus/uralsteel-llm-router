"""
gRPC Interceptor для проверки авторизации на уровне протокола.
"""

import grpc
from logger import logger


class AuthInterceptor(grpc.ServerInterceptor):
    """
    gRPC Interceptor для проверки авторизации на уровне протокола.
    
    Преимущества использования Interceptor вместо проверок в методах:
    1. Проверка выполняется ДО вызова бизнес-логики (минимум затрат CPU)
    2. Неавторизованные запросы отклоняются на уровне сети
    3. Нет дублирования кода в каждом методе
    4. Легко добавлять публичные методы без авторизации
    
    Методы без авторизации (public):
    - Ping: используется для health-check'ов, должен быть всегда доступен
    
    Методы требующие авторизации (protected):
    - NewMessage: взаимодействие с LLM, критичный ресурс
    - AvailableModelsText2Text: получение информации о моделях
    - AvailableModelsSpeech2Text: получение информации о моделях
    """

    # Методы, которые НЕ требуют авторизацию (public)
    PUBLIC_METHODS = {
        '/llm.Llm/Ping',
        '/grpc.health.v1.Health/Check',
        '/grpc.health.v1.Health/Watch',
    }

    def __init__(self, secret_key: str):
        """
        Инициализирует AuthInterceptor с переданным секретным ключом.
        
        Args:
            secret_key: Секретный ключ для проверки авторизации
        """
        self.secret_key = secret_key

    def intercept_service(self, continuation, handler_call_details):
        """
        Перехватывает каждый вызов RPC метода.
        
        handler_call_details.invocation_metadata содержит метаданные запроса,
        включая авторизационные заголовки.
        
        Метод должен вернуть либо обработанный ответ, либо вызвать
        continuation() для передачи запроса дальше.
        """
        method_name = handler_call_details.method

        # Если метод в списке публичных - пропускаем проверку авторизации
        if method_name in self.PUBLIC_METHODS:
            return continuation(handler_call_details)

        # Проверяем наличие авторизационных метаданных
        metadata = dict(handler_call_details.invocation_metadata or [])
        authorization = metadata.get('authorization', '')

        # Ожидаем формат "Bearer <SECRET_KEY>" или просто секретный ключ
        secret_from_header = authorization.replace('Bearer ', '').strip()

        # Проверка секретного ключа
        if not self.secret_key:
            # Если SECRET_KEY не задан в env - логируем предупреждение
            # но НЕ отклоняем запрос (для совместимости с dev средой)
            logger.warning("SECRET_KEY not set, auth skip for method %s",
                           method_name)
            return continuation(handler_call_details)

        if secret_from_header != self.secret_key:
            # Секретный ключ неверен - отклоняем запрос
            logger.error("Bad SECRET_KEY for method %s", method_name)
            # Возвращаем обработчик, который всегда абортит вызов
            def abort_unary_unary(_, context):
                context.abort(grpc.StatusCode.UNAUTHENTICATED,
                              "Invalid or missing authorization")

            return grpc.RpcMethodHandler(
                request_streaming=False,
                response_streaming=False,
                unary_unary=abort_unary_unary,
                unary_stream=None,
                stream_unary=None,
                stream_stream=None,
            )

        # Секретный ключ верен - пропускаем запрос дальше
        return continuation(handler_call_details)
