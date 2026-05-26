class StrategyBuilderError(Exception):
    pass

class ConfigurationError(StrategyBuilderError):
    pass

class DataError(StrategyBuilderError):
    pass

class ValidationError(StrategyBuilderError):
    pass

class BacktestError(StrategyBuilderError):
    pass
