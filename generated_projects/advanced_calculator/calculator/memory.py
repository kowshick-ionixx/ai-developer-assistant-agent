"""Memory functions for calculator (MC, MR, M+, M-, MS)."""


class CalculatorMemory:
    def __init__(self):
        self.memory = 0.0

    def memory_clear(self):
        self.memory = 0.0
        return self.memory

    def memory_recall(self) -> float:
        return self.memory

    def memory_store(self, value: float) -> float:
        try:
            self.memory = float(value)
        except Exception:
            raise ValueError("Invalid memory value")
        return self.memory

    def memory_add(self, value: float) -> float:
        try:
            self.memory += float(value)
        except Exception:
            raise ValueError("Invalid value for memory addition")
        return self.memory

    def memory_subtract(self, value: float) -> float:
        try:
            self.memory -= float(value)
        except Exception:
            raise ValueError("Invalid value for memory subtraction")
        return self.memory

    # Aliases matching the method names app.py calls
    clear = memory_clear
    recall = memory_recall
    store = memory_store


# Alias matching the class name app.py imports
Memory = CalculatorMemory
